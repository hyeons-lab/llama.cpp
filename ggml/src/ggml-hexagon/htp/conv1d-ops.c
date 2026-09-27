#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#include "hex-common.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-vtcm.h"
#include "htp-tensor.h"
#include "work-queue.h"
#include "conv1d-ops.h"

struct htp_conv1d_context {
    struct htp_ops_context * octx;
    uint32_t                 stride;
    uint32_t                 pad;
    uint32_t                 dilation;
    uint32_t                 groups;
    uint32_t                 cout_per_thread;
};

static void conv1d_worker_f32(unsigned int nth, unsigned int ith, void * data) {
    struct htp_conv1d_context * cctx = (struct htp_conv1d_context *) data;
    struct htp_ops_context *    octx = cctx->octx;

    const struct htp_tensor * src0 = octx->src[0]; // input: [W_in, C_in, N, 1]
    const struct htp_tensor * src1 = octx->src[1]; // weight: [K, C_in/groups, C_out, 1]
    const struct htp_tensor * src2 = octx->src[2]; // bias: [C_out, 1, 1, 1] (NULL if none)
    const struct htp_tensor * dst  = octx->dst;    // output: [W_out, C_out, N, 1]

    const uint32_t w_in   = src0->ne[0];
    const uint32_t c_in   = src0->ne[1];
    const uint32_t n_in   = src0->ne[2];

    const uint32_t k_size = src1->ne[0];
    const uint32_t c_out  = dst->ne[1];
    const uint32_t w_out  = dst->ne[0];
    const uint32_t n_out  = dst->ne[2];

    const uint32_t stride   = cctx->stride;
    const uint32_t pad      = cctx->pad;
    const uint32_t dilation = cctx->dilation;
    const uint32_t groups   = cctx->groups;

    const uint32_t cin_per_group  = c_in / groups;
    const uint32_t cout_per_group = c_out / groups;

    const uint32_t cout_start = ith * cctx->cout_per_thread;
    const uint32_t cout_end   = (cout_start + cctx->cout_per_thread < c_out) ?
                                (cout_start + cctx->cout_per_thread) : c_out;

    if (cout_start >= c_out) {
        return;
    }

    const uint8_t * in_bytes     = (const uint8_t *) src0->data;
    const uint8_t * weight_bytes = (const uint8_t *) src1->data;
    const uint8_t * bias_bytes   = src2 ? (const uint8_t *) src2->data : NULL;
    uint8_t *       out_bytes    = (uint8_t *) dst->data;

    for (uint32_t b = 0; b < n_out; b++) {
        for (uint32_t oc = cout_start; oc < cout_end; oc++) {
            const uint32_t g        = oc / cout_per_group;
            const uint32_t oc_rel   = oc % cout_per_group;
            const uint32_t ic_start = g * cin_per_group;

            float bias_val = 0.0f;
            if (bias_bytes) {
                const float * bp = (const float *)(bias_bytes + oc * src2->nb[0]);
                bias_val = *bp;
            }

            // Depthwise fast path (groups == c_in == c_out, cin_per_group == 1)
            if (groups == c_in && groups == c_out && cin_per_group == 1) {
                const uint32_t ic = oc;

                // When stride is 1 and w_out >= 32, vectorize across output positions
                if (stride == 1 && w_out >= VLEN_FP32) {
                    const uint32_t nvec = w_out / VLEN_FP32;
                    const uint32_t nloe = w_out % VLEN_FP32;
                    const HVX_Vector v_bias = hvx_vec_splat_f32(bias_val);

                    for (uint32_t iv = 0; iv < nvec; iv++) {
                        const uint32_t ow_base = iv * VLEN_FP32;
                        HVX_Vector v_acc = v_bias;

                        for (uint32_t k = 0; k < k_size; k++) {
                            const float * wp = (const float *)(weight_bytes +
                                k * src1->nb[0] + oc * src1->nb[2]);
                            const float w_val = *wp;
                            const HVX_Vector v_w = hvx_vec_splat_f32(w_val);

                            int32_t iw_base = (int32_t) ow_base - (int32_t) pad + (int32_t)(k * dilation);

                            if (iw_base >= 0 && (uint32_t)(iw_base + VLEN_FP32) <= w_in) {
                                const float * in_p = (const float *)(in_bytes +
                                    iw_base * src0->nb[0] + ic * src0->nb[1] + b * src0->nb[2]);
                                HVX_Vector v_in = *(const HVX_Vector *) in_p;
                                HVX_Vector v_prod = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_in, v_w));
                                v_acc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_acc, v_prod));
                            } else {
                                // Boundary fallback: elementwise accumulation for this vector
                                float tmp_in[VLEN_FP32];
                                for (uint32_t vi = 0; vi < VLEN_FP32; vi++) {
                                    int32_t iw = iw_base + (int32_t) vi;
                                    if (iw >= 0 && (uint32_t) iw < w_in) {
                                        tmp_in[vi] = *(const float *)(in_bytes +
                                            iw * src0->nb[0] + ic * src0->nb[1] + b * src0->nb[2]);
                                    } else {
                                        tmp_in[vi] = 0.0f;
                                    }
                                }
                                HVX_Vector v_in = *(const HVX_Vector *) tmp_in;
                                HVX_Vector v_prod = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v_in, v_w));
                                v_acc = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v_acc, v_prod));
                            }
                        }

                        float * out_p = (float *)(out_bytes +
                            ow_base * dst->nb[0] + oc * dst->nb[1] + b * dst->nb[2]);
                        *(HVX_Vector *) out_p = v_acc;
                    }

                    // Trailing scalar positions
                    for (uint32_t ow = nvec * VLEN_FP32; ow < w_out; ow++) {
                        float sum = bias_val;
                        for (uint32_t k = 0; k < k_size; k++) {
                            int32_t iw = (int32_t)(ow * stride) - (int32_t) pad + (int32_t)(k * dilation);
                            if (iw >= 0 && (uint32_t) iw < w_in) {
                                const float * in_p = (const float *)(in_bytes +
                                    iw * src0->nb[0] + ic * src0->nb[1] + b * src0->nb[2]);
                                const float * wp   = (const float *)(weight_bytes +
                                    k * src1->nb[0] + oc * src1->nb[2]);
                                sum += (*in_p) * (*wp);
                            }
                        }
                        float * out_p = (float *)(out_bytes +
                            ow * dst->nb[0] + oc * dst->nb[1] + b * dst->nb[2]);
                        *out_p = sum;
                    }
                    continue;
                }
            }

            // General Conv1D path
            for (uint32_t ow = 0; ow < w_out; ow++) {
                float sum = bias_val;

                for (uint32_t k = 0; k < k_size; k++) {
                    int32_t iw = (int32_t)(ow * stride) - (int32_t) pad + (int32_t)(k * dilation);
                    if (iw < 0 || (uint32_t) iw >= w_in) {
                        continue;
                    }

                    for (uint32_t ic_rel = 0; ic_rel < cin_per_group; ic_rel++) {
                        const uint32_t ic = ic_start + ic_rel;
                        const float * in_p = (const float *)(in_bytes +
                            iw * src0->nb[0] + ic * src0->nb[1] + b * src0->nb[2]);
                        const float * wp   = (const float *)(weight_bytes +
                            k * src1->nb[0] + ic_rel * src1->nb[1] + oc * src1->nb[2]);

                        sum += (*in_p) * (*wp);
                    }
                }

                float * out_p = (float *)(out_bytes +
                    ow * dst->nb[0] + oc * dst->nb[1] + b * dst->nb[2]);
                *out_p = sum;
            }
        }
    }
}

int op_conv1d(struct htp_ops_context * octx) {
    if (octx->src[0]->type != HTP_TYPE_F32 || octx->src[1]->type != HTP_TYPE_F32) {
        FARF(ERROR, "op_conv1d: unsupported data type (f32 required)");
        return HTP_STATUS_NO_SUPPORT;
    }

    uint32_t stride   = octx->op_params[0] > 0 ? (uint32_t) octx->op_params[0] : 1;
    uint32_t pad      = (uint32_t) octx->op_params[1];
    uint32_t dilation = octx->op_params[2] > 0 ? (uint32_t) octx->op_params[2] : 1;
    uint32_t groups   = octx->op_params[3] > 0 ? (uint32_t) octx->op_params[3] : 1;

    const uint32_t c_out = octx->dst->ne[1];
    const uint32_t n_threads = octx->ctx->n_threads > 0 ? octx->ctx->n_threads : 1;

    struct htp_conv1d_context cctx;
    cctx.octx            = octx;
    cctx.stride          = stride;
    cctx.pad             = pad;
    cctx.dilation        = dilation;
    cctx.groups          = groups;
    cctx.cout_per_thread = (c_out + n_threads - 1) / n_threads;

    work_queue_run(octx->ctx->work_queue, conv1d_worker_f32, &cctx, n_threads);

    return HTP_STATUS_OK;
}
