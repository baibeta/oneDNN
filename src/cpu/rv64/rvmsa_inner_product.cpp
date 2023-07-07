/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_io_helper.hpp"

#include "rvmsa_inner_product.hpp"
#include "cpu/ref_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

status_t riscv_msa_inner_product_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    printf("Hello from riscv msa inner product fwd \r\n");

    status_t status = status::success;
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto ndims = pd()->ndims();
    const auto MB = pd()->MB();
    const auto OC = pd()->OC();
    const auto IC = pd()->IC();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();

    const format_tag_t desired_fmt_tag = utils::pick(ndims - 3,
        format_tag::ncw, format_tag::nchw, format_tag::ncdhw);
    bool unit_stride_src = memory_desc_matches_tag(*pd()->src_md(), desired_fmt_tag);

    const format_tag_t desired_fmt_tag_weights = utils::pick(ndims - 3,
        format_tag::oiw, format_tag::oihw, format_tag::oidhw);
    bool unit_stride_wei = memory_desc_matches_tag(*pd()->weights_md(0), desired_fmt_tag_weights);

    // mini batch
    for (dim_t mb = 0; mb < MB; ++mb) {
        // output channel
        for (dim_t oc = 0; oc < OC; ++oc) {
            float d = 0;
            // input channel
            for (dim_t ic = 0; ic < IC; ++ic) {
                // kernel depth
                for (dim_t kd = 0; kd < KD; ++kd) {
                    // kernel height
                    for (dim_t kh = 0; kh < KH; ++kh) {
                        // kernel width
                        for (dim_t kw = 0; kw < KW; ) {
                            const auto src_off = ref_ip_utils::get_data_off(
                                        src_d, ndims, mb, ic, kd, kh, kw);
                            const auto wei_off = ref_ip_utils::get_weights_off(
                                        weights_d, ndims, oc, ic, kd, kh, kw);
                            if (KW < 4 || KW - kw < 4) {
                                const float s = ((float *)src)[src_off];
                                const float w = ((float *)weights)[wei_off];
                                d += s * w;
                                kw++;
                            } else {
                                v4f32 v_s = (v4f32)__msa_ld_w((float *)src + src_off, 0);
                                v4f32 v_w = (v4f32)__msa_ld_w((float *)weights + wei_off, 0);

                                if (!unit_stride_src) {
                                    v_s = (v4f32)__msa_insert_w((v4i32)v_s, 1,
                                        *((int *) src + ref_ip_utils::get_data_off(src_d, ndims, mb, ic, kd, kh, kw+1)));
                                    v_s = (v4f32)__msa_insert_w((v4i32)v_s, 2,
                                        *((int *) src + ref_ip_utils::get_data_off(src_d, ndims, mb, ic, kd, kh, kw+2)));
                                    v_s = (v4f32)__msa_insert_w((v4i32)v_s, 3,
                                        *((int *) src + ref_ip_utils::get_data_off(src_d, ndims, mb, ic, kd, kh, kw+3)));
                                }
                                if (!unit_stride_wei) {
                                    v_w = (v4f32)__msa_insert_w((v4i32)v_w, 1,
                                        *((int *) weights + ref_ip_utils::get_weights_off(weights_d, ndims, oc, ic, kd, kh, kw + 1)));
                                    v_w = (v4f32)__msa_insert_w((v4i32)v_w, 2,
                                        *((int *) weights + ref_ip_utils::get_weights_off(weights_d, ndims, oc, ic, kd, kh, kw + 2)));
                                    v_w = (v4f32)__msa_insert_w((v4i32)v_w, 3,
                                        *((int *) weights + ref_ip_utils::get_weights_off(weights_d, ndims, oc, ic, kd, kh, kw + 3)));
                                }
                                v4f32 v_d = __msa_fmul_w(v_s, v_w);

                                d += v_d[0] + v_d[1] + v_d[2] + v_d[3];
                                kw += 4;
                            }
                        }
                    }
                }
            }

            if (bias) {
                const auto bias_off = bias_d.off(oc);
                const float b = ((float *)bias)[bias_off];
                d += b;
            }

            dim_t dst_off = dst_d.off(mb, oc);
            dim_t dst_l_off = (mb * OC + oc);
            ((float *)dst)[dst_off] = d;
        }
    }

    return status::success;
}


} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
