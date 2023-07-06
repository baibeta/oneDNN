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

#include "rvv_inner_product.hpp"
#include "cpu/ref_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

status_t riscv_rvv_inner_product_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    printf("Hello from riscv rvv inner product fwd \r\n");

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
                        const auto src_off = ref_ip_utils::get_data_off(
                                    src_d, ndims, mb, ic, kd, kh, 0);
                        const auto wei_off = ref_ip_utils::get_weights_off(
                                    weights_d, ndims, oc, ic, kd, kh, 0);

                        int stride_src = ref_ip_utils::get_data_off(src_d, ndims, mb, ic, kd, kh, 1) - src_off;
                        int stride_wei = ref_ip_utils::get_weights_off(weights_d, ndims, oc, ic, kd, kh, 1) - wei_off;

                        // VLMAX= 128 * 8 / 32 = 32 (KW <= 32)
                        size_t vl = vsetvl_e32m8(KW);

                        vfloat32m8_t v_s = vlse32_v_f32m8((float *)src + src_off, stride_src*4, vl);
                        vfloat32m8_t v_w = vlse32_v_f32m8((float *)weights + wei_off, stride_wei*4, vl);

                        vfloat32m8_t v_d = vfmul_vv_f32m8(v_s, v_w, vl);

                        // Sum results
                        vfloat32m1_t zero_scalar;
                        float zero = 0.0;
                        zero_scalar = vle32_v_f32m1(&zero, 1);
                        vfloat32m1_t vred_res;
                        vred_res = vfredosum_vs_f32m8_f32m1(vred_res, v_d, zero_scalar, vl);

                        float red_res;
                        vse32_v_f32m1(&red_res, vred_res, 1);
                        d += red_res;
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
