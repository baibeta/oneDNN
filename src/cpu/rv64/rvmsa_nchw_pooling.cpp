/******************************************************************************
* Copyright 2023 KNS Group LLC (YADRO)
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

#include "rvmsa_nchw_pooling.hpp"
#include <algorithm>
#include "stdio.h"
#include <cstddef>


namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace {
    void MaxPooling(const float* src, float* dst, std::size_t batch,
                    std::size_t channels, std::size_t outD,
                    std::size_t outH, std::size_t outW,
                    std::size_t inD, std::size_t inH,
                    std::size_t inW, std::size_t kerD,
                    std::size_t kerH, std::size_t kerW,
                    std::size_t strideD, std::size_t strideH,
                    std::size_t strideW, std::size_t padFront,
                    std::size_t padTop, std::size_t padLeft)
    {
	printf("Hello from rv_msa pooling \r\n");
        constexpr std::size_t max_kernel_width = riscv_msa_nchw_pooling_fwd_t<data_type::f32>::max_kernel_width;
	float arr_flt_min[max_kernel_width];
        for (std::size_t i = 0; i < max_kernel_width; i++)
            arr_flt_min[i] = -FLT_MAX;;

    for (int mb = 0; mb < batch; mb++)
        for (int c = 0; c < channels; c++)
            for (int od = 0; od < outD; od++)
                for (int oh = 0; oh < outH; oh++)
                    for (int ow = 0; ow < outW; ow++) {
                        const size_t dst_offset
                                = (size_t)outW * outH * outD * channels * mb
                                + (size_t)outW * outH * outD * c
                                + (size_t)outW * outH * od + (size_t)outW * oh
                                + (size_t)ow;
                        const auto src_offset = ((size_t)inW * inH * inD)
                                * ((size_t)channels * mb + c);
                        const auto local_src = &src[src_offset];
                        const auto IWH = (size_t)inW * inH;

                        int od_offset = od * strideD - padFront;
                        int oh_offset = oh * strideH - padTop;
                        int ow_offset = ow * strideW - padLeft;
                        size_t size = std::min(ow_offset + kerW, inW)
                                - std::max(ow_offset, 0);
			v4f32 vmax = (v4f32) __builtin_msa_ld_w((void*)arr_flt_min, 0);

                        for (int id = std::max(od_offset, 0);
                                id < std::min(od_offset + kerD, inD); id++)
                            for (int ih = std::max(oh_offset, 0);
                                    ih < std::min(oh_offset + kerH, inH);
                                    ih++) {
                                const auto local_src_offset = IWH * id
                                        + (size_t)inW * ih
                                        + std::max(ow_offset, 0);
                                for (size_t iw = 0; iw < size; iw += 4) {
				    int offs = local_src_offset + iw;
				    v4f32 vsrc = (v4f32) __builtin_msa_ld_w((float *)local_src + offs,0);
				    vmax = __builtin_msa_fmax_w(vsrc, vmax);
                                }
				// no tail for MSA pooling

		          dst[dst_offset] = vmax[0];
		          for(int i =1; i <4; i++){
				  dst[dst_offset] = vmax[i] > dst[dst_offset] ? vmax[i] : dst[dst_offset];
			  }
                        }
	            }
    }
} // namespace

template <data_type_t d_type>
riscv_msa_nchw_pooling_fwd_t<d_type>::riscv_msa_nchw_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd), ref_post_ops_(pd()->attr()->post_ops_) {}

template <>
status_t riscv_msa_nchw_pooling_fwd_t<data_type::f32>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    src += src_d.off_l(0);
    dst += dst_d.off_l(0);

    const dim_t MB = pd()->MB();
    const dim_t C = pd()->OC();
    const dim_t OD = pd()->OD();
    const dim_t OH = pd()->OH();
    const dim_t OW = pd()->OW();
    const dim_t ID = pd()->ID();
    const dim_t IH = pd()->IH();
    const dim_t IW = pd()->IW();
    const dim_t KD = pd()->KD();
    const dim_t KH = pd()->KH();
    const dim_t KW = pd()->KW();
    const dim_t SD = pd()->KSD();
    const dim_t SH = pd()->KSH();
    const dim_t SW = pd()->KSW();
    const dim_t padF = pd()->padFront();
    const dim_t padT = pd()->padT();
    const dim_t padL = pd()->padL();

    const auto alg = pd()->desc()->alg_kind;
    const bool is_max_pool = alg == alg_kind::pooling_max;

    if (!is_max_pool) { return status::unimplemented; }

    MaxPooling(src, dst, MB, C, OD, OH, OW, ID, IH, IW, KD, KH, KW, SD, SH, SW,
            padF, padT, padL);

    return status::success;
}

template struct riscv_msa_nchw_pooling_fwd_t<data_type::f32>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
