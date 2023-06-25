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
#include <limits>
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
	printf("Hello from rv_msa \r\n");
        constexpr std::size_t max_kernel_width = riscv_msa_nchw_pooling_fwd_t<data_type::f32>::max_kernel_width;
	float arr_flt_min[max_kernel_width];
        for (std::size_t i = 0; i < max_kernel_width; i++)
            arr_flt_min[i] = std::numeric_limits<float>::min();

#ifdef DNNL_RISCV_USE_MSA_INTRINSICS 
        printf("Using MSA to implement pooling \r\n");
        for (std::size_t mb = 0; mb < batch; mb++) {
            for (std::size_t c = 0; c < channels; c++) {
                for (std::size_t od = 0; od < outD; od++) {
                    for (std::size_t oh = 0; oh < outH; oh++) {
                        for (std::size_t ow = 0; ow < outW; ow++) {
                            const std::size_t dst_offset =
                                outW * outH * outD * channels * mb +
                                outW * outH * outD * c +
                                outW * outH * od + outW * oh + ow;
                            const std::size_t src_offset =
                                inW * inH * inD * (channels * mb + c);
                            const float* local_src = &src[src_offset];
                            const std::size_t IWH = inW * inH;

                            int od_offset = od * strideD - padFront;
                            int oh_offset = oh * strideH - padTop;
                            int ow_offset = ow * strideW - padLeft;
                            std::size_t size =
                                std::min(static_cast<std::size_t>(ow_offset + kerW), inW) -
                                std::max(static_cast<int>(ow_offset), 0);
                            std::size_t cycleLength = size / 4;


			    v4i32 varr_flt_min = __builtin_msa_ld_w(reinterpret_cast<const v4i32*>(&arr_flt_min[0]), 0);
			    v4f32 vmax;
                                for (int i = 0; i < 4; i++)
				{
				    v4f32 vsrc = (v4f32)__builtin_msa_ld_w(reinterpret_cast<const v4i32*>(&arr_flt_min[0]), 0);
				    vmax[i] = vsrc[0];
				}


                            for (int id = std::max(od_offset, 0); id < std::min(static_cast<int>(od_offset + kerD), static_cast<int>(inD)); id++) {
                                for (int ih = std::max(oh_offset, 0); ih < std::min(static_cast<int>(oh_offset + kerH), static_cast<int>(inH)); ih++) {
                                    const std::size_t local_src_offset = IWH * id +
                                        inW * ih +
                                        std::max(ow_offset, 0);

                                    std::size_t iw = 0;
                                    for (; iw < cycleLength; iw++) {
                                        v4f32 vsrc = (v4f32_w)__builtin_msa_ld_w(reinterpret_cast<const v4i32*>(&local_src[local_src_offset + iw * 4]), 0);
                                        vmax = __builtin_msa_fmax_w(vsrc, vmax);
                                    }

                                    for (; iw < size; iw++) {
                                        v4f32 vsrc = (v4f32_w)__builtin_msa_ld_w(reinterpret_cast<const v4i32*>(&local_src[local_src_offset + iw * 4]), 0);
                                        vmax = __builtin_msa_fmax_w(vsrc, vmax);
                                    }
                                }
                            }

                            float max_value = vmax[0];
                            for (std::size_t i = 1; i < 4; i++) {
                                max_value = std::max(max_value, vmax[i]);
                            }

                            dst[dst_offset] = max_value;
                        }
                    }
                }
            }
        }
    }


#else
        for (std::size_t mb = 0; mb < batch; mb++) {
            for (std::size_t c = 0; c < channels; c++) {
                for (std::size_t od = 0; od < outD; od++) {
                    for (std::size_t oh = 0; oh < outH; oh++) {
                        for (std::size_t ow = 0; ow < outW; ow++) {
                            const std::size_t dst_offset =
                                outW * outH * outD * channels * mb +
                                outW * outH * outD * c +
                                outW * outH * od + outW * oh + ow;
                            const std::size_t src_offset =
                                inW * inH * inD * (channels * mb + c);
                            const float* local_src = &src[src_offset];
                            const std::size_t IWH = inW * inH;

                            int od_offset = od * strideD - padFront;
                            int oh_offset = oh * strideH - padTop;
                            int ow_offset = ow * strideW - padLeft;
                            std::size_t size =
                                std::min(ow_offset + kerW, inW) -
                                std::max(ow_offset, 0);

                            float vmax = arr_flt_min[0];

                            for (int id = std::max(od_offset, 0);
                                 id < std::min(od_offset + kerD, inD); id++) {
                                for (int ih = std::max(oh_offset, 0);
                                     ih < std::min(oh_offset + kerH, inH);
                                     ih++) {
                                    const std::size_t local_src_offset =
                                        IWH * id + inW * ih + std::max(ow_offset, 0);

                                    for (std::size_t iw = 0; iw < size; iw++) {
                                        float vsrc = local_src[local_src_offset + iw];
                                        vmax = std::max(vsrc, vmax);
                                    }
                                }
                            }

                            dst[dst_offset] = vmax;
                        }
                    }
                }
            }
        }
    }
#endif
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
