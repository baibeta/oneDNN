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

#ifndef RV64_INNER_PRODUCT_HPP
#define RV64_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_inner_product_pd.hpp"
#include "msa.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct riscv_msa_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T("RISCV64GC_MSA", riscv_msa_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto bia_type = weights_md(1)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            bool ok = is_fwd() && platform::has_data_type_support(src_type)
                    && platform::has_data_type_support(wei_type)
                    && platform::has_data_type_support(bia_type)
                    && platform::has_data_type_support(dst_type)
                    && utils::one_of(src_type, f32)
                    && wei_type == src_type
                    && utils::one_of(dst_type, f32, src_type)
                    && IMPLICATION(
                            with_bias(), utils::one_of(bia_type, f32, src_type))
                    && attr()->has_default_values(
                            smask_t::post_ops | smask_t::sum_dt)
                    && attr()->post_ops_.check_sum_consistency(dst_type,
                            /* is_int8 */ false)
                    && attr_.set_default_formats(dst_md(0)) == status::success;

            return ok ? status::success : status::unimplemented;
        }
    };

    riscv_msa_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
