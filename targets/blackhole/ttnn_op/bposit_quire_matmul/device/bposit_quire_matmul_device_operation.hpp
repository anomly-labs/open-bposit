// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "bposit_quire_matmul_device_operation_types.hpp"
#include "bposit_quire_matmul_program_factory.hpp"

namespace ttnn::experimental::prim {

struct BpositQuireMatmulDeviceOperation {
    using operation_attributes_t = BpositQuireMatmulParams;
    using tensor_args_t = BpositQuireMatmulInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<BpositQuireMatmulProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
ttnn::experimental::prim::BpositQuireMatmulDeviceOperation::tensor_return_value_t bposit_quire_matmul(
    const Tensor& a, const Tensor& b, const std::optional<MemoryConfig>& memory_config, bool bp32_readout,
    bool input_float, bool input_bf16bits, bool output_float, bool output_bf16bits);
}  // namespace ttnn::prim
