// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bposit_quire_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct BpositQuireMatmulProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle kernel_id{};
        std::vector<CoreCoord> cores;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const BpositQuireMatmulParams& operation_attributes,
        const BpositQuireMatmulInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const BpositQuireMatmulParams& operation_attributes,
        const BpositQuireMatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
