// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "bposit_quire_matmul.hpp"
#include "device/bposit_quire_matmul_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor bposit_quire_matmul(
    const ttnn::Tensor& a, const ttnn::Tensor& b, const std::optional<ttnn::MemoryConfig>& memory_config,
    bool bp32_readout, bool input_float, bool input_bf16bits, bool output_float, bool output_bf16bits) {
    return ttnn::prim::bposit_quire_matmul(
        a, b, memory_config, bp32_readout, input_float, input_bf16bits, output_float, output_bf16bits);
}

}  // namespace ttnn::experimental
