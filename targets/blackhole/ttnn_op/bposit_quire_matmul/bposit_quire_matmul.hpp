// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Exact 256-bit Kulisch quire b-posit16 MATMUL  C[M,N] = A[M,K] · B[K,N].
// a, b: ROW_MAJOR uint32 tensors carrying b-posit16 codes in the low 16 bits.
// Returns a ROW_MAJOR uint32 [M,N] tensor of b-posit16 readout codes.
ttnn::Tensor bposit_quire_matmul(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    bool bp32_readout = false,
    bool input_float = false,
    bool input_bf16bits = false,
    bool output_float = false,
    bool output_bf16bits = false);

}  // namespace ttnn::experimental
