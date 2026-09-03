// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

// Exact 256-bit Kulisch quire b-posit16 MATMUL. Operands are ttnn ROW_MAJOR
// uint32 tensors carrying b-posit16 codes in the low 16 bits of each word.
struct BpositQuireMatmulParams {
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    bool bp32_readout = false;   // false = bp16 readout (low 16b), true = bp32 (full 32b code)
    bool input_float = false;    // operands are bf16 2B (encode on-device; blocked by ttnn ROW_MAJOR bf16 layout)
    bool input_bf16bits = false;  // operands uint32 w/ bf16 bits in low 16 (encode on-device; proven uint32 read)
    bool output_float = false;    // output is bf16 2B (decode on-device; blocked path)
    bool output_bf16bits = false; // output uint32 w/ bf16 bits in low 16 (decode on-device; proven uint32 write)
};

struct BpositQuireMatmulInputs {
    Tensor a;  // [M, K] ROW_MAJOR uint32 (bp16 codes)
    Tensor b;  // [K, N] ROW_MAJOR uint32 (bp16 codes)
};

}  // namespace ttnn::experimental::prim
