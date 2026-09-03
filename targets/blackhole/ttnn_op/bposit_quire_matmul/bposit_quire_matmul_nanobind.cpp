// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "bposit_quire_matmul_nanobind.hpp"

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "bposit_quire_matmul.hpp"
#include "ttnn-nanobind/bind_function.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::bposit_quire_matmul::detail {

void bind_bposit_quire_matmul(nb::module_& mod) {
    ttnn::bind_function<"bposit_quire_matmul", "ttnn.experimental.">(
        mod,
        R"doc(
        bposit_quire_matmul(a: ttnn.Tensor, b: ttnn.Tensor, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Exact 256-bit Kulisch quire b-posit16 MATMUL  C[M,N] = A[M,K] @ B[K,N], run
        on the Tensix grid (one output row per core). Every K-product is exact (no
        per-product rounding) and accumulated in a 256-bit quire; only the final
        quire->b-posit16 readout rounds — recovering accuracy a rounding MAC array
        loses.

        a : ttnn.Tensor  [M, K]  ROW_MAJOR, UINT32 (b-posit16 codes in the low 16 bits)
        b : ttnn.Tensor  [K, N]  ROW_MAJOR, UINT32 (b-posit16 codes in the low 16 bits)
        bp32_readout : bool  if True, read the quire out as a bposit32 code (full uint32) —
            preserves far more of the exact accumulation; if False (default), bp16 (low 16 bits).
        returns : ttnn.Tensor [M, N] ROW_MAJOR UINT32 (bp16 or bp32 readout codes)
        )doc",
        &ttnn::experimental::bposit_quire_matmul,
        nb::arg("a"),
        nb::arg("b"),
        nb::kw_only(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("bp32_readout") = false,
        nb::arg("input_float") = false,
        nb::arg("input_bf16bits") = false,
        nb::arg("output_float") = false,
        nb::arg("output_bf16bits") = false);
}

}  // namespace ttnn::operations::experimental::bposit_quire_matmul::detail
