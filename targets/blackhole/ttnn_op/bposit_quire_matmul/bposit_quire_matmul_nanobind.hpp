// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::operations::experimental::bposit_quire_matmul::detail {
namespace nb = nanobind;
void bind_bposit_quire_matmul(nb::module_& mod);
}  // namespace ttnn::operations::experimental::bposit_quire_matmul::detail
