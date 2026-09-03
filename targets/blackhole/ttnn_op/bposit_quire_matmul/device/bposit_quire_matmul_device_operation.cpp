// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "bposit_quire_matmul_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

BpositQuireMatmulDeviceOperation::program_factory_t BpositQuireMatmulDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return BpositQuireMatmulProgramFactory{};
}

void BpositQuireMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    TT_FATAL(
        a.storage_type() == StorageType::DEVICE && b.storage_type() == StorageType::DEVICE,
        "bposit_quire_matmul operands must be on device");
    TT_FATAL(a.device() == b.device(), "bposit_quire_matmul operands must be on the same device");
    TT_FATAL(
        a.layout() == Layout::ROW_MAJOR && b.layout() == Layout::ROW_MAJOR,
        "bposit_quire_matmul requires ROW_MAJOR operands");
    if (attrs.input_float) {
        TT_FATAL(
            a.dtype() == DataType::BFLOAT16 && b.dtype() == DataType::BFLOAT16,
            "bposit_quire_matmul(input_float) requires BFLOAT16 operands (encoded on-device)");
    } else {
        TT_FATAL(
            a.dtype() == DataType::UINT32 && b.dtype() == DataType::UINT32,
            "bposit_quire_matmul requires UINT32 operands (b-posit16 codes in the low 16 bits)");
    }
    const auto& as = a.logical_shape();
    const auto& bs = b.logical_shape();
    TT_FATAL(as.rank() == 2 && bs.rank() == 2, "bposit_quire_matmul expects 2D operands [M,K] and [K,N]");
    TT_FATAL(as[-1] == bs[-2], "bposit_quire_matmul inner dims must match: K={} vs {}", as[-1], bs[-2]);
    // Rows must clear the 64B DRAM read granularity: uint32 (4B) needs K,N>=16; bf16 (2B) needs >=32.
    const int min_dim = attrs.input_float ? 32 : 16;
    TT_FATAL(as[-1] >= min_dim && bs[-1] >= min_dim, "bposit_quire_matmul needs K,N >= {} (rows >= 64B)", min_dim);
    TT_FATAL(bs[-1] % 4 == 0, "bposit_quire_matmul needs N % 4 == 0 (16B-aligned column chunks)");
    // Streamed path: per-row B reads are chunk_cols*asz bytes and Blackhole DRAM transfers at 64-BYTE
    // granularity — a sub-64B-multiple read silently corrupts. The factory keeps chunk_cols a multiple
    // of gran=64/asz, but the LAST chunk's width is N - j0, so N itself must be a multiple of gran for
    // the streamed path. Fail LOUDLY here rather than corrupt (the float bridge pads N to satisfy this).
    const int64_t Mv = as[-2], Kv = as[-1], Nv = bs[-1];
    const bool streamed = attrs.input_float || attrs.input_bf16bits || attrs.output_float ||
                          attrs.output_bf16bits || (Mv * Kv + Kv * Nv + (Nv > 8 ? Nv : 8)) > 300000;
    const int gran = attrs.input_float ? 32 : 16;
    TT_FATAL(!streamed || (bs[-1] % gran == 0),
             "bposit_quire_matmul streamed path needs N % {} == 0 (64B DRAM granularity); pad N (the "
             "float bridge does this) — N={} is not a multiple of {}", gran, (int)Nv, gran);
}

void BpositQuireMatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}

BpositQuireMatmulDeviceOperation::spec_return_value_t BpositQuireMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto M = tensor_args.a.logical_shape()[-2];
    const auto N = tensor_args.b.logical_shape()[-1];
    const auto mem_config = attrs.output_mem_config.value_or(tensor_args.a.memory_config());
    const auto out_dtype = attrs.output_float ? DataType::BFLOAT16 : DataType::UINT32;
    return TensorSpec(ttnn::Shape({M, N}), TensorLayout(out_dtype, PageConfig(Layout::ROW_MAJOR), mem_config));
}

BpositQuireMatmulDeviceOperation::tensor_return_value_t BpositQuireMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.a.device());
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
ttnn::experimental::prim::BpositQuireMatmulDeviceOperation::tensor_return_value_t bposit_quire_matmul(
    const Tensor& a, const Tensor& b, const std::optional<MemoryConfig>& memory_config, bool bp32_readout,
    bool input_float, bool input_bf16bits, bool output_float, bool output_bf16bits) {
    using OperationType = ttnn::experimental::prim::BpositQuireMatmulDeviceOperation;
    OperationType::operation_attributes_t attrs{
        memory_config, bp32_readout, input_float, input_bf16bits, output_float, output_bf16bits};
    OperationType::tensor_args_t args{a, b};
    return ttnn::device_operation::launch<OperationType>(attrs, args);
}
}  // namespace ttnn::prim
