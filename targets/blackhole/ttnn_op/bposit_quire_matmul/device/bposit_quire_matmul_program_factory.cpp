// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "bposit_quire_matmul_device_operation.hpp"
#include "bposit_quire_matmul_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <algorithm>
#include <cstdlib>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::tt_metal;

static constexpr const char* kKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/bposit_quire_matmul/device/kernels/quire_matmul_op.cpp";
static constexpr const char* kStreamedKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/bposit_quire_matmul/device/kernels/quire_matmul_op_streamed.cpp";

// L1-fit thresholds (words). Non-streamed loads all of A+B; switch to streamed (per-unit
// A-row + B-chunk-columns) when that would overflow. ~1.5MB L1/core => keep well under.
static constexpr uint32_t kNonStreamedMaxWords = 300000u;  // ~1.2MB for A+B+C
static constexpr uint32_t kStreamedBColBudget = 200000u;   // ~0.8MB for B chunk-columns

BpositQuireMatmulProgramFactory::cached_program_t BpositQuireMatmulProgramFactory::create(
    const BpositQuireMatmulParams& attrs, const BpositQuireMatmulInputs& tensor_args, Tensor& output) {
    const uint32_t readout_bp32 = attrs.bp32_readout ? 1u : 0u;
    const uint32_t input_float = attrs.input_float ? 1u : 0u;          // bf16 2B operands (blocked)
    const uint32_t input_bf16bits = attrs.input_bf16bits ? 1u : 0u;    // uint32 operands w/ bf16 bits in low 16
    const uint32_t output_float = attrs.output_float ? 1u : 0u;        // output bf16 2B (decode; blocked)
    const uint32_t output_bf16bits = attrs.output_bf16bits ? 1u : 0u;  // output uint32 w/ bf16 bits in low 16
    const uint32_t asz_in = input_float ? 2u : 4u;                     // operand element bytes (bf16bits = uint32 = 4)
    Program program = CreateProgram();

    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    auto* a_buffer = a.buffer();
    auto* b_buffer = b.buffer();
    auto* c_buffer = output.buffer();

    const uint32_t WORD = sizeof(uint32_t);
    const uint32_t M = a.logical_shape()[-2];
    const uint32_t K = a.logical_shape()[-1];
    const uint32_t N = b.logical_shape()[-1];

    IDevice* device = a.device();
    auto grid = device->compute_with_storage_grid_size();

    // Split work into UNITS = (row, column-chunk). For small M, splitting columns
    // lets the matmul use more of the grid. Each chunk is a contiguous 16B-aligned
    // sub-range of its row: chunk_cols is a multiple of 4 and N % 4 == 0 (validated),
    // so concurrent cores writing different chunks of the same row page stay aligned.
    const uint32_t total_cores = grid.x * grid.y;
    // Streamed when loading all of A+B into L1 would overflow (large operands), OR for
    // float in/out (the streamed kernel carries the bf16 encode/decode + small L1).
    // ALSO when K or N is not a multiple of 16 (= 64 bytes, the Blackhole DRAM page-transfer
    // granularity): the NON-streamed kernel reads A/B via InterleavedAddrGen with page_size =
    // k_dim*WORD / n_dim*WORD, which mis-addresses rows when that page isn't 64B-aligned (a real
    // bug for K%16!=0 — it shifted row i's A data, only visible under high-magnitude cancellation).
    // The streamed kernel reads via TensorAccessor (alignment-agnostic), so route those there.
    const bool streamed = input_float || input_bf16bits || output_float || output_bf16bits ||
                          (M * K + K * N + (N > 8 ? N : 8)) > kNonStreamedMaxWords ||
                          (K % 16 != 0) || (N % 16 != 0);

    uint32_t nchunks = 1;
    uint32_t chunk_cols = 0;
    if (streamed) {
        // chunk_cols capped so B chunk-columns (K*chunk_cols) fit L1, while aiming for ~grid-many
        // units. MUST be a multiple of gran = 64/asz_in columns: each per-row B read is
        // chunk_cols*asz bytes and Blackhole DRAM transfers at 64-BYTE granularity — a sub-64B-multiple
        // strided read silently corrupts (this was the deep-K accumulation bug). gran = 16 (uint32) /
        // 32 (bf16). (Real model N are multiples of 16, so the ragged last chunk w stays 64B-aligned too.)
        const uint32_t gran = 64u / asz_in;
        uint32_t cc_fit = (kStreamedBColBudget / std::max(1u, K)) / gran * gran;   // round DOWN to gran
        if (cc_fit < gran) {
            cc_fit = gran;
        }
        const uint32_t want_units_per_row = std::max(1u, total_cores / std::max(1u, M));
        uint32_t cc_grid = ((N / want_units_per_row + gran - 1) / gran) * gran;     // round UP to gran
        if (cc_grid < gran) {
            cc_grid = gran;
        }
        chunk_cols = std::min(cc_fit, cc_grid);
        if (chunk_cols > N) {
            chunk_cols = ((N + gran - 1) / gran) * gran;
        }
        nchunks = (N + chunk_cols - 1) / chunk_cols;
    } else {
        if (M < total_cores && N >= 8) {
            const uint32_t want = total_cores / M;   // chunks needed to fill the grid
            const uint32_t max_chunks = N / 4;       // each chunk >= 4 columns (16 bytes)
            nchunks = std::max(1u, std::min(want, max_chunks));
        }
        chunk_cols = (N + nchunks - 1) / nchunks;   // ceil
        chunk_cols = ((chunk_cols + 3) / 4) * 4;    // round up to a multiple of 4
        nchunks = (N + chunk_cols - 1) / chunk_cols;
    }
    const uint32_t total_units = M * nchunks;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2] =
        split_work_to_cores(grid, total_units, /*row_major=*/true);

    // Per-core L1 scratch via circular buffers (used as plain scratch).
    auto mk_cb_bytes = [&](uint32_t index, uint32_t bytes) {
        bytes = ((bytes + WORD - 1) / WORD) * WORD;     // round up to a uint32
        CircularBufferConfig cfg(bytes, {{index, tt::DataFormat::UInt32}});
        cfg.set_page_size(index, bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    };
    auto mk_cb = [&](uint32_t index, uint32_t words) { mk_cb_bytes(index, words * WORD); };
    if (streamed) {
        mk_cb_bytes(0, K * asz_in);                     // one A row (bf16 or code)
        mk_cb_bytes(1, K * chunk_cols * asz_in);        // B chunk-columns (bf16 or code)
        mk_cb_bytes(2, (chunk_cols > 8 ? chunk_cols : 8) * WORD);  // output chunk (uint32-sized; bf16 uses half)
        if (input_float || input_bf16bits) {
            mk_cb_bytes(3, (1u << 16) * WORD);          // 64K-entry bf16->packed-bp16 LUT (fused codec fast path)
        }
    } else {
        mk_cb(0, M * K);                                // all A codes
        mk_cb(1, K * N);                                // all B codes
        mk_cb(2, N > 8 ? N : 8);                        // one output row
    }

    // The streamed kernel reads operands via TensorAccessor (correct for bf16 OR uint32
    // ttnn layouts) — append a/b/c TensorAccessorArgs as compile-time args. The
    // non-streamed kernel uses its own InterleavedAddrGen (uint32 code only).
    std::vector<uint32_t> ct_args;
    if (streamed) {
        TensorAccessorArgs(*a_buffer).append_to(ct_args);
        TensorAccessorArgs(*b_buffer).append_to(ct_args);
        TensorAccessorArgs(*c_buffer).append_to(ct_args);
    }
    auto kernel_id = CreateKernel(
        program,
        streamed ? kStreamedKernelPath : kKernelPath,
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct_args});

    // Iterate the actual worked cores (group ranges), exactly the proven standalone
    // dispatch — guarantees every core CreateKernel was given on all_cores gets
    // runtime args (manual {i/ny, i%ny} reconstruction can desync from the grid).
    std::vector<CoreCoord> cores;
    cores.reserve(num_cores);
    uint32_t unit_start = 0;
    const std::pair<CoreRangeSet, uint32_t> work_groups[] = {
        {core_group_1, units_per_core_g1},
        {core_group_2, units_per_core_g2},
    };
    for (const auto& [group, units] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                uint32_t unit_end = unit_start + units;
                std::vector<uint32_t> args = {
                    (uint32_t)a_buffer->address(),
                    (uint32_t)b_buffer->address(),
                    (uint32_t)c_buffer->address(),
                    M, K, N, unit_start, unit_end, nchunks, chunk_cols, readout_bp32};
                if (streamed) {
                    args.push_back(input_float);      // arg 11 (streamed kernel only)
                    args.push_back(output_float);     // arg 12
                    args.push_back(input_bf16bits);   // arg 13
                    args.push_back(output_bf16bits);  // arg 14
                    // arg 15: allow the fused codec LUT fast path (BPOSIT_NO_LUT=1 forces it off, for
                    // apples-to-apples LUT vs non-LUT validation). Default on.
                    args.push_back(std::getenv("BPOSIT_NO_LUT") ? 0u : 1u);
                }
                SetRuntimeArgs(program, kernel_id, core, args);
                cores.push_back(core);
                unit_start = unit_end;
            }
        }
    }

    return cached_program_t{std::move(program), {.kernel_id = kernel_id, .cores = std::move(cores)}};
}

void BpositQuireMatmulProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const BpositQuireMatmulParams&,
    const BpositQuireMatmulInputs& tensor_args,
    Tensor& output) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;
    uint32_t a_addr = tensor_args.a.buffer()->address();
    uint32_t b_addr = tensor_args.b.buffer()->address();
    uint32_t c_addr = output.buffer()->address();
    auto& runtime_args_by_core = GetRuntimeArgs(program, shared.kernel_id);
    for (const auto& core : shared.cores) {
        auto& args = runtime_args_by_core[core.x][core.y];
        args[0] = a_addr;
        args[1] = b_addr;
        args[2] = c_addr;
    }
}

}  // namespace ttnn::experimental::prim
