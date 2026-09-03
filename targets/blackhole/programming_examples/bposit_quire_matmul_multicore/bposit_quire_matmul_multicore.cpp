// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// bposit_quire_matmul_multicore — run Anomly's EXACT 256-bit Kulisch quire
// b-posit16 MATMUL  C[M,N] = A[M,K] · B[K,N]  (M=N=8, K=16, 64 output elements)
// across MANY Blackhole baby (RISC-V) cores, bit-exact vs the the scalar-kernel-oracle
// golden. This is the ROBUST throughput path: it reuses the ALREADY-BIT-EXACT
// scalar quire-dot kernel verbatim and only adds the multi-core dispatch.
//
// Each output element C[i,j] is an INDEPENDENT exact-quire dot of length K
// (no cross-core reduction), so the 64 outputs are distributed one-per-core onto
// a CoreRange grid. Every core runs the SAME proven scalar kernel
// (kernels/quire_matmul_mc_baby.cpp) on a disjoint [out_start,out_end) slice of
// the flat output indices; the per-element math is byte-for-byte the single-core
// kernel's (quire_matmul_baby.cpp), so the parallel result provably equals it.
//
// The single-core matmul example (bposit_quire_reduce, mode "matmul") is left
// UNTOUCHED — this is an additive, separate example.
//
// Host program + dispatch mirror the proven examples:
//   - programming_examples/bposit_quire_reduce/bposit_quire_reduce.cpp  (run_matmul host path, golden gate, host self-check)
//   - programming_examples/vecadd_multi_core/vecadd_multi_core.cpp      (CoreRange grid, split_work_to_cores, per-core SetRuntimeArgs)

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

// Baked golden (generated from the canonical oracle by gen_matmul_mc_golden.py).
#include "quire_matmul_mc_golden_cases.h"

// Host-side reference reuses the SAME golden-exact numerics the kernel uses, so
// the host independently recomputes the expected quire/readout from the input
// codes (belt-and-suspenders: the baked golden is checked against the headers at
// startup, and the device result against both). These freestanding headers carry
// the full op set; the host calls a subset, so suppress -Werror=unused-function.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../../kernel/bp16_quire.h"
#include "../../kernel/bp16_encode.h"
#pragma GCC diagnostic pop

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static std::string quire_to_hex_le(const uint32_t q[8]) {
    // 32-byte little-endian hex, matching golden "quire_le_hex".
    std::ostringstream os;
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 4; ++b)
            os << std::hex << std::setw(2) << std::setfill('0') << ((q[i] >> (8 * b)) & 0xFFu);
    return os.str();
}

int main(int /*argc*/, char** /*argv*/) {
    constexpr int QLIMBS = 8;
    constexpr uint32_t WORD = sizeof(uint32_t);

    const int M = QGOLD_MMMC_M;  // 8
    const int K = QGOLD_MMMC_K;  // 16
    const int N = QGOLD_MMMC_N;  // 8
    const int A_COUNT = M * K;   // 128
    const int B_COUNT = K * N;   // 128
    const int C_COUNT = M * N;   // 64 readouts (one per core)
    const int GATE_I = QGOLD_MMMC_GATE_I;
    const int GATE_J = QGOLD_MMMC_GATE_J;
    const int GATE_O = GATE_I * N + GATE_J;  // flat gate index

    std::vector<uint32_t> a_words(A_COUNT), b_words(B_COUNT);
    for (int i = 0; i < A_COUNT; ++i) a_words[i] = (uint32_t)(QGOLD_MMMC_A_CODES[i] & 0xFFFF);
    for (int i = 0; i < B_COUNT; ++i) b_words[i] = (uint32_t)(QGOLD_MMMC_B_CODES[i] & 0xFFFF);

    // ======================= HOST SELF-CHECK (belt) ==========================
    // Re-derive the golden from the reused exact headers BEFORE any device run:
    //   (a) recompute every output's exact quire dot + readout, confirm the baked
    //       per-output C readouts AND the baked full quire of output (gi,gj);
    //   (b) re-derive EACH per-element product alone == baked bposit16_mul code;
    //   (c) re-accumulate each output in REVERSE K-order, quire identical (assoc).
    bool per_elem_ok = true, readout_ok = true, assoc_ok = true, gate_ok = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            unsigned q[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
            unsigned q_rev[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int k = 0; k < K; ++k) {
                unsigned c[QLIMBS];
                int ac = (int)(a_words[i * K + k] & 0xFFFF);
                int bc = (int)(b_words[k * N + j] & 0xFFFF);
                bp16_prod_to_q256(ac, bc, c);
                q256_add(q, c);
                int per = bp16_encode_quire256(c) & 0xFFFF;
                int baked = QGOLD_MMMC_PROD_CODES[(i * K + k) * N + j] & 0xFFFF;
                if (per != baked) {
                    per_elem_ok = false;
                    std::cout << "  prod[" << i << "," << k << "," << j << "] exact=0x"
                              << std::hex << per << " baked-mul=0x" << baked << std::dec << "\n";
                }
            }
            for (int k = K - 1; k >= 0; --k) {  // reverse-order accumulation
                unsigned c[QLIMBS];
                bp16_prod_to_q256((int)(a_words[i * K + k] & 0xFFFF),
                                  (int)(b_words[k * N + j] & 0xFFFF), c);
                q256_add(q_rev, c);
            }
            for (int l = 0; l < QLIMBS; ++l)
                if (q[l] != q_rev[l]) assoc_ok = false;
            int ref_readout = bp16_encode_quire256(q) & 0xFFFF;
            if (ref_readout != (QGOLD_MMMC_C_READOUT[i * N + j] & 0xFFFF)) readout_ok = false;
            if (i == GATE_I && j == GATE_J)
                for (int l = 0; l < QLIMBS; ++l)
                    if (q[l] != QGOLD_MMMC_GATE_QUIRE[l]) gate_ok = false;
        }
    }
    if (!(per_elem_ok && readout_ok && assoc_ok && gate_ok)) {
        std::cout << "FAIL: baked MULTI-CORE MATMUL golden disagrees with the reused "
                     "headers (host self-check) — golden data drifted, regenerate from "
                     "oracle (per_elem=" << per_elem_ok << " readout=" << readout_ok
                  << " assoc=" << assoc_ok << " gate=" << gate_ok << ")\n";
        return 1;
    }

    // ============================ DEVICE RUN =================================
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();

    // ---- buffers ------------------------------------------------------------
    // INPUTS A,B: ONE page each (whole matrix) — the proven single-page/bulk
    // read rule; every core reads its own private copy. OUTPUT C: 1 word per page
    // (C_COUNT pages) so each core writes ONLY its own output slot(s) without
    // colliding. GATE: one 8-word page written by the single owning core.
    const size_t a_bytes = (size_t)A_COUNT * WORD;
    const size_t b_bytes = (size_t)B_COUNT * WORD;
    const size_t c_bytes = (size_t)C_COUNT * WORD;
    const size_t gate_bytes = (size_t)QLIMBS * WORD;
    // per-core L1 scratch sizes (each core holds full A, full B, and >=8 output words)
    const size_t a_l1_bytes = a_bytes;
    const size_t b_l1_bytes = b_bytes;
    const size_t c_l1_bytes = (size_t)QLIMBS * WORD;  // 1 readout word OR 8 gate limbs

    auto mk_paged = [&](size_t total, size_t page, BufferType type) {
        distributed::DeviceLocalBufferConfig lc{.page_size = page, .buffer_type = type};
        distributed::ReplicatedBufferConfig bc{.size = total};
        return distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    };
    // inputs: single page (whole buffer). output: 1-word pages. gate: single page.
    auto a_dram = mk_paged(a_bytes, a_bytes, BufferType::DRAM);
    auto b_dram = mk_paged(b_bytes, b_bytes, BufferType::DRAM);
    auto c_dram = mk_paged(c_bytes, WORD, BufferType::DRAM);     // C_COUNT pages
    auto gate_dram = mk_paged(gate_bytes, gate_bytes, BufferType::DRAM);
    auto a_l1 = mk_paged(a_l1_bytes, a_l1_bytes, BufferType::L1);
    auto b_l1 = mk_paged(b_l1_bytes, b_l1_bytes, BufferType::L1);
    auto c_l1 = mk_paged(c_l1_bytes, c_l1_bytes, BufferType::L1);

    // ---- core grid + work split --------------------------------------------
    // Distribute the C_COUNT (=64) flat output indices across the device's
    // baby-core grid. On Blackhole the storage grid is large enough that the
    // 64 outputs map one-per-core onto an 8x8 sub-grid; split_work_to_cores
    // handles whatever grid the device exposes (≥1 output/core, exact split).
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2,
          outs_per_core_g1, outs_per_core_g2] =
        tt::tt_metal::split_work_to_cores(core_grid, (uint32_t)C_COUNT, /*row_major=*/true);

    // ONE CreateKernel over the whole used core range — same proven scalar kernel.
    auto kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bposit_quire_matmul_multicore/kernels/quire_matmul_mc_baby.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Per-core runtime args: each core gets a disjoint [out_start,out_end) slice.
    auto work_groups = {std::make_pair(core_group_1, outs_per_core_g1),
                        std::make_pair(core_group_2, outs_per_core_g2)};
    uint32_t out_start = 0;
    for (const auto& [group, outs_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                uint32_t out_end = out_start + outs_per_core;
                // arg order MUST match get_arg_val<>() in quire_matmul_mc_baby.cpp.
                SetRuntimeArgs(program, kernel, core,
                               {(uint32_t)a_dram->address(), (uint32_t)b_dram->address(),
                                (uint32_t)c_dram->address(), (uint32_t)a_l1->address(),
                                (uint32_t)b_l1->address(), (uint32_t)c_l1->address(),
                                (uint32_t)M, (uint32_t)K, (uint32_t)N,
                                out_start, out_end,
                                (uint32_t)GATE_O, (uint32_t)gate_dram->address()});
                out_start = out_end;
            }
        }
    }

    distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_words, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_words, /*blocking=*/false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<uint32_t> c_result(C_COUNT, 0);
    distributed::EnqueueReadMeshBuffer(cq, c_result, c_dram, /*blocking=*/true);
    std::vector<uint32_t> gate_result(QLIMBS, 0);
    distributed::EnqueueReadMeshBuffer(cq, gate_result, gate_dram, /*blocking=*/true);

    // ============================ GATE vs GOLDEN =============================
    int bad_c = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int dev = (int)(c_result[i * N + j] & 0xFFFF);
            int gold = QGOLD_MMMC_C_READOUT[i * N + j] & 0xFFFF;
            if (dev != gold) {
                if (++bad_c <= 16)
                    std::cout << "  C[" << i << "," << j << "] device=0x" << std::hex << dev
                              << " golden=0x" << gold << std::dec << "\n";
            }
        }
    }
    uint32_t dev_q[QLIMBS];
    for (int i = 0; i < QLIMBS; ++i) dev_q[i] = gate_result[i];
    int bad_q = 0;
    for (int i = 0; i < QLIMBS; ++i) {
        if (dev_q[i] != QGOLD_MMMC_GATE_QUIRE[i]) {
            if (++bad_q <= 8)
                std::cout << "  gate limb[" << i << "] device=0x" << std::hex << dev_q[i]
                          << " golden=0x" << QGOLD_MMMC_GATE_QUIRE[i] << std::dec << "\n";
        }
    }

    const int total_checks = C_COUNT + QLIMBS;  // 64 readouts + 8 quire limbs
    const int matched = (C_COUNT - bad_c) + (QLIMBS - bad_q);
    bool pass = (bad_c == 0) && (bad_q == 0);

    std::cout << "[mm-mc] cores used = " << num_cores << " (grid " << core_grid.x
              << "x" << core_grid.y << "), outputs = " << C_COUNT
              << " (one+ per core)\n";
    std::cout << "[mm-mc] device C readouts (row-major):";
    for (int o = 0; o < C_COUNT; ++o) {
        if (o % N == 0) std::cout << "\n   ";
        std::cout << " 0x" << std::hex << std::setw(4) << std::setfill('0') << (c_result[o] & 0xFFFF);
    }
    std::cout << std::dec << "\n[mm-mc] golden C readouts (row-major):";
    for (int o = 0; o < C_COUNT; ++o) {
        if (o % N == 0) std::cout << "\n   ";
        std::cout << " 0x" << std::hex << std::setw(4) << std::setfill('0') << (QGOLD_MMMC_C_READOUT[o] & 0xFFFF);
    }
    std::cout << std::dec << "\n";
    std::cout << "[mm-mc] device gate(" << GATE_I << "," << GATE_J << ") quire = " << quire_to_hex_le(dev_q) << "\n";
    std::cout << "[mm-mc] golden gate(" << GATE_I << "," << GATE_J << ") quire = " << QGOLD_MMMC_GATE_QUIRE_HEX << "\n";
    std::cout << (pass ? "PASS" : "FAIL")
              << ": exact bp16 quire MATMUL (" << M << "x" << K << "x" << N
              << ") across " << num_cores << " cores on Blackhole, "
              << matched << "/" << total_checks << " match golden\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
