// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// bposit_quire_matmul_multicore_perf — THROUGHPUT variant of the proven
// bposit_quire_matmul_multicore. Runs Anomly's EXACT 256-bit Kulisch quire
// b-posit16 MATMUL  C[N,N] = A[N,K] . B[K,N]  (N=128, K=64 → 16384 output
// elements, 1,048,576 exact bp16 products) across the FULL ~140-core Blackhole
// baby (RISC-V) grid with MANY outputs per core, and TIMES the device matmul to
// report the EXACT-QUIRE throughput (the differentiator number).
//
// These are SCALAR-per-core exact-quire ops — NOT bf16-engine FLOPs. The number
// reported is exact-quire dots/sec and exact bp16 products/sec: every product is
// placed UNROUNDED into a 256-bit quire and only the per-output readout rounds.
//
// Each output element C[i,j] is an INDEPENDENT exact-quire dot of length K (no
// cross-core reduction), so the 16384 outputs are distributed across the device's
// baby-core grid via split_work_to_cores; every core runs the SAME proven scalar
// kernel (kernels/quire_matmul_perf_baby.cpp) on a disjoint [out_start,out_end)
// slice. The per-element math is byte-for-byte the on-silicon-proven small
// multicore kernel's, so the parallel result provably equals it.
//
// CORRECTNESS WITHOUT A HUGE-GOLDEN BOTTLENECK: the pure-Python Fraction oracle is
// too slow to gate all 1.05M products, so the host gates only a SAMPLE of S=64
// outputs spread across the grid (baked golden from gen_matmul_perf_golden.py)
// plus one full gate quire, while TIMING the FULL N*N matmul.
//
// The small multicore example (bposit_quire_matmul_multicore, the correctness
// gate) is left UNTOUCHED — this is an additive, separate example.
//
// Host program + dispatch mirror the proven examples:
//   - programming_examples/bposit_quire_matmul_multicore/bposit_quire_matmul_multicore.cpp
//   - programming_examples/vecadd_multi_core/vecadd_multi_core.cpp

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
#include <chrono>

// Baked golden (generated from the canonical oracle by gen_matmul_perf_golden.py).
#include "quire_matmul_perf_golden_cases.h"

// Host-side reference reuses the SAME golden-exact numerics the kernel uses, so
// the host independently recomputes the expected quire/readout for the SAMPLED
// outputs from the input codes (belt-and-suspenders). These freestanding headers
// carry the full op set; the host calls a subset, so suppress -Wunused-function.
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
    std::ostringstream os;
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 4; ++b)
            os << std::hex << std::setw(2) << std::setfill('0') << ((q[i] >> (8 * b)) & 0xFFu);
    return os.str();
}

int main(int /*argc*/, char** /*argv*/) {
    constexpr int QLIMBS = 8;
    constexpr uint32_t WORD = sizeof(uint32_t);

    const int N = QGOLD_MMP_N;  // 128
    const int K = QGOLD_MMP_K;  // 64
    const int S = QGOLD_MMP_S;  // 64 sampled outputs gated
    const int A_COUNT = N * K;  // 8192
    const int B_COUNT = K * N;  // 8192
    const int C_COUNT = N * N;  // 16384 readouts
    const int GATE_O = QGOLD_MMP_GATE_O;
    const int GATE_I = QGOLD_MMP_GATE_I;
    const int GATE_J = QGOLD_MMP_GATE_J;

    std::vector<uint32_t> a_words(A_COUNT), b_words(B_COUNT);
    for (int i = 0; i < A_COUNT; ++i) a_words[i] = (uint32_t)(QGOLD_MMP_A_CODES[i] & 0xFFFF);
    for (int i = 0; i < B_COUNT; ++i) b_words[i] = (uint32_t)(QGOLD_MMP_B_CODES[i] & 0xFFFF);

    // ======================= HOST SELF-CHECK (belt) ==========================
    // Re-derive the SAMPLED golden from the reused exact headers BEFORE any device
    // run: for each sampled output (a) recompute its exact quire dot + readout and
    // confirm the baked readout; (b) re-derive EACH product alone == bposit16_mul
    // implicitly via the readout chain; (c) re-accumulate in REVERSE K-order, quire
    // identical (associativity); plus the full gate quire.
    bool readout_ok = true, assoc_ok = true, gate_ok = true;
    for (int s = 0; s < S; ++s) {
        int o = QGOLD_MMP_SAMPLE_O[s];
        int i = o / N, j = o % N;
        unsigned q[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
        unsigned q_rev[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k = 0; k < K; ++k) {
            unsigned c[QLIMBS];
            bp16_prod_to_q256((int)(a_words[i * K + k] & 0xFFFF),
                              (int)(b_words[k * N + j] & 0xFFFF), c);
            q256_add(q, c);
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
        if (ref_readout != (QGOLD_MMP_SAMPLE_READOUT[s] & 0xFFFF)) readout_ok = false;
        if (o == GATE_O)
            for (int l = 0; l < QLIMBS; ++l)
                if (q[l] != QGOLD_MMP_GATE_QUIRE[l]) gate_ok = false;
    }
    if (!(readout_ok && assoc_ok && gate_ok)) {
        std::cout << "FAIL: baked PERF MATMUL sampled golden disagrees with the reused "
                     "headers (host self-check) — regenerate from oracle (readout="
                  << readout_ok << " assoc=" << assoc_ok << " gate=" << gate_ok << ")\n";
        return 1;
    }

    // ============================ DEVICE SETUP ===============================
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();

    // ---- buffers ------------------------------------------------------------
    // INPUTS A,B: ONE page each (whole matrix), the proven single-page/bulk read
    // rule; every core reads its own private copy (A=32 KB, B=32 KB — fits the
    // 1536 KB L1). OUTPUT C: 1 word per page (C_COUNT pages) so each core writes
    // ONLY its own output slot(s) without colliding. GATE: one 8-word page.
    const size_t a_bytes = (size_t)A_COUNT * WORD;
    const size_t b_bytes = (size_t)B_COUNT * WORD;
    const size_t c_bytes = (size_t)C_COUNT * WORD;
    const size_t gate_bytes = (size_t)QLIMBS * WORD;
    const size_t a_l1_bytes = a_bytes;
    const size_t b_l1_bytes = b_bytes;
    const size_t c_l1_bytes = (size_t)QLIMBS * WORD;  // 1 readout word OR 8 gate limbs

    auto mk_paged = [&](size_t total, size_t page, BufferType type) {
        distributed::DeviceLocalBufferConfig lc{.page_size = page, .buffer_type = type};
        distributed::ReplicatedBufferConfig bc{.size = total};
        return distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    };
    auto a_dram = mk_paged(a_bytes, a_bytes, BufferType::DRAM);
    auto b_dram = mk_paged(b_bytes, b_bytes, BufferType::DRAM);
    auto c_dram = mk_paged(c_bytes, WORD, BufferType::DRAM);     // C_COUNT pages
    auto gate_dram = mk_paged(gate_bytes, gate_bytes, BufferType::DRAM);
    auto a_l1 = mk_paged(a_l1_bytes, a_l1_bytes, BufferType::L1);
    auto b_l1 = mk_paged(b_l1_bytes, b_l1_bytes, BufferType::L1);
    auto c_l1 = mk_paged(c_l1_bytes, c_l1_bytes, BufferType::L1);

    // ---- core grid + work split --------------------------------------------
    // Distribute the C_COUNT (=16384) flat output indices across the device's full
    // baby-core grid (14x10 = 140 on Blackhole). split_work_to_cores gives each
    // core a disjoint, near-equal [out_start,out_end) slice (~117 outputs/core).
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2,
          outs_per_core_g1, outs_per_core_g2] =
        tt::tt_metal::split_work_to_cores(core_grid, (uint32_t)C_COUNT, /*row_major=*/true);

    // ONE CreateKernel over the whole used core range — same proven scalar kernel.
    auto kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bposit_quire_matmul_multicore_perf/kernels/quire_matmul_perf_baby.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Per-core runtime args: each core gets a disjoint [out_start,out_end) slice.
    auto work_groups = {std::make_pair(core_group_1, outs_per_core_g1),
                        std::make_pair(core_group_2, outs_per_core_g2)};
    uint32_t out_start = 0;
    uint32_t max_outs_per_core = 0;
    for (const auto& [group, outs_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                uint32_t out_end = out_start + outs_per_core;
                // arg order MUST match get_arg_val<>() in quire_matmul_perf_baby.cpp.
                SetRuntimeArgs(program, kernel, core,
                               {(uint32_t)a_dram->address(), (uint32_t)b_dram->address(),
                                (uint32_t)c_dram->address(), (uint32_t)a_l1->address(),
                                (uint32_t)b_l1->address(), (uint32_t)c_l1->address(),
                                (uint32_t)N, (uint32_t)K,
                                out_start, out_end,
                                (uint32_t)GATE_O, (uint32_t)gate_dram->address()});
                out_start = out_end;
                if (outs_per_core > max_outs_per_core) max_outs_per_core = outs_per_core;
            }
        }
    }

    // operands resident once (read-only, reused across warm-up + timed run).
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_words, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_words, /*blocking=*/false);
    workload.add_program(device_range, std::move(program));

    std::vector<uint32_t> c_result(C_COUNT, 0);
    std::vector<uint32_t> gate_result(QLIMBS, 0);

    // ---- WARM-UP run (excludes one-time JIT/compile + first-dispatch setup) --
    // The first EnqueueMeshWorkload triggers kernel JIT + program-cache fill; we
    // run + finish it once and DISCARD the timing, then time a second dispatch of
    // the SAME cached workload so the number reflects steady-state compute+NoC.
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    // ---- TIMED run: time EnqueueMeshWorkload -> Finish (the matmul itself) ---
    auto t0 = std::chrono::high_resolution_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);  // block until the device finishes the FULL N*N matmul
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // read back AFTER timing (read-back transfer is excluded from the matmul time).
    distributed::EnqueueReadMeshBuffer(cq, c_result, c_dram, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(cq, gate_result, gate_dram, /*blocking=*/true);

    // ===================== GATE vs GOLDEN (SAMPLE ONLY) ======================
    int bad_c = 0, matched_sample = 0;
    for (int s = 0; s < S; ++s) {
        int o = QGOLD_MMP_SAMPLE_O[s];
        int dev = (int)(c_result[o] & 0xFFFF);
        int gold = QGOLD_MMP_SAMPLE_READOUT[s] & 0xFFFF;
        if (dev == gold) {
            ++matched_sample;
        } else if (++bad_c <= 16) {
            int i = o / N, j = o % N;
            std::cout << "  C[" << i << "," << j << "] (o=" << o << ") device=0x" << std::hex
                      << dev << " golden=0x" << gold << std::dec << "\n";
        }
    }
    uint32_t dev_q[QLIMBS];
    for (int i = 0; i < QLIMBS; ++i) dev_q[i] = gate_result[i];
    int bad_q = 0;
    for (int i = 0; i < QLIMBS; ++i) {
        if (dev_q[i] != QGOLD_MMP_GATE_QUIRE[i]) {
            if (++bad_q <= 8)
                std::cout << "  gate limb[" << i << "] device=0x" << std::hex << dev_q[i]
                          << " golden=0x" << QGOLD_MMP_GATE_QUIRE[i] << std::dec << "\n";
        }
    }
    bool pass = (bad_c == 0) && (bad_q == 0);

    // ============================ THROUGHPUT =================================
    const double seconds = ms / 1000.0;
    const long long dots = (long long)N * N;             // exact-quire dots = N*N outputs
    const long long products = (long long)N * N * K;     // exact bp16 products = N*N*K
    const double dots_per_s = seconds > 0 ? (double)dots / seconds : 0.0;
    const double prods_per_s = seconds > 0 ? (double)products / seconds : 0.0;

    std::cout << "[mm-perf] grid " << core_grid.x << "x" << core_grid.y
              << ", cores used = " << num_cores
              << ", outputs = " << C_COUNT << " (~" << max_outs_per_core << "/core)\n";
    std::cout << "[mm-perf] device gate(" << GATE_I << "," << GATE_J << ") quire = "
              << quire_to_hex_le(dev_q) << "\n";
    std::cout << "[mm-perf] golden gate(" << GATE_I << "," << GATE_J << ") quire = "
              << QGOLD_MMP_GATE_QUIRE_HEX << "\n";
    std::cout << "[mm-perf] sample bit-exact = " << matched_sample << "/" << S
              << ", gate limbs = " << (QLIMBS - bad_q) << "/" << QLIMBS << "\n";

    // The headline throughput line (honest: SCALAR-per-core EXACT-QUIRE ops).
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "PERF: " << N << "x" << K << "x" << N
              << " exact-quire matmul across " << num_cores << " cores, "
              << ms << " ms, "
              << std::setprecision(0) << dots_per_s << " exact-quire-dots/s ("
              << prods_per_s << " exact-products/s); sample "
              << matched_sample << "/" << S << " bit-exact\n";

    std::cout << (pass ? "PASS" : "FAIL")
              << ": exact bp16 quire PERF MATMUL (" << N << "x" << K << "x" << N
              << ", " << dots << " dots, " << products << " exact products) across "
              << num_cores << " cores on Blackhole; sample " << matched_sample << "/"
              << S << " + gate " << (QLIMBS - bad_q) << "/" << QLIMBS << " bit-exact\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
