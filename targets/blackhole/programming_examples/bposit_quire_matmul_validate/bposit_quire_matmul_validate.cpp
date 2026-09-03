// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// bposit_quire_matmul_validate — RIGOROUS FULL-VALIDATION of Anomly's EXACT
// 256-bit Kulisch quire b-posit16 MATMUL  C[M,N] = A[M,K] . B[K,N]  (M=N=16,
// K=16) on real Blackhole silicon. Runs SEEDS=8 DISTINCT RANDOM matmuls and
// gates EVERY one of the M*N=256 outputs of EVERY seed bit-exact vs the exact
// 256-bit-quire oracle reference — 2048 fully-validated outputs and 32768 exact
// bp16 products — plus a full 256-bit quire byte-gate on one output per seed.
//
// WHY this example exists (correctness hardening): the existing matmul examples
// validate either small tiles FULLY (multicore, 64/64) or large tiles SAMPLED
// (perf, 64/16384). A paper reviewer / skeptic wants "we ran N random MxKxN
// matrices and ALL M*N*SEEDS outputs were bit-exact vs the exact-quire
// reference." This converts the headline claim from "sampled" to "fully
// validated across random data". There is NO sampling and NO tolerance here:
// every single output must match the oracle exactly or the run FAILS.
//
// REUSE (cited): this is byte-for-byte the proven multi-core matmul host +
// kernel, with a host-side SEED LOOP wrapped around the dispatch:
//   - programming_examples/bposit_quire_matmul_multicore/bposit_quire_matmul_multicore.cpp
//       (run host path, split_work_to_cores grid, per-core SetRuntimeArgs, gate)
//   - programming_examples/bposit_quire_matmul_multicore/kernels/quire_matmul_mc_baby.cpp
//       (the scalar exact-quire dot — reused VERBATIM as quire_matmul_validate_baby.cpp)
//   - programming_examples/bposit_quire_matmul_multicore_perf/...  (re-enqueue of a
//       cached workload after re-uploading operands — the per-seed re-dispatch pattern)
// The proven single-/multi-core matmul examples are left UNTOUCHED — additive.
//
// Each output element C[i,j] is an INDEPENDENT exact-quire dot of length K (no
// cross-core reduction); one device dispatch computes one seed's full matmul, and
// the host re-uploads A_seed,B_seed and re-enqueues the SAME cached workload once
// per seed. The per-element math is byte-identical to the on-silicon-proven
// kernel, so the parallel, multi-seed result provably equals the scalar oracle.

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

// Baked golden (generated from the canonical oracle by gen_matmul_validate_golden.py).
#include "quire_matmul_validate_golden_cases.h"

// Host-side reference reuses the SAME golden-exact numerics the kernel uses, so
// the host independently recomputes the expected quire/readout for EVERY output of
// EVERY seed from the input codes (belt-and-suspenders): the baked golden is
// checked against the headers at startup, and the device result against the baked
// golden. These freestanding headers carry the full op set; the host calls a
// subset, so suppress -Werror=unused-function.
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
    // 32-byte little-endian hex, matching golden "*_GATE_QUIRE_HEX".
    std::ostringstream os;
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 4; ++b)
            os << std::hex << std::setw(2) << std::setfill('0') << ((q[i] >> (8 * b)) & 0xFFu);
    return os.str();
}

int main(int /*argc*/, char** /*argv*/) {
    constexpr int QLIMBS = 8;
    constexpr uint32_t WORD = sizeof(uint32_t);

    const int M = QGOLD_MMV_M;          // 16
    const int K = QGOLD_MMV_K;          // 16
    const int N = QGOLD_MMV_N;          // 16
    const int SEEDS = QGOLD_MMV_SEEDS;  // 8
    const int A_COUNT = M * K;          // 256
    const int B_COUNT = K * N;          // 256
    const int C_COUNT = M * N;          // 256 readouts (one per core)
    const int GATE_I = QGOLD_MMV_GATE_I;
    const int GATE_J = QGOLD_MMV_GATE_J;
    const int GATE_O = GATE_I * N + GATE_J;  // flat gate index

    // ======================= HOST SELF-CHECK (belt) ==========================
    // BEFORE any device run, re-derive the ENTIRE golden from the reused exact
    // headers for ALL seeds: for every output (a) recompute its exact quire dot +
    // readout, confirm the baked readout; (b) re-derive EACH per-element product
    // alone == baked bposit16_mul; (c) re-accumulate in REVERSE K-order, quire
    // identical (associativity); plus the per-seed full gate quire. A drift FAILS
    // loudly here, before touching the device.
    {
        bool per_elem_ok = true, readout_ok = true, assoc_ok = true, gate_ok = true;
        for (int s = 0; s < SEEDS; ++s) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    unsigned q[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
                    unsigned q_rev[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (int k = 0; k < K; ++k) {
                        unsigned c[QLIMBS];
                        int ac = (int)(QGOLD_MMV_A_CODES[s][i * K + k] & 0xFFFF);
                        int bc = (int)(QGOLD_MMV_B_CODES[s][k * N + j] & 0xFFFF);
                        bp16_prod_to_q256(ac, bc, c);
                        q256_add(q, c);
                        int per = bp16_encode_quire256(c) & 0xFFFF;
                        int baked = QGOLD_MMV_PROD_CODES[s][(i * K + k) * N + j] & 0xFFFF;
                        if (per != baked) {
                            per_elem_ok = false;
                            std::cout << "  seed " << s << " prod[" << i << "," << k << "," << j
                                      << "] exact=0x" << std::hex << per << " baked-mul=0x"
                                      << baked << std::dec << "\n";
                        }
                    }
                    for (int k = K - 1; k >= 0; --k) {  // reverse-order accumulation
                        unsigned c[QLIMBS];
                        bp16_prod_to_q256((int)(QGOLD_MMV_A_CODES[s][i * K + k] & 0xFFFF),
                                          (int)(QGOLD_MMV_B_CODES[s][k * N + j] & 0xFFFF), c);
                        q256_add(q_rev, c);
                    }
                    for (int l = 0; l < QLIMBS; ++l)
                        if (q[l] != q_rev[l]) assoc_ok = false;
                    int ref_readout = bp16_encode_quire256(q) & 0xFFFF;
                    if (ref_readout != (QGOLD_MMV_C_READOUT[s][i * N + j] & 0xFFFF))
                        readout_ok = false;
                    if (i == GATE_I && j == GATE_J)
                        for (int l = 0; l < QLIMBS; ++l)
                            if (q[l] != QGOLD_MMV_GATE_QUIRE[s][l]) gate_ok = false;
                }
            }
        }
        if (!(per_elem_ok && readout_ok && assoc_ok && gate_ok)) {
            std::cout << "FAIL: baked FULL-VALIDATION golden disagrees with the reused "
                         "headers (host self-check) — golden data drifted, regenerate from "
                         "oracle (per_elem=" << per_elem_ok << " readout=" << readout_ok
                      << " assoc=" << assoc_ok << " gate=" << gate_ok << ")\n";
            return 1;
        }
    }

    // ============================ DEVICE SETUP ===============================
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();

    // ---- buffers (created ONCE; operands re-uploaded per seed) --------------
    // INPUTS A,B: ONE page each (whole matrix) — the proven single-page/bulk read
    // rule; every core reads its own private copy. OUTPUT C: 1 word per page
    // (C_COUNT pages) so each core writes ONLY its own slot. GATE: one 8-word page.
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

    // ---- core grid + work split (constant across seeds) ---------------------
    // Distribute the C_COUNT (=256) flat output indices across the device's
    // baby-core grid; split_work_to_cores gives each core a disjoint, near-equal
    // [out_start,out_end) slice. The grid + slicing are identical every seed
    // (only the operand DATA changes), so build the workload ONCE and re-enqueue.
    auto core_grid = mesh_device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2,
          outs_per_core_g1, outs_per_core_g2] =
        tt::tt_metal::split_work_to_cores(core_grid, (uint32_t)C_COUNT, /*row_major=*/true);

    // ONE CreateKernel over the whole used core range — same proven scalar kernel.
    auto kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bposit_quire_matmul_validate/kernels/quire_matmul_validate_baby.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Per-core runtime args: each core gets a disjoint [out_start,out_end) slice.
    // The buffer addresses, dims, slices and gate index are the SAME for every
    // seed (only the bytes inside a_dram/b_dram change), so set them once.
    auto work_groups = {std::make_pair(core_group_1, outs_per_core_g1),
                        std::make_pair(core_group_2, outs_per_core_g2)};
    uint32_t out_start = 0;
    uint32_t max_outs_per_core = 0;
    for (const auto& [group, outs_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                uint32_t out_end = out_start + outs_per_core;
                // arg order MUST match get_arg_val<>() in quire_matmul_validate_baby.cpp.
                SetRuntimeArgs(program, kernel, core,
                               {(uint32_t)a_dram->address(), (uint32_t)b_dram->address(),
                                (uint32_t)c_dram->address(), (uint32_t)a_l1->address(),
                                (uint32_t)b_l1->address(), (uint32_t)c_l1->address(),
                                (uint32_t)M, (uint32_t)K, (uint32_t)N,
                                out_start, out_end,
                                (uint32_t)GATE_O, (uint32_t)gate_dram->address()});
                out_start = out_end;
                if (outs_per_core > max_outs_per_core) max_outs_per_core = outs_per_core;
            }
        }
    }
    workload.add_program(device_range, std::move(program));

    // ===================== PER-SEED DEVICE RUN + FULL GATE ===================
    // For each random seed: upload A_seed,B_seed; re-enqueue the SAME cached
    // workload; BLOCKING read back C and the gate quire; gate EVERY output and the
    // full quire bit-exact. The reads are BLOCKING so the host never inspects a
    // half-written buffer (no non-blocking EnqueueReadMeshBuffer here).
    long long total_outputs = 0;     // grand total of outputs gated across all seeds
    long long matched_outputs = 0;   // grand total that matched bit-exact
    long long total_products = 0;    // exact bp16 products validated (M*N*K per seed)
    int gate_pass = 0;               // full-quire byte-gates that passed (one per seed)
    int seeds_pass = 0;              // seeds with ALL outputs + gate bit-exact
    bool all_pass = true;

    auto t0 = std::chrono::high_resolution_clock::now();
    long long rows_done = 0;         // M rows per seed -> SEEDS*M rows for a rows/s bonus

    for (int s = 0; s < SEEDS; ++s) {
        std::vector<uint32_t> a_words(A_COUNT), b_words(B_COUNT);
        for (int i = 0; i < A_COUNT; ++i) a_words[i] = (uint32_t)(QGOLD_MMV_A_CODES[s][i] & 0xFFFF);
        for (int i = 0; i < B_COUNT; ++i) b_words[i] = (uint32_t)(QGOLD_MMV_B_CODES[s][i] & 0xFFFF);

        // upload this seed's operands (blocking so the workload reads fresh data),
        // then run and BLOCKING-read the results.
        distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_words, /*blocking=*/true);
        distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_words, /*blocking=*/true);
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

        std::vector<uint32_t> c_result(C_COUNT, 0);
        std::vector<uint32_t> gate_result(QLIMBS, 0);
        distributed::EnqueueReadMeshBuffer(cq, c_result, c_dram, /*blocking=*/true);
        distributed::EnqueueReadMeshBuffer(cq, gate_result, gate_dram, /*blocking=*/true);

        // ---- gate EVERY output of this seed bit-exact vs oracle golden -------
        int bad_c = 0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int dev = (int)(c_result[i * N + j] & 0xFFFF);
                int gold = QGOLD_MMV_C_READOUT[s][i * N + j] & 0xFFFF;
                ++total_outputs;
                if (dev == gold) {
                    ++matched_outputs;
                } else if (++bad_c <= 8) {
                    std::cout << "  seed " << s << " C[" << i << "," << j << "] device=0x"
                              << std::hex << dev << " golden=0x" << gold << std::dec << "\n";
                }
            }
        }
        total_products += (long long)M * N * K;
        rows_done += M;

        // ---- full 256-bit quire byte-gate on this seed's gated output --------
        uint32_t dev_q[QLIMBS];
        for (int i = 0; i < QLIMBS; ++i) dev_q[i] = gate_result[i];
        int bad_q = 0;
        for (int i = 0; i < QLIMBS; ++i) {
            if (dev_q[i] != QGOLD_MMV_GATE_QUIRE[s][i]) {
                if (++bad_q <= QLIMBS)
                    std::cout << "  seed " << s << " gate limb[" << i << "] device=0x"
                              << std::hex << dev_q[i] << " golden=0x"
                              << QGOLD_MMV_GATE_QUIRE[s][i] << std::dec << "\n";
            }
        }
        bool seed_ok = (bad_c == 0) && (bad_q == 0);
        if (bad_q == 0) ++gate_pass;
        if (seed_ok) ++seeds_pass; else all_pass = false;

        std::cout << "[mm-val] seed " << s << ": outputs " << (M * N - bad_c) << "/" << (M * N)
                  << " bit-exact, gate(" << GATE_I << "," << GATE_J << ") quire "
                  << (bad_q == 0 ? "MATCH" : "MISMATCH") << "  dev=" << quire_to_hex_le(dev_q)
                  << "\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double seconds = ms / 1000.0;
    double rows_per_s = seconds > 0 ? (double)rows_done / seconds : 0.0;

    // ============================ VERDICT ===================================
    std::cout << "[mm-val] grid " << core_grid.x << "x" << core_grid.y
              << ", cores used = " << num_cores << ", outputs/seed = " << C_COUNT
              << " (~" << max_outs_per_core << "/core), seeds = " << SEEDS << "\n";
    std::cout << "[mm-val] seeds fully bit-exact = " << seeds_pass << "/" << SEEDS
              << ", full-quire byte-gates = " << gate_pass << "/" << SEEDS << "\n";
    std::cout << std::fixed << std::setprecision(1)
              << "[mm-val] wall " << ms << " ms for " << rows_done << " rows ("
              << std::setprecision(0) << rows_per_s << " rows/s, includes per-seed upload+readback)\n";

    std::cout << "VALIDATE: " << SEEDS << " seeds x " << M << "x" << K << "x" << N
              << " exact-quire matmul on " << num_cores << " cores — "
              << total_outputs << " outputs, " << total_products
              << " exact products, "
              << (all_pass ? "ALL bit-exact vs oracle" : "MISMATCH vs oracle") << "\n";

    bool pass = all_pass && (matched_outputs == total_outputs) && (gate_pass == SEEDS);
    std::cout << (pass ? "PASS" : "FAIL")
              << ": full-validation " << matched_outputs << "/" << total_outputs
              << " bit-exact (incl. " << gate_pass << " full-quire byte-gates) on real silicon\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
