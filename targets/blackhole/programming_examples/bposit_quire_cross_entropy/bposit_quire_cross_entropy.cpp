// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// bposit_quire_cross_entropy — run a REAL information-theoretic ML primitive
// on a Tenstorrent Blackhole baby (RISC-V) data-movement core, bit-exact vs the
// oracle golden, reusing the PROVEN exact-quire kernels.
//
//   CROSS-ENTROPY (the standard classification training loss):
//     H(p,q) = -Sum_i p_i * log2(q_i)
//   for a target distribution p and a model/predicted distribution q over the
//   same alphabet of N classes. Each term p_i*log2(q_i) is an EXACT bp16 product
//   accumulated in the EXACT 256-bit Kulisch quire (QFRAC=96), then negated:
//     log2q_i = bposit16_log2(q_i)            (HOST-precomputed; see split below)
//     term    = bposit16_mul(p_i, log2q_i)    (EXACT p_i*log2(q_i))       [DEVICE]
//     H_q    += quire(term)                   (EXACT 256-bit accumulation) [DEVICE]
//     H_q     = -H_q                          (positive cross-entropy)     [DEVICE]
//     H_bp32  = bp32_encode_quire256(H_q)     (the SINGLE rounding: readout)[DEVICE]
//   The crown jewel: the accumulation is in the EXACT quire, so it is
//   order-independent AND tiny-probability tails (q_i small => log2 large-neg)
//   are summed with FULL precision — a float dot drops those tail terms once the
//   partial sum dominates, exactly the regime cross-entropy cares about. This
//   exact-quire SUM is the load-bearing, rounding-stable result and runs on-device.
//
// DEVICE vs HOST split (be explicit — HONEST split):
//   - DEVICE (Blackhole baby core, kernels/cross_entropy_baby.cpp): the EXACT
//     loss accumulation — every EXACT bp16 product term = p_i*log2(q_i), the EXACT
//     256-bit quire accumulation, the negate, AND the quire->bposit32 readout. The
//     load-bearing, rounding-stable, tail-lossless quire SUM is 100% on-device.
//   - HOST (this file): precomputes the per-element log2(q_i) (bposit16_log2 ==
//     BP16_LOG2_LUT[q_i]) and marshals p[] + log2q[] to the device. The log2 is
//     host-side ONLY because the 65536-entry log2 LUT does NOT fit the BRISC baby
//     core's local DATA region (~0x11d0 bytes; big-L1 is a separate CB/buffer
//     space, not kernel .data) — a static LUT in the kernel TU overflows the .elf
//     at load. Plus the belt-and-suspenders self-check (re-derive the golden from
//     the reused EXACT headers before the device run) and the byte-/code-level
//     gate of the device output. The host log2 IS the canonical oracle, so the
//     result is bit-exact vs golden regardless of where log2 runs.
//
// Golden + oracle + reused headers (cited):
//   golden : golden/cross_entropy.json  (p_codes, q_codes, quire_le_hex, bp32_code, n)
//   oracle : mosyne-bposit/kernels/bposit16_reference.py
//            (bposit16_log2, bposit16_mul, bposit16_to_quire, quire256_to_bposit32,
//             decode_bposit32, decoded_to_fraction_32)
//   C ref  : kernel/cross_entropy_kernel.c  (proven RV32IM-clean exact-quire ref)
//   reused : kernel/bp16_mul.h (bposit16_mul), kernel/bp16_quire.h (bp16_to_q256,
//            q256_add, q256_negate), kernel/bp32_encode.h (bp32_encode_quire256);
//            kernel/bp16_log2_lut.h (BP16_LOG2_LUT) is included HOST-SIDE ONLY to
//            precompute log2(q_i) — it is NOT compiled into the device kernel.
//
// Host program + kernel dispatch mirror the proven sibling example
//   programming_examples/bposit_quire_causet/bposit_quire_causet.cpp.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

// Baked golden case data (flat arrays), generated from golden/cross_entropy.json +
// the oracle by gen_cross_entropy_golden.py and re-verified here at startup.
#include "cross_entropy_golden_cases.h"

// Host-side reference reuses the SAME golden-exact numerics the kernel uses, so
// the host independently recomputes the per-element log2(q_i) (BP16_LOG2_LUT,
// host-side only) and the expected quire + bp32 (belt-and-suspenders). These
// freestanding headers carry the full golden-exact op set + the log2 LUT; the
// host calls a subset, so suppress -Werror=unused-function (tt-metal is strict).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../../kernel/bp16_mul.h"
#include "../../kernel/bp16_quire.h"
#include "../../kernel/bp16_log2_lut.h"
#include "../../kernel/bp32_encode.h"
#pragma GCC diagnostic pop

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static std::string quire_to_hex_le(const uint32_t q[8]) {
    // 32-byte little-endian hex, matching golden "quire_le_hex" convention.
    std::ostringstream os;
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 4; ++b)
            os << std::hex << std::setw(2) << std::setfill('0') << ((q[i] >> (8 * b)) & 0xFFu);
    return os.str();
}

// Host precompute of the per-element log2(q_i) (the part the device kernel no
// longer does — the 65536-entry LUT does not fit the baby core). This IS the
// canonical oracle: BP16_LOG2_LUT[q] == bposit16_log2(q).
static int host_log2q(int q_code) { return (int)(BP16_LOG2_LUT[q_code & 0xFFFF] & 0xFFFF); }

// Host re-derivation of the DEVICE path: exact-quire accumulation of
// term_i = bposit16_mul(p_i, log2q_i), negate, readout. Takes the SAME log2q[]
// operand the device receives, so it mirrors the kernel exactly (no LUT here).
static void host_cross_entropy(const uint16_t* p, const uint16_t* log2q, int n,
                               unsigned acc[8], uint32_t& bp32_out) {
    for (int l = 0; l < 8; ++l) acc[l] = 0u;
    unsigned contrib[8];
    for (int i = 0; i < n; ++i) {
        int term = bposit16_mul(p[i] & 0xFFFF, log2q[i] & 0xFFFF);  // p_i*log2(q_i) EXACT
        bp16_to_q256(term, contrib);
        q256_add(acc, contrib);
    }
    q256_negate(acc);                                             // H = -Sum
    bp32_out = bp32_encode_quire256(acc);                         // quire -> bposit32
}

// =============== cross-entropy H(p,q) ===========================
static int run_cross_entropy(distributed::MeshDevice* mesh_device) {
    constexpr int QLIMBS = 8;
    constexpr uint32_t WORD = sizeof(uint32_t);

    const int N = QGOLD_CE_N;            // 4 classes
    const int N_OUT = QLIMBS + 1;        // 8 negated-quire limbs + 1 bp32 readout

    // p[] is the target distribution; log2q[] is the HOST-precomputed per-element
    // log2(q_i) (from the raw q codes) that the device kernel consumes in place of
    // an on-chip log2 LUT. Both are marshalled to the device as bp16 operands.
    std::vector<uint16_t> log2q(N);
    std::vector<uint32_t> p_words(N), log2q_words(N);
    for (int i = 0; i < N; ++i) {
        p_words[i] = (uint32_t)(QGOLD_CE_P[i] & 0xFFFF);
        log2q[i] = (uint16_t)host_log2q((int)QGOLD_CE_Q[i]);  // log2(q_i) on HOST
        log2q_words[i] = (uint32_t)log2q[i];
    }

    // ---- host self-check (re-derives the golden from the reused EXACT headers):
    //   (a) confirm host log2(q_i) == baked QGOLD_CE_LOG2Q (the split is faithful);
    //   (b) recompute the negated 256-bit quire (device path), confirm baked golden;
    //   (c) recompute the bposit32 readout, confirm baked golden bp32.
    bool log2_ok = true;
    for (int i = 0; i < N; ++i)
        if (log2q[i] != (QGOLD_CE_LOG2Q[i] & 0xFFFF)) log2_ok = false;
    unsigned host_q[QLIMBS];
    uint32_t host_bp32;
    host_cross_entropy(QGOLD_CE_P, log2q.data(), N, host_q, host_bp32);
    bool quire_ok = true, bp32_ok = true;
    for (int l = 0; l < QLIMBS; ++l)
        if (host_q[l] != QGOLD_CE_QUIRE[l]) quire_ok = false;
    if (host_bp32 != QGOLD_CE_BP32) bp32_ok = false;
    if (!(log2_ok && quire_ok && bp32_ok)) {
        std::cout << "FAIL: baked CROSS-ENTROPY golden disagrees with the reused headers "
                     "(host self-check) — golden data drifted, regenerate from oracle "
                     "(log2=" << log2_ok << " quire=" << quire_ok << " bp32=" << bp32_ok << ")\n";
        std::cout << "  host quire = " << quire_to_hex_le(host_q) << "\n";
        std::cout << "  gold quire = " << QGOLD_CE_QUIRE_HEX << "\n";
        std::cout << "  host bp32 = 0x" << std::hex << host_bp32
                  << " gold bp32 = 0x" << QGOLD_CE_BP32 << std::dec << "\n";
        return 1;
    }

    // ---- device run --------------------------------------------------------
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    const size_t prob_bytes = (size_t)N * WORD;
    const size_t out_bytes = (size_t)N_OUT * WORD;
    const size_t dbg_bytes = (size_t)(2 * N) * WORD;  // P[] then LOG2Q[]
    auto mk = [&](size_t total, BufferType type) {
        distributed::DeviceLocalBufferConfig lc{.page_size = total, .buffer_type = type};
        distributed::ReplicatedBufferConfig bc{.size = total};
        return distributed::MeshBuffer::create(bc, lc, mesh_device);
    };
    // ONE page per buffer (whole buffer) — same marshalling fix as causet/physics.
    auto p_dram = mk(prob_bytes, BufferType::DRAM);
    auto log2q_dram = mk(prob_bytes, BufferType::DRAM);
    auto out_dram = mk(out_bytes, BufferType::DRAM);
    auto p_l1 = mk(prob_bytes, BufferType::L1);
    auto log2q_l1 = mk(prob_bytes, BufferType::L1);
    auto out_l1 = mk(out_bytes, BufferType::L1);
    auto dbg_dram = mk(dbg_bytes, BufferType::DRAM);  // [DEBUG] echo P[] then LOG2Q[]

    auto kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bposit_quire_cross_entropy/kernels/cross_entropy_baby.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // arg order MUST match get_arg_val<>() in cross_entropy_baby.cpp.
    SetRuntimeArgs(program, kernel, core,
                   {(uint32_t)p_dram->address(), (uint32_t)log2q_dram->address(),
                    (uint32_t)out_dram->address(), (uint32_t)p_l1->address(),
                    (uint32_t)log2q_l1->address(), (uint32_t)out_l1->address(),
                    (uint32_t)N, (uint32_t)dbg_dram->address()});

    distributed::EnqueueWriteMeshBuffer(cq, p_dram, p_words, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, log2q_dram, log2q_words, /*blocking=*/false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::vector<uint32_t> result(N_OUT, 0);
    distributed::EnqueueReadMeshBuffer(cq, result, out_dram, /*blocking=*/true);

    // ---- DEBUG (removable): read back the P[]+LOG2Q[] codes the kernel saw in L1
    std::vector<uint32_t> dbg(2 * N, 0);
    distributed::EnqueueReadMeshBuffer(cq, dbg, dbg_dram, /*blocking=*/true);
    std::cout << "[cross_entropy] P codes (host)        : ";
    for (int i = 0; i < N; ++i) std::cout << "0x" << std::hex << std::setw(4) << std::setfill('0') << (p_words[i] & 0xFFFF) << " ";
    std::cout << std::dec << "\n[cross_entropy] P codes readback      : ";
    for (int i = 0; i < N; ++i) std::cout << "0x" << std::hex << std::setw(4) << std::setfill('0') << (dbg[i] & 0xFFFF) << " ";
    std::cout << std::dec << "\n[cross_entropy] log2(q) codes (host)  : ";
    for (int i = 0; i < N; ++i) std::cout << "0x" << std::hex << std::setw(4) << std::setfill('0') << (log2q_words[i] & 0xFFFF) << " ";
    std::cout << std::dec << "\n[cross_entropy] log2(q) codes readback: ";
    for (int i = 0; i < N; ++i) std::cout << "0x" << std::hex << std::setw(4) << std::setfill('0') << (dbg[N + i] & 0xFFFF) << " ";
    std::cout << std::dec << "\n";
    // ---- end DEBUG ----

    // ---- gate: 8 negated-quire limbs + the bp32 readout, byte-/code-identical -
    uint32_t dev_q[QLIMBS];
    for (int l = 0; l < QLIMBS; ++l) dev_q[l] = result[l];
    uint32_t dev_bp32 = result[QLIMBS];

    int bad_q = 0;
    for (int l = 0; l < QLIMBS; ++l)
        if (dev_q[l] != QGOLD_CE_QUIRE[l]) {
            if (++bad_q <= 8)
                std::cout << "  quire limb[" << l << "] device=0x" << std::hex << dev_q[l]
                          << " golden=0x" << QGOLD_CE_QUIRE[l] << std::dec << "\n";
        }
    int bad_bp32 = (dev_bp32 != QGOLD_CE_BP32) ? 1 : 0;
    if (bad_bp32)
        std::cout << "  bp32 device=0x" << std::hex << std::setw(8) << std::setfill('0') << dev_bp32
                  << " golden=0x" << std::setw(8) << QGOLD_CE_BP32 << std::dec << "\n";

    const int total_checks = QLIMBS + 1;                 // 8 quire limbs + bp32
    const int matched = (QLIMBS - bad_q) + (1 - bad_bp32);
    bool pass = (bad_q == 0) && (bad_bp32 == 0);

    std::cout << "[cross_entropy] device negated quire = " << quire_to_hex_le(dev_q) << "\n";
    std::cout << "[cross_entropy] golden  negated quire = " << QGOLD_CE_QUIRE_HEX << "\n";
    std::cout << "[cross_entropy] device H(p,q) bp32 = 0x" << std::hex << std::setw(8) << std::setfill('0')
              << dev_bp32 << "  golden = 0x" << std::setw(8) << QGOLD_CE_BP32 << std::dec << "\n";

    if (pass) {
        std::cout << "PASS: cross-entropy on a Blackhole baby core, bit-exact vs golden (real silicon)\n";
        std::cout << "      H(p,q) = -Sum_i p_i*log2(q_i) over N=" << N << " classes (the standard "
                     "classification training loss); H(p,q) = 2.625 bits (bp32 0x" << std::hex
                  << std::setw(8) << std::setfill('0') << QGOLD_CE_BP32 << std::dec << "), "
                  << matched << "/" << total_checks << " match golden (case cross_entropy)\n";
        std::cout << "      meaning: each term p_i*log2(q_i) is an EXACT bp16 product accumulated in the "
                     "EXACT 256-bit quire on-device, so the tiny-probability tails are summed losslessly "
                     "and the loss is rounding-stable and order-independent — the classification loss on "
                     "real silicon. SPLIT: the per-element log2(q_i) is HOST-precomputed (bposit16_log2; "
                     "the 65536-entry LUT does not fit the baby core), while the load-bearing exact-quire "
                     "SUM runs 100% on-device. KL(p||q) = H(p,q) - H(p); Shannon H(p) = H(p,p).\n";
    } else {
        std::cout << "FAIL: cross-entropy on a Blackhole baby core, "
                  << matched << "/" << total_checks << " match golden (case cross_entropy)\n";
    }
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? std::string(argv[1]) : std::string("cross_entropy");
    if (mode != "cross_entropy" && mode != "all") {
        std::cout << "usage: " << argv[0] << " [cross_entropy|all]\n";
        return 2;
    }

    // 1x1 unit mesh on device 0 (same idiom as bposit_quire_causet / physics).
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    int rc = run_cross_entropy(mesh_device.get());
    mesh_device->close();
    return rc;
}
