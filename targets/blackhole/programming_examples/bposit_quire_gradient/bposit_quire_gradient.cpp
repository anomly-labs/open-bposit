// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// bposit_quire_gradient — the -for-TRAINING result on real silicon: on a
// Tenstorrent Blackhole baby (RISC-V) core, accumulating a per-parameter GRADIENT
// g = Sum_t d_t (a sum of many tiny per-step contributions) in an EXACT 256-bit
// b-posit16 quire recovers the tiny-tail gradient mass that an IEEE fp32 running
// accumulator SWALLOWS. Both accumulations run on the SAME core over the SAME
// inputs; the host gates both against the exact mathematical ground truth (oracle
// exact Fraction arithmetic, baked into quire_gradient_cases.h by
// gen_gradient_golden.py).
//
// WHY (the real training problem): in large-batch / many-step training, a
// parameter's gradient is a SUM of many tiny per-example / per-step contributions.
// A large early contribution inflates the running accumulator to magnitude M; once
// the running sum is at M, each subsequent tiny d_t falls below ULP(M) and is LOST
// in low precision (the "swamping" / stale-gradient / gradient-underflow problem).
// The exact quire keeps every contribution exactly. Sweeping the step count T shows
// the fp32-lost gradient fraction GROW with the number of accumulation steps.
//
// For each config the kernel (kernels/quire_gradient_baby.cpp):
//   (1) accumulates Sum_t bp16_to_q256(code_t) into one exact 256-bit quire and
//       reads it back to bp16 (the proven reduce path, reusing the scalar-reference
//       bp16_quire.h / bp16_encode.h headers already bit-exact on this silicon);
//   (2) accumulates the SAME values as a genuine IEEE-754 binary32 running sum
//       (kernels/ieee_softfp32.h, reused from bposit_quire_vs_float — pure-integer
//       fp32 add, RTNE, host-verified bit-exact vs native C `float`).
// The host compares BOTH to the exact gradient and prints, per config:
//   T=<> running_sum~<M>: quire_g=<> fp32_g=<> exact_g=<>  fp32_lost=<frac>
// then an overall PASS line.
//
// Host program + dispatch mirror the proven bposit_quire_vs_float.cpp (single-page
// MeshBuffer marshalling, golden self-check before the device run, PASS/FAIL gate).

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

// Baked golden case data (generated from the oracle).
#include "quire_gradient_cases.h"

// Host independently re-derives the expected quire / readout / fp32 result from the
// SAME numerics the kernel uses, so a drifted golden is caught before any device run.
// These freestanding headers carry a wide op set the host only partly calls —
// suppress -Werror=unused-function (tt-metal is strict).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../../kernel/bp16_quire.h"
#include "../../kernel/bp16_encode.h"
#include "../../kernel/bp16_decode.h"
#pragma GCC diagnostic pop
#include "$TT_METAL_HOME/tt_metal/programming_examples/bposit_quire_vs_float/kernels/ieee_softfp32.h"  // fp32_add (same pure-integer add the kernel runs)

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

static float bits_to_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Same bp16-code -> fp32-bits conversion as the kernel (kept in lockstep). Exact for
// every representable bp16 value, so the float error is purely accumulation.
static uint32_t host_bp16_code_to_fp32_bits(int code) {
    int sign;
    unsigned M;
    int E2;
    bp16_decode(code, &sign, &M, &E2);
    if (M == 0u) return sign ? 0x80000000u : 0x00000000u;
    int msb = 31;
    while (msb >= 0 && !((M >> msb) & 1u)) msb--;
    uint32_t sig = (msb <= 23) ? (M << (23 - msb)) : (M >> (msb - 23));
    int biased = E2 + msb + 127;
    if (biased >= 0xFF) return ((uint32_t)sign << 31) | 0x7F800000u;
    if (biased <= 0) return (uint32_t)sign << 31;
    return ((uint32_t)sign << 31) | ((uint32_t)biased << 23) | (sig & 0x007FFFFFu);
}

// One config: self-check golden, run on device, gate both accumulators, report the
// fraction of the gradient fp32 lost.
static int run_config(distributed::MeshDevice* mesh_device, const qgrad_case_t* cs) {
    constexpr int QLIMBS = 8;
    constexpr int N_OUT = QLIMBS + 2;  // 8 quire limbs + bp16 readout + fp32 bits
    constexpr uint32_t WORD = sizeof(uint32_t);

    const int N = cs->n;
    std::vector<uint32_t> in_words(N);
    for (int i = 0; i < N; ++i) in_words[i] = (uint32_t)(cs->codes[i] & 0xFFFF);

    // ---- host self-check: re-derive quire + readout + fp32 from the headers ---
    unsigned ref_q[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t ref_facc = 0x00000000u;
    for (int i = 0; i < N; ++i) {
        int code = (int)(in_words[i] & 0xFFFF);
        unsigned c[QLIMBS];
        bp16_to_q256(code, c);
        q256_add(ref_q, c);
        ref_facc = fp32_add(ref_facc, host_bp16_code_to_fp32_bits(code));
    }
    int ref_readout = bp16_encode_quire256(ref_q) & 0xFFFF;

    bool golden_self_ok = (ref_readout == (cs->readout_bp16 & 0xFFFF)) &&
                          (ref_facc == cs->fp32_bits);
    for (int i = 0; i < QLIMBS; ++i)
        if (ref_q[i] != cs->quire[i]) golden_self_ok = false;
    if (!golden_self_ok) {
        std::cout << "FAIL: baked golden for config " << cs->name
                  << " disagrees with the reused headers (host self-check) — "
                     "regenerate from the oracle (gen_gradient_golden.py)\n";
        std::cout << "  host quire = " << quire_to_hex_le((const uint32_t*)ref_q)
                  << "  baked = " << cs->quire_hex << "\n";
        std::cout << "  host fp32  = 0x" << std::hex << ref_facc
                  << "  baked = 0x" << cs->fp32_bits << std::dec << "\n";
        return 1;
    }

    // ---- device run --------------------------------------------------------
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    const size_t in_bytes = (size_t)N * WORD;
    const size_t out_bytes = (size_t)N_OUT * WORD;
    auto mk = [&](size_t total, BufferType type) {
        distributed::DeviceLocalBufferConfig lc{.page_size = total, .buffer_type = type};
        distributed::ReplicatedBufferConfig bc{.size = total};
        return distributed::MeshBuffer::create(bc, lc, mesh_device);
    };
    auto in_dram = mk(in_bytes, BufferType::DRAM);
    auto out_dram = mk(out_bytes, BufferType::DRAM);
    auto in_l1 = mk(in_bytes, BufferType::L1);
    auto out_l1 = mk(out_bytes, BufferType::L1);
    auto dbg_dram = mk(in_bytes, BufferType::DRAM);  // [DEBUG]

    auto kernel = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "bposit_quire_gradient/kernels/quire_gradient_baby.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(program, kernel, core,
                   {(uint32_t)in_dram->address(), (uint32_t)out_dram->address(),
                    (uint32_t)in_l1->address(), (uint32_t)out_l1->address(),
                    (uint32_t)N, (uint32_t)dbg_dram->address()});

    distributed::EnqueueWriteMeshBuffer(cq, in_dram, in_words, /*blocking=*/false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    std::vector<uint32_t> result(N_OUT, 0);
    distributed::EnqueueReadMeshBuffer(cq, result, out_dram, /*blocking=*/true);

    std::vector<uint32_t> dbg(N, 0);
    distributed::EnqueueReadMeshBuffer(cq, dbg, dbg_dram, /*blocking=*/true);

    // ---- gate: quire (8 limbs) + readout, and fp32 result -------------------
    uint32_t dev_q[QLIMBS];
    for (int i = 0; i < QLIMBS; ++i) dev_q[i] = result[i];
    int dev_readout = (int)(result[QLIMBS] & 0xFFFF);
    uint32_t dev_fp32 = result[QLIMBS + 1];

    int bad_q = 0;
    for (int i = 0; i < QLIMBS; ++i)
        if (dev_q[i] != cs->quire[i]) {
            if (++bad_q <= 8)
                std::cout << "  limb[" << i << "] device=0x" << std::hex << dev_q[i]
                          << " golden=0x" << cs->quire[i] << std::dec << "\n";
        }
    bool quire_limbs_ok = (bad_q == 0);
    bool readout_ok = (dev_readout == (cs->readout_bp16 & 0xFFFF));
    bool fp32_ok = (dev_fp32 == cs->fp32_bits);  // device fp32 == genuine fp32 baseline

    // ---- interpret vs the EXACT gradient ------------------------------------
    // exact gradient = num/den. The quire readout decodes to exactly that value (gen
    // enforced readout==exact), so quire's gradient error == 0 whenever readout_ok.
    double exact_g = (double)cs->exact_num / (double)cs->exact_den;
    bool quire_exact = quire_limbs_ok && readout_ok;  // quire == exact gradient

    float dev_f = bits_to_f32(dev_fp32);
    // fraction of the gradient fp32 dropped: (exact - fp32)/exact. The swamped tiny
    // tail mass is exactly what fp32 lost; the quire recovered all of it.
    double fp32_lost_frac = (exact_g != 0.0)
        ? ((double)exact_g - (double)dev_f) / (double)exact_g
        : 0.0;
    bool fp32_drops = (dev_fp32 != cs->exact_fp32_bits);  // fp32 really lost tail mass

    // running-sum magnitude this config inflated the accumulator to (M ~ the big
    // early contribution); printed for the headline line. The first code is +B = 2^k.
    double running_M = (double)bits_to_f32(host_bp16_code_to_fp32_bits((int)(cs->codes[0] & 0xFFFF)));

    // ---- the headline per-config line ---------------------------------------
    double quire_g_val = bits_to_f32(host_bp16_code_to_fp32_bits(dev_readout));
    std::cout << "T=" << cs->t_steps
              << " running_sum~" << std::defaultfloat << running_M << ": "
              << "quire_g=" << std::setprecision(8) << quire_g_val
              << " fp32_g=" << dev_f
              << " exact_g=" << exact_g
              << "  fp32_lost=" << std::fixed << std::setprecision(3) << fp32_lost_frac
              << " (" << std::setprecision(1) << (fp32_lost_frac * 100.0) << "%)";
    std::cout << std::defaultfloat;
    std::cout << "  [quire " << (quire_exact ? "EXACT vs oracle" : "MISMATCH")
              << ", device fp32==baseline " << (fp32_ok ? "OK" : "MISMATCH") << "]\n";

    // a config PASSES iff: quire is bit-exact to the exact gradient AND the device
    // fp32 matches the genuine fp32 baseline AND that baseline really lost tail mass
    // (otherwise the config is not demonstrating the swamping problem).
    bool pass = quire_exact && fp32_ok && fp32_drops && (fp32_lost_frac > 0.0);
    if (!pass) {
        std::cout << "   -> config FAIL (quire_exact=" << quire_exact
                  << " fp32_matches_baseline=" << fp32_ok
                  << " fp32_drops_tail=" << fp32_drops
                  << " lost_frac=" << fp32_lost_frac << ")\n";
    }
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    // optional: run a single config by index; default runs all (the T sweep).
    int only = (argc > 1) ? std::atoi(argv[1]) : -1;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    std::cout << "=== exact 256-bit b-posit16 quire vs IEEE fp32 — GRADIENT "
                 "ACCUMULATION (the swamped-tiny-gradients training problem) on a "
                 "Blackhole baby core ===\n";
    std::cout << "    g = Sum_t d_t : a large early contribution inflates the running "
                 "sum to M, so each tiny per-step gradient < ULP(M) is dropped by fp32; "
                 "the exact quire keeps every one.\n";

    int passed = 0, total = 0;
    double max_lost = 0.0;
    for (int c = 0; c < QGRAD_NCASES; ++c) {
        if (only >= 0 && c != only) continue;
        ++total;
        // track the largest fraction fp32 loses across the sweep, for the PASS line.
        const qgrad_case_t* cs = &QGRAD_CASES[c];
        double exact_g = (double)cs->exact_num / (double)cs->exact_den;
        float fp32_baseline = bits_to_f32(cs->fp32_bits);
        double lost = (exact_g != 0.0) ? (exact_g - (double)fp32_baseline) / exact_g : 0.0;
        if (lost > max_lost) max_lost = lost;
        int rc = run_config(mesh_device.get(), cs);
        if (rc == 0) ++passed;
    }

    bool all = (passed == total) && (total > 0);
    std::cout << (all ? "PASS" : "FAIL")
              << ": exact-quire gradient accumulation recovers up to "
              << std::fixed << std::setprecision(1) << (max_lost * 100.0)
              << "% of gradient that fp32 drops; quire bit-exact vs oracle — real "
                 "silicon (" << passed << "/" << total << " configs)\n";
    std::cout << std::defaultfloat;

    mesh_device->close();
    return all ? 0 : 1;
}
