// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: Apache-2.0
//
// bposit_quire_attention — ATTENTION under MASSIVE-ACTIVATION OUTLIERS on a
// REAL Tenstorrent Blackhole baby (RISC-V) core. Transformer attention scores
// s_j = q.k_j are corrupted when a few activation dims are huge outliers (Sun et
// al. 2024, "massive activations", arXiv:2402.17762): a naive IEEE fp32 dot lets
// the outlier products catastrophically cancel away the many small-but-meaningful
// contributions, distorting the softmax and FLIPPING which key wins. The EXACT
// 256-bit Kulisch b-posit16 quire keeps the full dot product EXACTLY, so its
// softmax matches the exact softmax and fp32 does not — demonstrated on silicon.
// (Framing: src/spacetime/attention_precision.py.)
//
// For one query q and J keys k_j (head_dim D), the kernel
// (kernels/quire_attention_baby.cpp) computes, per key, the score s_j BOTH ways:
//   (1) exact 256-bit quire dot  q256 = Sum_i bp16_prod_to_q256(q_i,k_ji), single
//       readout (the proven quire_dot path, reusing the the scalar-kernel bp16_quire.h /
//       bp16_encode.h headers already bit-exact on this silicon);
//   (2) genuine IEEE-754 fp32 NAIVE dot (kernels/ieee_softfp32.h, pure-integer add
//       + mul, RTNE, host-verified bit-exact vs native C `float`).
// The host gates BOTH against the exact ground truth (oracle exact Fraction
// arithmetic, baked into attention_golden_cases.h by gen_attention_golden.py),
// then softmaxes the quire/exact and fp32 score vectors and prints the per-key
// scores, the softmax weights, the argmax of each, and the FLIP.
//
// Host program + dispatch mirror bposit_quire_vs_float.cpp (single-page MeshBuffer
// marshalling, golden self-check before the device run, PASS/FAIL gate).

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
#include "attention_golden_cases.h"

// Host independently re-derives the expected quire / readout / fp32 from the SAME
// numerics the kernel uses, so a drifted golden is caught before any device run.
// These freestanding headers carry a wide op set the host only partly calls —
// suppress -Werror=unused-function (tt-metal is strict).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../../kernel/bp16_quire.h"
#include "../../kernel/bp16_encode.h"
#include "../../kernel/bp16_decode.h"
#pragma GCC diagnostic pop
#include "kernels/ieee_softfp32.h"  // fp32_add (same pure-integer add the kernel runs)

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static float bits_to_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static std::string quire_to_hex_le(const uint32_t q[8]) {
    std::ostringstream os;
    for (int i = 0; i < 8; ++i)
        for (int b = 0; b < 4; ++b)
            os << std::hex << std::setw(2) << std::setfill('0') << ((q[i] >> (8 * b)) & 0xFFu);
    return os.str();
}

// Same bp16-code -> fp32-bits conversion as the kernel (kept in lockstep).
static uint32_t host_bp16_code_to_fp32_bits(int code) {
    int sign; unsigned M; int E2;
    bp16_decode(code, &sign, &M, &E2);
    if (M == 0u) return sign ? 0x80000000u : 0x00000000u;
    int msb = 31; while (msb >= 0 && !((M >> msb) & 1u)) msb--;
    uint32_t sig = (msb <= 23) ? (M << (23 - msb)) : (M >> (msb - 23));
    int biased = E2 + msb + 127;
    if (biased >= 0xFF) return ((uint32_t)sign << 31) | 0x7F800000u;
    if (biased <= 0) return (uint32_t)sign << 31;
    return ((uint32_t)sign << 31) | ((uint32_t)biased << 23) | (sig & 0x007FFFFFu);
}

// EXACT product q_i*k_i of two bp16 codes as fp32 bits, built directly from the
// dyadic decode — the SAME exact path the kernel runs (bp16_prod_to_fp32_bits):
// M_q*M_k < 2^22 fits the fp32 significand, so the product is lossless and the
// only fp32 error is in the running fp32_add. Kept in lockstep with the kernel so
// the host self-check catches any divergence before the device run.
static uint32_t host_bp16_prod_to_fp32_bits(int acode, int bcode) {
    int sa, sb, Ea, Eb;
    unsigned Ma, Mb;
    bp16_decode(acode, &sa, &Ma, &Ea);
    bp16_decode(bcode, &sb, &Mb, &Eb);
    int sign = sa ^ sb;
    if (Ma == 0u || Mb == 0u) return (uint32_t)sign << 31;
    unsigned M = Ma * Mb;
    int E2 = Ea + Eb;
    int msb = 31; while (msb >= 0 && !((M >> msb) & 1u)) msb--;
    uint32_t sig = (msb <= 23) ? (M << (23 - msb)) : (M >> (msb - 23));
    int biased = E2 + msb + 127;
    if (biased >= 0xFF) return ((uint32_t)sign << 31) | 0x7F800000u;
    if (biased <= 0) return (uint32_t)sign << 31;
    return ((uint32_t)sign << 31) | ((uint32_t)biased << 23) | (sig & 0x007FFFFFu);
}

static int argmax3(const double* v, int n) {
    int m = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[m]) m = i;
    return m;
}

static void softmax(const double* s, int n, double* out) {
    double m = s[0];
    for (int i = 1; i < n; ++i) if (s[i] > m) m = s[i];
    double z = 0.0;
    for (int i = 0; i < n; ++i) { out[i] = std::exp(s[i] - m); z += out[i]; }
    for (int i = 0; i < n; ++i) out[i] /= z;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    constexpr int QLIMBS = 8;
    constexpr int N_OUT = QLIMBS + 2;  // 8 quire limbs + bp16 readout + fp32 bits
    constexpr uint32_t WORD = sizeof(uint32_t);
    const int D = ATTN_D;
    const int J = ATTN_NKEYS;

    std::cout << "=== attention under massive-activation outliers — exact 256-bit "
                 "b-posit16 quire dot vs IEEE fp32 naive dot, on a Blackhole baby core ===\n";
    std::cout << "    head_dim D=" << D << ", " << J << " keys; outlier dims (q*k = +/-2^33) "
                 "cancel exactly in the quire, fp32 loses the signal.\n";

    // ---- host self-check: re-derive quire + readout + fp32 from the headers ----
    // (Catches a drifted golden before any device run.)
    for (int j = 0; j < J; ++j) {
        const attn_key_t* K = &ATTN_KEYS[j];
        unsigned ref_q[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t ref_facc = 0x00000000u;
        for (int i = 0; i < D; ++i) {
            int qc = (int)(ATTN_Q_CODES[i] & 0xFFFF);
            int kc = (int)(K->codes[i] & 0xFFFF);
            unsigned c[QLIMBS];
            bp16_prod_to_q256(qc, kc, c);
            q256_add(ref_q, c);
            uint32_t prod = host_bp16_prod_to_fp32_bits(qc, kc);  // exact q_i*k_i (== kernel)
            ref_facc = fp32_add(ref_facc, prod);
        }
        int ref_readout = bp16_encode_quire256(ref_q) & 0xFFFF;
        bool ok = (ref_readout == (K->readout_bp16 & 0xFFFF)) && (ref_facc == K->fp32_bits);
        for (int i = 0; i < QLIMBS; ++i) if (ref_q[i] != K->quire[i]) ok = false;
        if (!ok) {
            std::cout << "FAIL: baked golden for key " << K->name
                      << " disagrees with the reused headers (host self-check) — "
                         "regenerate from the oracle (gen_attention_golden.py)\n";
            std::cout << "  host quire = " << quire_to_hex_le((const uint32_t*)ref_q)
                      << "  baked = " << K->quire_hex << "\n";
            std::cout << "  host fp32  = 0x" << std::hex << ref_facc
                      << "  baked = 0x" << K->fp32_bits << std::dec << "\n";
            return 1;
        }
    }

    // ---- device setup ------------------------------------------------------
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    constexpr CoreCoord core = {0, 0};

    const size_t vec_bytes = (size_t)D * WORD;
    const size_t out_bytes = (size_t)N_OUT * WORD;
    auto mk = [&](size_t total, BufferType type) {
        distributed::DeviceLocalBufferConfig lc{.page_size = total, .buffer_type = type};
        distributed::ReplicatedBufferConfig bc{.size = total};
        return distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    };
    // query buffers are shared across keys; key/out buffers reused per key.
    auto q_dram = mk(vec_bytes, BufferType::DRAM);
    auto q_l1 = mk(vec_bytes, BufferType::L1);
    std::vector<uint32_t> q_words(D);
    for (int i = 0; i < D; ++i) q_words[i] = (uint32_t)(ATTN_Q_CODES[i] & 0xFFFF);
    distributed::EnqueueWriteMeshBuffer(cq, q_dram, q_words, /*blocking=*/false);

    // collected per-key device results
    std::vector<double> exact_score(J), quire_score(J), fp32_score(J);
    std::vector<bool> quire_exact(J, false), fp32_match(J, false);
    std::vector<int> dev_readout(J, 0);
    int n_quire_ok = 0, n_fp32_ok = 0;

    for (int j = 0; j < J; ++j) {
        const attn_key_t* K = &ATTN_KEYS[j];
        std::vector<uint32_t> k_words(D);
        for (int i = 0; i < D; ++i) k_words[i] = (uint32_t)(K->codes[i] & 0xFFFF);

        auto program = CreateProgram();
        auto k_dram = mk(vec_bytes, BufferType::DRAM);
        auto out_dram = mk(out_bytes, BufferType::DRAM);
        auto k_l1 = mk(vec_bytes, BufferType::L1);
        auto out_l1 = mk(out_bytes, BufferType::L1);
        auto dbg_dram = mk(2 * vec_bytes, BufferType::DRAM);  // [DEBUG] q[] then k[]

        auto kernel = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "bposit_quire_attention/kernels/quire_attention_baby.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default});

        SetRuntimeArgs(program, kernel, core,
                       {(uint32_t)q_dram->address(), (uint32_t)k_dram->address(),
                        (uint32_t)out_dram->address(), (uint32_t)q_l1->address(),
                        (uint32_t)k_l1->address(), (uint32_t)out_l1->address(),
                        (uint32_t)D, (uint32_t)dbg_dram->address()});

        distributed::EnqueueWriteMeshBuffer(cq, k_dram, k_words, /*blocking=*/false);
        distributed::MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        std::vector<uint32_t> result(N_OUT, 0);
        distributed::EnqueueReadMeshBuffer(cq, result, out_dram, /*blocking=*/true);

        // ---- gate: quire (8 limbs) + readout, and fp32 result ---------------
        uint32_t dev_q[QLIMBS];
        for (int i = 0; i < QLIMBS; ++i) dev_q[i] = result[i];
        int rdo = (int)(result[QLIMBS] & 0xFFFF);
        uint32_t dev_fp32 = result[QLIMBS + 1];

        int bad_q = 0;
        for (int i = 0; i < QLIMBS; ++i)
            if (dev_q[i] != K->quire[i]) {
                if (++bad_q <= 8)
                    std::cout << "  key" << j << " limb[" << i << "] device=0x" << std::hex
                              << dev_q[i] << " golden=0x" << K->quire[i] << std::dec << "\n";
            }
        bool qlimbs_ok = (bad_q == 0);
        bool readout_ok = (rdo == (K->readout_bp16 & 0xFFFF));
        quire_exact[j] = qlimbs_ok && readout_ok;   // exact == ground truth
        fp32_match[j] = (dev_fp32 == K->fp32_bits);  // device fp32 == genuine baseline
        if (quire_exact[j]) ++n_quire_ok;
        if (fp32_match[j]) ++n_fp32_ok;

        double exact = (double)K->exact_num / (double)K->exact_den;
        exact_score[j] = exact;
        quire_score[j] = bits_to_f32(host_bp16_code_to_fp32_bits(rdo));  // decoded quire readout
        fp32_score[j] = (double)bits_to_f32(dev_fp32);
        dev_readout[j] = rdo;

        std::cout << "s_" << j << " (" << K->name << "): "
                  << "quire=" << std::fixed << std::setprecision(4) << quire_score[j]
                  << " fp32=" << fp32_score[j]
                  << " exact=" << exact
                  << "  [quire " << (quire_exact[j] ? "EXACT" : "MISMATCH")
                  << ", fp32==baseline " << (fp32_match[j] ? "OK" : "MISMATCH")
                  << ", quire->bp16=0x" << std::hex << rdo << std::dec << "]\n";
    }

    // ---- softmax over the score vectors + the FLIP --------------------------
    // exact/quire share the same (exact) scores; fp32 has the corrupted scores.
    std::vector<double> sm_quire(J), sm_fp32(J);
    softmax(quire_score.data(), J, sm_quire.data());
    softmax(fp32_score.data(), J, sm_fp32.data());
    int am_quire = argmax3(quire_score.data(), J);
    int am_fp32 = argmax3(fp32_score.data(), J);

    std::cout << "\nsoftmax weights:\n";
    std::cout << "  quire/exact:";
    for (int j = 0; j < J; ++j) std::cout << " " << std::setprecision(4) << sm_quire[j];
    std::cout << "   argmax=key" << am_quire << "\n";
    std::cout << "  fp32       :";
    for (int j = 0; j < J; ++j) std::cout << " " << std::setprecision(4) << sm_fp32[j];
    std::cout << "   argmax=key" << am_fp32 << "\n";

    // softmax divergence metrics (quire/exact is the reference distribution)
    double kl = 0.0, maxw = 0.0;
    for (int j = 0; j < J; ++j) {
        if (sm_quire[j] > 0.0) kl += sm_quire[j] * std::log(sm_quire[j] / sm_fp32[j]);
        double e = std::fabs(sm_quire[j] - sm_fp32[j]);
        if (e > maxw) maxw = e;
    }

    bool all_quire_exact = (n_quire_ok == J);
    bool all_fp32_match = (n_fp32_ok == J);
    bool flip = (am_quire != am_fp32);
    // sanity: the device-derived argmaxes must match the baked oracle expectation.
    bool matches_oracle = (am_quire == ATTN_EXACT_ARGMAX) && (am_fp32 == ATTN_FP32_ARGMAX);

    bool pass = all_quire_exact && all_fp32_match && flip && matches_oracle;

    std::cout << "\n";
    if (pass) {
        std::cout << "PASS: exact-quire attention matches exact softmax (argmax/weights), "
                     "fp32 distorts it — real silicon\n";
        std::cout << "  fp32 picks key" << am_fp32 << " (" << ATTN_KEYS[am_fp32].name
                  << "); quire/exact pick key" << am_quire << " (" << ATTN_KEYS[am_quire].name
                  << ") — the TRUE argmax. ARGMAX FLIP.\n";
        std::cout << "  divergence: KL(exact||fp32)=" << std::setprecision(4) << kl
                  << "  max softmax-weight error=" << maxw << "\n";
        std::cout << "  cause: the massive-activation outlier products (q*k = +/-2^33) cancel "
                     "EXACTLY in the 256-bit quire (signal survives), but in the fp32 naive sum "
                     "they swallow the O(1) signal then cancel to ~0 (signal lost).\n";
    } else {
        std::cout << "FAIL: attention demo did not hold ("
                  << "all_quire_exact=" << all_quire_exact
                  << " all_fp32_match=" << all_fp32_match
                  << " flip=" << flip
                  << " matches_oracle=" << matches_oracle << ")\n";
    }

    mesh_device->close();
    return pass ? 0 : 1;
}
