// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_cancel_sfpu — the EXACTNESS-VALUE demonstration on real Blackhole: a catastrophic-
// cancellation dot product where bf16 AND fp32 accumulation collapse to a WRONG answer, but
// the SFPU exact-quire path (bit-exact to the 256-bit Kulisch quire = the true sum) is right.
//
// Construction (per lane): products  [ +BIG, +1 (x K-2), -BIG ]  with BIG = 2^40.
//   true exact sum = K-2 (the small terms survive in the quire).
//   bf16/fp32: BIG + 1 rounds back to BIG (1 << ulp(2^40)), so all small terms are ABSORBED,
//   then -BIG -> 0. Inexact accumulation loses 100% of the answer.
// The exact-quire result is read off the device (q[8]) and converted to a value; it matches the
// true sum bit-for-bit (the SFPU exact-quire chain was already proven bit-exact vs scalar).
//
// Reuses the in-kernel-K-loop dot kernels (../bposit_dotk_sfpu/) verbatim — one dispatch.

#include <fmt/ostream.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../../kernel/bp16_quire.h"
#pragma GCC diagnostic pop

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static constexpr uint32_t LANES = 32;
static constexpr uint32_t TILE_W = 32, TILE_H = 32, TILE_HW = TILE_W * TILE_H;
static constexpr uint32_t N_LIMB = 8;
static constexpr uint32_t K = 64;

static double bp16_value(int code) {
    int s, e2;
    unsigned M;
    bp16_decode(code, &s, &M, &e2);
    if (M == 0) return 0.0;
    double v = std::ldexp((double)M, e2);
    return s ? -v : v;
}
// round a double to bfloat16 precision (1 sign, 8 exp, 7 mantissa) and back to double.
static double to_bf16(double x) {
    float f = (float)x;
    uint32_t b;
    std::memcpy(&b, &f, 4);
    uint32_t r = (b + 0x8000u + ((b >> 16) & 1u)) & 0xFFFF0000u;  // round-to-nearest-even to bf16
    std::memcpy(&f, &r, 4);
    return (double)f;
}
// signed 256-bit (8x u32 LE two's-complement) quire / 2^96 -> double.
static double q256_to_double(const uint32_t q[8]) {
    bool neg = (q[7] & 0x80000000u) != 0;
    uint32_t m[8];
    for (int i = 0; i < 8; ++i) m[i] = q[i];
    if (neg) {
        unsigned long long c = 1;
        for (int i = 0; i < 8; ++i) { unsigned long long t = (unsigned long long)(~m[i]) + c; m[i] = (uint32_t)t; c = t >> 32; }
    }
    double v = 0;
    for (int i = 0; i < 8; ++i) v += std::ldexp((double)m[i], 32 * i);
    v = std::ldexp(v, -96);
    return neg ? -v : v;
}

static std::vector<uint32_t> lane_tile(const uint32_t lane_vals[LANES]) {
    std::vector<uint32_t> tile(TILE_HW);
    for (uint32_t r = 0; r < TILE_H; ++r)
        for (uint32_t c = 0; c < TILE_W; ++c) tile[r * TILE_W + c] = lane_vals[c];
    return tile;
}

int main() {
    // find a bp16 code whose value is exactly 2^20 (so BIG = product = 2^40), and -2^20.
    int big = -1, negbig = -1;
    for (int code = 0; code < 65536; ++code) {
        double v = bp16_value(code);
        if (v == 1048576.0 && big < 0) big = code;          // +2^20
        if (v == -1048576.0 && negbig < 0) negbig = code;   // -2^20
    }
    const int one = 0x4000;  // +1.0
    if (big < 0 || negbig < 0) { std::cout << "FAIL: could not find 2^20 bp16 codes\n"; return 2; }

    // Sequence per lane: k=0 (+BIG=big*big), k=1..K-2 (+1=one*one), k=K-1 (-BIG=big*negbig).
    uint32_t A[K][LANES], B[K][LANES];
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            if (k == 0) { A[k][lane] = (uint32_t)big; B[k][lane] = (uint32_t)big; }
            else if (k == K - 1) { A[k][lane] = (uint32_t)big; B[k][lane] = (uint32_t)negbig; }
            else { A[k][lane] = (uint32_t)one; B[k][lane] = (uint32_t)one; }
        }

    // references on the SAME products (lane 0): exact (scalar quire->double), fp32, bf16.
    unsigned gq[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double fp32 = 0.0f, bf16 = 0.0;
    float fp32f = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        bp16_madd_q256(gq, (int)A[k][0], (int)B[k][0]);
        double prod = bp16_value((int)A[k][0]) * bp16_value((int)B[k][0]);
        fp32f = fp32f + (float)prod;                 // fp32 accumulation
        bf16 = to_bf16(bf16 + to_bf16(prod));        // bf16 product + bf16 accumulate
    }
    fp32 = (double)fp32f;
    const double exact_true = q256_to_double(gq);    // the bit-exact 256-bit quire = true sum

    // ---- device: exact-quire dot (in-kernel K-loop), read q[8], convert ----
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t tile_bytes = sizeof(uint32_t) * TILE_HW;
    auto mk = [&](uint32_t n) {
        distributed::DeviceLocalBufferConfig lc{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig bc{.size = (size_t)n * tile_bytes};
        return distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    };
    auto q_dram = mk(N_LIMB), a_dram = mk(K), b_dram = mk(K), zero_dram = mk(1);
    std::vector<uint32_t> a_host(K * TILE_HW), b_host(K * TILE_HW);
    for (uint32_t k = 0; k < K; ++k) {
        auto at = lane_tile(A[k]); auto bt = lane_tile(B[k]);
        std::copy(at.begin(), at.end(), a_host.begin() + (size_t)k * TILE_HW);
        std::copy(bt.begin(), bt.end(), b_host.begin() + (size_t)k * TILE_HW);
    }
    std::vector<uint32_t> zero_host(TILE_HW, 0u);
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, zero_dram, zero_host, true);

    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();
    auto make_cb = [&](CBIndex idx, uint32_t n) {
        CircularBufferConfig cfg(n * tile_bytes, {{idx, tt::DataFormat::Int32}});
        cfg.set_page_size(idx, tile_bytes);
        CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, K); make_cb(CBIndex::c_1, K); make_cb(CBIndex::c_2, 1);
    make_cb(CBIndex::c_3, 1); make_cb(CBIndex::c_5, N_LIMB); make_cb(CBIndex::c_6, N_LIMB);
    make_cb(CBIndex::c_7, N_LIMB); make_cb(CBIndex::c_16, N_LIMB);

    std::vector<uint32_t> rargs;
    TensorAccessorArgs(*a_dram->get_backing_buffer()).append_to(rargs);
    TensorAccessorArgs(*b_dram->get_backing_buffer()).append_to(rargs);
    TensorAccessorArgs(*zero_dram->get_backing_buffer()).append_to(rargs);
    auto reader = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_dotk_sfpu/kernels/dataflow/read_dotk.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = rargs});
    std::vector<uint32_t> wargs;
    TensorAccessorArgs(*q_dram->get_backing_buffer()).append_to(wargs);
    auto writer = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_dotk_sfpu/kernels/dataflow/write_qk.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = wargs});
    auto compute = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_dotk_sfpu/kernels/compute/sfpu_bp16_dotk.cpp", core,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true});
    SetRuntimeArgs(program, reader, core, {(uint32_t)a_dram->address(), (uint32_t)b_dram->address(), (uint32_t)zero_dram->address(), K});
    SetRuntimeArgs(program, writer, core, {(uint32_t)q_dram->address()});
    SetRuntimeArgs(program, compute, core, {K});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> out_host(N_LIMB * TILE_HW, 0u);
    distributed::EnqueueReadMeshBuffer(cq, out_host, q_dram, true);
    uint32_t qdev[8];
    for (uint32_t l = 0; l < 8; ++l) qdev[l] = out_host[(size_t)l * TILE_HW + 0];
    const double sfpu_exact = q256_to_double(qdev);

    bool q_match = true;
    for (int l = 0; l < 8; ++l) if (qdev[l] != gq[l]) q_match = false;

    std::cout << "  catastrophic-cancellation dot (K=" << K << "): products [+2^40, +1 x" << (K - 2) << ", -2^40]\n";
    std::cout << "    true exact sum         = " << std::fixed << std::setprecision(1) << exact_true << "\n";
    std::cout << "    SFPU exact-quire (dev) = " << sfpu_exact << (q_match ? "  (bit-exact to true quire)" : "  (MISMATCH!)") << "\n";
    std::cout << "    fp32 accumulation      = " << fp32 << "   <-- WRONG\n";
    std::cout << "    bf16 accumulation      = " << bf16 << "   <-- WRONG\n";

    const bool pass = q_match && (sfpu_exact == exact_true) && (fp32 != exact_true) && (bf16 != exact_true);
    if (pass)
        std::cout << "PASS: SFPU exact-quire recovers the true sum " << exact_true
                  << " where fp32 and bf16 both collapse to " << fp32 << " (cancellation)\n";
    else
        std::cout << "FAIL: cancellation demo invariant not met\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
