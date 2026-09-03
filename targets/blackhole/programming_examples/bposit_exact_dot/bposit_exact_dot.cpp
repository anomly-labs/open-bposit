// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_exact_dot — arbitrary-length EXACT inner product on a real Blackhole baby core:
// the BLAS-1 dot a·b accumulated in a 256-bit b-posit16 quire with NO per-product rounding,
// where an IEEE fp32 dot loses the result to catastrophic cancellation. Reuses the proven
// quire_dot_baby.cpp kernel (Sum_i bp16_prod_to_q256(a_i, b_i) -> exact quire -> readout).
//
// The inner product is the core of Gram-Schmidt projections, least-squares normal equations,
// iterative-refinement residuals, and geometric determinants — so an exact dot on-silicon is
// the primitive those scientific results rest on.
//
// Case: a = [2^E, 1, 1, ..., 1 (K), 2^E],  b = [1, 1, ..., 1 (K), 1, -1] giving products
//   2^E, K x (+1), -2^E  ->  exact dot = K. fp32 drops the 1's once the partial hits 2^E.
//
//   BR_E (default 30), BR_K (default 64).  Real silicon (TT_METAL_SIMULATOR unset).

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../../kernel/bp16_quire.h"
#include "../../kernel/bp16_encode.h"
#include "../../kernel/bp16_decode.h"
#pragma GCC diagnostic pop

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static float bits_to_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static float code_to_f32(int code) {
    int sign; unsigned M; int E2;
    bp16_decode(code, &sign, &M, &E2);
    if (M == 0u) return 0.0f;
    int msb = 31; while (msb >= 0 && !((M >> msb) & 1u)) msb--;
    uint32_t sig = (msb <= 23) ? (M << (23 - msb)) : (M >> (msb - 23));
    int biased = E2 + msb + 127;
    uint32_t bits = (biased >= 0xFF) ? (((uint32_t)sign << 31) | 0x7F800000u)
                  : (biased <= 0) ? ((uint32_t)sign << 31)
                  : (((uint32_t)sign << 31) | ((uint32_t)biased << 23) | (sig & 0x007FFFFFu));
    return bits_to_f32(bits);
}

int main() {
    constexpr int QLIMBS = 8;
    constexpr int N_OUT = QLIMBS + 1;                    // 8 quire limbs + readout
    constexpr uint32_t WORD = sizeof(uint32_t);
    const int E = std::getenv("BR_E") ? std::atoi(std::getenv("BR_E")) : 30;
    const int K = std::getenv("BR_K") ? std::atoi(std::getenv("BR_K")) : 64;

    // products 2^E, K x (+1), -2^E  ->  exact dot = K
    std::vector<int> A, B;
    A.push_back(bp16_encode_signed(0, 1u, E)); B.push_back(bp16_encode_signed(0, 1u, 0));   // 2^E * 1
    for (int i = 0; i < K; ++i) { A.push_back(bp16_encode_signed(0, 1u, 0)); B.push_back(bp16_encode_signed(0, 1u, 0)); }  // 1*1
    A.push_back(bp16_encode_signed(0, 1u, E)); B.push_back(bp16_encode_signed(1, 1u, 0));   // 2^E * -1
    const int N = (int)A.size();

    // exact host reference + naive fp32 dot
    unsigned ref_q[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
    float fp32 = 0.f;
    for (int i = 0; i < N; ++i) {
        unsigned p[QLIMBS]; bp16_prod_to_q256(A[i] & 0xFFFF, B[i] & 0xFFFF, p); q256_add(ref_q, p);
        fp32 += code_to_f32(A[i] & 0xFFFF) * code_to_f32(B[i] & 0xFFFF);
    }
    int ref_readout = bp16_encode_quire256(ref_q) & 0xFFFF;
    float exact_dot = code_to_f32(ref_readout);

    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto range = distributed::MeshCoordinateRange(mesh->shape());
    auto program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    auto mk = [&](size_t total, BufferType type) {
        distributed::DeviceLocalBufferConfig lc{.page_size = total, .buffer_type = type};
        distributed::ReplicatedBufferConfig bc{.size = total};
        return distributed::MeshBuffer::create(bc, lc, mesh.get());
    };
    auto a_dram = mk((size_t)N * WORD, BufferType::DRAM);
    auto b_dram = mk((size_t)N * WORD, BufferType::DRAM);
    auto out_dram = mk((size_t)N_OUT * WORD, BufferType::DRAM);
    auto a_l1 = mk((size_t)N * WORD, BufferType::L1);
    auto b_l1 = mk((size_t)N * WORD, BufferType::L1);
    auto out_l1 = mk((size_t)N_OUT * WORD, BufferType::L1);
    auto dbg_dram = mk((size_t)2 * N * WORD, BufferType::DRAM);

    auto kernel = CreateKernel(program,
        OVERRIDE_KERNEL_PREFIX "bposit_quire_reduce/kernels/quire_dot_baby.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, core,
        {(uint32_t)a_dram->address(), (uint32_t)b_dram->address(), (uint32_t)out_dram->address(),
         (uint32_t)a_l1->address(), (uint32_t)b_l1->address(), (uint32_t)out_l1->address(),
         (uint32_t)N, (uint32_t)dbg_dram->address()});

    std::vector<uint32_t> aw(N), bw(N);
    for (int i = 0; i < N; ++i) { aw[i] = (uint32_t)(A[i] & 0xFFFF); bw[i] = (uint32_t)(B[i] & 0xFFFF); }
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, aw, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, bw, false);
    workload.add_program(range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    std::vector<uint32_t> res(N_OUT, 0);
    distributed::EnqueueReadMeshBuffer(cq, res, out_dram, true);

    int bad = 0;
    for (int i = 0; i < QLIMBS; ++i) if (res[i] != ref_q[i]) bad++;
    int dev_readout = (int)(res[QLIMBS] & 0xFFFF);
    float quire_dot = code_to_f32(dev_readout);

    std::cout << "=== exact 256-bit quire DOT PRODUCT vs IEEE fp32 — cancellation, real silicon ===\n";
    std::cout << "  a.b with products: 2^" << E << " + " << K << " x (+1) + -2^" << E
              << "   (N=" << N << " pairs, exact dot = " << K << ")\n";
    std::cout << "  quire dot (exact) = " << quire_dot << "   [device quire "
              << (bad == 0 ? "bit-exact vs oracle" : "MISMATCH") << "]\n";
    std::cout << "  fp32 dot          = " << fp32 << "   (lost the inner product)\n";
    bool pass = (bad == 0) && (dev_readout == ref_readout) && (fp32 != exact_dot);
    std::cout << (pass ? "PASS" : "FAIL")
              << ": exact-quire inner product recovered " << quire_dot << " where fp32 got "
              << fp32 << " — real silicon\n";
    mesh->close();
    return pass ? 0 : 1;
}
