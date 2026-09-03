// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_exact_gemm — arbitrary-size EXACT matrix multiply C[M,N] = A[M,K]·B[K,N] on a real
// Blackhole baby core: each output is an exact-quire dot of length K (no per-product
// rounding), where an IEEE fp32 GEMM loses every entry to catastrophic cancellation. Drives
// the proven quire_matmul_baby.cpp kernel (flexible M,K,N) with file/host-built bp16 codes.
// The BLAS-3 completion of the on-card exact set (reduce / dot / matmul).
//
// Case (every entry a cancellation): A[i,:] = [2^E, 1, ..., 1 (K-2), 2^E],
//   B[:,j] = [1, 1, ..., 1 (K-2), -1]  ->  C[i,j] = (K-2). fp32 collapses each to 0.
//
//   BR_M (4), BR_N (4), BR_E (30); inner count is K-2 (K = inner+2).  Real silicon.

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
    constexpr uint32_t WORD = sizeof(uint32_t);
    const int M = std::getenv("BR_M") ? std::atoi(std::getenv("BR_M")) : 4;
    const int N = std::getenv("BR_N") ? std::atoi(std::getenv("BR_N")) : 4;
    const int E = std::getenv("BR_E") ? std::atoi(std::getenv("BR_E")) : 30;
    const int INNER = std::getenv("BR_INNER") ? std::atoi(std::getenv("BR_INNER")) : 16;
    const int K = INNER + 2;
    const int exact_entry = INNER;                        // each C[i,j] = INNER

    // A[M,K] row-major, B[K,N] row-major (bp16 codes)
    std::vector<int> A((size_t)M * K), B((size_t)K * N);
    const int big = bp16_encode_signed(0, 1u, E);         // 2^E
    const int one = bp16_encode_signed(0, 1u, 0);         // +1
    const int neg = bp16_encode_signed(1, 1u, 0);         // -1
    for (int i = 0; i < M; ++i) {
        A[i * K + 0] = big;
        for (int k = 1; k <= INNER; ++k) A[i * K + k] = one;
        A[i * K + (K - 1)] = big;
    }
    for (int j = 0; j < N; ++j) {
        B[0 * N + j] = one;
        for (int k = 1; k <= INNER; ++k) B[k * N + j] = one;
        B[(K - 1) * N + j] = neg;
    }

    // exact host reference (per output) + fp32 GEMM
    std::vector<int> refC((size_t)M * N);
    std::vector<float> fp32C((size_t)M * N, 0.f);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            unsigned q[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0};
            float f = 0.f;
            for (int k = 0; k < K; ++k) {
                unsigned p[QLIMBS];
                bp16_prod_to_q256(A[i * K + k] & 0xFFFF, B[k * N + j] & 0xFFFF, p);
                q256_add(q, p);
                f += code_to_f32(A[i * K + k] & 0xFFFF) * code_to_f32(B[k * N + j] & 0xFFFF);
            }
            refC[i * N + j] = bp16_encode_quire256(q) & 0xFFFF;
            fp32C[i * N + j] = f;
        }

    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto rangeC = distributed::MeshCoordinateRange(mesh->shape());
    auto program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    const int C_COUNT = M * N;
    const int N_OUT = C_COUNT + QLIMBS;
    auto mk = [&](size_t total, BufferType type) {
        distributed::DeviceLocalBufferConfig lc{.page_size = total, .buffer_type = type};
        distributed::ReplicatedBufferConfig bc{.size = total};
        return distributed::MeshBuffer::create(bc, lc, mesh.get());
    };
    auto a_dram = mk((size_t)M * K * WORD, BufferType::DRAM);
    auto b_dram = mk((size_t)K * N * WORD, BufferType::DRAM);
    auto out_dram = mk((size_t)N_OUT * WORD, BufferType::DRAM);
    auto a_l1 = mk((size_t)M * K * WORD, BufferType::L1);
    auto b_l1 = mk((size_t)K * N * WORD, BufferType::L1);
    auto out_l1 = mk((size_t)N_OUT * WORD, BufferType::L1);
    auto dbg_dram = mk((size_t)(M * K + K * N) * WORD, BufferType::DRAM);

    auto kernel = CreateKernel(program,
        OVERRIDE_KERNEL_PREFIX "bposit_quire_reduce/kernels/quire_matmul_baby.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, core,
        {(uint32_t)a_dram->address(), (uint32_t)b_dram->address(), (uint32_t)out_dram->address(),
         (uint32_t)a_l1->address(), (uint32_t)b_l1->address(), (uint32_t)out_l1->address(),
         (uint32_t)M, (uint32_t)K, (uint32_t)N, 0u, 0u, (uint32_t)dbg_dram->address()});

    std::vector<uint32_t> aw(M * K), bw(K * N);
    for (int i = 0; i < M * K; ++i) aw[i] = (uint32_t)(A[i] & 0xFFFF);
    for (int i = 0; i < K * N; ++i) bw[i] = (uint32_t)(B[i] & 0xFFFF);
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, aw, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, bw, false);
    workload.add_program(rangeC, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    std::vector<uint32_t> res(N_OUT, 0);
    distributed::EnqueueReadMeshBuffer(cq, res, out_dram, true);

    int bad = 0, fp32_zero = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            if ((int)(res[i * N + j] & 0xFFFF) != refC[i * N + j]) bad++;
            if (fp32C[i * N + j] != (float)exact_entry) fp32_zero++;
        }
    float c00_quire = code_to_f32((int)(res[0] & 0xFFFF));

    std::cout << "=== exact 256-bit quire GEMM  C=A.B  on a Blackhole baby core (real silicon) ===\n";
    std::cout << "  M=" << M << " K=" << K << " N=" << N
              << "; every C[i,j] is a cancellation dot 2^" << E << " + " << INNER
              << " x(+1) - 2^" << E << "  (exact = " << exact_entry << ")\n";
    std::cout << "  exact-quire GEMM: " << (M * N - bad) << "/" << (M * N)
              << " outputs bit-exact vs oracle; C[0,0] = " << c00_quire << "\n";
    std::cout << "  fp32 GEMM: " << fp32_zero << "/" << (M * N)
              << " entries wrong (each collapsed to ~0)\n";
    bool pass = (bad == 0) && (fp32_zero == M * N);
    std::cout << (pass ? "PASS" : "FAIL")
              << ": exact-quire GEMM recovered every entry (" << exact_entry
              << ") that fp32 lost — real silicon\n";
    mesh->close();
    return pass ? 0 : 1;
}
