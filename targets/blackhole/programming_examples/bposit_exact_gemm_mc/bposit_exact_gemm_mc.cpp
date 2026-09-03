// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_exact_gemm_mc — MULTI-CORE exact-quire GEMM C[M,N]=A[M,K]·B[K,N] across the
// Blackhole grid: the M*N exact-quire dots are distributed over the Tensix cores
// (split_work_to_cores), each computed bit-exact with no per-product rounding. The
// throughput generalization of bposit_exact_gemm, reusing the proven
// quire_matmul_mc_baby.cpp kernel (per-element math byte-identical to the single-core path).
//
// Cancellation GEMM: every C[i,j] = 2^E + INNER x(+1) - 2^E = INNER; fp32 collapses each to 0.
//
//   BR_M (16), BR_N (16), BR_E (30), BR_INNER (16).  Real silicon.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/work_split.hpp>
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
    const int M = std::getenv("BR_M") ? std::atoi(std::getenv("BR_M")) : 16;
    const int N = std::getenv("BR_N") ? std::atoi(std::getenv("BR_N")) : 16;
    const int E = std::getenv("BR_E") ? std::atoi(std::getenv("BR_E")) : 30;
    const int INNER = std::getenv("BR_INNER") ? std::atoi(std::getenv("BR_INNER")) : 16;
    const int K = INNER + 2;

    std::vector<int> A((size_t)M * K), B((size_t)K * N);
    const int big = bp16_encode_signed(0, 1u, E), one = bp16_encode_signed(0, 1u, 0), neg = bp16_encode_signed(1, 1u, 0);
    for (int i = 0; i < M; ++i) { A[i * K + 0] = big; for (int k = 1; k <= INNER; ++k) A[i * K + k] = one; A[i * K + K - 1] = big; }
    for (int j = 0; j < N; ++j) { B[0 * N + j] = one; for (int k = 1; k <= INNER; ++k) B[k * N + j] = one; B[(K - 1) * N + j] = neg; }

    // exact reference readouts + fp32 GEMM
    std::vector<int> refC((size_t)M * N);
    int fp32_wrong = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            unsigned q[QLIMBS] = {0, 0, 0, 0, 0, 0, 0, 0}; float f = 0.f;
            for (int k = 0; k < K; ++k) {
                unsigned p[QLIMBS]; bp16_prod_to_q256(A[i * K + k] & 0xFFFF, B[k * N + j] & 0xFFFF, p); q256_add(q, p);
                f += code_to_f32(A[i * K + k] & 0xFFFF) * code_to_f32(B[k * N + j] & 0xFFFF);
            }
            refC[i * N + j] = bp16_encode_quire256(q) & 0xFFFF;
            if (f != (float)INNER) fp32_wrong++;
        }

    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto rg = distributed::MeshCoordinateRange(mesh->shape());
    auto program = CreateProgram();
    const int C_COUNT = M * N;
    CoreCoord grid = mesh->compute_with_storage_grid_size();
    auto [num_cores, all_cores, cg1, cg2, opc1, opc2] =
        split_work_to_cores(grid, (uint32_t)C_COUNT, /*row_major=*/true);
    uint32_t max_opc = std::max(opc1, opc2);

    auto mk = [&](size_t page, size_t sz, BufferType type) {
        distributed::DeviceLocalBufferConfig lc{.page_size = page, .buffer_type = type};
        distributed::ReplicatedBufferConfig bc{.size = sz};
        return distributed::MeshBuffer::create(bc, lc, mesh.get());
    };
    auto a_dram = mk((size_t)M * K * WORD, (size_t)M * K * WORD, BufferType::DRAM);   // single page, shared
    auto b_dram = mk((size_t)K * N * WORD, (size_t)K * N * WORD, BufferType::DRAM);
    auto c_dram = mk(WORD, (size_t)C_COUNT * WORD, BufferType::DRAM);                 // per-element paged
    auto gate_dram = mk((size_t)QLIMBS * WORD, (size_t)QLIMBS * WORD, BufferType::DRAM);
    auto a_l1 = mk((size_t)M * K * WORD, (size_t)M * K * WORD, BufferType::L1);
    auto b_l1 = mk((size_t)K * N * WORD, (size_t)K * N * WORD, BufferType::L1);
    auto c_l1 = mk((size_t)max_opc * WORD, (size_t)max_opc * WORD, BufferType::L1);

    auto kernel = CreateKernel(program,
        OVERRIDE_KERNEL_PREFIX "bposit_quire_matmul_multicore/kernels/quire_matmul_mc_baby.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    uint32_t out_start = 0;
    for (auto& [group, opc] : {std::make_pair(cg1, opc1), std::make_pair(cg2, opc2)})
        for (const auto& r : group.ranges())
            for (const auto& core : r) {
                uint32_t out_end = out_start + opc;
                SetRuntimeArgs(program, kernel, core,
                    {(uint32_t)a_dram->address(), (uint32_t)b_dram->address(), (uint32_t)c_dram->address(),
                     (uint32_t)a_l1->address(), (uint32_t)b_l1->address(), (uint32_t)c_l1->address(),
                     (uint32_t)M, (uint32_t)K, (uint32_t)N, out_start, out_end, 0u,
                     (uint32_t)gate_dram->address()});
                out_start = out_end;
            }

    std::vector<uint32_t> aw(M * K), bw(K * N);
    for (int i = 0; i < M * K; ++i) aw[i] = (uint32_t)(A[i] & 0xFFFF);
    for (int i = 0; i < K * N; ++i) bw[i] = (uint32_t)(B[i] & 0xFFFF);
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, aw, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, bw, false);
    workload.add_program(rg, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    std::vector<uint32_t> c_res(C_COUNT, 0);
    distributed::EnqueueReadMeshBuffer(cq, c_res, c_dram, true);

    int bad = 0;
    for (int o = 0; o < C_COUNT; ++o) if ((int)(c_res[o] & 0xFFFF) != refC[o]) bad++;
    float c00 = code_to_f32((int)(c_res[0] & 0xFFFF));

    std::cout << "=== MULTI-CORE exact 256-bit quire GEMM across the Blackhole grid (real silicon) ===\n";
    std::cout << "  C[" << M << "," << N << "] = A[" << M << "," << K << "]B[" << K << "," << N
              << "]; " << C_COUNT << " exact-quire dots across " << num_cores << " cores\n";
    std::cout << "  every C[i,j] is a cancellation dot 2^" << E << " + " << INNER
              << " x(+1) - 2^" << E << "  (exact = " << INNER << ")\n";
    std::cout << "  exact-quire GEMM: " << (C_COUNT - bad) << "/" << C_COUNT
              << " outputs bit-exact vs oracle; C[0,0] = " << c00 << "\n";
    std::cout << "  fp32 GEMM: " << fp32_wrong << "/" << C_COUNT << " entries collapsed to ~0\n";
    bool pass = (bad == 0) && (fp32_wrong == C_COUNT);
    std::cout << (pass ? "PASS" : "FAIL") << ": distributed exact GEMM, " << (C_COUNT - bad) << "/"
              << C_COUNT << " bit-exact across " << num_cores << " cores — real silicon\n";
    mesh->close();
    return pass ? 0 : 1;
}
