// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_matmul_sfpu — MILESTONE M4: a full EXACT-QUIRE bp16 MATMUL on a PHYSICAL
// Tenstorrent Blackhole, the capstone of the SFPU-vectorized chain. Computes
//   Y[m][n] = sum_k X[m][k] * W[k][n]   (exact 256-bit quire per output element)
// for an (M x K) @ (K x N) matmul with N = 32 (one SFPU lane per output column). Each
// output row is one length-K exact dot product per lane, reusing the PROVEN M3.5 dot
// kernel VERBATIM (kernels live in ../bposit_dot_sfpu/): per row m we broadcast X[m][k]
// to all lanes as operand A and feed W[k][n] as operand B, so lane n accumulates
// Y[m][n]. Gated bit-exact (M*N*8 limbs) vs the scalar bp16_madd_q256 chain.

#include <fmt/ostream.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <cstdint>
#include <cstdlib>
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

static constexpr uint32_t LANES = 32;             // = N (output columns)
static constexpr uint32_t TILE_W = 32, TILE_H = 32, TILE_HW = TILE_W * TILE_H;
static constexpr uint32_t N_LIMB = 8;
static constexpr uint32_t M = 4;                  // output rows
static constexpr uint32_t K = 8;                  // contraction length

static std::vector<uint32_t> bcast_tile(uint32_t v) {  // every element = v (broadcast scalar)
    return std::vector<uint32_t>(TILE_HW, v);
}
static std::vector<uint32_t> lane_tile(const uint32_t lane_vals[LANES]) {
    std::vector<uint32_t> tile(TILE_HW);
    for (uint32_t r = 0; r < TILE_H; ++r)
        for (uint32_t c = 0; c < TILE_W; ++c) tile[r * TILE_W + c] = lane_vals[c];
    return tile;
}

int main() {
    // X is (M x K) bp16 codes; W is (K x N) bp16 codes (N = LANES). Mixed signs.
    uint32_t X[M][K], W[K][LANES];
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < K; ++k) X[m][k] = (uint32_t)(((m * 1777u) + (k * 911u) + 3u) & 0xFFFFu);
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < LANES; ++n) W[k][n] = (uint32_t)(((k * 1303u) + (n * 1471u) + 7u) & 0xFFFFu);

    // golden: Y[m][n] quire = scalar bp16_madd_q256 chain over k of X[m][k]*W[k][n].
    uint32_t golden[M][LANES][N_LIMB];
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < LANES; ++n) {
            unsigned q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint32_t k = 0; k < K; ++k) bp16_madd_q256(q, (int)X[m][k], (int)W[k][n]);
            for (uint32_t l = 0; l < N_LIMB; ++l) golden[m][n][l] = q[l];
        }

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t tile_bytes = sizeof(uint32_t) * TILE_HW;

    auto mk = [&](uint32_t n) {
        distributed::DeviceLocalBufferConfig lc{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig bc{.size = (size_t)n * tile_bytes};
        return distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    };
    auto q_dram = mk(N_LIMB);   // running quire for the current row
    auto a_dram = mk(K);        // A[k] = X[m][k] broadcast (re-marshaled per row m)
    auto b_dram = mk(K);        // B[k][n] = W[k][n]
    auto zero_dram = mk(1);

    // W and zero are constant across rows.
    std::vector<uint32_t> b_host(K * TILE_HW);
    for (uint32_t k = 0; k < K; ++k) {
        auto bt = lane_tile(W[k]);
        std::copy(bt.begin(), bt.end(), b_host.begin() + (size_t)k * TILE_HW);
    }
    std::vector<uint32_t> zero_host(TILE_HW, 0u);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, zero_dram, zero_host, true);

    // One accumulation step (row m, pair k): q_dram += A[k]*B[k]  (reuses the M3.5 dot kernel).
    auto run_step = [&](uint32_t k) {
        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        auto program = CreateProgram();
        auto make_cb = [&](CBIndex idx, uint32_t n) {
            CircularBufferConfig cfg(n * tile_bytes, {{idx, tt::DataFormat::Int32}});
            cfg.set_page_size(idx, tile_bytes);
            CreateCircularBuffer(program, core, cfg);
        };
        make_cb(CBIndex::c_0, 1); make_cb(CBIndex::c_1, 1); make_cb(CBIndex::c_2, 1);
        make_cb(CBIndex::c_3, 1); make_cb(CBIndex::c_4, N_LIMB); make_cb(CBIndex::c_5, N_LIMB);
        make_cb(CBIndex::c_16, N_LIMB);

        std::vector<uint32_t> rargs;
        TensorAccessorArgs(*q_dram->get_backing_buffer()).append_to(rargs);
        TensorAccessorArgs(*a_dram->get_backing_buffer()).append_to(rargs);
        TensorAccessorArgs(*b_dram->get_backing_buffer()).append_to(rargs);
        TensorAccessorArgs(*zero_dram->get_backing_buffer()).append_to(rargs);
        auto reader = CreateKernel(
            program, OVERRIDE_KERNEL_PREFIX "bposit_dot_sfpu/kernels/dataflow/read_dot.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = rargs});
        std::vector<uint32_t> wargs;
        TensorAccessorArgs(*q_dram->get_backing_buffer()).append_to(wargs);
        auto writer = CreateKernel(
            program, OVERRIDE_KERNEL_PREFIX "bposit_dot_sfpu/kernels/dataflow/write_q.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = wargs});
        CreateKernel(program, OVERRIDE_KERNEL_PREFIX "bposit_dot_sfpu/kernels/compute/sfpu_bp16_dot.cpp", core,
                     ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true});
        SetRuntimeArgs(program, reader, core,
                       {(uint32_t)q_dram->address(), (uint32_t)a_dram->address(),
                        (uint32_t)b_dram->address(), (uint32_t)zero_dram->address(), k, k});
        SetRuntimeArgs(program, writer, core, {(uint32_t)q_dram->address()});
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    };

    // Y[m][n] (8-limb quire) for each output row, read back.
    uint32_t dev[M][LANES][N_LIMB];
    std::vector<uint32_t> q_zero(N_LIMB * TILE_HW, 0u);
    for (uint32_t m = 0; m < M; ++m) {
        // A[k] = X[m][k] broadcast to all lanes
        std::vector<uint32_t> a_host(K * TILE_HW);
        for (uint32_t k = 0; k < K; ++k) {
            auto at = bcast_tile(X[m][k]);
            std::copy(at.begin(), at.end(), a_host.begin() + (size_t)k * TILE_HW);
        }
        distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_host, false);
        distributed::EnqueueWriteMeshBuffer(cq, q_dram, q_zero, true);   // reset quire for this row
        for (uint32_t k = 0; k < K; ++k) run_step(k);
        std::vector<uint32_t> out(N_LIMB * TILE_HW, 0u);
        distributed::EnqueueReadMeshBuffer(cq, out, q_dram, true);
        for (uint32_t n = 0; n < LANES; ++n)
            for (uint32_t l = 0; l < N_LIMB; ++l)
                dev[m][n][l] = out[(size_t)l * TILE_HW + 0 * TILE_W + n];
    }

    int bad = 0; int first_m = -1, first_n = -1;
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < LANES; ++n) {
            bool ok = true;
            for (uint32_t l = 0; l < N_LIMB; ++l) if (dev[m][n][l] != golden[m][n][l]) ok = false;
            if (!ok) { ++bad; if (first_m < 0) { first_m = (int)m; first_n = (int)n; } }
        }

    std::cout << "  [diag] Y[0][0] dev/gold:";
    for (uint32_t l = 0; l < N_LIMB; ++l) std::cout << " " << dev[0][0][l] << "/" << golden[0][0][l];
    std::cout << "\n  [diag] Y[3][31] dev/gold:";
    for (uint32_t l = 0; l < N_LIMB; ++l) std::cout << " " << dev[M - 1][LANES - 1][l] << "/" << golden[M - 1][LANES - 1][l];
    std::cout << "\n";

    const uint32_t total = M * LANES;
    const bool pass = (bad == 0);
    if (pass)
        std::cout << "PASS: SFPU exact-quire bp16 MATMUL (" << M << "x" << K << " @ " << K << "x" << LANES
                  << "), " << (total - bad) << "/" << total << " output elements bit-exact (8-limb quire)\n";
    else
        std::cout << "FAIL: SFPU exact-quire bp16 MATMUL, " << (total - bad) << "/" << total
                  << " elements bit-exact; first bad Y[" << first_m << "][" << first_n << "]\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
