// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_matmul_full_sfpu — fully ARBITRARY-(M,N,K) multi-core exact-quire bp16 MATMUL. Removes the
// one-row-per-core limit of bposit_matmul_mc_sfpu: each core handles a SLICE [row_start,
// row_start+n_rows) of output rows, looping over them in-kernel (each row = a fresh exact
// K-dot, quire reset to 0). M can exceed the core count. Gated bit-exact vs scalar
// bp16_madd_q256; reports aggregate throughput. Compute kernel = sfpu_bp16_mc2.cpp (the dotk
// faces/helpers + a per-row outer loop).

#include <fmt/ostream.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>

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
static constexpr uint32_t K = 128;
static constexpr uint32_t M = 140;
static constexpr uint32_t STREAM_DEPTH = 4;

static std::vector<uint32_t> bcast_tile(uint32_t v) { return std::vector<uint32_t>(TILE_HW, v); }
static std::vector<uint32_t> lane_tile(const uint32_t lv[LANES]) {
    std::vector<uint32_t> t(TILE_HW);
    for (uint32_t r = 0; r < TILE_H; ++r) for (uint32_t c = 0; c < TILE_W; ++c) t[r * TILE_W + c] = lv[c];
    return t;
}

int main() {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    auto grid = mesh_device->compute_with_storage_grid_size();

    std::vector<std::vector<uint32_t>> X(M, std::vector<uint32_t>(K));
    uint32_t W[K][LANES];
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < K; ++k) X[m][k] = (uint32_t)(((m * 1777u) + (k * 911u) + 3u) & 0xFFFFu);
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < LANES; ++n) W[k][n] = (uint32_t)(((k * 1303u) + (n * 1471u) + 7u) & 0xFFFFu);

    std::vector<std::array<std::array<uint32_t, N_LIMB>, LANES>> golden(M);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < LANES; ++n) {
            unsigned q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint32_t k = 0; k < K; ++k) bp16_madd_q256(q, (int)X[m][k], (int)W[k][n]);
            for (uint32_t l = 0; l < N_LIMB; ++l) golden[m][n][l] = q[l];
        }

    constexpr uint32_t tile_bytes = sizeof(uint32_t) * TILE_HW;
    auto mk = [&](uint32_t n) {
        distributed::DeviceLocalBufferConfig lc{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig bc{.size = (size_t)n * tile_bytes};
        return distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    };
    auto a_dram = mk(M * K), b_dram = mk(K), q_dram = mk(M * N_LIMB), zero_dram = mk(1);

    std::vector<uint32_t> a_host(M * K * TILE_HW), b_host(K * TILE_HW);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < K; ++k) {
            auto at = bcast_tile(X[m][k]);
            std::copy(at.begin(), at.end(), a_host.begin() + (size_t)(m * K + k) * TILE_HW);
        }
    for (uint32_t k = 0; k < K; ++k) {
        auto bt = lane_tile(W[k]);
        std::copy(bt.begin(), bt.end(), b_host.begin() + (size_t)k * TILE_HW);
    }
    std::vector<uint32_t> zero_host(TILE_HW, 0u);
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, zero_dram, zero_host, true);

    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();
    auto [num_cores, all_cores, cg1, cg2, opc1, opc2] =
        tt::tt_metal::split_work_to_cores(grid, M, /*row_major=*/true);

    auto make_cb = [&](CBIndex idx, uint32_t n) {
        CircularBufferConfig cfg(n * tile_bytes, {{idx, tt::DataFormat::Int32}});
        cfg.set_page_size(idx, tile_bytes);
        CreateCircularBuffer(program, all_cores, cfg);
    };
    make_cb(CBIndex::c_0, STREAM_DEPTH); make_cb(CBIndex::c_1, STREAM_DEPTH); make_cb(CBIndex::c_2, 1);
    make_cb(CBIndex::c_3, 1); make_cb(CBIndex::c_5, N_LIMB); make_cb(CBIndex::c_6, N_LIMB);
    make_cb(CBIndex::c_7, N_LIMB); make_cb(CBIndex::c_16, N_LIMB);

    std::vector<uint32_t> rargs;
    TensorAccessorArgs(*a_dram->get_backing_buffer()).append_to(rargs);
    TensorAccessorArgs(*b_dram->get_backing_buffer()).append_to(rargs);
    TensorAccessorArgs(*zero_dram->get_backing_buffer()).append_to(rargs);
    auto reader = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_matmul_full_sfpu/kernels/dataflow/read_full.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = rargs});
    std::vector<uint32_t> wargs;
    TensorAccessorArgs(*q_dram->get_backing_buffer()).append_to(wargs);
    auto writer = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_matmul_full_sfpu/kernels/dataflow/write_full.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = wargs});
    auto compute = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_matmul_full_sfpu/kernels/compute/sfpu_bp16_full.cpp", all_cores,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true});

    uint32_t row_start = 0;
    for (const auto& [group, per] : {std::make_pair(cg1, opc1), std::make_pair(cg2, opc2)}) {
        for (const auto& range : group.ranges())
            for (const auto& core : range) {
                SetRuntimeArgs(program, reader, core,
                               {(uint32_t)a_dram->address(), (uint32_t)b_dram->address(),
                                (uint32_t)zero_dram->address(), K, row_start, per});
                SetRuntimeArgs(program, writer, core, {(uint32_t)q_dram->address(), row_start, per});
                SetRuntimeArgs(program, compute, core, {per, K});
                row_start += per;
            }
    }

    workload.add_program(device_range, std::move(program));
    auto t0 = std::chrono::high_resolution_clock::now();
    distributed::EnqueueMeshWorkload(cq, workload, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    std::vector<uint32_t> out_host(M * N_LIMB * TILE_HW, 0u);
    distributed::EnqueueReadMeshBuffer(cq, out_host, q_dram, true);
    auto dev = [&](uint32_t m, uint32_t n, uint32_t l) -> uint32_t {
        return out_host[(size_t)(m * N_LIMB + l) * TILE_HW + 0 * TILE_W + n];
    };

    uint32_t bad = 0; int fm = -1, fn = -1;
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < LANES; ++n) {
            bool ok = true;
            for (uint32_t l = 0; l < N_LIMB; ++l) if (dev(m, n, l) != golden[m][n][l]) ok = false;
            if (!ok) { ++bad; if (fm < 0) { fm = (int)m; fn = (int)n; } }
        }

    const uint32_t total = M * LANES;
    const uint64_t macs = (uint64_t)M * LANES * K;
    const uint32_t rows_per_core = (M + num_cores - 1) / num_cores;
    const bool pass = (bad == 0);
    std::cout << "  grid " << grid.x << "x" << grid.y << ", " << num_cores << " cores, M=" << M
              << " rows (~" << rows_per_core << " rows/core), N=" << LANES << ", K=" << K << "\n";
    if (pass) {
        std::cout << "PASS: ARBITRARY-(M,N,K) streaming exact-quire MATMUL, " << (total - bad) << "/" << total
                  << " output elements bit-exact (8-limb quire)\n";
        std::cout << "  [perf] " << macs << " exact MACs in 1 dispatch, wall " << std::fixed
                  << std::setprecision(1) << us << " us = " << std::setprecision(1) << ((double)macs / us)
                  << " MAC/us aggregate (incl. dispatch + DRAM I/O)\n";
    } else {
        std::cout << "FAIL: ARBITRARY-(M,N,K) streaming exact-quire MATMUL, " << (total - bad) << "/" << total
                  << " bit-exact; first bad Y[" << fm << "][" << fn << "]\n";
    }

    mesh_device->close();
    return pass ? 0 : 1;
}
