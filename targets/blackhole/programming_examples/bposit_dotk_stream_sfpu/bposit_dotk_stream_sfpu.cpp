// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_dotk_stream_sfpu — STREAMING-K exact bp16 dot on real Blackhole. The reader streams
// the K (a,b) pairs just-in-time into shallow ring buffers, so K is UNBOUNDED by L1 (the old
// all-K-resident dot capped K ~100). Demonstrated at K=512. Bit-exact vs scalar bp16_madd_q256.
// Kernel: sfpu_bp16_stream.cpp (dotk faces/helpers + a per-k streaming loop).

#include <fmt/ostream.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <cstdint>
#include <vector>
#include <array>
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
static constexpr uint32_t K = 512;        // >> the old all-K-resident L1 cap (~100)
static constexpr uint32_t STREAM_DEPTH = 4;  // shallow cb_a/cb_b ring buffers

static std::vector<uint32_t> lane_tile(const uint32_t lv[LANES]) {
    std::vector<uint32_t> t(TILE_HW);
    for (uint32_t r = 0; r < TILE_H; ++r) for (uint32_t c = 0; c < TILE_W; ++c) t[r * TILE_W + c] = lv[c];
    return t;
}

int main() {
    std::vector<std::array<uint32_t, LANES>> A(K), B(K);
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            A[k][lane] = (uint32_t)(((lane * 2129u) + (k * 911u) + 3u) & 0xFFFFu);
            B[k][lane] = (uint32_t)(((lane * 1471u) + (k * 1303u) + 7u) & 0xFFFFu);
        }
    uint32_t golden[N_LIMB][LANES];
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        unsigned q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t k = 0; k < K; ++k) bp16_madd_q256(q, (int)A[k][lane], (int)B[k][lane]);
        for (uint32_t l = 0; l < N_LIMB; ++l) golden[l][lane] = q[l];
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
    auto q_dram = mk(N_LIMB), a_dram = mk(K), b_dram = mk(K), zero_dram = mk(1);
    std::vector<uint32_t> a_host(K * TILE_HW), b_host(K * TILE_HW);
    for (uint32_t k = 0; k < K; ++k) {
        auto at = lane_tile(A[k].data()); auto bt = lane_tile(B[k].data());
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
    make_cb(CBIndex::c_0, STREAM_DEPTH); make_cb(CBIndex::c_1, STREAM_DEPTH); make_cb(CBIndex::c_2, 1);
    make_cb(CBIndex::c_3, 1); make_cb(CBIndex::c_5, N_LIMB); make_cb(CBIndex::c_6, N_LIMB);
    make_cb(CBIndex::c_7, N_LIMB); make_cb(CBIndex::c_16, N_LIMB);

    std::vector<uint32_t> rargs;
    TensorAccessorArgs(*a_dram->get_backing_buffer()).append_to(rargs);
    TensorAccessorArgs(*b_dram->get_backing_buffer()).append_to(rargs);
    TensorAccessorArgs(*zero_dram->get_backing_buffer()).append_to(rargs);
    auto reader = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_dotk_stream_sfpu/kernels/dataflow/read_stream.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = rargs});
    std::vector<uint32_t> wargs;
    TensorAccessorArgs(*q_dram->get_backing_buffer()).append_to(wargs);
    auto writer = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_dotk_stream_sfpu/kernels/dataflow/write_stream.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = wargs});
    auto compute = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_dotk_stream_sfpu/kernels/compute/sfpu_bp16_stream.cpp", core,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true});
    SetRuntimeArgs(program, reader, core, {(uint32_t)a_dram->address(), (uint32_t)b_dram->address(), (uint32_t)zero_dram->address(), K});
    SetRuntimeArgs(program, writer, core, {(uint32_t)q_dram->address()});
    SetRuntimeArgs(program, compute, core, {K});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    std::vector<uint32_t> out_host(N_LIMB * TILE_HW, 0u);
    distributed::EnqueueReadMeshBuffer(cq, out_host, q_dram, true);
    auto dev = [&](uint32_t limb, uint32_t lane) -> uint32_t { return out_host[(size_t)limb * TILE_HW + lane]; };

    int bad = 0, first_bad = -1;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        bool ok = true;
        for (uint32_t l = 0; l < N_LIMB; ++l) if (dev(l, lane) != golden[l][lane]) ok = false;
        if (!ok) { ++bad; if (first_bad < 0) first_bad = (int)lane; }
    }
    const bool pass = (bad == 0);
    if (pass)
        std::cout << "PASS: STREAMING-K exact bp16 dot (K=" << K << ", cb depth=" << STREAM_DEPTH << "), "
                  << (LANES - bad) << "/" << LANES << " lanes bit-exact (8-limb quire) — K unbounded by L1\n";
    else
        std::cout << "FAIL: STREAMING-K exact bp16 dot (K=" << K << "), " << (LANES - bad) << "/" << LANES
                  << " bit-exact; first bad lane " << first_bad << "\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
