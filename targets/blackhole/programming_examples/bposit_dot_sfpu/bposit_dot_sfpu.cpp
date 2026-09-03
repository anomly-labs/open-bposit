// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_dot_sfpu — MILESTONE M3.5: an exact bp16 DOT PRODUCT on a PHYSICAL Tenstorrent
// Blackhole. Per lane, accumulate q[8] += a_k*b_k over K pairs into a 256-bit quire, exactly
// (no per-product rounding), bit-identical to the scalar bp16_madd_q256 chain. The running
// quire lives in DRAM (zero-init); the host dispatches K accumulation steps, each computing
// the signed contribution (M3b) and adding it to q via the solved q256_add. Kernel:
// kernels/compute/sfpu_bp16_dot.cpp.

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

static constexpr uint32_t LANES = 32;
static constexpr uint32_t TILE_W = 32, TILE_H = 32, TILE_HW = TILE_W * TILE_H;
static constexpr uint32_t N_LIMB = 8;
static constexpr uint32_t K = 4;   // dot-product length (start small, validates accumulation)

static std::vector<uint32_t> make_lane_tile(const uint32_t lane_vals[LANES]) {
    std::vector<uint32_t> tile(TILE_HW);
    for (uint32_t r = 0; r < TILE_H; ++r)
        for (uint32_t c = 0; c < TILE_W; ++c) tile[r * TILE_W + c] = lane_vals[c];
    return tile;
}

int main() {
    // K code pairs per lane (mixed signs/regimes so the running quire sees adds AND subtracts).
    uint32_t A[K][LANES], B[K][LANES];
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            A[k][lane] = (uint32_t)(((lane * 2129u) + (k * 911u) + 3u) & 0xFFFFu);
            B[k][lane] = (uint32_t)(((lane * 1471u) + (k * 1303u) + 7u) & 0xFFFFu);
        }

    // golden: scalar bp16_madd_q256 chain into q[8] per lane.
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
    auto q_dram = mk(N_LIMB);
    auto a_dram = mk(K);
    auto b_dram = mk(K);
    auto zero_dram = mk(1);

    // marshal A,B (K tiles each), zero, and zero-init q_dram.
    std::vector<uint32_t> a_host(K * TILE_HW), b_host(K * TILE_HW);
    for (uint32_t k = 0; k < K; ++k) {
        auto at = make_lane_tile(A[k]);
        auto bt = make_lane_tile(B[k]);
        std::copy(at.begin(), at.end(), a_host.begin() + (size_t)k * TILE_HW);
        std::copy(bt.begin(), bt.end(), b_host.begin() + (size_t)k * TILE_HW);
    }
    std::vector<uint32_t> zero_host(TILE_HW, 0u);
    std::vector<uint32_t> q_zero(N_LIMB * TILE_HW, 0u);
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, zero_dram, zero_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, q_dram, q_zero, true);  // q starts at 0

    // One accumulation step (pair k): q_dram += A[k]*B[k].
    auto run_step = [&](uint32_t k) {
        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        auto program = CreateProgram();

        auto make_cb = [&](CBIndex idx, uint32_t n) {
            CircularBufferConfig cfg(n * tile_bytes, {{idx, tt::DataFormat::Int32}});
            cfg.set_page_size(idx, tile_bytes);
            CreateCircularBuffer(program, core, cfg);
        };
        make_cb(CBIndex::c_0, 1);       // code_a
        make_cb(CBIndex::c_1, 1);       // code_b
        make_cb(CBIndex::c_2, 1);       // zero
        make_cb(CBIndex::c_3, 1);       // carry_mid
        make_cb(CBIndex::c_4, N_LIMB);  // q_in
        make_cb(CBIndex::c_5, N_LIMB);  // C
        make_cb(CBIndex::c_16, N_LIMB); // q_out

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
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);  // q_dram updated in place
    };

    for (uint32_t k = 0; k < K; ++k) run_step(k);

    // read back final q and gate
    std::vector<uint32_t> out_host(N_LIMB * TILE_HW, 0u);
    distributed::EnqueueReadMeshBuffer(cq, out_host, q_dram, true);
    auto dev = [&](uint32_t limb, uint32_t lane) -> uint32_t {
        return out_host[(size_t)limb * TILE_HW + 0 * TILE_W + lane];
    };

    {
        const int diag_lanes[] = {0, 1, 5, 10, 16, 31};
        for (int lane : diag_lanes) {
            std::cout << "  [diag] lane " << std::setw(2) << lane << " | q dev/gold:";
            for (uint32_t l = 0; l < N_LIMB; ++l) std::cout << " " << dev(l, lane) << "/" << golden[l][lane];
            std::cout << "\n";
        }
    }

    int bad = 0, first_bad = -1;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        bool ok = true;
        for (uint32_t l = 0; l < N_LIMB; ++l)
            if (dev(l, lane) != golden[l][lane]) ok = false;
        if (!ok) {
            if (bad < 6) {
                std::cout << "  lane " << lane << " mismatch:";
                for (uint32_t l = 0; l < N_LIMB; ++l)
                    if (dev(l, lane) != golden[l][lane])
                        std::cout << " [l" << l << " dev=0x" << std::hex << dev(l, lane) << " gold=0x" << golden[l][lane] << std::dec << "]";
                std::cout << "\n";
            }
            ++bad;
            if (first_bad < 0) first_bad = (int)lane;
        }
    }

    const int good = (int)LANES - bad;
    const bool pass = (bad == 0);
    if (pass)
        std::cout << "PASS: SFPU lane-parallel bp16 exact dot (K=" << K << "), " << good << "/" << LANES
                  << " lanes bit-exact (8-limb quire)\n";
    else
        std::cout << "FAIL: SFPU lane-parallel bp16 exact dot (K=" << K << "), " << good << "/" << LANES
                  << " lanes bit-exact; first bad lane " << first_bad << "\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
