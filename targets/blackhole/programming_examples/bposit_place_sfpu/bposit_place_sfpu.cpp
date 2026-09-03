// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_place_sfpu — MILESTONE M3 of the SFPU-vectorized exact-quire b-posit matmul:
// lane-parallel WINDOWED PLACEMENT of a bp16xbp16 product into the 256-bit quire, on a
// PHYSICAL Tenstorrent Blackhole. M3b (full SIGNED): per lane, two codes (a,b) decode ->
// product -> place M_p<<(E2_p+96) into the 8-limb quire contribution, WITH the conditional
// two's-complement when sign_p (sa^sb). Gated BIT-EXACT (8 limbs x 32 lanes) against the
// FULL scalar bp16_prod_to_q256 across same- AND opposite-sign products. Kernel:
// kernels/compute/sfpu_bp16_place.cpp.

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

static std::vector<uint32_t> make_lane_tile(const uint32_t lane_vals[LANES]) {
    std::vector<uint32_t> tile(TILE_HW);
    for (uint32_t r = 0; r < TILE_H; ++r)
        for (uint32_t c = 0; c < TILE_W; ++c) tile[r * TILE_W + c] = lane_vals[c];
    return tile;
}

int main() {
    // 32 (a,b) pairs chosen to spread E2_p so the product lands at many limbs wi and the
    // drop cases (shift<0, wi>=8) are exercised. Low lanes are explicit edges.
    uint32_t ca[LANES], cb[LANES];
    const uint32_t ea_[10] = {0x0000, 0x4000, 0x7FFF, 0x0001, 0xC000, 0x7FFF, 0x0001, 0x6000, 0xC000, 0x2000};
    const uint32_t eb_[10] = {0x4000, 0xC000, 0x7FFF, 0x0001, 0x4000, 0x4000, 0x4000, 0x6000, 0x4000, 0xE000};
    for (uint32_t i = 0; i < 10; ++i) { ca[i] = ea_[i]; cb[i] = eb_[i]; }
    for (uint32_t i = 10; i < LANES; ++i) {
        ca[i] = (uint32_t)((i * 2129u) & 0xFFFFu);
        cb[i] = (uint32_t)(((i * 1471u) + 7u) & 0xFFFFu);
    }

    uint32_t golden[N_LIMB][LANES];
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        unsigned o[8];
        bp16_prod_to_q256((int)ca[lane], (int)cb[lane], o);  // FULL (incl. sign negate)
        for (uint32_t l = 0; l < N_LIMB; ++l) golden[l][lane] = o[l];
    }

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t tile_bytes = sizeof(uint32_t) * TILE_HW;

    auto mk = [&](uint32_t n) {
        distributed::DeviceLocalBufferConfig lc{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig bc{.size = (size_t)n * tile_bytes};
        return distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    };
    auto a_dram = mk(1), b_dram = mk(1), out_dram = mk(N_LIMB);

    auto make_cb = [&](CBIndex idx, uint32_t n) {
        CircularBufferConfig cfg(n * tile_bytes, {{idx, tt::DataFormat::Int32}});
        cfg.set_page_size(idx, tile_bytes);
        CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, 1);
    make_cb(CBIndex::c_1, 1);
    make_cb(CBIndex::c_16, N_LIMB);

    std::vector<uint32_t> rargs;
    TensorAccessorArgs(*a_dram->get_backing_buffer()).append_to(rargs);
    TensorAccessorArgs(*b_dram->get_backing_buffer()).append_to(rargs);
    auto reader = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_place_sfpu/kernels/dataflow/read_two_codes.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = rargs});

    std::vector<uint32_t> wargs;
    TensorAccessorArgs(*out_dram->get_backing_buffer()).append_to(wargs);
    auto writer = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_place_sfpu/kernels/dataflow/write_limbs8.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = wargs});

    auto compute = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_place_sfpu/kernels/compute/sfpu_bp16_place.cpp", core,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true});

    SetRuntimeArgs(program, reader, core, {(uint32_t)a_dram->address(), (uint32_t)b_dram->address()});
    SetRuntimeArgs(program, writer, core, {(uint32_t)out_dram->address(), N_LIMB});
    SetRuntimeArgs(program, compute, core, {});

    auto at = make_lane_tile(ca), bt = make_lane_tile(cb);
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, at, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, bt, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> out_host(N_LIMB * TILE_HW, 0u);
    distributed::EnqueueReadMeshBuffer(cq, out_host, out_dram, true);

    auto dev = [&](uint32_t limb, uint32_t lane) -> uint32_t {
        return out_host[(size_t)limb * TILE_HW + 0 * TILE_W + lane];
    };

    {
        const int diag_lanes[] = {0, 1, 2, 3, 4, 5, 8, 10};
        for (int lane : diag_lanes) {
            std::cout << "  [diag] lane " << std::setw(2) << lane << " a=0x" << std::hex << std::setw(4)
                      << std::setfill('0') << ca[lane] << " b=0x" << std::setw(4) << std::setfill('0') << cb[lane]
                      << " | limbs dev/gold:";
            for (uint32_t l = 0; l < N_LIMB; ++l)
                std::cout << " " << dev(l, lane) << "/" << golden[l][lane];
            std::cout << std::dec << "\n";
        }
    }

    int bad = 0, first_bad = -1;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        bool ok = true;
        for (uint32_t l = 0; l < N_LIMB; ++l)
            if (dev(l, lane) != golden[l][lane]) ok = false;
        if (!ok) {
            if (bad < 6) {
                std::cout << "  lane " << lane << " a=0x" << std::hex << ca[lane] << " b=0x" << cb[lane] << std::dec << " limb mismatch:";
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
        std::cout << "PASS: SFPU lane-parallel bp16 madd/placement (signed), " << good << "/" << LANES
                  << " lanes bit-exact (8 limbs each)\n";
    else
        std::cout << "FAIL: SFPU lane-parallel bp16 madd/placement (signed), " << good << "/" << LANES
                  << " lanes bit-exact (8 limbs each); first bad lane " << first_bad << "\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
