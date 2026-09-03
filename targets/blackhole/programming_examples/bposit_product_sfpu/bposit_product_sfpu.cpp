// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_product_sfpu — MILESTONE M2 of the SFPU-vectorized exact-quire b-posit matmul:
// a LANE-PARALLEL bp16 PRODUCT on a PHYSICAL Tenstorrent Blackhole. Per lane, two
// bposit-16 codes (a, b) are decoded and combined into the dyadic product
//   sign_p = sa^sb,  M_p = Ma*Mb,  E2_p = Ea+Eb
// gated BIT-EXACT per lane against the scalar oracle (bp16_decode then xor/mul/add).
// Mirrors bposit_decode_sfpu's host; kernel kernels/compute/sfpu_bp16_product.cpp.

#include <fmt/ostream.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/constants.hpp>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../../kernel/bp16_decode.h"
#pragma GCC diagnostic pop

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static constexpr uint32_t LANES = 32;
static constexpr uint32_t TILE_W = 32;
static constexpr uint32_t TILE_H = 32;
static constexpr uint32_t TILE_HW = TILE_W * TILE_H;
static constexpr uint32_t N_OUT = 3;

static std::vector<uint32_t> make_lane_tile(const uint32_t lane_vals[LANES]) {
    std::vector<uint32_t> tile(TILE_HW);
    for (uint32_t r = 0; r < TILE_H; ++r)
        for (uint32_t c = 0; c < TILE_W; ++c)
            tile[r * TILE_W + c] = lane_vals[c];
    return tile;
}

int main() {
    // ---- 32 (a,b) code pairs: edges in low lanes, odd-stride spread above --------
    uint32_t ca[LANES], cb[LANES];
    // Edge/special pairs (incl. ZERO and NaR operands, +1.0, max/min regimes, negatives).
    const uint32_t ea_[8] = {0x0000, 0x8000, 0x4000, 0x4000, 0x7FFF, 0x0001, 0xC000, 0x3A37};
    const uint32_t eb_[8] = {0x4000, 0x4000, 0x4000, 0xC000, 0x0001, 0x7FFF, 0xC000, 0x8510};
    for (uint32_t i = 0; i < 8; ++i) { ca[i] = ea_[i]; cb[i] = eb_[i]; }
    for (uint32_t i = 8; i < LANES; ++i) {
        ca[i] = (uint32_t)((i * 2129u) & 0xFFFFu);
        cb[i] = (uint32_t)(((i * 1471u) + 7u) & 0xFFFFu);
    }

    // ---- golden product via the REAL scalar bp16_decode --------------------------
    uint32_t g_sign[LANES], g_M[LANES], g_E2[LANES];
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        int sa = 0, ea = 0, sb = 0, eb = 0;
        unsigned Ma = 0, Mb = 0;
        bp16_decode((int)ca[lane], &sa, &Ma, &ea);
        bp16_decode((int)cb[lane], &sb, &Mb, &eb);
        g_sign[lane] = (uint32_t)(sa ^ sb);
        g_M[lane] = (uint32_t)(Ma * Mb);
        g_E2[lane] = (uint32_t)(ea + eb);
    }

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    constexpr uint32_t tile_bytes = sizeof(uint32_t) * TILE_HW;

    auto mk = [&](uint32_t n_tiles) {
        distributed::DeviceLocalBufferConfig lc{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig bc{.size = (size_t)n_tiles * tile_bytes};
        return distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    };
    auto a_dram = mk(1);
    auto b_dram = mk(1);
    auto out_dram = mk(N_OUT);

    auto make_cb = [&](CBIndex idx, uint32_t n_tiles) {
        CircularBufferConfig cfg(n_tiles * tile_bytes, {{idx, tt::DataFormat::Int32}});
        cfg.set_page_size(idx, tile_bytes);
        CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, 1);       // CODE_A
    make_cb(CBIndex::c_1, 1);       // CODE_B
    make_cb(CBIndex::c_16, N_OUT);  // OUT: sign_p, M_p, E2_p

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*a_dram->get_backing_buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*b_dram->get_backing_buffer()).append_to(reader_ct_args);
    auto reader = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_product_sfpu/kernels/dataflow/read_two_codes.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                           .compile_args = reader_ct_args});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*out_dram->get_backing_buffer()).append_to(writer_ct_args);
    auto writer = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_product_sfpu/kernels/dataflow/write_fields.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                           .compile_args = writer_ct_args});

    auto compute = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_product_sfpu/kernels/compute/sfpu_bp16_product.cpp", core,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true});

    SetRuntimeArgs(program, reader, core, {(uint32_t)a_dram->address(), (uint32_t)b_dram->address()});
    SetRuntimeArgs(program, writer, core, {(uint32_t)out_dram->address(), N_OUT});
    SetRuntimeArgs(program, compute, core, {});

    auto a_tile = make_lane_tile(ca);
    auto b_tile = make_lane_tile(cb);
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_tile, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_tile, /*blocking=*/false);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<uint32_t> out_host(N_OUT * TILE_HW, 0u);
    distributed::EnqueueReadMeshBuffer(cq, out_host, out_dram, /*blocking=*/true);

    auto dev = [&](uint32_t out_tile, uint32_t lane) -> uint32_t {
        return out_host[(size_t)out_tile * TILE_HW + 0 * TILE_W + lane];
    };

    {
        const int diag_lanes[] = {0, 1, 2, 3, 4, 5, 6, 7, 16};
        for (int lane : diag_lanes) {
            std::cout << "  [diag] lane " << std::setw(2) << lane
                      << " a=0x" << std::hex << std::setw(4) << std::setfill('0') << ca[lane]
                      << " b=0x" << std::setw(4) << std::setfill('0') << cb[lane] << std::dec
                      << " | golden sign=" << g_sign[lane] << " M=" << g_M[lane] << " E2=" << (int32_t)g_E2[lane]
                      << " | dev sign=" << dev(0, lane) << " M=" << dev(1, lane) << " E2=" << (int32_t)dev(2, lane)
                      << "\n";
        }
    }

    int bad_lanes = 0, first_bad = -1;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        bool ok = (dev(0, lane) == g_sign[lane]) && (dev(1, lane) == g_M[lane]) && (dev(2, lane) == g_E2[lane]);
        if (!ok) {
            if (bad_lanes < 8)
                std::cout << "  lane " << lane << " a=0x" << std::hex << ca[lane] << " b=0x" << cb[lane] << std::dec
                          << " dev(sign=" << dev(0, lane) << ",M=" << dev(1, lane) << ",E2=" << (int32_t)dev(2, lane) << ")"
                          << " golden(sign=" << g_sign[lane] << ",M=" << g_M[lane] << ",E2=" << (int32_t)g_E2[lane] << ")\n";
            ++bad_lanes;
            if (first_bad < 0) first_bad = (int)lane;
        }
    }

    const int good = (int)LANES - bad_lanes;
    const bool pass = (bad_lanes == 0);
    if (pass)
        std::cout << "PASS: SFPU lane-parallel bp16 product, " << good << "/" << LANES
                  << " lanes bit-exact (sign,M,E2)\n";
    else
        std::cout << "FAIL: SFPU lane-parallel bp16 product, " << good << "/" << LANES
                  << " lanes bit-exact (sign,M,E2); first bad lane " << first_bad << "\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
