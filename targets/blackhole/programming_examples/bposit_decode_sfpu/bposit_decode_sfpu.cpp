// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_decode_sfpu — MILESTONE M1 of the SFPU-vectorized exact-quire b-posit matmul:
// a LANE-PARALLEL bp16_decode on a PHYSICAL Tenstorrent Blackhole. 32 bposit-16 codes
// (one per SFPU lane) are decoded simultaneously to (sign, M, E2) and gated BIT-EXACT
// per lane against the scalar oracle bp16_decode (kernel/bp16_decode.h).
//
// Host structure mirrors the proven sibling bposit_quire_sfpu_add.cpp: pick 32 diverse
// codes (incl. the ZERO and NaR specials, both regime-saturation directions, positives,
// and negatives with the high bit set), derive the golden by calling the REAL scalar
// bp16_decode, dispatch the SFPU compute kernel, read back the 3x32 result, gate
// lane-by-lane. The kernel is kernels/compute/sfpu_bp16_decode.cpp.

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

// The scalar oracle — same decode the milestone targets (ES=3, useed=256), proven
// byte-identical x86 == qemu-rv32 == ttsim-BRISC. Suppress unused-function (freestanding).
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
static constexpr uint32_t TILE_HW = TILE_W * TILE_H;  // 1024
static constexpr uint32_t N_OUT = 3;                  // sign, M, E2

// Build a tile where element (r*32 + c) holds lane c's value, replicated over all rows.
static std::vector<uint32_t> make_lane_tile(const uint32_t lane_vals[LANES]) {
    std::vector<uint32_t> tile(TILE_HW);
    for (uint32_t r = 0; r < TILE_H; ++r)
        for (uint32_t c = 0; c < TILE_W; ++c)
            tile[r * TILE_W + c] = lane_vals[c];
    return tile;
}

int main() {
    // ---- 32 diverse bp16 codes (lane 0 = ZERO, lane 1 = NaR, rest span regimes/signs) ----
    uint32_t codes[LANES];
    codes[0] = 0x0000;  // ZERO   -> sign=0,M=0,E2=0
    codes[1] = 0x8000;  // NaR     -> sign=0,M=0,E2=0
    codes[2] = 0x7FFF;  // max positive regime (saturated)
    codes[3] = 0x0001;  // min positive magnitude
    codes[4] = 0x4000;  // +1.0
    for (uint32_t i = 5; i < LANES; ++i) codes[i] = (uint32_t)((i * 2129u) & 0xFFFFu);  // odd stride spreads regimes/signs

    // ---- golden via the REAL scalar bp16_decode -------------------------------
    uint32_t g_sign[LANES], g_M[LANES], g_E2[LANES];
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        int s = 0, e2 = 0;
        unsigned m = 0;
        bp16_decode((int)codes[lane], &s, &m, &e2);
        g_sign[lane] = (uint32_t)s;
        g_M[lane] = (uint32_t)m;
        g_E2[lane] = (uint32_t)e2;  // two's-complement bit pattern of the signed E2
    }

    // ---- device setup (unit mesh, idiom from bposit_quire_sfpu_add.cpp) --------
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
    auto codes_dram = mk(1);
    auto out_dram = mk(N_OUT);

    auto make_cb = [&](CBIndex idx, uint32_t n_tiles) {
        CircularBufferConfig cfg(n_tiles * tile_bytes, {{idx, tt::DataFormat::Int32}});
        cfg.set_page_size(idx, tile_bytes);
        CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, 1);       // CODES
    make_cb(CBIndex::c_16, N_OUT);  // OUT: sign, M, E2

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*codes_dram->get_backing_buffer()).append_to(reader_ct_args);
    auto reader = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_decode_sfpu/kernels/dataflow/read_codes.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                           .compile_args = reader_ct_args});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*out_dram->get_backing_buffer()).append_to(writer_ct_args);
    auto writer = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_decode_sfpu/kernels/dataflow/write_fields.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                           .compile_args = writer_ct_args});

    // INT32 on the SFPU requires fp32_dest_acc_en=true (DEST is a 16-bit file; 32-bit
    // tiles into a 16-bit-configured DEST corrupt — see sfpu_q256_add.cpp root-cause note).
    // ECHO_INPUT=1 repurposes cb_out to echo the loaded code back (verify the SFPU input).
    std::map<std::string, std::string> compute_defines;
    if (const char* echo = std::getenv("ECHO_INPUT")) {
        if (echo[0] != '\0' && echo[0] != '0') compute_defines["ECHO_INPUT"] = echo;
    }
    auto compute = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_decode_sfpu/kernels/compute/sfpu_bp16_decode.cpp", core,
        ComputeConfig{.fp32_dest_acc_en = true, .dst_full_sync_en = true, .defines = compute_defines});

    SetRuntimeArgs(program, reader, core, {(uint32_t)codes_dram->address()});
    SetRuntimeArgs(program, writer, core, {(uint32_t)out_dram->address(), N_OUT});
    SetRuntimeArgs(program, compute, core, {});

    // ---- marshal the single code tile -----------------------------------------
    auto codes_tile = make_lane_tile(codes);
    distributed::EnqueueWriteMeshBuffer(cq, codes_dram, codes_tile, /*blocking=*/false);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<uint32_t> out_host(N_OUT * TILE_HW, 0u);
    distributed::EnqueueReadMeshBuffer(cq, out_host, out_dram, /*blocking=*/true);

    auto dev = [&](uint32_t out_tile, uint32_t lane) -> uint32_t {
        return out_host[(size_t)out_tile * TILE_HW + /*row0*/ 0 * TILE_W + lane];
    };

    // ---- diagnostic: per-lane code/golden/device for a spread of lanes ---------
    {
        const char* echo = std::getenv("ECHO_INPUT");
        const int echo_n = (echo && echo[0] && echo[0] != '0') ? std::atoi(echo) : 0;
        std::cout << "  [diag] mode: " << (echo_n == 1 ? "ECHO_INPUT=1 (cb_out=loaded code)" : "normal (sign,M,E2)") << "\n";
        const int diag_lanes[] = {0, 1, 2, 3, 4, 7, 12, 16};
        for (int lane : diag_lanes) {
            std::cout << "  [diag] lane " << std::setw(2) << lane
                      << " code=0x" << std::hex << std::setw(4) << std::setfill('0') << codes[lane] << std::dec
                      << " | golden sign=" << g_sign[lane] << " M=" << g_M[lane] << " E2=" << (int32_t)g_E2[lane]
                      << " | dev sign=" << dev(0, lane) << " M=" << dev(1, lane) << " E2=" << (int32_t)dev(2, lane)
                      << "\n";
        }
    }

    // ---- gate bit-exact, lane-by-lane -----------------------------------------
    int bad_lanes = 0, first_bad = -1;
    for (uint32_t lane = 0; lane < LANES; ++lane) {
        bool ok = (dev(0, lane) == g_sign[lane]) && (dev(1, lane) == g_M[lane]) && (dev(2, lane) == g_E2[lane]);
        if (!ok) {
            if (bad_lanes < 8)
                std::cout << "  lane " << lane << " code=0x" << std::hex << codes[lane] << std::dec
                          << " dev(sign=" << dev(0, lane) << ",M=" << dev(1, lane) << ",E2=" << (int32_t)dev(2, lane) << ")"
                          << " golden(sign=" << g_sign[lane] << ",M=" << g_M[lane] << ",E2=" << (int32_t)g_E2[lane] << ")\n";
            ++bad_lanes;
            if (first_bad < 0) first_bad = (int)lane;
        }
    }

    const int good = (int)LANES - bad_lanes;
    const bool pass = (bad_lanes == 0);
    if (pass)
        std::cout << "PASS: SFPU lane-parallel bp16_decode, " << good << "/" << LANES
                  << " lanes bit-exact (sign,M,E2)\n";
    else
        std::cout << "FAIL: SFPU lane-parallel bp16_decode, " << good << "/" << LANES
                  << " lanes bit-exact (sign,M,E2); first bad lane " << first_bad << "\n";

    mesh_device->close();
    return pass ? 0 : 1;
}
