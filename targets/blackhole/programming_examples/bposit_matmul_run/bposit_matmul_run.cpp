// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_matmul_run — file-IO driver for the SFPU exact-quire bp16 matmul, so it is callable
// from Python (subprocess). Reads bp16 CODES from files, runs the fully-arbitrary-(M,N,K) SFPU
// matmul (reusing the bposit_matmul_full_sfpu kernels VERBATIM), and writes the 256-bit quire
// per output element to a file. NO internal golden/gate — the Python caller verifies.
//
//   argv: M  K  a_codes.bin  b_codes.bin  q_out.bin
//   a_codes.bin : M*K int32 bp16 codes (row-major X[m][k])
//   b_codes.bin : K*N int32 bp16 codes (row-major W[k][n]), N = 32
//   q_out.bin   : M*N*8 uint32 (row-major q[m][n][limb]) — the exact 256-bit quire per element

#include <fmt/ostream.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

static constexpr uint32_t LANES = 32;  // N
static constexpr uint32_t TILE_W = 32, TILE_H = 32, TILE_HW = TILE_W * TILE_H;
static constexpr uint32_t N_LIMB = 8;
static constexpr uint32_t STREAM_DEPTH = 4;

static std::vector<uint32_t> read_u32(const char* path, size_t n) {
    std::vector<uint32_t> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(uint32_t)));
    if (!f) { std::cerr << "short read " << path << "\n"; std::exit(2); }
    return v;
}

int main(int argc, char** argv) {
    if (argc != 6) { std::cerr << "usage: bposit_matmul_run M K a_codes.bin b_codes.bin q_out.bin\n"; return 2; }
    const uint32_t M = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    const uint32_t K = (uint32_t)std::strtoul(argv[2], nullptr, 10);
    auto Xcodes = read_u32(argv[3], (size_t)M * K);   // X[m][k]
    auto Wcodes = read_u32(argv[4], (size_t)K * LANES); // W[k][n]

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();
    auto grid = mesh_device->compute_with_storage_grid_size();
    constexpr uint32_t tile_bytes = sizeof(uint32_t) * TILE_HW;
    auto mk = [&](uint32_t n) {
        distributed::DeviceLocalBufferConfig lc{.page_size = tile_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig bc{.size = (size_t)n * tile_bytes};
        return distributed::MeshBuffer::create(bc, lc, mesh_device.get());
    };
    auto a_dram = mk(M * K), b_dram = mk(K), q_dram = mk(M * N_LIMB), zero_dram = mk(1);

    // a_dram page (m*K+k) = X[m][k] broadcast across the 32x32 tile; b_dram page k = W[k][0..31] per lane.
    std::vector<uint32_t> a_host((size_t)M * K * TILE_HW), b_host((size_t)K * TILE_HW), zero_host(TILE_HW, 0u);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t v = Xcodes[(size_t)m * K + k];
            uint32_t* t = &a_host[(size_t)(m * K + k) * TILE_HW];
            for (uint32_t i = 0; i < TILE_HW; ++i) t[i] = v;
        }
    for (uint32_t k = 0; k < K; ++k) {
        uint32_t* t = &b_host[(size_t)k * TILE_HW];
        for (uint32_t r = 0; r < TILE_H; ++r)
            for (uint32_t c = 0; c < TILE_W; ++c) t[r * TILE_W + c] = Wcodes[(size_t)k * LANES + c];
    }
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, a_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, b_host, false);
    distributed::EnqueueWriteMeshBuffer(cq, zero_dram, zero_host, true);

    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    auto program = CreateProgram();
    auto [num_cores, all_cores, cg1, cg2, opc1, opc2] = tt::tt_metal::split_work_to_cores(grid, M, true);

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
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> out_host((size_t)M * N_LIMB * TILE_HW, 0u);
    distributed::EnqueueReadMeshBuffer(cq, out_host, q_dram, true);

    // q_out[m][n][l] = out_host[(m*8+l)*TILE_HW + n]
    std::vector<uint32_t> q_out((size_t)M * LANES * N_LIMB);
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < LANES; ++n)
            for (uint32_t l = 0; l < N_LIMB; ++l)
                q_out[((size_t)m * LANES + n) * N_LIMB + l] = out_host[(size_t)(m * N_LIMB + l) * TILE_HW + n];

    std::ofstream of(argv[5], std::ios::binary);
    of.write(reinterpret_cast<const char*>(q_out.data()), (std::streamsize)(q_out.size() * sizeof(uint32_t)));
    of.close();
    std::cout << "OK: SFPU exact-quire matmul M=" << M << " K=" << K << " N=" << LANES << " across "
              << num_cores << " cores -> " << argv[5] << " (" << (M * LANES) << " quires)\n";

    mesh_device->close();
    return 0;
}
