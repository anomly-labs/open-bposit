// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bposit_bf16_codec — run the bf16<->bposit16-code round-trip codec (bp16_bf16.h) on a real
// Blackhole baby core and confirm it is BYTE-IDENTICAL to the same codec on x86. This is the
// on-silicon validation (step 1) for the host-round-trip-free exact-quire op path (task #18):
// if device == host here, the RV32 reader/writer encode/decode will match the host reference.
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../../kernel/bp16_bf16.h"
#pragma GCC diagnostic pop

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    const uint32_t N = 1024;
    const uint32_t nbytes = ((N * 2 + 63) / 64) * 64;  // pad to 64B page

    // Random in-range bf16 values (exp in [127-40, 127+40], skip zero/denorm/inf/nan).
    std::vector<uint16_t> in(N);
    uint32_t seed = 12345;
    for (uint32_t i = 0; i < N; ++i) {
        seed = seed * 1664525u + 1013904223u;
        uint32_t sign = (seed >> 31) & 1u;
        uint32_t exp = 87u + ((seed >> 8) % 81u);   // 87..167 -> unbiased -40..40
        uint32_t mant = (seed >> 3) & 0x7Fu;
        in[i] = (uint16_t)((sign << 15) | (exp << 7) | mant);
    }
    // Host reference: same codec on x86.
    std::vector<uint16_t> expected(N);
    for (uint32_t i = 0; i < N; ++i) {
        expected[i] = (uint16_t)bp16_code_to_bf16(bf16_to_bp16_code((unsigned)in[i]));
    }
    // pad input buffer to nbytes/2 uint16
    std::vector<uint32_t> in_words(nbytes / 4, 0);
    std::memcpy(in_words.data(), in.data(), N * 2);

    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh->mesh_command_queue();
    distributed::MeshWorkload wl;
    auto drange = distributed::MeshCoordinateRange(mesh->shape());
    auto program = CreateProgram();

    auto mk = [&](size_t bytes, BufferType t) {
        distributed::DeviceLocalBufferConfig lc{.page_size = bytes, .buffer_type = t};
        distributed::ReplicatedBufferConfig bc{.size = bytes};
        return distributed::MeshBuffer::create(bc, lc, mesh.get());
    };
    auto in_dram = mk(nbytes, BufferType::DRAM);
    auto out_dram = mk(nbytes, BufferType::DRAM);
    auto in_l1 = mk(nbytes, BufferType::L1);
    auto out_l1 = mk(nbytes, BufferType::L1);

    CoreCoord core = {0, 0};
    auto kernel = CreateKernel(
        program, OVERRIDE_KERNEL_PREFIX "bposit_bf16_codec/kernels/bf16_codec_baby.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, core,
                   {(uint32_t)in_dram->address(), (uint32_t)out_dram->address(),
                    (uint32_t)in_l1->address(), (uint32_t)out_l1->address(), N, nbytes});

    distributed::EnqueueWriteMeshBuffer(cq, in_dram, in_words, false);
    wl.add_program(drange, std::move(program));
    distributed::EnqueueMeshWorkload(cq, wl, false);
    std::vector<uint32_t> out_words(nbytes / 4, 0);
    distributed::EnqueueReadMeshBuffer(cq, out_words, out_dram, true);

    std::vector<uint16_t> dev(N);
    std::memcpy(dev.data(), out_words.data(), N * 2);
    int bad = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (dev[i] != expected[i]) {
            if (++bad <= 8)
                std::cout << "  mismatch i=" << i << " in=0x" << std::hex << in[i]
                          << " dev=0x" << dev[i] << " host=0x" << expected[i] << std::dec << "\n";
        }
    }
    std::cout << "[bf16-codec] device vs host codec: " << (N - bad) << "/" << N << " match\n";
    std::cout << (bad == 0 ? "PASS" : "FAIL")
              << ": bf16<->bp16 codec byte-identical x86==Blackhole RV32\n";
    mesh->close();
    return bad == 0 ? 0 : 1;
}
