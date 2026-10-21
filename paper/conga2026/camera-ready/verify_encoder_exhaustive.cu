// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Exhaustive verification of the closed-form b-posit32 encoder against the
// deployed reference, over EVERY 32-bit integer input.
//
// The reference builds the 31-bit field one bit at a time (a regime run, three
// exponent bits, then up to 28 fraction bits). The closed form computes the
// same field as three shifted masks. Because that is a rewrite rather than a
// tweak, a sampled check is not good enough: this program enumerates all
// 2^32 int32 values at the deployed scale shift, and the full boundary set at
// every scale shift the pipeline can use, comparing on the device so the whole
// sweep costs seconds and no host memory.
//
// Build:
//   nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 \
//        verify_encoder_exhaustive.cu -o verify_encoder_exhaustive
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "cuda %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    std::exit(2); } } while (0)

// ---- reference: the shipped encoder, bit-at-a-time ------------------------
__device__ __forceinline__ std::uint32_t ref_field(std::uint32_t mantissa_bits,
                                                   int total_e) {
    if (total_e > 48)  return 0x7FFFFFFFu;
    if (total_e < -48) return 0x00000001u;
    int k = total_e >> 3;
    int e = total_e & 7;
    std::uint32_t out = 0; int n_bits = 0;
    if (k >= 0) {
        for (int i = 0; i < k + 1 && n_bits < 31; ++i) { out = (out << 1) | 1u; ++n_bits; }
        if (n_bits < 31) { out <<= 1; ++n_bits; }
    } else {
        for (int i = 0; i < -k && n_bits < 31; ++i) { out <<= 1; ++n_bits; }
        if (n_bits < 31) { out = (out << 1) | 1u; ++n_bits; }
    }
    #pragma unroll
    for (int i = 2; i >= 0; --i) {
        if (n_bits >= 31) break;
        out = (out << 1) | ((e >> i) & 1u); ++n_bits;
    }
    int frac_pos = 30;
    while (n_bits < 31 && frac_pos >= 0) {
        out = (out << 1) | ((mantissa_bits >> frac_pos) & 1u); ++n_bits; --frac_pos;
    }
    out <<= (31 - n_bits);
    return out & 0x7FFFFFFFu;
}

// ---- candidate: closed form ----------------------------------------------
__device__ __forceinline__ std::uint32_t cand_field(std::uint32_t mantissa_bits,
                                                    int total_e) {
    if (total_e > 48)  return 0x7FFFFFFFu;
    if (total_e < -48) return 0x00000001u;
    const int k = total_e >> 3;
    const std::uint32_t e = std::uint32_t(total_e & 7);
    std::uint32_t out; int r;
    if (k >= 0) {
        const int ones = k + 1;
        r = ones + 1;
        out = ((1u << ones) - 1u) << (31 - ones);
    } else {
        const int zeros = -k;
        r = zeros + 1;
        out = 1u << (30 - zeros);
    }
    out |= e << (28 - r);
    out |= mantissa_bits >> (r + 3);
    return out & 0x7FFFFFFFu;
}

template <bool USE_CAND>
__device__ __forceinline__ std::uint32_t encode(std::int32_t I, int ss) {
    if (I == 0) return 0u;
    const int sign = (I < 0);
    const std::uint32_t a = sign ? std::uint32_t(-I) : std::uint32_t(I);
    const int msb = 31 - __clz(a);
    const std::uint32_t sh = (msb >= 31) ? a : (a << (31 - msb));
    std::uint32_t f = USE_CAND ? cand_field(sh & 0x7FFFFFFFu, msb - ss)
                               : ref_field(sh & 0x7FFFFFFFu, msb - ss);
    if (sign) f = ((~f) + 1u) & 0x7FFFFFFFu;
    return (std::uint32_t(sign) << 31) | f;
}

// Every input is derived from its index, so the sweep needs no memory at all.
__global__ void sweep(std::uint64_t base, std::uint32_t count, int ss,
                      unsigned long long* bad, std::int32_t* first_bad) {
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const std::int32_t I = std::int32_t(std::uint32_t(base + i));
    if (encode<false>(I, ss) != encode<true>(I, ss)) {
        if (atomicAdd(bad, 1ull) == 0) *first_bad = I;
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp p; CUDA_OK(cudaGetDeviceProperties(&p, 0));
    const int deployed_ss = 12;                 // 2 * BP8_SCALE_BITS
    unsigned long long* dBad; std::int32_t* dFirst;
    CUDA_OK(cudaMalloc(&dBad, 8)); CUDA_OK(cudaMalloc(&dFirst, 4));
    CUDA_OK(cudaMemset(dBad, 0, 8));
    std::printf("device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    const std::uint32_t CHUNK = 1u << 26;       // 64 Mi inputs per launch
    const int tpb = 256;
    std::printf("\nexhaustive sweep, scale_shift = %d (the deployed value):\n",
                deployed_ss);
    for (std::uint64_t base = 0; base < (1ull << 32); base += CHUNK) {
        sweep<<<int((CHUNK + tpb - 1) / tpb), tpb>>>(base, CHUNK, deployed_ss,
                                                     dBad, dFirst);
        CUDA_OK(cudaDeviceSynchronize());
    }
    unsigned long long bad = 0; std::int32_t first = 0;
    CUDA_OK(cudaMemcpy(&bad, dBad, 8, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&first, dFirst, 4, cudaMemcpyDeviceToHost));
    std::printf("  all 4,294,967,296 int32 inputs: %llu mismatches -- %s\n",
                bad, bad ? "FAIL" : "PASS");
    if (bad) std::printf("  first mismatching input: %d\n", first);

    // every other scale shift the pipeline can use, over the whole space
    std::printf("\nevery other scale_shift, 0..24, same exhaustive sweep:\n");
    unsigned long long total = bad;
    for (int ss = 0; ss <= 24; ++ss) {
        if (ss == deployed_ss) continue;
        CUDA_OK(cudaMemset(dBad, 0, 8));
        for (std::uint64_t base = 0; base < (1ull << 32); base += CHUNK) {
            sweep<<<int((CHUNK + tpb - 1) / tpb), tpb>>>(base, CHUNK, ss,
                                                         dBad, dFirst);
            CUDA_OK(cudaDeviceSynchronize());
        }
        unsigned long long b = 0;
        CUDA_OK(cudaMemcpy(&b, dBad, 8, cudaMemcpyDeviceToHost));
        total += b;
        std::printf("  ss=%2d: %llu%s", ss, b, (ss % 5 == 4) ? "\n" : "   ");
    }
    std::printf("\n\nTOTAL over 25 scale shifts x 2^32 inputs "
                "(%.3e comparisons): %llu mismatches -- %s\n",
                25.0 * 4294967296.0, total, total ? "FAIL" : "PASS");
    return total ? 1 : 0;
}
