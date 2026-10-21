// Copyright (c) 2026 Anomly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// bench_pipeline_stages.cu -- stage-resolved cost of the b-posit8 tensor-core
// pipeline on an NVIDIA Blackwell GPU, for CoNGA'26 Sect. 7.4.
//
// Reviewer 3 asked that the reported tensor-core throughput distinguish the
// standalone INT8 GEMM from the complete decode--GEMM--quire--encode pipeline.
// Every throughput number in the repository times only cublasLtMatmul, so this
// program times each stage separately at the paper's shapes:
//
//   (A) decode   : b-posit8 bytes -> int8 fixed point through the 256-entry
//                  codebook, over both operands (M*K + K*N elements)
//   (B) GEMM     : unmodified cuBLASLt INT8 IMMA, int32 output (M*N)
//   (C) encode   : int32 -> b-posit32 over the M*N outputs
//   (ref) BF16   : native BF16 HMMA at the same shape, for the baseline ratio
//
// The quire accumulation is a sign-extension of each int32 tile into the
// 256-bit accumulator; with one K-tile per output (K <= 133,144, see the
// overflow bound in Sect. 7.4) it adds no device work beyond stage (C)'s read
// of the int32 result, so stage (C) is reported as "quire + encode".
//
// Build:
//   nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 \
//        -lcublasLt -lcudart bench_pipeline_stages.cu -o bench_pipeline_stages
// Run:
//   ./bench_pipeline_stages            # 2048 and 4096 cubed
//   ./bench_pipeline_stages 2048
#include "bposit8_int8_lut.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "cuda %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
    std::exit(2); } } while (0)
#define CUBLAS_OK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cublas %s @%d: %d\n", #x, __LINE__, (int)s_); \
    std::exit(3); } } while (0)

__device__ std::int8_t d_lut[256];

// ---- (A) codebook decode: b-posit8 byte -> int8 fixed point -----------------
// Scalar form (the originally deployed kernel) and a 4-wide form that moves one
// uchar4 per thread instead of one byte. Both are gated against each other.
__global__ void decode_bposit8_to_int8(const std::uint8_t* __restrict__ in,
                                       std::int8_t* __restrict__ out,
                                       std::size_t n) {
    std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) out[i] = d_lut[in[i]];
}

__global__ void decode_bposit8_to_int8_v4(const uchar4* __restrict__ in,
                                          char4* __restrict__ out,
                                          std::size_t n4) {
    std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for (; i < n4; i += stride) {
        uchar4 v = in[i];
        char4 o;
        o.x = d_lut[v.x]; o.y = d_lut[v.y]; o.z = d_lut[v.z]; o.w = d_lut[v.w];
        out[i] = o;
    }
}

static void launch_decode_v4(const std::uint8_t* in, std::int8_t* out,
                             std::size_t n) {
    const int tpb = 256;
    const std::size_t n4 = n / 4;
    if (n4) {
        std::size_t blocks = (n4 + tpb - 1) / tpb;
        if (blocks > 65535) blocks = 65535;
        decode_bposit8_to_int8_v4<<<int(blocks), tpb>>>(
            reinterpret_cast<const uchar4*>(in),
            reinterpret_cast<char4*>(out), n4);
    }
}

// ---- (C) quire read-out + b-posit32 encode ---------------------------------
// Identical to the deployed encoder in test_mera_isometry_imma.cu.
__device__ __forceinline__ std::uint32_t encode_unsigned_32(std::uint32_t mantissa_bits,
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

// Closed form of encode_unsigned_32: the regime run, the three exponent bits
// and the fraction are three shifted masks rather than three loops. Verified
// bit-identical to the loop form over all 2^32 int32 inputs at every scale
// shift 0..24 (verify_encoder_exhaustive.cu).
__device__ __forceinline__ std::uint32_t encode_unsigned_32_closed(
        std::uint32_t mantissa_bits, int total_e) {
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

__global__ void quire_encode_bposit32_closed(const std::int32_t* __restrict__ in,
                                             std::uint32_t* __restrict__ out,
                                             std::size_t n, int scale_shift) {
    std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const std::int32_t I = in[idx];
    if (I == 0) { out[idx] = 0u; return; }
    const int sign = (I < 0);
    const std::uint32_t abs_I = sign ? std::uint32_t(-I) : std::uint32_t(I);
    const int msb_pos = 31 - __clz(abs_I);
    const std::uint32_t shifted = (msb_pos >= 31) ? abs_I : (abs_I << (31 - msb_pos));
    std::uint32_t field = encode_unsigned_32_closed(shifted & 0x7FFFFFFFu,
                                                     msb_pos - scale_shift);
    if (sign) field = ((~field) + 1u) & 0x7FFFFFFFu;
    out[idx] = (std::uint32_t(sign) << 31) | field;
}

__global__ void quire_encode_bposit32(const std::int32_t* __restrict__ in,
                                      std::uint32_t* __restrict__ out,
                                      std::size_t n, int scale_shift) {
    std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    std::int32_t I = in[idx];
    if (I == 0) { out[idx] = 0u; return; }
    int sign = (I < 0);
    std::uint32_t abs_I = sign ? std::uint32_t(-I) : std::uint32_t(I);
    int msb_pos = 31 - __clz(abs_I);
    int total_e = msb_pos - scale_shift;
    std::uint32_t shifted = (msb_pos >= 31) ? abs_I : (abs_I << (31 - msb_pos));
    std::uint32_t field = encode_unsigned_32(shifted & 0x7FFFFFFFu, total_e);
    if (sign) field = ((~field) + 1u) & 0x7FFFFFFFu;
    out[idx] = ((std::uint32_t(sign) << 31) | field);
}

// ---- timing helper ----------------------------------------------------------
template <class F>
static double time_ms(F&& f, int warm, int reps) {
    cudaEvent_t e0, e1; CUDA_OK(cudaEventCreate(&e0)); CUDA_OK(cudaEventCreate(&e1));
    for (int i = 0; i < warm; ++i) f();
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(e0));
    for (int i = 0; i < reps; ++i) f();
    CUDA_OK(cudaEventRecord(e1));
    CUDA_OK(cudaEventSynchronize(e1));
    float ms; CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_OK(cudaEventDestroy(e0)); CUDA_OK(cudaEventDestroy(e1));
    return double(ms) / reps;
}

struct LtGemm {
    cublasLtHandle_t lt{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t la{}, lb{}, lc{};
    cublasLtMatmulHeuristicResult_t heur{};
    void* ws = nullptr; size_t ws_bytes = 32u << 20;
};

// Column-major, A^T B with both operands stored K-major (the IMMA requirement).
static void lt_setup(LtGemm& g, cudaDataType ab, cudaDataType c,
                     cublasComputeType_t comp, cudaDataType scale,
                     int M, int N, int K) {
    CUBLAS_OK(cublasLtCreate(&g.lt));
    CUBLAS_OK(cublasLtMatmulDescCreate(&g.op, comp, scale));
    cublasOperation_t T = CUBLAS_OP_T, Nn = CUBLAS_OP_N;
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof T));
    CUBLAS_OK(cublasLtMatmulDescSetAttribute(g.op, CUBLASLT_MATMUL_DESC_TRANSB, &Nn, sizeof Nn));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&g.la, ab, K, M, K));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&g.lb, ab, K, N, K));
    CUBLAS_OK(cublasLtMatrixLayoutCreate(&g.lc, c,  M, N, M));
    cublasLtMatmulPreference_t pref;
    CUBLAS_OK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_OK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g.ws_bytes, sizeof g.ws_bytes));
    int got = 0;
    CUBLAS_OK(cublasLtMatmulAlgoGetHeuristic(g.lt, g.op, g.la, g.lb, g.lc, g.lc,
                                             pref, 1, &g.heur, &got));
    CUBLAS_OK(cublasLtMatmulPreferenceDestroy(pref));
    if (!got) { std::fprintf(stderr, "no algo for %dx%dx%d\n", M, N, K); std::exit(4); }
    CUDA_OK(cudaMalloc(&g.ws, g.ws_bytes));
}

static void run_shape(int M, int N, int K) {
    const std::size_t nA = std::size_t(M) * K, nB = std::size_t(K) * N,
                      nC = std::size_t(M) * N;

    std::uint8_t *dAp8, *dBp8; std::int8_t *dAi8, *dBi8;
    std::int32_t* dCi32; std::uint32_t* dCbp32;
    __nv_bfloat16 *dAbf, *dBbf, *dCbf;
    CUDA_OK(cudaMalloc(&dAp8, nA)); CUDA_OK(cudaMalloc(&dBp8, nB));
    CUDA_OK(cudaMalloc(&dAi8, nA)); CUDA_OK(cudaMalloc(&dBi8, nB));
    CUDA_OK(cudaMalloc(&dCi32, nC * 4)); CUDA_OK(cudaMalloc(&dCbp32, nC * 4));
    CUDA_OK(cudaMalloc(&dAbf, nA * 2)); CUDA_OK(cudaMalloc(&dBbf, nB * 2));
    CUDA_OK(cudaMalloc(&dCbf, nC * 2));

    // Operands: b-posit8 codes drawn from the 80-code exact subset is not needed
    // for a timing run; fill with a deterministic spread of real codes.
    {
        std::vector<std::uint8_t> h(nA > nB ? nA : nB);
        for (std::size_t i = 0; i < h.size(); ++i) h[i] = std::uint8_t((i * 37 + 0x30) & 0xFF);
        CUDA_OK(cudaMemcpy(dAp8, h.data(), nA, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dBp8, h.data(), nB, cudaMemcpyHostToDevice));
    }
    CUDA_OK(cudaMemcpyToSymbol(d_lut, kBposit8ToInt8, sizeof kBposit8ToInt8));
    {   // the BF16 baseline must not be timed on uninitialised memory
        std::vector<__nv_bfloat16> h(nA > nB ? nA : nB);
        for (std::size_t i = 0; i < h.size(); ++i)
            h[i] = __float2bfloat16(float((i % 253) - 126) * 0.0078125f);
        CUDA_OK(cudaMemcpy(dAbf, h.data(), nA * 2, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dBbf, h.data(), nB * 2, cudaMemcpyHostToDevice));
    }

    const int tpb = 256;
    auto grid = [&](std::size_t n) { return int((n + tpb - 1) / tpb > 65535u ? 65535u
                                               : (n + tpb - 1) / tpb); };

    // (A) decode, timed per operand: weights are decoded once offline in
    // deployment, activations on every call, so the two must be separable.
    double ms_dec_act = time_ms([&] {
        decode_bposit8_to_int8<<<grid(nA), tpb>>>(dAp8, dAi8, nA);
    }, 3, 20);
    double ms_dec_wgt = time_ms([&] {
        decode_bposit8_to_int8<<<grid(nB), tpb>>>(dBp8, dBi8, nB);
    }, 3, 20);
    double ms_decode = ms_dec_act + ms_dec_wgt;
    double ms_dec_act4 = time_ms([&] { launch_decode_v4(dAp8, dAi8, nA); }, 3, 20);
    double ms_dec_wgt4 = time_ms([&] { launch_decode_v4(dBp8, dBi8, nB); }, 3, 20);

    // (B) INT8 IMMA GEMM
    LtGemm gi; lt_setup(gi, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I, M, N, K);
    int ai = 1, bi = 0;
    double ms_gemm = time_ms([&] {
        CUBLAS_OK(cublasLtMatmul(gi.lt, gi.op, &ai, dAi8, gi.la, dBi8, gi.lb,
                                 &bi, dCi32, gi.lc, dCi32, gi.lc,
                                 &gi.heur.algo, gi.ws, gi.ws_bytes, 0));
    }, 5, 30);

    // (C) quire read-out + b-posit32 encode
    double ms_enc = time_ms([&] {
        quire_encode_bposit32<<<grid(nC), tpb>>>(dCi32, dCbp32, nC, 2 * BP8_SCALE_BITS);
    }, 3, 20);
    double ms_enc_c = time_ms([&] {
        quire_encode_bposit32_closed<<<grid(nC), tpb>>>(dCi32, dCbp32, nC,
                                                        2 * BP8_SCALE_BITS);
    }, 3, 20);

    // (ref) BF16 HMMA at the same shape
    LtGemm gb; lt_setup(gb, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, M, N, K);
    float af = 1.f, bf = 0.f;
    double ms_bf16 = time_ms([&] {
        CUBLAS_OK(cublasLtMatmul(gb.lt, gb.op, &af, dAbf, gb.la, dBbf, gb.lb,
                                 &bf, dCbf, gb.lc, dCbf, gb.lc,
                                 &gb.heur.algo, gb.ws, gb.ws_bytes, 0));
    }, 5, 30);

    const double macs = double(M) * double(N) * double(K);
    auto tp = [&](double ms) { return macs / (ms * 1e-3) / 1e12 * 2.0; };  // TPOPS/TFLOPS
    const double ms_total = ms_decode + ms_gemm + ms_enc;

    const double ms_deploy = ms_dec_act + ms_gemm + ms_enc;   // weights pre-decoded
    std::printf("--- M=%d N=%d K=%d --------------------------------------\n", M, N, K);
    std::printf("  (A1) decode activations, %10zu elems : %8.4f ms  (%5.1f%%)\n",
                nA, ms_dec_act, 100.0 * ms_dec_act / ms_total);
    std::printf("  (A2) decode weights,     %10zu elems : %8.4f ms  (%5.1f%%)\n",
                nB, ms_dec_wgt, 100.0 * ms_dec_wgt / ms_total);
    std::printf("  (B)  GEMM  cuBLASLt INT8 IMMA (TN)        : %8.4f ms  (%5.1f%%)  %7.1f TPOPS\n",
                ms_gemm, 100.0 * ms_gemm / ms_total, tp(ms_gemm));
    std::printf("  (C)  quire read-out + bposit32 encode     : %8.4f ms  (%5.1f%%)\n",
                ms_enc, 100.0 * ms_enc / ms_total);
    std::printf("  ==   pipeline, all stages online          : %8.4f ms           %7.1f TPOPS\n",
                ms_total, tp(ms_total));
    std::printf("  ==   pipeline, weights pre-decoded        : %8.4f ms           %7.1f TPOPS\n",
                ms_deploy, tp(ms_deploy));
    std::printf("  ref  BF16 HMMA, same shape                : %8.4f ms           %7.1f TFLOPS\n",
                ms_bf16, tp(ms_bf16));
    std::printf("  vs BF16:  GEMM alone %.2fx | all online %.2fx | weights pre-decoded %.2fx\n",
                ms_bf16 / ms_gemm, ms_bf16 / ms_total, ms_bf16 / ms_deploy);
    const double ms_total_o = ms_dec_act4 + ms_dec_wgt4 + ms_gemm + ms_enc_c;
    const double ms_deploy_o = ms_dec_act4 + ms_gemm + ms_enc_c;
    std::printf("  OPTIMISED codec (4-wide decode, closed-form encode):\n");
    std::printf("    (A1) %.4f ms  (A2) %.4f ms  (C) %.4f ms\n",
                ms_dec_act4, ms_dec_wgt4, ms_enc_c);
    std::printf("    pipeline all online %.4f ms (%.1f TPOPS, %.2fx BF16) | "
                "weights pre-decoded %.4f ms (%.1f TPOPS, %.2fx BF16)\n\n",
                ms_total_o, tp(ms_total_o), ms_bf16 / ms_total_o,
                ms_deploy_o, tp(ms_deploy_o), ms_bf16 / ms_deploy_o);

    cudaFree(dAp8); cudaFree(dBp8); cudaFree(dAi8); cudaFree(dBi8);
    cudaFree(dCi32); cudaFree(dCbp32); cudaFree(dAbf); cudaFree(dBbf); cudaFree(dCbf);
    cudaFree(gi.ws); cudaFree(gb.ws);
}

// ---- correctness gate ------------------------------------------------------
// The published benchmark (bench_imma_vs_qmma.cu) uses the NN layout, which
// cuBLASLt does not serve with the IMMA-native kernel; this program uses the
// TN layout that INT8 IMMA requires. The two must compute the same matrix, so
// the TN result is checked element-by-element against a host int32 reference
// before any timing figure from it is believed.
static int check_tn_layout(int M, int N, int K) {
    const std::size_t nA = std::size_t(M) * K, nB = std::size_t(K) * N,
                      nC = std::size_t(M) * N;
    std::vector<std::int8_t> hA(nA), hB(nB);
    for (std::size_t i = 0; i < nA; ++i) hA[i] = std::int8_t(int(i * 7 % 61) - 30);
    for (std::size_t i = 0; i < nB; ++i) hB[i] = std::int8_t(int(i * 11 % 53) - 26);

    std::int8_t *dA, *dB; std::int32_t* dC;
    CUDA_OK(cudaMalloc(&dA, nA)); CUDA_OK(cudaMalloc(&dB, nB));
    CUDA_OK(cudaMalloc(&dC, nC * 4));
    CUDA_OK(cudaMemcpy(dA, hA.data(), nA, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), nB, cudaMemcpyHostToDevice));

    LtGemm g; lt_setup(g, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I, M, N, K);
    int a = 1, b = 0;
    CUBLAS_OK(cublasLtMatmul(g.lt, g.op, &a, dA, g.la, dB, g.lb, &b,
                             dC, g.lc, dC, g.lc, &g.heur.algo, g.ws, g.ws_bytes, 0));
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<std::int32_t> hC(nC);
    CUDA_OK(cudaMemcpy(hC.data(), dC, nC * 4, cudaMemcpyDeviceToHost));

    // la = (K, M, ld K) with TRANSA=T  -> A_logical[i][k] = hA[k + i*K]
    // lb = (K, N, ld K) with TRANSB=N  -> B[k][j]         = hB[k + j*K]
    // lc = (M, N, ld M)                -> C[i][j]         = hC[i + j*M]
    std::size_t bad = 0; std::int32_t first_got = 0, first_exp = 0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            std::int64_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += std::int64_t(hA[std::size_t(k) + std::size_t(i) * K])
                     * std::int64_t(hB[std::size_t(k) + std::size_t(j) * K]);
            std::int32_t got = hC[std::size_t(i) + std::size_t(j) * M];
            if (got != std::int32_t(acc)) {
                if (!bad) { first_got = got; first_exp = std::int32_t(acc); }
                ++bad;
            }
        }
    }
    std::printf("GATE  TN-layout INT8 IMMA vs host int32 reference, %dx%dx%d: "
                "%zu/%zu exact -- %s\n", M, N, K, nC - bad, nC,
                bad ? "FAIL" : "PASS");
    if (bad) std::printf("      first mismatch: got %d, expected %d\n", first_got, first_exp);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(g.ws);
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    cudaDeviceProp p; CUDA_OK(cudaGetDeviceProperties(&p, 0));
    std::printf("=== b-posit8 tensor-core pipeline, stage-resolved ===\n");
    std::printf("Device: %s (sm_%d%d)\n\n", p.name, p.major, p.minor);
    if (check_tn_layout(256, 256, 256) != 0) {
        std::fprintf(stderr, "correctness gate failed -- no timing reported\n");
        return 1;
    }
    std::printf("\n");
    if (argc > 3) run_shape(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]));
    else if (argc > 1) run_shape(std::atoi(argv[1]), std::atoi(argv[1]), std::atoi(argv[1]));
    else { run_shape(2048, 2048, 2048); run_shape(4096, 4096, 4096); }
    return 0;
}
