// Standalone synthetic recurrence probe; link the same Axiom dependencies as
// axiom_qwen38_gdn_temporal_microbench.cu. Uses the implementation in gdn.cu.
// Full unrolling makes state indices compile-time constants, but register
// residency is compiler/architecture dependent: inspect ptxas/SASS for spills.
#include "../src/axiom_qwen38_gdn.cu"
#include <cstring>
#include <exception>
#include <stdexcept>

namespace {
uint32_t bench_seed_from_env() {
    const char *value = std::getenv("AXIOM_GDN_MICRO_SEED");
    if (!value) return 12345u;
    const char *message = "AXIOM_GDN_MICRO_SEED must be a decimal uint32 (0..4294967295)";
    if (!*value) throw std::runtime_error(message);
    uint32_t seed = 0u;
    for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9') throw std::runtime_error(message);
        const uint32_t digit = static_cast<uint32_t>(*p - '0');
        if (seed > (std::numeric_limits<uint32_t>::max() - digit) / 10u)
            throw std::runtime_error(message);
        seed = seed * 10u + digit;
    }
    return seed;
}

void bench_check(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    throw std::runtime_error(operation);
}
#define REGISTER_BENCH_CUDA(call) bench_check((call), #call)

float bench_sample(size_t index, uint32_t salt) {
    uint32_t x = static_cast<uint32_t>(index) + salt;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (static_cast<int>(x & 2047u) - 1024) * (1.0f / 1024.0f);
}

template <uint32_t ValueTile>
void report_register_resources() {
    cudaFuncAttributes attributes{};
    REGISTER_BENCH_CUDA(cudaFuncGetAttributes(&attributes, register_recurrence_kernel<ValueTile>));
    std::printf("register tile=%u threads=32 registers_per_thread=%d "
                "local_bytes_per_thread=%zu shared_bytes=%zu "
                "(inspect ptxas/SASS to confirm spill freedom)\n",
                ValueTile, attributes.numRegs, attributes.localSizeBytes,
                attributes.sharedSizeBytes);
}
}  // namespace

int main() {
    static_assert(kBatch == 8u, "This benchmark requires batch8");
    constexpr int runs = 20;
    // Separate allocations: qkv, g, beta, immutable input state, final state,
    // BF16-rounded float output, delta history, decay history.
    const size_t counts[] = {
        size_t(kBatch) * kConvDim, size_t(kBatch) * kValueHeads,
        size_t(kBatch) * kValueHeads, size_t(kValueHeads) * kHeadDim * kHeadDim,
        size_t(kValueHeads) * kHeadDim * kHeadDim,
        size_t(kBatch) * kValueDim, size_t(kBatch) * kValueDim,
        size_t(kBatch) * kValueHeads};
    const char *names[] = {"final_state", "out", "delta_history", "decay_history"};
    const char *variants[] = {"baseline_temporal8_tile8", "register_tile8",
                              "register_tile16", "register_tile32"};
    using Launch = decltype(&launch_gdn_recurrence_temporal8<8>);
    const Launch launches[] = {launch_gdn_recurrence_temporal8<8>,
        launch_register_recurrence<8>, launch_register_recurrence<16>,
        launch_register_recurrence<32>};
    float *device[8]{};
    cudaEvent_t start = nullptr, stop = nullptr;
    int result = EXIT_SUCCESS;
    try {
        const uint32_t seed = bench_seed_from_env();
        std::printf("seed=%u\n", static_cast<unsigned int>(seed));
        std::vector<float> inputs[4], reference[4];
        for (int b = 0; b < 4; ++b) {
            inputs[b].resize(counts[b]);
            for (size_t i = 0; i < counts[b]; ++i) {
                const float x = bench_sample(i, seed + 7919u * b);
                inputs[b][i] = b == 1 ? -0.25f + 0.125f * x :
                               b == 2 ? 0.5f + 0.25f * x : 0.03125f * x;
            }
        }
        for (int b = 0; b < 8; ++b)
            REGISTER_BENCH_CUDA(cudaMalloc(reinterpret_cast<void **>(&device[b]),
                                           counts[b] * sizeof(float)));
        REGISTER_BENCH_CUDA(cudaEventCreate(&start));
        REGISTER_BENCH_CUDA(cudaEventCreate(&stop));
        int ordinal = 0;
        cudaDeviceProp properties{};
        REGISTER_BENCH_CUDA(cudaGetDevice(&ordinal));
        REGISTER_BENCH_CUDA(cudaGetDeviceProperties(&properties, ordinal));
        std::printf("device=%s batch=%u runs=%d (kernel-only, synthetic, no model)\n",
                    properties.name, kBatch, runs);
        report_register_resources<8>();
        report_register_resources<16>();
        report_register_resources<32>();
        float baseline_ms = 0.0f;
        for (int variant = 0; variant < 4; ++variant) {
            float total_ms = 0.0f;
            size_t nonfinite[4]{}, mismatches[4]{};
            bool preserved_input = true;
            // One warmup and 20 reset executions. Transfers, poison, and all
            // comparisons are outside the CUDA-event timing interval.
            for (int run = -1; run < runs; ++run) {
                for (int b = 0; b < 4; ++b)
                    REGISTER_BENCH_CUDA(cudaMemcpy(device[b], inputs[b].data(),
                        counts[b] * sizeof(float), cudaMemcpyHostToDevice));
                for (int b = 4; b < 8; ++b)
                    REGISTER_BENCH_CUDA(cudaMemset(device[b], 0xff, counts[b] * sizeof(float)));
                REGISTER_BENCH_CUDA(cudaEventRecord(start, nullptr));
                const int rc = launches[variant](device[0], device[1], device[2],
                    device[3], device[4], device[5], device[6], device[7], nullptr);
                REGISTER_BENCH_CUDA(cudaGetLastError());
                if (rc != AXIOM_OK) {
                    std::fprintf(stderr, "%s launch rc=%d\n", variants[variant], rc);
                    throw std::runtime_error("recurrence launch failed");
                }
                REGISTER_BENCH_CUDA(cudaEventRecord(stop, nullptr));
                REGISTER_BENCH_CUDA(cudaEventSynchronize(stop));
                float ms = 0.0f;
                REGISTER_BENCH_CUDA(cudaEventElapsedTime(&ms, start, stop));
                if (run >= 0) total_ms += ms;
                // Validate every execution, including warmup. Baseline warmup
                // is the immutable reference for subsequent baseline runs too.
                for (int b = 0; b < 4; ++b) {
                    std::vector<float> got(counts[b + 4]);
                    REGISTER_BENCH_CUDA(cudaMemcpy(got.data(), device[b + 4],
                        got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < got.size(); ++i) {
                        nonfinite[b] += !std::isfinite(got[i]);
                        if (variant != 0 || run != -1)
                            mismatches[b] += std::memcmp(&got[i], &reference[b][i],
                                                         sizeof(float)) != 0;
                    }
                    if (variant == 0 && run == -1) reference[b] = std::move(got);
                }
                std::vector<float> preserved(counts[3]);
                REGISTER_BENCH_CUDA(cudaMemcpy(preserved.data(), device[3],
                    counts[3] * sizeof(float), cudaMemcpyDeviceToHost));
                if (std::memcmp(preserved.data(), inputs[3].data(), counts[3] * sizeof(float)))
                    preserved_input = false;
            }
            bool valid = preserved_input;
            for (int b = 0; b < 4; ++b) {
                std::printf("%s %s elements_per_run=%zu nonfinite=%zu bit_mismatches=%zu "
                            "(counts across warmup + 20 runs)\n", variants[variant],
                            names[b], counts[b + 4], nonfinite[b], mismatches[b]);
                if (nonfinite[b] || mismatches[b]) valid = false;
            }
            if (!preserved_input)
                std::fprintf(stderr, "%s input state modified\n", variants[variant]);
            if (!valid) result = EXIT_FAILURE;
            if (variant == 0) baseline_ms = total_ms;
            std::printf("%s mean_us=%.3f total_ms=%.6f baseline_over_variant=%.4f "
                        "validation=%s\n", variants[variant], total_ms * 1000.0f / runs,
                        total_ms, total_ms > 0.0f ? baseline_ms / total_ms : 0.0f,
                        valid ? "PASS" : "FAIL");
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "register microbench: %s\n", error.what());
        result = EXIT_FAILURE;
    }
    // Check every cleanup result even if another cleanup or the test failed.
    auto cleanup = [&](cudaError_t error, const char *operation) {
        if (error != cudaSuccess) {
            std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
            result = EXIT_FAILURE;
        }
    };
    cleanup(cudaDeviceSynchronize(), "cudaDeviceSynchronize (cleanup)");
    if (stop) cleanup(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    if (start) cleanup(cudaEventDestroy(start), "cudaEventDestroy(start)");
    for (float *pointer : device)
        if (pointer) cleanup(cudaFree(pointer), "cudaFree");
    std::printf("%s\n", result == EXIT_SUCCESS ? "PASS" : "FAIL");
    return result;
}
