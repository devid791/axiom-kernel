// Standalone tuning probe; the including executable must link Axiom dependencies.
#include "../src/axiom_qwen38_gdn.cu"
#include <cstring>
#include <exception>
#include <stdexcept>

namespace {
void bench_check(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    throw std::runtime_error(operation);
}
#define BENCH_CUDA(call) bench_check((call), #call)

float bench_sample(size_t index, uint32_t salt) {
    uint32_t x = static_cast<uint32_t>(index) + salt;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (static_cast<int>(x & 2047u) - 1024) * (1.0f / 1024.0f);
}
}  // namespace

int main() {
    static_assert(kBatch == 8u, "This benchmark requires batch8");
    constexpr int runs = 20;
    // qkv, g, beta, immutable input state, final_state, out, delta, decay.
    const size_t counts[] = {
        size_t(kBatch) * kConvDim, size_t(kBatch) * kValueHeads,
        size_t(kBatch) * kValueHeads, size_t(kValueHeads) * kHeadDim * kHeadDim,
        size_t(kValueHeads) * kHeadDim * kHeadDim,
        size_t(kBatch) * kValueDim, size_t(kBatch) * kValueDim,
        size_t(kBatch) * kValueHeads};
    const char *names[] = {"final_state", "out", "delta_history", "decay_history"};
    const unsigned tiles[] = {8, 16, 32, 64};
    using Launch = decltype(&launch_gdn_recurrence_temporal8<8>);
    const Launch launches[] = {launch_gdn_recurrence_temporal8<8>,
        launch_gdn_recurrence_temporal8<16>, launch_gdn_recurrence_temporal8<32>,
        launch_gdn_recurrence_temporal8<64>};
    float *device[8]{};
    cudaEvent_t start = nullptr, stop = nullptr;
    int result = EXIT_SUCCESS;
    try {
        std::vector<float> inputs[4], reference[4];
        for (int b = 0; b < 4; ++b) {
            inputs[b].resize(counts[b]);
            for (size_t i = 0; i < counts[b]; ++i) {
                const float x = bench_sample(i, 12345u + 7919u * b);
                inputs[b][i] = b == 1 ? -0.25f + 0.125f * x :
                               b == 2 ? 0.5f + 0.25f * x : 0.03125f * x;
            }
        }
        for (int b = 0; b < 8; ++b)
            BENCH_CUDA(cudaMalloc(reinterpret_cast<void **>(&device[b]),
                                  counts[b] * sizeof(float)));
        BENCH_CUDA(cudaEventCreate(&start));
        BENCH_CUDA(cudaEventCreate(&stop));
        int ordinal = 0;
        cudaDeviceProp properties{};
        BENCH_CUDA(cudaGetDevice(&ordinal));
        BENCH_CUDA(cudaGetDeviceProperties(&properties, ordinal));
        std::printf("device=%s batch=%u runs=%d (kernel-only timing)\n",
                    properties.name, kBatch, runs);
        for (int variant = 0; variant < 4; ++variant) {
            float total_ms = 0.0f;
            // One untimed warmup, then 20 independently reset executions.
            for (int run = -1; run < runs; ++run) {
                for (int b = 0; b < 4; ++b)
                    BENCH_CUDA(cudaMemcpy(device[b], inputs[b].data(),
                        counts[b] * sizeof(float), cudaMemcpyHostToDevice));
                // NaN poison catches any unwritten output elements.
                for (int b = 4; b < 8; ++b)
                    BENCH_CUDA(cudaMemset(device[b], 0xff, counts[b] * sizeof(float)));
                BENCH_CUDA(cudaEventRecord(start, nullptr));
                const int rc = launches[variant](device[0], device[1], device[2],
                    device[3], device[4], device[5], device[6], device[7], nullptr);
                if (rc != AXIOM_OK) {
                    std::fprintf(stderr, "tile=%u launch rc=%d\n", tiles[variant], rc);
                    throw std::runtime_error("recurrence launch failed");
                }
                BENCH_CUDA(cudaGetLastError());
                BENCH_CUDA(cudaEventRecord(stop, nullptr));
                BENCH_CUDA(cudaEventSynchronize(stop));
                float ms = 0.0f;
                BENCH_CUDA(cudaEventElapsedTime(&ms, start, stop));
                if (run >= 0) total_ms += ms;
            }
            for (int b = 0; b < 4; ++b) {
                std::vector<float> got(counts[b + 4]);
                BENCH_CUDA(cudaMemcpy(got.data(), device[b + 4],
                    got.size() * sizeof(float), cudaMemcpyDeviceToHost));
                size_t nonfinite = 0, mismatches = 0;
                for (size_t i = 0; i < got.size(); ++i) {
                    nonfinite += !std::isfinite(got[i]);
                    if (variant != 0)
                        mismatches += std::memcmp(&got[i], &reference[b][i],
                                                  sizeof(float)) != 0;
                }
                std::printf("tile=%u %s elements=%zu nonfinite=%zu bit_mismatches=%zu\n",
                            tiles[variant], names[b], got.size(), nonfinite, mismatches);
                if (nonfinite || mismatches) result = EXIT_FAILURE;
                if (variant == 0) reference[b] = std::move(got);
            }
            std::vector<float> preserved(counts[3]);
            BENCH_CUDA(cudaMemcpy(preserved.data(), device[3],
                counts[3] * sizeof(float), cudaMemcpyDeviceToHost));
            if (std::memcmp(preserved.data(), inputs[3].data(), counts[3] * sizeof(float))) {
                std::fprintf(stderr, "tile=%u input state modified\n", tiles[variant]);
                result = EXIT_FAILURE;
            }
            std::printf("tile=%u mean_us=%.3f total_ms=%.6f\n",
                        tiles[variant], total_ms * 1000.0f / runs, total_ms);
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "microbench: %s\n", error.what());
        result = EXIT_FAILURE;
    }
    // Best-effort cleanup checks every result, even after an earlier failure.
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
