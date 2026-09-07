/* Standalone synthetic-input attention parity/timing test. No model weights.
 * Build from repository root with the same flags as libaxiom:
 * nvcc -O3 --use_fast_math -arch=sm_120 --default-stream per-thread -std=c++17
 *   -Iinclude tools/axiom_qwen38_attention_microbench.cu -Llib
 *   -Xlinker -rpath -Xlinker '$ORIGIN/../lib'
 *   -laxiom -lcublasLt -lcublas -lcudart -o bin/axiom-attention-microbench
 */
#include "../src/axiom_qwen38_attention.cu"
#include <cmath>
#include <exception>
#include <stdexcept>

namespace attention_microbench {
constexpr uint32_t context = 8192u;
constexpr int runs = 20;
static_assert(kHeads == 24u && kBatch == 8u,
              "This benchmark requires the requested 24x8 score grid");
static_assert(kHeadDim == kExactScoreThreads,
              "Exact kernels require one thread per head dimension");

void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s (%d)\n", operation,
                     cudaGetErrorString(status), static_cast<int>(status));
        throw std::runtime_error(operation);
    }
}

size_t multiply(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("allocation size overflow");
    return a * b;
}

// Every successful allocation/event is registered before subsequent work.
// Cleanup failures also make main return failure, including during unwinding.
struct Resources {
    bool &cleanup_ok;
    void *allocations[12]{};
    size_t used = 0;
    cudaEvent_t start = nullptr, stop = nullptr;
    explicit Resources(bool &ok) : cleanup_ok(ok) {}
    Resources(const Resources &) = delete;
    Resources &operator=(const Resources &) = delete;
    void release_status(cudaError_t status, const char *what) noexcept {
        if (status != cudaSuccess) {
            cleanup_ok = false;
            std::fprintf(stderr, "FAIL: cleanup %s: %s (%d)\n", what,
                         cudaGetErrorString(status), static_cast<int>(status));
        }
    }
    ~Resources() {
        if (stop) release_status(cudaEventDestroy(stop), "stop event");
        if (start) release_status(cudaEventDestroy(start), "start event");
        while (used) release_status(cudaFree(allocations[--used]), "cudaFree");
    }
    template<class T> T *allocate(size_t count, const char *name) {
        if (!count || used == sizeof(allocations) / sizeof(allocations[0]))
            throw std::runtime_error("invalid allocation count/registry capacity");
        void *pointer = nullptr;
        check(cudaMalloc(&pointer, multiply(count, sizeof(T))), name);
        allocations[used++] = pointer;
        return static_cast<T *>(pointer);
    }
};

__device__ uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

__global__ void initialize(float *q, uint8_t *k, uint8_t *v,
                           size_t q_count, size_t kv_count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < kv_count || i < q_count; i += stride) {
        if (i < q_count) {
            const int value = static_cast<int>(mix(static_cast<uint32_t>(i) ^
                                                  0x12345678u) % 2049u) - 1024;
            q[i] = static_cast<float>(value) / 1024.0f;
        }
        if (i < kv_count) {
            const uint32_t kh = mix(static_cast<uint32_t>(i) ^ 0x31415926u);
            const uint32_t vh = mix(static_cast<uint32_t>(i) ^ 0x27182818u);
            // E4M3 magnitudes 0x00..0x40: finite, bounded by 2, with
            // subnormals, zeros and both signs; never the NaN code 0x7f.
            k[i] = static_cast<uint8_t>((kh & 0x80u) | ((kh >> 8) % 65u));
            v[i] = static_cast<uint8_t>((vh & 0x80u) | ((vh >> 8) % 65u));
        }
    }
}

struct Path {
    float *maxima, *denominators, *out;
};

void launch(bool precomputed, const float *q, const uint8_t *k,
            const uint8_t *v, float *scores, const uint32_t *position,
            const Path &path) {
    if (precomputed) {
        qwen38_attention_temporal_exact_score_kernel<<<
            dim3(kHeads, kBatch, 4u), kExactScoreThreads>>>(
                q, k, scores, 0u, position, context, true);
        check(cudaGetLastError(), "score kernel launch (full_scores=true)");
    }
    qwen38_attention_temporal_exact_fused_kernel<<<
        dim3(kHeads, kBatch), kHeadDim>>>(
            q, k, v, path.maxima, path.denominators, path.out,
            0u, position, context, true, precomputed ? scores : nullptr);
    check(cudaGetLastError(), "persistent fused kernel launch");
}

bool run(bool &cleanup_ok) {
    Resources resources(cleanup_ok);
    const size_t states = multiply(kHeads, kBatch);
    const size_t q_count = multiply(states, kHeadDim);
    const size_t kv_count = multiply(multiply(context, kKvHeads), kHeadDim);
    const size_t score_count = multiply(states, context);
    const size_t output_bytes = multiply(q_count, sizeof(float));
    const size_t state_bytes = multiply(states, sizeof(float));
    float *q = resources.allocate<float>(q_count, "allocate q");
    uint8_t *k = resources.allocate<uint8_t>(kv_count, "allocate k");
    uint8_t *v = resources.allocate<uint8_t>(kv_count, "allocate v");
    float *scores = resources.allocate<float>(score_count, "allocate scores");
    uint32_t *position = resources.allocate<uint32_t>(1, "allocate position");
    const bool cold_cache = std::getenv("AXIOM_MICRO_COLD_CACHE") != nullptr;
    const size_t eviction_bytes = 128u * 1024u * 1024u;
    uint8_t *eviction = cold_cache ? resources.allocate<uint8_t>(eviction_bytes, "allocate eviction") : nullptr;
    Path paths[2]{};
    for (auto &path : paths) {
        path.maxima = resources.allocate<float>(states, "allocate maxima");
        path.denominators = resources.allocate<float>(states, "allocate denominators");
        path.out = resources.allocate<float>(q_count, "allocate output");
    }
    // Only output bits are copied to host; all synthetic inputs originate on GPU.
    std::vector<uint32_t> host[2];
    for (auto &output : host) output.resize(q_count);
    static_assert(sizeof(uint32_t) == sizeof(float), "32-bit float required");
    initialize<<<256, 256>>>(q, k, v, q_count, kv_count);
    check(cudaGetLastError(), "input initialization launch");
    check(cudaDeviceSynchronize(), "input initialization execution");
    std::printf("geometry heads=%u batch=%u kv_heads=%u head_dim=%u context=%u\n",
                kHeads, kBatch, kKvHeads, kHeadDim, context);
    bool passed = true;
    const uint32_t positions[] = {0u, 1u, 255u, 256u, 1002u, 8184u};
    for (uint32_t pos : positions) {
        if (pos > context || kBatch > context - pos)
            throw std::out_of_range("position plus batch exceeds cache allocation");
        check(cudaMemcpy(position, &pos, sizeof(pos), cudaMemcpyHostToDevice),
              "upload position");
        check(cudaMemset(scores, 0xff, multiply(score_count, sizeof(float))),
              "poison scores");
        for (int p = 0; p < 2; ++p) {
            // Poison catches unwritten outputs/state rather than reusing prior results.
            check(cudaMemset(paths[p].maxima, 0xff, state_bytes), "poison maxima");
            check(cudaMemset(paths[p].denominators, 0xff, state_bytes), "poison denominators");
            check(cudaMemset(paths[p].out, 0xff, output_bytes), "poison output");
            launch(p != 0, q, k, v, scores, position, paths[p]);
            check(cudaDeviceSynchronize(), p ? "precomputed path execution" : "fused path execution");
            check(cudaMemcpy(host[p].data(), paths[p].out, output_bytes,
                             cudaMemcpyDeviceToHost), "download output bits");
        }
        size_t mismatches = 0, nonfinite = 0;
        for (size_t i = 0; i < q_count; ++i) {
            const uint32_t a = host[0][i], b = host[1][i];
            const bool bad = (a & 0x7f800000u) == 0x7f800000u ||
                             (b & 0x7f800000u) == 0x7f800000u;
            if ((a != b || bad) && mismatches + nonfinite == 0)
                std::fprintf(stderr, "FAIL: pos=%u first_index=%zu fused=0x%08x precomputed=0x%08x nonfinite=%d\n",
                             pos, i, a, b, static_cast<int>(bad));
            mismatches += a != b;
            nonfinite += bad;
        }
        const bool ok = mismatches == 0 && nonfinite == 0;
        uint64_t fingerprint = 14695981039346656037ull;
        for (uint32_t bits : host[0]) {
            fingerprint ^= bits;
            fingerprint *= 1099511628211ull;
        }
        std::printf("fingerprint pos=%u fnv64=%016llx\n", pos,
                    static_cast<unsigned long long>(fingerprint));
        passed = passed && ok;
        std::printf("%s pos=%u outputs=%zu bit_mismatches=%zu nonfinite_pairs=%zu\n",
                    ok ? "PASS" : "FAIL", pos, q_count, mismatches, nonfinite);
    }
    if (!passed) return false;

    const uint32_t timed_position = 1002u;
    check(cudaMemcpy(position, &timed_position, sizeof(timed_position),
                     cudaMemcpyHostToDevice), "upload timed position");
    check(cudaEventCreate(&resources.start), "create start event");
    check(cudaEventCreate(&resources.stop), "create stop event");
    float milliseconds[2]{};
    for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < 5; ++i)
            launch(p != 0, q, k, v, scores, position, paths[p]);
        check(cudaDeviceSynchronize(), "warmup execution");
        // tile_begin=0 resets all consumed state within each kernel invocation.
        // The precomputed path recomputes scores on EVERY timed iteration.
        if (cold_cache) {
            for (int i = 0; i < runs; ++i) {
                check(cudaMemsetAsync(eviction, 0, eviction_bytes), "cache eviction");
                check(cudaEventRecord(resources.start), "record cold start");
                launch(p != 0, q, k, v, scores, position, paths[p]);
                check(cudaEventRecord(resources.stop), "record cold stop");
                check(cudaEventSynchronize(resources.stop), "cold execution");
                float sample = 0.0f;
                check(cudaEventElapsedTime(&sample, resources.start, resources.stop), "cold time");
                milliseconds[p] += sample;
            }
        } else {
            check(cudaEventRecord(resources.start), "record start event");
            for (int i = 0; i < runs; ++i)
                launch(p != 0, q, k, v, scores, position, paths[p]);
            check(cudaEventRecord(resources.stop), "record stop event");
            check(cudaEventSynchronize(resources.stop), "timed path execution");
            check(cudaEventElapsedTime(&milliseconds[p], resources.start, resources.stop),
                  "elapsed GPU event time");
        }
        if (!std::isfinite(milliseconds[p]) || milliseconds[p] <= 0.0f)
            throw std::runtime_error("invalid elapsed GPU event time");
        std::printf("TIME path=%s pos=%u runs=%d total_ms=%.6f mean_us=%.6f\n",
                    p ? "score+fused" : "fused", timed_position, runs,
                    milliseconds[p], milliseconds[p] * 1000.0f / runs);
    }
    std::printf("fused_over_score_plus_fused_ratio=%.6f\n",
                milliseconds[0] / milliseconds[1]);
    return true;
}
} // namespace attention_microbench

int main() {
    bool cleanup_ok = true, passed = false;
    try {
        passed = attention_microbench::run(cleanup_ok);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "FAIL: unknown exception\n");
    }
    std::printf("RESULT %s\n", passed && cleanup_ok ? "PASS" : "FAIL");
    return passed && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
