/* Standalone synthetic denominator-overlap parity and kernel-only timing.
 * Requires the parent's template<bool OverlapDenominator=false> fused kernel.
 * Build from the repository root (Makefile NVCCFLAGS_PRECISE; NO fast math):
 * /usr/local/cuda/bin/nvcc -O3 -arch=sm_120 --default-stream per-thread \
 *   -std=c++17 -U_GNU_SOURCE -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L \
 *   -Iinclude -I/usr/local/cuda/include \
 *   tools/axiom_qwen38_attention_overlap_microbench.cu -Llib \
 *   -L/usr/local/cuda/lib64 -laxiom -lcublasLt -lcublas -lcudart \
 *   -Xlinker -rpath -Xlinker '$ORIGIN/../lib' \
 *   -o bin/axiom-qwen38-attention-overlap-microbench
 * Pass --sanity-only for parity checks without warmups, capture, or timing.
 * No fixtures, weights, services, or production dispatch are used.
 */
#include "../src/axiom_qwen38_attention.cu"
#include <cmath>
#include <exception>
#include <stdexcept>

namespace attention_overlap_microbench {
constexpr uint32_t context = 8192u;
constexpr size_t states = size_t(kHeads) * kBatch;
constexpr size_t q_count = states * kHeadDim;
constexpr size_t kv_count = size_t(context) * kKvHeads * kHeadDim;
constexpr size_t score_count = states * context;
constexpr int rounds = 10, launches_per_sample = 4, warmups = 2;
static_assert(kHeads == 24u && kBatch == 8u && kHeadDim == 256u &&
              kExactScoreThreads == 256u, "Requires 24x8 D256 attention");
static_assert(sizeof(float) == sizeof(uint32_t), "Requires 32-bit float");

void check(cudaError_t status, const char *what) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "FAIL: %s: %s (%d)\n", what,
                 cudaGetErrorString(status), int(status));
    throw std::runtime_error(what);
}
#define OVERLAP_CHECK(call) check((call), #call)

struct Resources {
    bool &cleanup_ok;
    void *allocations[11]{};
    size_t used = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaGraph_t graphs[2]{};
    cudaGraphExec_t execs[2]{};
    bool capturing = false;
    explicit Resources(bool &ok) : cleanup_ok(ok) {}
    Resources(const Resources &) = delete;
    Resources &operator=(const Resources &) = delete;
    void release(cudaError_t status, const char *what) noexcept {
        if (status == cudaSuccess) return;
        cleanup_ok = false;
        std::fprintf(stderr, "FAIL: cleanup %s: %s (%d)\n", what,
                     cudaGetErrorString(status), int(status));
    }
    ~Resources() {
        if (capturing) {
            cudaGraph_t abandoned = nullptr;
            release(cudaStreamEndCapture(stream, &abandoned), "end capture");
            if (abandoned) release(cudaGraphDestroy(abandoned), "abandoned graph");
        }
        if (stream) release(cudaStreamSynchronize(stream), "stream sync");
        for (int p = 0; p < 2; ++p) {
            if (execs[p]) release(cudaGraphExecDestroy(execs[p]), "graph exec");
            if (graphs[p]) release(cudaGraphDestroy(graphs[p]), "graph");
        }
        if (stop) release(cudaEventDestroy(stop), "stop event");
        if (start) release(cudaEventDestroy(start), "start event");
        while (used) release(cudaFree(allocations[--used]), "allocation");
        if (stream) release(cudaStreamDestroy(stream), "stream");
    }
    template<class T> T *allocate(size_t count) {
        if (!count || used == 11 || count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("invalid allocation size/registry capacity");
        void *pointer = nullptr;
        OVERLAP_CHECK(cudaMalloc(&pointer, count * sizeof(T)));
        allocations[used++] = pointer;
        return static_cast<T *>(pointer);
    }
};

// Copied from attention_microbench's GPU helper, with an XOR seed salt.
// seed=0 reproduces its q/k/v exactly; all inputs remain GPU-generated.
__device__ uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
__global__ void initialize(float *q, uint8_t *k, uint8_t *v, uint32_t seed) {
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < kv_count || i < q_count; i += stride) {
        if (i < q_count) {
            const int value = int(mix(uint32_t(i) ^ 0x12345678u ^ seed) % 2049u) - 1024;
            q[i] = float(value) / 1024.0f;
        }
        if (i < kv_count) {
            const uint32_t kh = mix(uint32_t(i) ^ 0x31415926u ^ seed);
            const uint32_t vh = mix(uint32_t(i) ^ 0x27182818u ^ seed);
            // Finite E4M3 magnitudes 0..2, both signs, zeros and subnormals.
            k[i] = uint8_t((kh & 0x80u) | ((kh >> 8) % 65u));
            v[i] = uint8_t((vh & 0x80u) | ((vh >> 8) % 65u));
        }
    }
}

enum Pattern { Qk, Random, Increasing, Decreasing, Tied };
const char *pattern_names[] = {"qk", "random", "increasing", "decreasing", "tied"};
__global__ void initialize_scores(float *scores, Pattern pattern, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < score_count; i += size_t(blockDim.x) * gridDim.x) {
        const uint32_t token = uint32_t(i % context);
        const float offset = float(int(mix(uint32_t(i / context) ^ seed) % 33u) - 16) / 8.0f;
        float value = offset;
        if (pattern == Random)
            value = float(int(mix(uint32_t(i) ^ seed) % 16385u) - 8192) / 1024.0f;
        else if (pattern == Increasing) value += float(token) / 256.0f;
        else if (pattern == Decreasing) value -= float(token) / 256.0f;
        scores[i] = value;
    }
}

struct Path { float *maxima, *denominators, *out; };
void launch(int p, const float *q, const uint8_t *k, const uint8_t *v,
            const float *scores, const uint32_t *position, const Path &path,
            cudaStream_t stream) {
    if (p == 0)
        qwen38_attention_temporal_exact_fused_kernel<false><<<dim3(kHeads, kBatch), 256, 0, stream>>>(
            q, k, v, path.maxima, path.denominators, path.out, 0u, position, context, true, scores);
    else
        qwen38_attention_temporal_exact_fused_kernel<true><<<dim3(kHeads, kBatch), 288, 0, stream>>>(
            q, k, v, path.maxima, path.denominators, path.out, 0u, position, context, true, scores);
    OVERLAP_CHECK(cudaGetLastError());
}

bool compare(const char *field, const float *a, const float *b, size_t count,
             uint32_t pos, uint32_t seed, Pattern pattern) {
    std::vector<uint32_t> bits[2];
    for (auto &buffer : bits) buffer.resize(count);
    OVERLAP_CHECK(cudaMemcpy(bits[0].data(), a, count * sizeof(float), cudaMemcpyDeviceToHost));
    OVERLAP_CHECK(cudaMemcpy(bits[1].data(), b, count * sizeof(float), cudaMemcpyDeviceToHost));
    size_t mismatches = 0, nonfinite = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t x = bits[0][i], y = bits[1][i];
        const bool bad = (x & 0x7f800000u) == 0x7f800000u ||
                         (y & 0x7f800000u) == 0x7f800000u;
        if ((x != y || bad) && mismatches + nonfinite == 0)
            std::fprintf(stderr, "FAIL pos=%u seed=%u pattern=%s field=%s index=%zu reference=%08x candidate=%08x\n",
                         pos, seed, pattern_names[pattern], field, i, x, y);
        mismatches += x != y;
        nonfinite += bad;
    }
    std::printf("%s pos=%u seed=%u pattern=%s field=%s count=%zu mismatches=%zu nonfinite_pairs=%zu\n",
                mismatches || nonfinite ? "FAIL" : "PASS", pos, seed,
                pattern_names[pattern], field, count, mismatches, nonfinite);
    return !mismatches && !nonfinite;
}

bool run(bool &cleanup_ok, bool sanity_only, bool race_smoke) {
    Resources r(cleanup_ok);
    OVERLAP_CHECK(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
    float *q = r.allocate<float>(q_count);
    uint8_t *k = r.allocate<uint8_t>(kv_count), *v = r.allocate<uint8_t>(kv_count);
    float *scores = r.allocate<float>(score_count);
    uint32_t *position = r.allocate<uint32_t>(1);
    Path paths[2];
    for (auto &p : paths)
        p = {r.allocate<float>(states), r.allocate<float>(states), r.allocate<float>(q_count)};
    std::printf("heads=%u batch=%u dim=%u context=%u reference_threads=256 candidate_threads=288\n",
                kHeads, kBatch, kHeadDim, context);
    const uint32_t positions[] = {0u, 1u, 255u, 256u, 1002u, 8184u};
    const uint32_t seeds[] = {0u, 1u, 0xdeadbeefu};
    const auto prepare_scores = [&](Pattern pattern, uint32_t seed) {
        OVERLAP_CHECK(cudaMemsetAsync(scores, 0xff, score_count * sizeof(float), r.stream));
        if (pattern == Qk)
            qwen38_attention_temporal_exact_score_kernel<<<dim3(kHeads, kBatch, 4u), kExactScoreThreads, 0, r.stream>>>(
                q, k, scores, 0u, position, context, true);
        else
            initialize_scores<<<256, 256, 0, r.stream>>>(scores, pattern, seed);
        OVERLAP_CHECK(cudaGetLastError());
    };
    bool passed = true;
    for (uint32_t seed : seeds) {
        if (race_smoke && seed != 0u) continue;
        initialize<<<256, 256, 0, r.stream>>>(q, k, v, seed);
        OVERLAP_CHECK(cudaGetLastError());
        for (uint32_t pos : positions) {
            if (race_smoke && pos != 1002u) continue;
            if (pos > context || kBatch > context - pos)
                throw std::out_of_range("position exceeds allocated cache");
            OVERLAP_CHECK(cudaMemcpyAsync(position, &pos, sizeof(pos), cudaMemcpyHostToDevice, r.stream));
            for (Pattern pattern : {Qk, Random, Increasing, Decreasing, Tied}) {
                if (race_smoke && pattern != Qk) continue;
                // Produce once; both kernels consume exactly the same score buffer.
                prepare_scores(pattern, seed);
                for (int p = 0; p < 2; ++p) {
                    OVERLAP_CHECK(cudaMemsetAsync(paths[p].out, 0xff, q_count * sizeof(float), r.stream));
                    OVERLAP_CHECK(cudaMemsetAsync(paths[p].maxima, 0xff, states * sizeof(float), r.stream));
                    OVERLAP_CHECK(cudaMemsetAsync(paths[p].denominators, 0xff, states * sizeof(float), r.stream));
                    launch(p, q, k, v, scores, position, paths[p], r.stream);
                }
                OVERLAP_CHECK(cudaStreamSynchronize(r.stream));
                passed = compare("output", paths[0].out, paths[1].out, q_count, pos, seed, pattern) && passed;
                passed = compare("maxima", paths[0].maxima, paths[1].maxima, states, pos, seed, pattern) && passed;
                passed = compare("denominators", paths[0].denominators, paths[1].denominators, states, pos, seed, pattern) && passed;
            }
        }
    }
    if (!passed) return false;
    if (sanity_only) {
        std::printf("SANITY ONLY: all %u cases passed; timing skipped\n", race_smoke ? 1u : 90u);
        return true;
    }

    const uint32_t timed_pos = 1002u;
    initialize<<<256, 256, 0, r.stream>>>(q, k, v, 0u);
    OVERLAP_CHECK(cudaGetLastError());
    OVERLAP_CHECK(cudaMemcpyAsync(position, &timed_pos, sizeof(timed_pos), cudaMemcpyHostToDevice, r.stream));
    prepare_scores(Qk, 0u);
    OVERLAP_CHECK(cudaStreamSynchronize(r.stream));
    OVERLAP_CHECK(cudaEventCreate(&r.start));
    OVERLAP_CHECK(cudaEventCreate(&r.stop));
    for (int p = 0; p < 2; ++p) {
        OVERLAP_CHECK(cudaStreamBeginCapture(r.stream, cudaStreamCaptureModeThreadLocal));
        r.capturing = true;
        launch(p, q, k, v, scores, position, paths[p], r.stream);
        const cudaError_t ended = cudaStreamEndCapture(r.stream, &r.graphs[p]);
        r.capturing = false;
        check(ended, "end graph capture");
        OVERLAP_CHECK(cudaGraphInstantiate(&r.execs[p], r.graphs[p], nullptr, nullptr, 0));
    }
    // Fixed bounded work: 2 warmups + 10 alternating AB/BA pairs of 4
    // launches per path/mode. Scores, initialization, and capture are untimed.
    // tile_begin=0 resets consumed state on every eager/graph invocation.
    for (int mode = 0; mode < 2; ++mode) {
        const auto submit = [&](int p) {
            if (mode) OVERLAP_CHECK(cudaGraphLaunch(r.execs[p], r.stream));
            else launch(p, q, k, v, scores, position, paths[p], r.stream);
        };
        for (int i = 0; i < warmups; ++i)
            for (int p = 0; p < 2; ++p) submit(p ^ (i & 1));
        OVERLAP_CHECK(cudaStreamSynchronize(r.stream));
        double totals[2]{};
        for (int round = 0; round < rounds; ++round) {
            for (int order = 0; order < 2; ++order) {
                const int p = order ^ (round & 1);
                OVERLAP_CHECK(cudaEventRecord(r.start, r.stream));
                for (int i = 0; i < launches_per_sample; ++i) submit(p);
                OVERLAP_CHECK(cudaEventRecord(r.stop, r.stream));
                OVERLAP_CHECK(cudaEventSynchronize(r.stop));
                float ms = 0.0f;
                OVERLAP_CHECK(cudaEventElapsedTime(&ms, r.start, r.stop));
                if (!std::isfinite(ms) || ms <= 0.0f)
                    throw std::runtime_error("invalid event timing");
                totals[p] += ms;
                std::printf("SAMPLE mode=%s round=%d order=%d path=%s total_ms=%.6f\n",
                            mode ? "graph" : "eager", round, order, p ? "candidate" : "reference", ms);
            }
        }
        std::printf("TIME mode=%s pos=%u kernel_only=1 rounds=%d launches_per_sample=%d reference_us=%.6f candidate_us=%.6f reference_over_candidate=%.6f\n",
                    mode ? "graph" : "eager", timed_pos, rounds, launches_per_sample,
                    totals[0] * 1000 / (rounds * launches_per_sample),
                    totals[1] * 1000 / (rounds * launches_per_sample), totals[0] / totals[1]);
        // Also validate state after repeated execution, including graph replay.
        passed = compare("output", paths[0].out, paths[1].out, q_count, timed_pos, 0u, Qk) && passed;
        passed = compare("maxima", paths[0].maxima, paths[1].maxima, states, timed_pos, 0u, Qk) && passed;
        passed = compare("denominators", paths[0].denominators, paths[1].denominators, states, timed_pos, 0u, Qk) && passed;
    }
    return passed;
}
#undef OVERLAP_CHECK
} // namespace attention_overlap_microbench

int main(int argc, char **argv) {
    const bool race_smoke = argc == 2 && std::strcmp(argv[1], "--race-smoke") == 0;
    const bool sanity_only = race_smoke || (argc == 2 && std::strcmp(argv[1], "--sanity-only") == 0);
    if (argc != 1 && !sanity_only) {
        std::fprintf(stderr, "Usage: %s [--sanity-only|--race-smoke]\n", argv[0]);
        return EXIT_FAILURE;
    }
    bool cleanup_ok = true, passed = false;
    try {
        passed = attention_overlap_microbench::run(cleanup_ok, sanity_only, race_smoke);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
    } catch (...) {
        std::fprintf(stderr, "FAIL: unknown exception\n");
    }
    std::printf("RESULT %s\n", passed && cleanup_ok ? "PASS" : "FAIL");
    return passed && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
