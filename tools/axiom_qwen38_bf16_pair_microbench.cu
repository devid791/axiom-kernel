// Bounded standalone probe of the shared production kernel. Compile with the exact
// production FP flags (including FMA contraction); do not enable fast math.
// This includes the implementation: link its Axiom/cuBLAS dependencies, but
// do not also link a second bf16_linear implementation object.
// HTTP context supplied by caller: pair ~53.5 ms / 1584 calls (~7.6%).
// Synthetic warm-buffer timings do NOT establish HTTP or matrix-pool gains.
#include "../src/axiom_qwen38_bf16_linear.cu"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>

namespace pair_microbench {
constexpr int runs = 20;
constexpr unsigned batch = 8;
static_assert(AXIOM_QWEN38_BF16_LINEAR_BATCH == batch, "requires batch eight");


bool cleanup_failed = false;
void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        throw std::runtime_error(operation);
    }
}
void cleanup(cudaError_t status, const char *operation) noexcept {
    if (status != cudaSuccess) {
        cleanup_failed = true;
        std::fprintf(stderr, "cleanup %s: %s\n", operation, cudaGetErrorString(status));
    }
}
#define PAIR_CHECK(call) ::pair_microbench::check((call), #call)
struct Resources {
    uint16_t *w1 = nullptr, *w2 = nullptr;
    float *x = nullptr, *y1 = nullptr, *y2 = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    ~Resources() {
        if (stream) cleanup(cudaStreamSynchronize(stream), "stream synchronize");
        if (stop) cleanup(cudaEventDestroy(stop), "stop event");
        if (start) cleanup(cudaEventDestroy(start), "start event");
        if (y2) cleanup(cudaFree(y2), "y2");
        if (y1) cleanup(cudaFree(y1), "y1");
        if (x) cleanup(cudaFree(x), "x");
        if (w2) cleanup(cudaFree(w2), "w2");
        if (w1) cleanup(cudaFree(w1), "w1");
        if (stream) cleanup(cudaStreamDestroy(stream), "stream destroy");
    }
};

uint32_t mix(uint32_t v) {
    v ^= v >> 16; v *= 0x7feb352du;
    v ^= v >> 15; v *= 0x846ca68bu;
    return v ^ (v >> 16);
}
float sample(uint32_t index, uint32_t seed) {
    const uint32_t bits = mix(index ^ seed);
    // Signed, finite, varied exponents and mantissas; deterministic dyadics.
    return std::ldexp(float(int(bits & 65535u) - 32768),
                      int((bits >> 16) % 9) - 19);
}
uint32_t bits_of(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <unsigned Threads>
void launch(Resources &r, unsigned rows, unsigned cols) {
    bf16_linear_pair_batch8_virtual_kernel<Threads><<<dim3(rows, batch), Threads, 0, r.stream>>>(
            r.w1, r.w2, r.x, r.y1, r.y2, rows, cols);
    PAIR_CHECK(cudaGetLastError());
}
template <>
void launch<256>(Resources &r, unsigned rows, unsigned cols) {
    bf16_linear_pair_batch8_reference_kernel<<<dim3(rows, batch), 256, 0, r.stream>>>(
            r.w1, r.w2, r.x, r.y1, r.y2, rows, cols);
    PAIR_CHECK(cudaGetLastError());
}
template <unsigned Threads>
void attributes() {
    cudaFuncAttributes attr{};
    PAIR_CHECK(cudaFuncGetAttributes(&attr, bf16_linear_pair_batch8_virtual_kernel<Threads>));
    std::printf("threads=%u registers/thread=%d local_bytes/thread=%zu shared_bytes=%zu max_threads=%d\n",
                Threads, attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes,
                attr.maxThreadsPerBlock);
}
void download(Resources &r, std::vector<float> &a, std::vector<float> &b) {
    PAIR_CHECK(cudaMemcpyAsync(a.data(), r.y1, a.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, r.stream));
    PAIR_CHECK(cudaMemcpyAsync(b.data(), r.y2, b.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, r.stream));
    PAIR_CHECK(cudaStreamSynchronize(r.stream));
}
void compare(const std::vector<float> &got, const std::vector<float> &ref,
             unsigned threads, unsigned rows, unsigned cols, uint32_t seed,
             int run, const char *projection) {
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(ref[i]) ||
            bits_of(got[i]) != bits_of(ref[i])) {
            std::fprintf(stderr, "FAIL threads=%u dims=%ux%u seed=%u run=%d projection=%s column=%zu row=%zu got=%08x ref=%08x\n",
                         threads, rows, cols, seed, run, projection, i / rows,
                         i % rows, bits_of(got[i]), bits_of(ref[i]));
            throw std::runtime_error("finite/bitwise comparison failed");
        }
    }
}
template <unsigned Threads>
void measure(Resources &r, unsigned rows, unsigned cols, uint32_t seed,
             const std::vector<float> &ref1, const std::vector<float> &ref2) {
    std::vector<float> a(ref1.size()), b(ref2.size());
    float elapsed[runs];
    for (int i = 0; i < 3; ++i) launch<Threads>(r, rows, cols);
    PAIR_CHECK(cudaStreamSynchronize(r.stream));
    for (int i = 0; i < runs; ++i) {
        // Poison outside the timed interval so unwritten outputs cannot pass.
        PAIR_CHECK(cudaMemsetAsync(r.y1, 0xff, a.size() * sizeof(float), r.stream));
        PAIR_CHECK(cudaMemsetAsync(r.y2, 0xff, b.size() * sizeof(float), r.stream));
        PAIR_CHECK(cudaEventRecord(r.start, r.stream));
        launch<Threads>(r, rows, cols);
        PAIR_CHECK(cudaEventRecord(r.stop, r.stream));
        PAIR_CHECK(cudaEventSynchronize(r.stop));
        PAIR_CHECK(cudaEventElapsedTime(&elapsed[i], r.start, r.stop));
        download(r, a, b);
        compare(a, ref1, Threads, rows, cols, seed, i, "first");
        compare(b, ref2, Threads, rows, cols, seed, i, "second");
    }
    float total = 0;
    for (float ms : elapsed) total += ms;
    std::sort(elapsed, elapsed + runs);
    std::printf("dims=%ux%u columns=8 seed=%u threads=%u runs=%d finite_bitwise=PASS both_projections mean_us=%.3f median_us=%.3f min_us=%.3f max_us=%.3f\n",
                rows, cols, seed, Threads, runs, total * 1000 / runs,
                (elapsed[9] + elapsed[10]) * 500, elapsed[0] * 1000,
                elapsed[runs - 1] * 1000);
}
void run_case(unsigned rows, unsigned cols, uint32_t seed) {
    const size_t weights = size_t(rows) * cols, inputs = size_t(batch) * cols;
    const size_t outputs = size_t(batch) * rows;
    std::vector<uint16_t> w1(weights), w2(weights);
    std::vector<float> x(inputs), ref1(outputs), ref2(outputs);
    // Synchronize/free device work before host upload buffers are destroyed,
    // including exception paths after an asynchronous upload.
    Resources r;
    for (size_t i = 0; i < weights; ++i) {
        w1[i] = float_to_bf16_host(sample(uint32_t(i), seed ^ 0x12345678u));
        w2[i] = float_to_bf16_host(sample(uint32_t(i), seed ^ 0x9abcdef0u));
    }
    for (size_t i = 0; i < inputs; ++i) x[i] = sample(uint32_t(i), seed ^ 0xfedcba98u);
    PAIR_CHECK(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
    PAIR_CHECK(cudaEventCreate(&r.start));
    PAIR_CHECK(cudaEventCreate(&r.stop));
    PAIR_CHECK(cudaMalloc(reinterpret_cast<void **>(&r.w1), weights * sizeof(uint16_t)));
    PAIR_CHECK(cudaMalloc(reinterpret_cast<void **>(&r.w2), weights * sizeof(uint16_t)));
    PAIR_CHECK(cudaMalloc(reinterpret_cast<void **>(&r.x), inputs * sizeof(float)));
    PAIR_CHECK(cudaMalloc(reinterpret_cast<void **>(&r.y1), outputs * sizeof(float)));
    PAIR_CHECK(cudaMalloc(reinterpret_cast<void **>(&r.y2), outputs * sizeof(float)));
    PAIR_CHECK(cudaMemcpyAsync(r.w1, w1.data(), weights * sizeof(uint16_t), cudaMemcpyHostToDevice, r.stream));
    PAIR_CHECK(cudaMemcpyAsync(r.w2, w2.data(), weights * sizeof(uint16_t), cudaMemcpyHostToDevice, r.stream));
    PAIR_CHECK(cudaMemcpyAsync(r.x, x.data(), inputs * sizeof(float), cudaMemcpyHostToDevice, r.stream));
    PAIR_CHECK(cudaMemsetAsync(r.y1, 0xff, outputs * sizeof(float), r.stream));
    PAIR_CHECK(cudaMemsetAsync(r.y2, 0xff, outputs * sizeof(float), r.stream));
    launch<256>(r, rows, cols);
    download(r, ref1, ref2);
    compare(ref1, ref1, 256, rows, cols, seed, -1, "first reference");
    compare(ref2, ref2, 256, rows, cols, seed, -1, "second reference");
    measure<256>(r, rows, cols, seed, ref1, ref2);
    measure<32>(r, rows, cols, seed, ref1, ref2);
    measure<64>(r, rows, cols, seed, ref1, ref2);
    measure<128>(r, rows, cols, seed, ref1, ref2);
}
} // namespace pair_microbench

int main() {
    using namespace pair_microbench;
    int result = 0;
    try {
        int device = 0;
        PAIR_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        PAIR_CHECK(cudaGetDeviceProperties(&prop, device));
        std::printf("device=%d name=%s cc=%d.%d; warm synthetic buffers, kernel-only CUDA events\n",
                    device, prop.name, prop.major, prop.minor);
        cudaFuncAttributes attr{};
        PAIR_CHECK(cudaFuncGetAttributes(&attr, bf16_linear_pair_batch8_reference_kernel));
        std::printf("baseline threads=256 registers/thread=%d local_bytes/thread=%zu shared_bytes=%zu max_threads=%d\n",
                    attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes, attr.maxThreadsPerBlock);
        attributes<32>(); attributes<64>(); attributes<128>();
        for (uint32_t seed : {1u, 0x31415926u, 0xdeadbeefu}) {
            run_case(48, 5120, seed);
            run_case(7, 513, seed);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "pair microbench failed: %s\n", e.what());
        result = 1;
    }
    return cleanup_failed ? 1 : result;
}
