// Compile twice with AXIOM_DSPARK_PARALLEL_SOFTMAX=0/1; compare hashes.
#include "../src/axiom_qwen38_dspark_compute.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define CHECK_CUDA(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

template<class T> T* allocate(size_t count) {
    T* p = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)));
    return p;
}

__global__ void initialize_test(float* q, float* k, float* v,
                              uint16_t* kc, uint16_t* vc,
                              size_t qc, size_t lc, size_t cc) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cc;
         i += gridDim.x * blockDim.x) {
        const float a = (static_cast<int>((i * 17u + 3u) % 257u) - 128) / 128.0f;
        const float b = (static_cast<int>((i * 31u + 7u) % 251u) - 125) / 128.0f;
        if (i < qc) q[i] = b;
        if (i < lc) { k[i] = a; v[i] = b; }
        kc[i] = static_cast<uint16_t>(__float_as_uint(a) >> 16u);
        vc[i] = static_cast<uint16_t>(__float_as_uint(b) >> 16u);
    }
}

int main() {
    constexpr uint32_t columns = 7u, context = 8192u;
    const size_t count = columns * kHidden;
    float* q = allocate<float>(count);
    float* k = allocate<float>(columns * kKvDim);
    float* v = allocate<float>(columns * kKvDim);
    uint16_t* kc = allocate<uint16_t>(context * kKvDim);
    uint16_t* vc = allocate<uint16_t>(context * kKvDim);
    float* out = allocate<float>(count);
    uint32_t* position = allocate<uint32_t>(1);
    initialize_test<<<256,256>>>(q,k,v,kc,vc,count,columns*kKvDim,context*kKvDim);
    CHECK_CUDA(cudaGetLastError());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    std::vector<float> host(count);
    for (uint32_t tokens : {0u,1u,25u,31u,32u,255u,256u,1002u,8184u}) {
        CHECK_CUDA(cudaMemcpy(position,&tokens,sizeof(tokens),cudaMemcpyHostToDevice));
        auto launch = [&]() {
            noncausal_gqa_attention_bf16_wide_kernel<32u><<<
                kKvHeads*kGqaHeadGroups*columns,kGqaBlockThreads>>>(
                q,k,v,kc,vc,out,nullptr,nullptr,nullptr,tokens,columns,position);
            CHECK_CUDA(cudaGetLastError());
        };
        launch();
        CHECK_CUDA(cudaMemcpy(host.data(),out,count*sizeof(float),cudaMemcpyDeviceToHost));
        uint64_t hash = 1469598103934665603ull;
        for (float x : host) if (!std::isfinite(x)) return 2;
        const auto* bytes = reinterpret_cast<const unsigned char*>(host.data());
        for (size_t i=0; i<count*sizeof(float); ++i) { hash ^= bytes[i]; hash *= 1099511628211ull; }
        CHECK_CUDA(cudaEventRecord(start));
        for (int i=0;i<20;++i) launch();
        CHECK_CUDA(cudaEventRecord(stop)); CHECK_CUDA(cudaEventSynchronize(stop));
        float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
        std::printf("tokens=%u hash=%016llx us=%.3f\n",tokens,
                    static_cast<unsigned long long>(hash),ms*1000/20);
    }
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(q)); CHECK_CUDA(cudaFree(k)); CHECK_CUDA(cudaFree(v));
    CHECK_CUDA(cudaFree(kc)); CHECK_CUDA(cudaFree(vc)); CHECK_CUDA(cudaFree(out));
    CHECK_CUDA(cudaFree(position));
    return 0;
}
