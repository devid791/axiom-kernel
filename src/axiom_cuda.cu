#include "axiom/axiom.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/barrier>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

using axiom_cuda_block_barrier = cuda::barrier<cuda::thread_scope_block>;

struct axiom_cuda_runtime {
    int device;
    cudaStream_t moe_stream0;
    cudaStream_t moe_stream1;
    void *argmax_partials;
    void *argmax_result;
    uint32_t argmax_capacity;
    void *output_hc_pre;
    void *router_logits;
    uint32_t router_logits_capacity;
    void *q8_0_tile32;
    uint32_t q8_0_tile32_capacity;
    void *rmsnorm_inv;
    void *attention_vec_scratch;
    uint64_t attention_vec_capacity_floats;
};

struct axiom_cuda_buffer {
    int device;
    void *ptr;
    uint64_t bytes;
};

struct axiom_cuda_q8_k_block {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(axiom_cuda_q8_k_block) == AXIOM_Q8_K_BLOCK_BYTES, "Q8_K block ABI mismatch");

__constant__ static axiom_cuda_q8_k_block axiom_moe_midq_const[48];

struct axiom_cuda_q8_0_tile32_block {
    float d;
    int8_t qs[32];
};

struct axiom_cuda_latent_link {
    axiom_cuda_runtime *runtime;
    axiom_latent_link_kind kind;
    axiom_latent_dtype dtype;
    uint32_t source_width;
    uint32_t target_width;
    uint32_t hidden_width;
    float eps;
    float *pre_ln_weight;
    float *pre_ln_bias;
    float *proj1_weight;
    float *proj1_bias;
    float *proj2_weight;
    float *proj2_bias;
    float *residual_weight;
    float *residual_bias;
    float *post_ln_weight;
    float *post_ln_bias;
    float *scratch_norm;
    float *scratch_hidden;
    float *scratch_out;
    uint32_t scratch_rows;
};

static int axiom_cuda_status(cudaError_t err) {
    return err == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

static int axiom_cuda_probe_device_index(int device, axiom_device_info *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    if (device < 0 || device >= count) return AXIOM_ERR_INVALID_ARGUMENT;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->name, prop.name, sizeof(out->name) - 1);
    out->name[sizeof(out->name) - 1] = '\0';
    out->major = prop.major;
    out->minor = prop.minor;
    out->total_global_mem = (uint64_t)prop.totalGlobalMem;
    out->multi_processor_count = prop.multiProcessorCount;
    return AXIOM_OK;
}

static bool axiom_cuda_env_enabled(const char *name) {
    const char *v = std::getenv(name);
    return v && v[0] != '\0' && std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 && std::strcmp(v, "False") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

static bool axiom_cuda_moe_q8k_internal_timing_enabled() {
    static int cached = -1;
    if (cached < 0) cached = axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_INTERNAL_TIMING") ? 1 : 0;
    return cached != 0;
}

static double axiom_cuda_monotonic_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static FILE *axiom_cuda_moe_q8k_internal_log_file() {
    static FILE *fp = nullptr;
    static bool initialized = false;
    if (!initialized) {
        const char *path = std::getenv("AXIOM_DS4_MOE_Q8K_INTERNAL_LOG");
        initialized = true;
        if (path && path[0]) fp = std::fopen(path, "a");
        if (!fp) fp = stderr;
        std::setvbuf(fp, nullptr, _IOLBF, 0);
    }
    return fp;
}

static double axiom_cuda_moe_q8k_internal_start() {
    return axiom_cuda_moe_q8k_internal_timing_enabled() ? axiom_cuda_monotonic_sec() : 0.0;
}

static cudaError_t axiom_cuda_moe_q8k_internal_finish(
        const char *name,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden,
        double start,
        cudaError_t launch_err) {
    if (launch_err != cudaSuccess || start <= 0.0 ||
        !axiom_cuda_moe_q8k_internal_timing_enabled()) {
        return launch_err;
    }
    const cudaError_t sync_err = cudaDeviceSynchronize();
    FILE *fp = axiom_cuda_moe_q8k_internal_log_file();
    std::fprintf(fp,
            "{\"event\":\"moe_q8k_internal\",\"name\":\"%s\","
            "\"topk\":%u,\"hidden\":%u,\"expert_hidden\":%u,"
            "\"elapsed_ms\":%.6f,\"cuda_status\":%d}\n",
            name,
            topk,
            hidden,
            expert_hidden,
            (axiom_cuda_monotonic_sec() - start) * 1000.0,
            (int)sync_err);
    return sync_err;
}

static bool axiom_cuda_launch_async_enabled() {
    return axiom_cuda_env_enabled("AXIOM_CUDA_LAUNCH_ASYNC");
}

static bool axiom_cuda_default_stream_capturing() {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(cudaStreamPerThread, &status);
    if (err == cudaSuccess && status != cudaStreamCaptureStatusNone) return true;
    status = cudaStreamCaptureStatusNone;
    err = cudaStreamIsCapturing((cudaStream_t)0, &status);
    return err == cudaSuccess && status != cudaStreamCaptureStatusNone;
}

static int axiom_cuda_env_int(const char *name, int fallback, int min_value, int max_value, int multiple) {
    const char *v = std::getenv(name);
    if (!v || v[0] == '\0') return fallback;
    char *end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0') return fallback;
    if (parsed < min_value || parsed > max_value) return fallback;
    if (multiple > 1 && (parsed % multiple) != 0) return fallback;
    return (int)parsed;
}

static float axiom_cuda_env_float(const char *name, float fallback, float min_value, float max_value) {
    const char *v = std::getenv(name);
    if (!v || v[0] == '\0') return fallback;
    char *end = nullptr;
    float parsed = std::strtof(v, &end);
    if (end == v || *end != '\0') return fallback;
    if (parsed < min_value || parsed > max_value) return fallback;
    return parsed;
}

static int axiom_cuda_finish_after_launch(cudaError_t err) {
    if (err != cudaSuccess) return axiom_cuda_status(err);
    if (axiom_cuda_launch_async_enabled() || axiom_cuda_default_stream_capturing()) return AXIOM_OK;
    return axiom_cuda_status(cudaDeviceSynchronize());
}

static int axiom_cuda_moe_streams(
        axiom_cuda_runtime *runtime,
        cudaStream_t *stream0,
        cudaStream_t *stream1) {
    if (!runtime || !stream0 || !stream1) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!runtime->moe_stream0) {
        cudaError_t err = cudaStreamCreateWithFlags(&runtime->moe_stream0, cudaStreamNonBlocking);
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }
    if (!runtime->moe_stream1) {
        cudaError_t err = cudaStreamCreateWithFlags(&runtime->moe_stream1, cudaStreamNonBlocking);
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }
    *stream0 = runtime->moe_stream0;
    *stream1 = runtime->moe_stream1;
    return AXIOM_OK;
}

extern "C" int axiom_cuda_runtime_create(void **out, uint32_t device) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = nullptr;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    if ((int)device < 0 || (int)device >= count) return AXIOM_ERR_INVALID_ARGUMENT;

    err = cudaSetDevice((int)device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    axiom_cuda_runtime *runtime =
            (axiom_cuda_runtime *)std::calloc(1, sizeof(*runtime));
    if (!runtime) return AXIOM_ERR_RUNTIME;
    runtime->device = (int)device;
    *out = runtime;
    return AXIOM_OK;
}

extern "C" int axiom_cuda_device_count(uint32_t *out_count) {
    if (!out_count) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_count = 0;
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    if (count < 0) return AXIOM_ERR_CUDA;
    *out_count = (uint32_t)count;
    return AXIOM_OK;
}

extern "C" int axiom_cuda_device_probe(uint32_t device, axiom_device_info *out) {
    return axiom_cuda_probe_device_index((int)device, out);
}

extern "C" void axiom_cuda_runtime_destroy(void *cuda_runtime) {
    if (!cuda_runtime) return;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaSetDevice(runtime->device);
    cudaDeviceSynchronize();
    if (runtime->moe_stream0) cudaStreamDestroy(runtime->moe_stream0);
    if (runtime->moe_stream1) cudaStreamDestroy(runtime->moe_stream1);
    cudaFree(runtime->argmax_partials);
    cudaFree(runtime->argmax_result);
    cudaFree(runtime->output_hc_pre);
    cudaFree(runtime->router_logits);
    cudaFree(runtime->q8_0_tile32);
    cudaFree(runtime->rmsnorm_inv);
    cudaFree(runtime->attention_vec_scratch);
    if (axiom_cuda_env_enabled("AXIOM_DEV_RESET")) {
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
    std::free(cuda_runtime);
}

extern "C" int axiom_cuda_runtime_probe(void *cuda_runtime, axiom_device_info *out) {
    if (!cuda_runtime || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    return axiom_cuda_probe_device_index(runtime->device, out);
}

extern "C" int axiom_cuda_device_buffer_create(
        void *cuda_runtime,
        void **out,
        uint64_t bytes) {
    if (!cuda_runtime || !out || bytes == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = nullptr;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_cuda_buffer *buffer = (axiom_cuda_buffer *)std::calloc(1, sizeof(*buffer));
    if (!buffer) return AXIOM_ERR_RUNTIME;
    buffer->device = runtime->device;
    buffer->bytes = bytes;
    err = cudaMalloc(&buffer->ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        std::free(buffer);
        return axiom_cuda_status(err);
    }
    *out = buffer;
    return AXIOM_OK;
}

extern "C" void axiom_cuda_device_buffer_destroy(void *opaque) {
    if (!opaque) return;
    axiom_cuda_buffer *buffer = (axiom_cuda_buffer *)opaque;
    cudaSetDevice(buffer->device);
    cudaFree(buffer->ptr);
    std::free(buffer);
}

extern "C" int axiom_cuda_device_buffer_upload(
        void *opaque,
        uint64_t offset,
        const void *src_host,
        uint64_t bytes) {
    if (!opaque || !src_host || bytes == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_buffer *buffer = (axiom_cuda_buffer *)opaque;
    if (offset > buffer->bytes || bytes > buffer->bytes - offset) return AXIOM_ERR_INVALID_ARGUMENT;
    cudaError_t err = cudaSetDevice(buffer->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    err = cudaMemcpy((uint8_t *)buffer->ptr + offset, src_host, (size_t)bytes, cudaMemcpyHostToDevice);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_device_buffer_download(
        void *opaque,
        uint64_t offset,
        void *dst_host,
        uint64_t bytes) {
    if (!opaque || !dst_host || bytes == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_buffer *buffer = (axiom_cuda_buffer *)opaque;
    if (offset > buffer->bytes || bytes > buffer->bytes - offset) return AXIOM_ERR_INVALID_ARGUMENT;
    cudaError_t err = cudaSetDevice(buffer->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    err = cudaMemcpy(dst_host, (const uint8_t *)buffer->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_device_buffer_copy(
        void *dst_opaque,
        uint64_t dst_offset,
        const void *src_opaque,
        uint64_t src_offset,
        uint64_t bytes) {
    if (!dst_opaque || !src_opaque || bytes == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_buffer *dst = (axiom_cuda_buffer *)dst_opaque;
    const axiom_cuda_buffer *src = (const axiom_cuda_buffer *)src_opaque;
    if (dst_offset > dst->bytes || bytes > dst->bytes - dst_offset ||
        src_offset > src->bytes || bytes > src->bytes - src_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (dst->device != src->device) {
        int can_access = 0;
        cudaError_t err = cudaDeviceCanAccessPeer(&can_access, dst->device, src->device);
        if (err == cudaSuccess && can_access) {
            err = cudaSetDevice(dst->device);
            if (err != cudaSuccess) return AXIOM_ERR_CUDA;
            err = cudaDeviceEnablePeerAccess(src->device, 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                (void)cudaGetLastError();
                err = cudaSuccess;
            }
            if (err == cudaSuccess) {
                err = cudaMemcpyPeer(
                        (uint8_t *)dst->ptr + dst_offset,
                        dst->device,
                        (const uint8_t *)src->ptr + src_offset,
                        src->device,
                        (size_t)bytes);
                if (err == cudaSuccess) return AXIOM_OK;
            }
        }

        const uint64_t preferred_chunk = 64ull * 1024ull * 1024ull;
        const uint64_t chunk_bytes_u64 = bytes < preferred_chunk ? bytes : preferred_chunk;
        const size_t chunk_bytes = (size_t)chunk_bytes_u64;
        void *host = nullptr;
        bool pinned = false;
        err = cudaMallocHost(&host, chunk_bytes);
        if (err == cudaSuccess) {
            pinned = true;
        } else {
            (void)cudaGetLastError();
            host = std::malloc(chunk_bytes);
            err = host ? cudaSuccess : cudaErrorMemoryAllocation;
        }
        if (!host) return AXIOM_ERR_RUNTIME;

        uint64_t copied = 0;
        while (err == cudaSuccess && copied < bytes) {
            const uint64_t remaining = bytes - copied;
            const size_t chunk = (size_t)(remaining < chunk_bytes_u64 ? remaining : chunk_bytes_u64);
            err = cudaSetDevice(src->device);
            if (err == cudaSuccess) {
                err = cudaMemcpy(
                        host,
                        (const uint8_t *)src->ptr + src_offset + copied,
                        chunk,
                        cudaMemcpyDeviceToHost);
            }
            if (err == cudaSuccess) {
                err = cudaSetDevice(dst->device);
            }
            if (err == cudaSuccess) {
                err = cudaMemcpy(
                        (uint8_t *)dst->ptr + dst_offset + copied,
                        host,
                        chunk,
                        cudaMemcpyHostToDevice);
            }
            copied += (uint64_t)chunk;
        }
        if (pinned) {
            cudaFreeHost(host);
        } else {
            std::free(host);
        }
        return axiom_cuda_status(err);
    }

    cudaError_t err = cudaSetDevice(dst->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    if (axiom_cuda_launch_async_enabled()) {
        err = cudaMemcpyAsync(
                (uint8_t *)dst->ptr + dst_offset,
                (const uint8_t *)src->ptr + src_offset,
                (size_t)bytes,
                cudaMemcpyDeviceToDevice,
                0);
    } else {
        err = cudaMemcpy(
                (uint8_t *)dst->ptr + dst_offset,
                (const uint8_t *)src->ptr + src_offset,
                (size_t)bytes,
                cudaMemcpyDeviceToDevice);
    }
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_device_buffer_device(const void *opaque, uint32_t *out_device) {
    if (!opaque || !out_device) return AXIOM_ERR_INVALID_ARGUMENT;
    const axiom_cuda_buffer *buffer = (const axiom_cuda_buffer *)opaque;
    *out_device = (uint32_t)buffer->device;
    return AXIOM_OK;
}

extern "C" int axiom_cuda_device_buffer_pointer(const void *opaque, void **out_pointer) {
    if (out_pointer) *out_pointer = nullptr;
    if (!opaque || !out_pointer) return AXIOM_ERR_INVALID_ARGUMENT;
    const axiom_cuda_buffer *buffer = (const axiom_cuda_buffer *)opaque;
    if (!buffer->ptr) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_pointer = buffer->ptr;
    return AXIOM_OK;
}

typedef struct {
    float value;
    uint32_t index;
} axiom_cuda_argmax_pair;

static int axiom_cuda_argmax_scratch(
        axiom_cuda_runtime *runtime,
        uint32_t capacity,
        axiom_cuda_argmax_pair **partials,
        axiom_cuda_argmax_pair **result) {
    if (!runtime || !partials || !result || capacity == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    if (runtime->argmax_capacity < capacity ||
        !runtime->argmax_partials || !runtime->argmax_result) {
        void *new_partials = nullptr;
        void *new_result = nullptr;
        cudaError_t err = cudaMalloc(&new_partials, (size_t)capacity * sizeof(axiom_cuda_argmax_pair));
        if (err != cudaSuccess) return axiom_cuda_status(err);
        err = cudaMalloc(&new_result, sizeof(axiom_cuda_argmax_pair));
        if (err != cudaSuccess) {
            cudaFree(new_partials);
            return axiom_cuda_status(err);
        }
        cudaFree(runtime->argmax_partials);
        cudaFree(runtime->argmax_result);
        runtime->argmax_partials = new_partials;
        runtime->argmax_result = new_result;
        runtime->argmax_capacity = capacity;
    }
    *partials = (axiom_cuda_argmax_pair *)runtime->argmax_partials;
    *result = (axiom_cuda_argmax_pair *)runtime->argmax_result;
    return AXIOM_OK;
}

static int axiom_cuda_router_logits_scratch(
        axiom_cuda_runtime *runtime,
        uint32_t capacity,
        float **logits) {
    if (!runtime || !logits || capacity == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    if (runtime->router_logits_capacity < capacity || !runtime->router_logits) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        void *new_logits = nullptr;
        err = cudaMalloc(&new_logits, (size_t)capacity * sizeof(float));
        if (err != cudaSuccess) return axiom_cuda_status(err);
        cudaFree(runtime->router_logits);
        runtime->router_logits = new_logits;
        runtime->router_logits_capacity = capacity;
    }
    *logits = (float *)runtime->router_logits;
    return AXIOM_OK;
}

static int axiom_cuda_q8_0_tile32_scratch(
        axiom_cuda_runtime *runtime,
        uint32_t capacity,
        axiom_cuda_q8_0_tile32_block **blocks) {
    if (!runtime || !blocks || capacity == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    if (runtime->q8_0_tile32_capacity < capacity || !runtime->q8_0_tile32) {
        void *new_blocks = nullptr;
        cudaError_t err = cudaMalloc(&new_blocks, (size_t)capacity * sizeof(axiom_cuda_q8_0_tile32_block));
        if (err != cudaSuccess) return axiom_cuda_status(err);
        cudaFree(runtime->q8_0_tile32);
        runtime->q8_0_tile32 = new_blocks;
        runtime->q8_0_tile32_capacity = capacity;
    }
    *blocks = (axiom_cuda_q8_0_tile32_block *)runtime->q8_0_tile32;
    return AXIOM_OK;
}

static int axiom_cuda_rmsnorm_inv_scratch(
        axiom_cuda_runtime *runtime,
        float **inv) {
    if (!runtime || !inv) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!runtime->rmsnorm_inv) {
        cudaError_t err = cudaMalloc(&runtime->rmsnorm_inv, sizeof(float));
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }
    *inv = (float *)runtime->rmsnorm_inv;
    return AXIOM_OK;
}

static int axiom_cuda_attention_vec_scratch(
        axiom_cuda_runtime *runtime,
        uint64_t partial_count,
        uint32_t head_dim,
        float **partial_m,
        float **partial_l,
        float **partial_acc) {
    if (!runtime || !partial_m || !partial_l || !partial_acc ||
        partial_count == 0 || head_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t capacity = partial_count * ((uint64_t)head_dim + 2u);
    if (capacity < partial_count) return AXIOM_ERR_INVALID_ARGUMENT;
    if (runtime->attention_vec_capacity_floats < capacity || !runtime->attention_vec_scratch) {
        if (axiom_cuda_default_stream_capturing()) return AXIOM_ERR_RUNTIME;
        void *new_scratch = nullptr;
        cudaError_t err = cudaMalloc(&new_scratch, (size_t)capacity * sizeof(float));
        if (err != cudaSuccess) return axiom_cuda_status(err);
        cudaFree(runtime->attention_vec_scratch);
        runtime->attention_vec_scratch = new_scratch;
        runtime->attention_vec_capacity_floats = capacity;
    }
    float *base = (float *)runtime->attention_vec_scratch;
    *partial_m = base;
    *partial_l = base + partial_count;
    *partial_acc = base + partial_count * 2u;
    return AXIOM_OK;
}

__device__ static axiom_cuda_argmax_pair axiom_argmax_better(
        axiom_cuda_argmax_pair a,
        axiom_cuda_argmax_pair b) {
    const int a_ok = !isnan(a.value);
    const int b_ok = !isnan(b.value);
    if (!a_ok) return b;
    if (!b_ok) return a;
    if (b.value > a.value) return b;
    if (b.value == a.value && b.index < a.index) return b;
    return a;
}

__global__ static void axiom_f32_argmax_stage1_kernel(
        const float *input,
        uint32_t count,
        axiom_cuda_argmax_pair *partials) {
    extern __shared__ axiom_cuda_argmax_pair shared[];
    axiom_cuda_argmax_pair best;
    best.value = -INFINITY;
    best.index = UINT32_MAX;
    const uint32_t tid = threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + tid; i < count; i += stride) {
        axiom_cuda_argmax_pair cur;
        cur.value = input[i];
        cur.index = i;
        best = axiom_argmax_better(best, cur);
    }
    shared[tid] = best;
    __syncthreads();
    for (uint32_t step = blockDim.x >> 1u; step > 0u; step >>= 1u) {
        if (tid < step) shared[tid] = axiom_argmax_better(shared[tid], shared[tid + step]);
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = shared[0];
}

__global__ static void axiom_f32_argmax_stage2_kernel(
        const axiom_cuda_argmax_pair *partials,
        uint32_t count,
        axiom_cuda_argmax_pair *out) {
    extern __shared__ axiom_cuda_argmax_pair shared[];
    const uint32_t tid = threadIdx.x;
    axiom_cuda_argmax_pair best;
    best.value = -INFINITY;
    best.index = UINT32_MAX;
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        best = axiom_argmax_better(best, partials[i]);
    }
    shared[tid] = best;
    __syncthreads();
    for (uint32_t width = blockDim.x; width > 1u; ) {
        const uint32_t half = (width + 1u) >> 1u;
        if (tid < width - half) shared[tid] = axiom_argmax_better(shared[tid], shared[tid + half]);
        __syncthreads();
        width = half;
    }
    if (tid == 0) out[0] = shared[0];
}

__global__ static void axiom_argmax_pair_index_to_u32_kernel(
        const axiom_cuda_argmax_pair *result,
        uint32_t *out_token_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out_token_id[0] = result[0].index;
}

extern "C" int axiom_cuda_f32_argmax_device(
        void *cuda_runtime,
        const void *input,
        uint64_t input_offset,
        uint32_t count,
        uint32_t *out_index,
        float *out_value) {
    if (!cuda_runtime || !input || !out_index || !out_value || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    if (input_offset > ibuf->bytes ||
        (uint64_t)count * sizeof(float) > ibuf->bytes - input_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint32_t block = 256u;
    uint32_t grid = (count + block - 1u) / block;
    if (grid == 0) grid = 1;
    if (grid > 512u) grid = 512u;
    axiom_cuda_argmax_pair *partials = nullptr;
    axiom_cuda_argmax_pair *result = nullptr;
    int rc = axiom_cuda_argmax_scratch(runtime, grid, &partials, &result);
    if (rc != AXIOM_OK) return rc;
    const size_t shared_bytes = (size_t)block * sizeof(axiom_cuda_argmax_pair);
    axiom_f32_argmax_stage1_kernel<<<grid, block, shared_bytes>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            count,
            partials);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        axiom_f32_argmax_stage2_kernel<<<1, block, shared_bytes>>>(partials, grid, result);
        err = cudaGetLastError();
    }
    axiom_cuda_argmax_pair host;
    if (err == cudaSuccess) {
        err = cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) return axiom_cuda_status(err);
    if (host.index == UINT32_MAX) return AXIOM_ERR_RUNTIME;
    *out_index = host.index;
    *out_value = host.value;
    return AXIOM_OK;
}

extern "C" int axiom_cuda_f32_argmax_to_buffer_device(
        void *cuda_runtime,
        const void *input,
        uint64_t input_offset,
        uint32_t count,
        void *out_token_id,
        uint64_t out_token_id_offset) {
    if (!cuda_runtime || !input || !out_token_id || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out_token_id;
    if (input_offset > ibuf->bytes ||
        (uint64_t)count * sizeof(float) > ibuf->bytes - input_offset ||
        out_token_id_offset > obuf->bytes ||
        sizeof(uint32_t) > obuf->bytes - out_token_id_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint32_t block = 256u;
    uint32_t grid = (count + block - 1u) / block;
    if (grid == 0) grid = 1;
    if (grid > 512u) grid = 512u;
    axiom_cuda_argmax_pair *partials = nullptr;
    axiom_cuda_argmax_pair *result = nullptr;
    int rc = axiom_cuda_argmax_scratch(runtime, grid, &partials, &result);
    if (rc != AXIOM_OK) return rc;
    const size_t shared_bytes = (size_t)block * sizeof(axiom_cuda_argmax_pair);
    axiom_f32_argmax_stage1_kernel<<<grid, block, shared_bytes>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            count,
            partials);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        axiom_f32_argmax_stage2_kernel<<<1, block, shared_bytes>>>(partials, grid, result);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess) {
        axiom_argmax_pair_index_to_u32_kernel<<<1, 1>>>(
                result,
                (uint32_t *)((uint8_t *)obuf->ptr + out_token_id_offset));
        err = cudaGetLastError();
    }
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_vector_add_kernel(
        const float *a,
        const float *b,
        float *out,
        size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] + b[i];
}

__global__ static void axiom_bf16_to_f32_kernel(
        const uint16_t *in,
        float *out,
        size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const uint32_t bits = ((uint32_t)in[i]) << 16u;
        out[i] = __uint_as_float(bits);
    }
}

__device__ __forceinline__ static uint16_t axiom_dev_le16(const uint8_t *p) {
    return *(const uint16_t *)p;
}

__device__ __forceinline__ static uint32_t axiom_dev_le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

__device__ __forceinline__ static float axiom_dev_f16_to_f32(uint16_t h) {
    __half_raw raw;
    raw.x = h;
    return __half2float(__half(raw));
}

__device__ __constant__ static const uint16_t axiom_iq2xxs_kgrid[256] = {
        0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
      100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
     1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
     1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
     2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
     4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
     5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
     8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
};

__device__ __constant__ static const uint64_t axiom_iq2xxs_grid64[256] = {
    0x0808080808080808ull, 0x080808080808082bull, 0x0808080808081919ull, 0x0808080808082b08ull, 0x0808080808082b2bull, 0x0808080808190819ull, 0x0808080808191908ull, 0x08080808082b0808ull, 0x08080808082b082bull, 0x08080808082b2b08ull, 0x08080808082b2b2bull, 0x0808080819080819ull, 0x0808080819081908ull, 0x0808080819190808ull, 0x0808080819192b08ull, 0x08080808192b0819ull,
    0x08080808192b1908ull, 0x080808082b080808ull, 0x080808082b08082bull, 0x080808082b082b2bull, 0x080808082b2b082bull, 0x0808081908080819ull, 0x0808081908081908ull, 0x0808081908190808ull, 0x0808081908191919ull, 0x0808081919080808ull, 0x080808192b081908ull, 0x080808192b192b08ull, 0x0808082b08080808ull, 0x0808082b0808082bull, 0x0808082b082b082bull, 0x0808082b2b08082bull,
    0x0808190808080819ull, 0x0808190808081908ull, 0x0808190808190808ull, 0x08081908082b0819ull, 0x08081908082b1908ull, 0x0808190819080808ull, 0x080819081908082bull, 0x0808190819082b08ull, 0x08081908192b0808ull, 0x080819082b080819ull, 0x080819082b081908ull, 0x080819082b190808ull, 0x080819082b2b1908ull, 0x0808191908080808ull, 0x080819190808082bull, 0x0808191908082b08ull,
    0x08081919082b0808ull, 0x080819191908192bull, 0x08081919192b2b19ull, 0x080819192b080808ull, 0x080819192b190819ull, 0x0808192b08082b19ull, 0x0808192b08190808ull, 0x0808192b19080808ull, 0x0808192b2b081908ull, 0x0808192b2b2b1908ull, 0x08082b0808080808ull, 0x08082b0808081919ull, 0x08082b0808082b08ull, 0x08082b0808191908ull, 0x08082b08082b2b08ull, 0x08082b0819080819ull,
    0x08082b0819081908ull, 0x08082b0819190808ull, 0x08082b081919082bull, 0x08082b082b082b08ull, 0x08082b1908081908ull, 0x08082b1919080808ull, 0x08082b2b0808082bull, 0x08082b2b08191908ull, 0x0819080808080819ull, 0x0819080808081908ull, 0x0819080808190808ull, 0x08190808082b0819ull, 0x0819080819080808ull, 0x08190808192b0808ull, 0x081908082b081908ull, 0x081908082b190808ull,
    0x081908082b191919ull, 0x0819081908080808ull, 0x0819081908082b08ull, 0x08190819082b0808ull, 0x0819081919190808ull, 0x0819081919192b2bull, 0x081908192b080808ull, 0x0819082b082b1908ull, 0x0819082b19081919ull, 0x0819190808080808ull, 0x0819190808082b08ull, 0x08191908082b0808ull, 0x08191908082b1919ull, 0x0819190819082b19ull, 0x081919082b080808ull, 0x0819191908192b08ull,
    0x08191919192b082bull, 0x0819192b08080808ull, 0x0819192b0819192bull, 0x08192b0808080819ull, 0x08192b0808081908ull, 0x08192b0808190808ull, 0x08192b0819080808ull, 0x08192b082b080819ull, 0x08192b1908080808ull, 0x08192b1908081919ull, 0x08192b192b2b0808ull, 0x08192b2b19190819ull, 0x082b080808080808ull, 0x082b08080808082bull, 0x082b080808082b2bull, 0x082b080819081908ull,
    0x082b0808192b0819ull, 0x082b08082b080808ull, 0x082b08082b08082bull, 0x082b0819082b2b19ull, 0x082b081919082b08ull, 0x082b082b08080808ull, 0x082b082b0808082bull, 0x082b190808080819ull, 0x082b190808081908ull, 0x082b190808190808ull, 0x082b190819080808ull, 0x082b19081919192bull, 0x082b191908080808ull, 0x082b191919080819ull, 0x082b1919192b1908ull, 0x082b192b2b190808ull,
    0x082b2b0808082b08ull, 0x082b2b08082b0808ull, 0x082b2b082b191908ull, 0x082b2b2b19081908ull, 0x1908080808080819ull, 0x1908080808081908ull, 0x1908080808190808ull, 0x1908080808192b08ull, 0x19080808082b0819ull, 0x19080808082b1908ull, 0x1908080819080808ull, 0x1908080819082b08ull, 0x190808081919192bull, 0x19080808192b0808ull, 0x190808082b080819ull, 0x190808082b081908ull,
    0x190808082b190808ull, 0x1908081908080808ull, 0x19080819082b0808ull, 0x19080819192b0819ull, 0x190808192b080808ull, 0x190808192b081919ull, 0x1908082b08080819ull, 0x1908082b08190808ull, 0x1908082b19082b08ull, 0x1908082b1919192bull, 0x1908082b192b2b08ull, 0x1908190808080808ull, 0x1908190808082b08ull, 0x19081908082b0808ull, 0x190819082b080808ull, 0x190819082b192b19ull,
    0x190819190819082bull, 0x19081919082b1908ull, 0x1908192b08080808ull, 0x19082b0808080819ull, 0x19082b0808081908ull, 0x19082b0808190808ull, 0x19082b0819080808ull, 0x19082b0819081919ull, 0x19082b1908080808ull, 0x19082b1919192b08ull, 0x19082b19192b0819ull, 0x19082b192b08082bull, 0x19082b2b19081919ull, 0x19082b2b2b190808ull, 0x1919080808080808ull, 0x1919080808082b08ull,
    0x1919080808190819ull, 0x1919080808192b19ull, 0x19190808082b0808ull, 0x191908082b080808ull, 0x191908082b082b08ull, 0x1919081908081908ull, 0x191908191908082bull, 0x191908192b2b1908ull, 0x1919082b2b190819ull, 0x191919082b190808ull, 0x191919082b19082bull, 0x1919191908082b2bull, 0x1919192b08080819ull, 0x1919192b19191908ull, 0x19192b0808080808ull, 0x19192b0808190819ull,
    0x19192b0808192b19ull, 0x19192b08192b1908ull, 0x19192b1919080808ull, 0x19192b2b08082b08ull, 0x192b080808081908ull, 0x192b080808190808ull, 0x192b080819080808ull, 0x192b0808192b2b08ull, 0x192b081908080808ull, 0x192b081919191919ull, 0x192b082b08192b08ull, 0x192b082b192b0808ull, 0x192b190808080808ull, 0x192b190808081919ull, 0x192b191908190808ull, 0x192b19190819082bull,
    0x192b19192b081908ull, 0x192b2b081908082bull, 0x2b08080808080808ull, 0x2b0808080808082bull, 0x2b08080808082b2bull, 0x2b08080819080819ull, 0x2b0808082b08082bull, 0x2b08081908081908ull, 0x2b08081908192b08ull, 0x2b08081919080808ull, 0x2b08082b08190819ull, 0x2b08190808080819ull, 0x2b08190808081908ull, 0x2b08190808190808ull, 0x2b08190808191919ull, 0x2b08190819080808ull,
    0x2b081908192b0808ull, 0x2b08191908080808ull, 0x2b0819191908192bull, 0x2b0819192b191908ull, 0x2b08192b08082b19ull, 0x2b08192b19080808ull, 0x2b08192b192b0808ull, 0x2b082b080808082bull, 0x2b082b1908081908ull, 0x2b082b2b08190819ull, 0x2b19080808081908ull, 0x2b19080808190808ull, 0x2b190808082b1908ull, 0x2b19080819080808ull, 0x2b1908082b2b0819ull, 0x2b1908190819192bull,
    0x2b1908192b080808ull, 0x2b19082b19081919ull, 0x2b19190808080808ull, 0x2b191908082b082bull, 0x2b19190819081908ull, 0x2b19191919190819ull, 0x2b192b082b080819ull, 0x2b192b19082b0808ull, 0x2b2b08080808082bull, 0x2b2b080819190808ull, 0x2b2b08082b081919ull, 0x2b2b081908082b19ull, 0x2b2b082b08080808ull, 0x2b2b190808192b08ull, 0x2b2b2b0819190808ull, 0x2b2b2b1908081908ull,
};

__device__ __constant__ static const uint8_t axiom_iq2xxs_sign_mask[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

__device__ __forceinline__ static float axiom_dev_iq2xxs_grid_value(uint32_t grid_idx, uint32_t lane) {
    const uint32_t v = ((uint32_t)axiom_iq2xxs_kgrid[grid_idx] >> (2u * lane)) & 3u;
    return (float)(8u + 17u * v + (v >> 1u));
}

__device__ __forceinline__ static int32_t axiom_dev_iq2xxs_grid_i8(uint32_t grid_idx, uint32_t lane) {
    const uint32_t v = ((uint32_t)axiom_iq2xxs_kgrid[grid_idx] >> (2u * lane)) & 3u;
    return (int32_t)(8u + 17u * v + (v >> 1u));
}

__global__ static void axiom_q8_k_pack_f32_kernel(
        const float *input,
        axiom_cuda_q8_k_block *out,
        uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t lane = threadIdx.x;
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float iscale;
    const float x = input[(uint64_t)b * 256u + lane];
    absmax[lane] = fabsf(x);
    maxv[lane] = x;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && absmax[lane + stride] > absmax[lane]) {
            absmax[lane] = absmax[lane + stride];
            maxv[lane] = maxv[lane + stride];
        }
        __syncthreads();
    }
    if (absmax[0] == 0.0f) {
        out[b].qs[lane] = 0;
        if (lane < 16u) out[b].bsums[lane] = 0;
        if (lane == 0u) out[b].d = 0.0f;
        return;
    }
    if (lane == 0u) iscale = -127.0f / maxv[0];
    __syncthreads();
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    out[b].qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += out[b].qs[lane * 16u + i];
        out[b].bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) out[b].d = 1.0f / iscale;
}

__global__ static void axiom_q8_k_pack_f32_fast_kernel(
        const float *__restrict__ input,
        axiom_cuda_q8_k_block *__restrict__ out,
        uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    __shared__ float warp_abs[8];
    __shared__ float warp_val[8];
    __shared__ float iscale;

    const float x = input[(uint64_t)b * 256u + tid];
    float best_abs = fabsf(x);
    float best_val = x;
    #pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
        const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
        if (other_abs > best_abs) {
            best_abs = other_abs;
            best_val = other_val;
        }
    }
    if (lane == 0u) {
        warp_abs[warp] = best_abs;
        warp_val[warp] = best_val;
    }
    __syncthreads();

    if (warp == 0u) {
        best_abs = lane < 8u ? warp_abs[lane] : -1.0f;
        best_val = lane < 8u ? warp_val[lane] : 0.0f;
        #pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
            const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
            if (other_abs > best_abs) {
                best_abs = other_abs;
                best_val = other_val;
            }
        }
        if (lane == 0u) {
            if (best_abs == 0.0f) {
                iscale = 0.0f;
                out[b].d = 0.0f;
            } else {
                iscale = -127.0f / best_val;
                out[b].d = 1.0f / iscale;
            }
        }
    }
    __syncthreads();

    if (iscale == 0.0f) {
        out[b].qs[tid] = 0;
    } else {
        int q = (int)lrintf(iscale * x);
        q = q < -128 ? -128 : q > 127 ? 127 : q;
        out[b].qs[tid] = (int8_t)q;
    }
    __syncthreads();

    if (tid < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += out[b].qs[tid * 16u + i];
        out[b].bsums[tid] = (int16_t)sum;
    }
}

__global__ static void axiom_q8_k_pack_f32_rn_kernel(
        const float *input,
        axiom_cuda_q8_k_block *out,
        uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t lane = threadIdx.x;
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float iscale;
    const float x = input[(uint64_t)b * 256u + lane];
    absmax[lane] = fabsf(x);
    maxv[lane] = x;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && absmax[lane + stride] > absmax[lane]) {
            absmax[lane] = absmax[lane + stride];
            maxv[lane] = maxv[lane + stride];
        }
        __syncthreads();
    }
    if (absmax[0] == 0.0f) {
        out[b].qs[lane] = 0;
        if (lane < 16u) out[b].bsums[lane] = 0;
        if (lane == 0u) out[b].d = 0.0f;
        return;
    }
    if (lane == 0u) iscale = -127.0f / maxv[0];
    __syncthreads();
    int q = __float2int_rn(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    out[b].qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += out[b].qs[lane * 16u + i];
        out[b].bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) out[b].d = 1.0f / iscale;
}

__global__ static void axiom_q8_k_pack_f32_with_inv_weight_rn_kernel(
        const float *__restrict__ weight,
        const float *__restrict__ input,
        const float *__restrict__ inv,
        axiom_cuda_q8_k_block *__restrict__ out,
        uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t tid = threadIdx.x;
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float iscale;
    const uint32_t idx = b * 256u + tid;
    const float y = input[idx] * inv[0] * weight[idx];
    absmax[tid] = fabsf(y);
    maxv[tid] = y;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride && absmax[tid + stride] > absmax[tid]) {
            absmax[tid] = absmax[tid + stride];
            maxv[tid] = maxv[tid + stride];
        }
        __syncthreads();
    }
    if (absmax[0] == 0.0f) {
        out[b].qs[tid] = 0;
        if (tid < 16u) out[b].bsums[tid] = 0;
        if (tid == 0u) out[b].d = 0.0f;
        return;
    }
    if (tid == 0u) iscale = -127.0f / maxv[0];
    __syncthreads();
    int q = __float2int_rn(iscale * y);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    out[b].qs[tid] = (int8_t)q;
    __syncthreads();
    if (tid < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += out[b].qs[tid * 16u + i];
        out[b].bsums[tid] = (int16_t)sum;
    }
    if (tid == 0u) out[b].d = 1.0f / iscale;
}

__global__ static void axiom_q8_k_pack_f32_input_nobsums_kernel(
        const float *input,
        axiom_cuda_q8_k_block *out,
        uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t lane = threadIdx.x;
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float iscale;
    const float x = input[(uint64_t)b * 256u + lane];
    absmax[lane] = fabsf(x);
    maxv[lane] = x;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && absmax[lane + stride] > absmax[lane]) {
            absmax[lane] = absmax[lane + stride];
            maxv[lane] = maxv[lane + stride];
        }
        __syncthreads();
    }
    if (absmax[0] == 0.0f) {
        out[b].qs[lane] = 0;
        if (lane == 0u) out[b].d = 0.0f;
        return;
    }
    if (lane == 0u) iscale = -127.0f / maxv[0];
    __syncthreads();
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    out[b].qs[lane] = (int8_t)q;
    if (lane == 0u) out[b].d = 1.0f / iscale;
}

__global__ static void axiom_q8_k_pack_f32_input_nobsums_fast_kernel(
        const float *__restrict__ input,
        axiom_cuda_q8_k_block *__restrict__ out,
        uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    __shared__ float warp_abs[8];
    __shared__ float warp_val[8];
    __shared__ float iscale;

    const float x = input[(uint64_t)b * 256u + tid];
    float best_abs = fabsf(x);
    float best_val = x;
    #pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
        const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
        if (other_abs > best_abs) {
            best_abs = other_abs;
            best_val = other_val;
        }
    }
    if (lane == 0u) {
        warp_abs[warp] = best_abs;
        warp_val[warp] = best_val;
    }
    __syncthreads();

    if (warp == 0u) {
        best_abs = lane < 8u ? warp_abs[lane] : -1.0f;
        best_val = lane < 8u ? warp_val[lane] : 0.0f;
        #pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
            const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
            if (other_abs > best_abs) {
                best_abs = other_abs;
                best_val = other_val;
            }
        }
        if (lane == 0u) {
            if (best_abs == 0.0f) {
                iscale = 0.0f;
                out[b].d = 0.0f;
            } else {
                iscale = -127.0f / best_val;
                out[b].d = 1.0f / iscale;
            }
        }
    }
    __syncthreads();

    if (iscale == 0.0f) {
        out[b].qs[tid] = 0;
    } else {
        int q = (int)lrintf(iscale * x);
        q = q < -128 ? -128 : q > 127 ? 127 : q;
        out[b].qs[tid] = (int8_t)q;
    }
}

__global__ static void axiom_q8_k_pack_f32_with_d_kernel(
        const float *__restrict__ input,
        axiom_cuda_q8_k_block *__restrict__ out,
        uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t tid = threadIdx.x;
    const float d = out[b].d;
    if (d == 0.0f) {
        out[b].qs[tid] = 0;
    } else {
        const float iscale = 1.0f / d;
        const float x = input[(uint64_t)b * 256u + tid];
        int q = (int)lrintf(iscale * x);
        q = q < -128 ? -128 : q > 127 ? 127 : q;
        out[b].qs[tid] = (int8_t)q;
    }
    __syncthreads();
    if (tid < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += out[b].qs[tid * 16u + i];
        out[b].bsums[tid] = (int16_t)sum;
    }
}

__global__ static void axiom_q8_k_pack_rows_f32_kernel(
        axiom_cuda_q8_k_block *__restrict__ out,
        const float *__restrict__ input,
        uint32_t in_dim,
        uint32_t rows) {
    const uint32_t b = (uint32_t)blockIdx.x;
    const uint32_t row = (uint32_t)blockIdx.y;
    if (row >= rows || b >= in_dim / 256u) return;
    const uint32_t lane = threadIdx.x;
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float iscale;
    const float *src = input + (uint64_t)row * in_dim + (uint64_t)b * 256u;
    const float x = src[lane];
    absmax[lane] = fabsf(x);
    maxv[lane] = x;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && absmax[lane + stride] > absmax[lane]) {
            absmax[lane] = absmax[lane + stride];
            maxv[lane] = maxv[lane + stride];
        }
        __syncthreads();
    }
    axiom_cuda_q8_k_block *dst = out + (uint64_t)row * (in_dim / 256u) + b;
    if (absmax[0] == 0.0f) {
        dst->qs[lane] = 0;
        if (lane < 16u) dst->bsums[lane] = 0;
        if (lane == 0u) dst->d = 0.0f;
        return;
    }
    if (lane == 0u) iscale = -127.0f / maxv[0];
    __syncthreads();
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    dst->qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += dst->qs[lane * 16u + i];
        dst->bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) dst->d = 1.0f / iscale;
}

__global__ static void axiom_q8_k_pack_rows_f32_fast_kernel(
        axiom_cuda_q8_k_block *__restrict__ out,
        const float *__restrict__ input,
        uint32_t in_dim,
        uint32_t rows) {
    const uint32_t b = (uint32_t)blockIdx.x;
    const uint32_t row = (uint32_t)blockIdx.y;
    if (row >= rows || b >= in_dim / 256u) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t warp = lane >> 5u;
    const uint32_t lane32 = lane & 31u;
    __shared__ float warp_abs[8];
    __shared__ float warp_val[8];
    __shared__ float iscale;

    const float *src = input + (uint64_t)row * in_dim + (uint64_t)b * 256u;
    const float x = lane < 256u ? src[lane] : 0.0f;
    float best_abs = fabsf(x);
    float best_val = x;
    if (lane < 256u) {
        #pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
            const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
            if (other_abs > best_abs) {
                best_abs = other_abs;
                best_val = other_val;
            }
        }
        if (lane32 == 0u) {
            warp_abs[warp] = best_abs;
            warp_val[warp] = best_val;
        }
    }
    __syncthreads();

    if (lane < 32u) {
        best_abs = lane32 < 8u ? warp_abs[lane32] : -1.0f;
        best_val = lane32 < 8u ? warp_val[lane32] : 0.0f;
        #pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
            const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
            if (other_abs > best_abs) {
                best_abs = other_abs;
                best_val = other_val;
            }
        }
        if (lane32 == 0u) iscale = best_abs == 0.0f ? 0.0f : -127.0f / best_val;
    }
    __syncthreads();

    axiom_cuda_q8_k_block *dst = out + (uint64_t)row * (in_dim / 256u) + b;
    if (lane >= 256u) return;
    if (iscale == 0.0f) {
        dst->qs[lane] = 0;
        if (lane < 16u) dst->bsums[lane] = 0;
        if (lane == 0u) dst->d = 0.0f;
        return;
    }
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    dst->qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += dst->qs[lane * 16u + i];
        dst->bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) dst->d = 1.0f / iscale;
}

__global__ static void axiom_q8_k_outlier_delta_kernel(
        const float *__restrict__ input,
        const axiom_cuda_q8_k_block *__restrict__ q8,
        uint32_t *__restrict__ count,
        uint32_t *__restrict__ indices,
        float *__restrict__ deltas,
        uint32_t n,
        float threshold,
        uint32_t cap) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = __ldg(input + i);
    if (fabsf(x) < threshold) return;
    const uint32_t slot = atomicAdd(count, 1u);
    if (slot >= cap) return;
    const axiom_cuda_q8_k_block *blk = q8 + (i >> 8u);
    const float deq = blk->d * (float)blk->qs[i & 255u];
    indices[slot] = i;
    deltas[slot] = x - deq;
}

__device__ static float axiom_dev_iq2_xxs_dot_f32(
        const uint8_t *row,
        const float *input,
        uint32_t blocks) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 66u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const uint8_t *q2 = blk + 2u;
        #pragma unroll
        for (uint32_t g32 = 0; g32 < 8u; ++g32) {
            const uint8_t *q = q2 + g32 * 8u;
            const uint32_t aux_g = axiom_dev_le32(q);
            const uint32_t aux_s = axiom_dev_le32(q + 4u);
            const float dl = d * (0.5f + (float)(aux_s >> 28u)) * 0.25f;
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) {
                const uint32_t grid_idx = (aux_g >> (8u * k)) & 0xffu;
                const uint32_t sign_idx = (aux_s >> (7u * k)) & 0x7fu;
                const uint32_t sign_mask = axiom_iq2xxs_sign_mask[sign_idx];
                const uint64_t base = (uint64_t)b * 256u + g32 * 32u + k * 8u;
                #pragma unroll
                for (uint32_t i = 0; i < 8u; ++i) {
                    float w = axiom_dev_iq2xxs_grid_value(grid_idx, i);
                    if (sign_mask & (1u << i)) w = -w;
                    acc += dl * w * __ldg(input + base + i);
                }
            }
        }
    }
    return acc;
}

__device__ static float axiom_dev_iq2_xxs_dot_f32_warp(
        const uint8_t *row,
        const float *input,
        uint32_t blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t k = lane >> 3u;
    const uint32_t i = lane & 7u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 66u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const uint8_t *q2 = blk + 2u;
        for (uint32_t g32 = 0; g32 < 8u; ++g32) {
            const uint8_t *q = q2 + g32 * 8u;
            const uint32_t aux_g = axiom_dev_le32(q);
            const uint32_t aux_s = axiom_dev_le32(q + 4u);
            const float dl = d * (0.5f + (float)(aux_s >> 28u)) * 0.25f;
            const uint32_t grid_idx = (aux_g >> (8u * k)) & 0xffu;
            const uint32_t sign_idx = (aux_s >> (7u * k)) & 0x7fu;
            const uint32_t sign_mask = axiom_iq2xxs_sign_mask[sign_idx];
            float w = axiom_dev_iq2xxs_grid_value(grid_idx, i);
            if (sign_mask & (1u << i)) w = -w;
            const uint64_t base = (uint64_t)b * 256u + g32 * 32u + lane;
            acc += dl * w * __ldg(input + base);
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__device__ static void axiom_dev_iq2_xxs_dual_dot_f32_warp(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const float *__restrict__ input,
        uint32_t blocks,
        float *gate_out,
        float *up_out) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t k = lane >> 3u;
    const uint32_t i = lane & 7u;
    const uint32_t grid_shift = 8u * k;
    const uint32_t sign_shift = 7u * k;
    const uint32_t sign_bit = 1u << i;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * 66u;
        const uint8_t *up_blk = up_row + (uint64_t)b * 66u;
        const float gate_d = axiom_dev_f16_to_f32(axiom_dev_le16(gate_blk));
        const float up_d = axiom_dev_f16_to_f32(axiom_dev_le16(up_blk));
        const uint8_t *gate_q = gate_blk + 2u;
        const uint8_t *up_q = up_blk + 2u;
        const float *x_ptr = input + (uint64_t)b * 256u + lane;
        for (uint32_t g32 = 0; g32 < 8u; ++g32) {
            const uint32_t gate_aux_g = axiom_dev_le32(gate_q);
            const uint32_t gate_aux_s = axiom_dev_le32(gate_q + 4u);
            const uint32_t up_aux_g = axiom_dev_le32(up_q);
            const uint32_t up_aux_s = axiom_dev_le32(up_q + 4u);
            const float gate_dl = gate_d * (0.5f + (float)(gate_aux_s >> 28u)) * 0.25f;
            const float up_dl = up_d * (0.5f + (float)(up_aux_s >> 28u)) * 0.25f;

            const uint32_t gate_grid_idx = (gate_aux_g >> grid_shift) & 0xffu;
            const uint32_t gate_sign_idx = (gate_aux_s >> sign_shift) & 0x7fu;
            const uint32_t gate_sign_mask = axiom_iq2xxs_sign_mask[gate_sign_idx];
            float gate_w = axiom_dev_iq2xxs_grid_value(gate_grid_idx, i);
            if (gate_sign_mask & sign_bit) gate_w = -gate_w;

            const uint32_t up_grid_idx = (up_aux_g >> grid_shift) & 0xffu;
            const uint32_t up_sign_idx = (up_aux_s >> sign_shift) & 0x7fu;
            const uint32_t up_sign_mask = axiom_iq2xxs_sign_mask[up_sign_idx];
            float up_w = axiom_dev_iq2xxs_grid_value(up_grid_idx, i);
            if (up_sign_mask & sign_bit) up_w = -up_w;

            const float x = __ldg(x_ptr);
            gate_acc += gate_dl * gate_w * x;
            up_acc += up_dl * up_w * x;
            gate_q += 8u;
            up_q += 8u;
            x_ptr += 32u;
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

template <bool ReadOnlyInput>
__device__ __forceinline__ static void axiom_dev_iq2_xxs_dual_dot_f32_hwarp16_impl(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const float *__restrict__ input,
        uint32_t blocks,
        float *gate_out,
        float *up_out) {
    const uint32_t lane16 = threadIdx.x & 15u;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = lane16; b < blocks; b += 16u) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * 66u;
        const uint8_t *up_blk = up_row + (uint64_t)b * 66u;
        const float gate_d = axiom_dev_f16_to_f32(axiom_dev_le16(gate_blk));
        const float up_d = axiom_dev_f16_to_f32(axiom_dev_le16(up_blk));
        const uint8_t *gate_q = gate_blk + 2u;
        const uint8_t *up_q = up_blk + 2u;
        const float *x_base = input + (uint64_t)b * 256u;
        for (uint32_t g32 = 0; g32 < 8u; ++g32) {
            const uint32_t gate_aux_g = axiom_dev_le32(gate_q);
            const uint32_t gate_aux_s = axiom_dev_le32(gate_q + 4u);
            const uint32_t up_aux_g = axiom_dev_le32(up_q);
            const uint32_t up_aux_s = axiom_dev_le32(up_q + 4u);
            const float gate_dl = gate_d * (0.5f + (float)(gate_aux_s >> 28u)) * 0.25f;
            const float up_dl = up_d * (0.5f + (float)(up_aux_s >> 28u)) * 0.25f;
            const float *x = x_base + g32 * 32u;
            for (uint32_t k = 0; k < 4u; ++k) {
                const uint32_t gate_grid_idx = (gate_aux_g >> (8u * k)) & 0xffu;
                const uint32_t gate_sign_idx = (gate_aux_s >> (7u * k)) & 0x7fu;
                const uint32_t gate_sign_mask = axiom_iq2xxs_sign_mask[gate_sign_idx];
                const uint32_t up_grid_idx = (up_aux_g >> (8u * k)) & 0xffu;
                const uint32_t up_sign_idx = (up_aux_s >> (7u * k)) & 0x7fu;
                const uint32_t up_sign_mask = axiom_iq2xxs_sign_mask[up_sign_idx];
                const float *xk = x + k * 8u;
                for (uint32_t i = 0; i < 8u; ++i) {
                    float gate_w = axiom_dev_iq2xxs_grid_value(gate_grid_idx, i);
                    float up_w = axiom_dev_iq2xxs_grid_value(up_grid_idx, i);
                    const uint32_t sign_bit = 1u << i;
                    if (gate_sign_mask & sign_bit) gate_w = -gate_w;
                    if (up_sign_mask & sign_bit) up_w = -up_w;
                    const float xv = ReadOnlyInput ? __ldg(xk + i) : xk[i];
                    gate_acc += gate_dl * gate_w * xv;
                    up_acc += up_dl * up_w * xv;
                }
            }
            gate_q += 8u;
            up_q += 8u;
        }
    }
    const uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (uint32_t offset = 8u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(mask, gate_acc, offset, 16);
        up_acc += __shfl_down_sync(mask, up_acc, offset, 16);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__device__ __forceinline__ static void axiom_dev_iq2_xxs_dual_dot_f32_hwarp16(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const float *__restrict__ input,
        uint32_t blocks,
        float *gate_out,
        float *up_out) {
    axiom_dev_iq2_xxs_dual_dot_f32_hwarp16_impl<true>(
            gate_row, up_row, input, blocks, gate_out, up_out);
}

__device__ __forceinline__ static void axiom_dev_iq2_xxs_dual_dot_f32_hwarp16_shared(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const float *__restrict__ input,
        uint32_t blocks,
        float *gate_out,
        float *up_out) {
    axiom_dev_iq2_xxs_dual_dot_f32_hwarp16_impl<false>(
            gate_row, up_row, input, blocks, gate_out, up_out);
}

__device__ __forceinline__ static int32_t axiom_dev_iq2xxs_signed_pack4(
        uint32_t aux_g,
        uint32_t aux_s,
        uint32_t lane8) {
    int32_t packed = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t grid_idx = (aux_g >> (8u * k)) & 255u;
        const uint32_t sign_idx = (aux_s >> (7u * k)) & 127u;
        int32_t w = axiom_dev_iq2xxs_grid_i8(grid_idx, lane8);
        if (axiom_iq2xxs_sign_mask[sign_idx] & (1u << lane8)) w = -w;
        packed |= (int32_t)((uint32_t)(uint8_t)(int8_t)w << (8u * k));
    }
    return packed;
}

__device__ __forceinline__ static int32_t axiom_dev_iq2xxs_signed_pack4_base(
        uint32_t aux_g,
        uint32_t aux_s,
        uint32_t base_lane) {
    int32_t packed = 0;
    #pragma unroll
    for (uint32_t n = 0; n < 4u; ++n) {
        const uint32_t lane = base_lane + n;
        const uint32_t k = lane >> 3u;
        const uint32_t i = lane & 7u;
        const uint32_t grid_idx = (aux_g >> (8u * k)) & 255u;
        const uint32_t sign_idx = (aux_s >> (7u * k)) & 127u;
        int32_t w = axiom_dev_iq2xxs_grid_i8(grid_idx, i);
        if (axiom_iq2xxs_sign_mask[sign_idx] & (1u << i)) w = -w;
        packed |= (int32_t)((uint32_t)(uint8_t)(int8_t)w << (8u * n));
    }
    return packed;
}

__device__ __forceinline__ static uint32_t axiom_dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static void axiom_dev_iq2_i8x8_lut(
        const uint64_t *__restrict__ grid,
        const uint8_t *__restrict__ signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = axiom_dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

__device__ __forceinline__ static int32_t axiom_dev_q8k_pack4_base(
        const int8_t *__restrict__ q,
        uint32_t base_lane) {
    return (int32_t)((uint32_t)(uint8_t)q[base_lane] |
           ((uint32_t)(uint8_t)q[base_lane + 1u] << 8u) |
           ((uint32_t)(uint8_t)q[base_lane + 2u] << 16u) |
           ((uint32_t)(uint8_t)q[base_lane + 3u] << 24u));
}

__device__ __forceinline__ static int32_t axiom_dev_q8k_pack4(
        const int8_t *q,
        uint32_t lane8) {
    return (int32_t)((uint32_t)(uint8_t)q[lane8] |
           ((uint32_t)(uint8_t)q[8u + lane8] << 8u) |
           ((uint32_t)(uint8_t)q[16u + lane8] << 16u) |
           ((uint32_t)(uint8_t)q[24u + lane8] << 24u));
}

__device__ __forceinline__ static float axiom_dev_iq2_xxs_weight_at(
        const uint8_t *__restrict__ row,
        uint32_t col) {
    const uint32_t b = col >> 8u;
    const uint32_t within = col & 255u;
    const uint32_t g32 = within >> 5u;
    const uint32_t k = (within >> 3u) & 3u;
    const uint32_t i = within & 7u;
    const uint8_t *blk = row + (uint64_t)b * 66u;
    const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
    const uint8_t *q = blk + 2u + g32 * 8u;
    const uint32_t aux_g = axiom_dev_le32(q);
    const uint32_t aux_s = axiom_dev_le32(q + 4u);
    const float dl = d * (0.5f + (float)(aux_s >> 28u)) * 0.25f;
    const uint32_t grid_idx = (aux_g >> (8u * k)) & 0xffu;
    const uint32_t sign_idx = (aux_s >> (7u * k)) & 0x7fu;
    float w = axiom_dev_iq2xxs_grid_value(grid_idx, i);
    if (axiom_iq2xxs_sign_mask[sign_idx] & (1u << i)) w = -w;
    return dl * w;
}

__global__ static void axiom_q8_0_tile32_pack_f32_kernel(
        const float *__restrict__ input,
        axiom_cuda_q8_0_tile32_block *__restrict__ out,
        uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    const uint32_t lane = (uint32_t)threadIdx.x;
    if (b >= blocks || lane >= 32u) return;
    __shared__ float absmax[32];
    __shared__ float scale;
    const float x = __ldg(input + (uint64_t)b * 32u + lane);
    absmax[lane] = fabsf(x);
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride && absmax[lane + stride] > absmax[lane]) {
            absmax[lane] = absmax[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0u) scale = absmax[0] > 0.0f ? absmax[0] / 127.0f : 0.0f;
    __syncthreads();
    if (scale == 0.0f) {
        out[b].qs[lane] = 0;
        if (lane == 0u) out[b].d = 0.0f;
        return;
    }
    int q = (int)lrintf(x / scale);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    out[b].qs[lane] = (int8_t)q;
    if (lane == 0u) out[b].d = scale;
}

__device__ static void axiom_dev_iq2_xxs_dual_dot_q8k_qwarp8(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const axiom_cuda_q8_k_block *__restrict__ input,
        uint32_t blocks,
        uint32_t lane8,
        float *gate_out,
        float *up_out) {
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * 66u;
        const uint8_t *up_blk = up_row + (uint64_t)b * 66u;
        const float gate_d = axiom_dev_f16_to_f32(axiom_dev_le16(gate_blk));
        const float up_d = axiom_dev_f16_to_f32(axiom_dev_le16(up_blk));
        const float x_d = input[b].d;
        const uint8_t *gate_q = gate_blk + 2u;
        const uint8_t *up_q = up_blk + 2u;
        for (uint32_t g32 = 0; g32 < 8u; ++g32) {
            const uint32_t gate_aux_g = axiom_dev_le32(gate_q);
            const uint32_t gate_aux_s = axiom_dev_le32(gate_q + 4u);
            const uint32_t up_aux_g = axiom_dev_le32(up_q);
            const uint32_t up_aux_s = axiom_dev_le32(up_q + 4u);
            const int32_t xpack = axiom_dev_q8k_pack4(input[b].qs + g32 * 32u, lane8);
            const int32_t gate_sum = __dp4a(axiom_dev_iq2xxs_signed_pack4(gate_aux_g, gate_aux_s, lane8), xpack, 0);
            const int32_t up_sum = __dp4a(axiom_dev_iq2xxs_signed_pack4(up_aux_g, up_aux_s, lane8), xpack, 0);
            gate_acc += gate_d * x_d * (0.5f + (float)(gate_aux_s >> 28u)) * 0.25f * (float)gate_sum;
            up_acc += up_d * x_d * (0.5f + (float)(up_aux_s >> 28u)) * 0.25f * (float)up_sum;
            gate_q += 8u;
            up_q += 8u;
        }
    }
    for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset, 8);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset, 8);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__device__ __forceinline__ static float axiom_dev_iq2_xxs_dot_q8k_block_full(
        const uint8_t *__restrict__ row_blk,
        const axiom_cuda_q8_k_block *__restrict__ input,
        uint32_t aux_offset) {
    const float row_d = axiom_dev_f16_to_f32(axiom_dev_le16(row_blk));
    const float x_d = input->d;
    const uint8_t *q = row_blk + aux_offset;
    const int8_t *q8 = input->qs;
    int32_t bsum = 0;
    #pragma unroll
    for (uint32_t g32 = 0; g32 < 8u; ++g32) {
        const uint32_t aux_g = axiom_dev_le32(q);
        const uint32_t aux_s = axiom_dev_le32(q + 4u);
        q += 8u;
        int32_t sumi = 0;
        #pragma unroll
        for (uint32_t base = 0u; base < 32u; base += 4u) {
            sumi = __dp4a(
                    axiom_dev_iq2xxs_signed_pack4_base(aux_g, aux_s, base),
                    axiom_dev_q8k_pack4_base(q8 + g32 * 32u, base),
                    sumi);
        }
        bsum += sumi * (int32_t)(2u * (aux_s >> 28u) + 1u);
    }
    return 0.125f * row_d * x_d * (float)bsum;
}

__device__ __forceinline__ static float axiom_dev_iq2_xxs_dot_q8k_block_lut_full(
        const uint8_t *__restrict__ row_blk,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint64_t *__restrict__ grid,
        const uint8_t *__restrict__ signs,
        uint32_t aux_offset) {
    const float row_d = axiom_dev_f16_to_f32(axiom_dev_le16(row_blk));
    const float x_d = input->d;
    const uint16_t *q2 = (const uint16_t *)(row_blk + aux_offset);
    const int8_t *q8 = input->qs;
    int32_t bsum = 0;
    #pragma unroll
    for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16u);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16u);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28u) + 1u);
        int32_t w[8];
        axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),          (aux1 >> 0u)  & 127u, &w[0], &w[1]);
        axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8u)  & 0xffu), (aux1 >> 7u)  & 127u, &w[2], &w[3]);
        axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16u) & 0xffu), (aux1 >> 14u) & 127u, &w[4], &w[5]);
        axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24u) & 0xffu), (aux1 >> 21u) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        const int8_t *q = q8 + ib32 * 32u;
        sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * row_d * x_d * (float)bsum;
}

__device__ __forceinline__ static void axiom_dev_iq2_xxs_dual_dot_q8k_block_lane(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const axiom_cuda_q8_k_block *__restrict__ input,
        uint32_t blocks,
        uint32_t block_stride,
        uint32_t aux_offset,
        uint32_t lane8,
        float *gate_out,
        float *up_out) {
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = lane8; b < blocks; b += 8u) {
        gate_acc += axiom_dev_iq2_xxs_dot_q8k_block_full(gate_row + (uint64_t)b * block_stride, input + b, aux_offset);
        up_acc += axiom_dev_iq2_xxs_dot_q8k_block_full(up_row + (uint64_t)b * block_stride, input + b, aux_offset);
    }
    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(mask, gate_acc, offset, 8);
        up_acc += __shfl_down_sync(mask, up_acc, offset, 8);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__device__ __forceinline__ static void axiom_dev_iq2_xxs_dual_dot_q8k_block_lut_lane(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const axiom_cuda_q8_k_block *__restrict__ input,
        uint32_t blocks,
        uint32_t block_stride,
        uint32_t aux_offset,
        uint32_t lane8,
        const uint64_t *__restrict__ grid,
        const uint8_t *__restrict__ signs,
        float *gate_out,
        float *up_out) {
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = lane8; b < blocks; b += 8u) {
        gate_acc += axiom_dev_iq2_xxs_dot_q8k_block_lut_full(gate_row + (uint64_t)b * block_stride, input + b, grid, signs, aux_offset);
        up_acc += axiom_dev_iq2_xxs_dot_q8k_block_lut_full(up_row + (uint64_t)b * block_stride, input + b, grid, signs, aux_offset);
    }
    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(mask, gate_acc, offset, 8);
        up_acc += __shfl_down_sync(mask, up_acc, offset, 8);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__device__ __forceinline__ static void axiom_dev_iq2_xxs_dual_dot_q8k_block_lut_reuse_lane(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const axiom_cuda_q8_k_block *__restrict__ input,
        uint32_t blocks,
        uint32_t block_stride,
        uint32_t aux_offset,
        uint32_t lane8,
        const uint64_t *__restrict__ grid,
        const uint8_t *__restrict__ signs,
        float *gate_out,
        float *up_out) {
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = lane8; b < blocks; b += 8u) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * block_stride;
        const uint8_t *up_blk = up_row + (uint64_t)b * block_stride;
        const float gate_d = axiom_dev_f16_to_f32(axiom_dev_le16(gate_blk));
        const float up_d = axiom_dev_f16_to_f32(axiom_dev_le16(up_blk));
        const float x_d = input[b].d;
        const uint16_t *gate_q2 = (const uint16_t *)(gate_blk + aux_offset);
        const uint16_t *up_q2 = (const uint16_t *)(up_blk + aux_offset);
        const int8_t *q8 = input[b].qs;
        int32_t gate_bsum = 0;
        int32_t up_bsum = 0;
        #pragma unroll
        for (uint32_t ib32 = 0; ib32 < 8u; ++ib32) {
            const uint32_t gate_aux0 = (uint32_t)gate_q2[0] | ((uint32_t)gate_q2[1] << 16u);
            const uint32_t gate_aux1 = (uint32_t)gate_q2[2] | ((uint32_t)gate_q2[3] << 16u);
            const uint32_t up_aux0 = (uint32_t)up_q2[0] | ((uint32_t)up_q2[1] << 16u);
            const uint32_t up_aux1 = (uint32_t)up_q2[2] | ((uint32_t)up_q2[3] << 16u);
            gate_q2 += 4;
            up_q2 += 4;
            const int32_t gate_ls = (int32_t)(2u * (gate_aux1 >> 28u) + 1u);
            const int32_t up_ls = (int32_t)(2u * (up_aux1 >> 28u) + 1u);
            int32_t gate_w[8];
            int32_t up_w[8];
            axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)(gate_aux0 & 0xffu),          (gate_aux1 >> 0u)  & 127u, &gate_w[0], &gate_w[1]);
            axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)((gate_aux0 >> 8u)  & 0xffu), (gate_aux1 >> 7u)  & 127u, &gate_w[2], &gate_w[3]);
            axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)((gate_aux0 >> 16u) & 0xffu), (gate_aux1 >> 14u) & 127u, &gate_w[4], &gate_w[5]);
            axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)((gate_aux0 >> 24u) & 0xffu), (gate_aux1 >> 21u) & 127u, &gate_w[6], &gate_w[7]);
            axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)(up_aux0 & 0xffu),          (up_aux1 >> 0u)  & 127u, &up_w[0], &up_w[1]);
            axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)((up_aux0 >> 8u)  & 0xffu), (up_aux1 >> 7u)  & 127u, &up_w[2], &up_w[3]);
            axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)((up_aux0 >> 16u) & 0xffu), (up_aux1 >> 14u) & 127u, &up_w[4], &up_w[5]);
            axiom_dev_iq2_i8x8_lut(grid, signs, (uint8_t)((up_aux0 >> 24u) & 0xffu), (up_aux1 >> 21u) & 127u, &up_w[6], &up_w[7]);
            int32_t gate_sumi = 0;
            int32_t up_sumi = 0;
            const int8_t *q = q8 + ib32 * 32u;
            const int32_t x0 = *(const int32_t *)(q + 0);
            const int32_t x1 = *(const int32_t *)(q + 4);
            const int32_t x2 = *(const int32_t *)(q + 8);
            const int32_t x3 = *(const int32_t *)(q + 12);
            const int32_t x4 = *(const int32_t *)(q + 16);
            const int32_t x5 = *(const int32_t *)(q + 20);
            const int32_t x6 = *(const int32_t *)(q + 24);
            const int32_t x7 = *(const int32_t *)(q + 28);
            gate_sumi = __dp4a(gate_w[0], x0, gate_sumi);
            up_sumi = __dp4a(up_w[0], x0, up_sumi);
            gate_sumi = __dp4a(gate_w[1], x1, gate_sumi);
            up_sumi = __dp4a(up_w[1], x1, up_sumi);
            gate_sumi = __dp4a(gate_w[2], x2, gate_sumi);
            up_sumi = __dp4a(up_w[2], x2, up_sumi);
            gate_sumi = __dp4a(gate_w[3], x3, gate_sumi);
            up_sumi = __dp4a(up_w[3], x3, up_sumi);
            gate_sumi = __dp4a(gate_w[4], x4, gate_sumi);
            up_sumi = __dp4a(up_w[4], x4, up_sumi);
            gate_sumi = __dp4a(gate_w[5], x5, gate_sumi);
            up_sumi = __dp4a(up_w[5], x5, up_sumi);
            gate_sumi = __dp4a(gate_w[6], x6, gate_sumi);
            up_sumi = __dp4a(up_w[6], x6, up_sumi);
            gate_sumi = __dp4a(gate_w[7], x7, gate_sumi);
            up_sumi = __dp4a(up_w[7], x7, up_sumi);
            gate_bsum += gate_sumi * gate_ls;
            up_bsum += up_sumi * up_ls;
        }
        gate_acc += 0.125f * gate_d * x_d * (float)gate_bsum;
        up_acc += 0.125f * up_d * x_d * (float)up_bsum;
    }
    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(mask, gate_acc, offset, 8);
        up_acc += __shfl_down_sync(mask, up_acc, offset, 8);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__device__ static void axiom_dev_iq2_xxs_dual_dot_q8t32_qwarp8(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        uint32_t blocks,
        uint32_t lane8,
        float *gate_out,
        float *up_out) {
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * 66u;
        const uint8_t *up_blk = up_row + (uint64_t)b * 66u;
        const float gate_d = axiom_dev_f16_to_f32(axiom_dev_le16(gate_blk));
        const float up_d = axiom_dev_f16_to_f32(axiom_dev_le16(up_blk));
        const uint8_t *gate_q = gate_blk + 2u;
        const uint8_t *up_q = up_blk + 2u;
        for (uint32_t g32 = 0; g32 < 8u; ++g32) {
            const axiom_cuda_q8_0_tile32_block *x = input + (uint64_t)b * 8u + g32;
            const uint32_t gate_aux_g = axiom_dev_le32(gate_q);
            const uint32_t gate_aux_s = axiom_dev_le32(gate_q + 4u);
            const uint32_t up_aux_g = axiom_dev_le32(up_q);
            const uint32_t up_aux_s = axiom_dev_le32(up_q + 4u);
            const int32_t xpack = axiom_dev_q8k_pack4(x->qs, lane8);
            const int32_t gate_sum = __dp4a(axiom_dev_iq2xxs_signed_pack4(gate_aux_g, gate_aux_s, lane8), xpack, 0);
            const int32_t up_sum = __dp4a(axiom_dev_iq2xxs_signed_pack4(up_aux_g, up_aux_s, lane8), xpack, 0);
            gate_acc += gate_d * x->d * (0.5f + (float)(gate_aux_s >> 28u)) * 0.25f * (float)gate_sum;
            up_acc += up_d * x->d * (0.5f + (float)(up_aux_s >> 28u)) * 0.25f * (float)up_sum;
            gate_q += 8u;
            up_q += 8u;
        }
    }
    for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset, 8);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset, 8);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__device__ static float axiom_dev_q2_k_dot_f32(
        const uint8_t *row,
        const float *input,
        uint32_t blocks) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 84u;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk + 80u));
        const float dmin = axiom_dev_f16_to_f32(axiom_dev_le16(blk + 82u));
        for (uint32_t il = 0; il < 16u; ++il) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4u);
            const uint8_t *q = qs + 32u * chunk + 16u * pair;
            const uint64_t base = (uint64_t)b * 256u + chunk * 128u +
                    ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16u; ++i) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * __ldg(input + base + i);
            }
        }
    }
    return acc;
}

__device__ __forceinline__ static int32_t axiom_dev_dot_q2_16_q8(
        const uint8_t *__restrict__ q2,
        const int8_t *__restrict__ q8,
        int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16u; i += 4u) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ __forceinline__ static float axiom_dev_q2_k_dot_q8k_block(
        const uint8_t *__restrict__ blk,
        const axiom_cuda_q8_k_block *__restrict__ xq) {
    const uint8_t *q2 = blk + 16u;
    const int8_t *q8 = xq->qs;
    const uint8_t *sc = blk;
    int32_t summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 16u; ++j) {
        summs += (int32_t)xq->bsums[j] * (int32_t)(sc[j] >> 4u);
    }
    const float d = xq->d * axiom_dev_f16_to_f32(axiom_dev_le16(blk + 80u));
    const float dmin = xq->d * axiom_dev_f16_to_f32(axiom_dev_le16(blk + 82u));
    int32_t isum = 0;
    uint32_t is = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 2u; ++k) {
        int shift = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            int32_t dl = (int32_t)(sc[is++] & 0x0fu);
            isum += dl * axiom_dev_dot_q2_16_q8(q2, q8, shift);
            dl = (int32_t)(sc[is++] & 0x0fu);
            isum += dl * axiom_dev_dot_q2_16_q8(q2 + 16u, q8 + 16u, shift);
            shift += 2;
            q8 += 32u;
        }
        q2 += 32u;
    }
    return d * (float)isum - dmin * (float)summs;
}

__device__ __forceinline__ static void axiom_dev_q4_k_get_scale_min(
        uint32_t j,
        const uint8_t *__restrict__ scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t axiom_dev_dot_q4_32_q8(
        const uint8_t *__restrict__ q4,
        const int8_t *__restrict__ q8,
        int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(q4 + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ __forceinline__ static float axiom_dev_q4_k_dot_q8k_block(
        const uint8_t *__restrict__ blk,
        const axiom_cuda_q8_k_block *__restrict__ xq) {
    const float d = xq->d * axiom_dev_f16_to_f32(axiom_dev_le16(blk));
    const float dmin = xq->d * axiom_dev_f16_to_f32(axiom_dev_le16(blk + 2u));
    const uint8_t *scales = blk + 4u;
    const uint8_t *qs = blk + 16u;
    int32_t isum = 0;
    int32_t summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        uint8_t sc = 0;
        uint8_t m = 0;
        axiom_dev_q4_k_get_scale_min(j, scales, &sc, &m);
        summs += (int32_t)m * (int32_t)(xq->bsums[2u * j] + xq->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int32_t)sc * axiom_dev_dot_q4_32_q8(qs + byte_off, xq->qs + j * 32u, shift);
    }
    return d * (float)isum - dmin * (float)summs;
}

__device__ static float axiom_dev_q2_k_dot_f32_warp(
        const uint8_t *__restrict__ row,
        const float *__restrict__ input,
        uint32_t blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t pair = lane >> 4u;
    const uint32_t i = lane & 15u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 84u;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk + 80u));
        const float dmin = axiom_dev_f16_to_f32(axiom_dev_le16(blk + 82u));
        const float *x_ptr = input + (uint64_t)b * 256u + lane;
        for (uint32_t segment = 0; segment < 8u; ++segment) {
            const uint32_t chunk = segment >> 2u;
            const uint32_t qseg = segment & 3u;
            const uint32_t il = chunk * 8u + qseg * 2u + pair;
            const uint8_t sc = scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4u);
            const uint8_t *q = qs + 32u * chunk + 16u * pair;
            const uint32_t shift = qseg * 2u;
            const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
            acc += w * __ldg(x_ptr);
            x_ptr += 32u;
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__device__ static float axiom_dev_q2_k_dot_f32_hwarp16(
        const uint8_t *__restrict__ row,
        const float *__restrict__ input,
        uint32_t blocks) {
    const uint32_t lane16 = threadIdx.x & 15u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 84u;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk + 80u));
        const float dmin = axiom_dev_f16_to_f32(axiom_dev_le16(blk + 82u));
        for (uint32_t segment = 0; segment < 8u; ++segment) {
            const uint32_t chunk = segment >> 2u;
            const uint32_t qseg = segment & 3u;
            const uint32_t shift = qseg * 2u;
            for (uint32_t pair = 0; pair < 2u; ++pair) {
                const uint32_t il = chunk * 8u + qseg * 2u + pair;
                const uint8_t sc = scales[il];
                const float dl = d * (float)(sc & 0x0fu);
                const float ml = dmin * (float)(sc >> 4u);
                const uint8_t *q = qs + 32u * chunk + 16u * pair;
                const float w = dl * (float)((q[lane16] >> shift) & 3u) - ml;
                const float x = __ldg(input + (uint64_t)b * 256u + segment * 32u + pair * 16u + lane16);
                acc += w * x;
            }
        }
    }
    const uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (uint32_t offset = 8u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(mask, acc, offset, 16);
    }
    return acc;
}

__device__ static float axiom_dev_q2_k_block_dot_f32_shared(
        const uint8_t *__restrict__ blk,
        const float *__restrict__ input) {
    const uint8_t *scales = blk;
    const uint8_t *qs = blk + 16u;
    const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk + 80u));
    const float dmin = axiom_dev_f16_to_f32(axiom_dev_le16(blk + 82u));
    float acc = 0.0f;
    for (uint32_t il = 0; il < 16u; ++il) {
        const uint32_t chunk = il / 8u;
        const uint32_t pair = il & 1u;
        const uint32_t shift = ((il / 2u) & 3u) * 2u;
        const uint8_t sc = scales[il];
        const float dl = d * (float)(sc & 0x0fu);
        const float ml = dmin * (float)(sc >> 4u);
        const uint8_t *q = qs + 32u * chunk + 16u * pair;
        const uint32_t base = chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
        for (uint32_t i = 0; i < 16u; ++i) {
            const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
            acc += w * input[base + i];
        }
    }
    return acc;
}

extern "C" int axiom_cuda_bf16_to_f32(
        void *cuda_runtime,
        const uint16_t *bf16_host,
        float *out_host,
        size_t count) {
    if (!cuda_runtime || !bf16_host || !out_host || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    uint16_t *in_dev = nullptr;
    float *out_dev = nullptr;
    err = cudaMalloc(&in_dev, count * sizeof(uint16_t));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, count * sizeof(float));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(in_dev, bf16_host, count * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int block = 256;
        const int grid = (int)((count + block - 1) / block);
        axiom_bf16_to_f32_kernel<<<grid, block>>>(in_dev, out_dev, count);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, count * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(in_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(in_dev);
    return axiom_cuda_status(err);
}

__device__ static float axiom_dev_q8_0_dot_f32(
        const uint8_t *row,
        const float *input,
        uint32_t blocks);
__device__ static float axiom_dev_q8_0_dot_f32_warp(
        const uint8_t *row,
        const float *input,
        uint32_t blocks);
__device__ static float axiom_dev_q8_0_dot_f32_warp_shared(
        const uint8_t *row,
        const float *input,
        uint32_t blocks);
__device__ static float axiom_dev_q8_0_dot_q8tile_qwarp8(
        const uint8_t *row,
        const axiom_cuda_q8_0_tile32_block *input,
        uint32_t blocks,
        uint32_t lane8,
        uint32_t mask);
__device__ __forceinline__ static int32_t axiom_dev_q8_0_dot_i8_block32(
        const int8_t *__restrict__ wq,
        const int8_t *__restrict__ xq);

__global__ static void axiom_q8_0_matvec_f32_kernel(
        const uint8_t *weight,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    out[row] = axiom_dev_q8_0_dot_f32(r, input, blocks);
}

__global__ static void axiom_q8_0_matvec_f32_warp_kernel(
        const uint8_t *weight,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    const float acc = axiom_dev_q8_0_dot_f32_warp(r, input, blocks);
    if (lane == 0u) out[row] = acc;
}

__device__ __forceinline__ static float axiom_dev_q8_0_dequant_value(
        const uint8_t *row,
        uint32_t idx) {
    const uint32_t block = idx >> 5u;
    const uint32_t lane = idx & 31u;
    const uint8_t *blk = row + (uint64_t)block * 34u;
    const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
    const int8_t *qs = (const int8_t *)(blk + 2u);
    return d * (float)qs[lane];
}

__global__ static void axiom_q8_0_embedding_gather_f32_kernel(
        const uint8_t *__restrict__ embedding,
        float *__restrict__ out,
        uint32_t token_id,
        uint32_t token_count,
        uint32_t hidden) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    if (token_id >= token_count) {
        out[i] = 0.0f;
        return;
    }
    const uint64_t row_bytes = (uint64_t)(hidden / 32u) * 34u;
    const uint8_t *row = embedding + (uint64_t)token_id * row_bytes;
    out[i] = axiom_dev_q8_0_dequant_value(row, i);
}

__global__ static void axiom_q8_0_embedding_gather_token_f32_kernel(
        const uint8_t *__restrict__ embedding,
        const uint32_t *__restrict__ token_id,
        float *__restrict__ out,
        uint32_t token_count,
        uint32_t hidden) {
    const uint32_t tok = token_id[0];
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    if (tok >= token_count) {
        out[i] = 0.0f;
        return;
    }
    const uint64_t row_bytes = (uint64_t)(hidden / 32u) * 34u;
    const uint8_t *row = embedding + (uint64_t)tok * row_bytes;
    out[i] = axiom_dev_q8_0_dequant_value(row, i);
}

extern "C" int axiom_cuda_q8_0_embedding_gather_f32_device(
        void *cuda_runtime,
        const void *embedding_q8,
        uint64_t embedding_offset,
        void *out,
        uint64_t out_offset,
        uint32_t token_id,
        uint32_t token_count,
        uint32_t hidden) {
    if (!cuda_runtime || !embedding_q8 || !out || token_count == 0 || hidden == 0 ||
        (hidden % 32u) != 0u || token_id >= token_count) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ebuf = (const axiom_cuda_buffer *)embedding_q8;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    const uint64_t row_bytes = (uint64_t)(hidden / 32u) * 34u;
    const uint64_t emb_bytes = row_bytes * (uint64_t)token_count;
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (embedding_offset > ebuf->bytes || emb_bytes > ebuf->bytes - embedding_offset ||
        out_offset > obuf->bytes || out_bytes > obuf->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((hidden + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_q8_0_embedding_gather_f32_kernel<<<grid, block>>>(
            (const uint8_t *)ebuf->ptr + embedding_offset,
            (float *)((uint8_t *)obuf->ptr + out_offset),
            token_id,
            token_count,
            hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_embedding_gather_token_f32_device(
        void *cuda_runtime,
        const void *embedding_q8,
        uint64_t embedding_offset,
        const void *token_id,
        uint64_t token_id_offset,
        void *out,
        uint64_t out_offset,
        uint32_t token_count,
        uint32_t hidden) {
    if (!cuda_runtime || !embedding_q8 || !token_id || !out || token_count == 0 || hidden == 0 ||
        (hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ebuf = (const axiom_cuda_buffer *)embedding_q8;
    const axiom_cuda_buffer *tbuf = (const axiom_cuda_buffer *)token_id;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    const uint64_t row_bytes = (uint64_t)(hidden / 32u) * 34u;
    const uint64_t emb_bytes = row_bytes * (uint64_t)token_count;
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (embedding_offset > ebuf->bytes || emb_bytes > ebuf->bytes - embedding_offset ||
        token_id_offset > tbuf->bytes || sizeof(uint32_t) > tbuf->bytes - token_id_offset ||
        out_offset > obuf->bytes || out_bytes > obuf->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((hidden + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_q8_0_embedding_gather_token_f32_kernel<<<grid, block>>>(
            (const uint8_t *)ebuf->ptr + embedding_offset,
            (const uint32_t *)((const uint8_t *)tbuf->ptr + token_id_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            token_count,
            hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__device__ __forceinline__ static float axiom_dev_q8_0_soa_dot_f32_warp(
        const int8_t *__restrict__ qrow,
        const uint16_t *__restrict__ srow,
        const float *__restrict__ input,
        uint32_t blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        float d = lane == 0u ? axiom_dev_f16_to_f32(srow[b]) : 0.0f;
        d = __shfl_sync(0xffffffffu, d, 0);
        const int8_t q = __ldg(qrow + (uint64_t)b * 32u + lane);
        const float x = __ldg(input + (uint64_t)b * 32u + lane);
        acc += d * (float)q * x;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__global__ static void axiom_q8_0_soa_matvec_f32_warp_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const float *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const int8_t *qrow = qs + (uint64_t)row * cols;
    const uint16_t *srow = scales + (uint64_t)row * blocks;
    const float acc = axiom_dev_q8_0_soa_dot_f32_warp(qrow, srow, input, blocks);
    if (lane == 0u) out[row] = acc;
}

__device__ __forceinline__ static float axiom_dev_q8_0_soa_dot_f32_blocklane(
        const int8_t *__restrict__ qrow,
        const uint16_t *__restrict__ srow,
        const float *__restrict__ input,
        uint32_t blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uchar4 *q4 = (const uchar4 *)(qrow + (uint64_t)b * 32u);
        const uint64_t base = (uint64_t)b * 32u;
        const float d = axiom_dev_f16_to_f32(srow[b]);
        float p = 0.0f;
        #pragma unroll
        for (uint32_t k = 0; k < 8u; ++k) {
            const uchar4 v = q4[k];
            p += (float)(int8_t)v.x * __ldg(input + base + k * 4u + 0u);
            p += (float)(int8_t)v.y * __ldg(input + base + k * 4u + 1u);
            p += (float)(int8_t)v.z * __ldg(input + base + k * 4u + 2u);
            p += (float)(int8_t)v.w * __ldg(input + base + k * 4u + 3u);
        }
        acc += d * p;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__global__ static void axiom_q8_0_soa_matvec_f32_blocklane_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const float *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const int8_t *qrow = qs + (uint64_t)row * cols;
    const uint16_t *srow = scales + (uint64_t)row * blocks;
    const float acc = axiom_dev_q8_0_soa_dot_f32_blocklane(qrow, srow, input, blocks);
    if (lane == 0u) out[row] = acc;
}

template <uint32_t Blocks>
__device__ __forceinline__ static float axiom_dev_q8_0_dot_f32_warp_const(
        const uint8_t *__restrict__ row,
        const float *__restrict__ input) {
    const uint32_t lane = threadIdx.x & 31u;
    float acc = 0.0f;
    #pragma unroll 4
    for (uint32_t b = 0; b < Blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 34u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2u);
        acc += d * (float)qs[lane] * __ldg(input + (uint64_t)b * 32u + lane);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

template <uint32_t Blocks>
__global__ static void axiom_q8_0_matvec_f32_warp_const_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint64_t row_bytes = (uint64_t)Blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    const float acc = axiom_dev_q8_0_dot_f32_warp_const<Blocks>(r, input);
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_matvec_f32_warp_sx_kernel(
        const uint8_t *weight,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    extern __shared__ float sx[];
    for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) sx[i] = __ldg(input + i);
    __syncthreads();

    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    const float acc = axiom_dev_q8_0_dot_f32_warp_shared(r, sx, blocks);
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_matvec_f32_dp4a_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t qwarp = lane >> 3u;
    const uint32_t lane8 = lane & 7u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t rows_per_block = warps_per_block * 4u;
    const uint32_t row = (uint32_t)blockIdx.x * rows_per_block + warp * 4u + qwarp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    const uint32_t mask = 0xffu << (qwarp * 8u);
    const float acc = axiom_dev_q8_0_dot_q8tile_qwarp8(r, input, blocks, lane8, mask);
    if (lane8 == 0u) out[row] = acc;
}

__device__ __forceinline__ static int32_t axiom_dev_q8_0_dot_i8_block32(
        const int8_t *__restrict__ wq,
        const int8_t *__restrict__ xq) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        sum = __dp4a(
                axiom_dev_q8k_pack4_base(wq, i),
                axiom_dev_q8k_pack4_base(xq, i),
                sum);
    }
    return sum;
}

__device__ __forceinline__ static int32_t axiom_dev_q8_0_dot_i8_block32_aligned(
        const int8_t *__restrict__ wq,
        const int8_t *__restrict__ xq) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        sum = __dp4a(*(const int32_t *)(wq + i), *(const int32_t *)(xq + i), sum);
    }
    return sum;
}

__global__ static void axiom_q8_0_matvec_f32_preq_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
        acc += wd * x->d * (float)dot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_matvec_add_preq_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        const float *__restrict__ residual,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
        acc += wd * x->d * (float)dot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) out[row] = residual[row] + acc;
}

__global__ static void axiom_q8_0_matvec2_preq_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        float *__restrict__ out0,
        float *__restrict__ out1,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x0 = input0 + b;
        const axiom_cuda_q8_0_tile32_block *x1 = input1 + b;
        acc0 += wd * x0->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x0->qs);
        acc1 += wd * x1->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x1->qs);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
    }
}

__global__ static void axiom_q8_0_matvec4_preq_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input2,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input3,
        float *__restrict__ out0,
        float *__restrict__ out1,
        float *__restrict__ out2,
        float *__restrict__ out3,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x0 = input0 + b;
        const axiom_cuda_q8_0_tile32_block *x1 = input1 + b;
        const axiom_cuda_q8_0_tile32_block *x2 = input2 + b;
        const axiom_cuda_q8_0_tile32_block *x3 = input3 + b;
        acc0 += wd * x0->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x0->qs);
        acc1 += wd * x1->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x1->qs);
        acc2 += wd * x2->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x2->qs);
        acc3 += wd * x3->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x3->qs);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
        out2[row] = acc2;
        out3[row] = acc3;
    }
}

__global__ static void axiom_q8_0_matvec2_f32_warp_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ input0,
        const float *__restrict__ input1,
        float *__restrict__ out0,
        float *__restrict__ out1,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2u);
        const uint64_t xoff = (uint64_t)b * 32u + lane;
        const float w = d * (float)qs[lane];
        acc0 += w * __ldg(input0 + xoff);
        acc1 += w * __ldg(input1 + xoff);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
    }
}

__global__ static void axiom_q8_0_matvec4_f32_warp_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ input0,
        const float *__restrict__ input1,
        const float *__restrict__ input2,
        const float *__restrict__ input3,
        float *__restrict__ out0,
        float *__restrict__ out1,
        float *__restrict__ out2,
        float *__restrict__ out3,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2u);
        const uint64_t xoff = (uint64_t)b * 32u + lane;
        const float w = d * (float)qs[lane];
        acc0 += w * __ldg(input0 + xoff);
        acc1 += w * __ldg(input1 + xoff);
        acc2 += w * __ldg(input2 + xoff);
        acc3 += w * __ldg(input3 + xoff);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
        out2[row] = acc2;
        out3[row] = acc3;
    }
}

// Dynamic-B batched Q8_0 matvec. Each warp owns one output row r; it streams the
// q8_0 weight row ONCE per chunk of up to AXIOM_Q8_BATCH_CAP batch elements and
// accumulates one dot-product per batch element b: dot(weightRow, X[b]) -> Y[b][r].
// The per-element dequant+accumulation is byte-identical to the single-vector
// warp kernel (axiom_dev_q8_0_dot_f32_warp): same f16 scale decode, same
// d*(float)qs[lane]*x accumulation order, same shfl_down reduction. So each
// output element is bit-for-bit identical to the single-vector path.
//   X layout: row-major [B][cols]  (X + (uint64_t)b * cols)
//   Y layout: y_layout==0 -> [B][rows] (per-seq contiguous, natural/recommended);
//             y_layout==1 -> [rows][B] (per-row contiguous / interleaved).
#ifndef AXIOM_Q8_BATCH_CAP
#define AXIOM_Q8_BATCH_CAP 8u
#endif
__global__ static void axiom_q8_0_batched_matvec_f32_warp_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ x,
        float *__restrict__ y,
        uint32_t batch,
        uint32_t rows,
        uint32_t cols,
        uint32_t y_layout) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    // batch is warp-uniform, so the j<bn predicates below never diverge a warp.
    for (uint32_t b0 = 0; b0 < batch; b0 += AXIOM_Q8_BATCH_CAP) {
        const uint32_t bn = (batch - b0) < AXIOM_Q8_BATCH_CAP ? (batch - b0) : AXIOM_Q8_BATCH_CAP;
        float acc[AXIOM_Q8_BATCH_CAP];
        #pragma unroll
        for (uint32_t j = 0; j < AXIOM_Q8_BATCH_CAP; ++j) acc[j] = 0.0f;
        for (uint32_t b = 0; b < blocks; ++b) {
            const uint8_t *blk = r + (uint64_t)b * 34u;
            const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int8_t *qs = (const int8_t *)(blk + 2u);
            const uint64_t xoff = (uint64_t)b * 32u + lane;
            const float qv = (float)qs[lane];
            #pragma unroll
            for (uint32_t j = 0; j < AXIOM_Q8_BATCH_CAP; ++j) {
                if (j < bn) {
                    acc[j] += d * qv * __ldg(x + (uint64_t)(b0 + j) * cols + xoff);
                }
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < AXIOM_Q8_BATCH_CAP; ++j) {
            if (j < bn) {
                float a = acc[j];
                for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                    a += __shfl_down_sync(0xffffffffu, a, offset);
                }
                if (lane == 0u) {
                    const uint32_t bb = b0 + j;
                    const uint64_t yidx = (y_layout == 1u)
                            ? (uint64_t)row * batch + bb
                            : (uint64_t)bb * rows + row;
                    y[yidx] = a;
                }
            }
        }
    }
}

// Dynamic-B batched PREQUANT Q8_0 matvec. Each warp owns one output row r; it
// streams the q8_0 weight row ONCE per chunk of up to AXIOM_Q8_BATCH_CAP batch
// elements and computes one prequant dot-product per batch element b:
// dot(weightRow, X[b]) -> Y[b][r]. The per-element dequant+accumulation is
// byte-identical to the single-vector preq warp kernel
// (axiom_q8_0_matvec_f32_preq_warp_kernel): same f16 weight-scale decode, same
// __dp4a int8 block dot (axiom_dev_q8_0_dot_i8_block32), same
// wd*x->d*(float)dot accumulation order over the same lane-strided block loop,
// same shfl_down reduction. So each output element is bit-for-bit identical to
// the single-vector preq path. This mirrors axiom_q8_0_matvec2_preq_warp_kernel
// / axiom_q8_0_matvec4_preq_warp_kernel generalized to dynamic B.
//   xq layout: per-element prequantized tiles, row-major [B][blocks]
//              (xq + (uint64_t)b * blocks).
//   Y layout:  y_layout==0 -> [B][rows] (per-seq contiguous, natural);
//              y_layout==1 -> [rows][B] (per-row contiguous / interleaved).
__global__ static void axiom_q8_0_batched_matvec_f32_preq_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ xq,
        float *__restrict__ y,
        uint32_t batch,
        uint32_t rows,
        uint32_t cols,
        uint32_t y_layout) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    // batch is warp-uniform, so the j<bn predicates below never diverge a warp.
    for (uint32_t b0 = 0; b0 < batch; b0 += AXIOM_Q8_BATCH_CAP) {
        const uint32_t bn = (batch - b0) < AXIOM_Q8_BATCH_CAP ? (batch - b0) : AXIOM_Q8_BATCH_CAP;
        float acc[AXIOM_Q8_BATCH_CAP];
        #pragma unroll
        for (uint32_t j = 0; j < AXIOM_Q8_BATCH_CAP; ++j) acc[j] = 0.0f;
        for (uint32_t b = lane; b < blocks; b += 32u) {
            const uint8_t *blk = r + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int8_t *wq = (const int8_t *)(blk + 2u);
            #pragma unroll
            for (uint32_t j = 0; j < AXIOM_Q8_BATCH_CAP; ++j) {
                if (j < bn) {
                    const axiom_cuda_q8_0_tile32_block *x =
                            xq + (uint64_t)(b0 + j) * blocks + b;
                    acc[j] += wd * x->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
                }
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < AXIOM_Q8_BATCH_CAP; ++j) {
            if (j < bn) {
                float a = acc[j];
                for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                    a += __shfl_down_sync(0xffffffffu, a, offset);
                }
                if (lane == 0u) {
                    const uint32_t bb = b0 + j;
                    const uint64_t yidx = (y_layout == 1u)
                            ? (uint64_t)row * batch + bb
                            : (uint64_t)bb * rows + row;
                    y[yidx] = a;
                }
            }
        }
    }
}

__global__ static void axiom_q8_0_matvec_f32_preq_sxq_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    extern __shared__ axiom_cuda_q8_0_tile32_block sxq[];
    const uint32_t blocks = cols / 32u;
    for (uint32_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        sxq[b] = input[b];
    }
    __syncthreads();

    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = sxq + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
        acc += wd * x->d * (float)dot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_matvec_f32_preq64_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint64_t row_bytes = 64ull * 34ull;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t i = 0u; i < 2u; ++i) {
        const uint32_t b = lane + i * 32u;
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
        acc += wd * x->d * (float)dot;
    }
    #pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_dual_matvec_f32_warp_kernel(
        const uint8_t *weight_a,
        const uint8_t *weight_b,
        const float *input,
        float *out_a,
        float *out_b,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t rows = rows_a + rows_b;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    if (row < rows_a) {
        const uint8_t *r = weight_a + (uint64_t)row * row_bytes;
        const float acc = axiom_dev_q8_0_dot_f32_warp(r, input, blocks);
        if (lane == 0u) out_a[row] = acc;
    } else {
        const uint32_t row_b = row - rows_a;
        const uint8_t *r = weight_b + (uint64_t)row_b * row_bytes;
        const float acc = axiom_dev_q8_0_dot_f32_warp(r, input, blocks);
        if (lane == 0u) out_b[row_b] = acc;
    }
}

__global__ static void axiom_q8_0_dual_matvec_preq_pair_warp_kernel(
        const uint8_t *__restrict__ weight_a,
        const uint8_t *__restrict__ weight_b,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out_a,
        float *__restrict__ out_b,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows_a && row >= rows_b) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        if (row < rows_a) {
            const uint8_t *blk = weight_a + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int32_t dot = axiom_dev_q8_0_dot_i8_block32((const int8_t *)(blk + 2u), x->qs);
            acc_a += wd * x->d * (float)dot;
        }
        if (row < rows_b) {
            const uint8_t *blk = weight_b + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int32_t dot = axiom_dev_q8_0_dot_i8_block32((const int8_t *)(blk + 2u), x->qs);
            acc_b += wd * x->d * (float)dot;
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0u) {
        if (row < rows_a) out_a[row] = acc_a;
        if (row < rows_b) out_b[row] = acc_b;
    }
}

__global__ static void axiom_q8_0_dual_matvec_silu_preq_pair_warp_kernel(
        const uint8_t *__restrict__ weight_gate,
        const uint8_t *__restrict__ weight_up,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        const uint8_t *gblk = weight_gate + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
        const float gd = axiom_dev_f16_to_f32(axiom_dev_le16(gblk));
        const int32_t gdot = axiom_dev_q8_0_dot_i8_block32((const int8_t *)(gblk + 2u), x->qs);
        gate += gd * x->d * (float)gdot;
        const uint8_t *ublk = weight_up + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
        const float ud = axiom_dev_f16_to_f32(axiom_dev_le16(ublk));
        const int32_t udot = axiom_dev_q8_0_dot_i8_block32((const int8_t *)(ublk + 2u), x->qs);
        up += ud * x->d * (float)udot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if (lane == 0u) out[row] = (gate / (1.0f + expf(-gate))) * up;
}

__global__ static void axiom_q8_0_qkv_matvec_preq_warp_kernel(
        const uint8_t *__restrict__ weight_q,
        const uint8_t *__restrict__ weight_k,
        const uint8_t *__restrict__ weight_v,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out_q,
        float *__restrict__ out_k,
        float *__restrict__ out_v,
        uint32_t rows_q,
        uint32_t rows_k,
        uint32_t rows_v,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows_q && row >= rows_k && row >= rows_v) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    float acc_q = 0.0f;
    float acc_k = 0.0f;
    float acc_v = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        if (row < rows_q) {
            const uint8_t *blk = weight_q + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int32_t dot = axiom_dev_q8_0_dot_i8_block32((const int8_t *)(blk + 2u), x->qs);
            acc_q += wd * x->d * (float)dot;
        }
        if (row < rows_k) {
            const uint8_t *blk = weight_k + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int32_t dot = axiom_dev_q8_0_dot_i8_block32((const int8_t *)(blk + 2u), x->qs);
            acc_k += wd * x->d * (float)dot;
        }
        if (row < rows_v) {
            const uint8_t *blk = weight_v + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int32_t dot = axiom_dev_q8_0_dot_i8_block32((const int8_t *)(blk + 2u), x->qs);
            acc_v += wd * x->d * (float)dot;
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc_q += __shfl_down_sync(0xffffffffu, acc_q, offset);
        acc_k += __shfl_down_sync(0xffffffffu, acc_k, offset);
        acc_v += __shfl_down_sync(0xffffffffu, acc_v, offset);
    }
    if (lane == 0u) {
        if (row < rows_q) out_q[row] = acc_q;
        if (row < rows_k) out_k[row] = acc_k;
        if (row < rows_v) out_v[row] = acc_v;
    }
}

__global__ static void axiom_q8_0_dual_matvec2_preq_pair_warp_kernel(
        const uint8_t *__restrict__ weight_a,
        const uint8_t *__restrict__ weight_b,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        float *__restrict__ out_a0,
        float *__restrict__ out_b0,
        float *__restrict__ out_a1,
        float *__restrict__ out_b1,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows_a && row >= rows_b) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    float acc_a0 = 0.0f;
    float acc_b0 = 0.0f;
    float acc_a1 = 0.0f;
    float acc_b1 = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const axiom_cuda_q8_0_tile32_block *x0 = input0 + b;
        const axiom_cuda_q8_0_tile32_block *x1 = input1 + b;
        if (row < rows_a) {
            const uint8_t *blk = weight_a + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int8_t *wq = (const int8_t *)(blk + 2u);
            acc_a0 += wd * x0->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x0->qs);
            acc_a1 += wd * x1->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x1->qs);
        }
        if (row < rows_b) {
            const uint8_t *blk = weight_b + (uint64_t)row * row_bytes + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int8_t *wq = (const int8_t *)(blk + 2u);
            acc_b0 += wd * x0->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x0->qs);
            acc_b1 += wd * x1->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x1->qs);
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc_a0 += __shfl_down_sync(0xffffffffu, acc_a0, offset);
        acc_b0 += __shfl_down_sync(0xffffffffu, acc_b0, offset);
        acc_a1 += __shfl_down_sync(0xffffffffu, acc_a1, offset);
        acc_b1 += __shfl_down_sync(0xffffffffu, acc_b1, offset);
    }
    if (lane == 0u) {
        if (row < rows_a) {
            out_a0[row] = acc_a0;
            out_a1[row] = acc_a1;
        }
        if (row < rows_b) {
            out_b0[row] = acc_b0;
            out_b1[row] = acc_b1;
        }
    }
}

__device__ static float axiom_dev_q8_0_dot_f32(
        const uint8_t *row,
        const float *input,
        uint32_t blocks) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 34u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2u);
        const uint64_t base = (uint64_t)b * 32u;
        for (uint32_t i = 0; i < 32u; ++i) {
            acc += d * (float)qs[i] * __ldg(input + base + i);
        }
    }
    return acc;
}

__device__ static float axiom_dev_q8_0_dot_f32_warp(
        const uint8_t *row,
        const float *input,
        uint32_t blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 34u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2u);
        acc += d * (float)qs[lane] * __ldg(input + (uint64_t)b * 32u + lane);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__device__ static float axiom_dev_q8_0_dot_f32_warp_shared(
        const uint8_t *row,
        const float *input,
        uint32_t blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 34u;
        const float d = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *qs = (const int8_t *)(blk + 2u);
        acc += d * (float)qs[lane] * input[(uint64_t)b * 32u + lane];
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__device__ static float axiom_dev_q8_0_dot_q8tile_qwarp8(
        const uint8_t *__restrict__ row,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        uint32_t blocks,
        uint32_t lane8,
        uint32_t mask) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        const int32_t sum = __dp4a(axiom_dev_q8k_pack4(wq, lane8), axiom_dev_q8k_pack4(x->qs, lane8), 0);
        acc += wd * x->d * (float)sum;
    }
    for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(mask, acc, offset, 8);
    }
    return acc;
}

__device__ static void axiom_dev_q8_0_dual_dot_f32_warp(
        const uint8_t *gate_row,
        const uint8_t *up_row,
        const float *input,
        uint32_t blocks,
        float *gate_out,
        float *up_out) {
    const uint32_t lane = threadIdx.x & 31u;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * 34u;
        const uint8_t *up_blk = up_row + (uint64_t)b * 34u;
        const float gate_d = axiom_dev_f16_to_f32(axiom_dev_le16(gate_blk));
        const float up_d = axiom_dev_f16_to_f32(axiom_dev_le16(up_blk));
        const int8_t *gate_qs = (const int8_t *)(gate_blk + 2u);
        const int8_t *up_qs = (const int8_t *)(up_blk + 2u);
        const float x = __ldg(input + (uint64_t)b * 32u + lane);
        gate_acc += gate_d * (float)gate_qs[lane] * x;
        up_acc += up_d * (float)up_qs[lane] * x;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__device__ static void axiom_dev_q8_0_dual_dot_q8tile_warp(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        uint32_t blocks,
        float *gate_out,
        float *up_out) {
    const uint32_t lane = threadIdx.x & 31u;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * 34u;
        const uint8_t *up_blk = up_row + (uint64_t)b * 34u;
        const float gate_d = axiom_dev_f16_to_f32(axiom_dev_le16(gate_blk));
        const float up_d = axiom_dev_f16_to_f32(axiom_dev_le16(up_blk));
        const int8_t *gate_qs = (const int8_t *)(gate_blk + 2u);
        const int8_t *up_qs = (const int8_t *)(up_blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        gate_acc += gate_d * x->d * (float)axiom_dev_q8_0_dot_i8_block32(gate_qs, x->qs);
        up_acc += up_d * x->d * (float)axiom_dev_q8_0_dot_i8_block32(up_qs, x->qs);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__global__ static void axiom_q8_0_grouped_matvec_f32_warp_kernel(
        const uint8_t *weight,
        const float *input,
        float *out,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t rows = groups * rows_per_group;
    if (row >= rows) return;
    const uint32_t group = row / rows_per_group;
    const uint32_t local_row = row - group * rows_per_group;
    const uint32_t blocks = input_per_group / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint64_t matrix_row = (uint64_t)group * rows_per_group + local_row;
    const uint8_t *r = weight + matrix_row * row_bytes;
    const float *x = input + (uint64_t)group * input_per_group;
    const float acc = axiom_dev_q8_0_dot_f32_warp(r, x, blocks);
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_soa_grouped_matvec_f32_warp_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const float *__restrict__ input,
        float *__restrict__ out,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t rows = groups * rows_per_group;
    if (row >= rows) return;
    const uint32_t group = row / rows_per_group;
    const uint32_t local_row = row - group * rows_per_group;
    const uint32_t blocks = input_per_group / 32u;
    const uint64_t matrix_row = (uint64_t)group * rows_per_group + local_row;
    const int8_t *qrow = qs + matrix_row * input_per_group;
    const uint16_t *srow = scales + matrix_row * blocks;
    const float *x = input + (uint64_t)group * input_per_group;
    const float acc = axiom_dev_q8_0_soa_dot_f32_warp(qrow, srow, x, blocks);
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_soa_grouped_matvec_f32_blocklane_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const float *__restrict__ input,
        float *__restrict__ out,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t rows = groups * rows_per_group;
    if (row >= rows) return;
    const uint32_t group = row / rows_per_group;
    const uint32_t local_row = row - group * rows_per_group;
    const uint32_t blocks = input_per_group / 32u;
    const uint64_t matrix_row = (uint64_t)group * rows_per_group + local_row;
    const int8_t *qrow = qs + matrix_row * input_per_group;
    const uint16_t *srow = scales + matrix_row * blocks;
    const float *x = input + (uint64_t)group * input_per_group;
    const float acc = axiom_dev_q8_0_soa_dot_f32_blocklane(qrow, srow, x, blocks);
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_grouped_matvec_f32_warp_ds4_o_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ out) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= 8192u) return;
    const uint32_t group = row >> 10u;
    const uint8_t *r = weight + (uint64_t)row * (128ull * 34ull);
    const float *x = input + (uint64_t)group * 4096ull;
    const float acc = axiom_dev_q8_0_dot_f32_warp_const<128>(r, x);
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_grouped_matvec_preq_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t rows = groups * rows_per_group;
    if (row >= rows) return;
    const uint32_t group = row / rows_per_group;
    const uint32_t local_row = row - group * rows_per_group;
    const uint32_t blocks = input_per_group / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint64_t matrix_row = (uint64_t)group * rows_per_group + local_row;
    const uint8_t *r = weight + matrix_row * row_bytes;
    const axiom_cuda_q8_0_tile32_block *xrow = input + (uint64_t)group * blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = xrow + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
        acc += wd * x->d * (float)dot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_grouped_matvec_preq_sxq_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    extern __shared__ axiom_cuda_q8_0_tile32_block sxq[];
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t rows = groups * rows_per_group;
    const uint32_t row_base = (uint32_t)blockIdx.x * warps_per_block;
    const uint32_t row_last = row_base + warps_per_block - 1u < rows ?
            row_base + warps_per_block - 1u : rows - 1u;
    const uint32_t blocks = input_per_group / 32u;
    const uint32_t group_base = row_base / rows_per_group;
    const uint32_t group_last = row_last / rows_per_group;
    const int shared_group = group_base == group_last;
    if (shared_group) {
        const axiom_cuda_q8_0_tile32_block *xg = input + (uint64_t)group_base * blocks;
        for (uint32_t b = threadIdx.x; b < blocks; b += blockDim.x) {
            sxq[b] = xg[b];
        }
    }
    __syncthreads();

    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = row_base + warp;
    if (row >= rows) return;
    const uint32_t group = row / rows_per_group;
    const uint32_t local_row = row - group * rows_per_group;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint64_t matrix_row = (uint64_t)group * rows_per_group + local_row;
    const uint8_t *r = weight + matrix_row * row_bytes;
    const axiom_cuda_q8_0_tile32_block *xrow = shared_group ? sxq :
            input + (uint64_t)group * blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = xrow + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
        acc += wd * x->d * (float)dot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_soa_matvec_preq_warp_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const int8_t *qrow = qs + (uint64_t)row * cols;
    const uint16_t *srow = scales + (uint64_t)row * blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32_aligned(qrow + (uint64_t)b * 32u, x->qs);
        acc += axiom_dev_f16_to_f32(srow[b]) * x->d * (float)dot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_soa_matvec2_preq_warp_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        float *__restrict__ out0,
        float *__restrict__ out1,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const int8_t *qrow = qs + (uint64_t)row * cols;
    const uint16_t *srow = scales + (uint64_t)row * blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const int8_t *wq = qrow + (uint64_t)b * 32u;
        const float wd = axiom_dev_f16_to_f32(srow[b]);
        const axiom_cuda_q8_0_tile32_block *x0 = input0 + b;
        const axiom_cuda_q8_0_tile32_block *x1 = input1 + b;
        const int32_t dot0 = axiom_dev_q8_0_dot_i8_block32_aligned(wq, x0->qs);
        const int32_t dot1 = axiom_dev_q8_0_dot_i8_block32_aligned(wq, x1->qs);
        acc0 += wd * x0->d * (float)dot0;
        acc1 += wd * x1->d * (float)dot1;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
    }
}

__global__ static void axiom_q8_0_soa_matvec4_preq_warp_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input2,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input3,
        float *__restrict__ out0,
        float *__restrict__ out1,
        float *__restrict__ out2,
        float *__restrict__ out3,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const int8_t *qrow = qs + (uint64_t)row * cols;
    const uint16_t *srow = scales + (uint64_t)row * blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const int8_t *wq = qrow + (uint64_t)b * 32u;
        const float wd = axiom_dev_f16_to_f32(srow[b]);
        const axiom_cuda_q8_0_tile32_block *x0 = input0 + b;
        const axiom_cuda_q8_0_tile32_block *x1 = input1 + b;
        const axiom_cuda_q8_0_tile32_block *x2 = input2 + b;
        const axiom_cuda_q8_0_tile32_block *x3 = input3 + b;
        acc0 += wd * x0->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(wq, x0->qs);
        acc1 += wd * x1->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(wq, x1->qs);
        acc2 += wd * x2->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(wq, x2->qs);
        acc3 += wd * x3->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(wq, x3->qs);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
        out2[row] = acc2;
        out3[row] = acc3;
    }
}

__global__ static void axiom_q8_0_soa_grouped_matvec_preq_warp_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ out,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t rows = groups * rows_per_group;
    if (row >= rows) return;
    const uint32_t group = row / rows_per_group;
    const uint32_t local_row = row - group * rows_per_group;
    const uint32_t blocks = input_per_group / 32u;
    const uint64_t matrix_row = (uint64_t)group * rows_per_group + local_row;
    const int8_t *qrow = qs + matrix_row * input_per_group;
    const uint16_t *srow = scales + matrix_row * blocks;
    const axiom_cuda_q8_0_tile32_block *xrow = input + (uint64_t)group * blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const axiom_cuda_q8_0_tile32_block *x = xrow + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32_aligned(qrow + (uint64_t)b * 32u, x->qs);
        acc += axiom_dev_f16_to_f32(srow[b]) * x->d * (float)dot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_soa_grouped_matvec2_preq_warp_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        float *__restrict__ out0,
        float *__restrict__ out1,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t rows = groups * rows_per_group;
    if (row >= rows) return;
    const uint32_t group = row / rows_per_group;
    const uint32_t local_row = row - group * rows_per_group;
    const uint32_t blocks = input_per_group / 32u;
    const uint64_t matrix_row = (uint64_t)group * rows_per_group + local_row;
    const int8_t *qrow = qs + matrix_row * input_per_group;
    const uint16_t *srow = scales + matrix_row * blocks;
    const axiom_cuda_q8_0_tile32_block *x0row = input0 + (uint64_t)group * blocks;
    const axiom_cuda_q8_0_tile32_block *x1row = input1 + (uint64_t)group * blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const int8_t *wq = qrow + (uint64_t)b * 32u;
        const float wd = axiom_dev_f16_to_f32(srow[b]);
        const axiom_cuda_q8_0_tile32_block *x0 = x0row + b;
        const axiom_cuda_q8_0_tile32_block *x1 = x1row + b;
        const int32_t dot0 = axiom_dev_q8_0_dot_i8_block32_aligned(wq, x0->qs);
        const int32_t dot1 = axiom_dev_q8_0_dot_i8_block32_aligned(wq, x1->qs);
        acc0 += wd * x0->d * (float)dot0;
        acc1 += wd * x1->d * (float)dot1;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
    }
}

__global__ static void axiom_q8_0_soa_grouped_matvec4_preq_warp_kernel(
        const int8_t *__restrict__ qs,
        const uint16_t *__restrict__ scales,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input2,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input3,
        float *__restrict__ out0,
        float *__restrict__ out1,
        float *__restrict__ out2,
        float *__restrict__ out3,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t rows = groups * rows_per_group;
    if (row >= rows) return;
    const uint32_t group = row / rows_per_group;
    const uint32_t local_row = row - group * rows_per_group;
    const uint32_t blocks = input_per_group / 32u;
    const uint64_t matrix_row = (uint64_t)group * rows_per_group + local_row;
    const int8_t *qrow = qs + matrix_row * input_per_group;
    const uint16_t *srow = scales + matrix_row * blocks;
    const axiom_cuda_q8_0_tile32_block *x0row = input0 + (uint64_t)group * blocks;
    const axiom_cuda_q8_0_tile32_block *x1row = input1 + (uint64_t)group * blocks;
    const axiom_cuda_q8_0_tile32_block *x2row = input2 + (uint64_t)group * blocks;
    const axiom_cuda_q8_0_tile32_block *x3row = input3 + (uint64_t)group * blocks;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const int8_t *wq = qrow + (uint64_t)b * 32u;
        const float wd = axiom_dev_f16_to_f32(srow[b]);
        const axiom_cuda_q8_0_tile32_block *x0 = x0row + b;
        const axiom_cuda_q8_0_tile32_block *x1 = x1row + b;
        const axiom_cuda_q8_0_tile32_block *x2 = x2row + b;
        const axiom_cuda_q8_0_tile32_block *x3 = x3row + b;
        acc0 += wd * x0->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(wq, x0->qs);
        acc1 += wd * x1->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(wq, x1->qs);
        acc2 += wd * x2->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(wq, x2->qs);
        acc3 += wd * x3->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(wq, x3->qs);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }
    if (lane == 0u) {
        out0[row] = acc0;
        out1[row] = acc1;
        out2[row] = acc2;
        out3[row] = acc3;
    }
}

__global__ static void axiom_q2_k_matvec_f32_kernel(
        const uint8_t *weight,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t blocks = cols / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 84u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    out[row] = axiom_dev_q2_k_dot_f32(r, input, blocks);
}

__global__ static void axiom_iq2_xxs_matvec_f32_kernel(
        const uint8_t *weight,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t blocks = cols / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    out[row] = axiom_dev_iq2_xxs_dot_f32(r, input, blocks);
}

__global__ static void axiom_iq2_xxs_matvec_f32_warp_kernel(
        const uint8_t *weight,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    const float acc = axiom_dev_iq2_xxs_dot_f32_warp(r, input, blocks);
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_q8_0_matvec_argmax_f32_stage1_kernel(
        const uint8_t *weight,
        const float *input,
        uint32_t rows,
        uint32_t cols,
        axiom_cuda_argmax_pair *partials) {
    __shared__ axiom_cuda_argmax_pair shared[32];
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row0 = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t row_stride = (uint32_t)gridDim.x * warps_per_block;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    axiom_cuda_argmax_pair best;
    best.value = -INFINITY;
    best.index = UINT32_MAX;
    for (uint32_t row = row0; row < rows; row += row_stride) {
        const uint8_t *r = weight + (uint64_t)row * row_bytes;
        const float acc = axiom_dev_q8_0_dot_f32_warp(r, input, blocks);
        if (lane == 0u) {
            axiom_cuda_argmax_pair cur;
            cur.value = acc;
            cur.index = row;
            best = axiom_argmax_better(best, cur);
        }
    }
    if (lane == 0u) shared[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0u) {
        axiom_cuda_argmax_pair out = shared[0];
        for (uint32_t i = 1u; i < warps_per_block; ++i) {
            out = axiom_argmax_better(out, shared[i]);
        }
        partials[blockIdx.x] = out;
    }
}

__global__ static void axiom_q8_0_matvec_argmax_preq_stage1_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        uint32_t rows,
        uint32_t cols,
        axiom_cuda_argmax_pair *partials) {
    __shared__ axiom_cuda_argmax_pair shared[64];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t qwarp = lane >> 3u;
    const uint32_t lane8 = lane & 7u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t qwarp_in_block = (threadIdx.x >> 3u);
    const uint32_t qwarps_per_block = blockDim.x >> 3u;
    const uint32_t row0 = (uint32_t)blockIdx.x * qwarps_per_block + warp * 4u + qwarp;
    const uint32_t row_stride = (uint32_t)gridDim.x * qwarps_per_block;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint32_t mask = 0xffu << (qwarp * 8u);
    axiom_cuda_argmax_pair best;
    best.value = -INFINITY;
    best.index = UINT32_MAX;
    for (uint32_t row = row0; row < rows; row += row_stride) {
        const uint8_t *r = weight + (uint64_t)row * row_bytes;
        const float acc = axiom_dev_q8_0_dot_q8tile_qwarp8(r, input, blocks, lane8, mask);
        if (lane8 == 0u) {
            axiom_cuda_argmax_pair cur;
            cur.value = acc;
            cur.index = row;
            best = axiom_argmax_better(best, cur);
        }
    }
    if (lane8 == 0u) shared[qwarp_in_block] = best;
    __syncthreads();
    if (threadIdx.x == 0u) {
        axiom_cuda_argmax_pair out = shared[0];
        for (uint32_t i = 1u; i < qwarps_per_block; ++i) {
            out = axiom_argmax_better(out, shared[i]);
        }
        partials[blockIdx.x] = out;
    }
}

__global__ static void axiom_q8_0_matvec_argmax_preq_warp32_stage1_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        uint32_t rows,
        uint32_t cols,
        axiom_cuda_argmax_pair *partials) {
    __shared__ axiom_cuda_argmax_pair shared[16];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row0 = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t row_stride = (uint32_t)gridDim.x * warps_per_block;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    axiom_cuda_argmax_pair best;
    best.value = -INFINITY;
    best.index = UINT32_MAX;
    for (uint32_t row = row0; row < rows; row += row_stride) {
        const uint8_t *r = weight + (uint64_t)row * row_bytes;
        float acc = 0.0f;
        for (uint32_t b = lane; b < blocks; b += 32u) {
            const uint8_t *blk = r + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int8_t *wq = (const int8_t *)(blk + 2u);
            const axiom_cuda_q8_0_tile32_block *x = input + b;
            const int32_t dot = axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
            acc += wd * x->d * (float)dot;
        }
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0u) {
            axiom_cuda_argmax_pair cur;
            cur.value = acc;
            cur.index = row;
            best = axiom_argmax_better(best, cur);
        }
    }
    if (lane == 0u) shared[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0u) {
        axiom_cuda_argmax_pair out = shared[0];
        for (uint32_t i = 1u; i < warps_per_block; ++i) {
            out = axiom_argmax_better(out, shared[i]);
        }
        partials[blockIdx.x] = out;
    }
}

__global__ static void axiom_q8_0_argmax_exact_value_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ input,
        uint32_t rows,
        uint32_t cols,
        axiom_cuda_argmax_pair *result) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = result->index;
    if (row >= rows || row == UINT32_MAX) {
        if (lane == 0u) result->value = -INFINITY;
        return;
    }
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    const float acc = axiom_dev_q8_0_dot_f32_warp(r, input, blocks);
    if (lane == 0u) result->value = acc;
}

__global__ static void axiom_q8_0_matvec2_argmax_f32_stage1_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ input0,
        const float *__restrict__ input1,
        uint32_t rows,
        uint32_t cols,
        axiom_cuda_argmax_pair *__restrict__ partials0,
        axiom_cuda_argmax_pair *__restrict__ partials1) {
    __shared__ axiom_cuda_argmax_pair shared0[16];
    __shared__ axiom_cuda_argmax_pair shared1[16];
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row0 = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t row_stride = (uint32_t)gridDim.x * warps_per_block;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    axiom_cuda_argmax_pair best0;
    axiom_cuda_argmax_pair best1;
    best0.value = -INFINITY;
    best0.index = UINT32_MAX;
    best1.value = -INFINITY;
    best1.index = UINT32_MAX;
    for (uint32_t row = row0; row < rows; row += row_stride) {
        const uint8_t *r = weight + (uint64_t)row * row_bytes;
        const float acc0 = axiom_dev_q8_0_dot_f32_warp(r, input0, blocks);
        const float acc1 = axiom_dev_q8_0_dot_f32_warp(r, input1, blocks);
        if (lane == 0u) {
            axiom_cuda_argmax_pair cur0;
            axiom_cuda_argmax_pair cur1;
            cur0.value = acc0;
            cur0.index = row;
            cur1.value = acc1;
            cur1.index = row;
            best0 = axiom_argmax_better(best0, cur0);
            best1 = axiom_argmax_better(best1, cur1);
        }
    }
    if (lane == 0u) {
        shared0[warp] = best0;
        shared1[warp] = best1;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        axiom_cuda_argmax_pair out0 = shared0[0];
        axiom_cuda_argmax_pair out1 = shared1[0];
        for (uint32_t i = 1u; i < warps_per_block; ++i) {
            out0 = axiom_argmax_better(out0, shared0[i]);
            out1 = axiom_argmax_better(out1, shared1[i]);
        }
        partials0[blockIdx.x] = out0;
        partials1[blockIdx.x] = out1;
    }
}

__global__ static void axiom_q8_0_matvec2_argmax_preq_warp32_stage1_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        uint32_t rows,
        uint32_t cols,
        axiom_cuda_argmax_pair *__restrict__ partials0,
        axiom_cuda_argmax_pair *__restrict__ partials1) {
    __shared__ axiom_cuda_argmax_pair shared0[16];
    __shared__ axiom_cuda_argmax_pair shared1[16];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row0 = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t row_stride = (uint32_t)gridDim.x * warps_per_block;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    axiom_cuda_argmax_pair best0;
    axiom_cuda_argmax_pair best1;
    best0.value = -INFINITY;
    best0.index = UINT32_MAX;
    best1.value = -INFINITY;
    best1.index = UINT32_MAX;
    for (uint32_t row = row0; row < rows; row += row_stride) {
        const uint8_t *r = weight + (uint64_t)row * row_bytes;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t b = lane; b < blocks; b += 32u) {
            const uint8_t *blk = r + (uint64_t)b * 34u;
            const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
            const int8_t *wq = (const int8_t *)(blk + 2u);
            const axiom_cuda_q8_0_tile32_block *x0 = input0 + b;
            const axiom_cuda_q8_0_tile32_block *x1 = input1 + b;
            acc0 += wd * x0->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x0->qs);
            acc1 += wd * x1->d * (float)axiom_dev_q8_0_dot_i8_block32(wq, x1->qs);
        }
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        }
        if (lane == 0u) {
            axiom_cuda_argmax_pair cur0;
            axiom_cuda_argmax_pair cur1;
            cur0.value = acc0;
            cur0.index = row;
            cur1.value = acc1;
            cur1.index = row;
            best0 = axiom_argmax_better(best0, cur0);
            best1 = axiom_argmax_better(best1, cur1);
        }
    }
    if (lane == 0u) {
        shared0[warp] = best0;
        shared1[warp] = best1;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        axiom_cuda_argmax_pair out0 = shared0[0];
        axiom_cuda_argmax_pair out1 = shared1[0];
        for (uint32_t i = 1u; i < warps_per_block; ++i) {
            out0 = axiom_argmax_better(out0, shared0[i]);
            out1 = axiom_argmax_better(out1, shared1[i]);
        }
        partials0[blockIdx.x] = out0;
        partials1[blockIdx.x] = out1;
    }
}

__global__ static void axiom_q8_0_matvec2_argmax_preq_qwarp8_stage1_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        uint32_t rows,
        uint32_t cols,
        axiom_cuda_argmax_pair *__restrict__ partials0,
        axiom_cuda_argmax_pair *__restrict__ partials1) {
    __shared__ axiom_cuda_argmax_pair shared0[64];
    __shared__ axiom_cuda_argmax_pair shared1[64];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t qwarp = lane >> 3u;
    const uint32_t lane8 = lane & 7u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t qwarp_in_block = threadIdx.x >> 3u;
    const uint32_t qwarps_per_block = blockDim.x >> 3u;
    const uint32_t row0 = (uint32_t)blockIdx.x * qwarps_per_block + warp * 4u + qwarp;
    const uint32_t row_stride = (uint32_t)gridDim.x * qwarps_per_block;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint32_t mask = 0xffu << (qwarp * 8u);
    axiom_cuda_argmax_pair best0;
    axiom_cuda_argmax_pair best1;
    best0.value = -INFINITY;
    best0.index = UINT32_MAX;
    best1.value = -INFINITY;
    best1.index = UINT32_MAX;
    for (uint32_t row = row0; row < rows; row += row_stride) {
        const uint8_t *r = weight + (uint64_t)row * row_bytes;
        const float acc0 = axiom_dev_q8_0_dot_q8tile_qwarp8(r, input0, blocks, lane8, mask);
        const float acc1 = axiom_dev_q8_0_dot_q8tile_qwarp8(r, input1, blocks, lane8, mask);
        if (lane8 == 0u) {
            axiom_cuda_argmax_pair cur0;
            axiom_cuda_argmax_pair cur1;
            cur0.value = acc0;
            cur0.index = row;
            cur1.value = acc1;
            cur1.index = row;
            best0 = axiom_argmax_better(best0, cur0);
            best1 = axiom_argmax_better(best1, cur1);
        }
    }
    if (lane8 == 0u) {
        shared0[qwarp_in_block] = best0;
        shared1[qwarp_in_block] = best1;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        axiom_cuda_argmax_pair out0 = shared0[0];
        axiom_cuda_argmax_pair out1 = shared1[0];
        for (uint32_t i = 1u; i < qwarps_per_block; ++i) {
            out0 = axiom_argmax_better(out0, shared0[i]);
            out1 = axiom_argmax_better(out1, shared1[i]);
        }
        partials0[blockIdx.x] = out0;
        partials1[blockIdx.x] = out1;
    }
}

__global__ static void axiom_argmax_pair_stage2_kernel(
        const axiom_cuda_argmax_pair *__restrict__ partials0,
        const axiom_cuda_argmax_pair *__restrict__ partials1,
        uint32_t count,
        axiom_cuda_argmax_pair *__restrict__ out0,
        axiom_cuda_argmax_pair *__restrict__ out1) {
    extern __shared__ axiom_cuda_argmax_pair shared[];
    axiom_cuda_argmax_pair *shared0 = shared;
    axiom_cuda_argmax_pair *shared1 = shared + blockDim.x;
    const uint32_t tid = threadIdx.x;
    axiom_cuda_argmax_pair best0;
    axiom_cuda_argmax_pair best1;
    best0.value = -INFINITY;
    best0.index = UINT32_MAX;
    best1.value = -INFINITY;
    best1.index = UINT32_MAX;
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        best0 = axiom_argmax_better(best0, partials0[i]);
        best1 = axiom_argmax_better(best1, partials1[i]);
    }
    shared0[tid] = best0;
    shared1[tid] = best1;
    __syncthreads();
    for (uint32_t width = blockDim.x; width > 1u; ) {
        const uint32_t half = (width + 1u) >> 1u;
        if (tid < width - half) {
            shared0[tid] = axiom_argmax_better(shared0[tid], shared0[tid + half]);
            shared1[tid] = axiom_argmax_better(shared1[tid], shared1[tid + half]);
        }
        __syncthreads();
        width = half;
    }
    if (tid == 0u) {
        out0[0] = shared0[0];
        out1[0] = shared1[0];
    }
}

__global__ static void axiom_q8_0_argmax2_exact_value_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ input0,
        const float *__restrict__ input1,
        uint32_t rows,
        uint32_t cols,
        axiom_cuda_argmax_pair *__restrict__ result0,
        axiom_cuda_argmax_pair *__restrict__ result1) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (warp > 1u) return;
    axiom_cuda_argmax_pair *result = warp == 0u ? result0 : result1;
    const float *input = warp == 0u ? input0 : input1;
    const uint32_t row = result->index;
    if (row >= rows || row == UINT32_MAX) {
        if (lane == 0u) result->value = -INFINITY;
        return;
    }
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    const float acc = axiom_dev_q8_0_dot_f32_warp(r, input, blocks);
    if (lane == 0u) result->value = acc;
}

extern "C" int axiom_cuda_q8_0_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q8 || !input || !out ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const uint32_t blocks = cols / 32u;
    const size_t shared_bytes = (size_t)cols * sizeof(float);
    const bool lm_head_full_logits = rows == 129280u && cols == 4096u;
    const bool use_dp4a = !lm_head_full_logits &&
            axiom_cuda_env_enabled("AXIOM_DS4_Q8_DP4A");
    const bool use_preq_warp = !lm_head_full_logits && !use_dp4a &&
            !axiom_cuda_env_enabled("AXIOM_DS4_Q8_PREQ_DISABLE");
    if (use_preq_warp || use_dp4a) {
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        if (use_preq_warp) {
            const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
            if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_PREQ_SXQ") && cols <= 4096u) {
                const size_t sxq_bytes = (size_t)blocks * sizeof(axiom_cuda_q8_0_tile32_block);
                axiom_q8_0_matvec_f32_preq_sxq_warp_kernel<<<grid, block, sxq_bytes>>>(
                        (const uint8_t *)wbuf->ptr + weight_offset,
                        xq,
                        (float *)((uint8_t *)obuf->ptr + out_offset),
                        rows,
                        cols);
            } else {
                axiom_q8_0_matvec_f32_preq_warp_kernel<<<grid, block>>>(
                        (const uint8_t *)wbuf->ptr + weight_offset,
                        xq,
                        (float *)((uint8_t *)obuf->ptr + out_offset),
                        rows,
                        cols);
            }
        } else {
            const int rows_per_block = warps_per_block * 4;
            const int grid = (int)((rows + (uint32_t)rows_per_block - 1u) / (uint32_t)rows_per_block);
            axiom_q8_0_matvec_f32_dp4a_kernel<<<grid, block>>>(
                    (const uint8_t *)wbuf->ptr + weight_offset,
                    xq,
                    (float *)((uint8_t *)obuf->ptr + out_offset),
                    rows,
                    cols);
        }
    } else if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_SHAPE_CONST") &&
               rows == 32768u && cols == 1024u) {
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        axiom_q8_0_matvec_f32_warp_const_kernel<32><<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                rows);
    } else if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_SHAPE_CONST") &&
               rows == 4096u && cols == 8192u) {
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        axiom_q8_0_matvec_f32_warp_const_kernel<256><<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                rows);
    } else if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_MATVEC_SX") && shared_bytes <= 49152u) {
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        axiom_q8_0_matvec_f32_warp_sx_kernel<<<grid, block, shared_bytes>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                rows,
                cols);
    } else {
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        axiom_q8_0_matvec_f32_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                rows,
                cols);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_matvec2_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q8 || !input0 || !input1 || !out0 || !out1 ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q8;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const bool lm_head_full_logits = rows == 129280u && cols == 4096u;
    if (lm_head_full_logits) {
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        axiom_q8_0_matvec2_f32_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
                (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
                (float *)((uint8_t *)o0buf->ptr + out0_offset),
                (float *)((uint8_t *)o1buf->ptr + out1_offset),
                rows,
                cols);
        err = cudaGetLastError();
        return axiom_cuda_finish_after_launch(err);
    }
    const uint32_t blocks = cols / 32u;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks * 2u, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_cuda_q8_0_tile32_block *x0 = xq;
    axiom_cuda_q8_0_tile32_block *x1 = xq + blocks;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
            x0,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
            x1,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_q8_0_matvec2_preq_warp_kernel<<<grid, block>>>(
            (const uint8_t *)wbuf->ptr + weight_offset,
            x0,
            x1,
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_matvec4_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        const void *input2,
        uint64_t input2_offset,
        const void *input3,
        uint64_t input3_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
        uint64_t out3_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q8 || !input0 || !input1 || !input2 || !input3 ||
        !out0 || !out1 || !out2 || !out3 ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q8;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    const axiom_cuda_buffer *i2buf = (const axiom_cuda_buffer *)input2;
    const axiom_cuda_buffer *i3buf = (const axiom_cuda_buffer *)input3;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    axiom_cuda_buffer *o2buf = (axiom_cuda_buffer *)out2;
    axiom_cuda_buffer *o3buf = (axiom_cuda_buffer *)out3;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    const bool lm_head_full_logits = rows == 129280u && cols == 4096u;
    if (lm_head_full_logits) {
        axiom_q8_0_matvec4_f32_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
                (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
                (const float *)((const uint8_t *)i2buf->ptr + input2_offset),
                (const float *)((const uint8_t *)i3buf->ptr + input3_offset),
                (float *)((uint8_t *)o0buf->ptr + out0_offset),
                (float *)((uint8_t *)o1buf->ptr + out1_offset),
                (float *)((uint8_t *)o2buf->ptr + out2_offset),
                (float *)((uint8_t *)o3buf->ptr + out3_offset),
                rows,
                cols);
        err = cudaGetLastError();
        return axiom_cuda_finish_after_launch(err);
    }
    const uint32_t blocks = cols / 32u;
    if (blocks > 0x3fffffffu) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks * 4u, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_cuda_q8_0_tile32_block *x0 = xq;
    axiom_cuda_q8_0_tile32_block *x1 = xq + blocks;
    axiom_cuda_q8_0_tile32_block *x2 = xq + (uint64_t)blocks * 2u;
    axiom_cuda_q8_0_tile32_block *x3 = xq + (uint64_t)blocks * 3u;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
            x0,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
            x1,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i2buf->ptr + input2_offset),
            x2,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i3buf->ptr + input3_offset),
            x3,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_matvec4_preq_warp_kernel<<<grid, block>>>(
            (const uint8_t *)wbuf->ptr + weight_offset,
            x0,
            x1,
            x2,
            x3,
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            (float *)((uint8_t *)o2buf->ptr + out2_offset),
            (float *)((uint8_t *)o3buf->ptr + out3_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_batched_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *x,
        uint64_t x_offset,
        void *y,
        uint64_t y_offset,
        uint32_t y_layout,
        uint32_t batch,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q8 || !x || !y ||
        batch == 0 || rows == 0 || cols == 0 || (cols % 32u) != 0u || y_layout > 1u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q8;
    const axiom_cuda_buffer *xbuf = (const axiom_cuda_buffer *)x;
    axiom_cuda_buffer *ybuf = (axiom_cuda_buffer *)y;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    // Mirror axiom_cuda_q8_0_matvec_f32_device's dispatch: default to the
    // prequant warp path (bit-exact, per row, to the single preq path), and
    // fall back to the plain batched warp kernel (bit-exact to the single plain
    // path) when AXIOM_DS4_Q8_PREQ_DISABLE is set or for the lm_head full-logits
    // shape (which the single entry never routes through preq).
    const bool lm_head_full_logits = rows == 129280u && cols == 4096u;
    const bool use_preq_warp = !lm_head_full_logits &&
            !axiom_cuda_env_enabled("AXIOM_DS4_Q8_PREQ_DISABLE");
    if (use_preq_warp) {
        const uint32_t blocks = cols / 32u;
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks * batch, &xq);
        if (rc != AXIOM_OK) return rc;
        const float *xbase = (const float *)((const uint8_t *)xbuf->ptr + x_offset);
        for (uint32_t b = 0; b < batch; ++b) {
            axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
                    xbase + (uint64_t)b * cols,
                    xq + (uint64_t)b * blocks,
                    blocks);
            err = cudaGetLastError();
            if (err != cudaSuccess) return axiom_cuda_status(err);
        }
        axiom_q8_0_batched_matvec_f32_preq_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                xq,
                (float *)((uint8_t *)ybuf->ptr + y_offset),
                batch,
                rows,
                cols,
                y_layout);
    } else {
        axiom_q8_0_batched_matvec_f32_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)xbuf->ptr + x_offset),
                (float *)((uint8_t *)ybuf->ptr + y_offset),
                batch,
                rows,
                cols,
                y_layout);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_soa_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_qs_i8 || !weight_scales_f16 || !input || !out ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)weight_qs_i8;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)weight_scales_f16;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_SOA_PREQ")) {
        const uint32_t blocks = cols / 32u;
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_q8_0_soa_matvec_preq_warp_kernel<<<grid, block>>>(
                (const int8_t *)qbuf->ptr,
                (const uint16_t *)sbuf->ptr,
                xq,
                (float *)((uint8_t *)obuf->ptr + out_offset),
                rows,
                cols);
    } else if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_SOA_BLOCKLANE")) {
        axiom_q8_0_soa_matvec_f32_blocklane_kernel<<<grid, block>>>(
                (const int8_t *)qbuf->ptr,
                (const uint16_t *)sbuf->ptr,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                rows,
                cols);
    } else {
        axiom_q8_0_soa_matvec_f32_warp_kernel<<<grid, block>>>(
                (const int8_t *)qbuf->ptr,
                (const uint16_t *)sbuf->ptr,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                rows,
                cols);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_soa_matvec2_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_qs_i8 || !weight_scales_f16 ||
        !input0 || !input1 || !out0 || !out1 ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)weight_qs_i8;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)weight_scales_f16;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    const uint32_t blocks = cols / 32u;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks * 2u, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_cuda_q8_0_tile32_block *x0 = xq;
    axiom_cuda_q8_0_tile32_block *x1 = xq + blocks;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
            x0,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
            x1,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_soa_matvec2_preq_warp_kernel<<<grid, block>>>(
            (const int8_t *)qbuf->ptr,
            (const uint16_t *)sbuf->ptr,
            x0,
            x1,
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_soa_matvec4_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        const void *input2,
        uint64_t input2_offset,
        const void *input3,
        uint64_t input3_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
        uint64_t out3_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_qs_i8 || !weight_scales_f16 ||
        !input0 || !input1 || !input2 || !input3 ||
        !out0 || !out1 || !out2 || !out3 ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)weight_qs_i8;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)weight_scales_f16;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    const axiom_cuda_buffer *i2buf = (const axiom_cuda_buffer *)input2;
    const axiom_cuda_buffer *i3buf = (const axiom_cuda_buffer *)input3;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    axiom_cuda_buffer *o2buf = (axiom_cuda_buffer *)out2;
    axiom_cuda_buffer *o3buf = (axiom_cuda_buffer *)out3;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    const uint32_t blocks = cols / 32u;
    if (blocks > 0x3fffffffu) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks * 4u, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_cuda_q8_0_tile32_block *x0 = xq;
    axiom_cuda_q8_0_tile32_block *x1 = xq + blocks;
    axiom_cuda_q8_0_tile32_block *x2 = xq + (uint64_t)blocks * 2u;
    axiom_cuda_q8_0_tile32_block *x3 = xq + (uint64_t)blocks * 3u;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
            x0,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
            x1,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i2buf->ptr + input2_offset),
            x2,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i3buf->ptr + input3_offset),
            x3,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_soa_matvec4_preq_warp_kernel<<<grid, block>>>(
            (const int8_t *)qbuf->ptr,
            (const uint16_t *)sbuf->ptr,
            x0,
            x1,
            x2,
            x3,
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            (float *)((uint8_t *)o2buf->ptr + out2_offset),
            (float *)((uint8_t *)o3buf->ptr + out3_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_dual_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_a_q8,
        uint64_t weight_a_offset,
        const void *weight_b_q8,
        uint64_t weight_b_offset,
        const void *input,
        uint64_t input_offset,
        void *out_a,
        uint64_t out_a_offset,
        void *out_b,
        uint64_t out_b_offset,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols) {
    if (!cuda_runtime || !weight_a_q8 || !weight_b_q8 || !input || !out_a || !out_b ||
        rows_a == 0 || rows_b == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wabuf = (const axiom_cuda_buffer *)weight_a_q8;
    const axiom_cuda_buffer *wbbuf = (const axiom_cuda_buffer *)weight_b_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *oabuf = (axiom_cuda_buffer *)out_a;
    axiom_cuda_buffer *obbuf = (axiom_cuda_buffer *)out_b;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const bool use_dp4a = axiom_cuda_env_enabled("AXIOM_DS4_Q8_DP4A");
    const bool use_preq_warp = !use_dp4a &&
            (!axiom_cuda_env_enabled("AXIOM_DS4_Q8_PREQ_DISABLE") ||
             axiom_cuda_env_enabled("AXIOM_DS4_Q8_DUAL_PREQ"));
    if (use_preq_warp) {
        const uint32_t blocks = cols / 32u;
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        const uint32_t rows = rows_a > rows_b ? rows_a : rows_b;
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        axiom_q8_0_dual_matvec_preq_pair_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wabuf->ptr + weight_a_offset,
                (const uint8_t *)wbbuf->ptr + weight_b_offset,
                xq,
                (float *)((uint8_t *)oabuf->ptr + out_a_offset),
                (float *)((uint8_t *)obbuf->ptr + out_b_offset),
                rows_a,
                rows_b,
                cols);
    } else if (use_dp4a) {
        int rc = axiom_cuda_q8_0_matvec_f32_device(
                cuda_runtime,
                weight_a_q8,
                weight_a_offset,
                input,
                input_offset,
                out_a,
                out_a_offset,
                rows_a,
                cols);
        if (rc == AXIOM_OK) {
            rc = axiom_cuda_q8_0_matvec_f32_device(
                    cuda_runtime,
                    weight_b_q8,
                    weight_b_offset,
                    input,
                    input_offset,
                    out_b,
                    out_b_offset,
                    rows_b,
                    cols);
        }
        return rc;
    } else {
        const uint32_t rows = rows_a + rows_b;
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        axiom_q8_0_dual_matvec_f32_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wabuf->ptr + weight_a_offset,
                (const uint8_t *)wbbuf->ptr + weight_b_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)oabuf->ptr + out_a_offset),
                (float *)((uint8_t *)obbuf->ptr + out_b_offset),
                rows_a,
                rows_b,
                cols);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_qkv_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q_q8,
        uint64_t weight_q_offset,
        const void *weight_k_q8,
        uint64_t weight_k_offset,
        const void *weight_v_q8,
        uint64_t weight_v_offset,
        const void *input,
        uint64_t input_offset,
        void *out_q,
        uint64_t out_q_offset,
        void *out_k,
        uint64_t out_k_offset,
        void *out_v,
        uint64_t out_v_offset,
        uint32_t rows_q,
        uint32_t rows_k,
        uint32_t rows_v,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q_q8 || !weight_k_q8 || !weight_v_q8 ||
        !input || !out_q || !out_k || !out_v ||
        rows_q == 0 || rows_k == 0 || rows_v == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wqbuf = (const axiom_cuda_buffer *)weight_q_q8;
    const axiom_cuda_buffer *wkbuf = (const axiom_cuda_buffer *)weight_k_q8;
    const axiom_cuda_buffer *wvbuf = (const axiom_cuda_buffer *)weight_v_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *oqbuf = (axiom_cuda_buffer *)out_q;
    axiom_cuda_buffer *okbuf = (axiom_cuda_buffer *)out_k;
    axiom_cuda_buffer *ovbuf = (axiom_cuda_buffer *)out_v;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const bool use_dp4a = axiom_cuda_env_enabled("AXIOM_DS4_Q8_DP4A");
    const bool use_preq_warp = !use_dp4a &&
            !axiom_cuda_env_enabled("AXIOM_DS4_Q8_PREQ_DISABLE");
    if (!use_preq_warp) {
        int rc = axiom_cuda_q8_0_matvec_f32_device(
                cuda_runtime,
                weight_q_q8,
                weight_q_offset,
                input,
                input_offset,
                out_q,
                out_q_offset,
                rows_q,
                cols);
        if (rc == AXIOM_OK) {
            rc = axiom_cuda_q8_0_matvec_f32_device(
                    cuda_runtime,
                    weight_k_q8,
                    weight_k_offset,
                    input,
                    input_offset,
                    out_k,
                    out_k_offset,
                    rows_k,
                    cols);
        }
        if (rc == AXIOM_OK) {
            rc = axiom_cuda_q8_0_matvec_f32_device(
                    cuda_runtime,
                    weight_v_q8,
                    weight_v_offset,
                    input,
                    input_offset,
                    out_v,
                    out_v_offset,
                    rows_v,
                    cols);
        }
        return rc;
    }
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const uint32_t blocks = cols / 32u;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            xq,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    uint32_t rows = rows_q > rows_k ? rows_q : rows_k;
    rows = rows > rows_v ? rows : rows_v;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_q8_0_qkv_matvec_preq_warp_kernel<<<grid, block>>>(
            (const uint8_t *)wqbuf->ptr + weight_q_offset,
            (const uint8_t *)wkbuf->ptr + weight_k_offset,
            (const uint8_t *)wvbuf->ptr + weight_v_offset,
            xq,
            (float *)((uint8_t *)oqbuf->ptr + out_q_offset),
            (float *)((uint8_t *)okbuf->ptr + out_k_offset),
            (float *)((uint8_t *)ovbuf->ptr + out_v_offset),
            rows_q,
            rows_k,
            rows_v,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_matvec_add_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        const void *residual,
        uint64_t residual_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q8 || !input || !residual || !out ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)residual;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_DP4A") ||
        axiom_cuda_env_enabled("AXIOM_DS4_Q8_PREQ_DISABLE")) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const uint32_t blocks = cols / 32u;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            xq,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_q8_0_matvec_add_preq_warp_kernel<<<grid, block>>>(
            (const uint8_t *)wbuf->ptr + weight_offset,
            xq,
            (const float *)((const uint8_t *)rbuf->ptr + residual_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_dual_matvec_silu_f32_device(
        void *cuda_runtime,
        const void *weight_gate_q8,
        uint64_t weight_gate_offset,
        const void *weight_up_q8,
        uint64_t weight_up_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_gate_q8 || !weight_up_q8 || !input || !out ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)weight_gate_q8;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)weight_up_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_DP4A") ||
        axiom_cuda_env_enabled("AXIOM_DS4_Q8_PREQ_DISABLE")) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const uint32_t blocks = cols / 32u;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            xq,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_q8_0_dual_matvec_silu_preq_pair_warp_kernel<<<grid, block>>>(
            (const uint8_t *)gbuf->ptr + weight_gate_offset,
            (const uint8_t *)ubuf->ptr + weight_up_offset,
            xq,
            (float *)((uint8_t *)obuf->ptr + out_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_dual_matvec2_f32_device(
        void *cuda_runtime,
        const void *weight_a_q8,
        uint64_t weight_a_offset,
        const void *weight_b_q8,
        uint64_t weight_b_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out_a0,
        uint64_t out_a0_offset,
        void *out_b0,
        uint64_t out_b0_offset,
        void *out_a1,
        uint64_t out_a1_offset,
        void *out_b1,
        uint64_t out_b1_offset,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols) {
    if (!cuda_runtime || !weight_a_q8 || !weight_b_q8 || !input0 || !input1 ||
        !out_a0 || !out_b0 || !out_a1 || !out_b1 ||
        rows_a == 0 || rows_b == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wabuf = (const axiom_cuda_buffer *)weight_a_q8;
    const axiom_cuda_buffer *wbbuf = (const axiom_cuda_buffer *)weight_b_q8;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    axiom_cuda_buffer *oa0buf = (axiom_cuda_buffer *)out_a0;
    axiom_cuda_buffer *ob0buf = (axiom_cuda_buffer *)out_b0;
    axiom_cuda_buffer *oa1buf = (axiom_cuda_buffer *)out_a1;
    axiom_cuda_buffer *ob1buf = (axiom_cuda_buffer *)out_b1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const uint32_t blocks = cols / 32u;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks * 2u, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_cuda_q8_0_tile32_block *x0 = xq;
    axiom_cuda_q8_0_tile32_block *x1 = xq + blocks;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
            x0,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
            (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
            x1,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    const uint32_t rows = rows_a > rows_b ? rows_a : rows_b;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_q8_0_dual_matvec2_preq_pair_warp_kernel<<<grid, block>>>(
            (const uint8_t *)wabuf->ptr + weight_a_offset,
            (const uint8_t *)wbbuf->ptr + weight_b_offset,
            x0,
            x1,
            (float *)((uint8_t *)oa0buf->ptr + out_a0_offset),
            (float *)((uint8_t *)ob0buf->ptr + out_b0_offset),
            (float *)((uint8_t *)oa1buf->ptr + out_a1_offset),
            (float *)((uint8_t *)ob1buf->ptr + out_b1_offset),
            rows_a,
            rows_b,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_matvec_argmax_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        uint32_t rows,
        uint32_t cols,
        uint32_t *out_index,
        float *out_value) {
    if (!cuda_runtime || !weight_q8 || !input || !out_index || !out_value ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint32_t block = (uint32_t)axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const uint32_t warps_per_block = block / 32u;
    uint32_t grid = (rows + warps_per_block - 1u) / warps_per_block;
    if (grid > 4096u) grid = 4096u;
    axiom_cuda_argmax_pair *partials = nullptr;
    axiom_cuda_argmax_pair *result = nullptr;
    int rc = axiom_cuda_argmax_scratch(runtime, grid, &partials, &result);
    if (rc != AXIOM_OK) return rc;
    const bool used_preq = rows == 129280u && cols == 4096u &&
            !axiom_cuda_env_enabled("AXIOM_DS4_Q8_ARGMAX_PREQ_DISABLE");
    if (used_preq) {
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        rc = axiom_cuda_q8_0_tile32_scratch(runtime, cols / 32u, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)(cols / 32u), 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                cols / 32u);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        if (!axiom_cuda_env_enabled("AXIOM_DS4_Q8_ARGMAX_PREQ_QWARP8")) {
            axiom_q8_0_matvec_argmax_preq_warp32_stage1_kernel<<<grid, block>>>(
                    (const uint8_t *)wbuf->ptr + weight_offset,
                    xq,
                    rows,
                    cols,
                    partials);
        } else {
            axiom_q8_0_matvec_argmax_preq_stage1_kernel<<<grid, block>>>(
                    (const uint8_t *)wbuf->ptr + weight_offset,
                    xq,
                    rows,
                    cols,
                    partials);
        }
    } else {
        axiom_q8_0_matvec_argmax_f32_stage1_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                rows,
                cols,
                partials);
    }
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        const size_t shared_bytes = block * sizeof(axiom_cuda_argmax_pair);
        axiom_f32_argmax_stage2_kernel<<<1, block, shared_bytes>>>(partials, grid, result);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess && used_preq) {
        axiom_q8_0_argmax_exact_value_kernel<<<1, 32>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                rows,
                cols,
                result);
        err = cudaGetLastError();
    }
    axiom_cuda_argmax_pair host;
    if (err == cudaSuccess) {
        err = cudaMemcpy(&host, result, sizeof(host), cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) return axiom_cuda_status(err);
    if (host.index == UINT32_MAX) return AXIOM_ERR_RUNTIME;
    *out_index = host.index;
    *out_value = host.value;
    return AXIOM_OK;
}

extern "C" int axiom_cuda_q8_0_matvec2_argmax_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        uint32_t rows,
        uint32_t cols,
        uint32_t *out_index0,
        float *out_value0,
        uint32_t *out_index1,
        float *out_value1) {
    if (!cuda_runtime || !weight_q8 || !input0 || !input1 ||
        !out_index0 || !out_value0 || !out_index1 || !out_value1 ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q8;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint32_t block = (uint32_t)axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const uint32_t warps_per_block = block / 32u;
    uint32_t grid = (rows + warps_per_block - 1u) / warps_per_block;
    if (grid == 0u) grid = 1u;
    if (grid > 4096u) grid = 4096u;
    if (grid > (UINT32_MAX - 2u) / 2u) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_argmax_pair *partials = nullptr;
    axiom_cuda_argmax_pair *unused_result = nullptr;
    const uint32_t scratch_count = grid * 2u + 2u;
    int rc = axiom_cuda_argmax_scratch(runtime, scratch_count, &partials, &unused_result);
    if (rc != AXIOM_OK) return rc;
    axiom_cuda_argmax_pair *partials0 = partials;
    axiom_cuda_argmax_pair *partials1 = partials + grid;
    axiom_cuda_argmax_pair *result0 = partials + grid * 2u;
    axiom_cuda_argmax_pair *result1 = result0 + 1u;
    const bool used_preq = rows == 129280u && cols == 4096u &&
            !axiom_cuda_env_enabled("AXIOM_DS4_Q8_ARGMAX_PREQ_DISABLE");
    if (used_preq) {
        const uint32_t blocks = cols / 32u;
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        rc = axiom_cuda_q8_0_tile32_scratch(runtime, blocks * 2u, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_cuda_q8_0_tile32_block *x0 = xq;
        axiom_cuda_q8_0_tile32_block *x1 = xq + blocks;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
                (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
                x0,
                blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)blocks, 32>>>(
                (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
                x1,
                blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_ARGMAX_PREQ_QWARP8")) {
            axiom_q8_0_matvec2_argmax_preq_qwarp8_stage1_kernel<<<grid, block>>>(
                    (const uint8_t *)wbuf->ptr + weight_offset,
                    x0,
                    x1,
                    rows,
                    cols,
                    partials0,
                    partials1);
        } else {
            axiom_q8_0_matvec2_argmax_preq_warp32_stage1_kernel<<<grid, block>>>(
                    (const uint8_t *)wbuf->ptr + weight_offset,
                    x0,
                    x1,
                    rows,
                    cols,
                    partials0,
                    partials1);
        }
    } else {
        axiom_q8_0_matvec2_argmax_f32_stage1_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
                (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
                rows,
                cols,
                partials0,
                partials1);
    }
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        const size_t shared_bytes = (size_t)block * 2u * sizeof(axiom_cuda_argmax_pair);
        axiom_argmax_pair_stage2_kernel<<<1, block, shared_bytes>>>(
                partials0, partials1, grid, result0, result1);
        err = cudaGetLastError();
    }
    if (err == cudaSuccess && used_preq) {
        axiom_q8_0_argmax2_exact_value_kernel<<<1, 64>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
                (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
                rows,
                cols,
                result0,
                result1);
        err = cudaGetLastError();
    }
    axiom_cuda_argmax_pair host[2];
    if (err == cudaSuccess) {
        err = cudaMemcpy(host, result0, sizeof(host), cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) return axiom_cuda_status(err);
    if (host[0].index == UINT32_MAX || host[1].index == UINT32_MAX) {
        return AXIOM_ERR_RUNTIME;
    }
    *out_index0 = host[0].index;
    *out_value0 = host[0].value;
    *out_index1 = host[1].index;
    *out_value1 = host[1].value;
    (void)unused_result;
    return AXIOM_OK;
}

extern "C" int axiom_cuda_q8_0_grouped_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    if (!cuda_runtime || !weight_q8 || !input || !out ||
        groups == 0 || rows_per_group == 0 || input_per_group == 0 ||
        (input_per_group % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const uint64_t rows = (uint64_t)groups * rows_per_group;
    const int warps_per_block = block / 32;
    const bool use_grouped_preq = groups == 8u && rows_per_group == 1024u &&
            input_per_group == 4096u &&
            !axiom_cuda_env_enabled("AXIOM_DS4_Q8_GROUPED_PREQ_DISABLE");
    if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_GROUPED_SHAPE_CONST") &&
        groups == 8u && rows_per_group == 1024u && input_per_group == 4096u) {
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        axiom_q8_0_grouped_matvec_f32_warp_ds4_o_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset));
    } else if (use_grouped_preq || axiom_cuda_env_enabled("AXIOM_DS4_Q8_GROUPED_PREQ")) {
        const uint32_t blocks = input_per_group / 32u;
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, groups * blocks, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)(groups * blocks), 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                groups * blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_GROUPED_PREQ_SXQ")) {
            const size_t sxq_bytes = (size_t)blocks * sizeof(axiom_cuda_q8_0_tile32_block);
            axiom_q8_0_grouped_matvec_preq_sxq_warp_kernel<<<grid, block, sxq_bytes>>>(
                    (const uint8_t *)wbuf->ptr + weight_offset,
                    xq,
                    (float *)((uint8_t *)obuf->ptr + out_offset),
                    groups,
                    rows_per_group,
                    input_per_group);
        } else {
            axiom_q8_0_grouped_matvec_preq_warp_kernel<<<grid, block>>>(
                    (const uint8_t *)wbuf->ptr + weight_offset,
                    xq,
                    (float *)((uint8_t *)obuf->ptr + out_offset),
                    groups,
                    rows_per_group,
                    input_per_group);
        }
    } else {
        const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
        axiom_q8_0_grouped_matvec_f32_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                groups,
                rows_per_group,
                input_per_group);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_soa_grouped_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    if (!cuda_runtime || !weight_qs_i8 || !weight_scales_f16 || !input || !out ||
        groups == 0 || rows_per_group == 0 || input_per_group == 0 ||
        (input_per_group % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)weight_qs_i8;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)weight_scales_f16;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const uint64_t rows = (uint64_t)groups * rows_per_group;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_SOA_PREQ")) {
        const uint32_t blocks = input_per_group / 32u;
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, groups * blocks, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)(groups * blocks), 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                groups * blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_q8_0_soa_grouped_matvec_preq_warp_kernel<<<grid, block>>>(
                (const int8_t *)qbuf->ptr,
                (const uint16_t *)sbuf->ptr,
                xq,
                (float *)((uint8_t *)obuf->ptr + out_offset),
                groups,
                rows_per_group,
                input_per_group);
    } else if (axiom_cuda_env_enabled("AXIOM_DS4_Q8_SOA_BLOCKLANE")) {
        axiom_q8_0_soa_grouped_matvec_f32_blocklane_kernel<<<grid, block>>>(
                (const int8_t *)qbuf->ptr,
                (const uint16_t *)sbuf->ptr,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                groups,
                rows_per_group,
                input_per_group);
    } else {
        axiom_q8_0_soa_grouped_matvec_f32_warp_kernel<<<grid, block>>>(
                (const int8_t *)qbuf->ptr,
                (const uint16_t *)sbuf->ptr,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                groups,
                rows_per_group,
                input_per_group);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_soa_grouped_matvec2_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    if (!cuda_runtime || !weight_qs_i8 || !weight_scales_f16 ||
        !input0 || !input1 || !out0 || !out1 ||
        groups == 0 || rows_per_group == 0 || input_per_group == 0 ||
        (input_per_group % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)weight_qs_i8;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)weight_scales_f16;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const uint64_t rows = (uint64_t)groups * rows_per_group;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    const uint32_t blocks = input_per_group / 32u;
    const uint32_t xblocks = groups * blocks;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, xblocks * 2u, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_cuda_q8_0_tile32_block *x0 = xq;
    axiom_cuda_q8_0_tile32_block *x1 = xq + xblocks;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)xblocks, 32>>>(
            (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
            x0,
            xblocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)xblocks, 32>>>(
            (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
            x1,
            xblocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_soa_grouped_matvec2_preq_warp_kernel<<<grid, block>>>(
            (const int8_t *)qbuf->ptr,
            (const uint16_t *)sbuf->ptr,
            x0,
            x1,
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            groups,
            rows_per_group,
            input_per_group);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_soa_grouped_matvec4_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        const void *input2,
        uint64_t input2_offset,
        const void *input3,
        uint64_t input3_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
        uint64_t out3_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group) {
    if (!cuda_runtime || !weight_qs_i8 || !weight_scales_f16 ||
        !input0 || !input1 || !input2 || !input3 ||
        !out0 || !out1 || !out2 || !out3 ||
        groups == 0 || rows_per_group == 0 || input_per_group == 0 ||
        (input_per_group % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)weight_qs_i8;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)weight_scales_f16;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    const axiom_cuda_buffer *i2buf = (const axiom_cuda_buffer *)input2;
    const axiom_cuda_buffer *i3buf = (const axiom_cuda_buffer *)input3;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    axiom_cuda_buffer *o2buf = (axiom_cuda_buffer *)out2;
    axiom_cuda_buffer *o3buf = (axiom_cuda_buffer *)out3;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const uint64_t rows = (uint64_t)groups * rows_per_group;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    const uint32_t blocks = input_per_group / 32u;
    const uint32_t xblocks = groups * blocks;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, xblocks * 4u, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_cuda_q8_0_tile32_block *x0 = xq;
    axiom_cuda_q8_0_tile32_block *x1 = xq + xblocks;
    axiom_cuda_q8_0_tile32_block *x2 = x1 + xblocks;
    axiom_cuda_q8_0_tile32_block *x3 = x2 + xblocks;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)xblocks, 32>>>(
            (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
            x0,
            xblocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)xblocks, 32>>>(
            (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
            x1,
            xblocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)xblocks, 32>>>(
            (const float *)((const uint8_t *)i2buf->ptr + input2_offset),
            x2,
            xblocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)xblocks, 32>>>(
            (const float *)((const uint8_t *)i3buf->ptr + input3_offset),
            x3,
            xblocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_soa_grouped_matvec4_preq_warp_kernel<<<grid, block>>>(
            (const int8_t *)qbuf->ptr,
            (const uint16_t *)sbuf->ptr,
            x0,
            x1,
            x2,
            x3,
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            (float *)((uint8_t *)o2buf->ptr + out2_offset),
            (float *)((uint8_t *)o3buf->ptr + out3_offset),
            groups,
            rows_per_group,
            input_per_group);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_q4_k_matvec_q8k_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_k_block *__restrict__ input_q8k,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols);

extern "C" int axiom_cuda_q2_k_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q2k,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q2k || !input || !out ||
        rows == 0 || cols == 0 || (cols % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q2k;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((rows + block - 1u) / block);
    axiom_q2_k_matvec_f32_kernel<<<grid, block>>>(
            (const uint8_t *)wbuf->ptr + weight_offset,
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q4_k_matvec_q8k_device(
        void *cuda_runtime,
        const void *weight_q4k,
        uint64_t weight_offset,
        const void *input_q8k,
        uint64_t input_q8k_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q4k || !input_q8k || !out ||
        rows == 0 || cols == 0 || (cols % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q4k;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input_q8k;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((rows + 31u) / 32u);
    axiom_q4_k_matvec_q8k_warp_kernel<<<grid, block>>>(
            (const uint8_t *)wbuf->ptr + weight_offset,
            (const axiom_cuda_q8_k_block *)((const uint8_t *)ibuf->ptr + input_q8k_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_pack_f32_to_q8k_device(
        void *cuda_runtime,
        const void *input,
        uint64_t input_offset,
        void *out_q8k,
        uint64_t out_q8k_offset,
        uint32_t count) {
    if (!cuda_runtime || !input || !out_q8k || count == 0 || (count % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out_q8k;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint32_t blocks = count / 256u;
    axiom_q8_k_pack_f32_kernel<<<(int)blocks, 256>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (axiom_cuda_q8_k_block *)((uint8_t *)obuf->ptr + out_q8k_offset),
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_iq2_xxs_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_iq2xxs,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_iq2xxs || !input || !out ||
        rows == 0 || cols == 0 || (cols % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_iq2xxs;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((rows + block - 1u) / block);
    axiom_iq2_xxs_matvec_f32_kernel<<<grid, block>>>(
            (const uint8_t *)wbuf->ptr + weight_offset,
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_iq2_xxs_matvec_f32_warp_device(
        void *cuda_runtime,
        const void *weight_iq2xxs,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_iq2xxs || !input || !out ||
        rows == 0 || cols == 0 || (cols % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_iq2xxs;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int warps_per_block = block / 32;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_iq2_xxs_matvec_f32_warp_kernel<<<grid, block>>>(
            (const uint8_t *)wbuf->ptr + weight_offset,
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_shared_gate_up_q8_warp_kernel(
        const uint8_t *gate,
        const uint8_t *up,
        const float *input,
        float *mid,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= expert_hidden) return;
    const uint32_t blocks = hidden / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *gate_row = gate + (uint64_t)row * row_bytes;
    const uint8_t *up_row = up + (uint64_t)row * row_bytes;
    float g0 = 0.0f;
    float u0 = 0.0f;
    axiom_dev_q8_0_dual_dot_f32_warp(gate_row, up_row, input, blocks, &g0, &u0);
    if (lane == 0u) {
        const float g = fminf(g0, 10.0f);
        const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
        mid[row] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ static void axiom_deepseek_shared_gate_up_q8_preq_warp_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ mid,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= expert_hidden) return;
    const uint32_t blocks = hidden / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *gate_row = gate + (uint64_t)row * row_bytes;
    const uint8_t *up_row = up + (uint64_t)row * row_bytes;
    float g0 = 0.0f;
    float u0 = 0.0f;
    axiom_dev_q8_0_dual_dot_q8tile_warp(gate_row, up_row, input, blocks, &g0, &u0);
    if (lane == 0u) {
        const float g = fminf(g0, 10.0f);
        const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
        mid[row] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ static void axiom_deepseek_shared_gate_up_q8_soa_preq_warp_kernel(
        const int8_t *__restrict__ gate_qs,
        const uint16_t *__restrict__ gate_scales,
        const int8_t *__restrict__ up_qs,
        const uint16_t *__restrict__ up_scales,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        float *__restrict__ mid,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= expert_hidden) return;
    const uint32_t blocks = hidden / 32u;
    const int8_t *gate_row = gate_qs + (uint64_t)row * hidden;
    const int8_t *up_row = up_qs + (uint64_t)row * hidden;
    const uint16_t *gate_srow = gate_scales + (uint64_t)row * blocks;
    const uint16_t *up_srow = up_scales + (uint64_t)row * blocks;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        const int32_t gate_dot = axiom_dev_q8_0_dot_i8_block32_aligned(
                gate_row + (uint64_t)b * 32u, x->qs);
        const int32_t up_dot = axiom_dev_q8_0_dot_i8_block32_aligned(
                up_row + (uint64_t)b * 32u, x->qs);
        gate_acc += axiom_dev_f16_to_f32(gate_srow[b]) * x->d * (float)gate_dot;
        up_acc += axiom_dev_f16_to_f32(up_srow[b]) * x->d * (float)up_dot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    if (lane == 0u) {
        const float g = fminf(gate_acc, 10.0f);
        const float u = fminf(fmaxf(up_acc, -10.0f), 10.0f);
        mid[row] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ static void axiom_deepseek_shared_gate_up_q8_soa_preq_batch2_warp_kernel(
        const int8_t *__restrict__ gate_qs,
        const uint16_t *__restrict__ gate_scales,
        const int8_t *__restrict__ up_qs,
        const uint16_t *__restrict__ up_scales,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input0,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input1,
        float *__restrict__ mid0,
        float *__restrict__ mid1,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= expert_hidden) return;
    const uint32_t blocks = hidden / 32u;
    const int8_t *gate_row = gate_qs + (uint64_t)row * hidden;
    const int8_t *up_row = up_qs + (uint64_t)row * hidden;
    const uint16_t *gate_srow = gate_scales + (uint64_t)row * blocks;
    const uint16_t *up_srow = up_scales + (uint64_t)row * blocks;
    float gate0 = 0.0f;
    float up0 = 0.0f;
    float gate1 = 0.0f;
    float up1 = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const axiom_cuda_q8_0_tile32_block *x0 = input0 + b;
        const axiom_cuda_q8_0_tile32_block *x1 = input1 + b;
        const int8_t *gq = gate_row + (uint64_t)b * 32u;
        const int8_t *uq = up_row + (uint64_t)b * 32u;
        const float gs = axiom_dev_f16_to_f32(gate_srow[b]);
        const float us = axiom_dev_f16_to_f32(up_srow[b]);
        gate0 += gs * x0->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(gq, x0->qs);
        up0 += us * x0->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(uq, x0->qs);
        gate1 += gs * x1->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(gq, x1->qs);
        up1 += us * x1->d * (float)axiom_dev_q8_0_dot_i8_block32_aligned(uq, x1->qs);
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        gate0 += __shfl_down_sync(0xffffffffu, gate0, offset);
        up0 += __shfl_down_sync(0xffffffffu, up0, offset);
        gate1 += __shfl_down_sync(0xffffffffu, gate1, offset);
        up1 += __shfl_down_sync(0xffffffffu, up1, offset);
    }
    if (lane == 0u) {
        const float g0 = fminf(gate0, 10.0f);
        const float u0 = fminf(fmaxf(up0, -10.0f), 10.0f);
        const float g1 = fminf(gate1, 10.0f);
        const float u1 = fminf(fmaxf(up1, -10.0f), 10.0f);
        mid0[row] = (g0 / (1.0f + expf(-g0))) * u0;
        mid1[row] = (g1 / (1.0f + expf(-g1))) * u1;
    }
}

__global__ static void axiom_deepseek_shared_down_q8_warp_kernel(
        const uint8_t *down,
        const float *mid,
        float *out,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *down_row = down + (uint64_t)row * row_bytes;
    const float y = axiom_dev_q8_0_dot_f32_warp(down_row, mid, blocks);
    if (lane == 0u) out[row] = y;
}

__global__ static void axiom_deepseek_shared_down_ffn_hc_post_q8_warp_kernel(
        const uint8_t *down,
        const float *mid,
        const float *moe,
        const float *residual,
        const float *post,
        const float *comb,
        float *out_a,
        float *out_b,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *down_row = down + (uint64_t)row * row_bytes;
    const float shared_y = axiom_dev_q8_0_dot_f32_warp(down_row, mid, blocks);
    if (lane != 0u) return;

    const double block_v = (double)shared_y + (double)moe[row];
    #pragma unroll
    for (uint32_t dst = 0; dst < 4u; ++dst) {
        double acc = (double)post[dst] * block_v;
        #pragma unroll
        for (uint32_t src = 0; src < 4u; ++src) {
            acc += (double)comb[src * 4u + dst] *
                   (double)residual[(uint64_t)src * hidden + row];
        }
        const uint64_t off = (uint64_t)dst * hidden + row;
        const float y = (float)acc;
        out_a[off] = y;
        out_b[off] = y;
    }
}

__global__ static void axiom_deepseek_shared_down_ffn_hc_post_q8_preq64_warp_kernel(
        const uint8_t *down,
        const axiom_cuda_q8_0_tile32_block *input,
        const float *moe,
        const float *residual,
        const float *post,
        const float *comb,
        float *out_a,
        float *out_b,
        uint32_t hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= hidden) return;
    const uint64_t row_bytes = 64ull * 34ull;
    const uint8_t *r = down + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t i = 0u; i < 2u; ++i) {
        const uint32_t b = lane + i * 32u;
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
        acc += wd * x->d * (float)dot;
    }
    #pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane != 0u) return;

    const double block_v = (double)acc + (double)moe[row];
    #pragma unroll
    for (uint32_t dst = 0; dst < 4u; ++dst) {
        double post_acc = (double)post[dst] * block_v;
        #pragma unroll
        for (uint32_t src = 0; src < 4u; ++src) {
            post_acc += (double)comb[src * 4u + dst] *
                    (double)residual[(uint64_t)src * hidden + row];
        }
        const uint64_t off = (uint64_t)dst * hidden + row;
        const float y = (float)post_acc;
        out_a[off] = y;
        out_b[off] = y;
    }
}

__global__ static void axiom_q8_0_matvec_hc_post_f32_warp_kernel(
        const uint8_t *weight,
        const float *input,
        float *block_out,
        const float *residual,
        const float *post,
        const float *comb,
        float *out,
        uint32_t hidden,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= hidden) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *wrow = weight + (uint64_t)row * row_bytes;
    const float y = axiom_dev_q8_0_dot_f32_warp(wrow, input, blocks);
    if (lane != 0u) return;

    block_out[row] = y;
    #pragma unroll
    for (uint32_t dst = 0; dst < 4u; ++dst) {
        double acc = (double)post[dst] * (double)y;
        #pragma unroll
        for (uint32_t src = 0; src < 4u; ++src) {
            acc += (double)comb[src * 4u + dst] *
                   (double)residual[(uint64_t)src * hidden + row];
        }
        out[(uint64_t)dst * hidden + row] = (float)acc;
    }
}

__global__ static void axiom_q8_0_matvec_hc_post_preq_warp_kernel(
        const uint8_t *weight,
        const axiom_cuda_q8_0_tile32_block *input,
        float *block_out,
        const float *residual,
        const float *post,
        const float *comb,
        float *out,
        uint32_t hidden,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= hidden) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *wrow = weight + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = wrow + (uint64_t)b * 34u;
        const float wd = axiom_dev_f16_to_f32(axiom_dev_le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const axiom_cuda_q8_0_tile32_block *x = input + b;
        const int32_t dot = axiom_dev_q8_0_dot_i8_block32(wq, x->qs);
        acc += wd * x->d * (float)dot;
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane != 0u) return;

    block_out[row] = acc;
    #pragma unroll
    for (uint32_t dst = 0; dst < 4u; ++dst) {
        double hc = (double)post[dst] * (double)acc;
        #pragma unroll
        for (uint32_t src = 0; src < 4u; ++src) {
            hc += (double)comb[src * 4u + dst] *
                  (double)residual[(uint64_t)src * hidden + row];
        }
        out[(uint64_t)dst * hidden + row] = (float)hc;
    }
}

extern "C" int axiom_cuda_deepseek_shared_expert_q8_f32_device(
        void *cuda_runtime,
        const void *gate_q8,
        uint64_t gate_offset,
        const void *up_q8,
        uint64_t up_offset,
        const void *down_q8,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_q8 || !up_q8 || !down_q8 || !input || !out ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_q8;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_q8;
    const axiom_cuda_buffer *dbuf = (const axiom_cuda_buffer *)down_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *mid = nullptr;
    err = cudaMalloc(&mid, (size_t)expert_hidden * sizeof(float));
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const int block = 128;
    const int warps_per_block = block / 32;
    const int up_grid = (int)((expert_hidden + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_deepseek_shared_gate_up_q8_warp_kernel<<<up_grid, block>>>(
            (const uint8_t *)gbuf->ptr + gate_offset,
            (const uint8_t *)ubuf->ptr + up_offset,
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            mid,
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(mid);
        return axiom_cuda_status(err);
    }

    const int down_grid = (int)((hidden + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_deepseek_shared_down_q8_warp_kernel<<<down_grid, block>>>(
            (const uint8_t *)dbuf->ptr + down_offset,
            mid,
            (float *)((uint8_t *)obuf->ptr + out_offset),
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(mid);
        return axiom_cuda_status(err);
    }
    err = cudaDeviceSynchronize();
    cudaFree(mid);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_deepseek_shared_expert_q8_f32_scratch_device(
        void *cuda_runtime,
        const void *gate_q8,
        uint64_t gate_offset,
        const void *up_q8,
        uint64_t up_offset,
        const void *down_q8,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        void *scratch_mid,
        uint64_t scratch_mid_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_q8 || !up_q8 || !down_q8 || !input || !scratch_mid || !out ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_q8;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_q8;
    const axiom_cuda_buffer *dbuf = (const axiom_cuda_buffer *)down_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *sbuf = (axiom_cuda_buffer *)scratch_mid;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *mid = (float *)((uint8_t *)sbuf->ptr + scratch_mid_offset);
    const int block = 128;
    const int warps_per_block = block / 32;
    const int up_grid = (int)((expert_hidden + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    if (axiom_cuda_env_enabled("AXIOM_DS4_SHARED_Q8_GATEUP_PREQ") && hidden == 4096u) {
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, hidden / 32u, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)(hidden / 32u), 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                hidden / 32u);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_deepseek_shared_gate_up_q8_preq_warp_kernel<<<up_grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                mid,
                hidden,
                expert_hidden);
    } else {
        axiom_deepseek_shared_gate_up_q8_warp_kernel<<<up_grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                mid,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const int down_preq64 =
            axiom_cuda_env_enabled("AXIOM_DS4_SHARED_Q8_DOWN_PREQ64") &&
            hidden == 4096u && expert_hidden == 2048u;
    const int down_block = down_preq64 ? axiom_cuda_env_int(
            "AXIOM_DS4_SHARED_Q8_DOWN_PREQ64_BLOCK", 128, 128, 512, 32) : block;
    const int down_warps_per_block = down_block / 32;
    const int down_grid = (int)((hidden + (uint32_t)down_warps_per_block - 1u) /
            (uint32_t)down_warps_per_block);
    if (axiom_cuda_env_enabled("AXIOM_DS4_SHARED_Q8_DOWN_PREQ") && expert_hidden == 2048u) {
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, expert_hidden / 32u, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)(expert_hidden / 32u), 32>>>(
                mid,
                xq,
                expert_hidden / 32u);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        if (down_preq64) {
            axiom_q8_0_matvec_f32_preq64_warp_kernel<<<down_grid, down_block>>>(
                    (const uint8_t *)dbuf->ptr + down_offset,
                    xq,
                    (float *)((uint8_t *)obuf->ptr + out_offset),
                    hidden);
        } else {
            axiom_q8_0_matvec_f32_preq_warp_kernel<<<down_grid, down_block>>>(
                    (const uint8_t *)dbuf->ptr + down_offset,
                    xq,
                    (float *)((uint8_t *)obuf->ptr + out_offset),
                    hidden,
                    expert_hidden);
        }
    } else {
        axiom_deepseek_shared_down_q8_warp_kernel<<<down_grid, down_block>>>(
                (const uint8_t *)dbuf->ptr + down_offset,
                mid,
                (float *)((uint8_t *)obuf->ptr + out_offset),
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_shared_expert_q8_soa_f32_scratch_device(
        void *cuda_runtime,
        const void *gate_qs_i8,
        const void *gate_scales_f16,
        const void *up_qs_i8,
        const void *up_scales_f16,
        const void *down_qs_i8,
        const void *down_scales_f16,
        const void *input,
        uint64_t input_offset,
        void *scratch_mid,
        uint64_t scratch_mid_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_qs_i8 || !gate_scales_f16 || !up_qs_i8 || !up_scales_f16 ||
        !down_qs_i8 || !down_scales_f16 || !input || !scratch_mid || !out ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gqbuf = (const axiom_cuda_buffer *)gate_qs_i8;
    const axiom_cuda_buffer *gsbuf = (const axiom_cuda_buffer *)gate_scales_f16;
    const axiom_cuda_buffer *uqbuf = (const axiom_cuda_buffer *)up_qs_i8;
    const axiom_cuda_buffer *usbuf = (const axiom_cuda_buffer *)up_scales_f16;
    const axiom_cuda_buffer *dqbuf = (const axiom_cuda_buffer *)down_qs_i8;
    const axiom_cuda_buffer *dsbuf = (const axiom_cuda_buffer *)down_scales_f16;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *sbuf = (axiom_cuda_buffer *)scratch_mid;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const int block = axiom_cuda_env_int("AXIOM_DS4_SHARED_Q8_SOA_BLOCK", 128, 128, 512, 32);
    const int warps_per_block = block / 32;
    const int up_grid = (int)((expert_hidden + (uint32_t)warps_per_block - 1u) /
            (uint32_t)warps_per_block);
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, hidden / 32u, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)(hidden / 32u), 32>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            xq,
            hidden / 32u);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    float *mid = (float *)((uint8_t *)sbuf->ptr + scratch_mid_offset);
    axiom_deepseek_shared_gate_up_q8_soa_preq_warp_kernel<<<up_grid, block>>>(
            (const int8_t *)gqbuf->ptr,
            (const uint16_t *)gsbuf->ptr,
            (const int8_t *)uqbuf->ptr,
            (const uint16_t *)usbuf->ptr,
            xq,
            mid,
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);

    rc = axiom_cuda_q8_0_tile32_scratch(runtime, expert_hidden / 32u, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)(expert_hidden / 32u), 32>>>(
            mid,
            xq,
            expert_hidden / 32u);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    const int down_grid = (int)((hidden + (uint32_t)warps_per_block - 1u) /
            (uint32_t)warps_per_block);
    axiom_q8_0_soa_matvec_preq_warp_kernel<<<down_grid, block>>>(
            (const int8_t *)dqbuf->ptr,
            (const uint16_t *)dsbuf->ptr,
            xq,
            (float *)((uint8_t *)obuf->ptr + out_offset),
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_shared_expert_q8_soa_f32_scratch2_device(
        void *cuda_runtime,
        const void *gate_qs_i8,
        const void *gate_scales_f16,
        const void *up_qs_i8,
        const void *up_scales_f16,
        const void *down_qs_i8,
        const void *down_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *scratch_mid0,
        uint64_t scratch_mid0_offset,
        void *scratch_mid1,
        uint64_t scratch_mid1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_qs_i8 || !gate_scales_f16 || !up_qs_i8 || !up_scales_f16 ||
        !down_qs_i8 || !down_scales_f16 || !input0 || !input1 ||
        !scratch_mid0 || !scratch_mid1 || !out0 || !out1 ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gqbuf = (const axiom_cuda_buffer *)gate_qs_i8;
    const axiom_cuda_buffer *gsbuf = (const axiom_cuda_buffer *)gate_scales_f16;
    const axiom_cuda_buffer *uqbuf = (const axiom_cuda_buffer *)up_qs_i8;
    const axiom_cuda_buffer *usbuf = (const axiom_cuda_buffer *)up_scales_f16;
    const axiom_cuda_buffer *dqbuf = (const axiom_cuda_buffer *)down_qs_i8;
    const axiom_cuda_buffer *dsbuf = (const axiom_cuda_buffer *)down_scales_f16;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    axiom_cuda_buffer *m0buf = (axiom_cuda_buffer *)scratch_mid0;
    axiom_cuda_buffer *m1buf = (axiom_cuda_buffer *)scratch_mid1;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const uint32_t hidden_blocks = hidden / 32u;
    const uint32_t mid_blocks = expert_hidden / 32u;
    const uint32_t scratch_blocks = (hidden_blocks > mid_blocks ? hidden_blocks : mid_blocks) * 2u;
    axiom_cuda_q8_0_tile32_block *xq = nullptr;
    int rc = axiom_cuda_q8_0_tile32_scratch(runtime, scratch_blocks, &xq);
    if (rc != AXIOM_OK) return rc;
    axiom_cuda_q8_0_tile32_block *x0 = xq;
    axiom_cuda_q8_0_tile32_block *x1 = xq + (scratch_blocks / 2u);

    axiom_q8_0_tile32_pack_f32_kernel<<<(int)hidden_blocks, 32>>>(
            (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
            x0,
            hidden_blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)hidden_blocks, 32>>>(
            (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
            x1,
            hidden_blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const int block = axiom_cuda_env_int("AXIOM_DS4_SHARED_Q8_SOA_BLOCK", 128, 128, 512, 32);
    const int warps_per_block = block / 32;
    const int up_grid = (int)((expert_hidden + (uint32_t)warps_per_block - 1u) /
            (uint32_t)warps_per_block);
    float *mid0 = (float *)((uint8_t *)m0buf->ptr + scratch_mid0_offset);
    float *mid1 = (float *)((uint8_t *)m1buf->ptr + scratch_mid1_offset);
    axiom_deepseek_shared_gate_up_q8_soa_preq_batch2_warp_kernel<<<up_grid, block>>>(
            (const int8_t *)gqbuf->ptr,
            (const uint16_t *)gsbuf->ptr,
            (const int8_t *)uqbuf->ptr,
            (const uint16_t *)usbuf->ptr,
            x0,
            x1,
            mid0,
            mid1,
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);

    axiom_q8_0_tile32_pack_f32_kernel<<<(int)mid_blocks, 32>>>(mid0, x0, mid_blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_q8_0_tile32_pack_f32_kernel<<<(int)mid_blocks, 32>>>(mid1, x1, mid_blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const int down_grid = (int)((hidden + (uint32_t)warps_per_block - 1u) /
            (uint32_t)warps_per_block);
    axiom_q8_0_soa_matvec2_preq_warp_kernel<<<down_grid, block>>>(
            (const int8_t *)dqbuf->ptr,
            (const uint16_t *)dsbuf->ptr,
            x0,
            x1,
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_shared_gate_up_q8_f32_device(
        void *cuda_runtime,
        const void *gate_q8,
        uint64_t gate_offset,
        const void *up_q8,
        uint64_t up_offset,
        const void *input,
        uint64_t input_offset,
        void *mid,
        uint64_t mid_offset,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_q8 || !up_q8 || !input || !mid ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_q8;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *mbuf = (axiom_cuda_buffer *)mid;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const int block = 128;
    const int warps_per_block = block / 32;
    const int up_grid = (int)((expert_hidden + (uint32_t)warps_per_block - 1u) /
            (uint32_t)warps_per_block);
    if (axiom_cuda_env_enabled("AXIOM_DS4_SHARED_Q8_GATEUP_PREQ") && hidden == 4096u) {
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, hidden / 32u, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)(hidden / 32u), 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                hidden / 32u);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_deepseek_shared_gate_up_q8_preq_warp_kernel<<<up_grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                (float *)((uint8_t *)mbuf->ptr + mid_offset),
                hidden,
                expert_hidden);
    } else {
        axiom_deepseek_shared_gate_up_q8_warp_kernel<<<up_grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)mbuf->ptr + mid_offset),
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_matvec_hc_post_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *block_out,
        uint64_t block_out_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q8 || !input || !block_out || !residual || !post ||
        !comb || !out || hidden == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_q8;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *bbuf = (axiom_cuda_buffer *)block_out;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)residual;
    const axiom_cuda_buffer *pbuf = (const axiom_cuda_buffer *)post;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)comb;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const int block = axiom_cuda_env_int("AXIOM_DS4_Q8_BLOCK", 256, 64, 512, 32);
    const int warps_per_block = block / 32;
    const int grid = (int)((hidden + (uint32_t)warps_per_block - 1u) /
            (uint32_t)warps_per_block);
    if (axiom_cuda_env_enabled("AXIOM_DS4_ATTN_OUT_B_HC_PREQ")) {
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, cols / 32u, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)(cols / 32u), 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                cols / 32u);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_q8_0_matvec_hc_post_preq_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                xq,
                (float *)((uint8_t *)bbuf->ptr + block_out_offset),
                (const float *)((const uint8_t *)rbuf->ptr + residual_offset),
                (const float *)((const uint8_t *)pbuf->ptr + post_offset),
                (const float *)((const uint8_t *)cbuf->ptr + comb_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                hidden,
                cols);
    } else {
        axiom_q8_0_matvec_hc_post_f32_warp_kernel<<<grid, block>>>(
                (const uint8_t *)wbuf->ptr + weight_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)bbuf->ptr + block_out_offset),
                (const float *)((const uint8_t *)rbuf->ptr + residual_offset),
                (const float *)((const uint8_t *)pbuf->ptr + post_offset),
                (const float *)((const uint8_t *)cbuf->ptr + comb_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                hidden,
                cols);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_shared_down_ffn_hc_post_q8_f32_device(
        void *cuda_runtime,
        const void *down_q8,
        uint64_t down_offset,
        const void *shared_mid,
        uint64_t shared_mid_offset,
        const void *moe,
        uint64_t moe_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out_a,
        uint64_t out_a_offset,
        void *out_b,
        uint64_t out_b_offset,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !down_q8 || !shared_mid || !moe || !residual || !post || !comb ||
        !out_a || !out_b || hidden == 0 || expert_hidden == 0 ||
        (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *dbuf = (const axiom_cuda_buffer *)down_q8;
    const axiom_cuda_buffer *mbuf = (const axiom_cuda_buffer *)shared_mid;
    const axiom_cuda_buffer *moebuf = (const axiom_cuda_buffer *)moe;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)residual;
    const axiom_cuda_buffer *pbuf = (const axiom_cuda_buffer *)post;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)comb;
    axiom_cuda_buffer *oabuf = (axiom_cuda_buffer *)out_a;
    axiom_cuda_buffer *obbuf = (axiom_cuda_buffer *)out_b;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const int block = axiom_cuda_env_int(
            "AXIOM_DS4_SHARED_Q8_DOWN_PREQ64_BLOCK", 128, 128, 512, 32);
    const int warps_per_block = block / 32;
    const int grid = (int)((hidden + (uint32_t)warps_per_block - 1u) /
            (uint32_t)warps_per_block);
    if (axiom_cuda_env_enabled("AXIOM_DS4_SHARED_Q8_DOWN_PREQ") &&
        hidden == 4096u && expert_hidden == 2048u) {
        axiom_cuda_q8_0_tile32_block *xq = nullptr;
        int rc = axiom_cuda_q8_0_tile32_scratch(runtime, expert_hidden / 32u, &xq);
        if (rc != AXIOM_OK) return rc;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)(expert_hidden / 32u), 32>>>(
                (const float *)((const uint8_t *)mbuf->ptr + shared_mid_offset),
                xq,
                expert_hidden / 32u);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_deepseek_shared_down_ffn_hc_post_q8_preq64_warp_kernel<<<grid, block>>>(
                (const uint8_t *)dbuf->ptr + down_offset,
                xq,
                (const float *)((const uint8_t *)moebuf->ptr + moe_offset),
                (const float *)((const uint8_t *)rbuf->ptr + residual_offset),
                (const float *)((const uint8_t *)pbuf->ptr + post_offset),
                (const float *)((const uint8_t *)cbuf->ptr + comb_offset),
                (float *)((uint8_t *)oabuf->ptr + out_a_offset),
                (float *)((uint8_t *)obbuf->ptr + out_b_offset),
                hidden);
        err = cudaGetLastError();
        return axiom_cuda_finish_after_launch(err);
    }
    axiom_deepseek_shared_down_ffn_hc_post_q8_warp_kernel<<<grid, block>>>(
            (const uint8_t *)dbuf->ptr + down_offset,
            (const float *)((const uint8_t *)mbuf->ptr + shared_mid_offset),
            (const float *)((const uint8_t *)moebuf->ptr + moe_offset),
            (const float *)((const uint8_t *)rbuf->ptr + residual_offset),
            (const float *)((const uint8_t *)pbuf->ptr + post_offset),
            (const float *)((const uint8_t *)cbuf->ptr + comb_offset),
            (float *)((uint8_t *)oabuf->ptr + out_a_offset),
            (float *)((uint8_t *)obbuf->ptr + out_b_offset),
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_sliding_attention_single_kernel(
        const float *q,
        const float *kv,
        const float *attn_sink,
        float *out,
        uint32_t head_dim,
        float eps) {
    const uint32_t head = (uint32_t)blockIdx.x;
    if (threadIdx.x != 0) return;
    const float *qh = q + (uint64_t)head * head_dim;
    double dot = 0.0;
    for (uint32_t i = 0; i < head_dim; ++i) {
        dot += (double)qh[i] * (double)kv[i];
    }
    const float score = (float)dot * rsqrtf((float)head_dim);
    const float sink = attn_sink[head];
    const float m = fmaxf(score, sink);
    const float p = expf(score - m) / (expf(score - m) + expf(sink - m));
    float *oh = out + (uint64_t)head * head_dim;
    for (uint32_t i = 0; i < head_dim; ++i) {
        oh[i] = p * kv[i];
    }
}

extern "C" int axiom_cuda_deepseek_sliding_attention_single_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *kv,
        uint64_t kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    if (!cuda_runtime || !q || !kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *kvbuf = (const axiom_cuda_buffer *)kv;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_sliding_attention_single_kernel<<<heads, 1>>>(
            (const float *)((const uint8_t *)qbuf->ptr + q_offset),
            (const float *)((const uint8_t *)kvbuf->ptr + kv_offset),
            (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            head_dim,
            eps);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_sliding_attention_ring_kernel(
        const float *q,
        const float *current_kv,
        const float *ring_kv,
        const float *attn_sink,
        float *out,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ double reduce[256];
    __shared__ float scores[128];
    __shared__ float weights[128];
    __shared__ float m_s;
    __shared__ double denom_s;
    __shared__ float current_weight_s;
    const float *qh = q + (uint64_t)head * head_dim;

    if (tid == 0u) {
        m_s = attn_sink[head];
    }
    __syncthreads();

    double local = 0.0;
    const uint32_t oldest = ring_count == ring_slots ? ring_head : 0u;
    for (uint32_t k = 0; k < ring_count; ++k) {
        uint32_t slot = oldest + k;
        if (slot >= ring_slots) slot -= ring_slots;
        const float *kvh = ring_kv + (uint64_t)slot * head_dim;
        local = 0.0;
        for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
            local += (double)qh[i] * (double)kvh[i];
        }
        reduce[tid] = local;
        __syncthreads();
        for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            const float score = (float)reduce[0] * rsqrtf((float)head_dim);
            scores[k] = score;
            if (score > m_s) m_s = score;
        }
        __syncthreads();
    }

    local = 0.0;
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        local += (double)qh[i] * (double)current_kv[i];
    }
    reduce[tid] = local;
    __syncthreads();
    for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0u) {
        const float current_score = (float)reduce[0] * rsqrtf((float)head_dim);
        if (current_score > m_s) m_s = current_score;
        current_weight_s = expf(current_score - m_s);
        double denom = expf(attn_sink[head] - m_s) + current_weight_s;
        for (uint32_t k = 0; k < ring_count; ++k) {
            const float w = expf(scores[k] - m_s);
            weights[k] = w;
            denom += w;
        }
        denom_s = denom;
    }
    __syncthreads();

    float *oh = out + (uint64_t)head * head_dim;
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        double acc = (double)current_weight_s * (double)current_kv[i];
        for (uint32_t k = 0; k < ring_count; ++k) {
            uint32_t slot = oldest + k;
            if (slot >= ring_slots) slot -= ring_slots;
            const float *kvh = ring_kv + (uint64_t)slot * head_dim;
            acc += (double)weights[k] * (double)kvh[i];
        }
        oh[i] = denom_s > 0.0 ? (float)(acc / denom_s) : 0.0f;
    }
}

extern "C" int axiom_cuda_deepseek_sliding_attention_ring_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *current_kv,
        uint64_t current_kv_offset,
        const void *ring_kv,
        uint64_t ring_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps) {
    if (!cuda_runtime || !q || !current_kv || !ring_kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || ring_slots == 0 || ring_slots > 128u ||
        ring_head >= ring_slots || ring_count > ring_slots || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)current_kv;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)ring_kv;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_sliding_attention_ring_kernel<<<heads, 256>>>(
            (const float *)((const uint8_t *)qbuf->ptr + q_offset),
            (const float *)((const uint8_t *)cbuf->ptr + current_kv_offset),
            (const float *)((const uint8_t *)rbuf->ptr + ring_kv_offset),
            (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            head_dim,
            ring_slots,
            ring_head,
            ring_count,
            eps);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__device__ __forceinline__ static void axiom_deepseek_sliding_attention_context_inner(
        const float *qh,
        const float *current_kv,
        const float *ring_kv,
        const float *extra_kv,
        float sink,
        float *out_h,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_start,
        uint32_t ring_entries,
        uint32_t has_extra,
        uint32_t tid,
        double *reduce,
        float *scores,
        float *weights,
        float *m_s,
        double *denom_s,
        float *current_weight_s) {
    if (tid == 0u) *m_s = sink;
    __syncthreads();

    for (uint32_t k = 0; k < ring_entries; ++k) {
        uint32_t slot = ring_start + k;
        if (slot >= ring_slots) slot -= ring_slots;
        const float *kvh = ring_kv + (uint64_t)slot * head_dim;
        double local = 0.0;
        for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
            local += (double)qh[i] * (double)kvh[i];
        }
        reduce[tid] = local;
        __syncthreads();
        for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            const float score = (float)reduce[0] * rsqrtf((float)head_dim);
            scores[k] = score;
            if (score > *m_s) *m_s = score;
        }
        __syncthreads();
    }

    if (has_extra) {
        double local = 0.0;
        for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
            local += (double)qh[i] * (double)extra_kv[i];
        }
        reduce[tid] = local;
        __syncthreads();
        for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            const float score = (float)reduce[0] * rsqrtf((float)head_dim);
            scores[ring_entries] = score;
            if (score > *m_s) *m_s = score;
        }
        __syncthreads();
    }

    double local = 0.0;
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        local += (double)qh[i] * (double)current_kv[i];
    }
    reduce[tid] = local;
    __syncthreads();
    for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0u) {
        const float current_score = (float)reduce[0] * rsqrtf((float)head_dim);
        if (current_score > *m_s) *m_s = current_score;
        *current_weight_s = expf(current_score - *m_s);
        double denom = expf(sink - *m_s) + (double)(*current_weight_s);
        const uint32_t total_prev = ring_entries + (has_extra ? 1u : 0u);
        for (uint32_t k = 0; k < total_prev; ++k) {
            const float w = expf(scores[k] - *m_s);
            weights[k] = w;
            denom += w;
        }
        *denom_s = denom;
    }
    __syncthreads();

    const uint32_t total_prev = ring_entries + (has_extra ? 1u : 0u);
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        double acc = (double)(*current_weight_s) * (double)current_kv[i];
        for (uint32_t k = 0; k < ring_entries; ++k) {
            uint32_t slot = ring_start + k;
            if (slot >= ring_slots) slot -= ring_slots;
            const float *kvh = ring_kv + (uint64_t)slot * head_dim;
            acc += (double)weights[k] * (double)kvh[i];
        }
        if (has_extra) {
            acc += (double)weights[total_prev - 1u] * (double)extra_kv[i];
        }
        out_h[i] = *denom_s > 0.0 ? (float)(acc / *denom_s) : 0.0f;
    }
    __syncthreads();
}

__global__ static void axiom_deepseek_sliding_attention_ring2_causal_kernel(
        const float *q0,
        const float *q1,
        const float *kv0,
        const float *kv1,
        const float *ring_kv,
        const float *attn_sink,
        float *out0,
        float *out1,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ double reduce[256];
    __shared__ float scores[129];
    __shared__ float weights[129];
    __shared__ float m_s;
    __shared__ double denom_s;
    __shared__ float current_weight_s;
    const uint32_t old_start = ring_count == ring_slots ? ring_head : 0u;
    const uint32_t token1_entries = ring_count == ring_slots ? ring_slots - 1u : ring_count;
    uint32_t token1_start = ring_count == ring_slots ? ring_head + 1u : 0u;
    if (token1_start >= ring_slots) token1_start -= ring_slots;

    axiom_deepseek_sliding_attention_context_inner(
            q0 + (uint64_t)head * head_dim,
            kv0,
            ring_kv,
            NULL,
            attn_sink[head],
            out0 + (uint64_t)head * head_dim,
            head_dim,
            ring_slots,
            old_start,
            ring_count,
            0u,
            tid,
            reduce,
            scores,
            weights,
            &m_s,
            &denom_s,
            &current_weight_s);

    axiom_deepseek_sliding_attention_context_inner(
            q1 + (uint64_t)head * head_dim,
            kv1,
            ring_kv,
            kv0,
            attn_sink[head],
            out1 + (uint64_t)head * head_dim,
            head_dim,
            ring_slots,
            token1_start,
            token1_entries,
            1u,
            tid,
            reduce,
            scores,
            weights,
            &m_s,
            &denom_s,
            &current_weight_s);
}

__device__ __forceinline__ static const float *axiom_dev_sliding_extra_kv(
        const float *kv0,
        const float *kv1,
        const float *kv2,
        uint32_t idx) {
    return idx == 0u ? kv0 : (idx == 1u ? kv1 : kv2);
}

__device__ __forceinline__ static void axiom_dev_sliding_ring4_window(
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        uint32_t step,
        uint32_t *ring_start,
        uint32_t *ring_entries,
        uint32_t *extra_count) {
    uint32_t total_prev = ring_count + step;
    if (total_prev > ring_slots) total_prev = ring_slots;
    *extra_count = step;
    *ring_entries = total_prev >= step ? total_prev - step : 0u;
    uint32_t evict = ring_count + step > ring_slots ? ring_count + step - ring_slots : 0u;
    uint32_t start = ring_count == ring_slots ? ring_head : 0u;
    start += evict;
    while (start >= ring_slots) start -= ring_slots;
    *ring_start = start;
}

__device__ __forceinline__ static void axiom_deepseek_sliding_attention_context_multi_extra_inner(
        const float *qh,
        const float *current_kv,
        const float *ring_kv,
        const float *extra0,
        const float *extra1,
        const float *extra2,
        uint32_t extra_count,
        float sink,
        float *out_h,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_start,
        uint32_t ring_entries,
        uint32_t tid,
        double *reduce,
        float *scores,
        float *weights,
        float *m_s,
        double *denom_s,
        float *current_weight_s) {
    if (tid == 0u) *m_s = sink;
    __syncthreads();

    for (uint32_t k = 0; k < ring_entries; ++k) {
        uint32_t slot = ring_start + k;
        if (slot >= ring_slots) slot -= ring_slots;
        const float *kvh = ring_kv + (uint64_t)slot * head_dim;
        double local = 0.0;
        for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
            local += (double)qh[i] * (double)kvh[i];
        }
        reduce[tid] = local;
        __syncthreads();
        for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            const float score = (float)reduce[0] * rsqrtf((float)head_dim);
            scores[k] = score;
            if (score > *m_s) *m_s = score;
        }
        __syncthreads();
    }

    for (uint32_t e = 0; e < extra_count; e++) {
        const float *kvh = axiom_dev_sliding_extra_kv(extra0, extra1, extra2, e);
        double local = 0.0;
        for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
            local += (double)qh[i] * (double)kvh[i];
        }
        reduce[tid] = local;
        __syncthreads();
        for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            const float score = (float)reduce[0] * rsqrtf((float)head_dim);
            scores[ring_entries + e] = score;
            if (score > *m_s) *m_s = score;
        }
        __syncthreads();
    }

    double local = 0.0;
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        local += (double)qh[i] * (double)current_kv[i];
    }
    reduce[tid] = local;
    __syncthreads();
    for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0u) {
        const float current_score = (float)reduce[0] * rsqrtf((float)head_dim);
        if (current_score > *m_s) *m_s = current_score;
        *current_weight_s = expf(current_score - *m_s);
        double denom = expf(sink - *m_s) + (double)(*current_weight_s);
        const uint32_t total_prev = ring_entries + extra_count;
        for (uint32_t k = 0; k < total_prev; ++k) {
            const float w = expf(scores[k] - *m_s);
            weights[k] = w;
            denom += w;
        }
        *denom_s = denom;
    }
    __syncthreads();

    const uint32_t total_prev = ring_entries + extra_count;
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        double acc = (double)(*current_weight_s) * (double)current_kv[i];
        for (uint32_t k = 0; k < ring_entries; ++k) {
            uint32_t slot = ring_start + k;
            if (slot >= ring_slots) slot -= ring_slots;
            const float *kvh = ring_kv + (uint64_t)slot * head_dim;
            acc += (double)weights[k] * (double)kvh[i];
        }
        for (uint32_t e = 0; e < extra_count; e++) {
            const float *kvh = axiom_dev_sliding_extra_kv(extra0, extra1, extra2, e);
            acc += (double)weights[ring_entries + e] * (double)kvh[i];
        }
        out_h[i] = *denom_s > 0.0 ? (float)(acc / *denom_s) : 0.0f;
    }
    (void)total_prev;
    __syncthreads();
}

__global__ static void axiom_deepseek_sliding_attention_ring4_causal_kernel(
        const float *q0,
        const float *q1,
        const float *q2,
        const float *q3,
        const float *kv0,
        const float *kv1,
        const float *kv2,
        const float *kv3,
        const float *ring_kv,
        const float *attn_sink,
        float *out0,
        float *out1,
        float *out2,
        float *out3,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ double reduce[256];
    __shared__ float scores[131];
    __shared__ float weights[131];
    __shared__ float m_s;
    __shared__ double denom_s;
    __shared__ float current_weight_s;
    uint32_t ring_start = 0u;
    uint32_t ring_entries = 0u;
    uint32_t extra_count = 0u;

    axiom_dev_sliding_ring4_window(
            ring_slots, ring_head, ring_count, 0u,
            &ring_start, &ring_entries, &extra_count);
    axiom_deepseek_sliding_attention_context_multi_extra_inner(
            q0 + (uint64_t)head * head_dim,
            kv0,
            ring_kv,
            kv0,
            kv1,
            kv2,
            extra_count,
            attn_sink[head],
            out0 + (uint64_t)head * head_dim,
            head_dim,
            ring_slots,
            ring_start,
            ring_entries,
            tid,
            reduce,
            scores,
            weights,
            &m_s,
            &denom_s,
            &current_weight_s);

    axiom_dev_sliding_ring4_window(
            ring_slots, ring_head, ring_count, 1u,
            &ring_start, &ring_entries, &extra_count);
    axiom_deepseek_sliding_attention_context_multi_extra_inner(
            q1 + (uint64_t)head * head_dim,
            kv1,
            ring_kv,
            kv0,
            kv1,
            kv2,
            extra_count,
            attn_sink[head],
            out1 + (uint64_t)head * head_dim,
            head_dim,
            ring_slots,
            ring_start,
            ring_entries,
            tid,
            reduce,
            scores,
            weights,
            &m_s,
            &denom_s,
            &current_weight_s);

    axiom_dev_sliding_ring4_window(
            ring_slots, ring_head, ring_count, 2u,
            &ring_start, &ring_entries, &extra_count);
    axiom_deepseek_sliding_attention_context_multi_extra_inner(
            q2 + (uint64_t)head * head_dim,
            kv2,
            ring_kv,
            kv0,
            kv1,
            kv2,
            extra_count,
            attn_sink[head],
            out2 + (uint64_t)head * head_dim,
            head_dim,
            ring_slots,
            ring_start,
            ring_entries,
            tid,
            reduce,
            scores,
            weights,
            &m_s,
            &denom_s,
            &current_weight_s);

    axiom_dev_sliding_ring4_window(
            ring_slots, ring_head, ring_count, 3u,
            &ring_start, &ring_entries, &extra_count);
    axiom_deepseek_sliding_attention_context_multi_extra_inner(
            q3 + (uint64_t)head * head_dim,
            kv3,
            ring_kv,
            kv0,
            kv1,
            kv2,
            extra_count,
            attn_sink[head],
            out3 + (uint64_t)head * head_dim,
            head_dim,
            ring_slots,
            ring_start,
            ring_entries,
            tid,
            reduce,
            scores,
            weights,
            &m_s,
            &denom_s,
            &current_weight_s);
}

extern "C" int axiom_cuda_deepseek_sliding_attention_ring2_causal_f32_device(
        void *cuda_runtime,
        const void *q0,
        uint64_t q0_offset,
        const void *q1,
        uint64_t q1_offset,
        const void *kv0,
        uint64_t kv0_offset,
        const void *kv1,
        uint64_t kv1_offset,
        const void *ring_kv,
        uint64_t ring_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps) {
    (void)eps;
    if (!cuda_runtime || !q0 || !q1 || !kv0 || !kv1 || !ring_kv || !attn_sink || !out0 || !out1 ||
        heads == 0 || head_dim == 0 || ring_slots == 0 || ring_slots > 128u ||
        ring_head >= ring_slots || ring_count > ring_slots) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *q0buf = (const axiom_cuda_buffer *)q0;
    const axiom_cuda_buffer *q1buf = (const axiom_cuda_buffer *)q1;
    const axiom_cuda_buffer *kv0buf = (const axiom_cuda_buffer *)kv0;
    const axiom_cuda_buffer *kv1buf = (const axiom_cuda_buffer *)kv1;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)ring_kv;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_sliding_attention_ring2_causal_kernel<<<heads, 256>>>(
            (const float *)((const uint8_t *)q0buf->ptr + q0_offset),
            (const float *)((const uint8_t *)q1buf->ptr + q1_offset),
            (const float *)((const uint8_t *)kv0buf->ptr + kv0_offset),
            (const float *)((const uint8_t *)kv1buf->ptr + kv1_offset),
            (const float *)((const uint8_t *)rbuf->ptr + ring_kv_offset),
            (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            head_dim,
            ring_slots,
            ring_head,
            ring_count);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_sliding_attention_ring4_causal_f32_device(
        void *cuda_runtime,
        const void *q0,
        uint64_t q0_offset,
        const void *q1,
        uint64_t q1_offset,
        const void *q2,
        uint64_t q2_offset,
        const void *q3,
        uint64_t q3_offset,
        const void *kv0,
        uint64_t kv0_offset,
        const void *kv1,
        uint64_t kv1_offset,
        const void *kv2,
        uint64_t kv2_offset,
        const void *kv3,
        uint64_t kv3_offset,
        const void *ring_kv,
        uint64_t ring_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
        uint64_t out3_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps) {
    (void)eps;
    if (!cuda_runtime || !q0 || !q1 || !q2 || !q3 ||
        !kv0 || !kv1 || !kv2 || !kv3 || !ring_kv || !attn_sink ||
        !out0 || !out1 || !out2 || !out3 ||
        heads == 0 || head_dim == 0 || ring_slots == 0 || ring_slots > 128u ||
        ring_head >= ring_slots || ring_count > ring_slots) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *q0buf = (const axiom_cuda_buffer *)q0;
    const axiom_cuda_buffer *q1buf = (const axiom_cuda_buffer *)q1;
    const axiom_cuda_buffer *q2buf = (const axiom_cuda_buffer *)q2;
    const axiom_cuda_buffer *q3buf = (const axiom_cuda_buffer *)q3;
    const axiom_cuda_buffer *kv0buf = (const axiom_cuda_buffer *)kv0;
    const axiom_cuda_buffer *kv1buf = (const axiom_cuda_buffer *)kv1;
    const axiom_cuda_buffer *kv2buf = (const axiom_cuda_buffer *)kv2;
    const axiom_cuda_buffer *kv3buf = (const axiom_cuda_buffer *)kv3;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)ring_kv;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    axiom_cuda_buffer *o2buf = (axiom_cuda_buffer *)out2;
    axiom_cuda_buffer *o3buf = (axiom_cuda_buffer *)out3;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_sliding_attention_ring4_causal_kernel<<<heads, 256>>>(
            (const float *)((const uint8_t *)q0buf->ptr + q0_offset),
            (const float *)((const uint8_t *)q1buf->ptr + q1_offset),
            (const float *)((const uint8_t *)q2buf->ptr + q2_offset),
            (const float *)((const uint8_t *)q3buf->ptr + q3_offset),
            (const float *)((const uint8_t *)kv0buf->ptr + kv0_offset),
            (const float *)((const uint8_t *)kv1buf->ptr + kv1_offset),
            (const float *)((const uint8_t *)kv2buf->ptr + kv2_offset),
            (const float *)((const uint8_t *)kv3buf->ptr + kv3_offset),
            (const float *)((const uint8_t *)rbuf->ptr + ring_kv_offset),
            (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            (float *)((uint8_t *)o2buf->ptr + out2_offset),
            (float *)((uint8_t *)o3buf->ptr + out3_offset),
            head_dim,
            ring_slots,
            ring_head,
            ring_count);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_attention_multi_kv_single_kernel(
        const float *q,
        const float *kv,
        const float *attn_sink,
        float *out,
        uint32_t head_dim,
        uint32_t kv_count,
        float eps) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ double reduce[256];
    __shared__ float scores[32];
    __shared__ float weights[32];
    __shared__ double denom_s;
    const float *qh = q + (uint64_t)head * head_dim;
    double local = 0.0;
    for (uint32_t k = 0; k < kv_count; ++k) {
        const float *kvh = kv + (uint64_t)k * head_dim;
        local = 0.0;
        for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
            local += (double)qh[i] * (double)kvh[i];
        }
        reduce[tid] = local;
        __syncthreads();
        for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) scores[k] = (float)reduce[0] * rsqrtf((float)head_dim);
        __syncthreads();
    }
    if (tid == 0u) {
        float m = attn_sink[head];
        for (uint32_t k = 0; k < kv_count; ++k) {
            if (scores[k] > m) m = scores[k];
        }
        double denom = expf(attn_sink[head] - m);
        for (uint32_t k = 0; k < kv_count; ++k) {
            const float weight = expf(scores[k] - m);
            weights[k] = weight;
            denom += weight;
        }
        denom_s = denom;
    }
    __syncthreads();
    float *oh = out + (uint64_t)head * head_dim;
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        double acc = 0.0;
        for (uint32_t k = 0; k < kv_count; ++k) {
            acc += (double)weights[k] * (double)kv[(uint64_t)k * head_dim + i];
        }
        oh[i] = denom_s > 0.0 ? (float)(acc / denom_s) : 0.0f;
    }
}

__global__ static void axiom_deepseek_attention_current_history_kernel(
        const float *q,
        const float *current_kv,
        const float *history_kv,
        const float *attn_sink,
        float *out,
        uint32_t head_dim,
        uint32_t has_history,
        float eps) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ double reduce[256];
    __shared__ float scores[2];
    __shared__ float weights[2];
    __shared__ double denom_s;
    const float *qh = q + (uint64_t)head * head_dim;
    double local = 0.0;
    const uint32_t kv_count = has_history ? 2u : 1u;
    for (uint32_t k = 0; k < kv_count; ++k) {
        const float *kvh = k == 0u ? current_kv : history_kv;
        local = 0.0;
        for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
            local += (double)qh[i] * (double)kvh[i];
        }
        reduce[tid] = local;
        __syncthreads();
        for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) scores[k] = (float)reduce[0] * rsqrtf((float)head_dim);
        __syncthreads();
    }
    if (tid == 0u) {
        float m = attn_sink[head];
        for (uint32_t k = 0; k < kv_count; ++k) {
            if (scores[k] > m) m = scores[k];
        }
        double denom = expf(attn_sink[head] - m);
        for (uint32_t k = 0; k < kv_count; ++k) {
            const float weight = expf(scores[k] - m);
            weights[k] = weight;
            denom += weight;
        }
        denom_s = denom;
    }
    __syncthreads();
    float *oh = out + (uint64_t)head * head_dim;
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        double acc = (double)weights[0] * (double)current_kv[i];
        if (has_history) acc += (double)weights[1] * (double)history_kv[i];
        oh[i] = denom_s > 0.0 ? (float)(acc / denom_s) : 0.0f;
    }
}

extern "C" int axiom_cuda_deepseek_attention_multi_kv_single_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *kv,
        uint64_t kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t kv_count,
        float eps) {
    if (!cuda_runtime || !q || !kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || kv_count == 0 || kv_count > 32 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *kvbuf = (const axiom_cuda_buffer *)kv;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_attention_multi_kv_single_kernel<<<heads, 256>>>(
            (const float *)((const uint8_t *)qbuf->ptr + q_offset),
            (const float *)((const uint8_t *)kvbuf->ptr + kv_offset),
            (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            head_dim,
            kv_count,
            eps);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_attention_current_history_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *current_kv,
        uint64_t current_kv_offset,
        const void *history_kv,
        uint64_t history_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t has_history,
        float eps) {
    if (!cuda_runtime || !q || !current_kv || !history_kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || has_history > 1u || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *ckvbuf = (const axiom_cuda_buffer *)current_kv;
    const axiom_cuda_buffer *hkvbuf = (const axiom_cuda_buffer *)history_kv;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_attention_current_history_kernel<<<heads, 256>>>(
            (const float *)((const uint8_t *)qbuf->ptr + q_offset),
            (const float *)((const uint8_t *)ckvbuf->ptr + current_kv_offset),
            (const float *)((const uint8_t *)hkvbuf->ptr + history_kv_offset),
            (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            head_dim,
            has_history,
            eps);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__device__ static uint32_t axiom_dev_ring_slot(uint32_t oldest, uint32_t i, uint32_t ratio);

__global__ static void axiom_deepseek_attention_raw_comp_ring_kernel(
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *attn_sink,
        float *out,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        float eps) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t total = raw_count + comp_count;
    extern __shared__ unsigned char smem[];
    double *reduce = (double *)smem;
    float *scores = (float *)(reduce + 256);
    float *weights = scores + total;
    __shared__ double denom_s;
    const float *qh = q + (uint64_t)head * head_dim;
    double local = 0.0;

    const uint32_t oldest = raw_count == raw_slots ? raw_head : 0u;
    for (uint32_t k = 0; k < total; ++k) {
        const float *kvh;
        if (k < raw_count) {
            const uint32_t slot = axiom_dev_ring_slot(oldest, k, raw_slots);
            kvh = raw_kv + (uint64_t)slot * head_dim;
        } else {
            kvh = comp_kv + (uint64_t)(k - raw_count) * head_dim;
        }
        local = 0.0;
        for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
            local += (double)qh[i] * (double)kvh[i];
        }
        reduce[tid] = local;
        __syncthreads();
        for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) scores[k] = (float)reduce[0] * rsqrtf((float)head_dim);
        __syncthreads();
    }
    if (tid == 0u) {
        float m = attn_sink[head];
        for (uint32_t k = 0; k < total; ++k) if (scores[k] > m) m = scores[k];
        double denom = expf(attn_sink[head] - m);
        for (uint32_t k = 0; k < total; ++k) {
            const float weight = expf(scores[k] - m);
            weights[k] = weight;
            denom += weight;
        }
        denom_s = denom;
    }
    __syncthreads();
    float *oh = out + (uint64_t)head * head_dim;
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        double acc = 0.0;
        for (uint32_t k = 0; k < total; ++k) {
            const float *kvh;
            if (k < raw_count) {
                const uint32_t slot = axiom_dev_ring_slot(oldest, k, raw_slots);
                kvh = raw_kv + (uint64_t)slot * head_dim;
            } else {
                kvh = comp_kv + (uint64_t)(k - raw_count) * head_dim;
            }
            acc += (double)weights[k] * (double)kvh[i];
        }
        oh[i] = denom_s > 0.0 ? (float)(acc / denom_s) : 0.0f;
    }
}

__device__ __forceinline__ static float axiom_dev_dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static bool axiom_dev_attn_sink_forces_zero(float sink) {
    return isnan(sink) || (isinf(sink) && sink > 0.0f);
}

__device__ __forceinline__ static bool axiom_dev_attn_sink_is_active(float sink) {
    return isfinite(sink);
}

__global__ static void axiom_deepseek_attention_raw_comp_ring_heads8_kernel(
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ attn_sink,
        float *__restrict__ out,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = (uint32_t)blockIdx.x * 8u + warp;
    if (warp >= 8u) return;
    __shared__ float4 kv_shared[4u * 128u];
    const uint32_t head_dim = 512u;
    const uint32_t total = raw_count + comp_count;
    const uint32_t oldest = raw_count == raw_slots ? raw_head : 0u;
    const bool valid_head = head < 64u;
    const float4 *q4 = valid_head ? (const float4 *)(q + (uint64_t)head * head_dim) : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0;
    float4 q2 = q0;
    float4 q3 = q0;
    if (valid_head) {
        q0 = q4[lane + 0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }
    const float scale = rsqrtf((float)head_dim);
    float max_s = -3.4028234663852886e+38F;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0;
    float4 o2 = o0;
    float4 o3 = o0;
    for (uint32_t row0 = 0; row0 < total; row0 += 4u) {
        const uint32_t nr = total - row0 < 4u ? total - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src;
            if (sr < raw_count) {
                const uint32_t slot = axiom_dev_ring_slot(oldest, sr, raw_slots);
                src = (const float4 *)(raw_kv + (uint64_t)slot * head_dim);
            } else {
                src = (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            }
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                const float4 k0 = kv4[lane + 0u];
                const float4 k1 = kv4[lane + 32u];
                const float4 k2 = kv4[lane + 64u];
                const float4 k3 = kv4[lane + 96u];
                float score = axiom_dev_dot4(q0, k0) +
                              axiom_dev_dot4(q1, k1) +
                              axiom_dev_dot4(q2, k2) +
                              axiom_dev_dot4(q3, k3);
                #pragma unroll
                for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                    score += __shfl_down_sync(0xffffffffu, score, offset);
                }
                score = __shfl_sync(0xffffffffu, score, 0) * scale;
                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        const float sink = attn_sink[head];
        if (axiom_dev_attn_sink_forces_zero(sink)) {
            o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            o1 = o0;
            o2 = o0;
            o3 = o0;
            sum_s = 1.0f;
        } else if (axiom_dev_attn_sink_is_active(sink)) {
            const float new_m = fmaxf(max_s, sink);
            const float old_scale = expf(max_s - new_m);
            const float sink_scale = expf(sink - new_m);
            sum_s = sum_s * old_scale + sink_scale;
            o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
            o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
            o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
            o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;
        }
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(out + (uint64_t)head * head_dim);
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void axiom_deepseek_csa_indexer_score_kernel(
        float *scores,
        const float *q,
        const float *index_weights,
        const float *index_comp,
        uint32_t comp_count) {
    const uint32_t comp = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    if (comp >= comp_count || tid >= 128u) return;
    const float *kv = index_comp + (uint64_t)comp * 128u;
    __shared__ float partial[128];
    float total = 0.0f;
    for (uint32_t h = 0; h < 64u; ++h) {
        const float *qh = q + (uint64_t)h * 128u;
        float dot = 0.0f;
        dot += qh[tid] * kv[tid];
        partial[tid] = dot;
        __syncthreads();
        for (uint32_t stride = 64u; stride > 0u; stride >>= 1u) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            total += fmaxf(partial[0], 0.0f) * index_weights[h];
        }
        __syncthreads();
    }
    if (tid == 0u) scores[comp] = total * 0.011048543456039804f;
}

__global__ static void axiom_deepseek_csa_indexer_topk_kernel(
        uint32_t *selected,
        float *scores,
        uint32_t comp_count,
        uint32_t topk) {
    if (threadIdx.x != 0) return;
    for (uint32_t k = 0; k < topk; ++k) {
        uint32_t best = 0u;
        float best_score = scores[0];
        for (uint32_t c = 1u; c < comp_count; ++c) {
            const float v = scores[c];
            if (v > best_score || (v == best_score && c < best)) {
                best = c;
                best_score = v;
            }
        }
        selected[k] = best;
        scores[best] = -INFINITY;
    }
}

__global__ static void axiom_deepseek_attention_raw_selected_comp_ring_kernel(
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const uint32_t *selected,
        const float *attn_sink,
        float *out,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t selected_count,
        uint32_t *bad_selected) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t total = raw_count + selected_count;
    extern __shared__ unsigned char smem[];
    double *reduce = (double *)smem;
    float *scores = (float *)(reduce + 256);
    float *weights = scores + total;
    __shared__ double denom_s;
    const float *qh = q + (uint64_t)head * head_dim;
    const uint32_t oldest = raw_count == raw_slots ? raw_head : 0u;

    for (uint32_t k = 0; k < total; ++k) {
        const float *kvh = NULL;
        uint32_t valid = 1u;
        if (k < raw_count) {
            const uint32_t slot = axiom_dev_ring_slot(oldest, k, raw_slots);
            kvh = raw_kv + (uint64_t)slot * head_dim;
        } else {
            const uint32_t idx = selected[k - raw_count];
            valid = idx < comp_count ? 1u : 0u;
            if (valid) kvh = comp_kv + (uint64_t)idx * head_dim;
        }
        double local = 0.0;
        if (valid) {
            for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
                local += (double)qh[i] * (double)kvh[i];
            }
        }
        reduce[tid] = local;
        __syncthreads();
        for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0u) {
            if (valid) {
                scores[k] = (float)reduce[0] * rsqrtf((float)head_dim);
            } else {
                scores[k] = -INFINITY;
                if (bad_selected) atomicExch(bad_selected, 1u);
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        float m = attn_sink[head];
        for (uint32_t k = 0; k < total; ++k) if (scores[k] > m) m = scores[k];
        double denom = expf(attn_sink[head] - m);
        for (uint32_t k = 0; k < total; ++k) {
            const float weight = expf(scores[k] - m);
            weights[k] = weight;
            denom += weight;
        }
        denom_s = denom;
    }
    __syncthreads();
    float *oh = out + (uint64_t)head * head_dim;
    for (uint32_t i = tid; i < head_dim; i += (uint32_t)blockDim.x) {
        double acc = 0.0;
        for (uint32_t k = 0; k < total; ++k) {
            const float *kvh = NULL;
            uint32_t valid = 1u;
            if (k < raw_count) {
                const uint32_t slot = axiom_dev_ring_slot(oldest, k, raw_slots);
                kvh = raw_kv + (uint64_t)slot * head_dim;
            } else {
                const uint32_t idx = selected[k - raw_count];
                valid = idx < comp_count ? 1u : 0u;
                if (valid) kvh = comp_kv + (uint64_t)idx * head_dim;
            }
            if (valid) acc += (double)weights[k] * (double)kvh[i];
        }
        oh[i] = denom_s > 0.0 ? (float)(acc / denom_s) : 0.0f;
    }
}

__global__ static void axiom_deepseek_attention_raw_selected_comp_ring_heads8_kernel(
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const uint32_t *__restrict__ selected,
        const float *__restrict__ attn_sink,
        float *__restrict__ out,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t selected_count,
        uint32_t *bad_selected) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = (uint32_t)blockIdx.x * 8u + warp;
    if (warp >= 8u) return;
    __shared__ float4 kv_shared[4u * 128u];
    const uint32_t head_dim = 512u;
    const uint32_t total = raw_count + selected_count;
    const uint32_t oldest = raw_count == raw_slots ? raw_head : 0u;
    const bool valid_head = head < 64u;
    const float4 *q4 = valid_head ? (const float4 *)(q + (uint64_t)head * head_dim) : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0;
    float4 q2 = q0;
    float4 q3 = q0;
    if (valid_head) {
        q0 = q4[lane + 0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }
    const float scale = rsqrtf((float)head_dim);
    float max_s = -3.4028234663852886e+38F;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0;
    float4 o2 = o0;
    float4 o3 = o0;
    for (uint32_t row0 = 0; row0 < total; row0 += 4u) {
        const uint32_t nr = total - row0 < 4u ? total - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = NULL;
            uint32_t valid = 1u;
            if (sr < raw_count) {
                const uint32_t slot = axiom_dev_ring_slot(oldest, sr, raw_slots);
                src = (const float4 *)(raw_kv + (uint64_t)slot * head_dim);
            } else {
                const uint32_t idx = selected[sr - raw_count];
                valid = idx < comp_count ? 1u : 0u;
                if (valid) src = (const float4 *)(comp_kv + (uint64_t)idx * head_dim);
            }
            if (valid) {
                kv_shared[off] = src[c4];
            } else {
                kv_shared[off] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (bad_selected) atomicExch(bad_selected, 1u);
            }
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                const float4 k0 = kv4[lane + 0u];
                const float4 k1 = kv4[lane + 32u];
                const float4 k2 = kv4[lane + 64u];
                const float4 k3 = kv4[lane + 96u];
                float score = axiom_dev_dot4(q0, k0) +
                              axiom_dev_dot4(q1, k1) +
                              axiom_dev_dot4(q2, k2) +
                              axiom_dev_dot4(q3, k3);
                #pragma unroll
                for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                    score += __shfl_down_sync(0xffffffffu, score, offset);
                }
                score = __shfl_sync(0xffffffffu, score, 0) * scale;
                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        const float sink = attn_sink[head];
        if (axiom_dev_attn_sink_forces_zero(sink)) {
            o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            o1 = o0;
            o2 = o0;
            o3 = o0;
            sum_s = 1.0f;
        } else if (axiom_dev_attn_sink_is_active(sink)) {
            const float new_m = fmaxf(max_s, sink);
            const float old_scale = expf(max_s - new_m);
            const float sink_scale = expf(sink - new_m);
            sum_s = sum_s * old_scale + sink_scale;
            o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
            o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
            o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
            o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;
        }
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(out + (uint64_t)head * head_dim);
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__device__ __forceinline__ static const float *axiom_dev_raw_comp_ring2_source(
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ kv0,
        const float *__restrict__ kv1,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t row,
        uint32_t slot0,
        uint32_t slot1,
        uint32_t include_kv1) {
    if (row < raw_count) {
        const uint32_t oldest = raw_count == raw_slots ? raw_head : 0u;
        uint32_t slot = oldest + row;
        if (slot >= raw_slots) slot -= raw_slots;
        if (include_kv1 && slot == slot1) return kv1;
        if (slot == slot0) return kv0;
        return raw_kv + (uint64_t)slot * head_dim;
    }
    return comp_kv + (uint64_t)(row - raw_count) * head_dim;
}

__device__ static void axiom_deepseek_attention_raw_comp_ring2_heads8_inner(
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ kv0,
        const float *__restrict__ kv1,
        const float *__restrict__ attn_sink,
        float *__restrict__ out,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t slot0,
        uint32_t slot1,
        uint32_t include_kv1,
        float4 *kv_shared) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = (uint32_t)blockIdx.x * 8u + warp;
    if (warp >= 8u) return;
    const uint32_t head_dim = 512u;
    const uint32_t total = raw_count + comp_count;
    const bool valid_head = head < 64u;
    const float4 *q4 = valid_head ? (const float4 *)(q + (uint64_t)head * head_dim) : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0;
    float4 q2 = q0;
    float4 q3 = q0;
    if (valid_head) {
        q0 = q4[lane + 0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }
    const float scale = rsqrtf((float)head_dim);
    float max_s = -3.4028234663852886e+38F;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0;
    float4 o2 = o0;
    float4 o3 = o0;
    for (uint32_t row0 = 0; row0 < total; row0 += 4u) {
        const uint32_t nr = total - row0 < 4u ? total - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = (const float4 *)axiom_dev_raw_comp_ring2_source(
                    raw_kv, comp_kv, kv0, kv1, head_dim, raw_slots, raw_head,
                    raw_count, sr, slot0, slot1, include_kv1);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                const float4 k0 = kv4[lane + 0u];
                const float4 k1 = kv4[lane + 32u];
                const float4 k2 = kv4[lane + 64u];
                const float4 k3 = kv4[lane + 96u];
                float score = axiom_dev_dot4(q0, k0) +
                              axiom_dev_dot4(q1, k1) +
                              axiom_dev_dot4(q2, k2) +
                              axiom_dev_dot4(q3, k3);
                #pragma unroll
                for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                    score += __shfl_down_sync(0xffffffffu, score, offset);
                }
                score = __shfl_sync(0xffffffffu, score, 0) * scale;
                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        const float sink = attn_sink[head];
        if (axiom_dev_attn_sink_forces_zero(sink)) {
            o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            o1 = o0;
            o2 = o0;
            o3 = o0;
            sum_s = 1.0f;
        } else if (axiom_dev_attn_sink_is_active(sink)) {
            const float new_m = fmaxf(max_s, sink);
            const float old_scale = expf(max_s - new_m);
            const float sink_scale = expf(sink - new_m);
            sum_s = sum_s * old_scale + sink_scale;
            o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
            o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
            o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
            o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;
        }
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(out + (uint64_t)head * head_dim);
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
    __syncthreads();
}

__global__ static void axiom_deepseek_attention_raw_comp_ring2_causal_heads8_kernel(
        const float *__restrict__ q0,
        const float *__restrict__ q1,
        const float *__restrict__ kv0,
        const float *__restrict__ kv1,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ attn_sink,
        float *__restrict__ out0,
        float *__restrict__ out1,
        uint32_t raw_slots,
        uint32_t raw_head0,
        uint32_t raw_count0,
        uint32_t comp_count0,
        uint32_t comp_count1) {
    __shared__ float4 kv_shared[4u * 128u];
    const uint32_t slot0 = raw_head0 == 0u ? raw_slots - 1u : raw_head0 - 1u;
    const uint32_t slot1 = raw_head0;
    uint32_t raw_head1 = raw_head0 + 1u;
    if (raw_head1 >= raw_slots) raw_head1 = 0u;
    const uint32_t raw_count1 = raw_count0 < raw_slots ? raw_count0 + 1u : raw_slots;

    axiom_deepseek_attention_raw_comp_ring2_heads8_inner(
            q0, raw_kv, comp_kv, kv0, kv1, attn_sink, out0,
            raw_slots, raw_head0, raw_count0, comp_count0,
            slot0, slot1, 0u, kv_shared);
    axiom_deepseek_attention_raw_comp_ring2_heads8_inner(
            q1, raw_kv, comp_kv, kv0, kv1, attn_sink, out1,
            raw_slots, raw_head1, raw_count1, comp_count1,
            slot0, slot1, 1u, kv_shared);
}

__device__ __forceinline__ static const float *axiom_dev_raw_comp_ring4_source(
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ kv0,
        const float *__restrict__ kv1,
        const float *__restrict__ kv2,
        const float *__restrict__ kv3,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t row,
        uint32_t slot0,
        uint32_t slot1,
        uint32_t slot2,
        uint32_t slot3,
        uint32_t include_count) {
    if (row < raw_count) {
        const uint32_t oldest = raw_count == raw_slots ? raw_head : 0u;
        uint32_t slot = oldest + row;
        if (slot >= raw_slots) slot -= raw_slots;
        if (include_count > 3u && slot == slot3) return kv3;
        if (include_count > 2u && slot == slot2) return kv2;
        if (include_count > 1u && slot == slot1) return kv1;
        if (slot == slot0) return kv0;
        return raw_kv + (uint64_t)slot * head_dim;
    }
    return comp_kv + (uint64_t)(row - raw_count) * head_dim;
}

__device__ static void axiom_deepseek_attention_raw_comp_ring4_heads8_inner(
        const float *__restrict__ q,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ kv0,
        const float *__restrict__ kv1,
        const float *__restrict__ kv2,
        const float *__restrict__ kv3,
        const float *__restrict__ attn_sink,
        float *__restrict__ out,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t slot0,
        uint32_t slot1,
        uint32_t slot2,
        uint32_t slot3,
        uint32_t include_count,
        float4 *kv_shared) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = (uint32_t)blockIdx.x * 8u + warp;
    if (warp >= 8u) return;
    const uint32_t head_dim = 512u;
    const uint32_t total = raw_count + comp_count;
    const bool valid_head = head < 64u;
    const float4 *q4 = valid_head ? (const float4 *)(q + (uint64_t)head * head_dim) : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0;
    float4 q2 = q0;
    float4 q3 = q0;
    if (valid_head) {
        q0 = q4[lane + 0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }
    const float scale = rsqrtf((float)head_dim);
    float max_s = -3.4028234663852886e+38F;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0;
    float4 o2 = o0;
    float4 o3 = o0;
    for (uint32_t row0 = 0; row0 < total; row0 += 4u) {
        const uint32_t nr = total - row0 < 4u ? total - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = (const float4 *)axiom_dev_raw_comp_ring4_source(
                    raw_kv, comp_kv, kv0, kv1, kv2, kv3, head_dim, raw_slots,
                    raw_head, raw_count, sr, slot0, slot1, slot2, slot3, include_count);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                const float4 k0 = kv4[lane + 0u];
                const float4 k1 = kv4[lane + 32u];
                const float4 k2 = kv4[lane + 64u];
                const float4 k3 = kv4[lane + 96u];
                float score = axiom_dev_dot4(q0, k0) +
                              axiom_dev_dot4(q1, k1) +
                              axiom_dev_dot4(q2, k2) +
                              axiom_dev_dot4(q3, k3);
                #pragma unroll
                for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                    score += __shfl_down_sync(0xffffffffu, score, offset);
                }
                score = __shfl_sync(0xffffffffu, score, 0) * scale;
                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        const float sink = attn_sink[head];
        if (axiom_dev_attn_sink_forces_zero(sink)) {
            o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            o1 = o0;
            o2 = o0;
            o3 = o0;
            sum_s = 1.0f;
        } else if (axiom_dev_attn_sink_is_active(sink)) {
            const float new_m = fmaxf(max_s, sink);
            const float old_scale = expf(max_s - new_m);
            const float sink_scale = expf(sink - new_m);
            sum_s = sum_s * old_scale + sink_scale;
            o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
            o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
            o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
            o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;
        }
        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(out + (uint64_t)head * head_dim);
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
    __syncthreads();
}

__global__ static void axiom_deepseek_attention_raw_comp_ring4_causal_heads8_kernel(
        const float *__restrict__ q0,
        const float *__restrict__ q1,
        const float *__restrict__ q2,
        const float *__restrict__ q3,
        const float *__restrict__ kv0,
        const float *__restrict__ kv1,
        const float *__restrict__ kv2,
        const float *__restrict__ kv3,
        const float *__restrict__ raw_kv,
        const float *__restrict__ comp_kv,
        const float *__restrict__ attn_sink,
        float *__restrict__ out0,
        float *__restrict__ out1,
        float *__restrict__ out2,
        float *__restrict__ out3,
        uint32_t raw_slots,
        uint32_t raw_head0,
        uint32_t raw_count0,
        uint32_t comp_count0,
        uint32_t comp_count1,
        uint32_t comp_count2,
        uint32_t comp_count3) {
    __shared__ float4 kv_shared[4u * 128u];
    const uint32_t slot0 = raw_head0 == 0u ? raw_slots - 1u : raw_head0 - 1u;
    uint32_t slot1 = raw_head0;
    uint32_t slot2 = slot1 + 1u;
    if (slot2 >= raw_slots) slot2 -= raw_slots;
    uint32_t slot3 = slot2 + 1u;
    if (slot3 >= raw_slots) slot3 -= raw_slots;
    uint32_t raw_head1 = raw_head0 + 1u;
    if (raw_head1 >= raw_slots) raw_head1 = 0u;
    uint32_t raw_head2 = raw_head1 + 1u;
    if (raw_head2 >= raw_slots) raw_head2 = 0u;
    uint32_t raw_head3 = raw_head2 + 1u;
    if (raw_head3 >= raw_slots) raw_head3 = 0u;
    const uint32_t raw_count1 = raw_count0 < raw_slots ? raw_count0 + 1u : raw_slots;
    const uint32_t raw_count2 = raw_count1 < raw_slots ? raw_count1 + 1u : raw_slots;
    const uint32_t raw_count3 = raw_count2 < raw_slots ? raw_count2 + 1u : raw_slots;

    axiom_deepseek_attention_raw_comp_ring4_heads8_inner(
            q0, raw_kv, comp_kv, kv0, kv1, kv2, kv3, attn_sink, out0,
            raw_slots, raw_head0, raw_count0, comp_count0,
            slot0, slot1, slot2, slot3, 1u, kv_shared);
    axiom_deepseek_attention_raw_comp_ring4_heads8_inner(
            q1, raw_kv, comp_kv, kv0, kv1, kv2, kv3, attn_sink, out1,
            raw_slots, raw_head1, raw_count1, comp_count1,
            slot0, slot1, slot2, slot3, 2u, kv_shared);
    axiom_deepseek_attention_raw_comp_ring4_heads8_inner(
            q2, raw_kv, comp_kv, kv0, kv1, kv2, kv3, attn_sink, out2,
            raw_slots, raw_head2, raw_count2, comp_count2,
            slot0, slot1, slot2, slot3, 3u, kv_shared);
    axiom_deepseek_attention_raw_comp_ring4_heads8_inner(
            q3, raw_kv, comp_kv, kv0, kv1, kv2, kv3, attn_sink, out3,
            raw_slots, raw_head3, raw_count3, comp_count3,
            slot0, slot1, slot2, slot3, 4u, kv_shared);
}

extern "C" int axiom_cuda_deepseek_attention_raw_comp_ring2_causal_f32_device(
        void *cuda_runtime,
        const void *q0,
        uint64_t q0_offset,
        const void *q1,
        uint64_t q1_offset,
        const void *kv0,
        uint64_t kv0_offset,
        const void *kv1,
        uint64_t kv1_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head0,
        uint32_t raw_count0,
        uint32_t comp_count0,
        uint32_t comp_count1,
        float eps) {
    (void)eps;
    if (!cuda_runtime || !q0 || !q1 || !kv0 || !kv1 || !raw_kv_ring || !comp_kv ||
        !attn_sink || !out0 || !out1 || heads != 64u || head_dim != 512u ||
        raw_slots == 0 || raw_slots > 128u || raw_head0 >= raw_slots ||
        raw_count0 == 0 || raw_count0 > raw_slots ||
        raw_count0 + comp_count0 == 0u || raw_count0 + comp_count0 > 16384u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t raw_count1 = raw_count0 < raw_slots ? raw_count0 + 1u : raw_slots;
    if (raw_count1 + comp_count1 == 0u || raw_count1 + comp_count1 > 16384u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *q0buf = (const axiom_cuda_buffer *)q0;
    const axiom_cuda_buffer *q1buf = (const axiom_cuda_buffer *)q1;
    const axiom_cuda_buffer *kv0buf = (const axiom_cuda_buffer *)kv0;
    const axiom_cuda_buffer *kv1buf = (const axiom_cuda_buffer *)kv1;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)raw_kv_ring;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)comp_kv;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int grid = (int)((heads + 7u) / 8u);
    axiom_deepseek_attention_raw_comp_ring2_causal_heads8_kernel<<<grid, 256>>>(
            (const float *)((const uint8_t *)q0buf->ptr + q0_offset),
            (const float *)((const uint8_t *)q1buf->ptr + q1_offset),
            (const float *)((const uint8_t *)kv0buf->ptr + kv0_offset),
            (const float *)((const uint8_t *)kv1buf->ptr + kv1_offset),
            (const float *)((const uint8_t *)rbuf->ptr + raw_kv_ring_offset),
            (const float *)((const uint8_t *)cbuf->ptr + comp_kv_offset),
            (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            raw_slots,
            raw_head0,
            raw_count0,
            comp_count0,
            comp_count1);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_attention_raw_comp_ring4_causal_f32_device(
        void *cuda_runtime,
        const void *q0,
        uint64_t q0_offset,
        const void *q1,
        uint64_t q1_offset,
        const void *q2,
        uint64_t q2_offset,
        const void *q3,
        uint64_t q3_offset,
        const void *kv0,
        uint64_t kv0_offset,
        const void *kv1,
        uint64_t kv1_offset,
        const void *kv2,
        uint64_t kv2_offset,
        const void *kv3,
        uint64_t kv3_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
        uint64_t out3_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head0,
        uint32_t raw_count0,
        uint32_t comp_count0,
        uint32_t comp_count1,
        uint32_t comp_count2,
        uint32_t comp_count3,
        float eps) {
    (void)eps;
    if (!cuda_runtime || !q0 || !q1 || !q2 || !q3 ||
        !kv0 || !kv1 || !kv2 || !kv3 || !raw_kv_ring || !comp_kv ||
        !attn_sink || !out0 || !out1 || !out2 || !out3 ||
        heads != 64u || head_dim != 512u ||
        raw_slots == 0 || raw_slots > 128u || raw_head0 >= raw_slots ||
        raw_count0 == 0 || raw_count0 > raw_slots ||
        raw_count0 + comp_count0 == 0u || raw_count0 + comp_count0 > 16384u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t raw_count1 = raw_count0 < raw_slots ? raw_count0 + 1u : raw_slots;
    const uint32_t raw_count2 = raw_count1 < raw_slots ? raw_count1 + 1u : raw_slots;
    const uint32_t raw_count3 = raw_count2 < raw_slots ? raw_count2 + 1u : raw_slots;
    if (raw_count1 + comp_count1 == 0u || raw_count1 + comp_count1 > 16384u ||
        raw_count2 + comp_count2 == 0u || raw_count2 + comp_count2 > 16384u ||
        raw_count3 + comp_count3 == 0u || raw_count3 + comp_count3 > 16384u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *q0buf = (const axiom_cuda_buffer *)q0;
    const axiom_cuda_buffer *q1buf = (const axiom_cuda_buffer *)q1;
    const axiom_cuda_buffer *q2buf = (const axiom_cuda_buffer *)q2;
    const axiom_cuda_buffer *q3buf = (const axiom_cuda_buffer *)q3;
    const axiom_cuda_buffer *kv0buf = (const axiom_cuda_buffer *)kv0;
    const axiom_cuda_buffer *kv1buf = (const axiom_cuda_buffer *)kv1;
    const axiom_cuda_buffer *kv2buf = (const axiom_cuda_buffer *)kv2;
    const axiom_cuda_buffer *kv3buf = (const axiom_cuda_buffer *)kv3;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)raw_kv_ring;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)comp_kv;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    axiom_cuda_buffer *o2buf = (axiom_cuda_buffer *)out2;
    axiom_cuda_buffer *o3buf = (axiom_cuda_buffer *)out3;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int grid = (int)((heads + 7u) / 8u);
    axiom_deepseek_attention_raw_comp_ring4_causal_heads8_kernel<<<grid, 256>>>(
            (const float *)((const uint8_t *)q0buf->ptr + q0_offset),
            (const float *)((const uint8_t *)q1buf->ptr + q1_offset),
            (const float *)((const uint8_t *)q2buf->ptr + q2_offset),
            (const float *)((const uint8_t *)q3buf->ptr + q3_offset),
            (const float *)((const uint8_t *)kv0buf->ptr + kv0_offset),
            (const float *)((const uint8_t *)kv1buf->ptr + kv1_offset),
            (const float *)((const uint8_t *)kv2buf->ptr + kv2_offset),
            (const float *)((const uint8_t *)kv3buf->ptr + kv3_offset),
            (const float *)((const uint8_t *)rbuf->ptr + raw_kv_ring_offset),
            (const float *)((const uint8_t *)cbuf->ptr + comp_kv_offset),
            (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            (float *)((uint8_t *)o2buf->ptr + out2_offset),
            (float *)((uint8_t *)o3buf->ptr + out3_offset),
            raw_slots,
            raw_head0,
            raw_count0,
            comp_count0,
            comp_count1,
            comp_count2,
            comp_count3);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_attention_raw_comp_ring_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        float eps) {
    if (!cuda_runtime || !q || !raw_kv_ring || !comp_kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || raw_slots == 0 || raw_head >= raw_slots ||
        raw_count == 0 || raw_count > raw_slots || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t total = raw_count + comp_count;
    if (total == 0 || total > 16384u) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)raw_kv_ring;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)comp_kv;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    if (heads == 64u && head_dim == 512u &&
        !axiom_cuda_env_enabled("AXIOM_DS4_RAWCOMP_GENERIC")) {
        const int grid = (int)((heads + 7u) / 8u);
        axiom_deepseek_attention_raw_comp_ring_heads8_kernel<<<grid, 256>>>(
                (const float *)((const uint8_t *)qbuf->ptr + q_offset),
                (const float *)((const uint8_t *)rbuf->ptr + raw_kv_ring_offset),
                (const float *)((const uint8_t *)cbuf->ptr + comp_kv_offset),
                (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                raw_slots,
                raw_head,
                raw_count,
                comp_count);
    } else {
        const size_t shared_bytes = 256u * sizeof(double) + (size_t)2u * total * sizeof(float);
        err = cudaFuncSetAttribute(
                axiom_deepseek_attention_raw_comp_ring_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)shared_bytes);
        if (err != cudaSuccess) return AXIOM_ERR_CUDA;
        axiom_deepseek_attention_raw_comp_ring_kernel<<<heads, 256, shared_bytes>>>(
                (const float *)((const uint8_t *)qbuf->ptr + q_offset),
                (const float *)((const uint8_t *)rbuf->ptr + raw_kv_ring_offset),
                (const float *)((const uint8_t *)cbuf->ptr + comp_kv_offset),
                (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                head_dim,
                raw_slots,
                raw_head,
                raw_count,
                comp_count,
                eps);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_csa_indexer_topk_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *index_weights,
        uint64_t index_weights_offset,
        const void *index_comp,
        uint64_t index_comp_offset,
        void *selected,
        uint64_t selected_offset,
        uint32_t comp_count,
        uint32_t topk) {
    if (!cuda_runtime || !q || !index_weights || !index_comp || !selected ||
        comp_count == 0 || topk == 0 || topk > 512u || topk > comp_count) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)index_weights;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)index_comp;
    axiom_cuda_buffer *sbuf = (axiom_cuda_buffer *)selected;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    float *scores = NULL;
    err = cudaMalloc(&scores, (size_t)comp_count * sizeof(float));
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_csa_indexer_score_kernel<<<comp_count, 128>>>(
            scores,
            (const float *)((const uint8_t *)qbuf->ptr + q_offset),
            (const float *)((const uint8_t *)wbuf->ptr + index_weights_offset),
            (const float *)((const uint8_t *)cbuf->ptr + index_comp_offset),
            comp_count);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        axiom_deepseek_csa_indexer_topk_kernel<<<1, 1>>>(
                (uint32_t *)((uint8_t *)sbuf->ptr + selected_offset),
                scores,
                comp_count,
                topk);
        err = cudaGetLastError();
    }
    const cudaError_t free_err = cudaFree(scores);
    if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_csa_indexer_topk_scratch_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *index_weights,
        uint64_t index_weights_offset,
        const void *index_comp,
        uint64_t index_comp_offset,
        void *scores,
        uint64_t scores_offset,
        void *selected,
        uint64_t selected_offset,
        uint32_t comp_count,
        uint32_t topk) {
    if (!cuda_runtime || !q || !index_weights || !index_comp || !scores || !selected ||
        comp_count == 0 || topk == 0 || topk > 512u || topk > comp_count) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)index_weights;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)index_comp;
    axiom_cuda_buffer *scorebuf = (axiom_cuda_buffer *)scores;
    axiom_cuda_buffer *sbuf = (axiom_cuda_buffer *)selected;
    if (scores_offset > scorebuf->bytes ||
        (uint64_t)comp_count * sizeof(float) > scorebuf->bytes - scores_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    float *score_ptr = (float *)((uint8_t *)scorebuf->ptr + scores_offset);
    axiom_deepseek_csa_indexer_score_kernel<<<comp_count, 128>>>(
            score_ptr,
            (const float *)((const uint8_t *)qbuf->ptr + q_offset),
            (const float *)((const uint8_t *)wbuf->ptr + index_weights_offset),
            (const float *)((const uint8_t *)cbuf->ptr + index_comp_offset),
            comp_count);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        axiom_deepseek_csa_indexer_topk_kernel<<<1, 1>>>(
                (uint32_t *)((uint8_t *)sbuf->ptr + selected_offset),
                score_ptr,
                comp_count,
                topk);
        err = cudaGetLastError();
    }
    return axiom_cuda_finish_after_launch(err);
}

static int axiom_cuda_deepseek_attention_raw_selected_comp_ring_f32_impl(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *selected,
        uint64_t selected_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t selected_count,
        float eps,
        int trusted_selected) {
    if (!cuda_runtime || !q || !raw_kv_ring || !comp_kv || !selected || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || raw_slots == 0 || raw_head >= raw_slots ||
        raw_count == 0 || raw_count > raw_slots || selected_count > 512u ||
        selected_count > comp_count || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t total = raw_count + selected_count;
    if (total == 0 || total > 16384u) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qbuf = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)raw_kv_ring;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)comp_kv;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)selected;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)attn_sink;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    uint32_t *bad_selected = NULL;
    if (!trusted_selected) {
        err = cudaMalloc(&bad_selected, sizeof(uint32_t));
        if (err != cudaSuccess) return AXIOM_ERR_CUDA;
        err = cudaMemset(bad_selected, 0, sizeof(uint32_t));
        if (err != cudaSuccess) {
            cudaError_t free_err = cudaFree(bad_selected);
            (void)free_err;
            return AXIOM_ERR_CUDA;
        }
    }
    if (heads == 64u && head_dim == 512u &&
        !axiom_cuda_env_enabled("AXIOM_DS4_RAWCOMP_GENERIC")) {
        const int grid = (int)((heads + 7u) / 8u);
        axiom_deepseek_attention_raw_selected_comp_ring_heads8_kernel<<<grid, 256>>>(
                (const float *)((const uint8_t *)qbuf->ptr + q_offset),
                (const float *)((const uint8_t *)rbuf->ptr + raw_kv_ring_offset),
                (const float *)((const uint8_t *)cbuf->ptr + comp_kv_offset),
                (const uint32_t *)((const uint8_t *)ibuf->ptr + selected_offset),
                (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                raw_slots,
                raw_head,
                raw_count,
                comp_count,
                selected_count,
                bad_selected);
    } else {
        const size_t shared_bytes = 256u * sizeof(double) + (size_t)2u * total * sizeof(float);
        err = cudaFuncSetAttribute(
                axiom_deepseek_attention_raw_selected_comp_ring_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)shared_bytes);
        if (err == cudaSuccess) {
            axiom_deepseek_attention_raw_selected_comp_ring_kernel<<<heads, 256, shared_bytes>>>(
                    (const float *)((const uint8_t *)qbuf->ptr + q_offset),
                    (const float *)((const uint8_t *)rbuf->ptr + raw_kv_ring_offset),
                    (const float *)((const uint8_t *)cbuf->ptr + comp_kv_offset),
                    (const uint32_t *)((const uint8_t *)ibuf->ptr + selected_offset),
                    (const float *)((const uint8_t *)sbuf->ptr + attn_sink_offset),
                    (float *)((uint8_t *)obuf->ptr + out_offset),
                    head_dim,
                    raw_slots,
                    raw_head,
                    raw_count,
                    comp_count,
                    selected_count,
                    bad_selected);
        }
    }
    if (err == cudaSuccess) err = cudaGetLastError();
    if (!trusted_selected) {
        uint32_t bad_host = 0;
        if (err == cudaSuccess) {
            err = cudaMemcpy(&bad_host, bad_selected, sizeof(bad_host), cudaMemcpyDeviceToHost);
        }
        const cudaError_t free_err = cudaFree(bad_selected);
        if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
        if (err != cudaSuccess) return axiom_cuda_status(err);
        if (bad_host) return AXIOM_ERR_INVALID_ARGUMENT;
        return axiom_cuda_finish_after_launch(cudaSuccess);
    }
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_attention_raw_selected_comp_ring_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *selected,
        uint64_t selected_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t selected_count,
        float eps) {
    return axiom_cuda_deepseek_attention_raw_selected_comp_ring_f32_impl(
            cuda_runtime, q, q_offset, raw_kv_ring, raw_kv_ring_offset,
            comp_kv, comp_kv_offset, selected, selected_offset, attn_sink,
            attn_sink_offset, out, out_offset, heads, head_dim, raw_slots,
            raw_head, raw_count, comp_count, selected_count, eps, 0);
}

extern "C" int axiom_cuda_deepseek_attention_raw_selected_comp_ring_trusted_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *selected,
        uint64_t selected_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t selected_count,
        float eps) {
    return axiom_cuda_deepseek_attention_raw_selected_comp_ring_f32_impl(
            cuda_runtime, q, q_offset, raw_kv_ring, raw_kv_ring_offset,
            comp_kv, comp_kv_offset, selected, selected_offset, attn_sink,
            attn_sink_offset, out, out_offset, heads, head_dim, raw_slots,
            raw_head, raw_count, comp_count, selected_count, eps, 1);
}

__global__ static void axiom_deepseek_csa_cold_window_kernel(
        const float *kv,
        const float *gate,
        const float *bias,
        float *out,
        uint32_t ratio,
        uint32_t head_dim) {
    const uint32_t d = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    float m = -3.4028234663852886e+38F;
    for (uint32_t i = 0; i < ratio; ++i) {
        const uint64_t off = ((uint64_t)i * 2u * head_dim) + head_dim + d;
        const float v = gate[off] + bias[off];
        if (v > m) m = v;
    }
    float sum = 0.0f;
    double acc = 0.0;
    for (uint32_t i = 0; i < ratio; ++i) {
        const uint64_t off = ((uint64_t)i * 2u * head_dim) + head_dim + d;
        const float w = expf((gate[off] + bias[off]) - m);
        sum += w;
        acc += (double)w * (double)kv[off];
    }
    out[d] = sum > 0.0f ? (float)(acc / (double)sum) : 0.0f;
}

__device__ static uint32_t axiom_dev_ring_slot(uint32_t oldest, uint32_t i, uint32_t ratio) {
    const uint32_t slot = oldest + i;
    if ((ratio & (ratio - 1u)) == 0u) return slot & (ratio - 1u);
    return slot >= ratio ? slot - ratio : slot;
}

extern "C" int axiom_cuda_deepseek_csa_cold_window_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim) {
    if (!cuda_runtime || !kv || !gate || !bias || !out || ratio == 0 || head_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *kvbuf = (const axiom_cuda_buffer *)kv;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)bias;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)((head_dim + block - 1u) / block);
    axiom_deepseek_csa_cold_window_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)kvbuf->ptr + kv_offset),
            (const float *)((const uint8_t *)gbuf->ptr + gate_offset),
            (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            ratio,
            head_dim);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_csa_ring_window_kernel(
        const float *kv,
        const float *gate,
        const float *bias,
        float *out,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head,
        uint32_t ring_count) {
    const uint32_t d = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    if (ring_count == 0) {
        out[d] = 0.0f;
        return;
    }
    const uint32_t oldest = ring_count == ratio ? ring_head : 0u;
    float m = -3.4028234663852886e+38F;
    for (uint32_t i = 0; i < ring_count; ++i) {
        const uint32_t slot = axiom_dev_ring_slot(oldest, i, ratio);
        const uint64_t slot_off = ((uint64_t)slot * 2u * head_dim) + head_dim + d;
        const uint64_t bias_off = ((uint64_t)i * 2u * head_dim) + head_dim + d;
        const float v = gate[slot_off] + bias[bias_off];
        if (v > m) m = v;
    }
    float sum = 0.0f;
    double acc = 0.0;
    for (uint32_t i = 0; i < ring_count; ++i) {
        const uint32_t slot = axiom_dev_ring_slot(oldest, i, ratio);
        const uint64_t slot_off = ((uint64_t)slot * 2u * head_dim) + head_dim + d;
        const uint64_t bias_off = ((uint64_t)i * 2u * head_dim) + head_dim + d;
        const float w = expf((gate[slot_off] + bias[bias_off]) - m);
        sum += w;
        acc += (double)w * (double)kv[slot_off];
    }
    out[d] = sum > 0.0f ? (float)(acc / (double)sum) : 0.0f;
}

extern "C" int axiom_cuda_deepseek_csa_ring_window_count_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head,
        uint32_t ring_count) {
    if (!cuda_runtime || !kv || !gate || !bias || !out ||
        ratio == 0 || head_dim == 0 || ring_head >= ratio || ring_count > ratio) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *kvbuf = (const axiom_cuda_buffer *)kv;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)bias;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)((head_dim + block - 1u) / block);
    axiom_deepseek_csa_ring_window_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)kvbuf->ptr + kv_offset),
            (const float *)((const uint8_t *)gbuf->ptr + gate_offset),
            (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            ratio,
            head_dim,
            ring_head,
            ring_count);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_csa_ring_window_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head) {
    return axiom_cuda_deepseek_csa_ring_window_count_f32_device(
            cuda_runtime, kv, kv_offset, gate, gate_offset, bias, bias_offset,
            out, out_offset, ratio, head_dim, ring_head, ratio);
}

__global__ static void axiom_deepseek_csa_state_pool_kernel(
        const float *kv,
        const float *gate,
        float *out,
        uint32_t head_dim,
        uint32_t use_previous) {
    const uint32_t d = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    const uint32_t width = 2u * head_dim;
    float m = -3.4028234663852886e+38F;
    if (use_previous) {
        for (uint32_t r = 0; r < 4u; ++r) {
            const float v = gate[(uint64_t)r * width + d];
            if (v > m) m = v;
        }
    }
    for (uint32_t r = 0; r < 4u; ++r) {
        const float v = gate[(uint64_t)(4u + r) * width + head_dim + d];
        if (v > m) m = v;
    }
    float sum = 0.0f;
    double acc = 0.0;
    if (use_previous) {
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint64_t off = (uint64_t)r * width + d;
            const float w = expf(gate[off] - m);
            sum += w;
            acc += (double)w * (double)kv[off];
        }
    }
    for (uint32_t r = 0; r < 4u; ++r) {
        const uint64_t off = (uint64_t)(4u + r) * width + head_dim + d;
        const float w = expf(gate[off] - m);
        sum += w;
        acc += (double)w * (double)kv[off];
    }
    out[d] = sum > 0.0f ? (float)(acc / (double)sum) : 0.0f;
}

extern "C" int axiom_cuda_deepseek_csa_state_pool_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        void *out,
        uint64_t out_offset,
        uint32_t head_dim,
        uint32_t use_previous) {
    if (!cuda_runtime || !kv || !gate || !out || head_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *kvbuf = (const axiom_cuda_buffer *)kv;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)((head_dim + block - 1u) / block);
    axiom_deepseek_csa_state_pool_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)kvbuf->ptr + kv_offset),
            (const float *)((const uint8_t *)gbuf->ptr + gate_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            head_dim,
            use_previous ? 1u : 0u);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_hca_window_kernel(
        const float *kv,
        const float *gate,
        const float *bias,
        float *out,
        uint32_t ratio,
        uint32_t head_dim) {
    const uint32_t d = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    float m = -3.4028234663852886e+38F;
    for (uint32_t i = 0; i < ratio; ++i) {
        const uint64_t off = (uint64_t)i * head_dim + d;
        const float v = gate[off] + bias[off];
        if (v > m) m = v;
    }
    float sum = 0.0f;
    double acc = 0.0;
    for (uint32_t i = 0; i < ratio; ++i) {
        const uint64_t off = (uint64_t)i * head_dim + d;
        const float w = expf((gate[off] + bias[off]) - m);
        sum += w;
        acc += (double)w * (double)kv[off];
    }
    out[d] = sum > 0.0f ? (float)(acc / (double)sum) : 0.0f;
}

extern "C" int axiom_cuda_deepseek_hca_window_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim) {
    if (!cuda_runtime || !kv || !gate || !bias || !out || ratio == 0 || head_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *kvbuf = (const axiom_cuda_buffer *)kv;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)bias;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)((head_dim + block - 1u) / block);
    axiom_deepseek_hca_window_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)kvbuf->ptr + kv_offset),
            (const float *)((const uint8_t *)gbuf->ptr + gate_offset),
            (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            ratio,
            head_dim);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_hca_ring_window_kernel(
        const float *kv,
        const float *gate,
        const float *bias,
        float *out,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head,
        uint32_t ring_count) {
    const uint32_t d = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    if (ring_count == 0) {
        out[d] = 0.0f;
        return;
    }
    const uint32_t oldest = ring_count == ratio ? ring_head : 0u;
    float m = -3.4028234663852886e+38F;
    for (uint32_t i = 0; i < ring_count; ++i) {
        const uint32_t slot = axiom_dev_ring_slot(oldest, i, ratio);
        const uint64_t slot_off = (uint64_t)slot * head_dim + d;
        const uint64_t bias_off = (uint64_t)i * head_dim + d;
        const float v = gate[slot_off] + bias[bias_off];
        if (v > m) m = v;
    }
    float sum = 0.0f;
    double acc = 0.0;
    for (uint32_t i = 0; i < ring_count; ++i) {
        const uint32_t slot = axiom_dev_ring_slot(oldest, i, ratio);
        const uint64_t slot_off = (uint64_t)slot * head_dim + d;
        const uint64_t bias_off = (uint64_t)i * head_dim + d;
        const float w = expf((gate[slot_off] + bias[bias_off]) - m);
        sum += w;
        acc += (double)w * (double)kv[slot_off];
    }
    out[d] = sum > 0.0f ? (float)(acc / (double)sum) : 0.0f;
}

extern "C" int axiom_cuda_deepseek_hca_ring_window_count_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head,
        uint32_t ring_count) {
    if (!cuda_runtime || !kv || !gate || !bias || !out ||
        ratio == 0 || head_dim == 0 || ring_head >= ratio || ring_count > ratio) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *kvbuf = (const axiom_cuda_buffer *)kv;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)bias;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)((head_dim + block - 1u) / block);
    axiom_deepseek_hca_ring_window_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)kvbuf->ptr + kv_offset),
            (const float *)((const uint8_t *)gbuf->ptr + gate_offset),
            (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            ratio,
            head_dim,
            ring_head,
            ring_count);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_hca_ring_window_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head) {
    return axiom_cuda_deepseek_hca_ring_window_count_f32_device(
            cuda_runtime, kv, kv_offset, gate, gate_offset, bias, bias_offset,
            out, out_offset, ratio, head_dim, ring_head, ratio);
}

__device__ static float axiom_dev_sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ static void axiom_deepseek_hc_pre_params_kernel(
        const uint16_t *fn,
        const float *scale,
        const float *base,
        const float *streams,
        float *out,
        float *post,
        float *comb,
        uint32_t hidden,
        float eps) {
    const uint32_t hc = 4u;
    const uint32_t mix_count = 24u;
    const uint32_t flat = hc * hidden;
    const uint32_t tid = threadIdx.x;
    __shared__ float ss_parts[256];
    __shared__ float dot_parts[24][256];
    __shared__ float pre_vals[4];
    float ss = 0.0f;
    float dots[24];
    for (uint32_t row = 0; row < mix_count; ++row) dots[row] = 0.0f;
    for (uint32_t i = tid; i < flat; i += blockDim.x) {
        const float x = streams[i];
        ss += x * x;
        for (uint32_t row = 0; row < mix_count; ++row) {
            dots[row] += axiom_dev_f16_to_f32(fn[(uint64_t)row * flat + i]) * x;
        }
    }
    ss_parts[tid] = ss;
    for (uint32_t row = 0; row < mix_count; ++row) dot_parts[row][tid] = dots[row];
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            ss_parts[tid] += ss_parts[tid + stride];
            for (uint32_t row = 0; row < mix_count; ++row) {
                dot_parts[row][tid] += dot_parts[row][tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0u) {
        const float inv = rsqrtf(ss_parts[0] / (float)flat + eps);
        float mixes[24];
        for (uint32_t row = 0; row < mix_count; ++row) {
            mixes[row] = dot_parts[row][0] * inv;
        }
        for (uint32_t i = 0; i < hc; ++i) {
            pre_vals[i] = axiom_dev_sigmoid_f32(mixes[i] * scale[0] + base[i]) + eps;
            post[i] = 2.0f * axiom_dev_sigmoid_f32(mixes[i + hc] * scale[1] + base[i + hc]);
        }
        for (uint32_t r = 0; r < hc; ++r) {
            float row_max = -3.4028234663852886e+38F;
            for (uint32_t c = 0; c < hc; ++c) {
                const uint32_t idx = hc * 2u + r * hc + c;
                const float v = mixes[idx] * scale[2] + base[idx];
                comb[r * hc + c] = v;
                row_max = fmaxf(row_max, v);
            }
            float row_sum = 0.0f;
            for (uint32_t c = 0; c < hc; ++c) {
                const float e = expf(comb[r * hc + c] - row_max);
                comb[r * hc + c] = e;
                row_sum += e;
            }
            for (uint32_t c = 0; c < hc; ++c) comb[r * hc + c] = comb[r * hc + c] / row_sum + eps;
        }
        for (uint32_t iter = 0; iter < 20u; ++iter) {
            if (iter > 0u) {
                for (uint32_t r = 0; r < hc; ++r) {
                    float row_sum = 0.0f;
                    for (uint32_t c = 0; c < hc; ++c) row_sum += comb[r * hc + c];
                    for (uint32_t c = 0; c < hc; ++c) comb[r * hc + c] /= row_sum + eps;
                }
            }
            for (uint32_t c = 0; c < hc; ++c) {
                float col_sum = 0.0f;
                for (uint32_t r = 0; r < hc; ++r) col_sum += comb[r * hc + c];
                for (uint32_t r = 0; r < hc; ++r) comb[r * hc + c] /= col_sum + eps;
            }
        }
    }
    __syncthreads();
    for (uint32_t d = tid; d < hidden; d += blockDim.x) {
        double acc = 0.0;
        for (uint32_t h = 0; h < hc; ++h) {
            acc += (double)pre_vals[h] * (double)streams[(uint64_t)h * hidden + d];
        }
        out[d] = (float)acc;
    }
}

__global__ static void axiom_deepseek_hc_pre_mix_kernel(
        const float *streams,
        const float *pre,
        float *out,
        uint32_t hidden) {
    const uint32_t d = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= hidden) return;
    double acc = 0.0;
    for (uint32_t h = 0; h < 4u; ++h) {
        acc += (double)pre[h] * (double)streams[(uint64_t)h * hidden + d];
    }
    out[d] = (float)acc;
}

__global__ static void axiom_deepseek_hc_pre_scratch_kernel(
        const uint16_t *fn,
        const float *streams,
        float *scratch,
        uint32_t hidden) {
    const uint32_t flat = 4u * hidden;
    const uint32_t row = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ float parts[256];
    float acc = 0.0f;
    if (row == 0u) {
        for (uint32_t i = tid; i < flat; i += (uint32_t)blockDim.x) {
            const float x = streams[i];
            acc += x * x;
        }
    } else {
        const uint32_t mix_row = row - 1u;
        const uint16_t *fn_row = fn + (uint64_t)mix_row * flat;
        for (uint32_t i = tid; i < flat; i += (uint32_t)blockDim.x) {
            acc += axiom_dev_f16_to_f32(fn_row[i]) * streams[i];
        }
    }
    parts[tid] = acc;
    __syncthreads();
    for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) parts[tid] += parts[tid + stride];
        __syncthreads();
    }
    if (tid == 0u) scratch[row] = parts[0];
}

__global__ static void axiom_deepseek_hc_pre_scratch_fn_f32_kernel(
        const float *fn,
        const float *streams,
        float *scratch,
        uint32_t hidden) {
    const uint32_t flat = 4u * hidden;
    const uint32_t row = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ float parts[256];
    float acc = 0.0f;
    if (row == 0u) {
        for (uint32_t i = tid; i < flat; i += (uint32_t)blockDim.x) {
            const float x = streams[i];
            acc += x * x;
        }
    } else {
        const uint32_t mix_row = row - 1u;
        const float *fn_row = fn + (uint64_t)mix_row * flat;
        for (uint32_t i = tid; i < flat; i += (uint32_t)blockDim.x) {
            acc += fn_row[i] * streams[i];
        }
    }
    parts[tid] = acc;
    __syncthreads();
    for (uint32_t stride = (uint32_t)blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) parts[tid] += parts[tid + stride];
        __syncthreads();
    }
    if (tid == 0u) scratch[row] = parts[0];
}

__global__ static void axiom_deepseek_hc_pre_finalize_mix_kernel(
        const float *scale,
        const float *base,
        const float *streams,
        float *scratch_out,
        float *post,
        float *comb,
        uint32_t hidden,
        float eps) {
    const uint32_t hc = 4u;
    const uint32_t flat = hc * hidden;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ float pre_vals[4];
    if (tid == 0u) {
        const float inv = rsqrtf(scratch_out[0] / (float)flat + eps);
        float mixes[24];
        for (uint32_t row = 0; row < 24u; ++row) {
            mixes[row] = scratch_out[row + 1u] * inv;
        }
        for (uint32_t i = 0; i < hc; ++i) {
            pre_vals[i] = axiom_dev_sigmoid_f32(mixes[i] * scale[0] + base[i]) + eps;
            post[i] = 2.0f * axiom_dev_sigmoid_f32(mixes[i + hc] * scale[1] + base[i + hc]);
        }
        for (uint32_t r = 0; r < hc; ++r) {
            float row_max = -3.4028234663852886e+38F;
            for (uint32_t c = 0; c < hc; ++c) {
                const uint32_t idx = hc * 2u + r * hc + c;
                const float v = mixes[idx] * scale[2] + base[idx];
                comb[r * hc + c] = v;
                row_max = fmaxf(row_max, v);
            }
            float row_sum = 0.0f;
            for (uint32_t c = 0; c < hc; ++c) {
                const float e = expf(comb[r * hc + c] - row_max);
                comb[r * hc + c] = e;
                row_sum += e;
            }
            for (uint32_t c = 0; c < hc; ++c) comb[r * hc + c] = comb[r * hc + c] / row_sum + eps;
        }
        for (uint32_t iter = 0; iter < 20u; ++iter) {
            if (iter > 0u) {
                for (uint32_t r = 0; r < hc; ++r) {
                    float row_sum = 0.0f;
                    for (uint32_t c = 0; c < hc; ++c) row_sum += comb[r * hc + c];
                    for (uint32_t c = 0; c < hc; ++c) comb[r * hc + c] /= row_sum + eps;
                }
            }
            for (uint32_t c = 0; c < hc; ++c) {
                float col_sum = 0.0f;
                for (uint32_t r = 0; r < hc; ++r) col_sum += comb[r * hc + c];
                for (uint32_t r = 0; r < hc; ++r) comb[r * hc + c] /= col_sum + eps;
            }
        }
    }
    __syncthreads();
    for (uint32_t d = tid; d < hidden; d += (uint32_t)blockDim.x) {
        double acc = 0.0;
        for (uint32_t h = 0; h < hc; ++h) {
            acc += (double)pre_vals[h] * (double)streams[(uint64_t)h * hidden + d];
        }
        scratch_out[d] = (float)acc;
    }
}

extern "C" int axiom_cuda_deepseek_hc_pre_f32_device(
        void *cuda_runtime,
        const void *fn_f16,
        uint64_t fn_offset,
        const void *scale_f32,
        uint64_t scale_offset,
        const void *base_f32,
        uint64_t base_offset,
        const void *streams,
        uint64_t streams_offset,
        void *out,
        uint64_t out_offset,
        void *post,
        uint64_t post_offset,
        void *comb,
        uint64_t comb_offset,
        uint32_t hidden,
        float eps) {
    if (!cuda_runtime || !fn_f16 || !scale_f32 || !base_f32 || !streams ||
        !out || !post || !comb || hidden == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *fbuf = (const axiom_cuda_buffer *)fn_f16;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)scale_f32;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)base_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)streams;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    axiom_cuda_buffer *pbuf = (axiom_cuda_buffer *)post;
    axiom_cuda_buffer *cbuf = (axiom_cuda_buffer *)comb;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    if (axiom_cuda_env_enabled("AXIOM_DS4_HC_PRE_MONO")) {
        axiom_deepseek_hc_pre_params_kernel<<<1, 256>>>(
                (const uint16_t *)((const uint8_t *)fbuf->ptr + fn_offset),
                (const float *)((const uint8_t *)sbuf->ptr + scale_offset),
                (const float *)((const uint8_t *)bbuf->ptr + base_offset),
                (const float *)((const uint8_t *)ibuf->ptr + streams_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                (float *)((uint8_t *)pbuf->ptr + post_offset),
                (float *)((uint8_t *)cbuf->ptr + comb_offset),
                hidden,
                eps);
    } else {
        axiom_deepseek_hc_pre_scratch_kernel<<<25, 256>>>(
                (const uint16_t *)((const uint8_t *)fbuf->ptr + fn_offset),
                (const float *)((const uint8_t *)ibuf->ptr + streams_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                hidden);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_deepseek_hc_pre_finalize_mix_kernel<<<1, 256>>>(
                (const float *)((const uint8_t *)sbuf->ptr + scale_offset),
                (const float *)((const uint8_t *)bbuf->ptr + base_offset),
                (const float *)((const uint8_t *)ibuf->ptr + streams_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                (float *)((uint8_t *)pbuf->ptr + post_offset),
                (float *)((uint8_t *)cbuf->ptr + comb_offset),
                hidden,
                eps);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_hc_pre_fn_f32_device(
        void *cuda_runtime,
        const void *fn_f32,
        uint64_t fn_offset,
        const void *scale_f32,
        uint64_t scale_offset,
        const void *base_f32,
        uint64_t base_offset,
        const void *streams,
        uint64_t streams_offset,
        void *out,
        uint64_t out_offset,
        void *post,
        uint64_t post_offset,
        void *comb,
        uint64_t comb_offset,
        uint32_t hidden,
        float eps) {
    if (!cuda_runtime || !fn_f32 || !scale_f32 || !base_f32 || !streams ||
        !out || !post || !comb || hidden == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *fbuf = (const axiom_cuda_buffer *)fn_f32;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)scale_f32;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)base_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)streams;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    axiom_cuda_buffer *pbuf = (axiom_cuda_buffer *)post;
    axiom_cuda_buffer *cbuf = (axiom_cuda_buffer *)comb;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    axiom_deepseek_hc_pre_scratch_fn_f32_kernel<<<25, 256>>>(
            (const float *)((const uint8_t *)fbuf->ptr + fn_offset),
            (const float *)((const uint8_t *)ibuf->ptr + streams_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_deepseek_hc_pre_finalize_mix_kernel<<<1, 256>>>(
            (const float *)((const uint8_t *)sbuf->ptr + scale_offset),
            (const float *)((const uint8_t *)bbuf->ptr + base_offset),
            (const float *)((const uint8_t *)ibuf->ptr + streams_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            (float *)((uint8_t *)pbuf->ptr + post_offset),
            (float *)((uint8_t *)cbuf->ptr + comb_offset),
            hidden,
            eps);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_output_hc_params_kernel(
        const uint16_t *fn,
        const float *scale,
        const float *base,
        const float *streams,
        float *pre,
        uint32_t hidden,
        float eps) {
    const uint32_t hc = 4u;
    const uint32_t flat = hc * hidden;
    const uint32_t tid = threadIdx.x;
    __shared__ float ss_parts[256];
    __shared__ float dot_parts[4][256];
    float ss = 0.0f;
    float dots[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t i = tid; i < flat; i += blockDim.x) {
        const float x = streams[i];
        ss += x * x;
        for (uint32_t row = 0; row < hc; ++row) {
            dots[row] += axiom_dev_f16_to_f32(fn[(uint64_t)row * flat + i]) * x;
        }
    }
    ss_parts[tid] = ss;
    for (uint32_t row = 0; row < hc; ++row) dot_parts[row][tid] = dots[row];
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            ss_parts[tid] += ss_parts[tid + stride];
            for (uint32_t row = 0; row < hc; ++row) {
                dot_parts[row][tid] += dot_parts[row][tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid != 0) return;
    const float inv = rsqrtf(ss_parts[0] / (float)flat + eps);
    for (uint32_t row = 0; row < hc; ++row) {
        const float mix = dot_parts[row][0] * inv;
        pre[row] = axiom_dev_sigmoid_f32(mix * scale[0] + base[row]) + eps;
    }
}

__global__ static void axiom_deepseek_output_hc_params_fn_f32_kernel(
        const float *fn,
        const float *scale,
        const float *base,
        const float *streams,
        float *pre,
        uint32_t hidden,
        float eps) {
    const uint32_t hc = 4u;
    const uint32_t flat = hc * hidden;
    const uint32_t tid = threadIdx.x;
    __shared__ float ss_parts[256];
    __shared__ float dot_parts[4][256];
    float ss = 0.0f;
    float dots[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t i = tid; i < flat; i += blockDim.x) {
        const float x = streams[i];
        ss += x * x;
        for (uint32_t row = 0; row < hc; ++row) {
            dots[row] += fn[(uint64_t)row * flat + i] * x;
        }
    }
    ss_parts[tid] = ss;
    for (uint32_t row = 0; row < hc; ++row) dot_parts[row][tid] = dots[row];
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            ss_parts[tid] += ss_parts[tid + stride];
            for (uint32_t row = 0; row < hc; ++row) {
                dot_parts[row][tid] += dot_parts[row][tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid != 0) return;
    const float inv = rsqrtf(ss_parts[0] / (float)flat + eps);
    for (uint32_t row = 0; row < hc; ++row) {
        const float mix = dot_parts[row][0] * inv;
        pre[row] = axiom_dev_sigmoid_f32(mix * scale[0] + base[row]) + eps;
    }
}

extern "C" int axiom_cuda_deepseek_output_hc_f32_device(
        void *cuda_runtime,
        const void *fn_f16,
        uint64_t fn_offset,
        const void *scale_f32,
        uint64_t scale_offset,
        const void *base_f32,
        uint64_t base_offset,
        const void *streams,
        uint64_t streams_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        float eps) {
    if (!cuda_runtime || !fn_f16 || !scale_f32 || !base_f32 || !streams ||
        !out || hidden == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *fbuf = (const axiom_cuda_buffer *)fn_f16;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)scale_f32;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)base_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)streams;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    if (!runtime->output_hc_pre) {
        err = cudaMalloc(&runtime->output_hc_pre, 4u * sizeof(float));
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }
    float *pre = (float *)runtime->output_hc_pre;
    axiom_deepseek_output_hc_params_kernel<<<1, 256>>>(
            (const uint16_t *)((const uint8_t *)fbuf->ptr + fn_offset),
            (const float *)((const uint8_t *)sbuf->ptr + scale_offset),
            (const float *)((const uint8_t *)bbuf->ptr + base_offset),
            (const float *)((const uint8_t *)ibuf->ptr + streams_offset),
            pre,
            hidden,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    const int block = 128;
    const int grid = (int)((hidden + block - 1u) / block);
    axiom_deepseek_hc_pre_mix_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)ibuf->ptr + streams_offset),
            pre,
            (float *)((uint8_t *)obuf->ptr + out_offset),
            hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_output_hc_fn_f32_device(
        void *cuda_runtime,
        const void *fn_f32,
        uint64_t fn_offset,
        const void *scale_f32,
        uint64_t scale_offset,
        const void *base_f32,
        uint64_t base_offset,
        const void *streams,
        uint64_t streams_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        float eps) {
    if (!cuda_runtime || !fn_f32 || !scale_f32 || !base_f32 || !streams ||
        !out || hidden == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *fbuf = (const axiom_cuda_buffer *)fn_f32;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)scale_f32;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)base_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)streams;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    if (!runtime->output_hc_pre) {
        err = cudaMalloc(&runtime->output_hc_pre, 4u * sizeof(float));
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }
    float *pre = (float *)runtime->output_hc_pre;
    axiom_deepseek_output_hc_params_fn_f32_kernel<<<1, 256>>>(
            (const float *)((const uint8_t *)fbuf->ptr + fn_offset),
            (const float *)((const uint8_t *)sbuf->ptr + scale_offset),
            (const float *)((const uint8_t *)bbuf->ptr + base_offset),
            (const float *)((const uint8_t *)ibuf->ptr + streams_offset),
            pre,
            hidden,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    const int block = 128;
    const int grid = (int)((hidden + block - 1u) / block);
    axiom_deepseek_hc_pre_mix_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)ibuf->ptr + streams_offset),
            pre,
            (float *)((uint8_t *)obuf->ptr + out_offset),
            hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_hc_post_kernel(
        const float *x,
        const float *residual,
        const float *post,
        const float *comb,
        float *out,
        uint32_t hidden) {
    const uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = 4u * hidden;
    if (idx >= total) return;
    const uint32_t dst = idx / hidden;
    const uint32_t d = idx - dst * hidden;
    double acc = (double)post[dst] * (double)x[d];
    for (uint32_t src = 0; src < 4u; ++src) {
        acc += (double)comb[src * 4u + dst] * (double)residual[(uint64_t)src * hidden + d];
    }
    out[idx] = (float)acc;
}

extern "C" int axiom_cuda_deepseek_hc_post_f32_device(
        void *cuda_runtime,
        const void *x,
        uint64_t x_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden) {
    if (!cuda_runtime || !x || !residual || !post || !comb || !out || hidden == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *xbuf = (const axiom_cuda_buffer *)x;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)residual;
    const axiom_cuda_buffer *pbuf = (const axiom_cuda_buffer *)post;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)comb;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)(((uint64_t)4u * hidden + block - 1u) / block);
    axiom_deepseek_hc_post_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)xbuf->ptr + x_offset),
            (const float *)((const uint8_t *)rbuf->ptr + residual_offset),
            (const float *)((const uint8_t *)pbuf->ptr + post_offset),
            (const float *)((const uint8_t *)cbuf->ptr + comb_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_ffn_hc_post_kernel(
        const float *shared,
        const float *moe,
        const float *residual,
        const float *post,
        const float *comb,
        float *out,
        uint32_t hidden) {
    const uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = 4u * hidden;
    if (idx >= total) return;
    const uint32_t dst = idx / hidden;
    const uint32_t d = idx - dst * hidden;
    double acc = (double)post[dst] * ((double)shared[d] + (double)moe[d]);
    for (uint32_t src = 0; src < 4u; ++src) {
        acc += (double)comb[src * 4u + dst] * (double)residual[(uint64_t)src * hidden + d];
    }
    out[idx] = (float)acc;
}

__global__ static void axiom_deepseek_ffn_hc_post_dual_out_kernel(
        const float *shared,
        const float *moe,
        const float *residual,
        const float *post,
        const float *comb,
        float *out_a,
        float *out_b,
        uint32_t hidden) {
    const uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = 4u * hidden;
    if (idx >= total) return;
    const uint32_t dst = idx / hidden;
    const uint32_t d = idx - dst * hidden;
    double acc = (double)post[dst] * ((double)shared[d] + (double)moe[d]);
    for (uint32_t src = 0; src < 4u; ++src) {
        acc += (double)comb[src * 4u + dst] * (double)residual[(uint64_t)src * hidden + d];
    }
    const float y = (float)acc;
    out_a[idx] = y;
    out_b[idx] = y;
}

__global__ static void axiom_deepseek_ffn_hc_post2_kernel(
        const float *shared0,
        const float *moe0,
        const float *residual0,
        const float *post0,
        const float *comb0,
        const float *shared1,
        const float *moe1,
        const float *residual1,
        const float *post1,
        const float *comb1,
        float *out0,
        float *out1,
        uint32_t hidden) {
    const uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = 4u * hidden;
    if (idx >= 2u * total) return;
    const uint32_t lane = idx >= total ? 1u : 0u;
    const uint32_t local = lane ? idx - total : idx;
    const uint32_t dst = local / hidden;
    const uint32_t d = local - dst * hidden;
    const float *shared = lane ? shared1 : shared0;
    const float *moe = lane ? moe1 : moe0;
    const float *residual = lane ? residual1 : residual0;
    const float *post = lane ? post1 : post0;
    const float *comb = lane ? comb1 : comb0;
    double acc = (double)post[dst] * ((double)shared[d] + (double)moe[d]);
    for (uint32_t src = 0; src < 4u; ++src) {
        acc += (double)comb[src * 4u + dst] * (double)residual[(uint64_t)src * hidden + d];
    }
    if (lane) out1[local] = (float)acc;
    else out0[local] = (float)acc;
}

extern "C" int axiom_cuda_deepseek_ffn_hc_post_f32_device(
        void *cuda_runtime,
        const void *shared,
        uint64_t shared_offset,
        const void *moe,
        uint64_t moe_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden) {
    if (!cuda_runtime || !shared || !moe || !residual || !post || !comb || !out || hidden == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)shared;
    const axiom_cuda_buffer *mbuf = (const axiom_cuda_buffer *)moe;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)residual;
    const axiom_cuda_buffer *pbuf = (const axiom_cuda_buffer *)post;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)comb;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)(((uint64_t)4u * hidden + block - 1u) / block);
    axiom_deepseek_ffn_hc_post_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)sbuf->ptr + shared_offset),
            (const float *)((const uint8_t *)mbuf->ptr + moe_offset),
            (const float *)((const uint8_t *)rbuf->ptr + residual_offset),
            (const float *)((const uint8_t *)pbuf->ptr + post_offset),
            (const float *)((const uint8_t *)cbuf->ptr + comb_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_ffn_hc_post_dual_out_f32_device(
        void *cuda_runtime,
        const void *shared,
        uint64_t shared_offset,
        const void *moe,
        uint64_t moe_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out_a,
        uint64_t out_a_offset,
        void *out_b,
        uint64_t out_b_offset,
        uint32_t hidden) {
    if (!cuda_runtime || !shared || !moe || !residual || !post || !comb || !out_a || !out_b || hidden == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *sbuf = (const axiom_cuda_buffer *)shared;
    const axiom_cuda_buffer *mbuf = (const axiom_cuda_buffer *)moe;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)residual;
    const axiom_cuda_buffer *pbuf = (const axiom_cuda_buffer *)post;
    const axiom_cuda_buffer *cbuf = (const axiom_cuda_buffer *)comb;
    axiom_cuda_buffer *oabuf = (axiom_cuda_buffer *)out_a;
    axiom_cuda_buffer *obbuf = (axiom_cuda_buffer *)out_b;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)(((uint64_t)4u * hidden + block - 1u) / block);
    axiom_deepseek_ffn_hc_post_dual_out_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)sbuf->ptr + shared_offset),
            (const float *)((const uint8_t *)mbuf->ptr + moe_offset),
            (const float *)((const uint8_t *)rbuf->ptr + residual_offset),
            (const float *)((const uint8_t *)pbuf->ptr + post_offset),
            (const float *)((const uint8_t *)cbuf->ptr + comb_offset),
            (float *)((uint8_t *)oabuf->ptr + out_a_offset),
            (float *)((uint8_t *)obbuf->ptr + out_b_offset),
            hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_ffn_hc_post2_f32_device(
        void *cuda_runtime,
        const void *shared0,
        uint64_t shared0_offset,
        const void *moe0,
        uint64_t moe0_offset,
        const void *residual0,
        uint64_t residual0_offset,
        const void *post0,
        uint64_t post0_offset,
        const void *comb0,
        uint64_t comb0_offset,
        const void *shared1,
        uint64_t shared1_offset,
        const void *moe1,
        uint64_t moe1_offset,
        const void *residual1,
        uint64_t residual1_offset,
        const void *post1,
        uint64_t post1_offset,
        const void *comb1,
        uint64_t comb1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t hidden) {
    if (!cuda_runtime || !shared0 || !moe0 || !residual0 || !post0 || !comb0 ||
        !shared1 || !moe1 || !residual1 || !post1 || !comb1 || !out0 || !out1 ||
        hidden == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *s0buf = (const axiom_cuda_buffer *)shared0;
    const axiom_cuda_buffer *m0buf = (const axiom_cuda_buffer *)moe0;
    const axiom_cuda_buffer *r0buf = (const axiom_cuda_buffer *)residual0;
    const axiom_cuda_buffer *p0buf = (const axiom_cuda_buffer *)post0;
    const axiom_cuda_buffer *c0buf = (const axiom_cuda_buffer *)comb0;
    const axiom_cuda_buffer *s1buf = (const axiom_cuda_buffer *)shared1;
    const axiom_cuda_buffer *m1buf = (const axiom_cuda_buffer *)moe1;
    const axiom_cuda_buffer *r1buf = (const axiom_cuda_buffer *)residual1;
    const axiom_cuda_buffer *p1buf = (const axiom_cuda_buffer *)post1;
    const axiom_cuda_buffer *c1buf = (const axiom_cuda_buffer *)comb1;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)(((uint64_t)8u * hidden + block - 1u) / block);
    axiom_deepseek_ffn_hc_post2_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)s0buf->ptr + shared0_offset),
            (const float *)((const uint8_t *)m0buf->ptr + moe0_offset),
            (const float *)((const uint8_t *)r0buf->ptr + residual0_offset),
            (const float *)((const uint8_t *)p0buf->ptr + post0_offset),
            (const float *)((const uint8_t *)c0buf->ptr + comb0_offset),
            (const float *)((const uint8_t *)s1buf->ptr + shared1_offset),
            (const float *)((const uint8_t *)m1buf->ptr + moe1_offset),
            (const float *)((const uint8_t *)r1buf->ptr + residual1_offset),
            (const float *)((const uint8_t *)p1buf->ptr + post1_offset),
            (const float *)((const uint8_t *)c1buf->ptr + comb1_offset),
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            hidden);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_moe_gate_up_kernel(
        const uint8_t *gate,
        const uint8_t *up,
        const float *input,
        float *mid,
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = experts * expert_hidden;
    if (idx >= total) return;
    const uint32_t expert = idx / expert_hidden;
    const uint32_t row = idx - expert * expert_hidden;
    const uint32_t blocks = hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    const float g0 = axiom_dev_iq2_xxs_dot_f32(gate_row, input, blocks);
    const float u0 = axiom_dev_iq2_xxs_dot_f32(up_row, input, blocks);
    const float g = fminf(g0, 10.0f);
    const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
    mid[idx] = (g / (1.0f + expf(-g))) * u;
}

__global__ static void axiom_deepseek_moe_down_kernel(
        const uint8_t *down,
        const float *mid,
        const float *router_weights,
        float *out,
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 84u;
    const uint64_t expert_bytes = (uint64_t)hidden * row_bytes;
    float acc = 0.0f;
    for (uint32_t expert = 0; expert < experts; ++expert) {
        const uint8_t *down_row = down + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const float *mid_row = mid + (uint64_t)expert * expert_hidden;
        acc += router_weights[expert] * axiom_dev_q2_k_dot_f32(down_row, mid_row, blocks);
    }
    out[row] = acc;
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_kernel(
        const uint8_t *gate,
        const uint8_t *up,
        const float *input,
        const uint32_t *indices,
        float *mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = topk * expert_hidden;
    if (idx >= total) return;
    const uint32_t slot = idx / expert_hidden;
    const uint32_t row = idx - slot * expert_hidden;
    const uint32_t expert = indices[slot];
    if (expert >= experts) {
        mid[idx] = 0.0f;
        return;
    }
    const uint32_t blocks = hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    const float g0 = axiom_dev_iq2_xxs_dot_f32(gate_row, input, blocks);
    const float u0 = axiom_dev_iq2_xxs_dot_f32(up_row, input, blocks);
    const float g = fminf(g0, 10.0f);
    const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
    mid[idx] = (g / (1.0f + expf(-g))) * u;
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_warp_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const float *__restrict__ input,
        const uint32_t *__restrict__ indices,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t idx = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t total = topk * expert_hidden;
    if (idx >= total) return;
    const uint32_t slot = idx / expert_hidden;
    const uint32_t row = idx - slot * expert_hidden;
    const uint32_t expert = indices[slot];
    if (expert >= experts) {
        if (lane == 0u) mid[idx] = 0.0f;
        return;
    }
    const uint32_t blocks = hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    float g0 = 0.0f;
    float u0 = 0.0f;
    axiom_dev_iq2_xxs_dual_dot_f32_warp(gate_row, up_row, input, blocks, &g0, &u0);
    if (lane == 0u) {
        const float g = fminf(g0, 10.0f);
        const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
        mid[idx] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_hwarp16_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const float *__restrict__ input,
        const uint32_t *__restrict__ indices,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t rows_per_block = blockDim.x >> 4u;
    const uint32_t idx = (uint32_t)blockIdx.x * rows_per_block + (threadIdx.x >> 4u);
    const uint32_t total = topk * expert_hidden;
    if (idx >= total) return;
    const uint32_t slot = idx / expert_hidden;
    const uint32_t row = idx - slot * expert_hidden;
    const uint32_t expert = indices[slot];
    if (expert >= experts) {
        if (lane16 == 0u) mid[idx] = 0.0f;
        return;
    }
    const uint32_t blocks = hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    float g0 = 0.0f;
    float u0 = 0.0f;
    axiom_dev_iq2_xxs_dual_dot_f32_hwarp16(gate_row, up_row, input, blocks, &g0, &u0);
    if (lane16 == 0u) {
        const float g = fminf(g0, 10.0f);
        const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
        mid[idx] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_hwarp16_sx_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const float *__restrict__ input,
        const uint32_t *__restrict__ indices,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    extern __shared__ float sx[];
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) sx[i] = __ldg(input + i);
    __syncthreads();

    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t rows_per_block = blockDim.x >> 4u;
    const uint32_t idx = (uint32_t)blockIdx.x * rows_per_block + (threadIdx.x >> 4u);
    const uint32_t total = topk * expert_hidden;
    if (idx >= total) return;
    const uint32_t slot = idx / expert_hidden;
    const uint32_t row = idx - slot * expert_hidden;
    const uint32_t expert = indices[slot];
    if (expert >= experts) {
        if (lane16 == 0u) mid[idx] = 0.0f;
        return;
    }
    const uint32_t blocks = hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    float g0 = 0.0f;
    float u0 = 0.0f;
    axiom_dev_iq2_xxs_dual_dot_f32_hwarp16_shared(gate_row, up_row, sx, blocks, &g0, &u0);
    if (lane16 == 0u) {
        const float g = fminf(g0, 10.0f);
        const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
        mid[idx] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_hwarp16_cpasync_sx_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const float *__restrict__ input,
        const uint32_t *__restrict__ indices,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(16) unsigned char sbytes[];
    float *sx = reinterpret_cast<float *>(sbytes);
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ axiom_cuda_block_barrier bar;
    if (threadIdx.x == 0u) init(&bar, blockDim.x);
    __syncthreads();

    const uint32_t shared_bytes = hidden * (uint32_t)sizeof(float);
    if (threadIdx.x == 0u) {
        cuda::memcpy_async(sbytes, input, cuda::aligned_size_t<16>(shared_bytes), bar);
    }
    axiom_cuda_block_barrier::arrival_token token = bar.arrive();
    bar.wait(cuda::std::move(token));
#else
    extern __shared__ float sx[];
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) sx[i] = __ldg(input + i);
    __syncthreads();
#endif

    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t rows_per_block = blockDim.x >> 4u;
    const uint32_t idx = (uint32_t)blockIdx.x * rows_per_block + (threadIdx.x >> 4u);
    const uint32_t total = topk * expert_hidden;
    if (idx >= total) return;
    const uint32_t slot = idx / expert_hidden;
    const uint32_t row = idx - slot * expert_hidden;
    const uint32_t expert = indices[slot];
    if (expert >= experts) {
        if (lane16 == 0u) mid[idx] = 0.0f;
        return;
    }
    const uint32_t blocks = hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    float g0 = 0.0f;
    float u0 = 0.0f;
    axiom_dev_iq2_xxs_dual_dot_f32_hwarp16_shared(gate_row, up_row, sx, blocks, &g0, &u0);
    if (lane16 == 0u) {
        const float g = fminf(g0, 10.0f);
        const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
        mid[idx] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8k_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t qwarp = lane >> 3u;
    const uint32_t lane8 = lane & 7u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t rows_per_block = warps_per_block * 4u;
    const uint32_t idx = (uint32_t)blockIdx.x * rows_per_block + warp * 4u + qwarp;
    const uint32_t total = topk * expert_hidden;
    if (idx >= total) return;
    const uint32_t slot = idx / expert_hidden;
    const uint32_t row = idx - slot * expert_hidden;
    const uint32_t expert = indices[slot];
    if (expert >= experts) {
        if (lane8 == 0u) mid[idx] = 0.0f;
        return;
    }
    const uint32_t blocks = hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
    float g0 = 0.0f;
    float u0 = 0.0f;
    axiom_dev_iq2_xxs_dual_dot_q8k_qwarp8(gate_row, up_row, input, blocks, lane8, &g0, &u0);
    if (lane8 == 0u) {
        const float g = fminf(g0, 10.0f);
        const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
        mid[idx] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8k_qwarp32_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t slot = (uint32_t)blockIdx.y;
    if (slot >= topk) return;
    const uint32_t expert = indices[slot];
    if (expert >= experts) return;

    const uint32_t blocks = hidden / 256u;
    __shared__ axiom_cuda_q8_k_block sxq[16];
    const axiom_cuda_q8_k_block *xq = input;
    if (blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < blocks; i += blockDim.x) sxq[i] = input[i];
        __syncthreads();
        xq = sxq;
    }

    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = (uint32_t)blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_hidden) continue;
        const uint32_t idx = slot * expert_hidden + row;
        const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        float g0 = 0.0f;
        float u0 = 0.0f;
        axiom_dev_iq2_xxs_dual_dot_q8k_qwarp8(gate_row, up_row, xq, blocks, lane8, &g0, &u0);
        if (lane8 == 0u) {
            const float g = fminf(g0, 10.0f);
            const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
            mid[idx] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8k_outlier_qwarp32_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint32_t *__restrict__ outlier_count,
        const uint32_t *__restrict__ outlier_indices,
        const float *__restrict__ outlier_deltas,
        uint32_t outlier_cap,
        const uint32_t *__restrict__ indices,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t slot = (uint32_t)blockIdx.y;
    if (slot >= topk) return;
    const uint32_t expert = indices[slot];
    if (expert >= experts) return;

    const uint32_t blocks = hidden / 256u;
    __shared__ axiom_cuda_q8_k_block sxq[16];
    const axiom_cuda_q8_k_block *xq = input;
    if (blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < blocks; i += blockDim.x) sxq[i] = input[i];
        __syncthreads();
        xq = sxq;
    }

    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint32_t outliers = min(outlier_count[0], outlier_cap);
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = (uint32_t)blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_hidden) continue;
        const uint32_t idx = slot * expert_hidden + row;
        const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        float g0 = 0.0f;
        float u0 = 0.0f;
        axiom_dev_iq2_xxs_dual_dot_q8k_qwarp8(gate_row, up_row, xq, blocks, lane8, &g0, &u0);

        float gc = 0.0f;
        float uc = 0.0f;
        for (uint32_t j = lane8; j < outliers; j += 8u) {
            const uint32_t col = outlier_indices[j];
            const float delta = outlier_deltas[j];
            gc += axiom_dev_iq2_xxs_weight_at(gate_row, col) * delta;
            uc += axiom_dev_iq2_xxs_weight_at(up_row, col) * delta;
        }
        const uint32_t mask = 0xffu << (threadIdx.x & 24u);
        for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
            gc += __shfl_down_sync(mask, gc, offset, 8);
            uc += __shfl_down_sync(mask, uc, offset, 8);
        }
        if (lane8 == 0u) {
            const float g = fminf(g0 + gc, 10.0f);
            const float u = fminf(fmaxf(u0 + uc, -10.0f), 10.0f);
            mid[idx] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8t32_qwarp32_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t slot = (uint32_t)blockIdx.y;
    if (slot >= topk) return;
    const uint32_t expert = indices[slot];
    if (expert >= experts) return;

    const uint32_t blocks = hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = (uint32_t)blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_hidden) continue;
        const uint32_t idx = slot * expert_hidden + row;
        const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        float g0 = 0.0f;
        float u0 = 0.0f;
        axiom_dev_iq2_xxs_dual_dot_q8t32_qwarp8(gate_row, up_row, input, blocks, lane8, &g0, &u0);
        if (lane8 == 0u) {
            const float g = fminf(g0, 10.0f);
            const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
            mid[idx] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8k_weighted_qwarp32_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t slot = (uint32_t)blockIdx.y;
    if (slot >= topk) return;
    const uint32_t expert = indices[slot];
    if (expert >= experts) return;

    const uint32_t blocks = hidden / 256u;
    __shared__ axiom_cuda_q8_k_block sxq[16];
    const axiom_cuda_q8_k_block *xq = input;
    if (blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < blocks; i += blockDim.x) sxq[i] = input[i];
        __syncthreads();
        xq = sxq;
    }

    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const float rw = router_weights[slot];
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = (uint32_t)blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_hidden) continue;
        const uint32_t idx = slot * expert_hidden + row;
        const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        float g0 = 0.0f;
        float u0 = 0.0f;
        axiom_dev_iq2_xxs_dual_dot_q8k_qwarp8(gate_row, up_row, xq, blocks, lane8, &g0, &u0);
        if (lane8 == 0u) {
            const float g = fminf(g0, 10.0f);
            const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
            mid[idx] = (g / (1.0f + expf(-g))) * u * rw;
        }
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8k_weighted_block_qwarp32_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t slot = (uint32_t)blockIdx.y;
    if (slot >= topk) return;
    const uint32_t expert = indices[slot];
    if (expert >= experts) return;

    const uint32_t blocks = hidden / 256u;
    __shared__ axiom_cuda_q8_k_block sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    const axiom_cuda_q8_k_block *xq = input;
    if (blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < blocks; i += blockDim.x) sxq[i] = input[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = axiom_iq2xxs_grid64[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = axiom_iq2xxs_sign_mask[i];
        __syncthreads();
        xq = sxq;
    }

    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const float rw = router_weights[slot];
    for (uint32_t rr = 0; rr < 4u; ++rr) {
        const uint32_t row = (uint32_t)blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_hidden) continue;
        const uint32_t idx = slot * expert_hidden + row;
        const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        float g0 = 0.0f;
        float u0 = 0.0f;
        if (blocks <= 16u) {
            axiom_dev_iq2_xxs_dual_dot_q8k_block_lut_lane(gate_row, up_row, xq, blocks, 66u, 2u, lane8, s_iq2_grid, s_iq2_signs, &g0, &u0);
        } else {
            axiom_dev_iq2_xxs_dual_dot_q8k_block_lane(gate_row, up_row, xq, blocks, 66u, 2u, lane8, &g0, &u0);
        }
        if (lane8 == 0u) {
            const float g = fminf(g0, 10.0f);
            const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
            mid[idx] = (g / (1.0f + expf(-g))) * u * rw;
        }
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8k_weighted_pack_midq_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        axiom_cuda_q8_k_block *__restrict__ midq,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden,
        uint32_t no_lut,
        uint32_t fast_reduce) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t rows_per_round = blockDim.x >> 3u;
    const uint32_t rounds = 256u / rows_per_round;
    const uint32_t slot = (uint32_t)blockIdx.y;
    const uint32_t mid_block = (uint32_t)blockIdx.x;
    if (slot >= topk || mid_block >= expert_hidden / 256u) return;
    const uint32_t expert = indices[slot];

    const uint32_t blocks = hidden / 256u;
    __shared__ axiom_cuda_q8_k_block sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    __shared__ float smid[256];
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float warp_abs[8];
    __shared__ float warp_val[8];
    __shared__ float iscale;

    const axiom_cuda_q8_k_block *xq = input;
    if (blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < blocks; i += blockDim.x) sxq[i] = input[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = axiom_iq2xxs_grid64[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = axiom_iq2xxs_sign_mask[i];
    }
    __syncthreads();
    if (blocks <= 16u) xq = sxq;

    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const float rw = router_weights[slot];
    for (uint32_t rr = 0; rr < rounds; ++rr) {
        const uint32_t local_row = row_lane + rr * rows_per_round;
        const uint32_t row = mid_block * 256u + local_row;
        float g0 = 0.0f;
        float u0 = 0.0f;
        if (expert < experts) {
            const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
            const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
            if (blocks <= 16u && !no_lut) {
                axiom_dev_iq2_xxs_dual_dot_q8k_block_lut_lane(gate_row, up_row, xq, blocks, 66u, 2u, lane8, s_iq2_grid, s_iq2_signs, &g0, &u0);
            } else {
                axiom_dev_iq2_xxs_dual_dot_q8k_block_lane(gate_row, up_row, xq, blocks, 66u, 2u, lane8, &g0, &u0);
            }
        }
        if (lane8 == 0u) {
            const float g = fminf(g0, 10.0f);
            const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
            smid[local_row] = expert < experts ? (g / (1.0f + expf(-g))) * u * rw : 0.0f;
        }
    }
    __syncthreads();

    const uint32_t lane = threadIdx.x;
    const float x = lane < 256u ? smid[lane] : 0.0f;
    axiom_cuda_q8_k_block *dst = midq + (uint64_t)slot * (expert_hidden / 256u) + mid_block;
    if (fast_reduce) {
        const uint32_t warp = lane >> 5u;
        const uint32_t lane32 = lane & 31u;
        float best_abs = fabsf(x);
        float best_val = x;
        if (lane < 256u) {
            #pragma unroll
            for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
                const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
                if (other_abs > best_abs) {
                    best_abs = other_abs;
                    best_val = other_val;
                }
            }
            if (lane32 == 0u) {
                warp_abs[warp] = best_abs;
                warp_val[warp] = best_val;
            }
        }
        __syncthreads();

        if (lane < 32u) {
            best_abs = lane32 < 8u ? warp_abs[lane32] : -1.0f;
            best_val = lane32 < 8u ? warp_val[lane32] : 0.0f;
            #pragma unroll
            for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
                const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
                if (other_abs > best_abs) {
                    best_abs = other_abs;
                    best_val = other_val;
                }
            }
            if (lane32 == 0u) iscale = best_abs == 0.0f ? 0.0f : -127.0f / best_val;
        }
    } else {
        if (lane < 256u) {
            absmax[lane] = fabsf(x);
            maxv[lane] = x;
        }
        __syncthreads();
        for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride && absmax[lane + stride] > absmax[lane]) {
                absmax[lane] = absmax[lane + stride];
                maxv[lane] = maxv[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0u) iscale = absmax[0] == 0.0f ? 0.0f : -127.0f / maxv[0];
    }
    __syncthreads();
    if (lane >= 256u) return;
    if (iscale == 0.0f) {
        dst->qs[lane] = 0;
        if (lane < 16u) dst->bsums[lane] = 0;
        if (lane == 0u) dst->d = 0.0f;
        return;
    }
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    dst->qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += dst->qs[lane * 16u + i];
        dst->bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) dst->d = 1.0f / iscale;
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q4k_weighted_pack_midq_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        axiom_cuda_q8_k_block *__restrict__ midq,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden,
        uint32_t fast_reduce) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t rows_per_round = blockDim.x >> 3u;
    const uint32_t rounds = 256u / rows_per_round;
    const uint32_t slot = (uint32_t)blockIdx.y;
    const uint32_t mid_block = (uint32_t)blockIdx.x;
    if (slot >= topk || mid_block >= expert_hidden / 256u) return;
    const uint32_t expert = indices[slot];

    const uint32_t blocks = hidden / 256u;
    __shared__ axiom_cuda_q8_k_block sxq[16];
    __shared__ float smid[256];
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float warp_abs[8];
    __shared__ float warp_val[8];
    __shared__ float iscale;

    const axiom_cuda_q8_k_block *xq = input;
    if (blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < blocks; i += blockDim.x) sxq[i] = input[i];
    }
    __syncthreads();
    if (blocks <= 16u) xq = sxq;

    const uint64_t row_bytes = (uint64_t)blocks * 144u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const float rw = router_weights[slot];
    for (uint32_t rr = 0; rr < rounds; ++rr) {
        const uint32_t local_row = row_lane + rr * rows_per_round;
        const uint32_t row = mid_block * 256u + local_row;
        float g0 = 0.0f;
        float u0 = 0.0f;
        if (expert < experts) {
            const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
            const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
            for (uint32_t b = lane8; b < blocks; b += 8u) {
                g0 += axiom_dev_q4_k_dot_q8k_block(gate_row + (uint64_t)b * 144u, xq + b);
                u0 += axiom_dev_q4_k_dot_q8k_block(up_row + (uint64_t)b * 144u, xq + b);
            }
            const uint32_t mask = 0xffu << (threadIdx.x & 24u);
            for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                g0 += __shfl_down_sync(mask, g0, off, 8);
                u0 += __shfl_down_sync(mask, u0, off, 8);
            }
        }
        if (lane8 == 0u) {
            const float g = fminf(g0, 10.0f);
            const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
            smid[local_row] = expert < experts ? (g / (1.0f + expf(-g))) * u * rw : 0.0f;
        }
    }
    __syncthreads();

    const uint32_t lane = threadIdx.x;
    const float x = lane < 256u ? smid[lane] : 0.0f;
    axiom_cuda_q8_k_block *dst = midq + (uint64_t)slot * (expert_hidden / 256u) + mid_block;
    if (fast_reduce) {
        const uint32_t warp = lane >> 5u;
        const uint32_t lane32 = lane & 31u;
        float best_abs = fabsf(x);
        float best_val = x;
        if (lane < 256u) {
            #pragma unroll
            for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
                const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
                if (other_abs > best_abs) {
                    best_abs = other_abs;
                    best_val = other_val;
                }
            }
            if (lane32 == 0u) {
                warp_abs[warp] = best_abs;
                warp_val[warp] = best_val;
            }
        }
        __syncthreads();

        if (lane < 32u) {
            best_abs = lane32 < 8u ? warp_abs[lane32] : -1.0f;
            best_val = lane32 < 8u ? warp_val[lane32] : 0.0f;
            #pragma unroll
            for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
                const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
                if (other_abs > best_abs) {
                    best_abs = other_abs;
                    best_val = other_val;
                }
            }
            if (lane32 == 0u) iscale = best_abs == 0.0f ? 0.0f : -127.0f / best_val;
        }
    } else {
        if (lane < 256u) {
            absmax[lane] = fabsf(x);
            maxv[lane] = x;
        }
        __syncthreads();
        for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride && absmax[lane + stride] > absmax[lane]) {
                absmax[lane] = absmax[lane + stride];
                maxv[lane] = maxv[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0u) iscale = absmax[0] == 0.0f ? 0.0f : -127.0f / maxv[0];
    }
    __syncthreads();
    if (lane >= 256u) return;
    if (iscale == 0.0f) {
        dst->qs[lane] = 0;
        if (lane < 16u) dst->bsums[lane] = 0;
        if (lane == 0u) dst->d = 0.0f;
        return;
    }
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    dst->qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += dst->qs[lane * 16u + i];
        dst->bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) dst->d = 1.0f / iscale;
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8k_weighted_pack_midq_rowstage_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        axiom_cuda_q8_k_block *__restrict__ midq,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden,
        uint32_t no_lut,
        uint32_t fast_reduce,
        uint32_t block_stride,
        uint32_t aux_offset,
        uint32_t rows_per_round_arg,
        uint32_t lut_reuse) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t rows_per_round =
            rows_per_round_arg == 32u ? 32u : (rows_per_round_arg == 8u ? 8u : 16u);
    const uint32_t rounds = 256u / rows_per_round;
    const uint32_t slot = (uint32_t)blockIdx.y;
    const uint32_t mid_block = (uint32_t)blockIdx.x;
    if (slot >= topk || mid_block >= expert_hidden / 256u) return;
    const uint32_t expert = indices[slot];

    const uint32_t blocks = hidden / 256u;
    __shared__ axiom_cuda_q8_k_block sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    __shared__ float smid[256];
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float warp_abs[8];
    __shared__ float warp_val[8];
    __shared__ float iscale;
    extern __shared__ uint4 s_rows4[];

    const axiom_cuda_q8_k_block *xq = input;
    if (blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < blocks; i += blockDim.x) sxq[i] = input[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = axiom_iq2xxs_grid64[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = axiom_iq2xxs_sign_mask[i];
    }
    __syncthreads();
    if (blocks <= 16u) xq = sxq;

    const uint64_t row_bytes = (uint64_t)blocks * block_stride;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint32_t group_bytes = rows_per_round * (uint32_t)row_bytes;
    const uint32_t group_u4 = group_bytes / sizeof(uint4);
    uint4 *s_gate4 = s_rows4;
    uint4 *s_up4 = s_rows4 + group_u4;
    const float rw = router_weights[slot];
    for (uint32_t rr = 0; rr < rounds; ++rr) {
        const uint32_t first_local_row = rr * rows_per_round;
        const uint32_t first_row = mid_block * 256u + first_local_row;
        if (expert < experts) {
            const uint4 *gate4 = (const uint4 *)(gate + (uint64_t)expert * expert_bytes + (uint64_t)first_row * row_bytes);
            const uint4 *up4 = (const uint4 *)(up + (uint64_t)expert * expert_bytes + (uint64_t)first_row * row_bytes);
            for (uint32_t i = threadIdx.x; i < group_u4; i += blockDim.x) {
                s_gate4[i] = gate4[i];
                s_up4[i] = up4[i];
            }
        }
        __syncthreads();

        if (row_lane < rows_per_round) {
            const uint32_t local_row = row_lane + first_local_row;
            float g0 = 0.0f;
            float u0 = 0.0f;
            if (expert < experts) {
                const uint8_t *gate_row = (const uint8_t *)s_gate4 + (uint64_t)row_lane * row_bytes;
                const uint8_t *up_row = (const uint8_t *)s_up4 + (uint64_t)row_lane * row_bytes;
                if (blocks <= 16u && !no_lut && lut_reuse) {
                    axiom_dev_iq2_xxs_dual_dot_q8k_block_lut_reuse_lane(gate_row, up_row, xq, blocks, block_stride, aux_offset, lane8, s_iq2_grid, s_iq2_signs, &g0, &u0);
                } else if (blocks <= 16u && !no_lut) {
                    axiom_dev_iq2_xxs_dual_dot_q8k_block_lut_lane(gate_row, up_row, xq, blocks, block_stride, aux_offset, lane8, s_iq2_grid, s_iq2_signs, &g0, &u0);
                } else {
                    axiom_dev_iq2_xxs_dual_dot_q8k_block_lane(gate_row, up_row, xq, blocks, block_stride, aux_offset, lane8, &g0, &u0);
                }
            }
            if (lane8 == 0u) {
                const float g = fminf(g0, 10.0f);
                const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
                smid[local_row] = expert < experts ? (g / (1.0f + expf(-g))) * u * rw : 0.0f;
            }
        }
        __syncthreads();
    }

    const uint32_t lane = threadIdx.x;
    const float x = lane < 256u ? smid[lane] : 0.0f;
    axiom_cuda_q8_k_block *dst = midq + (uint64_t)slot * (expert_hidden / 256u) + mid_block;
    if (fast_reduce) {
        const uint32_t warp = lane >> 5u;
        const uint32_t lane32 = lane & 31u;
        float best_abs = fabsf(x);
        float best_val = x;
        if (lane < 256u) {
            #pragma unroll
            for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
                const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
                if (other_abs > best_abs) {
                    best_abs = other_abs;
                    best_val = other_val;
                }
            }
            if (lane32 == 0u) {
                warp_abs[warp] = best_abs;
                warp_val[warp] = best_val;
            }
        }
        __syncthreads();

        if (lane < 32u) {
            best_abs = lane32 < 8u ? warp_abs[lane32] : -1.0f;
            best_val = lane32 < 8u ? warp_val[lane32] : 0.0f;
            #pragma unroll
            for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
                const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
                if (other_abs > best_abs) {
                    best_abs = other_abs;
                    best_val = other_val;
                }
            }
            if (lane32 == 0u) iscale = best_abs == 0.0f ? 0.0f : -127.0f / best_val;
        }
    } else {
        if (lane < 256u) {
            absmax[lane] = fabsf(x);
            maxv[lane] = x;
        }
        __syncthreads();
        for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride && absmax[lane + stride] > absmax[lane]) {
                absmax[lane] = absmax[lane + stride];
                maxv[lane] = maxv[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0u) iscale = absmax[0] == 0.0f ? 0.0f : -127.0f / maxv[0];
    }
    __syncthreads();
    if (lane >= 256u) return;
    if (iscale == 0.0f) {
        dst->qs[lane] = 0;
        if (lane < 16u) dst->bsums[lane] = 0;
        if (lane == 0u) dst->d = 0.0f;
        return;
    }
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    dst->qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += dst->qs[lane * 16u + i];
        dst->bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) dst->d = 1.0f / iscale;
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8k_weighted_mid_rowstage_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_k_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        float *__restrict__ mid,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden,
        uint32_t no_lut,
        uint32_t block_stride,
        uint32_t aux_offset,
        uint32_t rows_per_round_arg,
        uint32_t rounds_per_block_arg,
        uint32_t lut_reuse) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t rows_per_round =
            rows_per_round_arg == 32u ? 32u : (rows_per_round_arg == 8u ? 8u : 16u);
    const uint32_t rounds_total = 256u / rows_per_round;
    const uint32_t rounds_per_block =
            rounds_per_block_arg >= 16u ? 16u :
            (rounds_per_block_arg >= 8u ? 8u :
            (rounds_per_block_arg >= 4u ? 4u :
            (rounds_per_block_arg >= 2u ? 2u : 1u)));
    const uint32_t slot = (uint32_t)blockIdx.y;
    const uint32_t mid_block = (uint32_t)blockIdx.x;
    const uint32_t rr_base = (uint32_t)blockIdx.z * rounds_per_block;
    if (slot >= topk || mid_block >= expert_hidden / 256u || rr_base >= rounds_total) return;
    const uint32_t expert = indices[slot];

    const uint32_t blocks = hidden / 256u;
    __shared__ axiom_cuda_q8_k_block sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    extern __shared__ uint4 s_rows4[];

    const axiom_cuda_q8_k_block *xq = input;
    if (blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < blocks; i += blockDim.x) sxq[i] = input[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = axiom_iq2xxs_grid64[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = axiom_iq2xxs_sign_mask[i];
    }
    __syncthreads();
    if (blocks <= 16u) xq = sxq;

    const uint64_t row_bytes = (uint64_t)blocks * block_stride;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint32_t group_bytes = rows_per_round * (uint32_t)row_bytes;
    const uint32_t group_u4 = group_bytes / sizeof(uint4);
    uint4 *s_gate4 = s_rows4;
    uint4 *s_up4 = s_rows4 + group_u4;
    const float rw = router_weights[slot];

    for (uint32_t r = 0; r < rounds_per_block && rr_base + r < rounds_total; ++r) {
        const uint32_t rr = rr_base + r;
        const uint32_t first_local_row = rr * rows_per_round;
        const uint32_t first_row = mid_block * 256u + first_local_row;

        if (expert < experts) {
            const uint4 *gate4 = (const uint4 *)(gate + (uint64_t)expert * expert_bytes + (uint64_t)first_row * row_bytes);
            const uint4 *up4 = (const uint4 *)(up + (uint64_t)expert * expert_bytes + (uint64_t)first_row * row_bytes);
            for (uint32_t i = threadIdx.x; i < group_u4; i += blockDim.x) {
                s_gate4[i] = gate4[i];
                s_up4[i] = up4[i];
            }
        }
        __syncthreads();

        if (row_lane < rows_per_round) {
            const uint32_t local_row = row_lane + first_local_row;
            const uint32_t row = mid_block * 256u + local_row;
            float g0 = 0.0f;
            float u0 = 0.0f;
            if (expert < experts) {
                const uint8_t *gate_row = (const uint8_t *)s_gate4 + (uint64_t)row_lane * row_bytes;
                const uint8_t *up_row = (const uint8_t *)s_up4 + (uint64_t)row_lane * row_bytes;
                if (blocks <= 16u && !no_lut && lut_reuse) {
                    axiom_dev_iq2_xxs_dual_dot_q8k_block_lut_reuse_lane(gate_row, up_row, xq, blocks, block_stride, aux_offset, lane8, s_iq2_grid, s_iq2_signs, &g0, &u0);
                } else if (blocks <= 16u && !no_lut) {
                    axiom_dev_iq2_xxs_dual_dot_q8k_block_lut_lane(gate_row, up_row, xq, blocks, block_stride, aux_offset, lane8, s_iq2_grid, s_iq2_signs, &g0, &u0);
                } else {
                    axiom_dev_iq2_xxs_dual_dot_q8k_block_lane(gate_row, up_row, xq, blocks, block_stride, aux_offset, lane8, &g0, &u0);
                }
            }
            if (lane8 == 0u) {
                const float g = fminf(g0, 10.0f);
                const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
                mid[(uint64_t)slot * expert_hidden + row] =
                        expert < experts ? (g / (1.0f + expf(-g))) * u * rw : 0.0f;
            }
        }
        __syncthreads();
    }
}

__global__ static void axiom_deepseek_moe_gate_up_indexed_q8t32_weighted_pack_midq_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const axiom_cuda_q8_0_tile32_block *__restrict__ input,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        axiom_cuda_q8_k_block *__restrict__ midq,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t slot = (uint32_t)blockIdx.y;
    const uint32_t mid_block = (uint32_t)blockIdx.x;
    if (slot >= topk || mid_block >= expert_hidden / 256u) return;
    const uint32_t expert = indices[slot];

    const uint32_t blocks = hidden / 256u;
    const uint32_t t32_blocks = hidden / 32u;
    __shared__ axiom_cuda_q8_0_tile32_block sxq[128];
    __shared__ float smid[256];
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float iscale;

    for (uint32_t i = threadIdx.x; i < t32_blocks; i += blockDim.x) {
        sxq[i] = input[i];
    }
    __syncthreads();

    const uint64_t row_bytes = (uint64_t)blocks * 66u;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const float rw = router_weights[slot];
    for (uint32_t rr = 0; rr < 8u; ++rr) {
        const uint32_t local_row = row_lane + rr * 32u;
        const uint32_t row = mid_block * 256u + local_row;
        float g0 = 0.0f;
        float u0 = 0.0f;
        if (expert < experts) {
            const uint8_t *gate_row = gate + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
            const uint8_t *up_row = up + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
            axiom_dev_iq2_xxs_dual_dot_q8t32_qwarp8(
                    gate_row, up_row, sxq, blocks, lane8, &g0, &u0);
        }
        if (lane8 == 0u) {
            const float g = fminf(g0, 10.0f);
            const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
            smid[local_row] = expert < experts ? (g / (1.0f + expf(-g))) * u * rw : 0.0f;
        }
    }
    __syncthreads();

    const uint32_t lane = threadIdx.x;
    const float x = smid[lane];
    absmax[lane] = fabsf(x);
    maxv[lane] = x;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && absmax[lane + stride] > absmax[lane]) {
            absmax[lane] = absmax[lane + stride];
            maxv[lane] = maxv[lane + stride];
        }
        __syncthreads();
    }

    axiom_cuda_q8_k_block *dst = midq + (uint64_t)slot * (expert_hidden / 256u) + mid_block;
    if (absmax[0] == 0.0f) {
        dst->qs[lane] = 0;
        if (lane < 16u) dst->bsums[lane] = 0;
        if (lane == 0u) dst->d = 0.0f;
        return;
    }
    if (lane == 0u) iscale = -127.0f / maxv[0];
    __syncthreads();
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    dst->qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += dst->qs[lane * 16u + i];
        dst->bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) dst->d = 1.0f / iscale;
}

__global__ static void axiom_deepseek_moe_down_q8k_sum6_qwarp32_kernel(
        const uint8_t *__restrict__ down,
        const axiom_cuda_q8_k_block *__restrict__ midq,
        const uint32_t *__restrict__ indices,
        float *__restrict__ out,
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row = (uint32_t)blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 84u;
    const uint64_t expert_bytes = (uint64_t)hidden * row_bytes;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; ++slot) {
        const uint32_t expert = indices[slot];
        if (expert >= experts) continue;
        const uint8_t *down_row = down + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const axiom_cuda_q8_k_block *xq = midq + (uint64_t)slot * blocks;
        float acc = 0.0f;
        for (uint32_t b = lane8; b < blocks; b += 8u) {
            acc += axiom_dev_q2_k_dot_q8k_block(down_row + (uint64_t)b * 84u, xq + b);
        }
        const uint32_t mask = 0xffu << (threadIdx.x & 24u);
        for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(mask, acc, offset, 8);
        }
        if (lane8 == 0u) total += acc;
    }
    if (lane8 == 0u) out[row] = total;
}

__global__ static void axiom_deepseek_moe_down_q8k_sum6_qwarp32_smid_kernel(
        const uint8_t *__restrict__ down,
        const axiom_cuda_q8_k_block *__restrict__ midq,
        const uint32_t *__restrict__ indices,
        float *__restrict__ out,
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row = (uint32_t)blockIdx.x * 32u + (threadIdx.x >> 3u);
    __shared__ axiom_cuda_q8_k_block smidq[48];
    for (uint32_t i = threadIdx.x; i < 48u; i += blockDim.x) {
        smidq[i] = midq[i];
    }
    __syncthreads();
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 84u;
    const uint64_t expert_bytes = (uint64_t)hidden * row_bytes;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; ++slot) {
        const uint32_t expert = indices[slot];
        if (expert >= experts) continue;
        const uint8_t *down_row = down + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const axiom_cuda_q8_k_block *xq = smidq + slot * 8u;
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t b = lane8; b < 8u; b += 8u) {
            acc += axiom_dev_q2_k_dot_q8k_block(down_row + (uint64_t)b * 84u, xq + b);
        }
        const uint32_t mask = 0xffu << (threadIdx.x & 24u);
        for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(mask, acc, offset, 8);
        }
        if (lane8 == 0u) total += acc;
    }
    if (lane8 == 0u) out[row] = total;
}

__global__ static void axiom_deepseek_moe_down_q4k_sum6_qwarp32_smid_kernel(
        const uint8_t *__restrict__ down,
        const axiom_cuda_q8_k_block *__restrict__ midq,
        const uint32_t *__restrict__ indices,
        float *__restrict__ out,
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row = (uint32_t)blockIdx.x * 32u + (threadIdx.x >> 3u);
    __shared__ axiom_cuda_q8_k_block smidq[48];
    for (uint32_t i = threadIdx.x; i < 48u; i += blockDim.x) {
        smidq[i] = midq[i];
    }
    __syncthreads();
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 144u;
    const uint64_t expert_bytes = (uint64_t)hidden * row_bytes;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; ++slot) {
        const uint32_t expert = indices[slot];
        if (expert >= experts) continue;
        const uint8_t *down_row = down + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const axiom_cuda_q8_k_block *xq = smidq + slot * 8u;
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t b = lane8; b < 8u; b += 8u) {
            acc += axiom_dev_q4_k_dot_q8k_block(down_row + (uint64_t)b * 144u, xq + b);
        }
        const uint32_t mask = 0xffu << (threadIdx.x & 24u);
        for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(mask, acc, offset, 8);
        }
        if (lane8 == 0u) total += acc;
    }
    if (lane8 == 0u) out[row] = total;
}

__global__ static void axiom_deepseek_moe_down_q8k_sum6_qwarp32_cmid_kernel(
        const uint8_t *__restrict__ down,
        const uint32_t *__restrict__ indices,
        float *__restrict__ out,
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row = (uint32_t)blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 84u;
    const uint64_t expert_bytes = (uint64_t)hidden * row_bytes;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; ++slot) {
        const uint32_t expert = indices[slot];
        if (expert >= experts) continue;
        const uint8_t *down_row = down + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const axiom_cuda_q8_k_block *xq = axiom_moe_midq_const + slot * 8u;
        float acc = 0.0f;
        #pragma unroll
        for (uint32_t b = lane8; b < 8u; b += 8u) {
            acc += axiom_dev_q2_k_dot_q8k_block(down_row + (uint64_t)b * 84u, xq + b);
        }
        const uint32_t mask = 0xffu << (threadIdx.x & 24u);
        for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(mask, acc, offset, 8);
        }
        if (lane8 == 0u) total += acc;
    }
    if (lane8 == 0u) out[row] = total;
}

__global__ static void axiom_q4_k_matvec_q8k_warp_kernel(
        const uint8_t *__restrict__ weight,
        const axiom_cuda_q8_k_block *__restrict__ input_q8k,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t lane8 = threadIdx.x & 7u;
    const uint32_t row = (uint32_t)blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= rows) return;
    const uint32_t blocks = cols / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 144u;
    const uint8_t *row_base = weight + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = lane8; b < blocks; b += 8u) {
        acc += axiom_dev_q4_k_dot_q8k_block(row_base + (uint64_t)b * 144u, input_q8k + b);
    }
    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(mask, acc, offset, 8);
    }
    if (lane8 == 0u) out[row] = acc;
}

__global__ static void axiom_deepseek_moe_down_indexed_kernel(
        const uint8_t *down,
        const float *mid,
        const uint32_t *indices,
        const float *router_weights,
        float *out,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 84u;
    const uint64_t expert_bytes = (uint64_t)hidden * row_bytes;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < topk; ++slot) {
        const uint32_t expert = indices[slot];
        if (expert >= experts) continue;
        const uint8_t *down_row = down + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const float *mid_row = mid + (uint64_t)slot * expert_hidden;
        acc += router_weights[slot] * axiom_dev_q2_k_dot_f32(down_row, mid_row, blocks);
    }
    out[row] = acc;
}

__global__ static void axiom_deepseek_moe_down_indexed_warp_kernel(
        const uint8_t *__restrict__ down,
        const float *__restrict__ mid,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        float *__restrict__ out,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 84u;
    const uint64_t expert_bytes = (uint64_t)hidden * row_bytes;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < topk; ++slot) {
        const uint32_t expert = indices[slot];
        if (expert >= experts) continue;
        const uint8_t *down_row = down + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const float *mid_row = mid + (uint64_t)slot * expert_hidden;
        const float dot = axiom_dev_q2_k_dot_f32_warp(down_row, mid_row, blocks);
        if (lane == 0u) acc += router_weights[slot] * dot;
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_deepseek_moe_down_indexed_warp_topk6_kernel(
        const uint8_t *__restrict__ down,
        const float *__restrict__ mid,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        float *__restrict__ out,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (topk != 6u) return;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 84u;
    const uint64_t expert_bytes = (uint64_t)hidden * row_bytes;
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; ++slot) {
        const uint32_t expert = indices[slot];
        if (expert >= experts) continue;
        const uint8_t *down_row = down + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const float *mid_row = mid + (uint64_t)slot * expert_hidden;
        const float dot = axiom_dev_q2_k_dot_f32_warp(down_row, mid_row, blocks);
        if (lane == 0u) acc += router_weights[slot] * dot;
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_deepseek_moe_down_indexed_hwarp16_kernel(
        const uint8_t *__restrict__ down,
        const float *__restrict__ mid,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        float *__restrict__ out,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t rows_per_block = blockDim.x >> 4u;
    const uint32_t row = (uint32_t)blockIdx.x * rows_per_block + (threadIdx.x >> 4u);
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * 84u;
    const uint64_t expert_bytes = (uint64_t)hidden * row_bytes;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < topk; ++slot) {
        const uint32_t expert = indices[slot];
        if (expert >= experts) continue;
        const uint8_t *down_row = down + (uint64_t)expert * expert_bytes + (uint64_t)row * row_bytes;
        const float *mid_row = mid + (uint64_t)slot * expert_hidden;
        const float dot = axiom_dev_q2_k_dot_f32_hwarp16(down_row, mid_row, blocks);
        if (lane16 == 0u) acc += router_weights[slot] * dot;
    }
    if (lane16 == 0u) out[row] = acc;
}

__global__ static void axiom_deepseek_moe_indexed_writeset_fused_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const uint8_t *__restrict__ down,
        const float *__restrict__ input,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_weights,
        float *__restrict__ out,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    extern __shared__ float mid_tile[];
    const uint32_t slot = (uint32_t)blockIdx.x;
    const uint32_t qblk = (uint32_t)blockIdx.y;
    if (slot >= topk || qblk >= expert_hidden / 256u) return;
    const uint32_t expert = indices[slot];
    if (expert >= experts) return;

    const uint32_t lane16 = threadIdx.x & 15u;
    const uint32_t hrow = threadIdx.x >> 4u;
    const uint32_t rows_per_block = blockDim.x >> 4u;
    const uint32_t blocks = hidden / 256u;
    const uint64_t gate_row_bytes = (uint64_t)blocks * 66u;
    const uint64_t gate_expert_bytes = (uint64_t)expert_hidden * gate_row_bytes;
    for (uint32_t tile = hrow; tile < 256u; tile += rows_per_block) {
        const uint32_t erow = qblk * 256u + tile;
        const uint8_t *gate_row = gate + (uint64_t)expert * gate_expert_bytes + (uint64_t)erow * gate_row_bytes;
        const uint8_t *up_row = up + (uint64_t)expert * gate_expert_bytes + (uint64_t)erow * gate_row_bytes;
        float g0 = 0.0f;
        float u0 = 0.0f;
        axiom_dev_iq2_xxs_dual_dot_f32_hwarp16(gate_row, up_row, input, blocks, &g0, &u0);
        if (lane16 == 0u) {
            const float g = fminf(g0, 10.0f);
            const float u = fminf(fmaxf(u0, -10.0f), 10.0f);
            mid_tile[tile] = (g / (1.0f + expf(-g))) * u;
        }
    }
    __syncthreads();

    const uint32_t down_blocks = expert_hidden / 256u;
    const uint64_t down_row_bytes = (uint64_t)down_blocks * 84u;
    const uint64_t down_expert_bytes = (uint64_t)hidden * down_row_bytes;
    const float rw = router_weights[slot];
    for (uint32_t row = threadIdx.x; row < hidden; row += blockDim.x) {
        const uint8_t *down_blk = down + (uint64_t)expert * down_expert_bytes +
                (uint64_t)row * down_row_bytes + (uint64_t)qblk * 84u;
        const float partial = rw * axiom_dev_q2_k_block_dot_f32_shared(down_blk, mid_tile);
        atomicAdd(out + row, partial);
    }
}

__device__ static float axiom_dev_sqrtsoftplus(float x) {
    if (x > 20.0f) return sqrtf(x);
    if (x < -20.0f) return expf(0.5f * x);
    return sqrtf(log1pf(expf(x)));
}

__global__ static void axiom_deepseek_router_f16_kernel(
        const uint16_t *router,
        const float *input,
        float *logits,
        uint32_t experts,
        uint32_t hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t expert = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (expert >= experts) return;
    const uint16_t *row = router + (uint64_t)expert * hidden;
    float acc = 0.0f;
    for (uint32_t i = lane; i < hidden; i += 32u) {
        acc += axiom_dev_f16_to_f32(row[i]) * input[i];
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) logits[expert] = acc;
}

__global__ static void axiom_deepseek_router_f32_kernel(
        const float *router,
        const float *input,
        float *logits,
        uint32_t experts,
        uint32_t hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t expert = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (expert >= experts) return;
    const float *row = router + (uint64_t)expert * hidden;
    float acc = 0.0f;
    for (uint32_t i = lane; i < hidden; i += 32u) {
        acc += row[i] * input[i];
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) logits[expert] = acc;
}

__global__ static void axiom_deepseek_router_score_f16_kernel(
        const uint16_t *router,
        const float *input,
        float *scores,
        uint32_t experts,
        uint32_t hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t expert = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (expert >= experts) return;
    const uint16_t *row = router + (uint64_t)expert * hidden;
    float acc = 0.0f;
    for (uint32_t i = lane; i < hidden; i += 32u) {
        acc += axiom_dev_f16_to_f32(row[i]) * input[i];
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) scores[expert] = axiom_dev_sqrtsoftplus(acc);
}

__global__ static void axiom_deepseek_router_topk_kernel(
        const float *logits,
        uint32_t *out_indices,
        float *out_weights,
        uint32_t experts,
        uint32_t topk) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t idx[16];
    float val[16];
    for (uint32_t i = 0; i < topk; ++i) {
        idx[i] = 0xffffffffu;
        val[i] = -3.4028234663852886e+38F;
    }
    for (uint32_t expert = 0; expert < experts; ++expert) {
        uint32_t slot = topk;
        const float x = logits[expert];
        if (!isfinite(x)) continue;
        for (uint32_t j = 0; j < topk; ++j) {
            if (x > val[j]) {
                slot = j;
                break;
            }
        }
        if (slot == topk) continue;
        for (uint32_t j = topk - 1u; j > slot; --j) {
            idx[j] = idx[j - 1u];
            val[j] = val[j - 1u];
        }
        idx[slot] = expert;
        val[slot] = x;
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < topk; ++i) {
        out_indices[i] = idx[i];
        out_weights[i] = idx[i] != 0xffffffffu ? axiom_dev_sqrtsoftplus(val[i]) : 0.0f;
        sum += out_weights[i];
    }
    if (sum <= 0.0f) {
        const float uniform = 1.5f / (float)topk;
        for (uint32_t i = 0; i < topk; ++i) out_weights[i] = uniform;
    } else {
        const float inv = 1.5f / sum;
        for (uint32_t i = 0; i < topk; ++i) out_weights[i] *= inv;
    }
}

__global__ static void axiom_deepseek_router_topk_biased_kernel(
        const float *logits,
        const float *bias,
        uint32_t *out_indices,
        float *out_weights,
        uint32_t experts,
        uint32_t topk) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t idx[16];
    float rank_val[16];
    float score_val[16];
    for (uint32_t i = 0; i < topk; ++i) {
        idx[i] = 0xffffffffu;
        rank_val[i] = -3.4028234663852886e+38F;
        score_val[i] = 0.0f;
    }
    for (uint32_t expert = 0; expert < experts; ++expert) {
        const float score = axiom_dev_sqrtsoftplus(logits[expert]);
        if (!isfinite(score) || !isfinite(bias[expert])) continue;
        const float rank = score + bias[expert];
        uint32_t slot = topk;
        for (uint32_t j = 0; j < topk; ++j) {
            if (rank > rank_val[j]) {
                slot = j;
                break;
            }
        }
        if (slot == topk) continue;
        for (uint32_t j = topk - 1u; j > slot; --j) {
            idx[j] = idx[j - 1u];
            rank_val[j] = rank_val[j - 1u];
            score_val[j] = score_val[j - 1u];
        }
        idx[slot] = expert;
        rank_val[slot] = rank;
        score_val[slot] = score;
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < topk; ++i) sum += score_val[i];
    for (uint32_t i = 0; i < topk; ++i) {
        out_indices[i] = idx[i];
        out_weights[i] = sum > 0.0f ? score_val[i] * 1.5f / sum : 1.5f / (float)topk;
    }
}

__global__ static void axiom_deepseek_router_topk_biased_parallel_kernel(
        const float *logits,
        const float *bias,
        uint32_t *out_indices,
        float *out_weights,
        uint32_t experts,
        uint32_t topk) {
    __shared__ float rank_shared[256];
    __shared__ float score_shared[256];
    const uint32_t tid = (uint32_t)threadIdx.x;
    if (tid < experts) {
        const float score = axiom_dev_sqrtsoftplus(logits[tid]);
        const float b = bias[tid];
        if (isfinite(score) && isfinite(b)) {
            score_shared[tid] = score;
            rank_shared[tid] = score + b;
        } else {
            score_shared[tid] = 0.0f;
            rank_shared[tid] = -3.4028234663852886e+38F;
        }
    }
    __syncthreads();
    if (tid != 0u) return;
    uint32_t idx[16];
    float rank_val[16];
    float score_val[16];
    for (uint32_t i = 0; i < topk; ++i) {
        idx[i] = 0xffffffffu;
        rank_val[i] = -3.4028234663852886e+38F;
        score_val[i] = 0.0f;
    }
    for (uint32_t expert = 0; expert < experts; ++expert) {
        const float rank = rank_shared[expert];
        uint32_t slot = topk;
        for (uint32_t j = 0; j < topk; ++j) {
            if (rank > rank_val[j]) {
                slot = j;
                break;
            }
        }
        if (slot == topk) continue;
        for (uint32_t j = topk - 1u; j > slot; --j) {
            idx[j] = idx[j - 1u];
            rank_val[j] = rank_val[j - 1u];
            score_val[j] = score_val[j - 1u];
        }
        idx[slot] = expert;
        rank_val[slot] = rank;
        score_val[slot] = score_shared[expert];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < topk; ++i) sum += score_val[i];
    for (uint32_t i = 0; i < topk; ++i) {
        out_indices[i] = idx[i];
        out_weights[i] = sum > 0.0f ? score_val[i] * 1.5f / sum : 1.5f / (float)topk;
    }
}

__global__ static void axiom_deepseek_router_topk_biased_score_parallel_kernel(
        const float *scores,
        const float *bias,
        uint32_t *out_indices,
        float *out_weights,
        uint32_t experts,
        uint32_t topk) {
    __shared__ float rank_shared[256];
    const uint32_t tid = (uint32_t)threadIdx.x;
    if (tid < experts) {
        const float score = scores[tid];
        const float b = bias[tid];
        rank_shared[tid] = (isfinite(score) && isfinite(b)) ? score + b : -3.4028234663852886e+38F;
    }
    __syncthreads();
    if (tid != 0u) return;
    uint32_t idx[16];
    float rank_val[16];
    float score_val[16];
    for (uint32_t i = 0; i < topk; ++i) {
        idx[i] = 0xffffffffu;
        rank_val[i] = -3.4028234663852886e+38F;
        score_val[i] = 0.0f;
    }
    for (uint32_t expert = 0; expert < experts; ++expert) {
        const float rank = rank_shared[expert];
        uint32_t slot = topk;
        for (uint32_t j = 0; j < topk; ++j) {
            if (rank > rank_val[j]) {
                slot = j;
                break;
            }
        }
        if (slot == topk) continue;
        for (uint32_t j = topk - 1u; j > slot; --j) {
            idx[j] = idx[j - 1u];
            rank_val[j] = rank_val[j - 1u];
            score_val[j] = score_val[j - 1u];
        }
        idx[slot] = expert;
        rank_val[slot] = rank;
        score_val[slot] = scores[expert];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < topk; ++i) sum += score_val[i];
    for (uint32_t i = 0; i < topk; ++i) {
        out_indices[i] = idx[i];
        out_weights[i] = sum > 0.0f ? score_val[i] * 1.5f / sum : 1.5f / (float)topk;
    }
}

__global__ static void axiom_deepseek_router_topk_biased_score_reduce_kernel(
        const float *scores,
        const float *bias,
        uint32_t *out_indices,
        float *out_weights,
        uint32_t experts,
        uint32_t topk) {
    __shared__ float rank_shared[256];
    __shared__ float score_shared[256];
    __shared__ float reduce_rank[256];
    __shared__ uint32_t reduce_idx[256];
    __shared__ float selected_score[16];
    const uint32_t tid = (uint32_t)threadIdx.x;
    if (tid < experts) {
        const float score = scores[tid];
        const float b = bias[tid];
        score_shared[tid] = (isfinite(score) && isfinite(b)) ? score : 0.0f;
        rank_shared[tid] = (isfinite(score) && isfinite(b)) ? score + b : -3.4028234663852886e+38F;
    } else {
        score_shared[tid] = 0.0f;
        rank_shared[tid] = -3.4028234663852886e+38F;
    }
    __syncthreads();

    for (uint32_t slot = 0; slot < topk; ++slot) {
        reduce_rank[tid] = rank_shared[tid];
        reduce_idx[tid] = tid < experts ? tid : 0xffffffffu;
        __syncthreads();
        for (uint32_t step = 128u; step > 0u; step >>= 1u) {
            if (tid < step) {
                const float lhs = reduce_rank[tid];
                const float rhs = reduce_rank[tid + step];
                const uint32_t lhs_idx = reduce_idx[tid];
                const uint32_t rhs_idx = reduce_idx[tid + step];
                if (rhs > lhs || (rhs == lhs && rhs_idx < lhs_idx)) {
                    reduce_rank[tid] = rhs;
                    reduce_idx[tid] = rhs_idx;
                }
            }
            __syncthreads();
        }
        const uint32_t best = reduce_idx[0];
        if (tid == 0u) {
            out_indices[slot] = best;
            selected_score[slot] = best != 0xffffffffu ? score_shared[best] : 0.0f;
        }
        if (tid == best) {
            rank_shared[tid] = -3.4028234663852886e+38F;
        }
        __syncthreads();
    }
    if (tid < topk) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < topk; ++i) sum += selected_score[i];
        out_weights[tid] = sum > 0.0f ? selected_score[tid] * 1.5f / sum : 1.5f / (float)topk;
    }
}

extern "C" int axiom_cuda_deepseek_router_topk_f16_f32_device(
        void *cuda_runtime,
        const void *router_f16,
        uint64_t router_offset,
        const void *input,
        uint64_t input_offset,
        void *out_indices,
        uint64_t out_indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk) {
    if (!cuda_runtime || !router_f16 || !input || !out_indices || !out_weights ||
        experts == 0 || hidden == 0 || topk == 0 || topk > 16 || topk > experts) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_f16;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *idxbuf = (axiom_cuda_buffer *)out_indices;
    axiom_cuda_buffer *wbuf = (axiom_cuda_buffer *)out_weights;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *logits = nullptr;
    int rc = axiom_cuda_router_logits_scratch(runtime, experts, &logits);
    if (rc != AXIOM_OK) return rc;

    const int block = 128;
    const int warps_per_block = block / 32;
    const int grid = (int)((experts + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_deepseek_router_f16_kernel<<<grid, block>>>(
            (const uint16_t *)((const uint8_t *)rbuf->ptr + router_offset),
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            logits,
            experts,
            hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    axiom_deepseek_router_topk_kernel<<<1, 1>>>(
            logits,
            (uint32_t *)((uint8_t *)idxbuf->ptr + out_indices_offset),
            (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
            experts,
            topk);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_router_topk_biased_f16_f32_device(
        void *cuda_runtime,
        const void *router_f16,
        uint64_t router_offset,
        const void *bias_f32,
        uint64_t bias_offset,
        const void *input,
        uint64_t input_offset,
        void *out_indices,
        uint64_t out_indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk) {
    if (!cuda_runtime || !router_f16 || !bias_f32 || !input || !out_indices || !out_weights ||
        experts == 0 || hidden == 0 || topk == 0 || topk > 16 || topk > experts) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_f16;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)bias_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *idxbuf = (axiom_cuda_buffer *)out_indices;
    axiom_cuda_buffer *wbuf = (axiom_cuda_buffer *)out_weights;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *logits = nullptr;
    err = cudaMalloc(&logits, (size_t)experts * sizeof(float));
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const int block = 128;
    const int warps_per_block = block / 32;
    const int grid = (int)((experts + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    if (experts <= 256u) {
        axiom_deepseek_router_score_f16_kernel<<<grid, block>>>(
                (const uint16_t *)((const uint8_t *)rbuf->ptr + router_offset),
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                logits,
                experts,
                hidden);
    } else {
        axiom_deepseek_router_f16_kernel<<<grid, block>>>(
                (const uint16_t *)((const uint8_t *)rbuf->ptr + router_offset),
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                logits,
                experts,
                hidden);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(logits);
        return axiom_cuda_status(err);
    }
    if (experts <= 256u) {
        if (axiom_cuda_env_enabled("AXIOM_DS4_ROUTER_TOPK_SINGLE")) {
            axiom_deepseek_router_topk_biased_score_parallel_kernel<<<1, 256>>>(
                    logits,
                    (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
                    (uint32_t *)((uint8_t *)idxbuf->ptr + out_indices_offset),
                    (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                    experts,
                    topk);
        } else {
        axiom_deepseek_router_topk_biased_score_reduce_kernel<<<1, 256>>>(
                logits,
                (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
                (uint32_t *)((uint8_t *)idxbuf->ptr + out_indices_offset),
                (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                experts,
                topk);
        }
    } else {
        axiom_deepseek_router_topk_biased_kernel<<<1, 1>>>(
                logits,
                (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
                (uint32_t *)((uint8_t *)idxbuf->ptr + out_indices_offset),
                (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                experts,
                topk);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(logits);
        return axiom_cuda_status(err);
    }
    err = cudaDeviceSynchronize();
    cudaFree(logits);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_deepseek_router_topk_biased_f32_f32_device(
        void *cuda_runtime,
        const void *router_f32,
        uint64_t router_offset,
        const void *bias_f32,
        uint64_t bias_offset,
        const void *input,
        uint64_t input_offset,
        void *out_indices,
        uint64_t out_indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk) {
    if (!cuda_runtime || !router_f32 || !bias_f32 || !input || !out_indices || !out_weights ||
        experts == 0 || hidden == 0 || topk == 0 || topk > 16 || topk > experts) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_f32;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)bias_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *idxbuf = (axiom_cuda_buffer *)out_indices;
    axiom_cuda_buffer *wbuf = (axiom_cuda_buffer *)out_weights;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *logits = nullptr;
    err = cudaMalloc(&logits, (size_t)experts * sizeof(float));
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const int block = 128;
    const int warps_per_block = block / 32;
    const int grid = (int)((experts + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_deepseek_router_f32_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)rbuf->ptr + router_offset),
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            logits,
            experts,
            hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(logits);
        return axiom_cuda_status(err);
    }
    if (experts <= 256u) {
        axiom_deepseek_router_topk_biased_parallel_kernel<<<1, 256>>>(
                logits,
                (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
                (uint32_t *)((uint8_t *)idxbuf->ptr + out_indices_offset),
                (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                experts,
                topk);
    } else {
        axiom_deepseek_router_topk_biased_kernel<<<1, 1>>>(
                logits,
                (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
                (uint32_t *)((uint8_t *)idxbuf->ptr + out_indices_offset),
                (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                experts,
                topk);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(logits);
        return axiom_cuda_status(err);
    }
    err = cudaDeviceSynchronize();
    cudaFree(logits);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_deepseek_router_topk_biased_f16_f32_scratch_device(
        void *cuda_runtime,
        const void *router_f16,
        uint64_t router_offset,
        const void *bias_f32,
        uint64_t bias_offset,
        const void *input,
        uint64_t input_offset,
        void *scratch_logits,
        uint64_t scratch_logits_offset,
        void *out_indices,
        uint64_t out_indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk) {
    if (!cuda_runtime || !router_f16 || !bias_f32 || !input || !scratch_logits ||
        !out_indices || !out_weights || experts == 0 || hidden == 0 ||
        topk == 0 || topk > 16 || topk > experts) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_f16;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)bias_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *lbuf = (axiom_cuda_buffer *)scratch_logits;
    axiom_cuda_buffer *idxbuf = (axiom_cuda_buffer *)out_indices;
    axiom_cuda_buffer *wbuf = (axiom_cuda_buffer *)out_weights;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *logits = (float *)((uint8_t *)lbuf->ptr + scratch_logits_offset);
    const int block = 128;
    const int warps_per_block = block / 32;
    const int grid = (int)((experts + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    if (experts <= 256u) {
        axiom_deepseek_router_score_f16_kernel<<<grid, block>>>(
                (const uint16_t *)((const uint8_t *)rbuf->ptr + router_offset),
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                logits,
                experts,
                hidden);
    } else {
        axiom_deepseek_router_f16_kernel<<<grid, block>>>(
                (const uint16_t *)((const uint8_t *)rbuf->ptr + router_offset),
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                logits,
                experts,
                hidden);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    if (experts <= 256u) {
        if (axiom_cuda_env_enabled("AXIOM_DS4_ROUTER_TOPK_SINGLE")) {
            axiom_deepseek_router_topk_biased_score_parallel_kernel<<<1, 256>>>(
                    logits,
                    (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
                    (uint32_t *)((uint8_t *)idxbuf->ptr + out_indices_offset),
                    (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                    experts,
                    topk);
        } else {
        axiom_deepseek_router_topk_biased_score_reduce_kernel<<<1, 256>>>(
                logits,
                (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
                (uint32_t *)((uint8_t *)idxbuf->ptr + out_indices_offset),
                (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                experts,
                topk);
        }
    } else {
        axiom_deepseek_router_topk_biased_kernel<<<1, 1>>>(
                logits,
                (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
                (uint32_t *)((uint8_t *)idxbuf->ptr + out_indices_offset),
                (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                experts,
                topk);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_router_hash_kernel(
        const uint16_t *router,
        const float *input,
        const uint32_t *indices,
        float *out_weights,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float sum = 0.0f;
    for (uint32_t i = 0; i < topk; ++i) {
        const uint32_t expert = indices[i];
        if (expert >= experts) {
            out_weights[i] = 0.0f;
            continue;
        }
        const uint16_t *row = router + (uint64_t)expert * hidden;
        float acc = 0.0f;
        for (uint32_t j = 0; j < hidden; ++j) {
            acc += axiom_dev_f16_to_f32(row[j]) * input[j];
        }
        const float w = axiom_dev_sqrtsoftplus(acc);
        out_weights[i] = w;
        sum += w;
    }
    if (sum <= 0.0f) {
        const float uniform = 1.5f / (float)topk;
        for (uint32_t i = 0; i < topk; ++i) out_weights[i] = uniform;
    } else {
        const float inv = 1.5f / sum;
        for (uint32_t i = 0; i < topk; ++i) out_weights[i] *= inv;
    }
}

__global__ static void axiom_deepseek_router_hash_parallel_kernel(
        const uint16_t *router,
        const float *input,
        const uint32_t *indices,
        float *out_weights,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk) {
    __shared__ float partial[256];
    __shared__ float scores[16];

    const uint32_t tid = threadIdx.x;
    if (blockIdx.x != 0 || tid >= 256u) return;

    for (uint32_t i = 0; i < topk; ++i) {
        const uint32_t expert = indices[i];
        float acc = 0.0f;
        if (expert < experts) {
            const uint16_t *row = router + (uint64_t)expert * hidden;
            for (uint32_t j = tid; j < hidden; j += 256u) {
                acc += axiom_dev_f16_to_f32(row[j]) * input[j];
            }
        }
        partial[tid] = acc;
        __syncthreads();

        for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) partial[tid] += partial[tid + stride];
            __syncthreads();
        }

        if (tid == 0u) {
            const float w = expert < experts ? axiom_dev_sqrtsoftplus(partial[0]) : 0.0f;
            scores[i] = w;
            out_weights[i] = w;
        }
        __syncthreads();
    }

    if (tid == 0u) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < topk; ++i) sum += scores[i];
        if (sum <= 0.0f) {
            const float uniform = 1.5f / (float)topk;
            for (uint32_t i = 0; i < topk; ++i) out_weights[i] = uniform;
        } else {
            const float inv = 1.5f / sum;
            for (uint32_t i = 0; i < topk; ++i) out_weights[i] *= inv;
        }
    }
}

extern "C" int axiom_cuda_deepseek_router_hash_f16_f32_device(
        void *cuda_runtime,
        const void *router_f16,
        uint64_t router_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk) {
    if (!cuda_runtime || !router_f16 || !input || !indices || !out_weights ||
        experts == 0 || hidden == 0 || topk == 0 || topk > 16) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_f16;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    const axiom_cuda_buffer *idxbuf = (const axiom_cuda_buffer *)indices;
    axiom_cuda_buffer *wbuf = (axiom_cuda_buffer *)out_weights;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    if (axiom_cuda_env_enabled("AXIOM_DS4_ROUTER_HASH_PARALLEL")) {
        axiom_deepseek_router_hash_parallel_kernel<<<1, 256>>>(
                (const uint16_t *)((const uint8_t *)rbuf->ptr + router_offset),
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                experts,
                hidden,
                topk);
    } else {
        axiom_deepseek_router_hash_kernel<<<1, 1>>>(
                (const uint16_t *)((const uint8_t *)rbuf->ptr + router_offset),
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)wbuf->ptr + out_weights_offset),
                experts,
                hidden,
                topk);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_moe_topk_f32_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_iq2xxs || !up_iq2xxs || !down_q2k || !input ||
        !router_weights || !out || experts == 0 || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_iq2xxs;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_iq2xxs;
    const axiom_cuda_buffer *dbuf = (const axiom_cuda_buffer *)down_q2k;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_weights;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *mid = nullptr;
    err = cudaMalloc(&mid, (size_t)experts * expert_hidden * sizeof(float));
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const int block = 128;
    const int gate_grid = (int)(((uint64_t)experts * expert_hidden + block - 1u) / block);
    axiom_deepseek_moe_gate_up_kernel<<<gate_grid, block>>>(
            (const uint8_t *)gbuf->ptr + gate_offset,
            (const uint8_t *)ubuf->ptr + up_offset,
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            mid,
            experts,
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(mid);
        return axiom_cuda_status(err);
    }

    const int down_grid = (int)((hidden + block - 1u) / block);
    axiom_deepseek_moe_down_kernel<<<down_grid, block>>>(
            (const uint8_t *)dbuf->ptr + down_offset,
            mid,
            (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            experts,
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(mid);
        return axiom_cuda_status(err);
    }
    err = cudaDeviceSynchronize();
    cudaFree(mid);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_deepseek_moe_indexed_f32_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_iq2xxs || !up_iq2xxs || !down_q2k || !input ||
        !indices || !router_weights || !out || experts == 0 || topk == 0 || topk > 16 ||
        topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_iq2xxs;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_iq2xxs;
    const axiom_cuda_buffer *dbuf = (const axiom_cuda_buffer *)down_q2k;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    const axiom_cuda_buffer *idxbuf = (const axiom_cuda_buffer *)indices;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_weights;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *mid = nullptr;
    err = cudaMalloc(&mid, (size_t)topk * expert_hidden * sizeof(float));
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const int block = 256;
    if (expert_hidden <= 2048u) {
        if (!axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_WARP32")) {
            const int gate_block = axiom_cuda_env_int("AXIOM_DS4_MOE_GATEUP_BLOCK", 512, 64, 512, 16);
            const int rows_per_block = gate_block / 16;
            const int gate_grid = (int)(((uint64_t)topk * expert_hidden + rows_per_block - 1u) / rows_per_block);
            const size_t shared_bytes = (size_t)hidden * sizeof(float);
            if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_CPASYNC") && shared_bytes <= 49152u) {
                axiom_deepseek_moe_gate_up_indexed_hwarp16_cpasync_sx_kernel<<<gate_grid, gate_block, shared_bytes>>>(
                        (const uint8_t *)gbuf->ptr + gate_offset,
                        (const uint8_t *)ubuf->ptr + up_offset,
                        (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        mid,
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            } else if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_SX") && shared_bytes <= 49152u) {
                axiom_deepseek_moe_gate_up_indexed_hwarp16_sx_kernel<<<gate_grid, gate_block, shared_bytes>>>(
                        (const uint8_t *)gbuf->ptr + gate_offset,
                        (const uint8_t *)ubuf->ptr + up_offset,
                        (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        mid,
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            } else {
                axiom_deepseek_moe_gate_up_indexed_hwarp16_kernel<<<gate_grid, gate_block>>>(
                        (const uint8_t *)gbuf->ptr + gate_offset,
                        (const uint8_t *)ubuf->ptr + up_offset,
                        (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        mid,
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            }
        } else {
            const int gate_block = 128;
            const int warps_per_block = gate_block / 32;
            const int gate_grid = (int)(((uint64_t)topk * expert_hidden + warps_per_block - 1u) / warps_per_block);
            axiom_deepseek_moe_gate_up_indexed_warp_kernel<<<gate_grid, gate_block>>>(
                    (const uint8_t *)gbuf->ptr + gate_offset,
                    (const uint8_t *)ubuf->ptr + up_offset,
                    (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                    (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                    mid,
                    experts,
                    topk,
                    hidden,
                    expert_hidden);
        }
    } else {
        const int gate_grid = (int)(((uint64_t)topk * expert_hidden + block - 1u) / block);
        axiom_deepseek_moe_gate_up_indexed_kernel<<<gate_grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                mid,
                experts,
                topk,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(mid);
        return axiom_cuda_status(err);
    }

    if (expert_hidden <= 2048u) {
        if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_H16")) {
            const int rows_per_block = block / 16;
            const int down_grid = (int)((hidden + (uint32_t)rows_per_block - 1u) / (uint32_t)rows_per_block);
            axiom_deepseek_moe_down_indexed_hwarp16_kernel<<<down_grid, block>>>(
                    (const uint8_t *)dbuf->ptr + down_offset,
                    mid,
                    (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                    (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                    (float *)((uint8_t *)obuf->ptr + out_offset),
                    experts,
                    topk,
                    hidden,
                    expert_hidden);
        } else {
            const int warps_per_block = block / 32;
            const int down_grid = (int)((hidden + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
            if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_UNROLL6") && topk == 6u) {
                axiom_deepseek_moe_down_indexed_warp_topk6_kernel<<<down_grid, block>>>(
                        (const uint8_t *)dbuf->ptr + down_offset,
                        mid,
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                        (float *)((uint8_t *)obuf->ptr + out_offset),
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            } else {
                axiom_deepseek_moe_down_indexed_warp_kernel<<<down_grid, block>>>(
                        (const uint8_t *)dbuf->ptr + down_offset,
                        mid,
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                        (float *)((uint8_t *)obuf->ptr + out_offset),
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            }
        }
    } else {
        const int down_grid = (int)((hidden + block - 1u) / block);
        axiom_deepseek_moe_down_indexed_kernel<<<down_grid, block>>>(
                (const uint8_t *)dbuf->ptr + down_offset,
                mid,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(mid);
        return axiom_cuda_status(err);
    }
    err = cudaDeviceSynchronize();
    cudaFree(mid);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_deepseek_moe_indexed_f32_scratch_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *scratch_mid,
        uint64_t scratch_mid_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_iq2xxs || !up_iq2xxs || !down_q2k || !input ||
        !indices || !router_weights || !scratch_mid || !out || experts == 0 ||
        topk == 0 || topk > 16 || topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_iq2xxs;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_iq2xxs;
    const axiom_cuda_buffer *dbuf = (const axiom_cuda_buffer *)down_q2k;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    const axiom_cuda_buffer *idxbuf = (const axiom_cuda_buffer *)indices;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_weights;
    axiom_cuda_buffer *sbuf = (axiom_cuda_buffer *)scratch_mid;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *mid = (float *)((uint8_t *)sbuf->ptr + scratch_mid_offset);
    const int block = 256;
    if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_WRITESET_FUSED") &&
        expert_hidden <= 2048u && (expert_hidden % 256u) == 0u) {
        err = cudaMemsetAsync((float *)((uint8_t *)obuf->ptr + out_offset), 0,
                (size_t)hidden * sizeof(float));
        if (err != cudaSuccess) return axiom_cuda_status(err);
        const dim3 grid((unsigned int)topk, (unsigned int)(expert_hidden / 256u));
        axiom_deepseek_moe_indexed_writeset_fused_kernel<<<grid, block, 256u * sizeof(float)>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                (const uint8_t *)dbuf->ptr + down_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
        err = cudaGetLastError();
        return axiom_cuda_finish_after_launch(err);
    }
    if (expert_hidden <= 2048u) {
        if (!axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_WARP32")) {
            const int gate_block = axiom_cuda_env_int("AXIOM_DS4_MOE_GATEUP_BLOCK", 512, 64, 512, 16);
            const int rows_per_block = gate_block / 16;
            const int gate_grid = (int)(((uint64_t)topk * expert_hidden + rows_per_block - 1u) / rows_per_block);
            const size_t shared_bytes = (size_t)hidden * sizeof(float);
            if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_CPASYNC") && shared_bytes <= 49152u) {
                axiom_deepseek_moe_gate_up_indexed_hwarp16_cpasync_sx_kernel<<<gate_grid, gate_block, shared_bytes>>>(
                        (const uint8_t *)gbuf->ptr + gate_offset,
                        (const uint8_t *)ubuf->ptr + up_offset,
                        (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        mid,
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            } else if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_SX") && shared_bytes <= 49152u) {
                axiom_deepseek_moe_gate_up_indexed_hwarp16_sx_kernel<<<gate_grid, gate_block, shared_bytes>>>(
                        (const uint8_t *)gbuf->ptr + gate_offset,
                        (const uint8_t *)ubuf->ptr + up_offset,
                        (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        mid,
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            } else {
                axiom_deepseek_moe_gate_up_indexed_hwarp16_kernel<<<gate_grid, gate_block>>>(
                        (const uint8_t *)gbuf->ptr + gate_offset,
                        (const uint8_t *)ubuf->ptr + up_offset,
                        (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        mid,
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            }
        } else {
            const int warps_per_block = block / 32;
            const int gate_grid = (int)(((uint64_t)topk * expert_hidden + warps_per_block - 1u) / warps_per_block);
            axiom_deepseek_moe_gate_up_indexed_warp_kernel<<<gate_grid, block>>>(
                    (const uint8_t *)gbuf->ptr + gate_offset,
                    (const uint8_t *)ubuf->ptr + up_offset,
                    (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                    (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                    mid,
                    experts,
                    topk,
                    hidden,
                    expert_hidden);
        }
    } else {
        const int gate_grid = (int)(((uint64_t)topk * expert_hidden + block - 1u) / block);
        axiom_deepseek_moe_gate_up_indexed_kernel<<<gate_grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                mid,
                experts,
                topk,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);

    if (expert_hidden <= 2048u) {
        if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_H16")) {
            const int rows_per_block = block / 16;
            const int down_grid = (int)((hidden + (uint32_t)rows_per_block - 1u) / (uint32_t)rows_per_block);
            axiom_deepseek_moe_down_indexed_hwarp16_kernel<<<down_grid, block>>>(
                    (const uint8_t *)dbuf->ptr + down_offset,
                    mid,
                    (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                    (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                    (float *)((uint8_t *)obuf->ptr + out_offset),
                    experts,
                    topk,
                    hidden,
                    expert_hidden);
        } else {
            const int warps_per_block = block / 32;
            const int down_grid = (int)((hidden + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
            if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_UNROLL6") && topk == 6u) {
                axiom_deepseek_moe_down_indexed_warp_topk6_kernel<<<down_grid, block>>>(
                        (const uint8_t *)dbuf->ptr + down_offset,
                        mid,
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                        (float *)((uint8_t *)obuf->ptr + out_offset),
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            } else {
                axiom_deepseek_moe_down_indexed_warp_kernel<<<down_grid, block>>>(
                        (const uint8_t *)dbuf->ptr + down_offset,
                        mid,
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                        (float *)((uint8_t *)obuf->ptr + out_offset),
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            }
        }
    } else {
        const int down_grid = (int)((hidden + block - 1u) / block);
        axiom_deepseek_moe_down_indexed_kernel<<<down_grid, block>>>(
                (const uint8_t *)dbuf->ptr + down_offset,
                mid,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_moe_gate_up_indexed_f32_scratch_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        void *out_mid,
        uint64_t out_mid_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_iq2xxs || !up_iq2xxs || !input ||
        !indices || !out_mid || experts == 0 || topk == 0 || topk > 16 ||
        topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_iq2xxs;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_iq2xxs;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    const axiom_cuda_buffer *idxbuf = (const axiom_cuda_buffer *)indices;
    axiom_cuda_buffer *mbuf = (axiom_cuda_buffer *)out_mid;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const int block = 256;
    if (expert_hidden <= 2048u) {
        const int warps_per_block = block / 32;
        const int grid = (int)(((uint64_t)topk * expert_hidden + warps_per_block - 1u) / warps_per_block);
        axiom_deepseek_moe_gate_up_indexed_warp_kernel<<<grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)mbuf->ptr + out_mid_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
    } else {
        const int grid = (int)(((uint64_t)topk * expert_hidden + block - 1u) / block);
        axiom_deepseek_moe_gate_up_indexed_kernel<<<grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)mbuf->ptr + out_mid_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_moe_gate_up_indexed_q8k_scratch_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        void *scratch_q8k,
        uint64_t scratch_q8k_offset,
        void *out_mid,
        uint64_t out_mid_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_iq2xxs || !up_iq2xxs || !input ||
        !indices || !scratch_q8k || !out_mid || experts == 0 || topk == 0 ||
        topk > 16 || topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_iq2xxs;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_iq2xxs;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    const axiom_cuda_buffer *idxbuf = (const axiom_cuda_buffer *)indices;
    axiom_cuda_buffer *qbuf = (axiom_cuda_buffer *)scratch_q8k;
    axiom_cuda_buffer *mbuf = (axiom_cuda_buffer *)out_mid;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const uint32_t blocks = hidden / 256u;
    if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_Q8T32")) {
        const uint32_t t32_blocks = hidden / 32u;
        axiom_cuda_q8_0_tile32_block *xq32 =
                (axiom_cuda_q8_0_tile32_block *)((uint8_t *)qbuf->ptr + scratch_q8k_offset);
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)t32_blocks, 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq32,
                t32_blocks);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);

        const dim3 block(256, 1, 1);
        const dim3 grid((unsigned int)((expert_hidden + 127u) / 128u), (unsigned int)topk, 1u);
        axiom_deepseek_moe_gate_up_indexed_q8t32_qwarp32_kernel<<<grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq32,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)mbuf->ptr + out_mid_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
        err = cudaGetLastError();
        return axiom_cuda_finish_after_launch(err);
    }

    axiom_cuda_q8_k_block *xq = (axiom_cuda_q8_k_block *)((uint8_t *)qbuf->ptr + scratch_q8k_offset);
    axiom_q8_k_pack_f32_kernel<<<(int)blocks, 256>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            xq,
            blocks);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);

    if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_Q8K_OUTLIER")) {
        const uint32_t outlier_cap = (uint32_t)axiom_cuda_env_int(
                "AXIOM_DS4_MOE_GATEUP_Q8K_OUTLIER_CAP", 128, 1, 256, 1);
        const float outlier_thr = axiom_cuda_env_float(
                "AXIOM_DS4_MOE_GATEUP_Q8K_OUTLIER_THR", 6.0f, 0.0f, 100.0f);
        const uint64_t q8_bytes = (uint64_t)blocks * sizeof(axiom_cuda_q8_k_block);
        uint8_t *scratch = (uint8_t *)qbuf->ptr + scratch_q8k_offset;
        const uint64_t count_off = q8_bytes;
        const uint64_t idx_off = count_off + sizeof(uint32_t);
        const uint64_t delta_off = idx_off + (uint64_t)outlier_cap * sizeof(uint32_t);
        const uint64_t need_bytes = delta_off + (uint64_t)outlier_cap * sizeof(float);
        if (scratch_q8k_offset > qbuf->bytes || need_bytes > qbuf->bytes - scratch_q8k_offset) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        uint32_t *outlier_count = (uint32_t *)(scratch + count_off);
        uint32_t *outlier_indices = (uint32_t *)(scratch + idx_off);
        float *outlier_deltas = (float *)(scratch + delta_off);
        err = cudaMemset(outlier_count, 0, sizeof(uint32_t));
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_q8_k_outlier_delta_kernel<<<(int)((hidden + 255u) / 256u), 256>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                outlier_count,
                outlier_indices,
                outlier_deltas,
                hidden,
                outlier_thr,
                outlier_cap);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);

        const dim3 block(256, 1, 1);
        const dim3 grid((unsigned int)((expert_hidden + 127u) / 128u), (unsigned int)topk, 1u);
        axiom_deepseek_moe_gate_up_indexed_q8k_outlier_qwarp32_kernel<<<grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                outlier_count,
                outlier_indices,
                outlier_deltas,
                outlier_cap,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)mbuf->ptr + out_mid_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
        err = cudaGetLastError();
        return axiom_cuda_finish_after_launch(err);
    }

    if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_Q8K_OLD")) {
        const int block = 64;
        const int warps_per_block = block / 32;
        const int rows_per_block = warps_per_block * 4;
        const int grid = (int)(((uint64_t)topk * expert_hidden + rows_per_block - 1u) / rows_per_block);
        axiom_deepseek_moe_gate_up_indexed_q8k_kernel<<<grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)mbuf->ptr + out_mid_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
    } else {
        const dim3 block(256, 1, 1);
        const dim3 grid((unsigned int)((expert_hidden + 127u) / 128u), (unsigned int)topk, 1u);
        axiom_deepseek_moe_gate_up_indexed_q8k_qwarp32_kernel<<<grid, block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)mbuf->ptr + out_mid_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_moe_down_indexed_f32_device(
        void *cuda_runtime,
        const void *down_q2k,
        uint64_t down_offset,
        const void *mid,
        uint64_t mid_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !down_q2k || !mid || !indices || !router_weights || !out ||
        experts == 0 || topk == 0 || topk > 16 || topk > experts ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *dbuf = (const axiom_cuda_buffer *)down_q2k;
    const axiom_cuda_buffer *mbuf = (const axiom_cuda_buffer *)mid;
    const axiom_cuda_buffer *idxbuf = (const axiom_cuda_buffer *)indices;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_weights;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const int block = 256;
    if (expert_hidden <= 2048u) {
        if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_H16")) {
            const int rows_per_block = block / 16;
            const int down_grid = (int)((hidden + (uint32_t)rows_per_block - 1u) / (uint32_t)rows_per_block);
            axiom_deepseek_moe_down_indexed_hwarp16_kernel<<<down_grid, block>>>(
                    (const uint8_t *)dbuf->ptr + down_offset,
                    (const float *)((const uint8_t *)mbuf->ptr + mid_offset),
                    (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                    (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                    (float *)((uint8_t *)obuf->ptr + out_offset),
                    experts,
                    topk,
                    hidden,
                    expert_hidden);
        } else {
            const int warps_per_block = block / 32;
            const int down_grid = (int)((hidden + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
            if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_UNROLL6") && topk == 6u) {
                axiom_deepseek_moe_down_indexed_warp_topk6_kernel<<<down_grid, block>>>(
                        (const uint8_t *)dbuf->ptr + down_offset,
                        (const float *)((const uint8_t *)mbuf->ptr + mid_offset),
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                        (float *)((uint8_t *)obuf->ptr + out_offset),
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            } else {
                axiom_deepseek_moe_down_indexed_warp_kernel<<<down_grid, block>>>(
                        (const uint8_t *)dbuf->ptr + down_offset,
                        (const float *)((const uint8_t *)mbuf->ptr + mid_offset),
                        (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                        (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                        (float *)((uint8_t *)obuf->ptr + out_offset),
                        experts,
                        topk,
                        hidden,
                        expert_hidden);
            }
        }
    } else {
        const int down_grid = (int)((hidden + block - 1u) / block);
        axiom_deepseek_moe_down_indexed_kernel<<<down_grid, block>>>(
                (const uint8_t *)dbuf->ptr + down_offset,
                (const float *)((const uint8_t *)mbuf->ptr + mid_offset),
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                experts,
                topk,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_moe_q8k_full_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *scratch_xq,
        uint64_t scratch_xq_offset,
        void *scratch_midq,
        uint64_t scratch_midq_offset,
        void *scratch_mid,
        uint64_t scratch_mid_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_iq2xxs || !up_iq2xxs || !down_q2k || !input ||
        !indices || !router_weights || !scratch_xq || !scratch_midq || !scratch_mid || !out ||
        experts == 0 || topk != 6u || topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_iq2xxs;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_iq2xxs;
    const axiom_cuda_buffer *dbuf = (const axiom_cuda_buffer *)down_q2k;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    const axiom_cuda_buffer *idxbuf = (const axiom_cuda_buffer *)indices;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_weights;
    axiom_cuda_buffer *xqbuf = (axiom_cuda_buffer *)scratch_xq;
    axiom_cuda_buffer *midqbuf = (axiom_cuda_buffer *)scratch_midq;
    axiom_cuda_buffer *midbuf = (axiom_cuda_buffer *)scratch_mid;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const uint32_t x_blocks = hidden / 256u;
    const uint32_t mid_blocks = expert_hidden / 256u;
    const uint32_t gate_q8t32_pack_midq =
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_Q8T32_PACK_MIDQ") &&
            hidden == 4096u &&
            expert_hidden == 2048u;
    axiom_cuda_q8_k_block *xq = (axiom_cuda_q8_k_block *)((uint8_t *)xqbuf->ptr + scratch_xq_offset);
    axiom_cuda_q8_k_block *midq = (axiom_cuda_q8_k_block *)((uint8_t *)midqbuf->ptr + scratch_midq_offset);
    float *mid = (float *)((uint8_t *)midbuf->ptr + scratch_mid_offset);

    const uint32_t input_prepacked =
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_INPUT_PREPACKED") &&
            !gate_q8t32_pack_midq;
    const uint32_t input_scale_prepacked =
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_INPUT_SCALE_PREPACKED") &&
            !gate_q8t32_pack_midq;

    double t_moe = axiom_cuda_moe_q8k_internal_start();
    if (input_prepacked) {
        if (axiom_cuda_moe_q8k_internal_timing_enabled()) {
            FILE *fp = axiom_cuda_moe_q8k_internal_log_file();
            std::fprintf(fp,
                    "{\"event\":\"moe_q8k_internal\",\"name\":\"pack_input_q8k_prepacked\","
                    "\"topk\":%u,\"hidden\":%u,\"expert_hidden\":%u,"
                    "\"elapsed_ms\":0.000000,\"cuda_status\":0}\n",
                    topk, hidden, expert_hidden);
        }
    } else if (input_scale_prepacked) {
        axiom_q8_k_pack_f32_with_d_kernel<<<(int)x_blocks, 256>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq,
                x_blocks);
    } else if (gate_q8t32_pack_midq) {
        axiom_cuda_q8_0_tile32_block *xq32 =
                (axiom_cuda_q8_0_tile32_block *)((uint8_t *)xqbuf->ptr + scratch_xq_offset);
        const uint32_t t32_blocks = hidden / 32u;
        axiom_q8_0_tile32_pack_f32_kernel<<<(int)t32_blocks, 32>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                xq32,
                t32_blocks);
    } else {
        if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_INPUT_NO_BSUMS_FAST")) {
            axiom_q8_k_pack_f32_input_nobsums_fast_kernel<<<(int)x_blocks, 256>>>(
                    (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                    xq,
                    x_blocks);
        } else if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_INPUT_NO_BSUMS")) {
            axiom_q8_k_pack_f32_input_nobsums_kernel<<<(int)x_blocks, 256>>>(
                    (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                    xq,
                    x_blocks);
        } else if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_PACK_RN_INTRIN")) {
            axiom_q8_k_pack_f32_rn_kernel<<<(int)x_blocks, 256>>>(
                    (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                    xq,
                    x_blocks);
        } else if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_PACK_FAST")) {
            axiom_q8_k_pack_f32_fast_kernel<<<(int)x_blocks, 256>>>(
                    (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                    xq,
                    x_blocks);
        } else {
            axiom_q8_k_pack_f32_kernel<<<(int)x_blocks, 256>>>(
                    (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                    xq,
                    x_blocks);
        }
    }
    if (!input_prepacked) {
        err = cudaGetLastError();
        err = axiom_cuda_moe_q8k_internal_finish(
                input_scale_prepacked ? "pack_input_q8k_with_d" :
                gate_q8t32_pack_midq ? "pack_input_q8t32" :
                        (axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_INPUT_NO_BSUMS_FAST") ?
                                "pack_input_q8k_nobsums_fast" :
                        (axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_INPUT_NO_BSUMS") ?
                                "pack_input_q8k_nobsums" :
                        (axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_PACK_RN_INTRIN") ?
                                "pack_input_q8k_rn_intrin" :
                        (axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_PACK_FAST") ?
                                "pack_input_q8k_fast" : "pack_input_q8k")))),
                topk, hidden, expert_hidden, t_moe, err);
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }

    const uint32_t gate_pack_midq =
            !gate_q8t32_pack_midq &&
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ") &&
            expert_hidden == 2048u;
    const uint32_t gate_pack_midq_rowstage =
            gate_pack_midq &&
            hidden == 4096u &&
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_ROWSTAGE");
    const uint32_t gate_pack_midq_rowstage8 =
            gate_pack_midq_rowstage &&
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_ROWSTAGE8");
    const uint32_t gate_pack_midq_rowstage32 =
            gate_pack_midq_rowstage &&
            !gate_pack_midq_rowstage8 &&
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_ROWSTAGE32");
    const uint32_t gate_pack_midq_a68 =
            gate_pack_midq_rowstage &&
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_IQ2_A68");
    const uint32_t gate_pack_midq_rowstage_split_floatmid =
            gate_pack_midq_rowstage &&
            hidden == 4096u &&
            expert_hidden == 2048u &&
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_ROWSTAGE_SPLIT_FLOATMID");
    if (gate_pack_midq_rowstage_split_floatmid) {
        const uint64_t xq_start = (uint64_t)(uintptr_t)xq;
        const uint64_t xq_end = xq_start + (uint64_t)x_blocks * sizeof(axiom_cuda_q8_k_block);
        const uint64_t mid_start = (uint64_t)(uintptr_t)mid;
        const uint64_t mid_end = mid_start + (uint64_t)topk * expert_hidden * sizeof(float);
        if (xq_start < mid_end && mid_start < xq_end) return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t gate_block_threads = 256u;
    if (!gate_pack_midq_rowstage && gate_pack_midq && axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_BLOCK512")) {
        gate_block_threads = 512u;
    }
    const dim3 gate_block(gate_block_threads, 1, 1);
    const dim3 gate_grid(
            (unsigned int)((expert_hidden + ((gate_pack_midq || gate_q8t32_pack_midq) ? 255u : 127u)) /
                    ((gate_pack_midq || gate_q8t32_pack_midq) ? 256u : 128u)),
            (unsigned int)topk,
            1u);
    t_moe = axiom_cuda_moe_q8k_internal_start();
    if (gate_q8t32_pack_midq) {
        axiom_deepseek_moe_gate_up_indexed_q8t32_weighted_pack_midq_kernel<<<gate_grid, gate_block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                (const axiom_cuda_q8_0_tile32_block *)((uint8_t *)xqbuf->ptr + scratch_xq_offset),
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                midq,
                experts,
                topk,
                hidden,
                expert_hidden);
    } else if (gate_pack_midq_rowstage_split_floatmid) {
        const uint32_t iq2_block_stride = gate_pack_midq_a68 ? 68u : 66u;
        const uint32_t iq2_aux_offset = gate_pack_midq_a68 ? 4u : 2u;
        const uint32_t row_bytes = (hidden / 256u) * iq2_block_stride;
        const uint32_t rowstage_rows =
                gate_pack_midq_rowstage32 ? 32u : (gate_pack_midq_rowstage8 ? 8u : 16u);
        const size_t rowstage_shared = (size_t)2u * rowstage_rows * row_bytes;
        if (rowstage_rows > 16u) {
            err = cudaFuncSetAttribute(
                    axiom_deepseek_moe_gate_up_indexed_q8k_weighted_mid_rowstage_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)rowstage_shared);
            if (err != cudaSuccess) return axiom_cuda_status(err);
        }
        const uint32_t split_rounds_per_block = (uint32_t)axiom_cuda_env_int(
                "AXIOM_DS4_MOE_GATEUP_ROWSTAGE_SPLIT_FLOATMID_RPB", 1, 1, 16, 1);
        const dim3 split_gate_grid(
                (unsigned int)mid_blocks,
                (unsigned int)topk,
                (unsigned int)(((256u / rowstage_rows) + split_rounds_per_block - 1u) /
                        split_rounds_per_block));
        axiom_deepseek_moe_gate_up_indexed_q8k_weighted_mid_rowstage_kernel<<<split_gate_grid, gate_block, rowstage_shared>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                mid,
                experts,
                topk,
                hidden,
                expert_hidden,
                axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_NOLUT") ? 1u : 0u,
                iq2_block_stride,
                iq2_aux_offset,
                rowstage_rows,
                split_rounds_per_block,
                axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_LUT_REUSE") ? 1u : 0u);
    } else if (gate_pack_midq_rowstage) {
        const uint32_t iq2_block_stride = gate_pack_midq_a68 ? 68u : 66u;
        const uint32_t iq2_aux_offset = gate_pack_midq_a68 ? 4u : 2u;
        const uint32_t row_bytes = (hidden / 256u) * iq2_block_stride;
        const uint32_t rowstage_rows =
                gate_pack_midq_rowstage32 ? 32u : (gate_pack_midq_rowstage8 ? 8u : 16u);
        const size_t rowstage_shared = (size_t)2u * rowstage_rows * row_bytes;
        if (rowstage_rows > 16u) {
            err = cudaFuncSetAttribute(
                    axiom_deepseek_moe_gate_up_indexed_q8k_weighted_pack_midq_rowstage_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)rowstage_shared);
            if (err != cudaSuccess) return axiom_cuda_status(err);
        }
        axiom_deepseek_moe_gate_up_indexed_q8k_weighted_pack_midq_rowstage_kernel<<<gate_grid, gate_block, rowstage_shared>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                midq,
                experts,
                topk,
                hidden,
                expert_hidden,
                axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_NOLUT") ? 1u : 0u,
                axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_FASTREDUCE") ? 1u : 0u,
                iq2_block_stride,
                iq2_aux_offset,
                rowstage_rows,
                axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_LUT_REUSE") ? 1u : 0u);
    } else if (gate_pack_midq) {
        axiom_deepseek_moe_gate_up_indexed_q8k_weighted_pack_midq_kernel<<<gate_grid, gate_block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                midq,
                experts,
                topk,
                hidden,
                expert_hidden,
                axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_NOLUT") ? 1u : 0u,
                axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_FASTREDUCE") ? 1u : 0u);
    } else if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_Q8K_WEIGHTED_OLD")) {
        axiom_deepseek_moe_gate_up_indexed_q8k_weighted_qwarp32_kernel<<<gate_grid, gate_block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                mid,
                experts,
                topk,
                hidden,
                expert_hidden);
    } else {
        axiom_deepseek_moe_gate_up_indexed_q8k_weighted_block_qwarp32_kernel<<<gate_grid, gate_block>>>(
                (const uint8_t *)gbuf->ptr + gate_offset,
                (const uint8_t *)ubuf->ptr + up_offset,
                xq,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
                mid,
                experts,
                topk,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    err = axiom_cuda_moe_q8k_internal_finish(
            gate_q8t32_pack_midq ? "gateup_weighted_q8t32_pack_midq" :
                    (gate_pack_midq_rowstage_split_floatmid ? "gateup_weighted_mid_rowstage_split_floatmid" :
                    (gate_pack_midq_a68 ? "gateup_weighted_pack_midq_rowstage_a68" :
                    (gate_pack_midq_rowstage32 ? "gateup_weighted_pack_midq_rowstage32" :
                    (gate_pack_midq_rowstage8 ? "gateup_weighted_pack_midq_rowstage8" :
                    (gate_pack_midq_rowstage ? "gateup_weighted_pack_midq_rowstage" :
                    (gate_pack_midq ? "gateup_weighted_pack_midq" :
                    (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_Q8K_WEIGHTED_OLD") ?
                            "gateup_weighted_old" : "gateup_weighted_block"))))))),
            topk, hidden, expert_hidden, t_moe, err);
    if (err != cudaSuccess) return axiom_cuda_status(err);

    if (gate_pack_midq_rowstage_split_floatmid) {
        const dim3 midq_grid(mid_blocks, topk, 1);
        t_moe = axiom_cuda_moe_q8k_internal_start();
        axiom_q8_k_pack_rows_f32_fast_kernel<<<midq_grid, 256>>>(
                midq,
                mid,
                expert_hidden,
                topk);
        err = cudaGetLastError();
        err = axiom_cuda_moe_q8k_internal_finish(
                "pack_mid_q8k_split_floatmid_fast", topk, hidden, expert_hidden, t_moe, err);
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }

    if (!gate_pack_midq && !gate_q8t32_pack_midq) {
        const dim3 midq_grid(mid_blocks, topk, 1);
        t_moe = axiom_cuda_moe_q8k_internal_start();
        axiom_q8_k_pack_rows_f32_kernel<<<midq_grid, 256>>>(
                midq,
                mid,
                expert_hidden,
                topk);
        err = cudaGetLastError();
        err = axiom_cuda_moe_q8k_internal_finish(
                "pack_mid_q8k", topk, hidden, expert_hidden, t_moe, err);
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }

    const dim3 down_grid((unsigned int)((hidden + 31u) / 32u), 1u, 1u);
    t_moe = axiom_cuda_moe_q8k_internal_start();
    const int down_const_mid =
            (gate_pack_midq || gate_q8t32_pack_midq) &&
            expert_hidden == 2048u &&
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_Q8K_CONST_MID");
    if (down_const_mid) {
        err = cudaMemcpyToSymbol(
                axiom_moe_midq_const,
                midq,
                48u * sizeof(axiom_cuda_q8_k_block),
                0,
                cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_deepseek_moe_down_q8k_sum6_qwarp32_cmid_kernel<<<down_grid, 256>>>(
                (const uint8_t *)dbuf->ptr + down_offset,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                experts,
                hidden,
                expert_hidden);
    } else if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_Q8K_SHARED_MID") && expert_hidden == 2048u) {
        axiom_deepseek_moe_down_q8k_sum6_qwarp32_smid_kernel<<<down_grid, 256>>>(
                (const uint8_t *)dbuf->ptr + down_offset,
                midq,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                experts,
                hidden,
                expert_hidden);
    } else {
        axiom_deepseek_moe_down_q8k_sum6_qwarp32_kernel<<<down_grid, 256>>>(
                (const uint8_t *)dbuf->ptr + down_offset,
                midq,
                (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                experts,
                hidden,
                expert_hidden);
    }
    err = cudaGetLastError();
    err = axiom_cuda_moe_q8k_internal_finish(
            (down_const_mid ? "down_q8k_cmid" :
                    ((axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_Q8K_SHARED_MID") && expert_hidden == 2048u) ?
                            "down_q8k_smid" : "down_q8k")),
            topk, hidden, expert_hidden, t_moe, err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_moe_q8k_full2_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *indices0,
        uint64_t indices0_offset,
        const void *router_weights0,
        uint64_t router_weights0_offset,
        void *scratch_xq0,
        uint64_t scratch_xq0_offset,
        void *scratch_midq0,
        uint64_t scratch_midq0_offset,
        void *scratch_mid0,
        uint64_t scratch_mid0_offset,
        void *out0,
        uint64_t out0_offset,
        const void *input1,
        uint64_t input1_offset,
        const void *indices1,
        uint64_t indices1_offset,
        const void *router_weights1,
        uint64_t router_weights1_offset,
        void *scratch_xq1,
        uint64_t scratch_xq1_offset,
        void *scratch_midq1,
        uint64_t scratch_midq1_offset,
        void *scratch_mid1,
        uint64_t scratch_mid1_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_iq2xxs || !up_iq2xxs || !down_q2k ||
        !input0 || !indices0 || !router_weights0 || !scratch_xq0 || !scratch_midq0 || !scratch_mid0 || !out0 ||
        !input1 || !indices1 || !router_weights1 || !scratch_xq1 || !scratch_midq1 || !scratch_mid1 || !out1 ||
        experts == 0 || topk != 6u || topk > experts || hidden != 4096u || expert_hidden != 2048u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    if (axiom_cuda_launch_async_enabled() ||
        !axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_INPUT_PREPACKED") ||
        !axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ") ||
        !axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_ROWSTAGE") ||
        !axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_ROWSTAGE_SPLIT_FLOATMID") ||
        !axiom_cuda_env_enabled("AXIOM_DS4_MOE_DOWN_Q8K_SHARED_MID")) {
        int rc = axiom_cuda_deepseek_moe_q8k_full_device(
                cuda_runtime, gate_iq2xxs, gate_offset, up_iq2xxs, up_offset,
                down_q2k, down_offset, input0, input0_offset, indices0, indices0_offset,
                router_weights0, router_weights0_offset, scratch_xq0, scratch_xq0_offset,
                scratch_midq0, scratch_midq0_offset, scratch_mid0, scratch_mid0_offset,
                out0, out0_offset, experts, topk, hidden, expert_hidden);
        if (rc != AXIOM_OK) return rc;
        return axiom_cuda_deepseek_moe_q8k_full_device(
                cuda_runtime, gate_iq2xxs, gate_offset, up_iq2xxs, up_offset,
                down_q2k, down_offset, input1, input1_offset, indices1, indices1_offset,
                router_weights1, router_weights1_offset, scratch_xq1, scratch_xq1_offset,
                scratch_midq1, scratch_midq1_offset, scratch_mid1, scratch_mid1_offset,
                out1, out1_offset, experts, topk, hidden, expert_hidden);
    }

    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    cudaStream_t s0 = nullptr;
    cudaStream_t s1 = nullptr;
    int src = axiom_cuda_moe_streams(runtime, &s0, &s1);
    if (src != AXIOM_OK) return src;

    const uint32_t mid_blocks = expert_hidden / 256u;
    const uint32_t gate_block_threads = 256u;
    const dim3 gate_block(gate_block_threads, 1, 1);
    const uint32_t rowstage_rows =
            axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_ROWSTAGE32") ? 32u :
            (axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_ROWSTAGE8") ? 8u : 16u);
    const uint32_t split_rounds_per_block = (uint32_t)axiom_cuda_env_int(
            "AXIOM_DS4_MOE_GATEUP_ROWSTAGE_SPLIT_FLOATMID_RPB", 1, 1, 16, 1);
    const uint32_t iq2_block_stride = axiom_cuda_env_enabled("AXIOM_DS4_MOE_IQ2_A68") ? 68u : 66u;
    const uint32_t iq2_aux_offset = axiom_cuda_env_enabled("AXIOM_DS4_MOE_IQ2_A68") ? 4u : 2u;
    const uint32_t row_bytes = (hidden / 256u) * iq2_block_stride;
    const size_t rowstage_shared = (size_t)2u * rowstage_rows * row_bytes;
    if (rowstage_rows > 16u) {
        err = cudaFuncSetAttribute(
                axiom_deepseek_moe_gate_up_indexed_q8k_weighted_mid_rowstage_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)rowstage_shared);
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }
    const dim3 split_gate_grid(
            (unsigned int)mid_blocks,
            (unsigned int)topk,
            (unsigned int)(((256u / rowstage_rows) + split_rounds_per_block - 1u) /
                    split_rounds_per_block));
    const dim3 midq_grid(mid_blocks, topk, 1);
    const dim3 down_grid((unsigned int)((hidden + 31u) / 32u), 1u, 1u);
    const uint32_t no_lut = axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_PACK_MIDQ_NOLUT") ? 1u : 0u;
    const uint32_t lut_reuse = axiom_cuda_env_enabled("AXIOM_DS4_MOE_GATEUP_LUT_REUSE") ? 1u : 0u;

    auto launch_lane = [&](cudaStream_t stream,
                           const void *input, uint64_t input_offset,
                           const void *indices, uint64_t indices_offset,
                           const void *router_weights, uint64_t router_weights_offset,
                           void *scratch_midq, uint64_t scratch_midq_offset,
                           void *scratch_mid, uint64_t scratch_mid_offset,
                           void *out, uint64_t out_offset) {
        axiom_cuda_q8_k_block *xq = (axiom_cuda_q8_k_block *)((uint8_t *)((axiom_cuda_buffer *)input)->ptr + input_offset);
        axiom_cuda_q8_k_block *midq = (axiom_cuda_q8_k_block *)((uint8_t *)((axiom_cuda_buffer *)scratch_midq)->ptr + scratch_midq_offset);
        float *mid = (float *)((uint8_t *)((axiom_cuda_buffer *)scratch_mid)->ptr + scratch_mid_offset);
        axiom_deepseek_moe_gate_up_indexed_q8k_weighted_mid_rowstage_kernel<<<
                split_gate_grid, gate_block, rowstage_shared, stream>>>(
                (const uint8_t *)((const axiom_cuda_buffer *)gate_iq2xxs)->ptr + gate_offset,
                (const uint8_t *)((const axiom_cuda_buffer *)up_iq2xxs)->ptr + up_offset,
                xq,
                (const uint32_t *)((const uint8_t *)((const axiom_cuda_buffer *)indices)->ptr + indices_offset),
                (const float *)((const uint8_t *)((const axiom_cuda_buffer *)router_weights)->ptr + router_weights_offset),
                mid,
                experts,
                topk,
                hidden,
                expert_hidden,
                no_lut,
                iq2_block_stride,
                iq2_aux_offset,
                rowstage_rows,
                split_rounds_per_block,
                lut_reuse);
        axiom_q8_k_pack_rows_f32_fast_kernel<<<midq_grid, 256, 0, stream>>>(
                midq,
                mid,
                expert_hidden,
                topk);
        axiom_deepseek_moe_down_q8k_sum6_qwarp32_smid_kernel<<<down_grid, 256, 0, stream>>>(
                (const uint8_t *)((const axiom_cuda_buffer *)down_q2k)->ptr + down_offset,
                midq,
                (const uint32_t *)((const uint8_t *)((const axiom_cuda_buffer *)indices)->ptr + indices_offset),
                (float *)((uint8_t *)((axiom_cuda_buffer *)out)->ptr + out_offset),
                experts,
                hidden,
                expert_hidden);
    };

    launch_lane(s0, scratch_xq0, scratch_xq0_offset, indices0, indices0_offset,
                router_weights0, router_weights0_offset, scratch_midq0, scratch_midq0_offset,
                scratch_mid0, scratch_mid0_offset, out0, out0_offset);
    launch_lane(s1, scratch_xq1, scratch_xq1_offset, indices1, indices1_offset,
                router_weights1, router_weights1_offset, scratch_midq1, scratch_midq1_offset,
                scratch_mid1, scratch_mid1_offset, out1, out1_offset);

    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    err = cudaStreamSynchronize(s0);
    if (err != cudaSuccess) return axiom_cuda_status(err);
    err = cudaStreamSynchronize(s1);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_deepseek_moe_q4k_full_device(
        void *cuda_runtime,
        const void *gate_q4k,
        uint64_t gate_offset,
        const void *up_q4k,
        uint64_t up_offset,
        const void *down_q4k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *scratch_xq,
        uint64_t scratch_xq_offset,
        void *scratch_midq,
        uint64_t scratch_midq_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    if (!cuda_runtime || !gate_q4k || !up_q4k || !down_q4k || !input ||
        !indices || !router_weights || !scratch_xq || !scratch_midq || !out ||
        experts == 0 || topk != 6u || topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u || expert_hidden != 2048u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate_q4k;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up_q4k;
    const axiom_cuda_buffer *dbuf = (const axiom_cuda_buffer *)down_q4k;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    const axiom_cuda_buffer *idxbuf = (const axiom_cuda_buffer *)indices;
    const axiom_cuda_buffer *rbuf = (const axiom_cuda_buffer *)router_weights;
    axiom_cuda_buffer *xqbuf = (axiom_cuda_buffer *)scratch_xq;
    axiom_cuda_buffer *midqbuf = (axiom_cuda_buffer *)scratch_midq;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const uint32_t x_blocks = hidden / 256u;
    const uint32_t mid_blocks = expert_hidden / 256u;
    axiom_cuda_q8_k_block *xq = (axiom_cuda_q8_k_block *)((uint8_t *)xqbuf->ptr + scratch_xq_offset);
    axiom_cuda_q8_k_block *midq = (axiom_cuda_q8_k_block *)((uint8_t *)midqbuf->ptr + scratch_midq_offset);

    double t_moe = axiom_cuda_moe_q8k_internal_start();
    axiom_q8_k_pack_f32_kernel<<<(int)x_blocks, 256>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            xq,
            x_blocks);
    err = cudaGetLastError();
    err = axiom_cuda_moe_q8k_internal_finish(
            "pack_input_q8k_q4full", topk, hidden, expert_hidden, t_moe, err);
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const dim3 gate_grid((unsigned int)mid_blocks, (unsigned int)topk, 1u);
    const dim3 gate_block(256u, 1u, 1u);
    t_moe = axiom_cuda_moe_q8k_internal_start();
    axiom_deepseek_moe_gate_up_indexed_q4k_weighted_pack_midq_kernel<<<gate_grid, gate_block>>>(
            (const uint8_t *)gbuf->ptr + gate_offset,
            (const uint8_t *)ubuf->ptr + up_offset,
            xq,
            (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
            (const float *)((const uint8_t *)rbuf->ptr + router_weights_offset),
            midq,
            experts,
            topk,
            hidden,
            expert_hidden,
            axiom_cuda_env_enabled("AXIOM_MTP_Q4K_FULL_FASTREDUCE") ? 1u : 0u);
    err = cudaGetLastError();
    err = axiom_cuda_moe_q8k_internal_finish(
            "gateup_weighted_q4k_pack_midq", topk, hidden, expert_hidden, t_moe, err);
    if (err != cudaSuccess) return axiom_cuda_status(err);

    const dim3 down_grid((unsigned int)((hidden + 31u) / 32u), 1u, 1u);
    t_moe = axiom_cuda_moe_q8k_internal_start();
    axiom_deepseek_moe_down_q4k_sum6_qwarp32_smid_kernel<<<down_grid, 256>>>(
            (const uint8_t *)dbuf->ptr + down_offset,
            midq,
            (const uint32_t *)((const uint8_t *)idxbuf->ptr + indices_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            experts,
            hidden,
            expert_hidden);
    err = cudaGetLastError();
    err = axiom_cuda_moe_q8k_internal_finish(
            "down_q4k_smid", topk, hidden, expert_hidden, t_moe, err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_q8_0_matvec_f32(
        void *cuda_runtime,
        const uint8_t *weight_q8_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q8_host || !input_host || !out_host ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    uint8_t *weight_dev = nullptr;
    float *input_dev = nullptr;
    float *out_dev = nullptr;
    const size_t weight_bytes = (size_t)rows * (cols / 32u) * 34u;
    const size_t input_bytes = (size_t)cols * sizeof(float);
    const size_t out_bytes = (size_t)rows * sizeof(float);
    err = cudaMalloc(&weight_dev, weight_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&input_dev, input_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, out_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(weight_dev, weight_q8_host, weight_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(input_dev, input_host, input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int block = 128;
        const int grid = (int)((rows + block - 1u) / block);
        axiom_q8_0_matvec_f32_kernel<<<grid, block>>>(weight_dev, input_dev, out_dev, rows, cols);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(weight_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(weight_dev);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_q2_k_matvec_f32(
        void *cuda_runtime,
        const uint8_t *weight_q2k_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_q2k_host || !input_host || !out_host ||
        rows == 0 || cols == 0 || (cols % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    uint8_t *weight_dev = nullptr;
    float *input_dev = nullptr;
    float *out_dev = nullptr;
    const size_t weight_bytes = (size_t)rows * (cols / 256u) * 84u;
    const size_t input_bytes = (size_t)cols * sizeof(float);
    const size_t out_bytes = (size_t)rows * sizeof(float);
    err = cudaMalloc(&weight_dev, weight_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&input_dev, input_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, out_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(weight_dev, weight_q2k_host, weight_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(input_dev, input_host, input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int block = 128;
        const int grid = (int)((rows + block - 1u) / block);
        axiom_q2_k_matvec_f32_kernel<<<grid, block>>>(weight_dev, input_dev, out_dev, rows, cols);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(weight_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(weight_dev);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_iq2_xxs_matvec_f32(
        void *cuda_runtime,
        const uint8_t *weight_iq2xxs_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_iq2xxs_host || !input_host || !out_host ||
        rows == 0 || cols == 0 || (cols % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    uint8_t *weight_dev = nullptr;
    float *input_dev = nullptr;
    float *out_dev = nullptr;
    const size_t weight_bytes = (size_t)rows * (cols / 256u) * 66u;
    const size_t input_bytes = (size_t)cols * sizeof(float);
    const size_t out_bytes = (size_t)rows * sizeof(float);
    err = cudaMalloc(&weight_dev, weight_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&input_dev, input_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, out_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(weight_dev, weight_iq2xxs_host, weight_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(input_dev, input_host, input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int block = 128;
        const int grid = (int)((rows + block - 1u) / block);
        axiom_iq2_xxs_matvec_f32_kernel<<<grid, block>>>(weight_dev, input_dev, out_dev, rows, cols);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(weight_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(weight_dev);
    return axiom_cuda_status(err);
}

__global__ static void axiom_bf16_matvec_f32_kernel(
        const uint16_t *weight,
        const uint16_t *bias,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float acc = 0.0f;
    if (bias) {
        acc = __uint_as_float(((uint32_t)bias[row]) << 16u);
    }
    const uint16_t *w = weight + (uint64_t)row * cols;
    for (uint32_t col = 0; col < cols; ++col) {
        const float wf = __uint_as_float(((uint32_t)w[col]) << 16u);
        acc += wf * input[col];
    }
    out[row] = acc;
}

extern "C" int axiom_cuda_bf16_matvec_f32(
        void *cuda_runtime,
        const uint16_t *weight_bf16_host,
        const uint16_t *bias_bf16_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_bf16_host || !input_host || !out_host ||
        rows == 0 || cols == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    uint16_t *weight_dev = nullptr;
    uint16_t *bias_dev = nullptr;
    float *input_dev = nullptr;
    float *out_dev = nullptr;
    const size_t weight_bytes = (size_t)rows * cols * sizeof(uint16_t);
    const size_t bias_bytes = (size_t)rows * sizeof(uint16_t);
    const size_t input_bytes = (size_t)cols * sizeof(float);
    const size_t out_bytes = (size_t)rows * sizeof(float);

    err = cudaMalloc(&weight_dev, weight_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&input_dev, input_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, out_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(weight_dev, weight_bf16_host, weight_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(input_dev, input_host, input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    if (bias_bf16_host) {
        err = cudaMalloc(&bias_dev, bias_bytes);
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(bias_dev, bias_bf16_host, bias_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    {
        const int block = 128;
        const int grid = (int)((rows + block - 1) / block);
        axiom_bf16_matvec_f32_kernel<<<grid, block>>>(
                weight_dev,
                bias_dev,
                input_dev,
                out_dev,
                rows,
                cols);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(bias_dev);
    cudaFree(weight_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(bias_dev);
    cudaFree(weight_dev);
    return axiom_cuda_status(err);
}

/* P4-C: DEVICE-RESIDENT-weight variant of axiom_cuda_bf16_matvec_f32. The
 * weight bytes were uploaded ONCE (axiom_model_tensor_device_resident) into an
 * axiom_cuda_buffer owned by the model; this entry launches the SAME
 * axiom_bf16_matvec_f32_kernel with the SAME launch config (block=128) — only
 * the per-call weight cudaMalloc + H2D upload + cudaFree are skipped. The
 * kernel input state (weight bytes, input bytes, bias bytes, rows, cols) is
 * byte-identical to the per-op path, so the output is bit-identical: same
 * math, same order — only the weight bytes' RESIDENCE changes. Bias (when
 * present) keeps the per-op upload: it was never the cost and the bytes are
 * identical either way. weight_byte_offset supports rank-3 slice views into a
 * whole resident expert slab. */
extern "C" int axiom_cuda_bf16_matvec_f32_resident(
        void *cuda_runtime,
        const void *weight_buffer_opaque,
        uint64_t weight_byte_offset,
        const uint16_t *bias_bf16_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_buffer_opaque || !input_host || !out_host ||
        rows == 0 || cols == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *weight_buffer =
            (const axiom_cuda_buffer *)weight_buffer_opaque;
    const uint64_t weight_bytes64 = (uint64_t)rows * cols * sizeof(uint16_t);
    if (weight_buffer->device != runtime->device ||
        weight_byte_offset > weight_buffer->bytes ||
        weight_bytes64 > weight_buffer->bytes - weight_byte_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    const uint16_t *weight_dev =
            (const uint16_t *)((const uint8_t *)weight_buffer->ptr + weight_byte_offset);
    uint16_t *bias_dev = nullptr;
    float *input_dev = nullptr;
    float *out_dev = nullptr;
    const size_t bias_bytes = (size_t)rows * sizeof(uint16_t);
    const size_t input_bytes = (size_t)cols * sizeof(float);
    const size_t out_bytes = (size_t)rows * sizeof(float);

    err = cudaMalloc(&input_dev, input_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, out_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(input_dev, input_host, input_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    if (bias_bf16_host) {
        err = cudaMalloc(&bias_dev, bias_bytes);
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(bias_dev, bias_bf16_host, bias_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    {
        const int block = 128;
        const int grid = (int)((rows + block - 1) / block);
        axiom_bf16_matvec_f32_kernel<<<grid, block>>>(
                weight_dev,
                bias_dev,
                input_dev,
                out_dev,
                rows,
                cols);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(bias_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(bias_dev);
    return axiom_cuda_status(err);
}

__global__ static void axiom_f16_matvec_f32_device_warp_kernel(
        const uint16_t *weight,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint16_t *w = weight + (uint64_t)row * cols;
    float acc = 0.0f;
    for (uint32_t col = lane; col < cols; col += 32u) {
        acc += axiom_dev_f16_to_f32(w[col]) * input[col];
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void axiom_f16_dual_matvec_f32_device_warp_kernel(
        const uint16_t *__restrict__ weight_a,
        const uint16_t *__restrict__ weight_b,
        const float *__restrict__ input,
        float *__restrict__ out_a,
        float *__restrict__ out_b,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows_a && row >= rows_b) return;
    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (uint32_t col = lane; col < cols; col += 32u) {
        const float x = input[col];
        if (row < rows_a) {
            acc_a += axiom_dev_f16_to_f32(weight_a[(uint64_t)row * cols + col]) * x;
        }
        if (row < rows_b) {
            acc_b += axiom_dev_f16_to_f32(weight_b[(uint64_t)row * cols + col]) * x;
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc_a += __shfl_down_sync(0xffffffffu, acc_a, offset);
        acc_b += __shfl_down_sync(0xffffffffu, acc_b, offset);
    }
    if (lane == 0u) {
        if (row < rows_a) out_a[row] = acc_a;
        if (row < rows_b) out_b[row] = acc_b;
    }
}

extern "C" int axiom_cuda_f16_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_f16,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (!cuda_runtime || !weight_f16 || !input || !out || rows == 0 || cols == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_f16;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int warps_per_block = block / 32;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_f16_matvec_f32_device_warp_kernel<<<grid, block>>>(
            (const uint16_t *)((const uint8_t *)wbuf->ptr + weight_offset),
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            rows,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_f16_dual_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_a_f16,
        uint64_t weight_a_offset,
        const void *weight_b_f16,
        uint64_t weight_b_offset,
        const void *input,
        uint64_t input_offset,
        void *out_a,
        uint64_t out_a_offset,
        void *out_b,
        uint64_t out_b_offset,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols) {
    if (!cuda_runtime || !weight_a_f16 || !weight_b_f16 || !input || !out_a || !out_b ||
        rows_a == 0 || rows_b == 0 || cols == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wabuf = (const axiom_cuda_buffer *)weight_a_f16;
    const axiom_cuda_buffer *wbbuf = (const axiom_cuda_buffer *)weight_b_f16;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *oabuf = (axiom_cuda_buffer *)out_a;
    axiom_cuda_buffer *obbuf = (axiom_cuda_buffer *)out_b;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int warps_per_block = block / 32;
    const uint32_t rows = rows_a > rows_b ? rows_a : rows_b;
    const int grid = (int)((rows + (uint32_t)warps_per_block - 1u) / (uint32_t)warps_per_block);
    axiom_f16_dual_matvec_f32_device_warp_kernel<<<grid, block>>>(
            (const uint16_t *)((const uint8_t *)wabuf->ptr + weight_a_offset),
            (const uint16_t *)((const uint8_t *)wbbuf->ptr + weight_b_offset),
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)oabuf->ptr + out_a_offset),
            (float *)((uint8_t *)obbuf->ptr + out_b_offset),
            rows_a,
            rows_b,
            cols);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_f16_embedding_streams_device_kernel(
        const uint16_t *embedding,
        float *out,
        uint32_t token_id,
        uint32_t hidden,
        uint32_t streams) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)hidden * streams;
    if (idx >= total) return;
    const uint32_t h = (uint32_t)(idx % hidden);
    out[idx] = axiom_dev_f16_to_f32(embedding[(uint64_t)token_id * hidden + h]);
}

extern "C" int axiom_cuda_f16_embedding_streams_device(
        void *cuda_runtime,
        const void *embedding_f16,
        uint64_t embedding_offset,
        void *out_streams,
        uint64_t out_offset,
        uint32_t token_id,
        uint32_t hidden,
        uint32_t streams) {
    if (!cuda_runtime || !embedding_f16 || !out_streams || hidden == 0 || streams == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ebuf = (const axiom_cuda_buffer *)embedding_f16;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out_streams;
    const uint64_t token_rows = (uint64_t)token_id + 1u;
    if ((uint64_t)hidden > UINT64_MAX / token_rows) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t row_elems = token_rows * (uint64_t)hidden;
    if (row_elems > UINT64_MAX / sizeof(uint16_t)) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t row_bytes = row_elems * sizeof(uint16_t);
    if ((uint64_t)hidden > UINT64_MAX / (uint64_t)streams) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t out_elems = (uint64_t)hidden * (uint64_t)streams;
    if (out_elems > UINT64_MAX / sizeof(float)) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t out_bytes = out_elems * sizeof(float);
    if (embedding_offset > ebuf->bytes || row_bytes > ebuf->bytes - embedding_offset ||
        out_offset > obuf->bytes || out_bytes > obuf->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint64_t total = (uint64_t)hidden * streams;
    const uint32_t block = 256;
    const uint32_t grid = (uint32_t)((total + block - 1) / block);
    axiom_f16_embedding_streams_device_kernel<<<grid, block>>>(
            (const uint16_t *)((const uint8_t *)ebuf->ptr + embedding_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            token_id,
            hidden,
            streams);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_rmsnorm_f32_kernel(
        const uint16_t *weight,
        const float *input,
        float *out,
        uint32_t count,
        float eps) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float ss = 0.0f;
    for (uint32_t i = 0; i < count; ++i) ss += input[i] * input[i];
    const float inv = rsqrtf(ss / (float)count + eps);
    for (uint32_t i = 0; i < count; ++i) {
        const float w = __uint_as_float(((uint32_t)weight[i]) << 16u);
        out[i] = input[i] * inv * w;
    }
}

__global__ static void axiom_rmsnorm_f32_device_kernel(
        const float *weight,
        const float *input,
        float *out,
        uint32_t count,
        float eps) {
    __shared__ float sums[256];
    const uint32_t tid = threadIdx.x;
    float ss = 0.0f;
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        ss += input[i] * input[i];
    }
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / (float)count + eps);
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        out[i] = input[i] * inv * weight[i];
    }
}

/* Shared HP entry point: double sum-of-squares + double sqrt inverse, matching
 * the historical host-reference path used by gemma-3. Do not weaken this ABI for
 * a family-specific speedup; add a scoped entry point instead. */
__global__ static void axiom_rmsnorm_f32_hp_device_kernel(
        const float *weight,
        const float *input,
        float *out,
        uint32_t count,
        float eps) {
    __shared__ double sums[256];
    const uint32_t tid = threadIdx.x;
    double ss = 0.0;
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        const double x = (double)input[i];
        ss += x * x;
    }
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt(sums[0] / (double)count + (double)eps));
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        out[i] = input[i] * inv * weight[i];
    }
}

/* Qwen B1 decode-only fast path: FP32 sum-of-squares, warp-shuffle + CTA
 * reduction, rsqrtf inverse, and float4 traffic when aligned. This preserves
 * the Fix 2 throughput win without changing the shared hp-rmsnorm ABI that
 * gemma-3 relies on. */
__global__ static void axiom_rmsnorm_f32_b1_device_kernel(
        const float *weight,
        const float *input,
        float *out,
        uint32_t count,
        float eps) {
    __shared__ float sums[8];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t vec_count = count >> 2u;
    const uint32_t tail = vec_count << 2u;
    const bool input_vec_ok = (((uintptr_t)input & 15u) == 0u);
    const bool apply_vec_ok =
            input_vec_ok && ((((uintptr_t)weight | (uintptr_t)out) & 15u) == 0u);

    float ss = 0.0f;
    if (input_vec_ok) {
        const float4 *input4 = (const float4 *)input;
        for (uint32_t i = tid; i < vec_count; i += blockDim.x) {
            const float4 x = input4[i];
            ss = fmaf(x.x, x.x, ss);
            ss = fmaf(x.y, x.y, ss);
            ss = fmaf(x.z, x.z, ss);
            ss = fmaf(x.w, x.w, ss);
        }
        for (uint32_t i = tail + tid; i < count; i += blockDim.x) {
            const float x = input[i];
            ss = fmaf(x, x, ss);
        }
    } else {
        for (uint32_t i = tid; i < count; i += blockDim.x) {
            const float x = input[i];
            ss = fmaf(x, x, ss);
        }
    }

#pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        ss += __shfl_down_sync(0xffffffffu, ss, offset);
    }
    if (lane == 0u) sums[warp] = ss;
    __syncthreads();

    ss = warp == 0u && lane < 8u ? sums[lane] : 0.0f;
    if (warp == 0u) {
#pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            ss += __shfl_down_sync(0xffffffffu, ss, offset);
        }
        if (lane == 0u) sums[0] = ss;
    }
    __syncthreads();
    const float inv = rsqrtf(sums[0] / (float)count + eps);

    if (apply_vec_ok) {
        const float4 *input4 = (const float4 *)input;
        const float4 *weight4 = (const float4 *)weight;
        float4 *out4 = (float4 *)out;
        for (uint32_t i = tid; i < vec_count; i += blockDim.x) {
            const float4 x = input4[i];
            const float4 w = weight4[i];
            out4[i] = make_float4(
                    x.x * inv * w.x,
                    x.y * inv * w.y,
                    x.z * inv * w.z,
                    x.w * inv * w.w);
        }
        for (uint32_t i = tail + tid; i < count; i += blockDim.x) {
            out[i] = input[i] * inv * weight[i];
        }
        return;
    }

    for (uint32_t i = tid; i < count; i += blockDim.x) {
        out[i] = input[i] * inv * weight[i];
    }
}

__global__ static void axiom_rmsnorm_f32_dual_device_kernel(
        const float *weight,
        const float *input0,
        float *out0,
        const float *input1,
        float *out1,
        uint32_t count,
        float eps) {
    const uint32_t row = (uint32_t)blockIdx.x;
    if (row >= 2u) return;
    const float *input = row == 0u ? input0 : input1;
    float *out = row == 0u ? out0 : out1;
    __shared__ float sums[256];
    const uint32_t tid = threadIdx.x;
    float ss = 0.0f;
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        ss += input[i] * input[i];
    }
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / (float)count + eps);
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        out[i] = input[i] * inv * weight[i];
    }
}

__global__ static void axiom_rmsnorm_f32_q8k_device_kernel(
        const float *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ out,
        axiom_cuda_q8_k_block *__restrict__ out_q8k,
        uint32_t count,
        float eps) {
    __shared__ float sums[256];
    __shared__ float warp_abs[8];
    __shared__ float warp_val[8];
    __shared__ float iscale;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    float ss = 0.0f;
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        const float x = input[i];
        ss += x * x;
    }
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / (float)count + eps);
    const uint32_t blocks = count / 256u;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t idx = b * 256u + tid;
        const float y = input[idx] * inv * weight[idx];
        out[idx] = y;

        float best_abs = fabsf(y);
        float best_val = y;
        #pragma unroll
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
            const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
            if (other_abs > best_abs) {
                best_abs = other_abs;
                best_val = other_val;
            }
        }
        if (lane == 0u) {
            warp_abs[warp] = best_abs;
            warp_val[warp] = best_val;
        }
        __syncthreads();
        if (warp == 0u) {
            best_abs = lane < 8u ? warp_abs[lane] : -1.0f;
            best_val = lane < 8u ? warp_val[lane] : 0.0f;
            #pragma unroll
            for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
                const float other_abs = __shfl_down_sync(0xffffffffu, best_abs, offset);
                const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
                if (other_abs > best_abs) {
                    best_abs = other_abs;
                    best_val = other_val;
                }
            }
            if (lane == 0u) {
                if (best_abs == 0.0f) {
                    iscale = 0.0f;
                    out_q8k[b].d = 0.0f;
                } else {
                    iscale = -127.0f / best_val;
                    out_q8k[b].d = 1.0f / iscale;
                }
            }
        }
        __syncthreads();
        if (iscale == 0.0f) {
            out_q8k[b].qs[tid] = 0;
        } else {
            int q = (int)lrintf(iscale * y);
            q = q < -128 ? -128 : q > 127 ? 127 : q;
            out_q8k[b].qs[tid] = (int8_t)q;
        }
        __syncthreads();
        if (tid < 16u) {
            int sum = 0;
            for (uint32_t i = 0; i < 16u; ++i) sum += out_q8k[b].qs[tid * 16u + i];
            out_q8k[b].bsums[tid] = (int16_t)sum;
        }
        __syncthreads();
    }
}

__global__ static void axiom_rmsnorm_f32_q8k_scale_device_kernel(
        const float *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ out,
        axiom_cuda_q8_k_block *__restrict__ out_q8k,
        uint32_t count,
        float eps) {
    __shared__ float sums[256];
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    const uint32_t tid = threadIdx.x;
    float ss = 0.0f;
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        const float x = input[i];
        ss += x * x;
    }
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / (float)count + eps);
    const uint32_t blocks = count / 256u;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint32_t idx = b * 256u + tid;
        const float y = input[idx] * inv * weight[idx];
        out[idx] = y;
        absmax[tid] = fabsf(y);
        maxv[tid] = y;
        __syncthreads();
        for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride && absmax[tid + stride] > absmax[tid]) {
                absmax[tid] = absmax[tid + stride];
                maxv[tid] = maxv[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            if (absmax[0] == 0.0f) {
                out_q8k[b].d = 0.0f;
            } else {
                const float iscale = -127.0f / maxv[0];
                out_q8k[b].d = 1.0f / iscale;
            }
        }
        __syncthreads();
    }
}

__global__ static void axiom_rmsnorm_f32_inv_device_kernel(
        const float *__restrict__ input,
        float *__restrict__ inv_out,
        uint32_t count,
        float eps) {
    __shared__ float sums[256];
    const uint32_t tid = threadIdx.x;
    float ss = 0.0f;
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        const float x = input[i];
        ss += x * x;
    }
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    if (tid == 0u) {
        inv_out[0] = rsqrtf(sums[0] / (float)count + eps);
    }
}

__global__ static void axiom_rmsnorm_f32_apply_device_kernel(
        const float *__restrict__ weight,
        const float *__restrict__ input,
        const float *__restrict__ inv,
        float *__restrict__ out,
        uint32_t count) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = input[i] * inv[0] * weight[i];
}

__global__ static void axiom_q8_k_pack_f32_with_inv_weight_kernel(
        const float *__restrict__ weight,
        const float *__restrict__ input,
        const float *__restrict__ inv,
        axiom_cuda_q8_k_block *__restrict__ out,
        uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t tid = threadIdx.x;
    __shared__ float absmax[256];
    __shared__ float maxv[256];
    __shared__ float iscale;
    const uint32_t idx = b * 256u + tid;
    const float y = input[idx] * inv[0] * weight[idx];
    absmax[tid] = fabsf(y);
    maxv[tid] = y;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride && absmax[tid + stride] > absmax[tid]) {
            absmax[tid] = absmax[tid + stride];
            maxv[tid] = maxv[tid + stride];
        }
        __syncthreads();
    }
    if (absmax[0] == 0.0f) {
        out[b].qs[tid] = 0;
        if (tid < 16u) out[b].bsums[tid] = 0;
        if (tid == 0u) out[b].d = 0.0f;
        return;
    }
    if (tid == 0u) iscale = -127.0f / maxv[0];
    __syncthreads();
    int q = (int)lrintf(iscale * y);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    out[b].qs[tid] = (int8_t)q;
    __syncthreads();
    if (tid < 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += out[b].qs[tid * 16u + i];
        out[b].bsums[tid] = (int16_t)sum;
    }
    if (tid == 0u) out[b].d = 1.0f / iscale;
}

extern "C" int axiom_cuda_rmsnorm_f32_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count,
        float eps) {
    if (!cuda_runtime || !weight_f32 || !input || !out || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_rmsnorm_f32_device_kernel<<<1, 256>>>(
            (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            count,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_rmsnorm_f32_hp_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count,
        float eps) {
    if (!cuda_runtime || !weight_f32 || !input || !out || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_rmsnorm_f32_hp_device_kernel<<<1, 256>>>(
            (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            count,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_rmsnorm_f32_b1_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count,
        float eps) {
    if (!cuda_runtime || !weight_f32 || !input || !out || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_rmsnorm_f32_b1_device_kernel<<<1, 256>>>(
            (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            count,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_rmsnorm_f32_dual_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input0,
        uint64_t input0_offset,
        void *out0,
        uint64_t out0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t count,
        float eps) {
    if (!cuda_runtime || !weight_f32 || !input0 || !out0 || !input1 || !out1 || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_f32;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_rmsnorm_f32_dual_device_kernel<<<2, 256>>>(
            (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
            (const float *)((const uint8_t *)i0buf->ptr + input0_offset),
            (float *)((uint8_t *)o0buf->ptr + out0_offset),
            (const float *)((const uint8_t *)i1buf->ptr + input1_offset),
            (float *)((uint8_t *)o1buf->ptr + out1_offset),
            count,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_rmsnorm_f32_q8k_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        void *out_q8k,
        uint64_t out_q8k_offset,
        uint32_t count,
        float eps) {
    if (!cuda_runtime || !weight_f32 || !input || !out || !out_q8k ||
        count == 0 || (count % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    axiom_cuda_buffer *qbuf = (axiom_cuda_buffer *)out_q8k;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    if (axiom_cuda_env_enabled("AXIOM_DS4_FFN_RMS_Q8K_SPLIT_PACK")) {
        float *inv = nullptr;
        int rc = axiom_cuda_rmsnorm_inv_scratch(runtime, &inv);
        if (rc != AXIOM_OK) return rc;
        axiom_rmsnorm_f32_inv_device_kernel<<<1, 256>>>(
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                inv,
                count,
                eps);
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        const uint32_t blocks = count / 256u;
        if (axiom_cuda_env_enabled("AXIOM_DS4_MOE_Q8K_PACK_RN_INTRIN")) {
            axiom_q8_k_pack_f32_with_inv_weight_rn_kernel<<<(int)blocks, 256>>>(
                    (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
                    (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                    inv,
                    (axiom_cuda_q8_k_block *)((uint8_t *)qbuf->ptr + out_q8k_offset),
                    blocks);
        } else {
            axiom_q8_k_pack_f32_with_inv_weight_kernel<<<(int)blocks, 256>>>(
                    (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
                    (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                    inv,
                    (axiom_cuda_q8_k_block *)((uint8_t *)qbuf->ptr + out_q8k_offset),
                    blocks);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) return axiom_cuda_status(err);
        axiom_rmsnorm_f32_apply_device_kernel<<<(int)((count + 255u) / 256u), 256>>>(
                (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                inv,
                (float *)((uint8_t *)obuf->ptr + out_offset),
                count);
    } else if (axiom_cuda_env_enabled("AXIOM_DS4_FFN_RMS_Q8K_SCALE_ONLY")) {
        axiom_rmsnorm_f32_q8k_scale_device_kernel<<<1, 256>>>(
                (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                (axiom_cuda_q8_k_block *)((uint8_t *)qbuf->ptr + out_q8k_offset),
                count,
                eps);
    } else {
        axiom_rmsnorm_f32_q8k_device_kernel<<<1, 256>>>(
                (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
                (const float *)((const uint8_t *)ibuf->ptr + input_offset),
                (float *)((uint8_t *)obuf->ptr + out_offset),
                (axiom_cuda_q8_k_block *)((uint8_t *)qbuf->ptr + out_q8k_offset),
                count,
                eps);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_head_rmsnorm_f32_device_kernel(
        float *x,
        uint32_t head_dim,
        float eps) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ float sums[256];
    float *row = x + (uint64_t)head * head_dim;
    float ss = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        ss += row[i] * row[i];
    }
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / (float)head_dim + eps);
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        row[i] *= inv;
    }
}

__global__ static void axiom_head_rmsnorm_f32_dual_device_kernel(
        float *x0,
        float *x1,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    const uint32_t row_id = (uint32_t)blockIdx.x;
    const uint32_t token = row_id / heads;
    const uint32_t head = row_id - token * heads;
    if (token >= 2u || head >= heads) return;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ float sums[256];
    float *x = token == 0u ? x0 : x1;
    float *row = x + (uint64_t)head * head_dim;
    float ss = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        ss += row[i] * row[i];
    }
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / (float)head_dim + eps);
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) {
        row[i] *= inv;
    }
}

extern "C" int axiom_cuda_head_rmsnorm_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    if (!cuda_runtime || !x || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_head_rmsnorm_f32_device_kernel<<<heads, 256>>>(
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            head_dim,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

/* P2 STEP 2 + precise fix: the dense NeoX RoPE KERNEL now lives in src/axiom_cuda_precise.cu,
 * compiled WITHOUT --use_fast_math so its powf/cosf/sinf are precise libm (byte-close to the host
 * rope_neox). The fast-math trig that USED to be here made the device attention core ~86% vs the
 * oracle (systematically worse); precise trig removes that. This thunk keeps the libaxiom buffer
 * plumbing and delegates the launch to the precise unit. */
extern "C" int axiom_rope_neox_precise_launch(
        int device, float *x, uint32_t heads, uint32_t head_dim, uint32_t pos, float theta);

extern "C" int axiom_cuda_rope_neox_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t pos,
        float theta) {
    if (!cuda_runtime || !x || heads == 0 || head_dim == 0 || (head_dim & 1u)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    const int prc = axiom_rope_neox_precise_launch(
            runtime->device,
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            heads, head_dim, pos, theta);
    return prc == 0 ? AXIOM_OK : AXIOM_ERR_CUDA;
}

/* P2 STEP 7 (Nemotron): partial NeoX RoPE — rotate only the first n_rot dims, pass-through the
 * rest. The KERNEL lives in src/axiom_cuda_precise.cu (non-fast-math, precise libm trig, same
 * unit as the dense rope above so the trig is byte-close to the host rope_neox_partial). This
 * thunk keeps the libaxiom buffer plumbing and delegates the launch. */
extern "C" int axiom_rope_neox_partial_precise_launch(
        int device, float *x, uint32_t heads, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos, float theta);

extern "C" int axiom_cuda_rope_neox_partial_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos,
        float theta) {
    if (!cuda_runtime || !x || heads == 0 || head_dim == 0 || (head_dim & 1u) ||
        n_rot == 0 || (n_rot & 1u) || n_rot > head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    const int prc = axiom_rope_neox_partial_precise_launch(
            runtime->device,
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            heads, head_dim, n_rot, pos, theta);
    return prc == 0 ? AXIOM_OK : AXIOM_ERR_CUDA;
}

/* P2 launch-fusion: fused per-head QK-RMSNorm(+weight)+NeoX rope in the precise unit. */
extern "C" int axiom_qknorm_rope_precise_launch(
        int device, float *x, const float *w, uint32_t heads, uint32_t head_dim,
        uint32_t pos, float theta, float eps);
extern "C" int axiom_qknorm_rope_pos_precise_launch(
        int device, float *x, const float *w, const uint32_t *pos,
        uint32_t heads, uint32_t head_dim, float theta, float eps);
extern "C" int axiom_qknorm_rope_pos_dual_precise_launch(
        int device,
        float *q, const float *qw,
        float *k, const float *kw,
        const uint32_t *pos,
        uint32_t q_heads, uint32_t k_heads, uint32_t head_dim,
        float theta, float eps);

extern "C" int axiom_cuda_qknorm_rope_f32_device(
        void *cuda_runtime,
        void *x, uint64_t x_offset,
        const void *w, uint64_t w_offset,
        uint32_t heads, uint32_t head_dim, uint32_t pos, float theta, float eps) {
    if (!cuda_runtime || !x || !w || heads == 0 || head_dim == 0 ||
        (head_dim & (head_dim - 1u)) != 0u || head_dim > 1024u || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)w;
    const int prc = axiom_qknorm_rope_precise_launch(
            runtime->device,
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            (const float *)((const uint8_t *)wbuf->ptr + w_offset),
            heads, head_dim, pos, theta, eps);
    return prc == 0 ? AXIOM_OK : AXIOM_ERR_CUDA;
}

extern "C" int axiom_cuda_qknorm_rope_pos_f32_device(
        void *cuda_runtime,
        void *x, uint64_t x_offset,
        const void *w, uint64_t w_offset,
        const void *pos, uint64_t pos_offset,
        uint32_t heads, uint32_t head_dim, float theta, float eps) {
    if (!cuda_runtime || !x || !w || !pos || heads == 0 || head_dim == 0 ||
        (head_dim & (head_dim - 1u)) != 0u || head_dim > 1024u || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)w;
    const axiom_cuda_buffer *pbuf = (const axiom_cuda_buffer *)pos;
    const int prc = axiom_qknorm_rope_pos_precise_launch(
            runtime->device,
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            (const float *)((const uint8_t *)wbuf->ptr + w_offset),
            (const uint32_t *)((const uint8_t *)pbuf->ptr + pos_offset),
            heads, head_dim, theta, eps);
    return prc == 0 ? AXIOM_OK : AXIOM_ERR_CUDA;
}

extern "C" int axiom_cuda_qknorm_rope_pos_dual_f32_device(
        void *cuda_runtime,
        void *q, uint64_t q_offset,
        const void *qw, uint64_t qw_offset,
        void *k, uint64_t k_offset,
        const void *kw, uint64_t kw_offset,
        const void *pos, uint64_t pos_offset,
        uint32_t q_heads, uint32_t k_heads, uint32_t head_dim, float theta, float eps) {
    if (!cuda_runtime || !q || !qw || !k || !kw || !pos ||
        q_heads == 0 || k_heads == 0 || head_dim == 0 ||
        (head_dim & (head_dim - 1u)) != 0u || head_dim > 1024u || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *qbuf = (axiom_cuda_buffer *)q;
    axiom_cuda_buffer *kbuf = (axiom_cuda_buffer *)k;
    const axiom_cuda_buffer *qwbuf = (const axiom_cuda_buffer *)qw;
    const axiom_cuda_buffer *kwbuf = (const axiom_cuda_buffer *)kw;
    const axiom_cuda_buffer *pbuf = (const axiom_cuda_buffer *)pos;
    const int prc = axiom_qknorm_rope_pos_dual_precise_launch(
            runtime->device,
            (float *)((uint8_t *)qbuf->ptr + q_offset),
            (const float *)((const uint8_t *)qwbuf->ptr + qw_offset),
            (float *)((uint8_t *)kbuf->ptr + k_offset),
            (const float *)((const uint8_t *)kwbuf->ptr + kw_offset),
            (const uint32_t *)((const uint8_t *)pbuf->ptr + pos_offset),
            q_heads, k_heads, head_dim, theta, eps);
    return prc == 0 ? AXIOM_OK : AXIOM_ERR_CUDA;
}

/* P2 STEP 3: per-head QK-RMSNorm WITH learned weight, faithful to the host qk_norm
 * (tools/axiom_qwen.cpp:531-539): DOUBLE sum-of-squares + (float)(1.0/sqrt(ss/hd+eps)) then
 * row[i]*inv*w[i]. The existing axiom_head_rmsnorm_f32_device is float-accum AND weightless so
 * it cannot be reused. One block per head, double shared-mem tree reduce (the layernorm
 * double-reduce idiom). w is a head_dim vector shared across all heads (ly->q_norm / k_norm). */
__global__ static void axiom_qwen_qk_rmsnorm_w_f32_kernel(
        float *x,
        const float *w,
        uint32_t head_dim,
        float eps) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ double sums[256];
    float *row = x + (uint64_t)head * head_dim;
    double ss = 0.0;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) ss += (double)row[i] * (double)row[i];
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt(sums[0] / (double)head_dim + (double)eps));
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) row[i] = row[i] * inv * w[i];
}

extern "C" int axiom_cuda_qk_rmsnorm_w_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        const void *w,
        uint64_t w_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    if (!cuda_runtime || !x || !w || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)w;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen_qk_rmsnorm_w_f32_kernel<<<heads, 256>>>(
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            (const float *)((const uint8_t *)wbuf->ptr + w_offset),
            head_dim,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

/* P2 STEP 7 (Gemma-4) thunk: DECOUPLED partial NeoX rope (active_pairs range, exp_dim denominator
 * are SEPARATE — global rotates 64 pairs of head_dim 512 but the denominator is 512). The KERNEL
 * lives in src/axiom_cuda_precise.cu (precise libm trig, same unit as the dense/partial ropes). */
extern "C" int axiom_rope_neox_decoupled_partial_precise_launch(
        int device, float *x, uint32_t heads, uint32_t head_dim, uint32_t active_pairs,
        uint32_t exp_dim, uint32_t pos, float theta);

extern "C" int axiom_cuda_rope_neox_decoupled_partial_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t active_pairs,
        uint32_t exp_dim,
        uint32_t pos,
        float theta) {
    if (!cuda_runtime || !x || heads == 0 || head_dim == 0 || (head_dim & 1u) ||
        active_pairs == 0 || exp_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    const int prc = axiom_rope_neox_decoupled_partial_precise_launch(
            runtime->device,
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            heads, head_dim, active_pairs, exp_dim, pos, theta);
    return prc == 0 ? AXIOM_OK : AXIOM_ERR_CUDA;
}

/* P2 STEP 7 (Gemma-4): per-head RMSNorm — out = row[i]*inv*w[i] (PLAIN weight), faithful to the
 * gemma-4 host rmsnorm1p/qknorm1p (tools/axiom_gemma4.cpp): DOUBLE sum-of-squares +
 * (float)(1.0/sqrt(ss/hd+eps)) inverse, then row[i]*inv*w[i]. NOTE: gemma-4 uses PLAIN *w (NOT
 * *(1+w) — the host comment "Gemma4 differs from Gemma2/3"). When w==NULL it is the v-norm
 * no-scale path (row[i]*inv, with_scale=False). One block per head, double shared-mem tree reduce
 * (same idiom as axiom_qwen_qk_rmsnorm_w_f32_kernel, but w is nullable). */
__global__ static void axiom_gemma4_qknorm1p_f32_kernel(
        float *x,
        const float *w,
        uint32_t head_dim,
        float eps) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    __shared__ double sums[256];
    float *row = x + (uint64_t)head * head_dim;
    double ss = 0.0;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) ss += (double)row[i] * (double)row[i];
    sums[tid] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt(sums[0] / (double)head_dim + (double)eps));
    if (w) { for (uint32_t i = tid; i < head_dim; i += blockDim.x) row[i] = row[i] * inv * w[i]; }
    else   { for (uint32_t i = tid; i < head_dim; i += blockDim.x) row[i] = row[i] * inv; }
}

extern "C" int axiom_cuda_gemma4_qknorm1p_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        const void *w,
        uint64_t w_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    if (!cuda_runtime || !x || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)w;  /* may be NULL: v-norm no-scale */
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_gemma4_qknorm1p_f32_kernel<<<heads, 256>>>(
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            wbuf ? (const float *)((const uint8_t *)wbuf->ptr + w_offset) : (const float *)nullptr,
            head_dim,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_head_rmsnorm_f32_dual_device(
        void *cuda_runtime,
        void *x0,
        uint64_t x0_offset,
        void *x1,
        uint64_t x1_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    if (!cuda_runtime || !x0 || !x1 || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *x0buf = (axiom_cuda_buffer *)x0;
    axiom_cuda_buffer *x1buf = (axiom_cuda_buffer *)x1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_head_rmsnorm_f32_dual_device_kernel<<<heads * 2u, 256>>>(
            (float *)((uint8_t *)x0buf->ptr + x0_offset),
            (float *)((uint8_t *)x1buf->ptr + x1_offset),
            heads,
            head_dim,
            eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

__device__ static float axiom_dsv4_e4m3fn_value(int i) {
    const int exp = (i >> 3) & 15;
    const int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float axiom_dsv4_e4m3fn_roundtrip(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (axiom_dsv4_e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float bd = fabsf(ax - axiom_dsv4_e4m3fn_value(best));
        const float nd = fabsf(ax - axiom_dsv4_e4m3fn_value(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) {
            best++;
        }
    }
    return sign * axiom_dsv4_e4m3fn_value(best);
}

__device__ static float axiom_dsv4_e2m1fn_value(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ static float axiom_dsv4_e2m1fn_roundtrip(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - axiom_dsv4_e2m1fn_value(0));
    for (int i = 1; i < 8; ++i) {
        const float diff = fabsf(ax - axiom_dsv4_e2m1fn_value(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * axiom_dsv4_e2m1fn_value(best);
}

__global__ static void axiom_deepseek_csa_indexer_qat_f32_kernel(
        float *x,
        uint32_t rows,
        uint32_t head_dim) {
    const uint32_t row = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    if (row >= rows || head_dim != 128u || tid >= 128u) return;
    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            const uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            const float a = vals[base];
            const float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    const float v = vals[tid] * 0.08838834764831845f;
    const uint32_t fp4_block = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();
    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(
                    absbuf[block_base + lane],
                    absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }
    const float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    const float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    const float clipped = fminf(6.0f, fmaxf(-6.0f, v / scale));
    xr[tid] = axiom_dsv4_e2m1fn_roundtrip(clipped) * scale;
}

extern "C" int axiom_cuda_deepseek_csa_indexer_qat_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t rows,
        uint32_t head_dim) {
    if (!cuda_runtime || !x || rows == 0 || head_dim != 128u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_csa_indexer_qat_f32_kernel<<<rows, 128>>>(
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            rows,
            head_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_deepseek_fp8_kv_quantize_f32_kernel(
        float *x,
        uint32_t head_dim,
        uint32_t n_rot) {
    const uint32_t row = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64u) {
        const uint32_t idx = off + tid;
        const float v = idx < n_nope ? xr[idx] : 0.0f;
        scratch[tid] = idx < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32u; stride > 0u; stride >>= 1u) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        const float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (idx < n_nope) {
            const float clipped = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[idx] = axiom_dsv4_e4m3fn_roundtrip(clipped) * scale;
        }
        __syncthreads();
    }
}

__global__ static void axiom_deepseek_fp8_kv_quantize_f32_dual_kernel(
        float *x0,
        float *x1,
        uint32_t rows,
        uint32_t head_dim,
        uint32_t n_rot) {
    const uint32_t row_id = (uint32_t)blockIdx.x;
    const uint32_t token = row_id / rows;
    const uint32_t row = row_id - token * rows;
    if (token >= 2u || row >= rows) return;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    float *x = token == 0u ? x0 : x1;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64u) {
        const uint32_t idx = off + tid;
        const float v = idx < n_nope ? xr[idx] : 0.0f;
        scratch[tid] = idx < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32u; stride > 0u; stride >>= 1u) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        const float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (idx < n_nope) {
            const float clipped = fminf(448.0f, fmaxf(-448.0f, v / scale));
            xr[idx] = axiom_dsv4_e4m3fn_roundtrip(clipped) * scale;
        }
        __syncthreads();
    }
}

extern "C" int axiom_cuda_deepseek_fp8_kv_quantize_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t rows,
        uint32_t head_dim,
        uint32_t n_rot) {
    if (!cuda_runtime || !x || rows == 0 || head_dim == 0 || n_rot >= head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_fp8_kv_quantize_f32_kernel<<<rows, 64>>>(
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            head_dim,
            n_rot);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_fp8_kv_quantize_dual_f32_device(
        void *cuda_runtime,
        void *x0,
        uint64_t x0_offset,
        void *x1,
        uint64_t x1_offset,
        uint32_t rows,
        uint32_t head_dim,
        uint32_t n_rot) {
    if (!cuda_runtime || !x0 || !x1 || rows == 0 || head_dim == 0 || n_rot >= head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *x0buf = (axiom_cuda_buffer *)x0;
    axiom_cuda_buffer *x1buf = (axiom_cuda_buffer *)x1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_deepseek_fp8_kv_quantize_f32_dual_kernel<<<rows * 2u, 64>>>(
            (float *)((uint8_t *)x0buf->ptr + x0_offset),
            (float *)((uint8_t *)x1buf->ptr + x1_offset),
            rows,
            head_dim,
            n_rot);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_f32_f16_round_kernel(float *x, uint32_t count) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] = __half2float(__float2half(x[i]));
}

__global__ static void axiom_f32_f16_round_dual_kernel(float *x0, float *x1, uint32_t count) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = count * 2u;
    if (i >= total) return;
    if (i < count) {
        x0[i] = __half2float(__float2half(x0[i]));
    } else {
        const uint32_t j = i - count;
        x1[j] = __half2float(__float2half(x1[j]));
    }
}

extern "C" int axiom_cuda_f32_f16_round_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t count) {
    if (!cuda_runtime || !x || count == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *xbuf = (axiom_cuda_buffer *)x;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((count + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_f32_f16_round_kernel<<<grid, block>>>(
            (float *)((uint8_t *)xbuf->ptr + x_offset),
            count);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_f32_f16_round_dual_device(
        void *cuda_runtime,
        void *x0,
        uint64_t x0_offset,
        void *x1,
        uint64_t x1_offset,
        uint32_t count) {
    if (!cuda_runtime || !x0 || !x1 || count == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_buffer *x0buf = (axiom_cuda_buffer *)x0;
    axiom_cuda_buffer *x1buf = (axiom_cuda_buffer *)x1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const uint32_t total = count * 2u;
    const int grid = (int)((total + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_f32_f16_round_dual_kernel<<<grid, block>>>(
            (float *)((uint8_t *)x0buf->ptr + x0_offset),
            (float *)((uint8_t *)x1buf->ptr + x1_offset),
            count);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_add_f32_device_kernel(
        const float *a,
        const float *b,
        float *out,
        uint32_t count) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] + b[i];
}

extern "C" int axiom_cuda_add_f32_device(
        void *cuda_runtime,
        const void *a,
        uint64_t a_offset,
        const void *b,
        uint64_t b_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count) {
    if (!cuda_runtime || !a || !b || !out || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *abuf = (const axiom_cuda_buffer *)a;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)b;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 128;
    const int grid = (int)((count + block - 1u) / block);
    axiom_add_f32_device_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)abuf->ptr + a_offset),
            (const float *)((const uint8_t *)bbuf->ptr + b_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            count);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_rmsnorm_f32(
        void *cuda_runtime,
        const uint16_t *weight_bf16_host,
        const float *input_host,
        float *out_host,
        uint32_t count,
        float eps) {
    if (!cuda_runtime || !weight_bf16_host || !input_host || !out_host || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    uint16_t *weight_dev = nullptr;
    float *input_dev = nullptr;
    float *out_dev = nullptr;
    const size_t weight_bytes = (size_t)count * sizeof(uint16_t);
    const size_t float_bytes = (size_t)count * sizeof(float);

    err = cudaMalloc(&weight_dev, weight_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&input_dev, float_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, float_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(weight_dev, weight_bf16_host, weight_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(input_dev, input_host, float_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    axiom_rmsnorm_f32_kernel<<<1, 1>>>(weight_dev, input_dev, out_dev, count, eps);

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, float_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(weight_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(input_dev);
    cudaFree(weight_dev);
    return axiom_cuda_status(err);
}

__global__ static void axiom_rope_f32_kernel(
        const float *input,
        float *out,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t position,
        float rope_theta) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pair_count = heads * (head_dim / 2u);
    if (i >= pair_count) return;
    const uint32_t head = i / (head_dim / 2u);
    const uint32_t pair = i % (head_dim / 2u);
    const uint32_t offset = head * head_dim + pair * 2u;
    const float x0 = input[offset];
    const float x1 = input[offset + 1u];
    const float inv_freq = powf(rope_theta, -((float)(pair * 2u) / (float)head_dim));
    const float angle = (float)position * inv_freq;
    float s = 0.0f;
    float c = 1.0f;
    sincosf(angle, &s, &c);
    out[offset] = x0 * c - x1 * s;
    out[offset + 1u] = x0 * s + x1 * c;
}

__global__ static void axiom_deepseek_rope_tail_f32_kernel(
        const float *input,
        float *out,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t position,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        uint32_t n_ctx_orig,
        uint32_t inverse) {
    const uint32_t pair_id = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pairs_per_head = n_rot / 2u;
    const uint32_t pair_count = heads * pairs_per_head;
    if (pair_id >= pair_count) return;

    const uint32_t head = pair_id / pairs_per_head;
    const uint32_t pair = pair_id - head * pairs_per_head;
    const uint32_t tail = head_dim - n_rot;
    const uint32_t offset = head * head_dim + tail + pair * 2u;

    const float x0 = input[offset];
    const float x1 = input[offset + 1u];
    const float sin_sign = inverse ? -1.0f : 1.0f;
    const float pair_f = (float)pair;
    float theta = (float)position * powf(freq_base, -2.0f * pair_f / (float)n_rot);
    float mscale = attn_factor;

    if (ext_factor != 0.0f && n_ctx_orig > 0u && freq_scale > 0.0f) {
        const float two_pi = 6.2831853071795864769f;
        const float denom = 2.0f * logf(freq_base);
        const float fast = (float)n_rot * logf((float)n_ctx_orig / (beta_fast * two_pi)) / denom;
        const float slow = (float)n_rot * logf((float)n_ctx_orig / (beta_slow * two_pi)) / denom;
        const float start = floorf(fast);
        const float end = ceilf(slow);
        const float ramp_denom = fmaxf(0.001f, end - start);
        const float ramp_raw = (pair_f - start) / ramp_denom;
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, ramp_raw));
        const float interp = theta * freq_scale;
        theta = interp * (1.0f - ramp * ext_factor) + theta * (ramp * ext_factor);
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    float s = 0.0f;
    float c = 1.0f;
    sincosf(theta, &s, &c);
    s *= sin_sign;
    out[offset] = (x0 * c - x1 * s) * mscale;
    out[offset + 1u] = (x0 * s + x1 * c) * mscale;
}

__global__ static void axiom_deepseek_rope_tail_f32_dual_kernel(
        const float *input0,
        float *out0,
        uint32_t position0,
        const float *input1,
        float *out1,
        uint32_t position1,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        uint32_t n_ctx_orig,
        uint32_t inverse) {
    const uint32_t pair_id = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t pairs_per_head = n_rot / 2u;
    const uint32_t pairs_per_token = heads * pairs_per_head;
    const uint32_t token = pair_id / pairs_per_token;
    const uint32_t local_pair = pair_id - token * pairs_per_token;
    if (token >= 2u || local_pair >= pairs_per_token) return;

    const float *input = token == 0u ? input0 : input1;
    float *out = token == 0u ? out0 : out1;
    const uint32_t position = token == 0u ? position0 : position1;
    const uint32_t head = local_pair / pairs_per_head;
    const uint32_t pair = local_pair - head * pairs_per_head;
    const uint32_t tail = head_dim - n_rot;
    const uint32_t offset = head * head_dim + tail + pair * 2u;

    const float x0 = input[offset];
    const float x1 = input[offset + 1u];
    const float sin_sign = inverse ? -1.0f : 1.0f;
    const float pair_f = (float)pair;
    float theta = (float)position * powf(freq_base, -2.0f * pair_f / (float)n_rot);
    float mscale = attn_factor;

    if (ext_factor != 0.0f && n_ctx_orig > 0u && freq_scale > 0.0f) {
        const float two_pi = 6.2831853071795864769f;
        const float denom = 2.0f * logf(freq_base);
        const float fast = (float)n_rot * logf((float)n_ctx_orig / (beta_fast * two_pi)) / denom;
        const float slow = (float)n_rot * logf((float)n_ctx_orig / (beta_slow * two_pi)) / denom;
        const float start = floorf(fast);
        const float end = ceilf(slow);
        const float ramp_denom = fmaxf(0.001f, end - start);
        const float ramp_raw = (pair_f - start) / ramp_denom;
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, ramp_raw));
        const float interp = theta * freq_scale;
        theta = interp * (1.0f - ramp * ext_factor) + theta * (ramp * ext_factor);
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    float s = 0.0f;
    float c = 1.0f;
    sincosf(theta, &s, &c);
    s *= sin_sign;
    out[offset] = (x0 * c - x1 * s) * mscale;
    out[offset + 1u] = (x0 * s + x1 * c) * mscale;
}

extern "C" int axiom_cuda_deepseek_rope_tail_f32_device(
        void *cuda_runtime,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t position,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        uint32_t n_ctx_orig,
        uint32_t inverse) {
    if (!cuda_runtime || !input || !out || heads == 0 || head_dim == 0 ||
        n_rot == 0 || n_rot > head_dim || (n_rot & 1u) != 0u ||
        freq_base <= 0.0f || freq_scale <= 0.0f ||
        beta_fast <= 0.0f || beta_slow <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const size_t bytes = (size_t)heads * head_dim * sizeof(float);
    const uint8_t *src = (const uint8_t *)ibuf->ptr + input_offset;
    uint8_t *dst = (uint8_t *)obuf->ptr + out_offset;
    if (src != dst) {
        err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }
    const uint32_t pairs = heads * (n_rot / 2u);
    const int block = 256;
    const int grid = (int)((pairs + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_deepseek_rope_tail_f32_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            heads,
            head_dim,
            n_rot,
            position,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            n_ctx_orig,
            inverse);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_deepseek_rope_tail_dual_f32_device(
        void *cuda_runtime,
        const void *input0,
        uint64_t input0_offset,
        void *out0,
        uint64_t out0_offset,
        uint32_t position0,
        const void *input1,
        uint64_t input1_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t position1,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        uint32_t n_ctx_orig,
        uint32_t inverse) {
    if (!cuda_runtime || !input0 || !out0 || !input1 || !out1 || heads == 0 || head_dim == 0 ||
        n_rot == 0 || n_rot > head_dim || (n_rot & 1u) != 0u ||
        freq_base <= 0.0f || freq_scale <= 0.0f ||
        beta_fast <= 0.0f || beta_slow <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *i0buf = (const axiom_cuda_buffer *)input0;
    axiom_cuda_buffer *o0buf = (axiom_cuda_buffer *)out0;
    const axiom_cuda_buffer *i1buf = (const axiom_cuda_buffer *)input1;
    axiom_cuda_buffer *o1buf = (axiom_cuda_buffer *)out1;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const size_t bytes = (size_t)heads * head_dim * sizeof(float);
    const uint8_t *src0 = (const uint8_t *)i0buf->ptr + input0_offset;
    uint8_t *dst0 = (uint8_t *)o0buf->ptr + out0_offset;
    const uint8_t *src1 = (const uint8_t *)i1buf->ptr + input1_offset;
    uint8_t *dst1 = (uint8_t *)o1buf->ptr + out1_offset;
    if (src0 != dst0) {
        err = cudaMemcpy(dst0, src0, bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }
    if (src1 != dst1) {
        err = cudaMemcpy(dst1, src1, bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) return axiom_cuda_status(err);
    }
    const uint32_t pairs = heads * (n_rot / 2u) * 2u;
    const int block = 256;
    const int grid = (int)((pairs + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_deepseek_rope_tail_f32_dual_kernel<<<grid, block>>>(
            (const float *)src0,
            (float *)dst0,
            position0,
            (const float *)src1,
            (float *)dst1,
            position1,
            heads,
            head_dim,
            n_rot,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            n_ctx_orig,
            inverse);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_rope_f32(
        void *cuda_runtime,
        const float *input_host,
        float *out_host,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t position,
        float rope_theta) {
    if (!cuda_runtime || !input_host || !out_host || heads == 0 ||
        head_dim == 0 || (head_dim % 2u) != 0u || rope_theta <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *input_dev = nullptr;
    float *out_dev = nullptr;
    const size_t count = (size_t)heads * head_dim;
    const size_t bytes = count * sizeof(float);
    err = cudaMalloc(&input_dev, bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(input_dev, input_host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int block = 256;
        const int grid = (int)(((count / 2u) + block - 1) / block);
        axiom_rope_f32_kernel<<<grid, block>>>(
                input_dev,
                out_dev,
                heads,
                head_dim,
                position,
                rope_theta);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(input_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(input_dev);
    return axiom_cuda_status(err);
}

__global__ static void axiom_attention_single_f32_kernel(
        const float *q,
        const float *k,
        const float *v,
        float *out,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t count = q_heads * head_dim;
    if (i >= count) return;
    const uint32_t q_head = i / head_dim;
    const uint32_t dim = i % head_dim;
    const uint32_t group = q_heads / kv_heads;
    const uint32_t kv_head = q_head / group;
    (void)q;
    (void)k;
    out[i] = v[(uint64_t)kv_head * head_dim + dim];
}

extern "C" int axiom_cuda_attention_single_f32(
        void *cuda_runtime,
        const float *q_host,
        const float *k_host,
        const float *v_host,
        float *out_host,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim) {
    if (!cuda_runtime || !q_host || !k_host || !v_host || !out_host ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        (q_heads % kv_heads) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *q_dev = nullptr;
    float *k_dev = nullptr;
    float *v_dev = nullptr;
    float *out_dev = nullptr;
    const size_t q_bytes = (size_t)q_heads * head_dim * sizeof(float);
    const size_t kv_bytes = (size_t)kv_heads * head_dim * sizeof(float);
    err = cudaMalloc(&q_dev, q_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&k_dev, kv_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&v_dev, kv_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, q_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(q_dev, q_host, q_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(k_dev, k_host, kv_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(v_dev, v_host, kv_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int block = 256;
        const int grid = (int)((q_heads * head_dim + block - 1) / block);
        axiom_attention_single_f32_kernel<<<grid, block>>>(
                q_dev,
                k_dev,
                v_dev,
                out_dev,
                q_heads,
                kv_heads,
                head_dim);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, q_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(v_dev);
    cudaFree(k_dev);
    cudaFree(q_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(v_dev);
    cudaFree(k_dev);
    cudaFree(q_dev);
    return axiom_cuda_status(err);
}

__global__ static void axiom_attention_cache_f32_kernel(
        const float *q,
        const float *k_cache,
        const float *v_cache,
        float *out,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        uint32_t cache_tokens) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t count = q_heads * head_dim;
    if (i >= count) return;

    const uint32_t q_head = i / head_dim;
    const uint32_t dim = i % head_dim;
    const uint32_t group = q_heads / kv_heads;
    const uint32_t kv_head = q_head / group;
    const float *q_vec = q + (uint64_t)q_head * head_dim;
    const float scale = rsqrtf((float)head_dim);

    float max_score = -3.4028234663852886e+38F;
    for (uint32_t t = 0; t < cache_tokens; ++t) {
        const float *k_vec =
                k_cache + ((uint64_t)t * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        score *= scale;
        if (score > max_score) max_score = score;
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (uint32_t t = 0; t < cache_tokens; ++t) {
        const float *k_vec =
                k_cache + ((uint64_t)t * kv_heads + kv_head) * head_dim;
        const float *v_vec =
                v_cache + ((uint64_t)t * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        const float w = expf(score * scale - max_score);
        denom += w;
        acc += w * v_vec[dim];
    }
    out[i] = denom > 0.0f ? acc / denom : 0.0f;
}

/* P2 STEP 6: OPTIMIZED block-per-head GQA causal attention. The naive
 * axiom_attention_cache_f32_kernel uses one thread per OUTPUT ELEMENT and recomputes the full
 * head_dim dot q.k for EVERY output dim => head_dim-fold redundant (128x for qwen) and twice
 * (two passes). This kernel uses ONE BLOCK per query head (block = head_dim, one thread per dim):
 * each q.k score is computed ONCE via a cooperative head_dim tree-reduction, and the softmax is
 * ONLINE (running max/denom) so it is single-pass over the KV cache. Same cache layout
 * (t*kv_heads+kv_head)*head_dim, same scale 1/sqrt(head_dim), causal cache_tokens=pos+1.
 * Constraint: head_dim must be a power of two and <= 1024 (qwen/gemma/nemotron = 128).
 * SLIDING WINDOW: window>0 restricts the softmax to the last `window` keys
 * [cache_tokens-window, cache_tokens) (gemma sliding layers); window==0 => full causal
 * (global/full layers + qwen/nemotron) — byte-identical to the pre-window behavior. */
__global__ static void axiom_attention_core_blk_f32_kernel(
        const float *q, const float *k_cache, const float *v_cache, float *out,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens,
        uint32_t window) {
    const uint32_t qh = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t group = q_heads / kv_heads;
    const uint32_t kvh = qh / group;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t start = (window != 0u && cache_tokens > window) ? (cache_tokens - window) : 0u;
    extern __shared__ float sh[];                  /* [head_dim] dot-reduction scratch */
    const float qd = q[(uint64_t)qh * head_dim + tid];
    float m = -3.4028234663852886e+38F;            /* running max */
    float l = 0.0f;                                 /* running denom */
    float acc = 0.0f;                               /* running output for dim tid */
    for (uint32_t t = start; t < cache_tokens; ++t) {
        const uint64_t base = ((uint64_t)t * kv_heads + kvh) * head_dim;
        sh[tid] = qd * k_cache[base + tid];
        __syncthreads();
        for (uint32_t s = head_dim >> 1; s > 0u; s >>= 1u) {
            if (tid < s) sh[tid] += sh[tid + s];
            __syncthreads();
        }
        const float score = sh[0] * scale;
        __syncthreads();
        const float m_new = fmaxf(m, score);
        const float corr = expf(m - m_new);
        const float w = expf(score - m_new);
        l = l * corr + w;
        acc = acc * corr + w * v_cache[base + tid];
        m = m_new;
    }
    out[(uint64_t)qh * head_dim + tid] = l > 0.0f ? acc / l : 0.0f;
}

extern "C" int axiom_cuda_attention_core_fast_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_cache, uint64_t k_offset,
        const void *v_cache, uint64_t v_offset,
        void *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens,
        uint32_t window) {
    if (!cuda_runtime || !q || !k_cache || !v_cache || !out ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 || cache_tokens == 0 ||
        (q_heads % kv_heads) != 0 || (head_dim & (head_dim - 1u)) != 0u || head_dim > 1024u) {
        return AXIOM_ERR_INVALID_ARGUMENT;   /* caller falls back to the naive kernel */
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qb = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *kb = (const axiom_cuda_buffer *)k_cache;
    const axiom_cuda_buffer *vb = (const axiom_cuda_buffer *)v_cache;
    axiom_cuda_buffer *ob = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_attention_core_blk_f32_kernel<<<q_heads, head_dim, head_dim * sizeof(float)>>>(
            (const float *)((const uint8_t *)qb->ptr + q_offset),
            (const float *)((const uint8_t *)kb->ptr + k_offset),
            (const float *)((const uint8_t *)vb->ptr + v_offset),
            (float *)((uint8_t *)ob->ptr + out_offset),
            q_heads, kv_heads, head_dim, cache_tokens, window);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

/* P2 STEP 7 (Gemma-4): block-per-head online-softmax GQA attention with a CALLER-SUPPLIED scale.
 * Identical to axiom_attention_core_blk_f32_kernel except `scale` is a param instead of
 * rsqrtf(head_dim): gemma-4 attention uses scale = 1.0 (raw dot product, NO 1/sqrt(hd)). Handles
 * MQA (kv_heads=1, group=q_heads) and GQA (kv_heads=8). Cache layout token-major
 * (t*kv_heads+kvh)*head_dim, cache_tokens = pos+1. SLIDING WINDOW: window>0 restricts the
 * softmax to the last `window` keys [cache_tokens-window, cache_tokens) (gemma-4 sliding
 * layers, window=1024); window==0 => full causal — matches the host attention(). */
__global__ static void axiom_attention_core_scale_blk_f32_kernel(
        const float *q, const float *k_cache, const float *v_cache, float *out,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens, float scale,
        uint32_t window) {
    const uint32_t qh = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t group = q_heads / kv_heads;
    const uint32_t kvh = qh / group;
    const uint32_t start = (window != 0u && cache_tokens > window) ? (cache_tokens - window) : 0u;
    extern __shared__ float sh[];                  /* [head_dim] dot-reduction scratch */
    const float qd = q[(uint64_t)qh * head_dim + tid];
    float m = -3.4028234663852886e+38F;            /* running max */
    float l = 0.0f;                                 /* running denom */
    float acc = 0.0f;                               /* running output for dim tid */
    for (uint32_t t = start; t < cache_tokens; ++t) {
        const uint64_t base = ((uint64_t)t * kv_heads + kvh) * head_dim;
        sh[tid] = qd * k_cache[base + tid];
        __syncthreads();
        for (uint32_t s = head_dim >> 1; s > 0u; s >>= 1u) {
            if (tid < s) sh[tid] += sh[tid + s];
            __syncthreads();
        }
        const float score = sh[0] * scale;
        __syncthreads();
        const float m_new = fmaxf(m, score);
        const float corr = expf(m - m_new);
        const float w = expf(score - m_new);
        l = l * corr + w;
        acc = acc * corr + w * v_cache[base + tid];
        m = m_new;
    }
    out[(uint64_t)qh * head_dim + tid] = l > 0.0f ? acc / l : 0.0f;
}

extern "C" int axiom_cuda_attention_core_scale_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_cache, uint64_t k_offset,
        const void *v_cache, uint64_t v_offset,
        void *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens, float scale,
        uint32_t window) {
    if (!cuda_runtime || !q || !k_cache || !v_cache || !out ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 || cache_tokens == 0 ||
        (q_heads % kv_heads) != 0 || (head_dim & (head_dim - 1u)) != 0u || head_dim > 1024u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qb = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *kb = (const axiom_cuda_buffer *)k_cache;
    const axiom_cuda_buffer *vb = (const axiom_cuda_buffer *)v_cache;
    axiom_cuda_buffer *ob = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_attention_core_scale_blk_f32_kernel<<<q_heads, head_dim, head_dim * sizeof(float)>>>(
            (const float *)((const uint8_t *)qb->ptr + q_offset),
            (const float *)((const uint8_t *)kb->ptr + k_offset),
            (const float *)((const uint8_t *)vb->ptr + v_offset),
            (float *)((uint8_t *)ob->ptr + out_offset),
            q_heads, kv_heads, head_dim, cache_tokens, scale, window);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

/* W5a (continuous batching): PAGED variant of axiom_attention_core_blk_f32_kernel. SAME math —
 * the identical online-softmax accumulation in STRICTLY ascending t — but K/V live in a paged
 * device pool addressed through a per-active-seq block table instead of a per-seq contiguous
 * cache:
 *   lb = t / BT, slot = t % BT, pb = block_tables[a*tbl_stride + lb]
 *   element base = (((uint64_t)pb*NL + L)*BT + slot)*pool_stride + (uint64_t)kvh*head_dim
 * Paging is pure storage relocation: the same logical KV bytes are read in the same ascending-t
 * order, so the output is BYTE-IDENTICAL to the contiguous kernel given identical logical KV.
 * Layouts: q/out are per-active-seq token-major [A][q_heads*head_dim] — element
 * q[((uint64_t)a*q_heads + qh)*head_dim + d] (A=1 for prefill). Launch grid dim3(q_heads, A)
 * (blockIdx.x = query head, blockIdx.y = active-seq index a), block = head_dim, shmem =
 * head_dim*sizeof(float) — exactly the contiguous kernel's shape per seq. block_tables is a
 * DEVICE int32 array [A][tbl_stride]; cache_tokens a DEVICE uint32 array [A] (per-seq ragged
 * lengths in ONE launch). pool_stride is the ELEMENT stride of one (block,layer,slot) row and
 * is SEPARATE from kv_heads*head_dim (gemma-4 later packs its true KVD into a wider uniform
 * slab). kv-head mapping kvh = qh/(q_heads/kv_heads) EXACTLY as the contiguous kernel (GQA 8
 * and MQA 1). window: same semantics as the contiguous kernel, computed PER SEQ from
 * ct = cache_tokens[a]. The partial last block falls out of the t<ct bound (no special case).
 * A seq with ct==0 writes zeros (l==0 guard). */
__global__ static void axiom_attention_core_paged_blk_f32_kernel(
        const float *q, const float *kpool, const float *vpool, float *out,
        const int32_t *block_tables, const uint32_t *cache_tokens,
        uint32_t tbl_stride, uint32_t NL, uint32_t L, uint32_t BT, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
        uint32_t window) {
    const uint32_t a = (uint32_t)blockIdx.y;
    const uint32_t qh = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t group = q_heads / kv_heads;
    const uint32_t kvh = qh / group;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t cache_t = cache_tokens[a];
    const uint32_t start = (window != 0u && cache_t > window) ? (cache_t - window) : 0u;
    const int32_t *tbl = block_tables + (uint64_t)a * tbl_stride;
    const uint64_t kvh_off = (uint64_t)kvh * head_dim;
    extern __shared__ float sh[];                  /* [head_dim] dot-reduction scratch */
    const float qd = q[((uint64_t)a * q_heads + qh) * head_dim + tid];
    float m = -3.4028234663852886e+38F;            /* running max */
    float l = 0.0f;                                 /* running denom */
    float acc = 0.0f;                               /* running output for dim tid */
    for (uint32_t t = start; t < cache_t; ++t) {
        const uint32_t pb = (uint32_t)tbl[t / BT];
        const uint64_t base =
                (((uint64_t)pb * NL + L) * BT + (t % BT)) * pool_stride + kvh_off;
        sh[tid] = qd * kpool[base + tid];
        __syncthreads();
        for (uint32_t s = head_dim >> 1; s > 0u; s >>= 1u) {
            if (tid < s) sh[tid] += sh[tid + s];
            __syncthreads();
        }
        const float score = sh[0] * scale;
        __syncthreads();
        const float m_new = fmaxf(m, score);
        const float corr = expf(m - m_new);
        const float w = expf(score - m_new);
        l = l * corr + w;
        acc = acc * corr + w * vpool[base + tid];
        m = m_new;
    }
    out[((uint64_t)a * q_heads + qh) * head_dim + tid] = l > 0.0f ? acc / l : 0.0f;
}

extern "C" int axiom_cuda_attention_core_paged_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_pool, uint64_t k_offset,
        const void *v_pool, uint64_t v_offset,
        void *out, uint64_t out_offset,
        const void *block_tables, uint64_t tbl_offset,
        const void *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
        uint32_t window) {
    if (!cuda_runtime || !q || !k_pool || !v_pool || !out || !block_tables || !cache_tokens ||
        active == 0 || active > 65535u || tbl_stride == 0 ||
        n_layers == 0 || layer >= n_layers || block_tokens == 0 ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        (q_heads % kv_heads) != 0 || (head_dim & (head_dim - 1u)) != 0u || head_dim > 1024u ||
        (uint64_t)pool_stride < (uint64_t)kv_heads * head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qb = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *kb = (const axiom_cuda_buffer *)k_pool;
    const axiom_cuda_buffer *vb = (const axiom_cuda_buffer *)v_pool;
    axiom_cuda_buffer *ob = (axiom_cuda_buffer *)out;
    const axiom_cuda_buffer *tb = (const axiom_cuda_buffer *)block_tables;
    const axiom_cuda_buffer *cb = (const axiom_cuda_buffer *)cache_tokens;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_attention_core_paged_blk_f32_kernel<<<dim3(q_heads, active), head_dim,
            head_dim * sizeof(float)>>>(
            (const float *)((const uint8_t *)qb->ptr + q_offset),
            (const float *)((const uint8_t *)kb->ptr + k_offset),
            (const float *)((const uint8_t *)vb->ptr + v_offset),
            (float *)((uint8_t *)ob->ptr + out_offset),
            (const int32_t *)((const uint8_t *)tb->ptr + tbl_offset),
            (const uint32_t *)((const uint8_t *)cb->ptr + ct_offset),
            tbl_stride, n_layers, layer, block_tokens, pool_stride,
            q_heads, kv_heads, head_dim, window);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

static constexpr uint32_t AXIOM_ATTN_VEC_SPLIT_TOKENS = 4u;
static constexpr uint32_t AXIOM_ATTN_VEC_THREADS = 32u;

static int axiom_attention_vec_geometry(
        uint32_t active,
        uint32_t tbl_stride,
        uint32_t block_tokens,
        uint32_t q_heads,
        uint32_t head_dim,
        uint32_t window,
        uint32_t *tiles_out,
        uint32_t *partials_per_head_out,
        uint64_t *partial_count_out) {
    if (!tiles_out || !partials_per_head_out || !partial_count_out) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *tiles_out = 0;
    *partials_per_head_out = 0;
    *partial_count_out = 0;
    if (active != 1u || tbl_stride == 0 || block_tokens == 0 ||
        q_heads == 0 || head_dim != 128u || window != 0u) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    const uint64_t max_tokens = (uint64_t)tbl_stride * block_tokens;
    const uint64_t partials64 =
            (max_tokens + AXIOM_ATTN_VEC_SPLIT_TOKENS - 1u) / AXIOM_ATTN_VEC_SPLIT_TOKENS;
    if (partials64 == 0 || partials64 > 65535u) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t partial_count = (uint64_t)active * q_heads * partials64;
    if (partials64 > 65535u || partial_count == 0) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    *tiles_out = (uint32_t)partials64;
    *partials_per_head_out = (uint32_t)partials64;
    *partial_count_out = partial_count;
    return AXIOM_OK;
}

__global__ static void axiom_attention_core_paged_b1_vec_partial_f32_kernel(
        const float *q, const float *kpool, const float *vpool,
        const int32_t *block_tables, const uint32_t *cache_tokens,
        float *partial_m, float *partial_l, float *partial_acc,
        uint32_t tbl_stride, uint32_t NL, uint32_t L, uint32_t BT, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
        uint32_t partials_per_head) {
    const uint32_t qh = (uint32_t)blockIdx.x;
    const uint32_t a = (uint32_t)blockIdx.y;
    const uint32_t split = (uint32_t)blockIdx.z;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t lane = tid & 31u;

    const uint32_t group = q_heads / kv_heads;
    const uint32_t kvh = qh / group;
    const uint32_t cache_t = cache_tokens[a];
    const int32_t *tbl = block_tables + (uint64_t)a * tbl_stride;
    const uint64_t kvh_off = (uint64_t)kvh * head_dim;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t split_start = split * AXIOM_ATTN_VEC_SPLIT_TOKENS;
    uint32_t split_end = split_start + AXIOM_ATTN_VEC_SPLIT_TOKENS;
    const uint32_t max_tokens = tbl_stride * BT;
    if (split_end > cache_t) split_end = cache_t;
    if (split_end > max_tokens) split_end = max_tokens;

    float m = -3.4028234663852886e+38F;
    float l = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const uint64_t q_base = ((uint64_t)a * q_heads + qh) * head_dim;
    const uint32_t mask = 0xffffffffu;
    if (split_start < split_end) {
        const float q0 = q[q_base + lane];
        const float q1 = q[q_base + lane + 32u];
        const float q2 = q[q_base + lane + 64u];
        const float q3 = q[q_base + lane + 96u];
        for (uint32_t t = split_start; t < split_end; ++t) {
            const uint32_t pb = (uint32_t)tbl[t / BT];
            const uint64_t base =
                    (((uint64_t)pb * NL + L) * BT + (t % BT)) * pool_stride + kvh_off;
            float dot = q0 * kpool[base + lane] +
                        q1 * kpool[base + lane + 32u] +
                        q2 * kpool[base + lane + 64u] +
                        q3 * kpool[base + lane + 96u];
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_down_sync(mask, dot, off);
            const float score = __shfl_sync(mask, dot, 0) * scale;
            const float m_new = fmaxf(m, score);
            const float corr = expf(m - m_new);
            const float w = expf(score - m_new);
            l = l * corr + w;
            acc0 = acc0 * corr + w * vpool[base + lane];
            acc1 = acc1 * corr + w * vpool[base + lane + 32u];
            acc2 = acc2 * corr + w * vpool[base + lane + 64u];
            acc3 = acc3 * corr + w * vpool[base + lane + 96u];
            m = m_new;
        }
    }

    const uint64_t part =
            (((uint64_t)a * q_heads + qh) * partials_per_head) +
            split;
    if (lane == 0u) {
        partial_m[part] = m;
        partial_l[part] = l;
    }
    partial_acc[part * head_dim + lane] = acc0;
    partial_acc[part * head_dim + lane + 32u] = acc1;
    partial_acc[part * head_dim + lane + 64u] = acc2;
    partial_acc[part * head_dim + lane + 96u] = acc3;
}

__global__ static void axiom_attention_core_paged_b1_vec_combine_f32_kernel(
        const float *partial_m,
        const float *partial_l,
        const float *partial_acc,
        float *out,
        uint32_t q_heads,
        uint32_t head_dim,
        uint32_t partials_per_head) {
    const uint32_t qh = (uint32_t)blockIdx.x;
    const uint32_t a = (uint32_t)blockIdx.y;
    const uint32_t tid = (uint32_t)threadIdx.x;
    if (tid >= head_dim) return;

    const uint64_t part_base = ((uint64_t)a * q_heads + qh) * partials_per_head;
    float m = -3.4028234663852886e+38F;
    float l = 0.0f;
    float acc = 0.0f;
    for (uint32_t p = 0; p < partials_per_head; ++p) {
        const uint64_t part = part_base + p;
        const float lp = partial_l[part];
        if (lp <= 0.0f) continue;
        const float mp = partial_m[part];
        const float ap = partial_acc[part * head_dim + tid];
        if (l <= 0.0f) {
            m = mp;
            l = lp;
            acc = ap;
            continue;
        }
        const float m_new = fmaxf(m, mp);
        const float corr = expf(m - m_new);
        const float w = expf(mp - m_new);
        l = l * corr + lp * w;
        acc = acc * corr + ap * w;
        m = m_new;
    }
    out[((uint64_t)a * q_heads + qh) * head_dim + tid] = l > 0.0f ? acc / l : 0.0f;
}

extern "C" int axiom_cuda_attention_core_paged_b1_vec_reserve(
        void *cuda_runtime,
        uint32_t active,
        uint32_t tbl_stride,
        uint32_t block_tokens,
        uint32_t q_heads,
        uint32_t head_dim,
        uint32_t window) {
    if (!cuda_runtime) return AXIOM_ERR_INVALID_ARGUMENT;
    uint32_t tiles = 0, partials_per_head = 0;
    uint64_t partial_count = 0;
    int rc = axiom_attention_vec_geometry(
            active, tbl_stride, block_tokens, q_heads, head_dim, window,
            &tiles, &partials_per_head, &partial_count);
    if (rc != AXIOM_OK) return rc;
    (void)tiles;
    (void)partials_per_head;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    float *pm = nullptr, *pl = nullptr, *pa = nullptr;
    return axiom_cuda_attention_vec_scratch(runtime, partial_count, head_dim, &pm, &pl, &pa);
}

extern "C" int axiom_cuda_attention_core_paged_b1_vec_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_pool, uint64_t k_offset,
        const void *v_pool, uint64_t v_offset,
        void *out, uint64_t out_offset,
        const void *block_tables, uint64_t tbl_offset,
        const void *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
        uint32_t window) {
    if (!cuda_runtime || !q || !k_pool || !v_pool || !out || !block_tables || !cache_tokens ||
        n_layers == 0 || layer >= n_layers || kv_heads == 0 ||
        (q_heads % kv_heads) != 0 || (uint64_t)pool_stride < (uint64_t)kv_heads * head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t tiles = 0, partials_per_head = 0;
    uint64_t partial_count = 0;
    int rc = axiom_attention_vec_geometry(
            active, tbl_stride, block_tokens, q_heads, head_dim, window,
            &tiles, &partials_per_head, &partial_count);
    if (rc != AXIOM_OK) return rc;

    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qb = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *kb = (const axiom_cuda_buffer *)k_pool;
    const axiom_cuda_buffer *vb = (const axiom_cuda_buffer *)v_pool;
    axiom_cuda_buffer *ob = (axiom_cuda_buffer *)out;
    const axiom_cuda_buffer *tb = (const axiom_cuda_buffer *)block_tables;
    const axiom_cuda_buffer *cb = (const axiom_cuda_buffer *)cache_tokens;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *partial_m = nullptr, *partial_l = nullptr, *partial_acc = nullptr;
    rc = axiom_cuda_attention_vec_scratch(
            runtime, partial_count, head_dim, &partial_m, &partial_l, &partial_acc);
    if (rc != AXIOM_OK) return rc;

    axiom_attention_core_paged_b1_vec_partial_f32_kernel<<<
            dim3(q_heads, active, tiles), AXIOM_ATTN_VEC_THREADS>>>(
            (const float *)((const uint8_t *)qb->ptr + q_offset),
            (const float *)((const uint8_t *)kb->ptr + k_offset),
            (const float *)((const uint8_t *)vb->ptr + v_offset),
            (const int32_t *)((const uint8_t *)tb->ptr + tbl_offset),
            (const uint32_t *)((const uint8_t *)cb->ptr + ct_offset),
            partial_m, partial_l, partial_acc,
            tbl_stride, n_layers, layer, block_tokens, pool_stride,
            q_heads, kv_heads, head_dim, partials_per_head);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);

    axiom_attention_core_paged_b1_vec_combine_f32_kernel<<<dim3(q_heads, active), head_dim>>>(
            partial_m, partial_l, partial_acc,
            (float *)((uint8_t *)ob->ptr + out_offset),
            q_heads, head_dim, partials_per_head);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

/* W5a: PAGED variant of axiom_attention_core_scale_blk_f32_kernel — identical to
 * axiom_attention_core_paged_blk_f32_kernel above except `scale` is a CALLER-SUPPLIED param
 * instead of rsqrtf(head_dim) (gemma-4 uses 1.0). Same block-table addressing, same layouts,
 * same per-seq window semantics; byte-identical to the contiguous scale kernel given identical
 * logical KV. */
__global__ static void axiom_attention_core_scale_paged_blk_f32_kernel(
        const float *q, const float *kpool, const float *vpool, float *out,
        const int32_t *block_tables, const uint32_t *cache_tokens,
        uint32_t tbl_stride, uint32_t NL, uint32_t L, uint32_t BT, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, float scale,
        uint32_t window) {
    const uint32_t a = (uint32_t)blockIdx.y;
    const uint32_t qh = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const uint32_t group = q_heads / kv_heads;
    const uint32_t kvh = qh / group;
    const uint32_t cache_t = cache_tokens[a];
    const uint32_t start = (window != 0u && cache_t > window) ? (cache_t - window) : 0u;
    const int32_t *tbl = block_tables + (uint64_t)a * tbl_stride;
    const uint64_t kvh_off = (uint64_t)kvh * head_dim;
    extern __shared__ float sh[];                  /* [head_dim] dot-reduction scratch */
    const float qd = q[((uint64_t)a * q_heads + qh) * head_dim + tid];
    float m = -3.4028234663852886e+38F;            /* running max */
    float l = 0.0f;                                 /* running denom */
    float acc = 0.0f;                               /* running output for dim tid */
    for (uint32_t t = start; t < cache_t; ++t) {
        const uint32_t pb = (uint32_t)tbl[t / BT];
        const uint64_t base =
                (((uint64_t)pb * NL + L) * BT + (t % BT)) * pool_stride + kvh_off;
        sh[tid] = qd * kpool[base + tid];
        __syncthreads();
        for (uint32_t s = head_dim >> 1; s > 0u; s >>= 1u) {
            if (tid < s) sh[tid] += sh[tid + s];
            __syncthreads();
        }
        const float score = sh[0] * scale;
        __syncthreads();
        const float m_new = fmaxf(m, score);
        const float corr = expf(m - m_new);
        const float w = expf(score - m_new);
        l = l * corr + w;
        acc = acc * corr + w * vpool[base + tid];
        m = m_new;
    }
    out[((uint64_t)a * q_heads + qh) * head_dim + tid] = l > 0.0f ? acc / l : 0.0f;
}

extern "C" int axiom_cuda_attention_core_scale_paged_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_pool, uint64_t k_offset,
        const void *v_pool, uint64_t v_offset,
        void *out, uint64_t out_offset,
        const void *block_tables, uint64_t tbl_offset,
        const void *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, float scale,
        uint32_t window) {
    if (!cuda_runtime || !q || !k_pool || !v_pool || !out || !block_tables || !cache_tokens ||
        active == 0 || active > 65535u || tbl_stride == 0 ||
        n_layers == 0 || layer >= n_layers || block_tokens == 0 ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        (q_heads % kv_heads) != 0 || (head_dim & (head_dim - 1u)) != 0u || head_dim > 1024u ||
        (uint64_t)pool_stride < (uint64_t)kv_heads * head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qb = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *kb = (const axiom_cuda_buffer *)k_pool;
    const axiom_cuda_buffer *vb = (const axiom_cuda_buffer *)v_pool;
    axiom_cuda_buffer *ob = (axiom_cuda_buffer *)out;
    const axiom_cuda_buffer *tb = (const axiom_cuda_buffer *)block_tables;
    const axiom_cuda_buffer *cb = (const axiom_cuda_buffer *)cache_tokens;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_attention_core_scale_paged_blk_f32_kernel<<<dim3(q_heads, active), head_dim,
            head_dim * sizeof(float)>>>(
            (const float *)((const uint8_t *)qb->ptr + q_offset),
            (const float *)((const uint8_t *)kb->ptr + k_offset),
            (const float *)((const uint8_t *)vb->ptr + v_offset),
            (float *)((uint8_t *)ob->ptr + out_offset),
            (const int32_t *)((const uint8_t *)tb->ptr + tbl_offset),
            (const uint32_t *)((const uint8_t *)cb->ptr + ct_offset),
            tbl_stride, n_layers, layer, block_tokens, pool_stride,
            q_heads, kv_heads, head_dim, scale, window);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

/* W5a: scatter ONE staged K row and ONE staged V row per active seq (k_stage/v_stage are
 * [A][kv_dim] each — the current token's k/v for layer L) into the paged pools at token
 * position pos[a], through the SAME per-seq block-table addressing as the paged attention
 * kernels: lb = pos/BT, slot = pos%BT, pb = block_tables[a*tbl_stride+lb], row base
 * (((uint64_t)pb*NL + L)*BT + slot)*pool_stride. pos is a DEVICE uint32 array [A] — positions
 * are PER SEQ (seqs in one launch may store at different positions). kv_dim may exceed the
 * thread-block size (up to 4096+): grid-stride loop over the row (blockIdx.x covers the row,
 * blockIdx.y = active-seq index a). Lanes [kv_dim, pool_stride) of the row are NOT written. */
__global__ static void axiom_kv_pool_store_paged_f32_kernel(
        const float *k_stage, const float *v_stage, float *kpool, float *vpool,
        const int32_t *block_tables, const uint32_t *pos,
        uint32_t tbl_stride, uint32_t NL, uint32_t L, uint32_t BT, uint32_t pool_stride,
        uint32_t kv_dim) {
    const uint32_t a = (uint32_t)blockIdx.y;
    const uint32_t p = pos[a];
    const uint32_t pb = (uint32_t)block_tables[(uint64_t)a * tbl_stride + p / BT];
    const uint64_t base = (((uint64_t)pb * NL + L) * BT + (p % BT)) * pool_stride;
    const uint64_t src = (uint64_t)a * kv_dim;
    const uint32_t step = (uint32_t)gridDim.x * blockDim.x;
    for (uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x; i < kv_dim; i += step) {
        kpool[base + i] = k_stage[src + i];
        vpool[base + i] = v_stage[src + i];
    }
}

extern "C" int axiom_cuda_kv_pool_store_paged_f32_device(
        void *cuda_runtime,
        const void *k_stage, uint64_t k_stage_offset,
        const void *v_stage, uint64_t v_stage_offset,
        void *k_pool, uint64_t k_pool_offset,
        void *v_pool, uint64_t v_pool_offset,
        const void *block_tables, uint64_t tbl_offset,
        const void *pos, uint64_t pos_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t kv_dim) {
    if (!cuda_runtime || !k_stage || !v_stage || !k_pool || !v_pool || !block_tables || !pos ||
        active == 0 || active > 65535u || tbl_stride == 0 ||
        n_layers == 0 || layer >= n_layers || block_tokens == 0 ||
        kv_dim == 0 || pool_stride < kv_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ksb = (const axiom_cuda_buffer *)k_stage;
    const axiom_cuda_buffer *vsb = (const axiom_cuda_buffer *)v_stage;
    axiom_cuda_buffer *kb = (axiom_cuda_buffer *)k_pool;
    axiom_cuda_buffer *vb = (axiom_cuda_buffer *)v_pool;
    const axiom_cuda_buffer *tb = (const axiom_cuda_buffer *)block_tables;
    const axiom_cuda_buffer *pb = (const axiom_cuda_buffer *)pos;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint32_t block = 256u;
    const uint32_t grid_x = (kv_dim + block - 1u) / block;
    axiom_kv_pool_store_paged_f32_kernel<<<dim3(grid_x, active), block>>>(
            (const float *)((const uint8_t *)ksb->ptr + k_stage_offset),
            (const float *)((const uint8_t *)vsb->ptr + v_stage_offset),
            (float *)((uint8_t *)kb->ptr + k_pool_offset),
            (float *)((uint8_t *)vb->ptr + v_pool_offset),
            (const int32_t *)((const uint8_t *)tb->ptr + tbl_offset),
            (const uint32_t *)((const uint8_t *)pb->ptr + pos_offset),
            tbl_stride, n_layers, layer, block_tokens, pool_stride, kv_dim);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

/* P2 STEP 4: device-RESIDENT GQA causal attention core. Launches the SAME validated
 * axiom_attention_cache_f32_kernel as the host-glue axiom_cuda_attention_cache_f32, but on
 * buffers ALREADY resident on device (d_q / d_kcache / d_vcache / d_att) — no per-call malloc
 * or H2D/D2H. This is the round-trip the host attention core costs per layer. cache_tokens =
 * pos+1 (causal-inclusive); cache layout token-major (t*kv_heads+kv_head)*head_dim. */
extern "C" int axiom_cuda_attention_core_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_cache, uint64_t k_offset,
        const void *v_cache, uint64_t v_offset,
        void *out, uint64_t out_offset,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        uint32_t cache_tokens) {
    if (!cuda_runtime || !q || !k_cache || !v_cache || !out ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 || cache_tokens == 0 ||
        (q_heads % kv_heads) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *qb = (const axiom_cuda_buffer *)q;
    const axiom_cuda_buffer *kb = (const axiom_cuda_buffer *)k_cache;
    const axiom_cuda_buffer *vb = (const axiom_cuda_buffer *)v_cache;
    axiom_cuda_buffer *ob = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((q_heads * head_dim + block - 1) / block);
    axiom_attention_cache_f32_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)qb->ptr + q_offset),
            (const float *)((const uint8_t *)kb->ptr + k_offset),
            (const float *)((const uint8_t *)vb->ptr + v_offset),
            (float *)((uint8_t *)ob->ptr + out_offset),
            q_heads, kv_heads, head_dim, cache_tokens);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_attention_cache_f32(
        void *cuda_runtime,
        const float *q_host,
        const float *k_cache_host,
        const float *v_cache_host,
        float *out_host,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        uint32_t cache_tokens) {
    if (!cuda_runtime || !q_host || !k_cache_host || !v_cache_host || !out_host ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 || cache_tokens == 0 ||
        (q_heads % kv_heads) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *q_dev = nullptr;
    float *k_dev = nullptr;
    float *v_dev = nullptr;
    float *out_dev = nullptr;
    const size_t q_bytes = (size_t)q_heads * head_dim * sizeof(float);
    const size_t kv_bytes = (size_t)cache_tokens * kv_heads * head_dim * sizeof(float);
    err = cudaMalloc(&q_dev, q_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&k_dev, kv_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&v_dev, kv_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, q_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(q_dev, q_host, q_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(k_dev, k_cache_host, kv_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(v_dev, v_cache_host, kv_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int block = 256;
        const int grid = (int)((q_heads * head_dim + block - 1) / block);
        axiom_attention_cache_f32_kernel<<<grid, block>>>(
                q_dev,
                k_dev,
                v_dev,
                out_dev,
                q_heads,
                kv_heads,
                head_dim,
                cache_tokens);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, q_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(v_dev);
    cudaFree(k_dev);
    cudaFree(q_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(v_dev);
    cudaFree(k_dev);
    cudaFree(q_dev);
    return axiom_cuda_status(err);
}

/* LayerNorm WITH bias (Nemotron) — mean-subtract + variance, weight + bias.
 * mean and variance accumulated in DOUBLE + (float)(1.0/sqrt(var+eps)), matching
 * the host layernorm() in axiom_nemotron.cpp byte-for-byte:
 *   out[i] = ((in[i]-mean)*inv)*w[i] + b[i]. Single block (<=256 threads). */
__global__ static void axiom_layernorm_f32_device_kernel(
        const float *weight,
        const float *bias,
        const float *input,
        float *out,
        uint32_t count,
        float eps) {
    __shared__ double sm[256];
    const uint32_t tid = threadIdx.x;
    double p = 0.0;
    for (uint32_t i = tid; i < count; i += blockDim.x) p += (double)input[i];
    sm[tid] = p;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    const double mean = sm[0] / (double)count;
    __syncthreads();
    double pv = 0.0;
    for (uint32_t i = tid; i < count; i += blockDim.x) {
        const double d = (double)input[i] - mean;
        pv += d * d;
    }
    sm[tid] = pv;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    const double var = sm[0] / (double)count;
    const float fmean = (float)mean;
    const float inv = (float)(1.0 / sqrt(var + (double)eps));
    for (uint32_t i = tid; i < count; i += blockDim.x)
        out[i] = ((input[i] - fmean) * inv) * weight[i] + bias[i];
}

extern "C" int axiom_cuda_layernorm_f32_device(
        void *cuda_runtime,
        const void *weight_f32, uint64_t weight_offset,
        const void *bias_f32, uint64_t bias_offset,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t count, float eps) {
    if (!cuda_runtime || !weight_f32 || !bias_f32 || !input || !out || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weight_f32;
    const axiom_cuda_buffer *bbuf = (const axiom_cuda_buffer *)bias_f32;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_layernorm_f32_device_kernel<<<1, 256>>>(
            (const float *)((const uint8_t *)wbuf->ptr + weight_offset),
            (const float *)((const uint8_t *)bbuf->ptr + bias_offset),
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            count, eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

/* ReLU^2 (squared ReLU), Nemotron sequential FFN: out[i] = (max(in[i],0))^2.
 * Matches the host relu2() byte-for-byte. */
__global__ static void axiom_relu2_f32_device_kernel(
        const float *input, float *out, uint32_t count) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = input[i] > 0.0f ? input[i] : 0.0f;
    out[i] = v * v;
}

extern "C" int axiom_cuda_relu2_f32_device(
        void *cuda_runtime,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t count) {
    if (!cuda_runtime || !input || !out || count == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((count + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_relu2_f32_device_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)ibuf->ptr + input_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            count);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

/* GeGLU (gelu-tanh gate) * up — matches the host geglu_mul/geglu math in the
 * gemma3 and gemma4 runners byte-for-byte in form:
 *   g = 0.5*x*(1 + tanh(k*(x + 0.044715*x^3))),  out = g*up,  k=0.7978845608028654
 * Computed in DOUBLE so tanh is precise and NOT degraded by --use_fast_math
 * (the same high-precision discipline as axiom_rmsnorm_f32_hp). Shared by gemma3
 * and gemma4 on-device FFN residency. */
__global__ static void axiom_geglu_mul_f32_device_kernel(
        const float *gate,
        const float *up,
        float *out,
        uint32_t count) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double k = 0.7978845608028654;
    const double x = (double)gate[i];
    const double inner = k * (x + 0.044715 * x * x * x);
    const double g = 0.5 * x * (1.0 + tanh(inner));
    out[i] = (float)(g * (double)up[i]);
}

extern "C" int axiom_cuda_geglu_mul_f32_device(
        void *cuda_runtime,
        const void *gate,
        uint64_t gate_offset,
        const void *up,
        uint64_t up_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count) {
    if (!cuda_runtime || !gate || !up || !out || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((count + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_geglu_mul_f32_device_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)gbuf->ptr + gate_offset),
            (const float *)((const uint8_t *)ubuf->ptr + up_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            count);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_silu_mul_f32_kernel(
        const float *gate,
        const float *up,
        float *out,
        uint32_t count,
        float clamp_abs) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float g = gate[i];
    float u = up[i];
    if (clamp_abs > 1.0e-6f) {
        g = fminf(g, clamp_abs);
        u = fminf(fmaxf(u, -clamp_abs), clamp_abs);
    }
    out[i] = (g / (1.0f + expf(-g))) * u;
}

extern "C" int axiom_cuda_silu_mul_f32_device(
        void *cuda_runtime,
        const void *gate,
        uint64_t gate_offset,
        const void *up,
        uint64_t up_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count) {
    if (!cuda_runtime || !gate || !up || !out || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((count + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_silu_mul_f32_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)gbuf->ptr + gate_offset),
            (const float *)((const uint8_t *)ubuf->ptr + up_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            count,
            0.0f);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_silu_mul_clamp_f32_device(
        void *cuda_runtime,
        const void *gate,
        uint64_t gate_offset,
        const void *up,
        uint64_t up_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count,
        float clamp_abs) {
    if (!cuda_runtime || !gate || !up || !out || count == 0 || clamp_abs < 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *gbuf = (const axiom_cuda_buffer *)gate;
    const axiom_cuda_buffer *ubuf = (const axiom_cuda_buffer *)up;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((count + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_silu_mul_f32_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)gbuf->ptr + gate_offset),
            (const float *)((const uint8_t *)ubuf->ptr + up_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            count,
            clamp_abs);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

__global__ static void axiom_weighted_sum_f32_kernel(
        const float *inputs,
        const float *weights,
        float *out,
        uint32_t slots,
        uint32_t count) {
    const uint32_t i = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < slots; ++slot) {
        acc += inputs[(uint64_t)slot * count + i] * weights[slot];
    }
    out[i] = acc;
}

extern "C" int axiom_cuda_weighted_sum_f32_device(
        void *cuda_runtime,
        const void *inputs,
        uint64_t inputs_offset,
        const void *weights,
        uint64_t weights_offset,
        void *out,
        uint64_t out_offset,
        uint32_t slots,
        uint32_t count) {
    if (!cuda_runtime || !inputs || !weights || !out || slots == 0 || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    const axiom_cuda_buffer *ibuf = (const axiom_cuda_buffer *)inputs;
    const axiom_cuda_buffer *wbuf = (const axiom_cuda_buffer *)weights;
    axiom_cuda_buffer *obuf = (axiom_cuda_buffer *)out;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const int block = 256;
    const int grid = (int)((count + (uint32_t)block - 1u) / (uint32_t)block);
    axiom_weighted_sum_f32_kernel<<<grid, block>>>(
            (const float *)((const uint8_t *)ibuf->ptr + inputs_offset),
            (const float *)((const uint8_t *)wbuf->ptr + weights_offset),
            (float *)((uint8_t *)obuf->ptr + out_offset),
            slots,
            count);
    err = cudaGetLastError();
    if (err != cudaSuccess) return axiom_cuda_status(err);
    return axiom_cuda_finish_after_launch(err);
}

extern "C" int axiom_cuda_silu_mul_f32(
        void *cuda_runtime,
        const float *gate_host,
        const float *up_host,
        float *out_host,
        uint32_t count) {
    if (!cuda_runtime || !gate_host || !up_host || !out_host || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *gate_dev = nullptr;
    float *up_dev = nullptr;
    float *out_dev = nullptr;
    const size_t bytes = (size_t)count * sizeof(float);
    err = cudaMalloc(&gate_dev, bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&up_dev, bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(gate_dev, gate_host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(up_dev, up_host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int block = 256;
        const int grid = (int)((count + block - 1) / block);
        axiom_silu_mul_f32_kernel<<<grid, block>>>(gate_dev, up_dev, out_dev, count, 0.0f);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(up_dev);
    cudaFree(gate_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(up_dev);
    cudaFree(gate_dev);
    return axiom_cuda_status(err);
}

__global__ static void axiom_topk_f32_kernel(
        const float *input,
        uint32_t count,
        uint32_t k,
        uint32_t *out_indices,
        float *out_values) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (uint32_t i = 0; i < k; ++i) {
        out_indices[i] = UINT32_MAX;
        out_values[i] = -3.4028234663852886e+38F;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const float v = input[i];
        uint32_t slot = k;
        for (uint32_t j = 0; j < k; ++j) {
            if (v > out_values[j]) {
                slot = j;
                break;
            }
        }
        if (slot == k) continue;
        for (uint32_t j = k - 1; j > slot; --j) {
            out_values[j] = out_values[j - 1];
            out_indices[j] = out_indices[j - 1];
        }
        out_values[slot] = v;
        out_indices[slot] = i;
    }
}

extern "C" int axiom_cuda_topk_f32(
        void *cuda_runtime,
        const float *input_host,
        uint32_t count,
        uint32_t k,
        uint32_t *out_indices_host,
        float *out_values_host) {
    if (!cuda_runtime || !input_host || !out_indices_host || !out_values_host ||
        count == 0 || k == 0 || k > count) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *input_dev = nullptr;
    uint32_t *idx_dev = nullptr;
    float *val_dev = nullptr;
    err = cudaMalloc(&input_dev, (size_t)count * sizeof(float));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&idx_dev, (size_t)k * sizeof(uint32_t));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&val_dev, (size_t)k * sizeof(float));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(input_dev, input_host, (size_t)count * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    axiom_topk_f32_kernel<<<1, 1>>>(input_dev, count, k, idx_dev, val_dev);

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_indices_host, idx_dev, (size_t)k * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_values_host, val_dev, (size_t)k * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(val_dev);
    cudaFree(idx_dev);
    cudaFree(input_dev);
    return AXIOM_OK;

fail:
    cudaFree(val_dev);
    cudaFree(idx_dev);
    cudaFree(input_dev);
    return axiom_cuda_status(err);
}

extern "C" int axiom_cuda_smoke_vector_add(
        void *cuda_runtime,
        const float *a_host,
        const float *b_host,
        float *out_host,
        size_t count) {
    if (!cuda_runtime || !a_host || !b_host || !out_host || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    float *a_dev = nullptr;
    float *b_dev = nullptr;
    float *out_dev = nullptr;
    const size_t bytes = count * sizeof(float);

    err = cudaMalloc(&a_dev, bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&b_dev, bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&out_dev, bytes);
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpy(a_dev, a_host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(b_dev, b_host, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    {
        const int block = 256;
        const int grid = (int)((count + block - 1) / block);
        axiom_vector_add_kernel<<<grid, block>>>(a_dev, b_dev, out_dev, count);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) goto fail;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(out_host, out_dev, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto fail;

    cudaFree(out_dev);
    cudaFree(b_dev);
    cudaFree(a_dev);
    return AXIOM_OK;

fail:
    cudaFree(out_dev);
    cudaFree(b_dev);
    cudaFree(a_dev);
    return axiom_cuda_status(err);
}

__global__ static void axiom_latent_link_f32_kernel(
        const float *source,
        const float *pre_ln_weight,
        const float *pre_ln_bias,
        const float *proj1_weight,
        const float *proj1_bias,
        const float *proj2_weight,
        const float *proj2_bias,
        const float *residual_weight,
        const float *residual_bias,
        const float *post_ln_weight,
        const float *post_ln_bias,
        float *target,
        float *scratch_norm,
        float *scratch_hidden,
        float *scratch_out,
        uint32_t rows,
        uint32_t source_width,
        uint32_t target_width,
        uint32_t hidden_width,
        uint32_t source_stride,
        uint32_t target_stride,
        float eps) {
    const uint32_t row = (uint32_t)blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;

    float mean = 0.0f;
    for (uint32_t i = 0; i < source_width; i++) {
        mean += source[(uint64_t)row * source_stride + i];
    }
    mean /= (float)source_width;

    float var = 0.0f;
    for (uint32_t i = 0; i < source_width; i++) {
        const float d = source[(uint64_t)row * source_stride + i] - mean;
        var += d * d;
    }
    const float inv_std = rsqrtf(var / (float)source_width + eps);

    float *norm = scratch_norm + (uint64_t)row * source_width;
    float *hidden = scratch_hidden + (uint64_t)row * hidden_width;
    float *out = scratch_out + (uint64_t)row * target_width;

    for (uint32_t i = 0; i < source_width; i++) {
        const float x = source[(uint64_t)row * source_stride + i];
        norm[i] = (x - mean) * inv_std * pre_ln_weight[i] + pre_ln_bias[i];
    }

    for (uint32_t h = 0; h < hidden_width; h++) {
        float acc = proj1_bias[h];
        const float *w = proj1_weight + (uint64_t)h * source_width;
        for (uint32_t i = 0; i < source_width; i++) acc += norm[i] * w[i];
        hidden[h] = 0.5f * acc * (1.0f + erff(acc * 0.7071067811865475f));
    }

    for (uint32_t j = 0; j < target_width; j++) {
        float acc = proj2_bias[j];
        const float *w = proj2_weight + (uint64_t)j * hidden_width;
        for (uint32_t h = 0; h < hidden_width; h++) acc += hidden[h] * w[h];

        float residual = 0.0f;
        if (residual_weight) {
            residual += residual_bias ? residual_bias[j] : 0.0f;
            const float *rw = residual_weight + (uint64_t)j * source_width;
            for (uint32_t i = 0; i < source_width; i++) {
                residual += source[(uint64_t)row * source_stride + i] * rw[i];
            }
        } else if (source_width == target_width) {
            residual = source[(uint64_t)row * source_stride + j];
        }
        out[j] = acc + residual;
    }

    float post_mean = 0.0f;
    for (uint32_t j = 0; j < target_width; j++) post_mean += out[j];
    post_mean /= (float)target_width;

    float post_var = 0.0f;
    for (uint32_t j = 0; j < target_width; j++) {
        const float d = out[j] - post_mean;
        post_var += d * d;
    }
    const float post_inv_std = rsqrtf(post_var / (float)target_width + eps);

    for (uint32_t j = 0; j < target_width; j++) {
        target[(uint64_t)row * target_stride + j] =
                (out[j] - post_mean) * post_inv_std * post_ln_weight[j] + post_ln_bias[j];
    }
}

static int axiom_cuda_copy_f32(float **dst, const float *src, size_t count) {
    if (!dst || !src || count == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    float *dev = nullptr;
    cudaError_t err = cudaMalloc(&dev, count * sizeof(float));
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    err = cudaMemcpy(dev, src, count * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(dev);
        return AXIOM_ERR_CUDA;
    }
    cudaFree(*dst);
    *dst = dev;
    return AXIOM_OK;
}

extern "C" int axiom_cuda_latent_link_create(
        void *cuda_runtime,
        void **out,
        axiom_latent_link_kind kind,
        axiom_latent_dtype dtype,
        uint32_t source_width,
        uint32_t target_width,
        uint32_t hidden_width,
        float eps) {
    if (!cuda_runtime || !out || source_width == 0 || target_width == 0 ||
        hidden_width == 0 || dtype != AXIOM_LATENT_F32 ||
        (kind != AXIOM_LATENT_LINK_INNER && kind != AXIOM_LATENT_LINK_OUTER)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    axiom_cuda_latent_link *link =
            (axiom_cuda_latent_link *)std::calloc(1, sizeof(*link));
    if (!link) return AXIOM_ERR_RUNTIME;
    link->runtime = runtime;
    link->kind = kind;
    link->dtype = dtype;
    link->source_width = source_width;
    link->target_width = target_width;
    link->hidden_width = hidden_width;
    link->eps = eps > 0.0f ? eps : 1.0e-5f;
    *out = link;
    return AXIOM_OK;
}

extern "C" void axiom_cuda_latent_link_destroy(void *cuda_link) {
    if (!cuda_link) return;
    axiom_cuda_latent_link *link = (axiom_cuda_latent_link *)cuda_link;
    cudaFree(link->pre_ln_weight);
    cudaFree(link->pre_ln_bias);
    cudaFree(link->proj1_weight);
    cudaFree(link->proj1_bias);
    cudaFree(link->proj2_weight);
    cudaFree(link->proj2_bias);
    cudaFree(link->residual_weight);
    cudaFree(link->residual_bias);
    cudaFree(link->post_ln_weight);
    cudaFree(link->post_ln_bias);
    cudaFree(link->scratch_norm);
    cudaFree(link->scratch_hidden);
    cudaFree(link->scratch_out);
    std::free(link);
}

extern "C" int axiom_cuda_latent_link_load_f32(
        void *cuda_link,
        const axiom_latent_link_weights_f32 *weights) {
    if (!cuda_link || !weights || weights->abi_version != AXIOM_ABI_VERSION ||
        weights->dtype != AXIOM_LATENT_F32) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_latent_link *link = (axiom_cuda_latent_link *)cuda_link;
    cudaError_t err = cudaSetDevice(link->runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    int rc = AXIOM_OK;
    if ((rc = axiom_cuda_copy_f32(&link->pre_ln_weight, weights->pre_ln_weight, link->source_width)) != AXIOM_OK) return rc;
    if ((rc = axiom_cuda_copy_f32(&link->pre_ln_bias, weights->pre_ln_bias, link->source_width)) != AXIOM_OK) return rc;
    if ((rc = axiom_cuda_copy_f32(&link->proj1_weight, weights->proj1_weight, (size_t)link->hidden_width * link->source_width)) != AXIOM_OK) return rc;
    if ((rc = axiom_cuda_copy_f32(&link->proj1_bias, weights->proj1_bias, link->hidden_width)) != AXIOM_OK) return rc;
    if ((rc = axiom_cuda_copy_f32(&link->proj2_weight, weights->proj2_weight, (size_t)link->target_width * link->hidden_width)) != AXIOM_OK) return rc;
    if ((rc = axiom_cuda_copy_f32(&link->proj2_bias, weights->proj2_bias, link->target_width)) != AXIOM_OK) return rc;
    if ((rc = axiom_cuda_copy_f32(&link->post_ln_weight, weights->post_ln_weight, link->target_width)) != AXIOM_OK) return rc;
    if ((rc = axiom_cuda_copy_f32(&link->post_ln_bias, weights->post_ln_bias, link->target_width)) != AXIOM_OK) return rc;
    if (weights->residual_weight) {
        if ((rc = axiom_cuda_copy_f32(&link->residual_weight, weights->residual_weight, (size_t)link->target_width * link->source_width)) != AXIOM_OK) return rc;
    }
    if (weights->residual_bias) {
        if ((rc = axiom_cuda_copy_f32(&link->residual_bias, weights->residual_bias, link->target_width)) != AXIOM_OK) return rc;
    }
    return AXIOM_OK;
}

extern "C" int axiom_cuda_latent_link_apply(
        void *cuda_runtime,
        void *cuda_link,
        const void *source,
        void *target,
        uint32_t rows,
        uint32_t source_stride,
        uint32_t target_stride) {
    if (!cuda_runtime || !cuda_link || !source || !target || rows == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_latent_link *link = (axiom_cuda_latent_link *)cuda_link;
    if (source_stride < link->source_width || target_stride < link->target_width ||
        !link->pre_ln_weight || !link->pre_ln_bias || !link->proj1_weight ||
        !link->proj1_bias || !link->proj2_weight || !link->proj2_bias ||
        !link->post_ln_weight || !link->post_ln_bias) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    axiom_cuda_runtime *runtime = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(runtime->device);
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;

    if (link->scratch_rows < rows) {
        cudaFree(link->scratch_norm);
        cudaFree(link->scratch_hidden);
        cudaFree(link->scratch_out);
        link->scratch_norm = nullptr;
        link->scratch_hidden = nullptr;
        link->scratch_out = nullptr;
        link->scratch_rows = 0;
        err = cudaMalloc(&link->scratch_norm, (size_t)rows * link->source_width * sizeof(float));
        if (err != cudaSuccess) return AXIOM_ERR_CUDA;
        err = cudaMalloc(&link->scratch_hidden, (size_t)rows * link->hidden_width * sizeof(float));
        if (err != cudaSuccess) return AXIOM_ERR_CUDA;
        err = cudaMalloc(&link->scratch_out, (size_t)rows * link->target_width * sizeof(float));
        if (err != cudaSuccess) return AXIOM_ERR_CUDA;
        link->scratch_rows = rows;
    }

    const dim3 block(1);
    const dim3 grid(rows);
    axiom_latent_link_f32_kernel<<<grid, block>>>(
            (const float *)source,
            link->pre_ln_weight,
            link->pre_ln_bias,
            link->proj1_weight,
            link->proj1_bias,
            link->proj2_weight,
            link->proj2_bias,
            link->residual_weight,
            link->residual_bias,
            link->post_ln_weight,
            link->post_ln_bias,
            (float *)target,
            link->scratch_norm,
            link->scratch_hidden,
            link->scratch_out,
            rows,
            link->source_width,
            link->target_width,
            link->hidden_width,
            source_stride,
            target_stride,
            link->eps);
    err = cudaGetLastError();
    if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    err = cudaDeviceSynchronize();
    return axiom_cuda_status(err);
}

/* ============================================================================
 * Gated DeltaNet (Qwen3.6 / Qwen3-Next) - fp32 state in global.
 * Kernel bodies validated standalone vs tools/deltanet_ref.py (bit-compatible
 * with transformers + llama.cpp). See tools/axiom_deltanet_kernels.cu.
 * ========================================================================= */
__global__ static void axiom_deltanet_conv1d_silu_kernel(
        const float *__restrict__ in, const float *__restrict__ w,
        float *__restrict__ ring, float *__restrict__ out, uint32_t conv_dim) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float acc = w[c*4+0]*ring[c*3+0] + w[c*4+1]*ring[c*3+1] + w[c*4+2]*ring[c*3+2] + w[c*4+3]*in[c];
    out[c] = acc / (1.0f + expf(-acc));
    ring[c*3+0] = ring[c*3+1]; ring[c*3+1] = ring[c*3+2]; ring[c*3+2] = in[c];
}
__global__ static void axiom_deltanet_recurrence_kernel(
        const float *__restrict__ qc, const float *__restrict__ kc, const float *__restrict__ vc,
        const float *__restrict__ g, const float *__restrict__ beta,
        float *__restrict__ state, float *__restrict__ out, uint32_t nkh, uint32_t hd) {
    const uint32_t h = blockIdx.x, v = threadIdx.x;
    if (v >= hd) return;
    /* GQA repeat: transformers repeat_interleave -> v-head h uses k-head h/n_rep
     * (n_rep = n_v_heads/n_k_heads). Safetensors stores heads interleaved-paired;
     * the GGUF runner de-interleaves and uses h%nkh instead. For n_rep==1 both agree. */
    const uint32_t kh = h / (gridDim.x / nkh);
    const float *qv = qc + (size_t)kh*hd, *kv = kc + (size_t)kh*hd, *vin = vc + (size_t)h*hd;
    float *Sh = state + (size_t)h*hd*hd;
    extern __shared__ float dn_rsm[];
    float *qq = dn_rsm, *kk = dn_rsm + hd, *sums = dn_rsm + 2*hd;
    const float qval = qv[v], kval = kv[v], vreg = vin[v];
    sums[v] = qval*qval; __syncthreads();
    for (uint32_t s = hd>>1; s; s >>= 1) { if (v < s) sums[v] += sums[v+s]; __syncthreads(); }
    const float iq = rsqrtf(sums[0] + 1e-6f); __syncthreads();
    sums[v] = kval*kval; __syncthreads();
    for (uint32_t s = hd>>1; s; s >>= 1) { if (v < s) sums[v] += sums[v+s]; __syncthreads(); }
    const float ik = rsqrtf(sums[0] + 1e-6f); __syncthreads();
    qq[v] = qval * iq * rsqrtf((float)hd);
    kk[v] = kval * ik;
    __syncthreads();
    const float gt = expf(g[h]), bt = beta[h];
    for (uint32_t k = 0; k < hd; ++k) Sh[k*hd+v] *= gt;
    float kvm = 0.0f;
    for (uint32_t k = 0; k < hd; ++k) kvm += Sh[k*hd+v] * kk[k];
    const float dl = (vreg - kvm) * bt;
    for (uint32_t k = 0; k < hd; ++k) Sh[k*hd+v] += kk[k] * dl;
    float o = 0.0f;
    for (uint32_t k = 0; k < hd; ++k) o += Sh[k*hd+v] * qq[k];
    out[(size_t)h*hd+v] = o;
}
__global__ static void axiom_deltanet_gated_norm_kernel(
        const float *__restrict__ o, const float *__restrict__ wnorm, const float *__restrict__ z,
        float *__restrict__ out, uint32_t hd, float eps) {
    const uint32_t h = blockIdx.x, v = threadIdx.x;
    if (v >= hd) return;
    const float *oh = o + (size_t)h*hd;
    extern __shared__ float dn_gsm[];
    const float ov = oh[v];
    dn_gsm[v] = ov*ov; __syncthreads();
    for (uint32_t s = hd>>1; s; s >>= 1) { if (v < s) dn_gsm[v] += dn_gsm[v+s]; __syncthreads(); }
    const float inv = rsqrtf(dn_gsm[0]/(float)hd + eps);
    const float zv = z[(size_t)h*hd+v];
    out[(size_t)h*hd+v] = ov * inv * wnorm[v] * (zv / (1.0f + expf(-zv)));
}

static bool axiom_cuda_u32_is_power_of_two(uint32_t x) {
    return x != 0 && (x & (x - 1u)) == 0;
}

static bool axiom_cuda_mul_u64_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return false;
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool axiom_cuda_buffer_span_ok(const void *buffer, uint64_t offset, uint64_t bytes) {
    if (!buffer) return false;
    const axiom_cuda_buffer *b = (const axiom_cuda_buffer *)buffer;
    return offset <= b->bytes && bytes <= b->bytes - offset;
}

extern "C" int axiom_cuda_deltanet_conv1d_silu_f32_device(
        void *cuda_runtime, const void *in, uint64_t in_off, const void *w, uint64_t w_off,
        void *ring, uint64_t ring_off, void *out, uint64_t out_off, uint32_t conv_dim) {
    if (!cuda_runtime || !in || !w || !ring || !out || conv_dim == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t conv_bytes = 0, weight_bytes = 0, ring_bytes = 0;
    if (!axiom_cuda_mul_u64_checked((uint64_t)conv_dim, sizeof(float), &conv_bytes) ||
        !axiom_cuda_mul_u64_checked(conv_bytes, 4u, &weight_bytes) ||
        !axiom_cuda_mul_u64_checked(conv_bytes, 3u, &ring_bytes) ||
        !axiom_cuda_buffer_span_ok(in, in_off, conv_bytes) ||
        !axiom_cuda_buffer_span_ok(w, w_off, weight_bytes) ||
        !axiom_cuda_buffer_span_ok(ring, ring_off, ring_bytes) ||
        !axiom_cuda_buffer_span_ok(out, out_off, conv_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *rt = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(rt->device); if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const float *pin = (const float *)((const uint8_t *)((const axiom_cuda_buffer *)in)->ptr + in_off);
    const float *pw  = (const float *)((const uint8_t *)((const axiom_cuda_buffer *)w)->ptr + w_off);
    float *pring = (float *)((uint8_t *)((axiom_cuda_buffer *)ring)->ptr + ring_off);
    float *pout  = (float *)((uint8_t *)((axiom_cuda_buffer *)out)->ptr + out_off);
    axiom_deltanet_conv1d_silu_kernel<<<(conv_dim+255)/256, 256>>>(pin, pw, pring, pout, conv_dim);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}
extern "C" int axiom_cuda_deltanet_recurrence_f32_device(
        void *cuda_runtime, const void *qkv, uint64_t q_off, uint64_t k_off, uint64_t v_off,
        const void *g, uint64_t g_off, const void *beta, uint64_t beta_off,
        void *state, uint64_t state_off, void *out, uint64_t out_off,
        uint32_t n_v_heads, uint32_t n_k_heads, uint32_t head_dim) {
    if (!cuda_runtime || !qkv || !g || !beta || !state || !out ||
        n_v_heads == 0 || n_k_heads == 0 || head_dim == 0 || head_dim > 1024 ||
        !axiom_cuda_u32_is_power_of_two(head_dim) ||
        n_v_heads < n_k_heads || (n_v_heads % n_k_heads) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t kh_items = 0, vh_items = 0, kh_bytes = 0, vh_bytes = 0;
    uint64_t gate_bytes = 0, state_items = 0, state_bytes = 0;
    if (!axiom_cuda_mul_u64_checked((uint64_t)n_k_heads, (uint64_t)head_dim, &kh_items) ||
        !axiom_cuda_mul_u64_checked((uint64_t)n_v_heads, (uint64_t)head_dim, &vh_items) ||
        !axiom_cuda_mul_u64_checked(kh_items, sizeof(float), &kh_bytes) ||
        !axiom_cuda_mul_u64_checked(vh_items, sizeof(float), &vh_bytes) ||
        !axiom_cuda_mul_u64_checked((uint64_t)n_v_heads, sizeof(float), &gate_bytes) ||
        !axiom_cuda_mul_u64_checked(vh_items, (uint64_t)head_dim, &state_items) ||
        !axiom_cuda_mul_u64_checked(state_items, sizeof(float), &state_bytes) ||
        !axiom_cuda_buffer_span_ok(qkv, q_off, kh_bytes) ||
        !axiom_cuda_buffer_span_ok(qkv, k_off, kh_bytes) ||
        !axiom_cuda_buffer_span_ok(qkv, v_off, vh_bytes) ||
        !axiom_cuda_buffer_span_ok(g, g_off, gate_bytes) ||
        !axiom_cuda_buffer_span_ok(beta, beta_off, gate_bytes) ||
        !axiom_cuda_buffer_span_ok(state, state_off, state_bytes) ||
        !axiom_cuda_buffer_span_ok(out, out_off, vh_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *rt = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(rt->device); if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint8_t *qkvp = (const uint8_t *)((const axiom_cuda_buffer *)qkv)->ptr;
    const float *pq = (const float *)(qkvp + q_off);
    const float *pk = (const float *)(qkvp + k_off);
    const float *pv = (const float *)(qkvp + v_off);
    const float *pg    = (const float *)((const uint8_t *)((const axiom_cuda_buffer *)g)->ptr + g_off);
    const float *pbeta = (const float *)((const uint8_t *)((const axiom_cuda_buffer *)beta)->ptr + beta_off);
    float *pstate = (float *)((uint8_t *)((axiom_cuda_buffer *)state)->ptr + state_off);
    float *pout   = (float *)((uint8_t *)((axiom_cuda_buffer *)out)->ptr + out_off);
    const size_t shmem = (size_t)3*head_dim*sizeof(float);
    axiom_deltanet_recurrence_kernel<<<n_v_heads, head_dim, shmem>>>(pq, pk, pv, pg, pbeta, pstate, pout, n_k_heads, head_dim);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}
extern "C" int axiom_cuda_deltanet_gated_norm_f32_device(
        void *cuda_runtime, const void *o, uint64_t o_off, const void *wnorm, uint64_t wnorm_off,
        const void *z, uint64_t z_off, void *out, uint64_t out_off,
        uint32_t n_v_heads, uint32_t head_dim, float eps) {
    if (!cuda_runtime || !o || !wnorm || !z || !out ||
        n_v_heads == 0 || head_dim == 0 || head_dim > 1024 ||
        !axiom_cuda_u32_is_power_of_two(head_dim) || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t head_bytes = 0, out_items = 0, out_bytes = 0;
    if (!axiom_cuda_mul_u64_checked((uint64_t)head_dim, sizeof(float), &head_bytes) ||
        !axiom_cuda_mul_u64_checked((uint64_t)n_v_heads, (uint64_t)head_dim, &out_items) ||
        !axiom_cuda_mul_u64_checked(out_items, sizeof(float), &out_bytes) ||
        !axiom_cuda_buffer_span_ok(o, o_off, out_bytes) ||
        !axiom_cuda_buffer_span_ok(wnorm, wnorm_off, head_bytes) ||
        !axiom_cuda_buffer_span_ok(z, z_off, out_bytes) ||
        !axiom_cuda_buffer_span_ok(out, out_off, out_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cuda_runtime *rt = (axiom_cuda_runtime *)cuda_runtime;
    cudaError_t err = cudaSetDevice(rt->device); if (err != cudaSuccess) return AXIOM_ERR_CUDA;
    const float *po  = (const float *)((const uint8_t *)((const axiom_cuda_buffer *)o)->ptr + o_off);
    const float *pwn = (const float *)((const uint8_t *)((const axiom_cuda_buffer *)wnorm)->ptr + wnorm_off);
    const float *pz  = (const float *)((const uint8_t *)((const axiom_cuda_buffer *)z)->ptr + z_off);
    float *pout = (float *)((uint8_t *)((axiom_cuda_buffer *)out)->ptr + out_off);
    const size_t shmem = (size_t)head_dim*sizeof(float);
    axiom_deltanet_gated_norm_kernel<<<n_v_heads, head_dim, shmem>>>(po, pwn, pz, pout, head_dim, eps);
    err = cudaGetLastError();
    return axiom_cuda_finish_after_launch(err);
}

/* ---- Native NVFP4/FP8 matvec device adapters ------------------------------------
 * Bridge opaque axiom_device_buffer backend handles (+ byte offsets) to the raw-pointer
 * kernels defined in axiom_cuda_nvfp4.cu. The runtime wrappers
 * (axiom_runtime_{e2m1_nvfp4,fp8_e4m3_e8m0}_matvec_f32_device) validate args/bounds and
 * forward backend_buffer pointers here; we cast to axiom_cuda_buffer and apply ptr+offset.
 * The underlying kernels do their own cudaSetDevice + synchronize. */
extern "C" int axiom_cuda_e2m1_nvfp4_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);
extern "C" int axiom_cuda_fp8_e4m3_e8m0_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);

extern "C" int axiom_cuda_e2m1_nvfp4_matvec_f32_device(
        void *cuda_runtime,
        const void *weight, uint64_t weight_offset,
        const void *block_scale, uint64_t block_scale_offset, float global_scale,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset, uint32_t rows, uint32_t cols) {
    (void)cuda_runtime;
    if (!weight || !block_scale || !input || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    const axiom_cuda_buffer *wb = (const axiom_cuda_buffer *)weight;
    const axiom_cuda_buffer *sb = (const axiom_cuda_buffer *)block_scale;
    const axiom_cuda_buffer *ib = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *ob = (axiom_cuda_buffer *)out;
    return axiom_cuda_e2m1_nvfp4_matvec_f32(
            wb->device,
            (const uint8_t *)wb->ptr + weight_offset,
            (const uint8_t *)sb->ptr + block_scale_offset,
            global_scale,
            (const float *)((const uint8_t *)ib->ptr + input_offset),
            (float *)((uint8_t *)ob->ptr + out_offset),
            rows, cols);
}

extern "C" int axiom_cuda_fp8_e4m3_e8m0_matvec_f32_device(
        void *cuda_runtime,
        const void *weight, uint64_t weight_offset,
        const void *block_scale, uint64_t block_scale_offset,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset, uint32_t rows, uint32_t cols) {
    (void)cuda_runtime;
    if (!weight || !block_scale || !input || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    const axiom_cuda_buffer *wb = (const axiom_cuda_buffer *)weight;
    const axiom_cuda_buffer *sb = (const axiom_cuda_buffer *)block_scale;
    const axiom_cuda_buffer *ib = (const axiom_cuda_buffer *)input;
    axiom_cuda_buffer *ob = (axiom_cuda_buffer *)out;
    return axiom_cuda_fp8_e4m3_e8m0_matvec_f32(
            wb->device,
            (const uint8_t *)wb->ptr + weight_offset,
            (const uint8_t *)sb->ptr + block_scale_offset,
            (const float *)((const uint8_t *)ib->ptr + input_offset),
            (float *)((uint8_t *)ob->ptr + out_offset),
            rows, cols);
}

extern "C" int axiom_cuda_axpby_f32(int device, float *out, const float *in,
                                    float alpha, float beta, uint32_t n);
extern "C" int axiom_cuda_axpby_f32_device(
        void *cuda_runtime, void *out, uint64_t out_offset,
        const void *in, uint64_t in_offset, float alpha, float beta, uint32_t n) {
    (void)cuda_runtime;
    if (!out || !in) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cuda_buffer *ob = (axiom_cuda_buffer *)out;
    const axiom_cuda_buffer *ib = (const axiom_cuda_buffer *)in;
    return axiom_cuda_axpby_f32(
            ob->device,
            (float *)((uint8_t *)ob->ptr + out_offset),
            (const float *)((const uint8_t *)ib->ptr + in_offset),
            alpha, beta, n);
}
