/* Native Axiom SwiGLU composition for Unsloth's eight FP8 Qwen3.8 MLPs. */
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <new>
#include <string>

#include "axiom/axiom.h"
#include "axiom/qwen38_fp8.h"
#include "axiom/qwen38_fp8_mlp.h"
#include "axiom/qwen38_nvfp4_mlp.h"

namespace {

__global__ void silu_mul_batch_kernel(const float *gate, const float *up, float *out, uint32_t count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = gate[index];
    out[index] = (x / (1.0f + expf(-x))) * up[index];
}

int load_projection(
        axiom_model *model,
        int device,
        uint32_t layer,
        const char *projection,
        axiom_qwen38_fp8_linear **out) {
    char prefix[192];
    std::snprintf(prefix, sizeof(prefix), "model.language_model.layers.%u.mlp.%s_proj", layer, projection);
    const std::string base(prefix);
    return axiom_qwen38_fp8_linear_load(model, device,
                                         (base + ".weight").c_str(),
                                         (base + ".weight_scale").c_str(), out);
}

}  // namespace

struct axiom_qwen38_fp8_mlp {
    int device = -1;
    uint32_t layer = 0;
    axiom_qwen38_fp8_linear *gate = nullptr;
    axiom_qwen38_fp8_linear *up = nullptr;
    axiom_qwen38_fp8_linear *down = nullptr;
    float *gate_out = nullptr;
    float *up_out = nullptr;
    float *mid = nullptr;
    uint64_t device_bytes = 0;
};

extern "C" int axiom_qwen38_fp8_mlp_load(
        axiom_model *model,
        int device,
        uint32_t layer,
        axiom_qwen38_fp8_mlp **out) {
    if (out) *out = nullptr;
    if (!model || !out || device < 0 || layer < 56u || layer > 63u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen38_fp8_mlp *mlp = new (std::nothrow) axiom_qwen38_fp8_mlp();
    if (!mlp) return AXIOM_ERR_BUDGET;
    mlp->device = device;
    mlp->layer = layer;
    int rc = load_projection(model, device, layer, "gate", &mlp->gate);
    if (rc == AXIOM_OK) rc = load_projection(model, device, layer, "up", &mlp->up);
    if (rc == AXIOM_OK) rc = load_projection(model, device, layer, "down", &mlp->down);
    if (rc == AXIOM_OK &&
        (axiom_qwen38_fp8_linear_rows(mlp->gate) != AXIOM_QWEN38_NVFP4_FFN ||
         axiom_qwen38_fp8_linear_cols(mlp->gate) != AXIOM_QWEN38_NVFP4_HIDDEN ||
         axiom_qwen38_fp8_linear_rows(mlp->up) != AXIOM_QWEN38_NVFP4_FFN ||
         axiom_qwen38_fp8_linear_cols(mlp->up) != AXIOM_QWEN38_NVFP4_HIDDEN ||
         axiom_qwen38_fp8_linear_rows(mlp->down) != AXIOM_QWEN38_NVFP4_HIDDEN ||
         axiom_qwen38_fp8_linear_cols(mlp->down) != AXIOM_QWEN38_NVFP4_FFN)) {
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }
    const size_t ffn_batch_bytes = static_cast<size_t>(AXIOM_QWEN38_NVFP4_FFN) *
            AXIOM_QWEN38_FP8_BATCH * sizeof(float);
    cudaError_t cuda_status = cudaSuccess;
    if (rc == AXIOM_OK) cuda_status = cudaMalloc(&mlp->gate_out, ffn_batch_bytes);
    if (rc == AXIOM_OK && cuda_status == cudaSuccess) cuda_status = cudaMalloc(&mlp->up_out, ffn_batch_bytes);
    if (rc == AXIOM_OK && cuda_status == cudaSuccess) cuda_status = cudaMalloc(&mlp->mid, ffn_batch_bytes);
    if (rc == AXIOM_OK && cuda_status != cudaSuccess) {
        rc = cuda_status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
    }
    if (rc != AXIOM_OK) {
        axiom_qwen38_fp8_mlp_destroy(mlp);
        return rc;
    }
    mlp->device_bytes = axiom_qwen38_fp8_linear_device_bytes(mlp->gate) +
            axiom_qwen38_fp8_linear_device_bytes(mlp->up) +
            axiom_qwen38_fp8_linear_device_bytes(mlp->down) + 3u * ffn_batch_bytes;
    *out = mlp;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_fp8_mlp_destroy(axiom_qwen38_fp8_mlp *mlp) {
    if (!mlp) return;
    if (mlp->device >= 0) (void)cudaSetDevice(mlp->device);
    if (mlp->mid) (void)cudaFree(mlp->mid);
    if (mlp->up_out) (void)cudaFree(mlp->up_out);
    if (mlp->gate_out) (void)cudaFree(mlp->gate_out);
    axiom_qwen38_fp8_linear_destroy(mlp->down);
    axiom_qwen38_fp8_linear_destroy(mlp->up);
    axiom_qwen38_fp8_linear_destroy(mlp->gate);
    delete mlp;
}

extern "C" uint64_t axiom_qwen38_fp8_mlp_device_bytes(const axiom_qwen38_fp8_mlp *mlp) {
    return mlp ? mlp->device_bytes : 0u;
}

extern "C" int axiom_qwen38_fp8_mlp_forward_f32_device(
        axiom_qwen38_fp8_mlp *mlp,
        const float *input,
        float *out,
        void *stream) {
    if (!mlp || !input || !out || !mlp->gate || !mlp->up || !mlp->down ||
        !mlp->gate_out || !mlp->up_out || !mlp->mid) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(mlp->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    int rc = axiom_qwen38_fp8_linear_forward_f32_device(mlp->gate, input, mlp->gate_out, stream);
    if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_f32_device(mlp->up, input, mlp->up_out, stream);
    if (rc != AXIOM_OK) return rc;
    constexpr uint32_t threads = 256u;
    const uint32_t count = AXIOM_QWEN38_NVFP4_FFN * AXIOM_QWEN38_FP8_BATCH;
    silu_mul_batch_kernel<<<(count + threads - 1u) / threads, threads, 0,
                             static_cast<cudaStream_t>(stream)>>>(mlp->gate_out, mlp->up_out, mlp->mid, count);
    if (cudaPeekAtLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    return axiom_qwen38_fp8_linear_forward_f32_device(mlp->down, mlp->mid, out, stream);
}
