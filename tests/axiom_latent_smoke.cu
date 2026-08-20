#include "axiom/axiom.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "axiom-latent-smoke: %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

static void cpu_latent_link(
        const std::vector<float> &source,
        std::vector<float> &target,
        uint32_t rows,
        uint32_t source_width,
        uint32_t target_width,
        uint32_t hidden_width,
        uint32_t source_stride,
        uint32_t target_stride,
        const std::vector<float> &pre_ln_weight,
        const std::vector<float> &pre_ln_bias,
        const std::vector<float> &proj1_weight,
        const std::vector<float> &proj1_bias,
        const std::vector<float> &proj2_weight,
        const std::vector<float> &proj2_bias,
        const std::vector<float> &residual_weight,
        const std::vector<float> &residual_bias,
        const std::vector<float> &post_ln_weight,
        const std::vector<float> &post_ln_bias,
        float eps) {
    std::vector<float> norm(source_width);
    std::vector<float> hidden(hidden_width);
    std::vector<float> out(target_width);
    for (uint32_t r = 0; r < rows; r++) {
        float mean = 0.0f;
        for (uint32_t i = 0; i < source_width; i++) mean += source[(size_t)r * source_stride + i];
        mean /= (float)source_width;

        float var = 0.0f;
        for (uint32_t i = 0; i < source_width; i++) {
            const float d = source[(size_t)r * source_stride + i] - mean;
            var += d * d;
        }
        const float inv_std = 1.0f / sqrtf(var / (float)source_width + eps);

        for (uint32_t i = 0; i < source_width; i++) {
            const float x = source[(size_t)r * source_stride + i];
            norm[i] = (x - mean) * inv_std * pre_ln_weight[i] + pre_ln_bias[i];
        }

        for (uint32_t h = 0; h < hidden_width; h++) {
            float acc = proj1_bias[h];
            for (uint32_t i = 0; i < source_width; i++) {
                acc += norm[i] * proj1_weight[(size_t)h * source_width + i];
            }
            hidden[h] = gelu(acc);
        }

        for (uint32_t j = 0; j < target_width; j++) {
            float acc = proj2_bias[j];
            for (uint32_t h = 0; h < hidden_width; h++) {
                acc += hidden[h] * proj2_weight[(size_t)j * hidden_width + h];
            }
            float residual = residual_bias[j];
            for (uint32_t i = 0; i < source_width; i++) {
                residual += source[(size_t)r * source_stride + i] *
                        residual_weight[(size_t)j * source_width + i];
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
        const float post_inv_std = 1.0f / sqrtf(post_var / (float)target_width + eps);

        for (uint32_t j = 0; j < target_width; j++) {
            target[(size_t)r * target_stride + j] =
                    (out[j] - post_mean) * post_inv_std * post_ln_weight[j] + post_ln_bias[j];
        }
    }
}

int main(void) {
    axiom_config runtime_config{};
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = 0;

    axiom_runtime *runtime = nullptr;
    int rc = axiom_runtime_create(&runtime, &runtime_config);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "axiom-latent-smoke: runtime create failed: %s\n", axiom_status_string(rc));
        return 1;
    }

    const uint32_t rows = 4;
    const uint32_t source_width = 6;
    const uint32_t hidden_width = 8;
    const uint32_t target_width = 10;
    const uint32_t source_stride = 8;
    const uint32_t target_stride = 12;
    const float eps = 1.0e-5f;

    axiom_latent_link_config link_config{};
    link_config.abi_version = AXIOM_ABI_VERSION;
    link_config.kind = AXIOM_LATENT_LINK_OUTER;
    link_config.dtype = AXIOM_LATENT_F32;
    link_config.source_width = source_width;
    link_config.target_width = target_width;
    link_config.hidden_width = hidden_width;
    link_config.rank = 0;
    link_config.eps = eps;

    axiom_latent_link *link = nullptr;
    rc = axiom_latent_link_create(runtime, &link, &link_config);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "axiom-latent-smoke: link create failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    std::vector<float> pre_ln_weight(source_width);
    std::vector<float> pre_ln_bias(source_width);
    std::vector<float> proj1_weight((size_t)hidden_width * source_width);
    std::vector<float> proj1_bias(hidden_width);
    std::vector<float> proj2_weight((size_t)target_width * hidden_width);
    std::vector<float> proj2_bias(target_width);
    std::vector<float> residual_weight((size_t)target_width * source_width);
    std::vector<float> residual_bias(target_width);
    std::vector<float> post_ln_weight(target_width);
    std::vector<float> post_ln_bias(target_width);

    for (uint32_t i = 0; i < source_width; i++) {
        pre_ln_weight[i] = 0.9f + 0.03f * (float)i;
        pre_ln_bias[i] = -0.04f + 0.01f * (float)i;
    }
    for (uint32_t h = 0; h < hidden_width; h++) {
        proj1_bias[h] = -0.03f + 0.02f * (float)h;
        for (uint32_t i = 0; i < source_width; i++) {
            proj1_weight[(size_t)h * source_width + i] =
                    ((int)((h + 1) * (i + 3)) % 7 - 3) * 0.03125f;
        }
    }
    for (uint32_t j = 0; j < target_width; j++) {
        proj2_bias[j] = 0.02f - 0.004f * (float)j;
        residual_bias[j] = -0.01f + 0.002f * (float)j;
        post_ln_weight[j] = 1.0f + 0.01f * (float)j;
        post_ln_bias[j] = -0.02f + 0.003f * (float)j;
        for (uint32_t h = 0; h < hidden_width; h++) {
            proj2_weight[(size_t)j * hidden_width + h] =
                    ((int)((j + 5) * (h + 1)) % 11 - 5) * 0.015625f;
        }
        for (uint32_t i = 0; i < source_width; i++) {
            residual_weight[(size_t)j * source_width + i] =
                    ((int)((j + 2) * (i + 1)) % 5 - 2) * 0.0625f;
        }
    }

    axiom_latent_link_weights_f32 weights{};
    weights.abi_version = AXIOM_ABI_VERSION;
    weights.dtype = AXIOM_LATENT_F32;
    weights.pre_ln_weight = pre_ln_weight.data();
    weights.pre_ln_bias = pre_ln_bias.data();
    weights.proj1_weight = proj1_weight.data();
    weights.proj1_bias = proj1_bias.data();
    weights.proj2_weight = proj2_weight.data();
    weights.proj2_bias = proj2_bias.data();
    weights.residual_weight = residual_weight.data();
    weights.residual_bias = residual_bias.data();
    weights.post_ln_weight = post_ln_weight.data();
    weights.post_ln_bias = post_ln_bias.data();

    rc = axiom_latent_link_load_f32(link, &weights);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "axiom-latent-smoke: link load failed: %s\n", axiom_status_string(rc));
        axiom_latent_link_destroy(link);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    std::vector<float> host_source((size_t)rows * source_stride, 0.0f);
    std::vector<float> host_target((size_t)rows * target_stride, -99.0f);
    std::vector<float> host_ref((size_t)rows * target_stride, -99.0f);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < source_width; c++) {
            host_source[(size_t)r * source_stride + c] =
                    sinf((float)(r * source_width + c) * 0.17f) + 0.05f * (float)c;
        }
    }
    cpu_latent_link(host_source, host_ref,
                    rows, source_width, target_width, hidden_width,
                    source_stride, target_stride,
                    pre_ln_weight, pre_ln_bias,
                    proj1_weight, proj1_bias,
                    proj2_weight, proj2_bias,
                    residual_weight, residual_bias,
                    post_ln_weight, post_ln_bias,
                    eps);

    float *dev_source = nullptr;
    float *dev_target = nullptr;
    const size_t source_bytes = host_source.size() * sizeof(float);
    const size_t target_bytes = host_target.size() * sizeof(float);
    if (!cuda_ok(cudaMalloc(&dev_source, source_bytes), "cudaMalloc source") ||
        !cuda_ok(cudaMalloc(&dev_target, target_bytes), "cudaMalloc target") ||
        !cuda_ok(cudaMemcpy(dev_source, host_source.data(), source_bytes, cudaMemcpyHostToDevice), "copy source") ||
        !cuda_ok(cudaMemcpy(dev_target, host_target.data(), target_bytes, cudaMemcpyHostToDevice), "copy target")) {
        cudaFree(dev_target);
        cudaFree(dev_source);
        axiom_latent_link_destroy(link);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_latent_frame source{};
    source.abi_version = AXIOM_ABI_VERSION;
    source.dtype = AXIOM_LATENT_F32;
    source.rows = rows;
    source.cols = source_width;
    source.stride = source_stride;
    source.device_ptr = dev_source;

    axiom_latent_frame target{};
    target.abi_version = AXIOM_ABI_VERSION;
    target.dtype = AXIOM_LATENT_F32;
    target.rows = rows;
    target.cols = target_width;
    target.stride = target_stride;
    target.device_ptr = dev_target;

    rc = axiom_latent_link_apply(link, &source, &target);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "axiom-latent-smoke: link apply failed: %s\n", axiom_status_string(rc));
        cudaFree(dev_target);
        cudaFree(dev_source);
        axiom_latent_link_destroy(link);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    if (!cuda_ok(cudaMemcpy(host_target.data(), dev_target, target_bytes, cudaMemcpyDeviceToHost), "read target")) {
        cudaFree(dev_target);
        cudaFree(dev_source);
        axiom_latent_link_destroy(link);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    float max_abs = 0.0f;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < target_width; c++) {
            const float got = host_target[(size_t)r * target_stride + c];
            const float expected = host_ref[(size_t)r * target_stride + c];
            max_abs = fmaxf(max_abs, fabsf(got - expected));
        }
    }

    cudaFree(dev_target);
    cudaFree(dev_source);
    axiom_latent_link_destroy(link);
    axiom_runtime_destroy(runtime);

    if (max_abs > 0.0002f) {
        std::fprintf(stderr, "axiom-latent-smoke: mismatch max_abs=%f\n", (double)max_abs);
        return 1;
    }

    std::printf("axiom-latent-smoke: OK max_abs=%g\n", (double)max_abs);
    return 0;
}
