/* Real-device exact-parity and timing gate for grouped Qwen3.8 FP8 inputs.
 * This tool is intentionally not part of a production service target. */
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_fp8.h"

namespace {

constexpr uint32_t kHidden = 5120u;
constexpr uint32_t kThreads = 256u;

__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void pack_separate_bf16_kernel(
        const float *projection0,
        const float *projection1,
        const float *projection2,
        uint32_t rows0,
        uint32_t rows1,
        uint32_t rows2,
        float *out) {
    const uint32_t total_rows = rows0 + rows1 + rows2;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(total_rows) * AXIOM_QWEN38_FP8_BATCH;
    if (index >= count) return;
    const uint32_t column = static_cast<uint32_t>(index / total_rows);
    uint32_t row = static_cast<uint32_t>(index % total_rows);
    const float *projection = projection0;
    uint32_t projection_rows = rows0;
    if (row >= rows0) {
        row -= rows0;
        projection = projection1;
        projection_rows = rows1;
        if (row >= rows1) {
            row -= rows1;
            projection = projection2;
            projection_rows = rows2;
        }
    }
    out[index] = round_bf16(
            projection[static_cast<uint64_t>(column) * projection_rows + row]);
}

int fail(const char *what, int rc = AXIOM_ERR_RUNTIME) {
    std::fprintf(stderr, "axiom-qwen38-fp8-projection-group-probe: %s (status=%d)\n",
                 what, rc);
    return 1;
}

uint32_t runs_from_env() {
    const char *text = std::getenv("AXIOM_QWEN38_FP8_GROUP_RUNS");
    if (!text || !text[0]) return 100u;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return end != text && *end == '\0' && value > 0u && value <= 10000u
            ? static_cast<uint32_t>(value) : 0u;
}

bool parse_layer(const char *text, uint32_t *out) {
    if (!text || !text[0] || !out) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value >= 64u) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

void fill_input(std::vector<float> *input) {
    for (uint32_t column = 0u; column < AXIOM_QWEN38_FP8_BATCH; ++column) {
        const float phase = static_cast<float>(column) * 0.037f;
        for (uint32_t feature = 0u; feature < kHidden; ++feature) {
            (*input)[static_cast<size_t>(column) * kHidden + feature] =
                    0.63f * std::sin(static_cast<float>(feature) * 0.013f + phase) +
                    0.31f * std::cos(static_cast<float>(feature) * 0.0047f - phase) +
                    static_cast<float>(static_cast<int>(feature % 13u) - 6) * 0.009f;
        }
    }
}

template <typename Enqueue>
int time_calls(uint32_t runs, Enqueue enqueue, float *out_ms) {
    if (!out_ms || runs == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    for (uint32_t warmup = 0u; warmup < 5u; ++warmup) {
        const int rc = enqueue();
        if (rc != AXIOM_OK) return rc;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t status = cudaEventCreate(&start);
    if (status == cudaSuccess) status = cudaEventCreate(&stop);
    if (status == cudaSuccess) status = cudaEventRecord(start, nullptr);
    int rc = AXIOM_OK;
    for (uint32_t run = 0u; run < runs && rc == AXIOM_OK; ++run) rc = enqueue();
    if (status == cudaSuccess && rc == AXIOM_OK) status = cudaEventRecord(stop, nullptr);
    if (status == cudaSuccess && rc == AXIOM_OK) status = cudaEventSynchronize(stop);
    float elapsed = 0.0f;
    if (status == cudaSuccess && rc == AXIOM_OK) {
        status = cudaEventElapsedTime(&elapsed, start, stop);
    }
    if (stop) (void)cudaEventDestroy(stop);
    if (start) (void)cudaEventDestroy(start);
    if (rc != AXIOM_OK) return rc;
    if (status != cudaSuccess) return AXIOM_ERR_CUDA;
    *out_ms = elapsed / static_cast<float>(runs);
    return AXIOM_OK;
}

struct Resources {
    axiom_runtime *runtime = nullptr;
    axiom_model *model = nullptr;
    axiom_qwen38_fp8_projection_group *group = nullptr;
    axiom_qwen38_fp8_linear *linears[3]{};
    float *input = nullptr;
    float *group_out = nullptr;
    float *reference_out = nullptr;
    float *separate_out = nullptr;
    float *projection_out[3]{};

    ~Resources() {
        for (float *value : projection_out) {
            if (value) (void)cudaFree(value);
        }
        if (separate_out) (void)cudaFree(separate_out);
        if (reference_out) (void)cudaFree(reference_out);
        if (group_out) (void)cudaFree(group_out);
        if (input) (void)cudaFree(input);
        for (axiom_qwen38_fp8_linear *linear : linears) {
            axiom_qwen38_fp8_linear_destroy(linear);
        }
        axiom_qwen38_fp8_projection_group_destroy(group);
        if (model) axiom_model_close(model);
        if (runtime) axiom_runtime_destroy(runtime);
    }
};

struct Bf16ParityStats {
    uint64_t bit_mismatches = 0u;
    uint64_t outside_one_ulp = 0u;
    uint32_t max_ulp = 0u;
    float max_abs = 0.0f;
};

uint32_t ordered_bf16(float value) {
    uint32_t word = 0u;
    std::memcpy(&word, &value, sizeof(word));
    const uint16_t bits = static_cast<uint16_t>(word >> 16u);
    return (bits & 0x8000u) != 0u
            ? static_cast<uint32_t>(static_cast<uint16_t>(~bits))
            : static_cast<uint32_t>(bits) | 0x8000u;
}

Bf16ParityStats compare_bf16(
        const std::vector<float> &actual,
        const std::vector<float> &reference) {
    Bf16ParityStats stats{};
    for (size_t index = 0u; index < actual.size(); ++index) {
        uint32_t actual_bits = 0u;
        uint32_t reference_bits = 0u;
        std::memcpy(&actual_bits, &actual[index], sizeof(actual_bits));
        std::memcpy(&reference_bits, &reference[index], sizeof(reference_bits));
        if (actual_bits != reference_bits) ++stats.bit_mismatches;
        const float absolute = std::fabs(actual[index] - reference[index]);
        stats.max_abs = std::max(stats.max_abs, absolute);
        if (!std::isfinite(actual[index]) || !std::isfinite(reference[index])) {
            ++stats.outside_one_ulp;
            stats.max_ulp = std::numeric_limits<uint32_t>::max();
            continue;
        }
        const uint32_t actual_ordered = ordered_bf16(actual[index]);
        const uint32_t reference_ordered = ordered_bf16(reference[index]);
        const uint32_t ulp = actual_ordered >= reference_ordered
                ? actual_ordered - reference_ordered
                : reference_ordered - actual_ordered;
        stats.max_ulp = std::max(stats.max_ulp, ulp);
        if (ulp > 1u) ++stats.outside_one_ulp;
    }
    return stats;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s MODEL_DIR gdn|attention LAYER\n", argv[0]);
        return 2;
    }
    const bool gdn = std::strcmp(argv[2], "gdn") == 0;
    const bool attention = std::strcmp(argv[2], "attention") == 0;
    uint32_t layer = 0u;
    if ((!gdn && !attention) || !parse_layer(argv[3], &layer) ||
        (gdn && (layer % 4u) == 3u) || (attention && (layer % 4u) != 3u)) {
        return fail("mode/layer mismatch", AXIOM_ERR_INVALID_ARGUMENT);
    }
    const uint32_t runs = runs_from_env();
    if (runs == 0u) return fail("invalid AXIOM_QWEN38_FP8_GROUP_RUNS");

    const uint32_t projection_count = gdn ? 2u : 3u;
    const uint32_t expected_rows[3] = {
        gdn ? 10240u : 12288u,
        gdn ? 6144u : 1024u,
        gdn ? 0u : 1024u,
    };
    const char *suffixes_gdn[2] = {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z"};
    const char *suffixes_attention[3] = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"};
    std::string bases_storage[3];
    const char *bases[3]{};
    for (uint32_t projection = 0u; projection < projection_count; ++projection) {
        char value[192]{};
        const char *suffix = gdn ? suffixes_gdn[projection] : suffixes_attention[projection];
        const int written = std::snprintf(
                value, sizeof(value), "model.language_model.layers.%u.%s", layer, suffix);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(value)) {
            return fail("tensor base overflow", AXIOM_ERR_INVALID_ARGUMENT);
        }
        bases_storage[projection] = value;
        bases[projection] = bases_storage[projection].c_str();
    }

    Resources resources;
    axiom_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.backend = AXIOM_BACKEND_CUDA;
    config.device = 0u;
    int rc = axiom_runtime_create(&resources.runtime, &config);
    if (rc != AXIOM_OK) return fail("runtime create", rc);
    axiom_model_config model_config{};
    model_config.abi_version = AXIOM_ABI_VERSION;
    model_config.path = argv[1];
    model_config.name = "qwen3.8-27b-fp8-projection-group-probe";
    model_config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    model_config.placement.abi_version = AXIOM_ABI_VERSION;
    model_config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    rc = axiom_model_open(resources.runtime, &resources.model, &model_config);
    if (rc != AXIOM_OK) return fail("model open", rc);
    rc = axiom_qwen38_fp8_projection_group_load_auto(
            resources.model, 0, bases, projection_count, &resources.group);
    for (uint32_t projection = 0u;
         projection < projection_count && rc == AXIOM_OK; ++projection) {
        rc = axiom_qwen38_fp8_linear_load_auto(
                resources.model, 0, bases[projection], &resources.linears[projection]);
    }
    if (rc != AXIOM_OK) return fail("resident load", rc);

    axiom_qwen38_fp8_projection_group_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_qwen38_fp8_projection_group_info_get(resources.group, &info);
    uint32_t total_rows = 0u;
    for (uint32_t projection = 0u; projection < projection_count; ++projection) {
        total_rows += expected_rows[projection];
        if (axiom_qwen38_fp8_linear_rows(resources.linears[projection]) !=
                    expected_rows[projection] ||
            axiom_qwen38_fp8_linear_cols(resources.linears[projection]) != kHidden ||
            info.row_offsets[projection + 1u] != total_rows) {
            return fail("checkpoint/group geometry mismatch", AXIOM_ERR_INVALID_ARGUMENT);
        }
    }
    if (rc != AXIOM_OK || info.projection_count != projection_count ||
        info.input_features != kHidden || info.total_output_features != total_rows ||
        info.tensor_core_enabled != 1u || info.bf16_output_boundary != 1u) {
        return fail("group info/tensor-core gate", rc);
    }
    axiom_model_close(resources.model);
    resources.model = nullptr;
    axiom_runtime_destroy(resources.runtime);
    resources.runtime = nullptr;

    const size_t input_count = static_cast<size_t>(kHidden) * AXIOM_QWEN38_FP8_BATCH;
    const size_t output_count = static_cast<size_t>(total_rows) * AXIOM_QWEN38_FP8_BATCH;
    std::vector<float> host_input(input_count);
    std::vector<float> host_group(output_count);
    std::vector<float> host_reference(output_count);
    std::vector<float> host_separate(output_count);
    fill_input(&host_input);
    cudaError_t cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&resources.input), input_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&resources.group_out), output_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&resources.reference_out), output_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&resources.separate_out), output_count * sizeof(float));
    for (uint32_t projection = 0u;
         projection < projection_count && cuda_status == cudaSuccess; ++projection) {
        cuda_status = cudaMalloc(
                reinterpret_cast<void **>(&resources.projection_out[projection]),
                static_cast<size_t>(expected_rows[projection]) *
                        AXIOM_QWEN38_FP8_BATCH * sizeof(float));
    }
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(
            resources.input, host_input.data(), input_count * sizeof(float),
            cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) return fail("device allocation/upload", AXIOM_ERR_CUDA);

    const auto enqueue_separate = [&]() -> int {
        int local_rc = AXIOM_OK;
        for (uint32_t projection = 0u;
             projection < projection_count && local_rc == AXIOM_OK; ++projection) {
            local_rc = axiom_qwen38_fp8_linear_forward_f32_device(
                    resources.linears[projection], resources.input,
                    resources.projection_out[projection], nullptr);
        }
        if (local_rc != AXIOM_OK) return local_rc;
        const uint64_t count = static_cast<uint64_t>(total_rows) * AXIOM_QWEN38_FP8_BATCH;
        pack_separate_bf16_kernel<<<
                static_cast<uint32_t>((count + kThreads - 1u) / kThreads), kThreads>>>(
                resources.projection_out[0], resources.projection_out[1],
                resources.projection_out[2], expected_rows[0], expected_rows[1],
                expected_rows[2], resources.separate_out);
        return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
    };
    rc = axiom_qwen38_fp8_projection_group_forward_reference_f32_device(
            resources.group, resources.input, resources.reference_out, nullptr);
    if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_projection_group_forward_f32_device(
            resources.group, resources.input, resources.group_out, nullptr);
    if (rc == AXIOM_OK) rc = enqueue_separate();
    if (rc != AXIOM_OK || cudaDeviceSynchronize() != cudaSuccess) {
        return fail("parity enqueue", rc);
    }
    if (cudaMemcpy(host_group.data(), resources.group_out,
                   output_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_reference.data(), resources.reference_out,
                   output_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_separate.data(), resources.separate_out,
                   output_count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return fail("parity download", AXIOM_ERR_CUDA);
    }
    const Bf16ParityStats group_reference = compare_bf16(host_group, host_reference);
    const Bf16ParityStats separate_reference = compare_bf16(host_separate, host_reference);
    const Bf16ParityStats legacy_delta = compare_bf16(host_group, host_separate);
    float grouped_ms = 0.0f;
    float separate_ms = 0.0f;
    if (group_reference.outside_one_ulp == 0u &&
        separate_reference.outside_one_ulp == 0u) {
        rc = time_calls(runs, [&]() {
            return axiom_qwen38_fp8_projection_group_forward_f32_device(
                    resources.group, resources.input, resources.group_out, nullptr);
        }, &grouped_ms);
    }
    if (rc == AXIOM_OK && group_reference.outside_one_ulp == 0u &&
        separate_reference.outside_one_ulp == 0u) {
        rc = time_calls(runs, enqueue_separate, &separate_ms);
    }
    const bool pass = rc == AXIOM_OK && group_reference.outside_one_ulp == 0u &&
            separate_reference.outside_one_ulp == 0u;
    std::printf("{\"status\":\"%s\",\"mode\":\"%s\",\"layer\":%u,"
                "\"projections\":%u,\"rows\":%u,\"cols\":%u,\"batch\":8,"
                "\"schema\":%u,\"tensor_core\":true,\"bf16_boundary\":true,"
                "\"device_bytes\":%llu,\"runs\":%u,\"grouped_ms\":%.9g,"
                "\"separate_ms\":%.9g,\"speedup\":%.9g,"
                "\"group_vs_reference\":{\"bit_mismatches\":%llu,"
                "\"outside_one_bf16_ulp\":%llu,\"max_bf16_ulp\":%u,"
                "\"max_abs\":%.9g},"
                "\"separate_vs_reference\":{\"bit_mismatches\":%llu,"
                "\"outside_one_bf16_ulp\":%llu,\"max_bf16_ulp\":%u,"
                "\"max_abs\":%.9g},"
                "\"group_vs_separate_legacy_delta\":{\"bit_mismatches\":%llu,"
                "\"max_bf16_ulp\":%u,\"max_abs\":%.9g}}\n",
                pass ? "pass" : "fail", gdn ? "gdn-qkv-z" : "attention-q-k-v",
                layer, projection_count, total_rows, kHidden, info.scale_schema,
                static_cast<unsigned long long>(info.device_bytes), runs,
                static_cast<double>(grouped_ms), static_cast<double>(separate_ms),
                static_cast<double>(grouped_ms > 0.0f ? separate_ms / grouped_ms : 0.0f),
                static_cast<unsigned long long>(group_reference.bit_mismatches),
                static_cast<unsigned long long>(group_reference.outside_one_ulp),
                group_reference.max_ulp, static_cast<double>(group_reference.max_abs),
                static_cast<unsigned long long>(separate_reference.bit_mismatches),
                static_cast<unsigned long long>(separate_reference.outside_one_ulp),
                separate_reference.max_ulp, static_cast<double>(separate_reference.max_abs),
                static_cast<unsigned long long>(legacy_delta.bit_mismatches),
                legacy_delta.max_ulp, static_cast<double>(legacy_delta.max_abs));
    return pass ? 0 : 1;
}
