/* Bounded correctness gate for the native Qwen3.8 MTP CUDA executor. */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_mtp.h"
#include "axiom/qwen38_mtp_compute.h"

namespace {

constexpr uint32_t kMaxContext = 32u;
constexpr uint32_t kMaxColumns = 3u;
constexpr uint32_t kCommitPrefix = 2u;

struct DeviceAllocation {
    void *pointer = nullptr;

    DeviceAllocation() = default;
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;

    ~DeviceAllocation() { reset(); }

    int allocate(uint64_t bytes) {
        if (pointer || bytes == 0u || bytes > std::numeric_limits<size_t>::max()) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        return cudaMalloc(&pointer, static_cast<size_t>(bytes)) == cudaSuccess
                ? AXIOM_OK : AXIOM_ERR_CUDA;
    }

    void reset() {
        if (pointer) (void)cudaFree(pointer);
        pointer = nullptr;
    }

    template <typename T>
    T *as() {
        return static_cast<T *>(pointer);
    }
};

struct SyntheticTarget {
    const float *embedding_template_device = nullptr;
    uint32_t embedding_columns = 0u;
    uint32_t embed_calls = 0u;
    uint32_t lm_head_calls = 0u;
    uint32_t embed_columns = 0u;
    uint32_t lm_head_columns = 0u;
};

bool parse_device(const char *text, int *out) {
    if (!text || !out || !text[0]) return false;
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

bool checked_bytes(uint64_t elements, uint64_t element_bytes, uint64_t *out) {
    if (!out || element_bytes == 0u ||
        elements > std::numeric_limits<uint64_t>::max() / element_bytes) {
        return false;
    }
    *out = elements * element_bytes;
    return true;
}

int synthetic_embed_f32_device(
        void *user_data,
        const uint32_t *token_ids_device,
        float *out_hidden_device,
        uint32_t columns,
        void *stream_handle) {
    auto *target = static_cast<SyntheticTarget *>(user_data);
    if (!target || !token_ids_device || !out_hidden_device || columns == 0u ||
        columns > target->embedding_columns || !target->embedding_template_device) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t bytes = 0u;
    if (!checked_bytes(
                static_cast<uint64_t>(AXIOM_QWEN38_MTP_HIDDEN) * columns,
                sizeof(float), &bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const cudaError_t status = cudaMemcpyAsync(
            out_hidden_device, target->embedding_template_device,
            static_cast<size_t>(bytes), cudaMemcpyDeviceToDevice,
            reinterpret_cast<cudaStream_t>(stream_handle));
    if (status != cudaSuccess) return AXIOM_ERR_CUDA;
    ++target->embed_calls;
    target->embed_columns += columns;
    return AXIOM_OK;
}

int synthetic_lm_head_f32_device(
        void *user_data,
        const float *hidden_device,
        float *out_logits_device,
        uint32_t columns,
        void *stream_handle) {
    auto *target = static_cast<SyntheticTarget *>(user_data);
    if (!target || !hidden_device || !out_logits_device || columns == 0u ||
        columns > kMaxColumns) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t logit_bytes = 0u;
    if (!checked_bytes(
                static_cast<uint64_t>(AXIOM_QWEN38_MTP_VOCAB) * columns,
                sizeof(float), &logit_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    if (cudaMemsetAsync(
                out_logits_device, 0, static_cast<size_t>(logit_bytes), stream) !=
        cudaSuccess) {
        return AXIOM_ERR_CUDA;
    }

    /* A deterministic synthetic projection: preserve the first hidden_size
     * logits of each temporal column and zero the rest. This exercises the
     * callback with the real MTP output without loading a duplicate target
     * LM head or synchronizing inside the callback. */
    const size_t hidden_bytes =
            static_cast<size_t>(AXIOM_QWEN38_MTP_HIDDEN) * sizeof(float);
    for (uint32_t column = 0u; column < columns; ++column) {
        const float *source = hidden_device +
                static_cast<uint64_t>(column) * AXIOM_QWEN38_MTP_HIDDEN;
        float *destination = out_logits_device +
                static_cast<uint64_t>(column) * AXIOM_QWEN38_MTP_VOCAB;
        if (cudaMemcpyAsync(
                    destination, source, hidden_bytes, cudaMemcpyDeviceToDevice,
                    stream) != cudaSuccess) {
            return AXIOM_ERR_CUDA;
        }
    }
    ++target->lm_head_calls;
    target->lm_head_columns += columns;
    return AXIOM_OK;
}

bool finite_nonzero(const std::vector<float> &values, uint32_t columns, uint32_t rows) {
    if (columns == 0u || rows == 0u ||
        values.size() != static_cast<size_t>(columns) * rows) {
        return false;
    }
    for (uint32_t column = 0u; column < columns; ++column) {
        bool nonzero = false;
        const size_t base = static_cast<size_t>(column) * rows;
        for (uint32_t row = 0u; row < rows; ++row) {
            const float value = values[base + row];
            if (!std::isfinite(value)) return false;
            nonzero = nonzero || value != 0.0f;
        }
        if (!nonzero) return false;
    }
    return true;
}

int copy_to_device(void *destination, const void *source, uint64_t bytes) {
    if (!destination || !source || bytes == 0u ||
        bytes > std::numeric_limits<size_t>::max()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return cudaMemcpy(
                   destination, source, static_cast<size_t>(bytes),
                   cudaMemcpyHostToDevice) == cudaSuccess
            ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int copy_from_device(
        void *destination,
        const void *source,
        uint64_t bytes,
        cudaStream_t stream) {
    if (!destination || !source || bytes == 0u ||
        bytes > std::numeric_limits<size_t>::max()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaMemcpyAsync(
                destination, source, static_cast<size_t>(bytes),
                cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        return AXIOM_ERR_CUDA;
    }
    return cudaStreamSynchronize(stream) == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int compute_info(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_info *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
    return axiom_qwen38_mtp_compute_info_get(compute, out);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::printf("{\"status\":\"fail\",\"stage\":\"arguments\","
                    "\"rc\":%d,\"error\":\"usage: MODEL_DIR [DEVICE]\"}\n",
                    AXIOM_ERR_INVALID_ARGUMENT);
        return 2;
    }

    int device = 0;
    if (argc == 3 && !parse_device(argv[2], &device)) {
        std::printf("{\"status\":\"fail\",\"stage\":\"device\","
                    "\"rc\":%d,\"error\":\"invalid device\"}\n",
                    AXIOM_ERR_INVALID_ARGUMENT);
        return 2;
    }

    axiom_runtime *runtime = nullptr;
    axiom_model *model = nullptr;
    axiom_qwen38_mtp *mtp = nullptr;
    axiom_qwen38_mtp_compute *compute = nullptr;
    cudaStream_t stream = nullptr;
    DeviceAllocation embedding_template;
    DeviceAllocation token_ids;
    DeviceAllocation target_hidden;
    DeviceAllocation output_hidden;
    DeviceAllocation output_logits;
    DeviceAllocation output_top1;
    SyntheticTarget synthetic_target{};

    const char *stage = "cuda_device";
    int rc = cudaSetDevice(device) == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
    bool transaction_open = false;
    bool create_info_ok = false;
    bool width1_ok = false;
    bool width3_ok = false;
    bool top1_ok = false;
    bool deterministic_replay = false;
    bool abort_restored = false;
    bool commit_advanced = false;
    bool discontinuity_rejected = false;
    uint32_t position_after_abort = std::numeric_limits<uint32_t>::max();
    uint32_t position_after_commit = std::numeric_limits<uint32_t>::max();

    if (rc == AXIOM_OK) {
        stage = "stream_create";
        rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess
                ? AXIOM_OK : AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) {
        axiom_config config{};
        config.abi_version = AXIOM_ABI_VERSION;
        config.backend = AXIOM_BACKEND_CUDA;
        config.device = static_cast<uint32_t>(device);
        stage = "runtime_create";
        rc = axiom_runtime_create(&runtime, &config);
    }
    if (rc == AXIOM_OK) {
        axiom_model_config config{};
        config.abi_version = AXIOM_ABI_VERSION;
        config.path = argv[1];
        config.name = "qwen38-mtp-compute-gate";
        config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
        config.placement.abi_version = AXIOM_ABI_VERSION;
        config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
        config.placement.device_id = static_cast<uint32_t>(device);
        stage = "model_open";
        rc = axiom_model_open(runtime, &model, &config);
    }
    if (rc == AXIOM_OK) {
        stage = "mtp_load";
        rc = axiom_qwen38_mtp_load(model, runtime, device, &mtp);
    }

    const uint64_t hidden_elements =
            static_cast<uint64_t>(AXIOM_QWEN38_MTP_HIDDEN) * kMaxColumns;
    const uint64_t logit_elements =
            static_cast<uint64_t>(AXIOM_QWEN38_MTP_VOCAB) * kMaxColumns;
    uint64_t hidden_bytes = 0u;
    uint64_t logit_bytes = 0u;
    if (rc == AXIOM_OK &&
        (!checked_bytes(hidden_elements, sizeof(float), &hidden_bytes) ||
         !checked_bytes(logit_elements, sizeof(float), &logit_bytes))) {
        stage = "buffer_sizes";
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (rc == AXIOM_OK) {
        stage = "device_buffers";
        rc = embedding_template.allocate(hidden_bytes);
        if (rc == AXIOM_OK) rc = token_ids.allocate(kMaxColumns * sizeof(uint32_t));
        if (rc == AXIOM_OK) rc = target_hidden.allocate(hidden_bytes);
        if (rc == AXIOM_OK) rc = output_hidden.allocate(hidden_bytes);
        if (rc == AXIOM_OK) rc = output_logits.allocate(logit_bytes);
        if (rc == AXIOM_OK) rc = output_top1.allocate(kMaxColumns * sizeof(uint32_t));
    }

    std::vector<float> host_embedding(static_cast<size_t>(hidden_elements));
    std::vector<float> host_target_hidden(static_cast<size_t>(hidden_elements));
    for (uint32_t column = 0u; column < kMaxColumns; ++column) {
        for (uint32_t row = 0u; row < AXIOM_QWEN38_MTP_HIDDEN; ++row) {
            const size_t index = static_cast<size_t>(column) * AXIOM_QWEN38_MTP_HIDDEN + row;
            const int32_t embed_code = static_cast<int32_t>((row * 17u + column * 29u) % 257u) - 128;
            const int32_t hidden_code = static_cast<int32_t>((row * 31u + column * 11u) % 263u) - 131;
            host_embedding[index] = static_cast<float>(embed_code) / 2048.0f;
            host_target_hidden[index] = static_cast<float>(hidden_code) / 1024.0f;
        }
    }
    constexpr uint32_t host_token_ids[kMaxColumns] = {17u, 23u, 31u};
    if (rc == AXIOM_OK) {
        stage = "input_upload";
        rc = copy_to_device(embedding_template.pointer, host_embedding.data(), hidden_bytes);
        if (rc == AXIOM_OK) {
            rc = copy_to_device(
                    target_hidden.pointer, host_target_hidden.data(), hidden_bytes);
        }
        if (rc == AXIOM_OK) {
            rc = copy_to_device(
                    token_ids.pointer, host_token_ids,
                    static_cast<uint64_t>(sizeof(host_token_ids)));
        }
    }
    synthetic_target.embedding_template_device = embedding_template.as<float>();
    synthetic_target.embedding_columns = kMaxColumns;

    if (rc == AXIOM_OK) {
        axiom_qwen38_mtp_target_binding binding{};
        binding.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
        binding.user_data = &synthetic_target;
        binding.embed_f32_device = synthetic_embed_f32_device;
        binding.lm_head_f32_device = synthetic_lm_head_f32_device;

        axiom_qwen38_mtp_compute_config config{};
        config.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
        config.max_context = kMaxContext;
        config.cache_dtype = AXIOM_TENSOR_DTYPE_BF16;
        stage = "compute_create";
        rc = axiom_qwen38_mtp_compute_create_v1(
                mtp, runtime, device, &config, sizeof(config), &binding, &compute);
    }

    axiom_qwen38_mtp_compute_info info{};
    if (rc == AXIOM_OK) {
        stage = "compute_info";
        rc = compute_info(compute, &info);
        create_info_ok = rc == AXIOM_OK && info.device == static_cast<uint32_t>(device) &&
                info.max_context == kMaxContext &&
                info.hidden_size == AXIOM_QWEN38_MTP_HIDDEN &&
                info.max_columns == AXIOM_QWEN38_MTP_COMPUTE_MAX_COLUMNS &&
                info.cache_dtype == AXIOM_TENSOR_DTYPE_BF16 &&
                info.committed_position == 0u &&
                info.transaction_state == AXIOM_QWEN38_MTP_TRANSACTION_IDLE &&
                info.checkpoint_device_bytes == axiom_qwen38_mtp_device_bytes(mtp) &&
                info.kv_cache_device_bytes != 0u && info.device_bytes != 0u;
        if (rc == AXIOM_OK && !create_info_ok) rc = AXIOM_ERR_RUNTIME;
    }

    auto forward = [&](uint32_t first_position, uint32_t columns) {
        axiom_qwen38_mtp_compute_forward_request request{};
        request.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
        request.token_ids_device = token_ids.as<uint32_t>();
        request.prior_hidden_device = target_hidden.as<float>();
        request.out_hidden_device = output_hidden.as<float>();
        request.out_logits_device = output_logits.as<float>();
        request.out_top1_device = output_top1.as<uint32_t>();
        request.first_position = first_position;
        request.columns = columns;
        request.stream = reinterpret_cast<void *>(stream);
        return axiom_qwen38_mtp_compute_forward(compute, &request);
    };

    std::vector<float> first_hidden(AXIOM_QWEN38_MTP_HIDDEN);
    std::vector<float> replay_hidden(AXIOM_QWEN38_MTP_HIDDEN);
    std::vector<float> width3_hidden(static_cast<size_t>(hidden_elements));
    std::vector<float> width3_logits(static_cast<size_t>(logit_elements));
    std::vector<uint32_t> width3_top1(kMaxColumns);

    if (rc == AXIOM_OK) {
        stage = "width1_begin";
        rc = axiom_qwen38_mtp_compute_transaction_begin(compute, 0u, stream);
        transaction_open = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK) {
        stage = "width1_forward";
        rc = forward(0u, 1u);
    }
    if (rc == AXIOM_OK) {
        stage = "width1_commit";
        rc = axiom_qwen38_mtp_compute_transaction_commit_prefix(compute, 1u, stream);
        if (rc == AXIOM_OK) transaction_open = false;
    }
    if (rc == AXIOM_OK) {
        stage = "width1_download";
        rc = copy_from_device(
                first_hidden.data(), output_hidden.pointer,
                static_cast<uint64_t>(first_hidden.size()) * sizeof(float), stream);
        width1_ok = rc == AXIOM_OK && finite_nonzero(
                first_hidden, 1u, AXIOM_QWEN38_MTP_HIDDEN);
        if (rc == AXIOM_OK && !width1_ok) rc = AXIOM_ERR_CUDA;
    }

    if (rc == AXIOM_OK) {
        stage = "replay_reset";
        rc = axiom_qwen38_mtp_compute_reset(compute);
    }
    if (rc == AXIOM_OK) {
        stage = "replay_begin";
        rc = axiom_qwen38_mtp_compute_transaction_begin(compute, 0u, stream);
        transaction_open = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK) {
        stage = "replay_forward";
        rc = forward(0u, 1u);
    }
    if (rc == AXIOM_OK) {
        stage = "replay_commit";
        rc = axiom_qwen38_mtp_compute_transaction_commit_prefix(compute, 1u, stream);
        if (rc == AXIOM_OK) transaction_open = false;
    }
    if (rc == AXIOM_OK) {
        stage = "replay_download";
        rc = copy_from_device(
                replay_hidden.data(), output_hidden.pointer,
                static_cast<uint64_t>(replay_hidden.size()) * sizeof(float), stream);
        deterministic_replay = rc == AXIOM_OK &&
                std::memcmp(
                        first_hidden.data(), replay_hidden.data(),
                        first_hidden.size() * sizeof(float)) == 0;
        if (rc == AXIOM_OK && !deterministic_replay) rc = AXIOM_ERR_CUDA;
    }

    if (rc == AXIOM_OK) {
        stage = "abort_reset";
        rc = axiom_qwen38_mtp_compute_reset(compute);
    }
    if (rc == AXIOM_OK) {
        stage = "width3_begin";
        rc = axiom_qwen38_mtp_compute_transaction_begin(compute, 0u, stream);
        transaction_open = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK) {
        stage = "width3_forward";
        rc = forward(0u, 3u);
    }
    if (rc == AXIOM_OK) {
        stage = "width3_download";
        rc = copy_from_device(
                width3_hidden.data(), output_hidden.pointer, hidden_bytes, stream);
        if (rc == AXIOM_OK) {
            rc = copy_from_device(
                    width3_logits.data(), output_logits.pointer, logit_bytes, stream);
        }
        if (rc == AXIOM_OK) {
            rc = copy_from_device(
                    width3_top1.data(), output_top1.pointer,
                    static_cast<uint64_t>(width3_top1.size()) * sizeof(uint32_t), stream);
        }
        top1_ok = rc == AXIOM_OK && std::all_of(
                width3_top1.begin(), width3_top1.end(),
                [](uint32_t token) { return token < AXIOM_QWEN38_MTP_VOCAB; });
        width3_ok = rc == AXIOM_OK &&
                finite_nonzero(width3_hidden, 3u, AXIOM_QWEN38_MTP_HIDDEN) &&
                finite_nonzero(width3_logits, 3u, AXIOM_QWEN38_MTP_VOCAB) && top1_ok;
        if (rc == AXIOM_OK && !width3_ok) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) {
        stage = "transaction_abort";
        rc = axiom_qwen38_mtp_compute_transaction_abort(compute, stream);
        if (rc == AXIOM_OK) transaction_open = false;
    }
    if (rc == AXIOM_OK) {
        stage = "abort_info";
        rc = compute_info(compute, &info);
        position_after_abort = info.committed_position;
        abort_restored = rc == AXIOM_OK &&
                info.transaction_state == AXIOM_QWEN38_MTP_TRANSACTION_IDLE &&
                position_after_abort == 0u;
        if (rc == AXIOM_OK && !abort_restored) rc = AXIOM_ERR_RUNTIME;
    }

    if (rc == AXIOM_OK) {
        stage = "commit_begin";
        rc = axiom_qwen38_mtp_compute_transaction_begin(compute, 0u, stream);
        transaction_open = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK) {
        stage = "commit_forward";
        rc = forward(0u, 3u);
    }
    if (rc == AXIOM_OK) {
        stage = "commit_prefix";
        rc = axiom_qwen38_mtp_compute_transaction_commit_prefix(
                compute, kCommitPrefix, stream);
        if (rc == AXIOM_OK) transaction_open = false;
    }
    if (rc == AXIOM_OK) {
        stage = "commit_info";
        rc = compute_info(compute, &info);
        position_after_commit = info.committed_position;
        commit_advanced = rc == AXIOM_OK &&
                info.transaction_state == AXIOM_QWEN38_MTP_TRANSACTION_IDLE &&
                position_after_commit == kCommitPrefix;
        if (rc == AXIOM_OK && !commit_advanced) rc = AXIOM_ERR_RUNTIME;
    }

    if (rc == AXIOM_OK) {
        stage = "discontinuity_begin";
        rc = axiom_qwen38_mtp_compute_transaction_begin(
                compute, position_after_commit, stream);
        transaction_open = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK) {
        stage = "discontinuity_reject";
        const int discontinuity_rc = forward(position_after_commit + 1u, 1u);
        discontinuity_rejected = discontinuity_rc != AXIOM_OK;
        if (!discontinuity_rejected) rc = AXIOM_ERR_RUNTIME;
    }
    if (transaction_open) {
        const int abort_rc = axiom_qwen38_mtp_compute_transaction_abort(compute, stream);
        transaction_open = false;
        if (rc == AXIOM_OK && abort_rc != AXIOM_OK) {
            stage = "discontinuity_abort";
            rc = abort_rc;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "discontinuity_info";
        rc = compute_info(compute, &info);
        if (rc == AXIOM_OK && info.committed_position != position_after_commit) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK &&
        (synthetic_target.embed_calls < 4u || synthetic_target.lm_head_calls < 4u ||
         synthetic_target.embed_columns != 8u || synthetic_target.lm_head_columns != 8u)) {
        stage = "callback_counts";
        rc = AXIOM_ERR_RUNTIME;
    }

    if (transaction_open && compute) {
        const int abort_rc = axiom_qwen38_mtp_compute_transaction_abort(compute, stream);
        transaction_open = false;
        if (rc == AXIOM_OK && abort_rc != AXIOM_OK) {
            stage = "cleanup_abort";
            rc = abort_rc;
        }
    }

    if (stream) {
        const cudaError_t sync_status = cudaStreamSynchronize(stream);
        if (rc == AXIOM_OK && sync_status != cudaSuccess) {
            stage = "stream_sync";
            rc = AXIOM_ERR_CUDA;
        }
    }

    const bool passed = rc == AXIOM_OK && create_info_ok && width1_ok && width3_ok &&
            deterministic_replay && abort_restored && commit_advanced &&
            discontinuity_rejected && top1_ok;
    if (passed) stage = "complete";
    std::printf(
            "{\"status\":\"%s\",\"stage\":\"%s\",\"rc\":%d,"
            "\"max_context\":%u,\"cache_dtype\":\"BF16\","
            "\"create_info\":%u,\"forward_columns_1\":%u,"
            "\"forward_columns_3\":%u,\"finite_nonzero\":%u,"
            "\"top1_valid\":%u,"
            "\"deterministic_replay\":%u,\"abort_position\":%u,"
            "\"abort_restored\":%u,\"commit_prefix\":%u,"
            "\"commit_position\":%u,\"commit_advanced\":%u,"
            "\"discontinuity_rejected\":%u,\"embed_calls\":%u,"
            "\"lm_head_calls\":%u,\"embed_columns\":%u,"
            "\"lm_head_columns\":%u}\n",
            passed ? "pass" : "fail", stage, rc, kMaxContext,
            create_info_ok ? 1u : 0u, width1_ok ? 1u : 0u, width3_ok ? 1u : 0u,
            width1_ok && width3_ok ? 1u : 0u, top1_ok ? 1u : 0u,
            deterministic_replay ? 1u : 0u,
            position_after_abort, abort_restored ? 1u : 0u, kCommitPrefix,
            position_after_commit, commit_advanced ? 1u : 0u,
            discontinuity_rejected ? 1u : 0u, synthetic_target.embed_calls,
            synthetic_target.lm_head_calls, synthetic_target.embed_columns,
            synthetic_target.lm_head_columns);

    axiom_qwen38_mtp_compute_destroy(compute);
    compute = nullptr;
    output_top1.reset();
    output_logits.reset();
    output_hidden.reset();
    target_hidden.reset();
    token_ids.reset();
    embedding_template.reset();
    if (stream) (void)cudaStreamDestroy(stream);
    axiom_qwen38_mtp_destroy(mtp);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return passed ? 0 : 1;
}
