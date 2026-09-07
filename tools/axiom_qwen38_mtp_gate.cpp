/* Inventory and residency gate for the native Qwen3.8 BF16 MTP block. */

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "axiom/axiom.h"
#include "axiom/qwen38_mtp.h"

namespace {

constexpr uint64_t kExpectedCheckpointBytes = 849398784ull;

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

void print_failure(const char *stage, int rc) {
    std::printf("{\"status\":\"fail\",\"stage\":\"%s\",\"rc\":%d,"
                "\"error\":\"%s\"}\n",
                stage, rc, axiom_status_string(rc));
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [DEVICE]\n", argv[0]);
        return 2;
    }

    int device = 0;
    if (argc == 3 && !parse_device(argv[2], &device)) {
        std::fprintf(stderr, "axiom-qwen38-mtp-gate: invalid DEVICE\n");
        return 2;
    }

    axiom_config runtime_config{};
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = device;

    axiom_runtime *runtime = nullptr;
    axiom_model *model = nullptr;
    axiom_qwen38_mtp *mtp = nullptr;
    const char *stage = "runtime_create";
    int rc = axiom_runtime_create(&runtime, &runtime_config);

    if (rc == AXIOM_OK) {
        axiom_model_config model_config{};
        model_config.abi_version = AXIOM_ABI_VERSION;
        model_config.path = argv[1];
        model_config.name = "qwen38-mtp-gate";
        model_config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
        model_config.placement.abi_version = AXIOM_ABI_VERSION;
        model_config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
        model_config.placement.device_id = static_cast<uint32_t>(device);
        model_config.memory_budget_bytes = 0u;
        stage = "model_open";
        rc = axiom_model_open(runtime, &model, &model_config);
    }
    if (rc == AXIOM_OK) {
        stage = "mtp_load";
        rc = axiom_qwen38_mtp_load(model, runtime, device, &mtp);
    }

    axiom_qwen38_mtp_info mtp_info{};
    mtp_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) {
        stage = "mtp_info";
        rc = axiom_qwen38_mtp_info_get(mtp, &mtp_info);
    }

    axiom_qwen38_mtp_forward_contract contract{};
    contract.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) {
        stage = "forward_contract";
        rc = axiom_qwen38_mtp_forward_contract_get(mtp, &contract);
    }

    uint64_t enumerated_bytes = 0u;
    char source_file[sizeof(axiom_tensor_info{}.file)]{};
    uint32_t resident_buffers = 0u;
    for (uint32_t index = 0u;
         rc == AXIOM_OK && index < AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT;
         ++index) {
        axiom_qwen38_mtp_tensor_info tensor{};
        tensor.abi_version = AXIOM_ABI_VERSION;
        stage = "tensor_info";
        rc = axiom_qwen38_mtp_tensor_info_get(
                mtp, static_cast<axiom_qwen38_mtp_tensor>(index), &tensor);
        if (rc != AXIOM_OK) break;

        axiom_tensor_info checkpoint_tensor{};
        checkpoint_tensor.abi_version = AXIOM_ABI_VERSION;
        stage = "checkpoint_tensor_info";
        rc = axiom_model_tensor_info_get(model, tensor.name, &checkpoint_tensor);
        if (rc != AXIOM_OK) break;
        if (checkpoint_tensor.dtype != tensor.dtype || checkpoint_tensor.rank != tensor.rank ||
            checkpoint_tensor.byte_count != tensor.byte_count) {
            stage = "tensor_identity";
            rc = AXIOM_ERR_INVALID_ARGUMENT;
            break;
        }
        for (uint32_t dim = 0u; dim < tensor.rank; ++dim) {
            if (checkpoint_tensor.shape[dim] != tensor.shape[dim]) {
                stage = "tensor_shape";
                rc = AXIOM_ERR_INVALID_ARGUMENT;
                break;
            }
        }
        if (rc != AXIOM_OK) break;
        if (index == 0u) {
            const size_t length = std::strlen(checkpoint_tensor.file);
            if (length == 0u || length >= sizeof(source_file)) {
                stage = "source_file";
                rc = AXIOM_ERR_INVALID_ARGUMENT;
                break;
            }
            std::memcpy(source_file, checkpoint_tensor.file, length + 1u);
        } else if (std::strcmp(source_file, checkpoint_tensor.file) != 0) {
            stage = "source_file_consistency";
            rc = AXIOM_ERR_INVALID_ARGUMENT;
            break;
        }

        const axiom_device_buffer *buffer = nullptr;
        stage = "resident_buffer";
        rc = axiom_qwen38_mtp_tensor_buffer_get(
                mtp, static_cast<axiom_qwen38_mtp_tensor>(index), &buffer);
        if (rc != AXIOM_OK || !buffer) {
            if (rc == AXIOM_OK) rc = AXIOM_ERR_RUNTIME;
            break;
        }
        ++resident_buffers;
        enumerated_bytes += tensor.byte_count;
    }

    float dummy = 0.0f;
    axiom_qwen38_mtp_forward_request request{};
    request.abi_version = AXIOM_ABI_VERSION;
    request.target_hidden = &dummy;
    request.proposed_token_embedding = &dummy;
    request.draft_hidden = &dummy;
    request.batch = AXIOM_QWEN38_MTP_BATCH;
    if (rc == AXIOM_OK) {
        stage = "forward_request";
        rc = axiom_qwen38_mtp_forward_request_validate(mtp, &request);
    }

    if (rc == AXIOM_OK &&
        (mtp_info.tensor_count != AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT ||
         mtp_info.resident_dtype != AXIOM_TENSOR_DTYPE_BF16 ||
         mtp_info.checkpoint_bytes != kExpectedCheckpointBytes ||
         mtp_info.device_bytes != kExpectedCheckpointBytes ||
         enumerated_bytes != kExpectedCheckpointBytes ||
         resident_buffers != AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT ||
         contract.uses_base_embedding != 1u || contract.uses_base_lm_head != 1u ||
         contract.requires_mtp_kv_transaction != 1u ||
         contract.requires_target_cache_transaction != 1u ||
         contract.zero_centered_rmsnorm != 1u || contract.attention_output_gate != 1u)) {
        stage = "contract_gate";
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }

    if (rc == AXIOM_OK) {
        std::printf("{\"status\":\"pass\",\"source_file\":\"%s\","
                    "\"tensor_count\":%u,\"resident_buffers\":%u,"
                    "\"checkpoint_bytes\":%llu,\"device_bytes\":%llu,"
                    "\"dtype\":\"BF16\",\"executor\":false}\n",
                    source_file, mtp_info.tensor_count, resident_buffers,
                    static_cast<unsigned long long>(mtp_info.checkpoint_bytes),
                    static_cast<unsigned long long>(mtp_info.device_bytes));
    } else {
        print_failure(stage, rc);
    }

    axiom_qwen38_mtp_destroy(mtp);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return rc == AXIOM_OK ? 0 : 1;
}
