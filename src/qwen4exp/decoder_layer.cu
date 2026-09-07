#include "axiom/qwen4exp/decoder_layer.hpp"

#include "axiom/qwen4exp/gdn_adapter.hpp"
#include "axiom/qwen4exp/gdn_transaction.hpp"
#include "axiom/qwen4exp/mhc_adapter.hpp"
#include "axiom/qwen4exp/moe_adapter.hpp"
#include "axiom/qwen4exp/ple_adapter.hpp"
#include "axiom/qwen4exp/qsa_adapter.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace axiom::qwen4exp {
namespace {

constexpr std::size_t kDecoderTokenBatch = kDecoderMaxSpeculativeTokens;
constexpr std::size_t kF32Bytes = sizeof(float);
constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);

bool checked_mul(std::size_t left,
                 std::size_t right,
                 std::size_t *result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

void set_error(std::string *error, const std::string &message) noexcept {
    if (error == nullptr) return;
    try {
        *error = message;
    } catch (...) {
    }
}

decoder_layer_status fail(decoder_layer_status status,
                          std::string *error,
                          const std::string &message) noexcept {
    set_error(error, message);
    return status;
}

std::string layer_prefix(std::size_t layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".";
}

decoder_layer_status map_mhc(mhc_adapter_status status) noexcept {
    switch (status) {
        case mhc_adapter_status::ok:
            return decoder_layer_status::ok;
        case mhc_adapter_status::invalid_argument:
            return decoder_layer_status::invalid_argument;
        case mhc_adapter_status::unsupported_config:
        case mhc_adapter_status::tensor_contract_mismatch:
            return decoder_layer_status::unsupported_config;
        case mhc_adapter_status::allocation_failure:
            return decoder_layer_status::allocation_failure;
        case mhc_adapter_status::invalid_device_pointer:
            return decoder_layer_status::invalid_device_pointer;
        case mhc_adapter_status::cuda_error:
            return decoder_layer_status::cuda_error;
        case mhc_adapter_status::size_overflow:
        case mhc_adapter_status::linear_error:
            return decoder_layer_status::mhc_error;
    }
    return decoder_layer_status::mhc_error;
}

decoder_layer_status map_gdn(GdnAdapterStatus status) noexcept {
    switch (status) {
        case GdnAdapterStatus::kOk:
            return decoder_layer_status::ok;
        case GdnAdapterStatus::kInvalidArgument:
            return decoder_layer_status::invalid_argument;
        case GdnAdapterStatus::kUnsupportedLayer:
            return decoder_layer_status::unsupported_layer;
        case GdnAdapterStatus::kUnsupportedConfig:
            return decoder_layer_status::unsupported_config;
        case GdnAdapterStatus::kAllocationFailure:
            return decoder_layer_status::allocation_failure;
        case GdnAdapterStatus::kInvalidDevicePointer:
            return decoder_layer_status::invalid_device_pointer;
        case GdnAdapterStatus::kCudaError:
            return decoder_layer_status::cuda_error;
        case GdnAdapterStatus::kTensorNotFound:
        case GdnAdapterStatus::kDtypeMismatch:
        case GdnAdapterStatus::kShapeMismatch:
        case GdnAdapterStatus::kCheckpointIoError:
            return decoder_layer_status::invalid_checkpoint;
        case GdnAdapterStatus::kStateMismatch:
            return decoder_layer_status::invalid_state;
        case GdnAdapterStatus::kSizeOverflow:
        case GdnAdapterStatus::kLinearError:
        case GdnAdapterStatus::kGdnError:
        case GdnAdapterStatus::kUnsupportedDevice:
            return decoder_layer_status::gdn_error;
    }
    return decoder_layer_status::gdn_error;
}

decoder_layer_status map_qsa(QsaAdapterStatus status) noexcept {
    switch (status) {
        case QsaAdapterStatus::kOk:
            return decoder_layer_status::ok;
        case QsaAdapterStatus::kInvalidArgument:
            return decoder_layer_status::invalid_argument;
        case QsaAdapterStatus::kUnsupportedLayer:
            return decoder_layer_status::unsupported_layer;
        case QsaAdapterStatus::kUnsupportedConfig:
            return decoder_layer_status::unsupported_config;
        case QsaAdapterStatus::kAllocationFailure:
            return decoder_layer_status::allocation_failure;
        case QsaAdapterStatus::kInvalidDevicePointer:
            return decoder_layer_status::invalid_device_pointer;
        case QsaAdapterStatus::kCudaError:
            return decoder_layer_status::cuda_error;
        case QsaAdapterStatus::kTensorNotFound:
        case QsaAdapterStatus::kDtypeMismatch:
        case QsaAdapterStatus::kShapeMismatch:
        case QsaAdapterStatus::kCheckpointIoError:
            return decoder_layer_status::invalid_checkpoint;
        case QsaAdapterStatus::kStateMismatch:
            return decoder_layer_status::invalid_state;
        case QsaAdapterStatus::kSizeOverflow:
        case QsaAdapterStatus::kLinearError:
        case QsaAdapterStatus::kQsaError:
        case QsaAdapterStatus::kAttentionError:
        case QsaAdapterStatus::kDeviceRejected:
        case QsaAdapterStatus::kUnsupportedDevice:
            return decoder_layer_status::qsa_error;
    }
    return decoder_layer_status::qsa_error;
}

decoder_layer_status map_ple(ple_adapter_status status) noexcept {
    switch (status) {
        case ple_adapter_status::ok:
            return decoder_layer_status::ok;
        case ple_adapter_status::invalid_argument:
            return decoder_layer_status::invalid_argument;
        case ple_adapter_status::unsupported_layer:
            return decoder_layer_status::unsupported_layer;
        case ple_adapter_status::unsupported_config:
            return decoder_layer_status::unsupported_config;
        case ple_adapter_status::resource_exhausted:
            return decoder_layer_status::allocation_failure;
        case ple_adapter_status::invalid_device_pointer:
            return decoder_layer_status::invalid_device_pointer;
        case ple_adapter_status::cuda_error:
            return decoder_layer_status::cuda_error;
        case ple_adapter_status::invalid_checkpoint:
        case ple_adapter_status::tensor_not_found:
        case ple_adapter_status::dtype_mismatch:
        case ple_adapter_status::shape_mismatch:
        case ple_adapter_status::metadata_mismatch:
        case ple_adapter_status::checkpoint_io_error:
            return decoder_layer_status::invalid_checkpoint;
        case ple_adapter_status::invalid_state:
            return decoder_layer_status::invalid_state;
        case ple_adapter_status::size_overflow:
        case ple_adapter_status::unsupported_device:
        case ple_adapter_status::embedding_error:
        case ple_adapter_status::compute_error:
            return decoder_layer_status::ple_error;
    }
    return decoder_layer_status::ple_error;
}

decoder_layer_status map_moe(moe_adapter_status status) noexcept {
    switch (status) {
        case moe_adapter_status::ok:
            return decoder_layer_status::ok;
        case moe_adapter_status::invalid_argument:
            return decoder_layer_status::invalid_argument;
        case moe_adapter_status::unsupported_config:
            return decoder_layer_status::unsupported_config;
        case moe_adapter_status::invalid_device_pointer:
            return decoder_layer_status::invalid_device_pointer;
        case moe_adapter_status::allocation_failure:
            return decoder_layer_status::allocation_failure;
        case moe_adapter_status::checkpoint_error:
            return decoder_layer_status::invalid_checkpoint;
        case moe_adapter_status::busy:
        case moe_adapter_status::transaction_error:
            return decoder_layer_status::transaction_error;
        case moe_adapter_status::cuda_error:
            return decoder_layer_status::cuda_error;
        case moe_adapter_status::unsupported_device:
        case moe_adapter_status::router_error:
        case moe_adapter_status::pager_error:
        case moe_adapter_status::non_finite:
        case moe_adapter_status::kernel_error:
            return decoder_layer_status::moe_error;
    }
    return decoder_layer_status::moe_error;
}

decoder_layer_status map_gdn_transaction(GdnTransactionStatus status) noexcept {
    switch (status) {
        case GdnTransactionStatus::kOk:
            return decoder_layer_status::ok;
        case GdnTransactionStatus::kInvalidArgument:
            return decoder_layer_status::invalid_argument;
        case GdnTransactionStatus::kCudaError:
            return decoder_layer_status::cuda_error;
        case GdnTransactionStatus::kSizeOverflow:
        case GdnTransactionStatus::kSnapshotTooLarge:
            return decoder_layer_status::allocation_failure;
        case GdnTransactionStatus::kUnsupportedConfig:
        case GdnTransactionStatus::kDimensionMismatch:
        case GdnTransactionStatus::kUninitialized:
        case GdnTransactionStatus::kAlreadyInitialized:
        case GdnTransactionStatus::kStateMismatch:
        case GdnTransactionStatus::kUnsupportedDevice:
        case GdnTransactionStatus::kStreamMismatch:
        case GdnTransactionStatus::kTransactionOpen:
        case GdnTransactionStatus::kNoTransaction:
        case GdnTransactionStatus::kPartialAcceptUnsupported:
        case GdnTransactionStatus::kGdnError:
            return decoder_layer_status::transaction_error;
    }
    return decoder_layer_status::transaction_error;
}

bool current_sm120_device(int *device) noexcept {
    if (device == nullptr || cudaGetDevice(device) != cudaSuccess) return false;
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, *device) == cudaSuccess &&
           properties.major == 12 && properties.minor == 0;
}

struct device_range {
    const void *pointer = nullptr;
    std::size_t bytes = 0u;
};

bool valid_device_range(const device_range &range,
                        int expected_device) noexcept {
    if (range.pointer == nullptr || range.bytes == 0u) return false;
    cudaPointerAttributes attributes{};
    const cudaError_t runtime_status =
            cudaPointerGetAttributes(&attributes, range.pointer);
    if (runtime_status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice ||
        attributes.device != expected_device) {
        return false;
    }
    const std::uintptr_t address =
            reinterpret_cast<std::uintptr_t>(range.pointer);
    CUdeviceptr allocation_base = 0u;
    std::size_t allocation_bytes = 0u;
    if (cuMemGetAddressRange(&allocation_base, &allocation_bytes,
                             static_cast<CUdeviceptr>(address)) != CUDA_SUCCESS) {
        return false;
    }
    const std::uintptr_t base =
            static_cast<std::uintptr_t>(allocation_base);
    if (address < base || range.bytes > allocation_bytes) return false;
    const std::size_t offset = static_cast<std::size_t>(address - base);
    return offset <= allocation_bytes - range.bytes;
}

bool ranges_overlap(const device_range &left,
                    const device_range &right) noexcept {
    const std::uintptr_t left_begin =
            reinterpret_cast<std::uintptr_t>(left.pointer);
    const std::uintptr_t right_begin =
            reinterpret_cast<std::uintptr_t>(right.pointer);
    if (left.bytes > std::numeric_limits<std::uintptr_t>::max() - left_begin ||
        right.bytes >
                std::numeric_limits<std::uintptr_t>::max() - right_begin) {
        return true;
    }
    const std::uintptr_t left_end = left_begin + left.bytes;
    const std::uintptr_t right_end = right_begin + right.bytes;
    return left_begin < right_end && right_begin < left_end;
}

}  // namespace

struct decoder_layer::impl {
    decoder_layer_config config{};
    decoder_layer_kind kind = decoder_layer_kind::gated_deltanet;
    int device = -1;
    std::unique_ptr<resident_mhc_adapter> attention_mhc;
    std::unique_ptr<resident_mhc_adapter> mlp_mhc;
    std::unique_ptr<GdnLayerAdapter> gdn;
    std::unique_ptr<QsaLayerAdapter> qsa;
    std::unique_ptr<routed_moe_layer_adapter> moe;
    std::unique_ptr<ple_layer_adapter> ple;
    bool ready = false;
};

struct decoder_layer_session::impl {
    decoder_layer::impl *owner = nullptr;
    int device = -1;
    cudaStream_t bound_stream = nullptr;
    std::size_t context_capacity = 0u;
    std::uint64_t committed_tokens = 0u;
    std::size_t staged_tokens = 0u;
    std::size_t eagerly_validated_moe_rows = 0u;
    bool transaction_open = false;
    bool prepared_to_commit = false;
    bool poisoned = false;
    bool ready = false;

    std::vector<void *> allocations;
    mhc_adapter_scratch mhc_scratch{};
    float *ple_hidden = nullptr;
    float *attention_mixed = nullptr;
    float *attention_injection = nullptr;
    float *attention_block = nullptr;
    float *after_attention = nullptr;
    float *mlp_mixed = nullptr;
    float *mlp_injection = nullptr;
    float *mlp_block = nullptr;

    GdnDeviceState gdn_state{};
    GdnTransactionState gdn_transaction{};
    GdnAdapterScratch gdn_scratch{};

    QsaLayerCache qsa_cache{};
    QsaAdapterScratch qsa_scratch{};

    std::unique_ptr<ple_layer_session> ple_session;
    bool early_ple_pending = false;
    std::int64_t early_ple_token = 0;
    routed_moe_layer_adapter::prepared_generation moe_generation{};

    bool allocate_bytes(void **output,
                        std::size_t bytes,
                        std::string *error,
                        const char *label) noexcept {
        if (output == nullptr || bytes == 0u) {
            set_error(error, std::string("invalid allocation request: ") + label);
            return false;
        }
        *output = nullptr;
        void *pointer = nullptr;
        const cudaError_t status = cudaMalloc(&pointer, bytes);
        if (status != cudaSuccess) {
            set_error(error, std::string("cudaMalloc(") + label + "): " +
                                     cudaGetErrorString(status));
            return false;
        }
        try {
            allocations.push_back(pointer);
        } catch (...) {
            (void)cudaFree(pointer);
            set_error(error, std::string("allocation registry failed: ") + label);
            return false;
        }
        *output = pointer;
        return true;
    }

    template <typename T>
    bool allocate_typed(T **output,
                        std::size_t bytes,
                        std::string *error,
                        const char *label) noexcept {
        return allocate_bytes(reinterpret_cast<void **>(output), bytes, error, label);
    }

    bool rollback_components(cudaStream_t stream,
                             bool synchronize,
                             std::string *error) noexcept {
        bool success = true;
        std::string detail;
        if (moe_generation.valid() && !moe_generation.rollback(&detail)) {
            success = false;
        }
        moe_generation = routed_moe_layer_adapter::prepared_generation{};

        if (owner != nullptr && owner->qsa != nullptr &&
            qsa_cache.initialized && qsa_cache.transaction_open) {
            const QsaAdapterStatus status = owner->qsa->rollback(&qsa_cache);
            if (status != QsaAdapterStatus::kOk) {
                success = false;
                if (detail.empty()) detail = qsa_adapter_status_string(status);
            }
        }
        if (gdn_transaction.initialized && gdn_transaction.transaction_open) {
            const GdnTransactionStatus status =
                    gdn_transaction_rollback(&gdn_transaction, stream);
            if (status != GdnTransactionStatus::kOk) {
                success = false;
                if (detail.empty()) detail = gdn_transaction_status_string(status);
            }
        }
        if (ple_session != nullptr && ple_session->transaction_open()) {
            const ple_adapter_status status = ple_session->rollback(&detail);
            if (status != ple_adapter_status::ok) success = false;
        }
        early_ple_pending = false;
        if (synchronize && cudaStreamSynchronize(stream) != cudaSuccess) {
            success = false;
            if (detail.empty()) detail = "CUDA stream rollback failed";
        }
        transaction_open = false;
        prepared_to_commit = false;
        staged_tokens = 0u;
        eagerly_validated_moe_rows = 0u;
        if (!success) {
            poisoned = true;
            set_error(error, detail.empty() ? "decoder rollback failed" : detail);
        }
        return success;
    }

    ~impl() {
        if (device >= 0) (void)cudaSetDevice(device);
        if (transaction_open || early_ple_pending) {
            std::string ignored;
            (void)rollback_components(bound_stream, true, &ignored);
        }
        ple_session.reset();
        if (gdn_transaction.initialized && !gdn_transaction.transaction_open) {
            (void)gdn_transaction_state_release(&gdn_transaction);
        }
        if (gdn_state.initialized) (void)gdn_device_state_release(&gdn_state);
        for (auto iterator = allocations.rbegin();
             iterator != allocations.rend(); ++iterator) {
            if (*iterator != nullptr) (void)cudaFree(*iterator);
        }
    }
};

const char *decoder_layer_status_string(decoder_layer_status status) noexcept {
    switch (status) {
        case decoder_layer_status::ok: return "ok";
        case decoder_layer_status::invalid_argument: return "invalid_argument";
        case decoder_layer_status::unsupported_layer: return "unsupported_layer";
        case decoder_layer_status::unsupported_config: return "unsupported_config";
        case decoder_layer_status::invalid_checkpoint: return "invalid_checkpoint";
        case decoder_layer_status::allocation_failure: return "allocation_failure";
        case decoder_layer_status::invalid_device_pointer:
            return "invalid_device_pointer";
        case decoder_layer_status::invalid_state: return "invalid_state";
        case decoder_layer_status::capacity_exceeded: return "capacity_exceeded";
        case decoder_layer_status::mhc_error: return "mhc_error";
        case decoder_layer_status::gdn_error: return "gdn_error";
        case decoder_layer_status::qsa_error: return "qsa_error";
        case decoder_layer_status::ple_error: return "ple_error";
        case decoder_layer_status::moe_error: return "moe_error";
        case decoder_layer_status::transaction_error: return "transaction_error";
        case decoder_layer_status::cuda_error: return "cuda_error";
    }
    return "unknown";
}

decoder_layer_kind decoder_layer_kind_for(std::size_t layer_index) noexcept {
    return layer_index < kDecoderLayerCount && (layer_index % 4u) == 3u
            ? decoder_layer_kind::qsa
            : decoder_layer_kind::gated_deltanet;
}

bool decoder_layer_has_ple(std::size_t layer_index) noexcept {
    return layer_index == kPleAdapterLayerIndex;
}

decoder_layer_status decoder_layer_validate_config(
        const decoder_layer_config &config) noexcept {
    if (config.layer_index >= kDecoderLayerCount || config.max_batch == 0u) {
        return decoder_layer_status::invalid_argument;
    }
    if (config.max_batch > kDecoderTokenBatch || config.max_context == 0u ||
        config.max_context > kDecoderMaxContext ||
        config.upload_chunk_bytes < 256u * 1024u ||
        config.blas_workspace_bytes == 0u ||
        (config.blas_workspace_bytes % 256u) != 0u ||
        config.moe_ram_capacity_bytes == 0u ||
        config.moe_prefetch_workers == 0u) {
        return decoder_layer_status::unsupported_config;
    }
    return decoder_layer_status::ok;
}

decoder_layer_status decoder_layer_validate_checkpoint_schedule(
        const checkpoint_catalog &catalog,
        std::string *error) noexcept {
    if (catalog.tensor_count() == 0u) {
        return fail(decoder_layer_status::invalid_checkpoint, error,
                    "empty qwen4_exp checkpoint catalog");
    }
    for (std::size_t layer = 0u; layer < kDecoderLayerCount; ++layer) {
        const std::string prefix = layer_prefix(layer);
        const bool qsa = catalog.find(prefix + "self_attn.q_proj.weight") != nullptr;
        const bool gdn =
                catalog.find(prefix + "linear_attn.in_proj_qkv.weight") != nullptr;
        const bool expected_qsa = decoder_layer_kind_for(layer) ==
                                  decoder_layer_kind::qsa;
        if (qsa == gdn || qsa != expected_qsa) {
            return fail(decoder_layer_status::invalid_checkpoint, error,
                        "checkpoint layer " + std::to_string(layer) +
                                " violates the GDN,GDN,GDN,QSA schedule");
        }
        const std::array<std::string, 3> required{
            prefix + "attn_hyper_connection.hc_norm.weight",
            prefix + "mlp_hyper_connection.hc_norm.weight",
            prefix + "mlp.gate.weight",
        };
        for (const std::string &name : required) {
            if (catalog.find(name) == nullptr) {
                return fail(decoder_layer_status::invalid_checkpoint, error,
                            "checkpoint schedule tensor missing: " + name);
            }
        }
    }

    std::size_t ple_tensors = 0u;
    constexpr char expected_prefix[] =
            "model.language_model.layers.1.ple.";
    for (std::size_t index = 0u; index < catalog.tensor_count(); ++index) {
        const tensor_span *span = catalog.at(index);
        if (span == nullptr || span->name.find(".ple.") == std::string::npos) {
            continue;
        }
        ++ple_tensors;
        if (span->name.rfind(expected_prefix, 0u) != 0u) {
            return fail(decoder_layer_status::invalid_checkpoint, error,
                        "PLE tensor exists outside zero-based layer 1: " +
                                span->name);
        }
    }
    if (ple_tensors != kPleAdapterCheckpointTensors) {
        return fail(decoder_layer_status::invalid_checkpoint, error,
                    "unexpected PLE tensor count in checkpoint: " +
                            std::to_string(ple_tensors));
    }
    return decoder_layer_status::ok;
}

decoder_layer::decoder_layer() = default;
decoder_layer::~decoder_layer() = default;
decoder_layer::decoder_layer(decoder_layer &&) noexcept = default;
decoder_layer &decoder_layer::operator=(decoder_layer &&) noexcept = default;

decoder_layer_status decoder_layer::load(
        const checkpoint_catalog &catalog,
        const decoder_layer_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<decoder_layer> *out,
        std::string *error) noexcept {
    if (out == nullptr) {
        return fail(decoder_layer_status::invalid_argument, error,
                    "null decoder layer output");
    }
    out->reset();
    if (error != nullptr) error->clear();
    const decoder_layer_status config_status =
            decoder_layer_validate_config(config);
    if (config_status != decoder_layer_status::ok) {
        return fail(config_status, error, "invalid decoder layer configuration");
    }
    const decoder_layer_status schedule_status =
            decoder_layer_validate_checkpoint_schedule(catalog, error);
    if (schedule_status != decoder_layer_status::ok) return schedule_status;

    int device = -1;
    if (!current_sm120_device(&device)) {
        return fail(decoder_layer_status::unsupported_config, error,
                    "qwen4_exp decoder requires an active SM120 device");
    }

    try {
        auto result = std::make_unique<decoder_layer>();
        result->impl_ = std::make_unique<impl>();
        impl &state = *result->impl_;
        state.config = config;
        state.kind = decoder_layer_kind_for(config.layer_index);
        state.device = device;

        mhc_adapter_config mhc_config{};
        mhc_config.max_batch = config.max_batch;
        mhc_config.upload_chunk_bytes = config.upload_chunk_bytes;
        mhc_config.blas_workspace_bytes = config.blas_workspace_bytes;
        mhc_adapter_status mhc_status = resident_mhc_adapter::load(
                catalog, static_cast<std::uint32_t>(config.layer_index),
                mhc_adapter_site::attention, mhc_config,
                initialization_stream, &state.attention_mhc, error);
        if (mhc_status != mhc_adapter_status::ok) return map_mhc(mhc_status);
        mhc_status = resident_mhc_adapter::load(
                catalog, static_cast<std::uint32_t>(config.layer_index),
                mhc_adapter_site::mlp, mhc_config, initialization_stream,
                &state.mlp_mhc, error);
        if (mhc_status != mhc_adapter_status::ok) return map_mhc(mhc_status);

        if (state.kind == decoder_layer_kind::gated_deltanet) {
            GdnAdapterConfig gdn_config{};
            gdn_config.layer_index = config.layer_index;
            gdn_config.max_batch = config.max_batch;
            gdn_config.upload_chunk_bytes = config.upload_chunk_bytes;
            gdn_config.blas_workspace_bytes = config.blas_workspace_bytes;
            const GdnAdapterStatus status = GdnLayerAdapter::load(
                    catalog, gdn_config, initialization_stream, &state.gdn,
                    error);
            if (status != GdnAdapterStatus::kOk) return map_gdn(status);
        } else {
            QsaAdapterConfig qsa_config{};
            qsa_config.layer_index = config.layer_index;
            qsa_config.max_batch = config.max_batch;
            qsa_config.max_context = config.max_context;
            qsa_config.upload_chunk_bytes = config.upload_chunk_bytes;
            qsa_config.blas_workspace_bytes = config.blas_workspace_bytes;
            const QsaAdapterStatus status = QsaLayerAdapter::load(
                    catalog, qsa_config, initialization_stream, &state.qsa,
                    error);
            if (status != QsaAdapterStatus::kOk) return map_qsa(status);
        }

        moe_adapter_options moe_options{};
        moe_options.device = device;
        moe_options.layer = static_cast<std::uint32_t>(config.layer_index);
        moe_options.ram_capacity_bytes = config.moe_ram_capacity_bytes;
        moe_options.prefetch_workers = config.moe_prefetch_workers;
        moe_options.owned_direct_pread = config.moe_owned_direct_pread;
        moe_options.blas_workspace_bytes = config.blas_workspace_bytes;
        moe_options.shared_expert_arena = config.shared_expert_arena;
        moe_options.shared_checkpoint_catalog = &catalog;
        moe_options.shared_expert_catalog = config.shared_expert_catalog;
        const moe_adapter_status moe_status = routed_moe_layer_adapter::load(
                catalog.model_root(), moe_options, initialization_stream,
                &state.moe, error);
        if (moe_status != moe_adapter_status::ok) return map_moe(moe_status);

        if (config.enable_ple && decoder_layer_has_ple(config.layer_index)) {
            ple_adapter_config ple_config{};
            ple_config.layer_index = config.layer_index;
            const ple_adapter_status status = ple_layer_adapter::load(
                    catalog, ple_config, initialization_stream, &state.ple,
                    error);
            if (status != ple_adapter_status::ok) return map_ple(status);
        }
        if (cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
            return fail(decoder_layer_status::cuda_error, error,
                        "decoder layer initialization stream failed");
        }
        state.ready = true;
        *out = std::move(result);
        return decoder_layer_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(decoder_layer_status::allocation_failure, error,
                    "host allocation failed while loading decoder layer");
    } catch (const std::exception &exception) {
        return fail(decoder_layer_status::invalid_checkpoint, error,
                    std::string("decoder layer load exception: ") +
                            exception.what());
    } catch (...) {
        return fail(decoder_layer_status::invalid_checkpoint, error,
                    "unknown decoder layer load failure");
    }
}

decoder_layer_status decoder_layer::create_session(
        std::size_t context_capacity,
        cudaStream_t initialization_stream,
        std::unique_ptr<decoder_layer_session> *out,
        std::string *error) noexcept {
    if (out == nullptr || impl_ == nullptr || !impl_->ready) {
        return fail(decoder_layer_status::invalid_argument, error,
                    "invalid decoder session creation request");
    }
    out->reset();
    if (context_capacity == 0u || context_capacity > impl_->config.max_context) {
        return fail(decoder_layer_status::capacity_exceeded, error,
                    "decoder session context exceeds the admitted capacity");
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != impl_->device) {
        return fail(decoder_layer_status::unsupported_config, error,
                    "decoder session was created on the wrong CUDA device");
    }

    try {
        auto result = std::make_unique<decoder_layer_session>();
        result->impl_ = std::make_unique<decoder_layer_session::impl>();
        decoder_layer_session::impl &session = *result->impl_;
        session.owner = impl_.get();
        session.device = impl_->device;
        session.bound_stream = initialization_stream;
        session.context_capacity = context_capacity;

        mhc_adapter_config mhc_config{};
        mhc_config.max_batch = impl_->config.max_batch;
        mhc_config.upload_chunk_bytes = impl_->config.upload_chunk_bytes;
        mhc_config.blas_workspace_bytes = impl_->config.blas_workspace_bytes;
        mhc_adapter_workspace_requirements mhc_required{};
        const mhc_adapter_status mhc_status =
                mhc_adapter_get_workspace_requirements(mhc_config, &mhc_required);
        if (mhc_status != mhc_adapter_status::ok) return map_mhc(mhc_status);

        if (!session.allocate_typed(&session.mhc_scratch.normalized,
                                    mhc_required.normalized_f32_bytes, error,
                                    "mHC normalized") ||
            !session.allocate_typed(&session.mhc_scratch.low_rank,
                                    mhc_required.low_rank_f32_bytes, error,
                                    "mHC low rank") ||
            !session.allocate_typed(&session.mhc_scratch.mix_logits,
                                    mhc_required.mix_logits_f32_bytes, error,
                                    "mHC mix logits") ||
            !session.allocate_typed(&session.mhc_scratch.injection_logits,
                                    mhc_required.injection_logits_f32_bytes,
                                    error, "mHC injection logits") ||
            !session.allocate_typed(&session.mhc_scratch.linear_input_bf16,
                                    mhc_required.linear_input_bf16_bytes, error,
                                    "mHC BF16 input") ||
            !session.allocate_bytes(&session.mhc_scratch.blas_workspace,
                                    mhc_required.blas_workspace_bytes, error,
                                    "mHC BLAS workspace")) {
            return decoder_layer_status::allocation_failure;
        }
        session.mhc_scratch.linear_input_bf16_bytes =
                mhc_required.linear_input_bf16_bytes;
        session.mhc_scratch.blas_workspace_bytes =
                mhc_required.blas_workspace_bytes;

        const auto allocate_f32 = [&](float **pointer,
                                      std::size_t elements,
                                      const char *label) {
            std::size_t bytes = 0u;
            return checked_mul(elements, kF32Bytes, &bytes) &&
                   session.allocate_typed(pointer, bytes, error, label);
        };
        const std::size_t max_batch = impl_->config.max_batch;
        if (!allocate_f32(&session.ple_hidden, max_batch * kDecoderResidual,
                          "PLE hidden") ||
            !allocate_f32(&session.attention_mixed, max_batch * kDecoderHidden,
                          "attention mixed") ||
            !allocate_f32(&session.attention_injection,
                          max_batch * kDecoderStreams,
                          "attention injection") ||
            !allocate_f32(&session.attention_block, max_batch * kDecoderHidden,
                          "attention block") ||
            !allocate_f32(&session.after_attention, max_batch * kDecoderResidual,
                          "post-attention residual") ||
            !allocate_f32(&session.mlp_mixed, max_batch * kDecoderHidden,
                          "MLP mixed") ||
            !allocate_f32(&session.mlp_injection,
                          max_batch * kDecoderStreams,
                          "MLP injection") ||
            !allocate_f32(&session.mlp_block, max_batch * kDecoderHidden,
                          "MLP block")) {
            return decoder_layer_status::allocation_failure;
        }

        if (impl_->gdn != nullptr) {
            const GdnConfig state_config = gdn_qwen4_exp_config(1u);
            const GdnStatus state_status = gdn_device_state_init(
                    &session.gdn_state, state_config, initialization_stream);
            if (state_status != GdnStatus::kOk) {
                return fail(decoder_layer_status::gdn_error, error,
                            std::string("GDN state init: ") +
                                    gdn_status_string(state_status));
            }
            const GdnTransactionStatus transaction_status =
                    gdn_transaction_state_init(&session.gdn_transaction,
                                               &session.gdn_state,
                                               initialization_stream);
            if (transaction_status != GdnTransactionStatus::kOk) {
                return fail(map_gdn_transaction(transaction_status), error,
                            std::string("GDN transaction init: ") +
                                    gdn_transaction_status_string(
                                            transaction_status));
            }
            GdnAdapterConfig active = impl_->gdn->config();
            active.max_batch = impl_->config.max_batch;
            GdnAdapterWorkspaceRequirements required{};
            const GdnAdapterStatus requirement_status =
                    gdn_adapter_workspace_requirements(active, &required);
            if (requirement_status != GdnAdapterStatus::kOk) {
                return map_gdn(requirement_status);
            }
            auto &scratch = session.gdn_scratch;
            if (!session.allocate_typed(&scratch.linear_input_bf16,
                                        required.linear_input_bf16_bytes, error,
                                        "GDN BF16 input") ||
                !session.allocate_bytes(&scratch.blas_workspace,
                                        required.blas_workspace_bytes, error,
                                        "GDN BLAS workspace") ||
                !session.allocate_typed(&scratch.projected_qkv,
                                        required.projected_qkv_f32_bytes, error,
                                        "GDN qkv") ||
                !session.allocate_typed(&scratch.z, required.z_f32_bytes,
                                        error, "GDN z") ||
                !session.allocate_typed(&scratch.a, required.a_f32_bytes,
                                        error, "GDN a") ||
                !session.allocate_typed(&scratch.b, required.b_f32_bytes,
                                        error, "GDN b") ||
                !session.allocate_typed(&scratch.convolved_qkv,
                                        required.convolved_qkv_f32_bytes, error,
                                        "GDN convolved") ||
                !session.allocate_typed(&scratch.gdn_output,
                                        required.gdn_output_f32_bytes, error,
                                        "GDN output")) {
                return decoder_layer_status::allocation_failure;
            }
            scratch.linear_input_bf16_bytes = required.linear_input_bf16_bytes;
            scratch.blas_workspace_bytes = required.blas_workspace_bytes;
            scratch.projected_qkv_f32_bytes = required.projected_qkv_f32_bytes;
            scratch.z_f32_bytes = required.z_f32_bytes;
            scratch.a_f32_bytes = required.a_f32_bytes;
            scratch.b_f32_bytes = required.b_f32_bytes;
            scratch.convolved_qkv_f32_bytes = required.convolved_qkv_f32_bytes;
            scratch.gdn_output_f32_bytes = required.gdn_output_f32_bytes;
        } else {
            QsaAdapterConfig active = impl_->qsa->config();
            active.max_batch = impl_->config.max_batch;
            active.max_context = context_capacity;
            QsaAdapterWorkspaceRequirements required{};
            const QsaAdapterStatus requirement_status =
                    qsa_adapter_workspace_requirements(active, &required);
            if (requirement_status != QsaAdapterStatus::kOk) {
                return map_qsa(requirement_status);
            }
            std::size_t main_cache_elements = 0u;
            std::size_t index_cache_elements = 0u;
            if (!checked_mul(context_capacity,
                             attention::kCheckpointKvHeads *
                                     attention::kCheckpointHeadDim,
                             &main_cache_elements) ||
                !checked_mul(context_capacity, kQsaAdapterIndexerKeySize,
                             &index_cache_elements)) {
                return decoder_layer_status::allocation_failure;
            }
            std::uint16_t *key_cache = nullptr;
            std::uint16_t *value_cache = nullptr;
            std::uint16_t *index_cache = nullptr;
            std::size_t main_cache_bytes = 0u;
            std::size_t index_cache_bytes = 0u;
            if (!checked_mul(main_cache_elements, kBf16Bytes,
                             &main_cache_bytes) ||
                !checked_mul(index_cache_elements, kBf16Bytes,
                             &index_cache_bytes) ||
                !session.allocate_typed(&key_cache, main_cache_bytes, error,
                                        "QSA K cache") ||
                !session.allocate_typed(&value_cache, main_cache_bytes, error,
                                        "QSA V cache") ||
                !session.allocate_typed(&index_cache, index_cache_bytes, error,
                                        "QSA index cache")) {
                return decoder_layer_status::allocation_failure;
            }
            const QsaAdapterStatus cache_status = qsa_adapter_cache_initialize(
                    &session.qsa_cache, key_cache, value_cache, index_cache,
                    context_capacity, 0u);
            if (cache_status != QsaAdapterStatus::kOk) return map_qsa(cache_status);

            auto &scratch = session.qsa_scratch;
#define AXIOM_Q4_ALLOC_QSA(field, bytes, label)                              \
            if (!session.allocate_typed(&scratch.field, required.bytes, error, \
                                        label)) {                            \
                return decoder_layer_status::allocation_failure;             \
            }                                                                \
            scratch.bytes = required.bytes
            AXIOM_Q4_ALLOC_QSA(linear_input_bf16, linear_input_bf16_bytes,
                              "QSA BF16 input");
            AXIOM_Q4_ALLOC_QSA(q_projection, q_projection_f32_bytes,
                              "QSA q projection");
            AXIOM_Q4_ALLOC_QSA(k_projection, k_projection_f32_bytes,
                              "QSA k projection");
            AXIOM_Q4_ALLOC_QSA(v_projection, v_projection_f32_bytes,
                              "QSA v projection");
            AXIOM_Q4_ALLOC_QSA(index_projection, index_projection_f32_bytes,
                              "QSA index projection");
            AXIOM_Q4_ALLOC_QSA(q_projection_bf16, q_projection_bf16_bytes,
                              "QSA q BF16");
            AXIOM_Q4_ALLOC_QSA(k_projection_bf16, k_projection_bf16_bytes,
                              "QSA k BF16");
            AXIOM_Q4_ALLOC_QSA(v_projection_bf16, v_projection_bf16_bytes,
                              "QSA v BF16");
            AXIOM_Q4_ALLOC_QSA(index_query_bf16, index_query_bf16_bytes,
                              "QSA index query BF16");
            AXIOM_Q4_ALLOC_QSA(index_query_prepared,
                              index_query_prepared_f32_bytes,
                              "QSA prepared query");
            AXIOM_Q4_ALLOC_QSA(visible_indices, visible_indices_bytes,
                              "QSA visible indices");
            AXIOM_Q4_ALLOC_QSA(pooled_index_keys,
                              pooled_index_keys_f32_bytes,
                              "QSA pooled keys");
            AXIOM_Q4_ALLOC_QSA(index_scores, index_scores_f32_bytes,
                              "QSA index scores");
            AXIOM_Q4_ALLOC_QSA(selected_indices, selected_indices_bytes,
                              "QSA selected indices");
            AXIOM_Q4_ALLOC_QSA(selected_counts, selected_counts_bytes,
                              "QSA selected counts");
            AXIOM_Q4_ALLOC_QSA(attention_output_bf16,
                              attention_output_bf16_bytes,
                              "QSA attention BF16 output");
            AXIOM_Q4_ALLOC_QSA(attention_output_f32,
                              attention_output_f32_bytes,
                              "QSA attention F32 output");
#undef AXIOM_Q4_ALLOC_QSA
            if (!session.allocate_bytes(&scratch.blas_workspace,
                                        required.blas_workspace_bytes, error,
                                        "QSA BLAS workspace") ||
                !session.allocate_bytes(&scratch.selection_workspace,
                                        required.selection_workspace_bytes,
                                        error, "QSA selection workspace") ||
                !session.allocate_bytes(&scratch.attention_workspace,
                                        required.attention_workspace_bytes,
                                        error, "QSA attention workspace") ||
                !session.allocate_typed(&scratch.qsa_device_status,
                                        required.qsa_device_status_bytes, error,
                                        "QSA device status") ||
                !session.allocate_typed(&scratch.attention_device_status,
                                        required.attention_device_status_bytes,
                                        error, "attention device status")) {
                return decoder_layer_status::allocation_failure;
            }
            scratch.blas_workspace_bytes = required.blas_workspace_bytes;
            scratch.selection_workspace_bytes =
                    required.selection_workspace_bytes;
            scratch.attention_workspace_bytes =
                    required.attention_workspace_bytes;
        }

        if (impl_->ple != nullptr) {
            const ple_adapter_status status = impl_->ple->create_session(
                    initialization_stream, &session.ple_session, error);
            if (status != ple_adapter_status::ok) return map_ple(status);
        }
        if (cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
            return fail(decoder_layer_status::cuda_error, error,
                        "decoder session initialization stream failed");
        }
        session.ready = true;
        *out = std::move(result);
        return decoder_layer_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(decoder_layer_status::allocation_failure, error,
                    "host allocation failed while creating decoder session");
    } catch (...) {
        return fail(decoder_layer_status::allocation_failure, error,
                    "unexpected decoder session creation failure");
    }
}

const decoder_layer_config &decoder_layer::config() const noexcept {
    static const decoder_layer_config empty{};
    return impl_ != nullptr ? impl_->config : empty;
}

decoder_layer_kind decoder_layer::kind() const noexcept {
    return impl_ != nullptr ? impl_->kind : decoder_layer_kind::gated_deltanet;
}

bool decoder_layer::has_ple() const noexcept {
    return impl_ != nullptr && impl_->ple != nullptr;
}

int decoder_layer::device() const noexcept {
    return impl_ != nullptr ? impl_->device : -1;
}

bool decoder_layer::initialized() const noexcept {
    return impl_ != nullptr && impl_->ready;
}

ExpertPagerMetrics decoder_layer::expert_pager_metrics() const noexcept {
    return impl_ && impl_->moe
            ? impl_->moe->pager_metrics() : ExpertPagerMetrics{};
}

decoder_layer_session::decoder_layer_session() = default;
decoder_layer_session::~decoder_layer_session() = default;
decoder_layer_session::decoder_layer_session(decoder_layer_session &&) noexcept =
        default;
decoder_layer_session &decoder_layer_session::operator=(
        decoder_layer_session &&) noexcept = default;

decoder_layer_status decoder_layer_session::prefetch_first_ple_token(
        std::int64_t token_id, cudaStream_t stream, std::string *error) noexcept {
    if (!impl_ || !impl_->ready || !impl_->owner || impl_->poisoned ||
        impl_->transaction_open || impl_->prepared_to_commit ||
        impl_->early_ple_pending || stream != impl_->bound_stream) {
        return fail(decoder_layer_status::invalid_state, error,
                    "early PLE prefetch requires an idle layer on its bound stream");
    }
    auto &session = *impl_;
    if (!session.ple_session) return decoder_layer_status::ok;
    if (session.committed_tokens >= session.context_capacity) {
        return fail(decoder_layer_status::capacity_exceeded, error,
                    "early PLE prefetch exceeds decoder capacity");
    }
    auto status = session.ple_session->begin_transaction(error);
    if (status != ple_adapter_status::ok) return map_ple(status);
    status = session.ple_session->start_prefetch_token(token_id, error);
    if (status != ple_adapter_status::ok) {
        std::string ignored;
        if (session.ple_session->transaction_open() &&
            session.ple_session->rollback(&ignored) != ple_adapter_status::ok) {
            session.poisoned = true;
            return decoder_layer_status::transaction_error;
        }
        return map_ple(status);
    }
    session.early_ple_token = token_id;
    session.early_ple_pending = true;
    return decoder_layer_status::ok;
}

decoder_layer_status decoder_layer_session::stage_token(
        std::int64_t token_id,
        const float *residual_4x2560_f32,
        const float *full_cos_64_f32,
        const float *full_sin_64_f32,
        std::size_t position_count,
        float *output_4x2560_f32,
        cudaStream_t stream,
        std::string *error) noexcept {
    return stage_sequence(&token_id, 1u, residual_4x2560_f32,
                          full_cos_64_f32, full_sin_64_f32, position_count,
                          output_4x2560_f32, stream, error);
}

decoder_layer_status decoder_layer_session::stage_sequence(
        const std::int64_t *token_ids,
        std::size_t token_count,
        const float *residual_4x2560_f32,
        const float *full_cos_64_f32,
        const float *full_sin_64_f32,
        std::size_t position_count,
        float *output_4x2560_f32,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->owner == nullptr ||
        token_ids == nullptr || token_count == 0u ||
        token_count > impl_->owner->config.max_batch ||
        token_count > kDecoderMaxSpeculativeTokens ||
        residual_4x2560_f32 == nullptr || output_4x2560_f32 == nullptr ||
        residual_4x2560_f32 == output_4x2560_f32) {
        return fail(decoder_layer_status::invalid_argument, error,
                    "invalid decoder stage request");
    }
    impl &session = *impl_;
    if (session.poisoned || session.transaction_open ||
        session.prepared_to_commit || stream != session.bound_stream) {
        return fail(decoder_layer_status::invalid_state, error,
                    "decoder session is poisoned, busy, or on a wrong stream");
    }
    if (session.early_ple_pending && session.early_ple_token != token_ids[0]) {
        return fail(decoder_layer_status::invalid_state, error,
                    "early PLE token does not match the staged sequence");
    }
    if (token_count > session.context_capacity ||
        session.committed_tokens > session.context_capacity - token_count) {
        return fail(decoder_layer_status::capacity_exceeded, error,
                    "decoder context capacity reached");
    }
    if (session.owner->kind == decoder_layer_kind::qsa &&
        (full_cos_64_f32 == nullptr || full_sin_64_f32 == nullptr ||
         position_count < session.committed_tokens + token_count ||
         position_count > session.context_capacity)) {
        return fail(decoder_layer_status::invalid_argument, error,
                    "QSA stage requires complete in-range RoPE tables");
    }

    const std::size_t residual_bytes =
            token_count * kDecoderResidual * sizeof(float);
    const device_range input_range{residual_4x2560_f32, residual_bytes};
    const device_range output_range{output_4x2560_f32, residual_bytes};
    if (!valid_device_range(input_range, session.device) ||
        !valid_device_range(output_range, session.device) ||
        ranges_overlap(input_range, output_range)) {
        return fail(decoder_layer_status::invalid_device_pointer, error,
                    "decoder residual input/output must be disjoint CUDA ranges");
    }
    if (session.owner->kind == decoder_layer_kind::qsa) {
        std::size_t rope_elements = 0u;
        std::size_t rope_bytes = 0u;
        if (!checked_mul(position_count, attention::kCheckpointRotaryDim,
                         &rope_elements) ||
            !checked_mul(rope_elements, sizeof(float), &rope_bytes)) {
            return fail(decoder_layer_status::invalid_argument, error,
                        "QSA RoPE range overflows");
        }
        const device_range cosine_range{full_cos_64_f32, rope_bytes};
        const device_range sine_range{full_sin_64_f32, rope_bytes};
        if (!valid_device_range(cosine_range, session.device) ||
            !valid_device_range(sine_range, session.device) ||
            ranges_overlap(cosine_range, sine_range) ||
            ranges_overlap(input_range, cosine_range) ||
            ranges_overlap(input_range, sine_range) ||
            ranges_overlap(output_range, cosine_range) ||
            ranges_overlap(output_range, sine_range)) {
            return fail(decoder_layer_status::invalid_device_pointer, error,
                        "QSA RoPE tables must be disjoint CUDA ranges");
        }
    }

    auto abort = [&](decoder_layer_status status,
                     const std::string &message) noexcept {
        set_error(error, message);
        std::string rollback_error;
        if (!session.rollback_components(stream, true, &rollback_error)) {
            if (error != nullptr && !rollback_error.empty()) {
                try {
                    *error += "; rollback: " + rollback_error;
                } catch (...) {
                }
            }
            return decoder_layer_status::transaction_error;
        }
        return status;
    };

    session.transaction_open = true;
    session.staged_tokens = 0u;
    session.eagerly_validated_moe_rows = 0u;
    const float *attention_input = residual_4x2560_f32;
    if (session.ple_session != nullptr) {
        ple_adapter_status status = session.early_ple_pending
                ? ple_adapter_status::ok
                : session.ple_session->begin_transaction(error);
        if (status != ple_adapter_status::ok) {
            return abort(map_ple(status), error && !error->empty()
                    ? *error : "PLE transaction begin failed");
        }
        for (std::size_t row = 0u; row < token_count; ++row) {
            status = row == 0u && session.early_ple_pending
                    ? session.ple_session->finish_prefetch_token(stream, error)
                    : session.ple_session->prefetch_token(token_ids[row], stream, error);
            if (status != ple_adapter_status::ok) {
                return abort(map_ple(status), error && !error->empty()
                        ? *error : "PLE token prefetch failed");
            }
            session.early_ple_pending = false;
            status = session.ple_session->stage_prefetched_and_add_cuda(
                    residual_4x2560_f32 + row * kDecoderResidual,
                    session.ple_hidden + row * kDecoderResidual,
                    stream, error);
            if (status != ple_adapter_status::ok) {
                return abort(map_ple(status), error && !error->empty()
                        ? *error : "PLE stage failed");
            }
        }
        attention_input = session.ple_hidden;
    }

    mhc_adapter_status mhc_status = session.owner->attention_mhc->prepare(
            attention_input, token_count, session.mhc_scratch,
            session.attention_mixed, session.attention_injection, stream);
    if (mhc_status != mhc_adapter_status::ok) {
        return abort(map_mhc(mhc_status),
                     std::string("attention mHC prepare failed: ") +
                             mhc_adapter_status_string(mhc_status));
    }

    if (session.owner->qsa != nullptr) {
        QsaAdapterStatus qsa_status = session.owner->qsa->begin_transaction(
                &session.qsa_cache, session.qsa_scratch, stream);
        if (qsa_status != QsaAdapterStatus::kOk) {
            return abort(map_qsa(qsa_status), "QSA transaction begin failed");
        }
        qsa_status = session.owner->qsa->forward(
                session.attention_mixed, token_count, full_cos_64_f32,
                full_sin_64_f32, position_count, &session.qsa_cache,
                session.qsa_scratch, session.attention_block, stream);
        if (qsa_status != QsaAdapterStatus::kOk) {
            return abort(map_qsa(qsa_status),
                         std::string("QSA forward failed: ") +
                                 qsa_adapter_status_string(qsa_status));
        }
        QsaAdapterCollectedStatus collected{};
        qsa_status = session.owner->qsa->collect_device_status(
                &session.qsa_cache, session.qsa_scratch, &collected, stream);
        if (qsa_status != QsaAdapterStatus::kOk) {
            return abort(map_qsa(qsa_status),
                         std::string("QSA device gate failed: ") +
                                 qsa_adapter_status_string(qsa_status));
        }
    } else {
        const GdnTransactionStatus begin_status = gdn_transaction_begin(
                &session.gdn_transaction, stream);
        if (begin_status != GdnTransactionStatus::kOk) {
            return abort(map_gdn_transaction(begin_status),
                         "GDN transaction begin failed");
        }
        const GdnAdapterStatus gdn_status = token_count == 1u
                ? session.owner->gdn->forward(
                          session.attention_mixed, 1u, &session.gdn_state,
                          session.gdn_scratch, session.attention_block, stream)
                : session.owner->gdn->forward_sequence(
                          session.attention_mixed, token_count,
                          &session.gdn_state, session.gdn_scratch,
                          session.attention_block, stream);
        if (gdn_status != GdnAdapterStatus::kOk) {
            return abort(map_gdn(gdn_status),
                         std::string("GDN forward failed: ") +
                                 gdn_adapter_status_string(gdn_status));
        }
        /* GdnLayerAdapter owns the real projections and therefore performs the
         * public gdn_step_cuda call itself.  Bind that one verified step to the
         * already-open public transaction snapshot before publication. */
        const std::uint64_t expected_tokens =
                session.gdn_transaction.committed_tokens_seen + token_count;
        const std::uint32_t expected_cursor = static_cast<std::uint32_t>(
                (session.gdn_transaction.committed_conv_cursor + token_count) %
                session.gdn_state.config.conv_kernel);
        if (session.gdn_state.tokens_seen != expected_tokens ||
            session.gdn_state.conv_cursor != expected_cursor) {
            return abort(decoder_layer_status::transaction_error,
                         "GDN adapter did not advance the exact staged sequence");
        }
        session.gdn_transaction.staged_steps = token_count;
    }

    mhc_status = session.owner->attention_mhc->reinject(
            attention_input, session.attention_block,
            session.attention_injection, token_count,
            session.after_attention, stream);
    if (mhc_status != mhc_adapter_status::ok) {
        return abort(map_mhc(mhc_status), "attention mHC reinject failed");
    }
    mhc_status = session.owner->mlp_mhc->prepare(
            session.after_attention, token_count, session.mhc_scratch,
            session.mlp_mixed, session.mlp_injection, stream);
    if (mhc_status != mhc_adapter_status::ok) {
        return abort(map_mhc(mhc_status), "MLP mHC prepare failed");
    }

    for (std::size_t row = 0u; row < token_count; ++row) {
        float *const moe_input = session.mlp_mixed + row * kDecoderHidden;
        float *const moe_output = session.mlp_block + row * kDecoderHidden;
        moe_adapter_status moe_status = session.owner->moe->enqueue_route(
                moe_input, stream, error);
        if (moe_status != moe_adapter_status::ok) {
            return abort(map_moe(moe_status),
                         std::string("MoE route failed: ") +
                                 moe_adapter_status_string(moe_status));
        }
        routed_moe_layer_adapter::prepared_generation *generation =
                &session.moe_generation;
        routed_moe_layer_adapter::prepared_generation eager_generation{};
        if (token_count > 1u) generation = &eager_generation;
        moe_status = session.owner->moe->prepare_route(
                stream, generation, error);
        if (moe_status != moe_adapter_status::ok) {
            return abort(map_moe(moe_status),
                         std::string("MoE pager preparation failed: ") +
                                 moe_adapter_status_string(moe_status));
        }
        moe_status = generation->forward(
                moe_input, moe_output, stream, error);
        if (moe_status != moe_adapter_status::ok) {
            return abort(map_moe(moe_status),
                         std::string("MoE forward failed: ") +
                                 moe_adapter_status_string(moe_status));
        }
        if (token_count > 1u) {
            std::string validation_error;
            if (!generation->commit(&validation_error)) {
                return abort(decoder_layer_status::moe_error,
                             validation_error.empty()
                                     ? "MoE block-row validation failed"
                                     : validation_error);
            }
            ++session.eagerly_validated_moe_rows;
        }
    }
    mhc_status = session.owner->mlp_mhc->reinject(
            session.after_attention, session.mlp_block,
            session.mlp_injection, token_count,
            output_4x2560_f32, stream);
    if (mhc_status != mhc_adapter_status::ok) {
        return abort(map_mhc(mhc_status), "MLP mHC reinject failed");
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        return abort(decoder_layer_status::cuda_error,
                     "decoder layer final CUDA launch failed");
    }
    session.staged_tokens = token_count;
    return decoder_layer_status::ok;
}

decoder_layer_status decoder_layer_session::commit(
        cudaStream_t stream,
        std::string *error) noexcept {
    decoder_layer_session *layers[] = {this};
    const decoder_layer_status prepared =
            decoder_model_transaction::prepare_commit(layers, 1u, stream, error);
    if (prepared != decoder_layer_status::ok) return prepared;
    if (!decoder_model_transaction::commit_prepared_noexcept(layers, 1u)) {
        return fail(decoder_layer_status::transaction_error, error,
                    "validated decoder publication invariant failed");
    }
    return decoder_layer_status::ok;
}

decoder_layer_status decoder_layer_session::rollback(
        cudaStream_t stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready ||
        (!impl_->transaction_open && !impl_->early_ple_pending) ||
        stream != impl_->bound_stream) {
        return fail(decoder_layer_status::invalid_state, error,
                    "invalid decoder rollback request");
    }
    return impl_->rollback_components(stream, true, error)
            ? decoder_layer_status::ok
            : decoder_layer_status::transaction_error;
}

decoder_layer_status decoder_layer_session::reset(
        cudaStream_t stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->transaction_open ||
        stream != impl_->bound_stream) {
        return fail(decoder_layer_status::invalid_state, error,
                    "invalid decoder reset request");
    }
    impl &session = *impl_;
    if (session.early_ple_pending &&
        !session.rollback_components(stream, true, error)) {
        return decoder_layer_status::transaction_error;
    }
    if (session.owner->qsa != nullptr) {
        const QsaAdapterStatus status = qsa_adapter_cache_reset(&session.qsa_cache);
        if (status != QsaAdapterStatus::kOk) return map_qsa(status);
    } else {
        const GdnStatus status = gdn_device_state_reset(&session.gdn_state, stream);
        if (status != GdnStatus::kOk) {
            return fail(decoder_layer_status::gdn_error, error,
                        std::string("GDN reset failed: ") +
                                gdn_status_string(status));
        }
        session.gdn_transaction.committed_conv_cursor = 0u;
        session.gdn_transaction.committed_tokens_seen = 0u;
        session.gdn_transaction.staged_steps = 0u;
    }
    if (session.ple_session != nullptr) {
        const ple_adapter_status status = session.ple_session->reset(stream, error);
        if (status != ple_adapter_status::ok) return map_ple(status);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return fail(decoder_layer_status::cuda_error, error,
                    "decoder reset stream failed");
    }
    session.committed_tokens = 0u;
    session.staged_tokens = 0u;
    session.eagerly_validated_moe_rows = 0u;
    session.poisoned = false;
    return decoder_layer_status::ok;
}

decoder_layer_state decoder_layer_session::state() const noexcept {
    decoder_layer_state result{};
    if (impl_ == nullptr || impl_->owner == nullptr) return result;
    result.layer_index = impl_->owner->config.layer_index;
    result.kind = impl_->owner->kind;
    result.context_capacity = impl_->context_capacity;
    result.committed_tokens = impl_->committed_tokens;
    result.staged_tokens = impl_->staged_tokens;
    result.has_ple = impl_->ple_session != nullptr;
    result.transaction_open = impl_->transaction_open;
    result.prepared_to_commit = impl_->prepared_to_commit;
    result.poisoned = impl_->poisoned;
    return result;
}

bool decoder_layer_session::initialized() const noexcept {
    return impl_ != nullptr && impl_->ready;
}

decoder_layer_status decoder_model_transaction::prepare_commit(
        decoder_layer_session *const *layers,
        std::size_t layer_count,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (layers == nullptr || layer_count == 0u ||
        layer_count > kDecoderLayerCount) {
        return fail(decoder_layer_status::invalid_argument, error,
                    "invalid decoder model transaction set");
    }

    std::array<bool, kDecoderLayerCount> observed{};
    std::size_t staged_tokens = 0u;
    for (std::size_t index = 0u; index < layer_count; ++index) {
        decoder_layer_session *layer = layers[index];
        if (layer == nullptr || layer->impl_ == nullptr ||
            !layer->impl_->ready || layer->impl_->owner == nullptr ||
            layer->impl_->poisoned || !layer->impl_->transaction_open ||
            layer->impl_->prepared_to_commit ||
            layer->impl_->bound_stream != stream) {
            return fail(decoder_layer_status::invalid_state, error,
                        "model prepare found an unstaged or invalid layer");
        }
        const std::size_t layer_index = layer->impl_->owner->config.layer_index;
        if (layer_index >= kDecoderLayerCount || observed[layer_index]) {
            return fail(decoder_layer_status::invalid_state, error,
                        "model prepare contains a duplicate decoder layer");
        }
        observed[layer_index] = true;
        if (layer->impl_->staged_tokens == 0u ||
            layer->impl_->staged_tokens > kDecoderMaxSpeculativeTokens ||
            (staged_tokens != 0u &&
             layer->impl_->staged_tokens != staged_tokens)) {
            return fail(decoder_layer_status::invalid_state, error,
                        "model prepare found inconsistent staged token counts");
        }
        staged_tokens = layer->impl_->staged_tokens;
    }

    /* The sole model-wide device completion boundary.  Every later check is
     * host metadata or an already-completed event observation. */
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::string ignored;
        (void)rollback_all(layers, layer_count, stream, &ignored);
        return fail(decoder_layer_status::cuda_error, error,
                    "model-wide decoder synchronization failed");
    }

    auto reject = [&](decoder_layer_status status,
                      const std::string &message) noexcept {
        set_error(error, message);
        std::string rollback_error;
        const decoder_layer_status rollback_status =
                rollback_all(layers, layer_count, stream, &rollback_error);
        if (rollback_status != decoder_layer_status::ok) {
            if (error != nullptr && !rollback_error.empty()) {
                try {
                    *error += "; rollback: " + rollback_error;
                } catch (...) {
                }
            }
            return decoder_layer_status::transaction_error;
        }
        return status;
    };

    for (std::size_t index = 0u; index < layer_count; ++index) {
        decoder_layer_session::impl &session = *layers[index]->impl_;
        if (staged_tokens == 1u) {
            if (!session.moe_generation.valid() ||
                !session.moe_generation.forward_enqueued()) {
                return reject(decoder_layer_status::transaction_error,
                              "model prepare found an unvalidated MoE generation");
            }
            std::string moe_error;
            if (!session.moe_generation.commit(&moe_error)) {
                return reject(decoder_layer_status::moe_error,
                              moe_error.empty()
                                      ? "MoE device gate rejected a layer"
                                      : moe_error);
            }
            session.moe_generation =
                    routed_moe_layer_adapter::prepared_generation{};
        } else if (session.moe_generation.valid() ||
                   session.eagerly_validated_moe_rows != staged_tokens) {
            return reject(decoder_layer_status::transaction_error,
                          "model prepare found an incomplete MoE block gate");
        }

        if (session.owner->qsa != nullptr) {
            QsaAdapterCacheLengths lengths{};
            const QsaAdapterStatus status = qsa_adapter_cache_get_lengths(
                    &session.qsa_cache, &lengths);
            if (status != QsaAdapterStatus::kOk ||
                !session.qsa_cache.transaction_open ||
                lengths.committed != session.committed_tokens ||
                lengths.staged != staged_tokens ||
                lengths.visible != session.committed_tokens + staged_tokens ||
                lengths.capacity != session.context_capacity) {
                return reject(decoder_layer_status::transaction_error,
                              "QSA staged boundary failed model prepare");
            }
        } else {
            const GdnTransactionState &transaction = session.gdn_transaction;
            const std::uint64_t expected_tokens =
                    transaction.committed_tokens_seen + staged_tokens;
            const std::uint32_t expected_cursor = static_cast<std::uint32_t>(
                    (transaction.committed_conv_cursor + staged_tokens) %
                    transaction.bound_config.conv_kernel);
            if (!transaction.initialized || !transaction.transaction_open ||
                transaction.staged_steps != staged_tokens ||
                session.gdn_state.tokens_seen != expected_tokens ||
                session.gdn_state.conv_cursor != expected_cursor ||
                transaction.committed_tokens_seen != session.committed_tokens) {
                return reject(decoder_layer_status::transaction_error,
                              "GDN staged boundary failed model prepare");
            }
        }
        if (session.ple_session != nullptr &&
            (!session.ple_session->transaction_open() ||
             session.ple_session->token_prefetched() ||
             session.ple_session->prefetch_pending() ||
             session.early_ple_pending ||
             session.ple_session->poisoned() ||
             session.ple_session->staged_tokens() != staged_tokens ||
             session.ple_session->committed_tokens() !=
                     session.committed_tokens)) {
            return reject(decoder_layer_status::transaction_error,
                          "PLE staged boundary failed model prepare");
        }
        session.prepared_to_commit = true;
    }
    return decoder_layer_status::ok;
}

bool decoder_model_transaction::commit_prepared_noexcept(
        decoder_layer_session *const *layers,
        std::size_t layer_count) noexcept {
    if (layers == nullptr || layer_count == 0u ||
        layer_count > kDecoderLayerCount) {
        return false;
    }
    decoder_layer_session::impl *ple_owner = nullptr;
    std::array<bool, kDecoderLayerCount> observed{};
    std::size_t staged_tokens = 0u;
    for (std::size_t index = 0u; index < layer_count; ++index) {
        decoder_layer_session *layer = layers[index];
        if (layer == nullptr || layer->impl_ == nullptr ||
            !layer->impl_->ready || layer->impl_->poisoned ||
            !layer->impl_->transaction_open ||
            !layer->impl_->prepared_to_commit ||
            layer->impl_->owner == nullptr) {
            return false;
        }
        const std::size_t layer_index = layer->impl_->owner->config.layer_index;
        if (layer_index >= kDecoderLayerCount || observed[layer_index]) {
            return false;
        }
        observed[layer_index] = true;
        if (layer->impl_->staged_tokens == 0u ||
            layer->impl_->staged_tokens > kDecoderMaxSpeculativeTokens ||
            (staged_tokens != 0u &&
             layer->impl_->staged_tokens != staged_tokens)) {
            return false;
        }
        staged_tokens = layer->impl_->staged_tokens;
        if (layer->impl_->ple_session != nullptr) {
            if (ple_owner != nullptr) return false;
            ple_owner = layer->impl_.get();
        }
    }

    /* Publish the sole device-backed history first.  Its geometry and stream
     * were validated by prepare_commit; the API performs one fixed-shape
     * enqueue and no synchronization. */
    if (ple_owner != nullptr) {
        std::string ignored;
        if (ple_owner->ple_session->commit_prefix(
                    staged_tokens, ple_owner->bound_stream, &ignored) !=
            ple_adapter_status::ok) {
            ple_owner->poisoned = true;
            return false;
        }
    }

    /* All remaining publications are allocation-free host metadata updates
     * over boundaries proven above.  A mismatch here is an internal invariant
     * violation, not a recoverable runtime branch. */
    for (std::size_t index = 0u; index < layer_count; ++index) {
        decoder_layer_session::impl &session = *layers[index]->impl_;
        bool published = false;
        if (session.owner->qsa != nullptr) {
            published = session.owner->qsa->commit_prefix(
                                &session.qsa_cache, staged_tokens) ==
                        QsaAdapterStatus::kOk;
        } else {
            published = gdn_transaction_commit_all(
                                &session.gdn_transaction,
                                session.bound_stream) ==
                        GdnTransactionStatus::kOk;
        }
        if (!published) {
            for (std::size_t poison = 0u; poison < layer_count; ++poison) {
                if (layers[poison] != nullptr && layers[poison]->impl_ != nullptr) {
                    layers[poison]->impl_->poisoned = true;
                }
            }
            return false;
        }
        session.committed_tokens += staged_tokens;
        session.staged_tokens = 0u;
        session.eagerly_validated_moe_rows = 0u;
        session.transaction_open = false;
        session.prepared_to_commit = false;
    }
    return true;
}

decoder_layer_status decoder_model_transaction::rollback_all(
        decoder_layer_session *const *layers,
        std::size_t layer_count,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (layers == nullptr || layer_count == 0u ||
        layer_count > kDecoderLayerCount) {
        return fail(decoder_layer_status::invalid_argument, error,
                    "invalid model rollback set");
    }
    bool success = cudaStreamSynchronize(stream) == cudaSuccess;
    std::string first_error;
    for (std::size_t offset = layer_count; offset > 0u; --offset) {
        decoder_layer_session *layer = layers[offset - 1u];
        if (layer == nullptr || layer->impl_ == nullptr ||
            !layer->impl_->ready || layer->impl_->bound_stream != stream) {
            success = false;
            if (first_error.empty()) first_error = "invalid layer in model rollback";
            continue;
        }
        if (!layer->impl_->transaction_open && !layer->impl_->early_ple_pending) {
            layer->impl_->prepared_to_commit = false;
            continue;
        }
        std::string detail;
        if (!layer->impl_->rollback_components(stream, false, &detail)) {
            success = false;
            if (first_error.empty()) first_error = detail;
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        success = false;
        if (first_error.empty()) first_error = "model rollback stream failed";
    }
    if (!success) {
        return fail(decoder_layer_status::transaction_error, error,
                    first_error.empty() ? "model rollback failed" : first_error);
    }
    return decoder_layer_status::ok;
}

}  // namespace axiom::qwen4exp
