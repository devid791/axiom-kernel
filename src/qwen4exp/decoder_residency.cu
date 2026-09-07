#include "axiom/qwen4exp/decoder_residency.hpp"

#include "axiom/qwen4exp/provider_plan.hpp"
#include "axiom/qwen4exp_admission.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <utility>

namespace axiom::qwen4exp {
namespace {

void set_error(std::string *error, const std::string &message) noexcept {
    if (error == nullptr) return;
    try {
        *error = message;
    } catch (...) {
    }
}

decoder_residency_status fail(decoder_residency_status status,
                              std::string *error,
                              const std::string &message) noexcept {
    set_error(error, "qwen4_exp decoder residency: " + message);
    return status;
}

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t *out) noexcept {
    if (out == nullptr ||
        right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    *out = left + right;
    return true;
}

bool checked_multiply(std::uint64_t left,
                      std::uint64_t right,
                      std::uint64_t *out) noexcept {
    if (out == nullptr ||
        (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left)) {
        return false;
    }
    *out = left * right;
    return true;
}

decoder_residency_status map_layer_status(
        decoder_layer_status status) noexcept {
    switch (status) {
        case decoder_layer_status::ok:
            return decoder_residency_status::ok;
        case decoder_layer_status::invalid_argument:
        case decoder_layer_status::unsupported_layer:
        case decoder_layer_status::unsupported_config:
            return decoder_residency_status::invalid_argument;
        case decoder_layer_status::invalid_checkpoint:
            return decoder_residency_status::invalid_checkpoint;
        case decoder_layer_status::allocation_failure:
            return decoder_residency_status::allocation_failure;
        case decoder_layer_status::cuda_error:
            return decoder_residency_status::cuda_error;
        default:
            return decoder_residency_status::layer_load_failure;
    }
}

bool read_cuda_memory(std::uint64_t *free_bytes,
                      std::uint64_t *total_bytes) noexcept {
    if (free_bytes == nullptr || total_bytes == nullptr) return false;
    std::size_t free_value = 0u;
    std::size_t total_value = 0u;
    if (cudaMemGetInfo(&free_value, &total_value) != cudaSuccess) return false;
    *free_bytes = static_cast<std::uint64_t>(free_value);
    *total_bytes = static_cast<std::uint64_t>(total_value);
    return true;
}

std::uint64_t observed_from_clean(std::uint64_t clean_free,
                                  std::uint64_t current_free) noexcept {
    return clean_free > current_free ? clean_free - current_free : 0u;
}

decoder_layer_config make_layer_config(
        const decoder_residency_config &config,
        std::size_t layer_index) noexcept {
    decoder_layer_config result{};
    result.layer_index = layer_index;
    result.max_context = config.max_context;
    result.max_batch = kDecoderMaxSpeculativeTokens;
    result.upload_chunk_bytes = config.upload_chunk_bytes;
    result.blas_workspace_bytes = config.blas_workspace_bytes;
    result.moe_ram_capacity_bytes = config.moe_ram_capacity_bytes_per_layer;
    result.moe_prefetch_workers = config.moe_prefetch_workers_per_layer;
    result.moe_owned_direct_pread = config.moe_owned_direct_pread;
    result.shared_expert_arena = config.shared_expert_arena;
    result.shared_expert_catalog = config.shared_expert_catalog;
    result.enable_ple = config.enable_ple;
    return result;
}

}  // namespace

struct decoder_residency_manager::impl {
    struct slot {
        std::unique_ptr<decoder_layer> layer;
        std::uint64_t last_use = 0u;
    };

    const checkpoint_catalog *catalog = nullptr;
    decoder_residency_config config{};
    decoder_residency_metrics metrics{};
    std::array<slot, kDecoderLayerCount> slots{};
    int device = -1;
    std::uint64_t clock = 0u;
    mutable std::mutex mutex;
    bool ready = false;

    void update_observed_locked() noexcept {
        std::uint64_t free_bytes = 0u;
        std::uint64_t total_bytes = 0u;
        if (!read_cuda_memory(&free_bytes, &total_bytes)) return;
        metrics.cuda_total_bytes = total_bytes;
        metrics.current_observed_bytes =
                observed_from_clean(metrics.clean_free_bytes, free_bytes);
        metrics.peak_observed_bytes =
                std::max(metrics.peak_observed_bytes,
                         metrics.current_observed_bytes);
    }

    bool budget_can_accept_locked(std::uint64_t additional) const noexcept {
        return metrics.current_observed_bytes <= metrics.usable_limit_bytes &&
               additional <= metrics.usable_limit_bytes -
                                     metrics.current_observed_bytes;
    }

    decoder_residency_status load_layer_locked(
            std::size_t layer_index,
            cudaStream_t stream,
            bool hot_path,
            std::string *error) noexcept {
        if (layer_index >= kDecoderLayerCount || catalog == nullptr) {
            return fail(decoder_residency_status::invalid_argument, error,
                        "invalid layer index or checkpoint catalog");
        }
        if (metrics.mode != decoder_residency_mode::resident_all) {
            return fail(decoder_residency_status::invalid_state, error,
                        "bounded-window runtime is diagnostic-only until "
                        "decoder session ownership is decoupled");
        }
        auto &record = metrics.layer[layer_index];
        if (slots[layer_index].layer != nullptr) {
            slots[layer_index].last_use = ++clock;
            return decoder_residency_status::ok;
        }

        std::uint64_t estimate = 0u;
        if (!checked_add(record.checkpoint_immutable_bytes,
                         config.adapter_allowance_bytes_per_layer,
                         &estimate)) {
            return fail(decoder_residency_status::arithmetic_overflow, error,
                        "layer residency estimate overflowed");
        }
        update_observed_locked();
        if (!budget_can_accept_locked(estimate)) {
            std::ostringstream message;
            message << "layer " << layer_index << " needs " << estimate
                    << " bytes with " << metrics.current_observed_bytes
                    << " already observed; usable ceiling is "
                    << metrics.usable_limit_bytes;
            return fail(decoder_residency_status::budget_exceeded, error,
                        message.str());
        }

        std::uint64_t before_free = 0u;
        std::uint64_t total_bytes = 0u;
        if (!read_cuda_memory(&before_free, &total_bytes)) {
            return fail(decoder_residency_status::cuda_error, error,
                        "cudaMemGetInfo failed before layer load");
        }
        record.state = decoder_residency_state::loading;
        std::unique_ptr<decoder_layer> loaded;
        std::string detail;
        const decoder_layer_status layer_status = decoder_layer::load(
                *catalog, make_layer_config(config, layer_index), stream,
                &loaded, &detail);
        if (layer_status != decoder_layer_status::ok || loaded == nullptr ||
            !loaded->initialized()) {
            record.state = decoder_residency_state::nvme;
            return fail(map_layer_status(layer_status), error,
                        detail.empty()
                                ? std::string("layer load failed: ") +
                                          decoder_layer_status_string(layer_status)
                                : detail);
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            record.state = decoder_residency_state::nvme;
            return fail(decoder_residency_status::cuda_error, error,
                        "stream synchronization failed after layer load");
        }
        std::uint64_t after_free = 0u;
        if (!read_cuda_memory(&after_free, &total_bytes)) {
            record.state = decoder_residency_state::nvme;
            return fail(decoder_residency_status::cuda_error, error,
                        "cudaMemGetInfo failed after layer load");
        }
        const std::uint64_t delta = before_free > after_free
                                          ? before_free - after_free
                                          : 0u;
        const std::uint64_t current =
                observed_from_clean(metrics.clean_free_bytes, after_free);
        if (current > metrics.usable_limit_bytes ||
            current > metrics.hard_limit_bytes) {
            loaded.reset();
            (void)cudaStreamSynchronize(stream);
            record.state = decoder_residency_state::nvme;
            update_observed_locked();
            return fail(decoder_residency_status::budget_exceeded, error,
                        "measured CUDA allocation crossed the residency ceiling");
        }

        slots[layer_index].layer = std::move(loaded);
        slots[layer_index].last_use = ++clock;
        record.state = decoder_residency_state::resident;
        record.observed_cuda_bytes = delta;
        record.adapter_overhead_bytes =
                delta > record.checkpoint_immutable_bytes
                        ? delta - record.checkpoint_immutable_bytes
                        : 0u;
        ++record.load_count;
        if (record.load_count > 1u) ++metrics.reload_count;
        if (hot_path && record.load_count > 1u) ++metrics.hot_path_reload_count;
        ++metrics.resident_layers;
        metrics.current_observed_bytes = current;
        metrics.peak_observed_bytes =
                std::max(metrics.peak_observed_bytes, current);
        metrics.cuda_total_bytes = total_bytes;
        return decoder_residency_status::ok;
    }

    void unpin(std::size_t layer_index) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (layer_index >= kDecoderLayerCount) return;
        auto &record = metrics.layer[layer_index];
        if (record.pins != 0u) --record.pins;
    }
};

struct decoder_layer_lease::impl {
    decoder_residency_manager::impl *owner = nullptr;
    decoder_layer *layer = nullptr;
    std::size_t index = kDecoderLayerCount;

    ~impl() {
        if (owner != nullptr && index < kDecoderLayerCount) owner->unpin(index);
    }
};

const char *decoder_residency_status_string(
        decoder_residency_status status) noexcept {
    switch (status) {
        case decoder_residency_status::ok: return "ok";
        case decoder_residency_status::invalid_argument: return "invalid_argument";
        case decoder_residency_status::unsupported_device: return "unsupported_device";
        case decoder_residency_status::invalid_checkpoint: return "invalid_checkpoint";
        case decoder_residency_status::arithmetic_overflow: return "arithmetic_overflow";
        case decoder_residency_status::budget_exceeded: return "budget_exceeded";
        case decoder_residency_status::allocation_failure: return "allocation_failure";
        case decoder_residency_status::layer_load_failure: return "layer_load_failure";
        case decoder_residency_status::busy: return "busy";
        case decoder_residency_status::invalid_state: return "invalid_state";
        case decoder_residency_status::cuda_error: return "cuda_error";
    }
    return "unknown";
}

const char *decoder_residency_mode_string(
        decoder_residency_mode mode) noexcept {
    switch (mode) {
        case decoder_residency_mode::resident_all: return "resident_all";
        case decoder_residency_mode::bounded_window: return "bounded_window";
    }
    return "unknown";
}

decoder_layer_lease::decoder_layer_lease() noexcept = default;
decoder_layer_lease::~decoder_layer_lease() = default;
decoder_layer_lease::decoder_layer_lease(decoder_layer_lease &&) noexcept = default;
decoder_layer_lease &decoder_layer_lease::operator=(
        decoder_layer_lease &&) noexcept = default;
decoder_layer_lease::decoder_layer_lease(
        std::unique_ptr<impl> state) noexcept : impl_(std::move(state)) {}

decoder_layer *decoder_layer_lease::get() const noexcept {
    return impl_ != nullptr ? impl_->layer : nullptr;
}

decoder_layer &decoder_layer_lease::operator*() const noexcept {
    return *impl_->layer;
}

decoder_layer *decoder_layer_lease::operator->() const noexcept {
    return get();
}

std::size_t decoder_layer_lease::layer_index() const noexcept {
    return impl_ != nullptr ? impl_->index : kDecoderLayerCount;
}

bool decoder_layer_lease::valid() const noexcept {
    return impl_ != nullptr && impl_->owner != nullptr &&
           impl_->layer != nullptr && impl_->index < kDecoderLayerCount;
}

void decoder_layer_lease::reset() noexcept { impl_.reset(); }

decoder_residency_manager::decoder_residency_manager() = default;
decoder_residency_manager::~decoder_residency_manager() {
    if (impl_ == nullptr) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->device >= 0 && cudaSetDevice(impl_->device) == cudaSuccess) {
        (void)cudaDeviceSynchronize();
    }
    for (auto &slot : impl_->slots) slot.layer.reset();
}

decoder_residency_manager::decoder_residency_manager(
        decoder_residency_manager &&) noexcept = default;
decoder_residency_manager &decoder_residency_manager::operator=(
        decoder_residency_manager &&) noexcept = default;

decoder_residency_status decoder_residency_manager::create(
        const checkpoint_catalog &catalog,
        const decoder_residency_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<decoder_residency_manager> *out,
        std::string *error) noexcept {
    if (out == nullptr || config.hard_limit_bytes == 0u ||
        config.safety_margin_bytes >= config.hard_limit_bytes ||
        config.max_context == 0u || config.max_context > kDecoderMaxContext ||
        config.upload_chunk_bytes == 0u || config.blas_workspace_bytes == 0u ||
        config.moe_ram_capacity_bytes_per_layer == 0u ||
        config.moe_prefetch_workers_per_layer == 0u ||
        config.adapter_allowance_bytes_per_layer == 0u) {
        return fail(decoder_residency_status::invalid_argument, error,
                    "invalid output, budget, context or adapter configuration");
    }
    out->reset();
    if (error != nullptr) error->clear();
    try {
        int device = -1;
        cudaDeviceProp properties{};
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
            properties.major != 12) {
            return fail(decoder_residency_status::unsupported_device, error,
                        "qwen4_exp decoder residency requires an active SM120 GPU");
        }
        if (cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
            return fail(decoder_residency_status::cuda_error, error,
                        "initialization stream was not synchronized");
        }
        std::uint64_t free_bytes = 0u;
        std::uint64_t total_bytes = 0u;
        if (!read_cuda_memory(&free_bytes, &total_bytes)) {
            return fail(decoder_residency_status::cuda_error, error,
                        "cudaMemGetInfo failed during residency creation");
        }
        const std::uint64_t clean_free = config.clean_free_bytes == 0u
                                                ? free_bytes
                                                : config.clean_free_bytes;
        if (clean_free > total_bytes || clean_free < free_bytes) {
            return fail(decoder_residency_status::invalid_argument, error,
                        "clean CUDA baseline is outside current device memory");
        }

        provider_plan plan{};
        std::string detail;
        const provider_plan_status plan_status = build_provider_plan(
                &catalog, kWeightManifestSha256, &plan, &detail);
        if (plan_status != provider_plan_status::ok) {
            return fail(decoder_residency_status::invalid_checkpoint, error,
                        detail.empty() ? "checkpoint residency plan failed"
                                       : detail);
        }

        auto result = std::make_unique<decoder_residency_manager>();
        result->impl_ = std::make_unique<impl>();
        impl &state = *result->impl_;
        state.catalog = &catalog;
        state.config = config;
        state.device = device;
        state.metrics.cuda_total_bytes = total_bytes;
        state.metrics.clean_free_bytes = clean_free;
        state.metrics.external_bytes_at_create =
                observed_from_clean(clean_free, free_bytes);
        state.metrics.hard_limit_bytes = config.hard_limit_bytes;
        state.metrics.usable_limit_bytes =
                config.hard_limit_bytes - config.safety_margin_bytes;

        std::uint64_t immutable_bytes = 0u;
        for (std::size_t index = 0u; index < kDecoderLayerCount; ++index) {
            auto &record = state.metrics.layer[index];
            record.layer_index = index;
            record.kind = decoder_layer_kind_for(index);
            record.checkpoint_immutable_bytes = plan.layer[index].resident_bytes;
            if (!checked_add(immutable_bytes, record.checkpoint_immutable_bytes,
                             &immutable_bytes)) {
                return fail(decoder_residency_status::arithmetic_overflow, error,
                            "checkpoint immutable-byte sum overflowed");
            }
        }
        state.metrics.checkpoint_immutable_bytes = immutable_bytes;
        std::uint64_t adapter_allowance = 0u;
        std::uint64_t predicted = 0u;
        if (!checked_multiply(config.adapter_allowance_bytes_per_layer,
                              kDecoderLayerCount, &adapter_allowance) ||
            !checked_add(state.metrics.external_bytes_at_create,
                         immutable_bytes, &predicted) ||
            !checked_add(predicted, adapter_allowance, &predicted)) {
            return fail(decoder_residency_status::arithmetic_overflow, error,
                        "predicted residency peak overflowed");
        }
        state.metrics.predicted_peak_bytes = predicted;
        state.metrics.mode =
                predicted <= state.metrics.usable_limit_bytes
                        ? decoder_residency_mode::resident_all
                        : decoder_residency_mode::bounded_window;

        if (state.metrics.mode == decoder_residency_mode::resident_all) {
            state.metrics.resident_capacity = kDecoderLayerCount;
        } else {
            std::uint64_t largest = 0u;
            for (const auto &record : state.metrics.layer) {
                std::uint64_t estimate = 0u;
                if (!checked_add(record.checkpoint_immutable_bytes,
                                 config.adapter_allowance_bytes_per_layer,
                                 &estimate)) {
                    return fail(decoder_residency_status::arithmetic_overflow,
                                error, "layer capacity estimate overflowed");
                }
                largest = std::max(largest, estimate);
            }
            const std::uint64_t available =
                    state.metrics.external_bytes_at_create <
                            state.metrics.usable_limit_bytes
                            ? state.metrics.usable_limit_bytes -
                                      state.metrics.external_bytes_at_create
                            : 0u;
            const std::uint64_t capacity = largest == 0u ? 0u : available / largest;
            if (capacity == 0u) {
                return fail(decoder_residency_status::budget_exceeded, error,
                            "budget cannot admit even one complete decoder layer");
            }
            state.metrics.resident_capacity = static_cast<std::size_t>(
                    std::min<std::uint64_t>(capacity, kDecoderLayerCount));
        }
        state.metrics.current_observed_bytes =
                state.metrics.external_bytes_at_create;
        state.metrics.peak_observed_bytes =
                state.metrics.external_bytes_at_create;
        /* Admission planning never publishes runtime residency.  This bit is
         * set only after load_all() has retained all 48 layer objects; in
         * bounded-window mode load_all() fails closed before the first load. */
        state.metrics.all_48_admitted = false;
        state.ready = true;
        *out = std::move(result);
        return decoder_residency_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(decoder_residency_status::allocation_failure, error,
                    "host allocation failed during residency creation");
    } catch (const std::exception &exception) {
        return fail(decoder_residency_status::invalid_state, error,
                    std::string("residency creation exception: ") +
                            exception.what());
    } catch (...) {
        return fail(decoder_residency_status::invalid_state, error,
                    "unknown residency creation exception");
    }
}

decoder_residency_status decoder_residency_manager::load_all(
        cudaStream_t initialization_stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready) {
        return fail(decoder_residency_status::invalid_state, error,
                    "residency manager is not initialized");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->metrics.mode != decoder_residency_mode::resident_all) {
        return fail(decoder_residency_status::budget_exceeded, error,
                    "predicted resident set requires bounded-window mode");
    }
    for (std::size_t index = 0u; index < kDecoderLayerCount; ++index) {
        const decoder_residency_status status =
                impl_->load_layer_locked(index, initialization_stream, false,
                                         error);
        if (status != decoder_residency_status::ok) return status;
    }
    impl_->metrics.all_48_admitted = impl_->metrics.resident_layers ==
                                     kDecoderLayerCount;
    impl_->metrics.no_reload_decode_path =
            impl_->metrics.all_48_admitted &&
            impl_->metrics.reload_count == 0u;
    return impl_->metrics.all_48_admitted
                   ? decoder_residency_status::ok
                   : fail(decoder_residency_status::invalid_state, error,
                          "load_all returned without 48 resident layers");
}

decoder_residency_status decoder_residency_manager::acquire(
        std::size_t layer_index,
        cudaStream_t stream,
        decoder_layer_lease *lease,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || lease == nullptr ||
        layer_index >= kDecoderLayerCount) {
        return fail(decoder_residency_status::invalid_argument, error,
                    "invalid manager, layer index or lease output");
    }
    lease->reset();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->metrics.mode != decoder_residency_mode::resident_all ||
        !impl_->metrics.no_reload_decode_path ||
        impl_->metrics.resident_layers != kDecoderLayerCount) {
        return fail(decoder_residency_status::invalid_state, error,
                    "acquire is enabled only for a sealed 48-layer "
                    "resident-all model; bounded paging is diagnostic-only");
    }
    const bool was_resident = impl_->slots[layer_index].layer != nullptr;
    const decoder_residency_status status = impl_->load_layer_locked(
            layer_index, stream, true, error);
    if (status != decoder_residency_status::ok) return status;
    if (!was_resident) {
        return fail(decoder_residency_status::invalid_state, error,
                    "resident-all decode attempted an unexpected reload");
    }
    try {
        auto lease_state = std::make_unique<decoder_layer_lease::impl>();
        lease_state->owner = impl_.get();
        lease_state->layer = impl_->slots[layer_index].layer.get();
        lease_state->index = layer_index;
        ++impl_->metrics.layer[layer_index].pins;
        impl_->slots[layer_index].last_use = ++impl_->clock;
        *lease = decoder_layer_lease(std::move(lease_state));
        return decoder_residency_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(decoder_residency_status::allocation_failure, error,
                    "lease allocation failed");
    }
}

decoder_residency_status decoder_residency_manager::release_unpinned(
        cudaStream_t stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready) {
        return fail(decoder_residency_status::invalid_state, error,
                    "residency manager is not initialized");
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->metrics.mode == decoder_residency_mode::resident_all) {
        return decoder_residency_status::ok;
    }
    (void)stream;
    return fail(decoder_residency_status::invalid_state, error,
                "bounded-window runtime is diagnostic-only; no layer was "
                "loaded or evicted");
}

decoder_layer *decoder_residency_manager::resident_layer(
        std::size_t layer_index) const noexcept {
    if (impl_ == nullptr || layer_index >= kDecoderLayerCount) return nullptr;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->slots[layer_index].layer.get();
}

std::array<decoder_layer *, kDecoderLayerCount>
decoder_residency_manager::resident_layer_array() const noexcept {
    std::array<decoder_layer *, kDecoderLayerCount> result{};
    if (impl_ == nullptr) return result;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index] = impl_->slots[index].layer.get();
    }
    return result;
}

decoder_residency_metrics decoder_residency_manager::metrics() const noexcept {
    if (impl_ == nullptr) return {};
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->metrics;
}

decoder_residency_mode decoder_residency_manager::mode() const noexcept {
    if (impl_ == nullptr) return decoder_residency_mode::resident_all;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->metrics.mode;
}

bool decoder_residency_manager::initialized() const noexcept {
    return impl_ != nullptr && impl_->ready;
}

bool decoder_residency_manager::all_48_resident() const noexcept {
    if (impl_ == nullptr) return false;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->metrics.resident_layers == kDecoderLayerCount;
}

}  // namespace axiom::qwen4exp
