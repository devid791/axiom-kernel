#include "axiom/qwen4exp/decoder_residency.hpp"

#include "axiom/qwen4exp/gdn_adapter.hpp"
#include "axiom/qwen4exp/expert_bridge.hpp"
#include "axiom/qwen4exp/mhc_adapter.hpp"
#include "axiom/qwen4exp/moe_adapter.hpp"
#include "axiom/qwen4exp/output_head.hpp"
#include "axiom/qwen4exp/ple_adapter.hpp"
#include "axiom/qwen4exp/qsa_adapter.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace q4 = axiom::qwen4exp;

namespace {

struct memory_snapshot {
    std::uint64_t free_bytes = 0u;
    std::uint64_t total_bytes = 0u;
};

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void require_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

memory_snapshot snapshot() {
    std::size_t free_bytes = 0u;
    std::size_t total_bytes = 0u;
    require_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    return {static_cast<std::uint64_t>(free_bytes),
            static_cast<std::uint64_t>(total_bytes)};
}

std::uint64_t consumed(const memory_snapshot &before,
                       const memory_snapshot &after) {
    return before.free_bytes > after.free_bytes
                   ? before.free_bytes - after.free_bytes
                   : 0u;
}

class stream_owner final {
public:
    stream_owner() {
        require_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");
    }
    ~stream_owner() {
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }
    stream_owner(const stream_owner &) = delete;
    stream_owner &operator=(const stream_owner &) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

template <typename Object, typename Loader>
std::uint64_t measure_component(const char *name,
                                cudaStream_t stream,
                                Loader &&loader) {
    require_cuda(cudaStreamSynchronize(stream), "component pre-sync");
    const memory_snapshot before = snapshot();
    std::unique_ptr<Object> object;
    std::string error;
    require(loader(&object, &error), std::string(name) + " load failed: " + error);
    require(object != nullptr, std::string(name) + " returned null object");
    require_cuda(cudaStreamSynchronize(stream), "component post-load sync");
    const memory_snapshot after = snapshot();
    const std::uint64_t delta = consumed(before, after);
    std::cout << "component=" << name << " cuda_bytes=" << delta << '\n';
    object.reset();
    require_cuda(cudaStreamSynchronize(stream), "component release sync");
    return delta;
}

void require_residency(q4::decoder_residency_status status,
                       const char *operation,
                       const std::string &error) {
    if (status == q4::decoder_residency_status::ok) return;
    fail(std::string(operation) + ": " +
         q4::decoder_residency_status_string(status) +
         (error.empty() ? std::string{} : ": " + error));
}

void run_admission_only(const q4::checkpoint_catalog &catalog,
                        const memory_snapshot &clean,
                        cudaStream_t stream) {
    q4::decoder_residency_config resident_config{};
    resident_config.clean_free_bytes = clean.free_bytes;

    std::unique_ptr<q4::decoder_residency_manager> resident_manager;
    std::string error;
    require_residency(q4::decoder_residency_manager::create(
                              catalog, resident_config, stream,
                              &resident_manager, &error),
                      "resident-all admission create", error);
    require(resident_manager && resident_manager->initialized(),
            "resident-all admission did not initialize");
    const q4::decoder_residency_metrics resident_plan =
            resident_manager->metrics();
    require(resident_plan.mode == q4::decoder_residency_mode::resident_all,
            "qualified checkpoint was not admitted as resident_all");
    require(resident_plan.resident_capacity == q4::kDecoderLayerCount &&
                    resident_plan.resident_layers == 0u &&
                    !resident_plan.all_48_admitted &&
                    !resident_plan.no_reload_decode_path,
            "admission planning incorrectly published runtime residency");
    for (q4::decoder_layer *layer : resident_manager->resident_layer_array()) {
        require(layer == nullptr,
                "admission-only resident plan allocated a decoder layer");
    }

    std::uint64_t largest_layer_estimate = 0u;
    for (const q4::decoder_layer_residency_record &record :
         resident_plan.layer) {
        require(record.state == q4::decoder_residency_state::nvme &&
                        record.load_count == 0u &&
                        record.eviction_count == 0u && record.pins == 0u,
                "admission-only layer record contains runtime state");
        largest_layer_estimate = std::max(
                largest_layer_estimate,
                record.checkpoint_immutable_bytes +
                        resident_config.adapter_allowance_bytes_per_layer);
    }

    constexpr std::uint64_t kAdmissionMargin = 1ull * 1024ull * 1024ull;
    q4::decoder_residency_config bounded_config = resident_config;
    bounded_config.safety_margin_bytes = kAdmissionMargin;
    bounded_config.hard_limit_bytes =
            resident_plan.external_bytes_at_create + largest_layer_estimate +
            2u * kAdmissionMargin;
    require(bounded_config.hard_limit_bytes <
                    resident_plan.predicted_peak_bytes,
            "synthetic bounded admission ceiling does not force bounded mode");

    std::unique_ptr<q4::decoder_residency_manager> bounded_manager;
    error.clear();
    require_residency(q4::decoder_residency_manager::create(
                              catalog, bounded_config, stream,
                              &bounded_manager, &error),
                      "bounded admission create", error);
    require(bounded_manager && bounded_manager->initialized(),
            "bounded admission did not initialize");
    const q4::decoder_residency_metrics bounded_plan =
            bounded_manager->metrics();
    require(bounded_plan.mode == q4::decoder_residency_mode::bounded_window &&
                    bounded_plan.resident_capacity > 0u &&
                    bounded_plan.resident_capacity < q4::kDecoderLayerCount &&
                    bounded_plan.resident_layers == 0u &&
                    !bounded_plan.all_48_admitted &&
                    !bounded_plan.no_reload_decode_path,
            "bounded admission did not remain diagnostic-only");

    q4::decoder_layer_lease lease;
    error.clear();
    require(bounded_manager->acquire(0u, stream, &lease, &error) ==
                            q4::decoder_residency_status::invalid_state &&
                    !lease.valid(),
            "bounded acquire did not fail closed");
    error.clear();
    require(bounded_manager->release_unpinned(stream, &error) ==
                    q4::decoder_residency_status::invalid_state,
            "bounded release_unpinned did not fail closed");
    const q4::decoder_residency_metrics bounded_after =
            bounded_manager->metrics();
    require(bounded_after.resident_layers == 0u &&
                    bounded_after.reload_count == 0u &&
                    bounded_after.eviction_count == 0u &&
                    bounded_after.hot_path_reload_count == 0u,
            "bounded fail-closed checks changed runtime residency");
    for (q4::decoder_layer *layer : bounded_manager->resident_layer_array()) {
        require(layer == nullptr,
                "bounded admission check allocated a decoder layer");
    }

    std::cout << "qwen4exp-decoder-residency-admission-test: PASS"
              << " resident_mode="
              << q4::decoder_residency_mode_string(resident_plan.mode)
              << " resident_predicted_bytes="
              << resident_plan.predicted_peak_bytes
              << " resident_usable_bytes=" << resident_plan.usable_limit_bytes
              << " bounded_mode="
              << q4::decoder_residency_mode_string(bounded_plan.mode)
              << " bounded_capacity=" << bounded_plan.resident_capacity
              << " load_all=not_run"
              << " decoder_layers_loaded=" << bounded_after.resident_layers
              << " reloads=" << bounded_after.reload_count
              << " evictions=" << bounded_after.eviction_count << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    const bool admission_only =
            argc == 3 && std::string_view(argv[2]) == "--admission-only";
    if (argc < 2 || argc > 3 || (argc == 3 && !admission_only)) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL_ROOT [--admission-only]\n";
        return 2;
    }

    try {
        int device = -1;
        cudaDeviceProp properties{};
        require_cuda(cudaGetDevice(&device), "cudaGetDevice");
        require_cuda(cudaGetDeviceProperties(&properties, device),
                     "cudaGetDeviceProperties");
        require(properties.major == 12,
                "decoder residency gate requires an SM120 GPU");
        require_cuda(cudaFree(nullptr), "CUDA context warmup");

        stream_owner stream;
        require_cuda(cudaStreamSynchronize(stream.get()), "clean baseline sync");
        const memory_snapshot clean = snapshot();

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        std::string error;
        require(q4::checkpoint_catalog::open(argv[1], &catalog, &error) &&
                        catalog != nullptr,
                "checkpoint catalog open failed: " + error);

        if (admission_only) {
            run_admission_only(*catalog, clean, stream.get());
            return 0;
        }

        q4::output_head_config head_config{};
        head_config.max_tokens = 1u;
        std::unique_ptr<q4::resident_output_head> output_head;
        const memory_snapshot before_head = snapshot();
        const q4::output_head_status head_status =
                q4::resident_output_head::load(*catalog, head_config,
                                               stream.get(), &output_head,
                                               &error);
        require(head_status == q4::output_head_status::ok && output_head &&
                        output_head->initialized(),
                "output head load failed: " + error);
        require_cuda(cudaStreamSynchronize(stream.get()), "output head sync");
        const memory_snapshot after_head = snapshot();
        const std::uint64_t output_head_cuda = consumed(before_head, after_head);
        std::cout << "component=output_head cuda_bytes=" << output_head_cuda
                  << " checkpoint_bytes="
                  << output_head->resident_weight_bytes() << '\n';

        q4::mhc_adapter_config mhc_config{};
        mhc_config.max_batch = 1u;
        const std::uint64_t mhc_attention =
                measure_component<q4::resident_mhc_adapter>(
                        "mhc_attention_layer0", stream.get(),
                        [&](auto *out, std::string *detail) {
                            return q4::resident_mhc_adapter::load(
                                           *catalog, 0u,
                                           q4::mhc_adapter_site::attention,
                                           mhc_config, stream.get(), out,
                                           detail) == q4::mhc_adapter_status::ok;
                        });
        const std::uint64_t mhc_mlp =
                measure_component<q4::resident_mhc_adapter>(
                        "mhc_mlp_layer0", stream.get(),
                        [&](auto *out, std::string *detail) {
                            return q4::resident_mhc_adapter::load(
                                           *catalog, 0u,
                                           q4::mhc_adapter_site::mlp,
                                           mhc_config, stream.get(), out,
                                           detail) == q4::mhc_adapter_status::ok;
                        });

        q4::GdnAdapterConfig gdn_config{};
        gdn_config.layer_index = 0u;
        gdn_config.max_batch = 1u;
        const std::uint64_t gdn = measure_component<q4::GdnLayerAdapter>(
                "gdn_layer0", stream.get(),
                [&](auto *out, std::string *detail) {
                    return q4::GdnLayerAdapter::load(
                                   *catalog, gdn_config, stream.get(), out,
                                   detail) == q4::GdnAdapterStatus::kOk;
                });

        q4::QsaAdapterConfig qsa_config{};
        qsa_config.layer_index = 3u;
        qsa_config.max_batch = 1u;
        qsa_config.max_context = q4::kDecoderMaxContext;
        const std::uint64_t qsa = measure_component<q4::QsaLayerAdapter>(
                "qsa_layer3", stream.get(),
                [&](auto *out, std::string *detail) {
                    return q4::QsaLayerAdapter::load(
                                   *catalog, qsa_config, stream.get(), out,
                                   detail) == q4::QsaAdapterStatus::kOk;
                });

        q4::moe_adapter_options moe_options{};
        moe_options.device = device;
        moe_options.layer = 0u;
        const std::uint64_t moe =
                measure_component<q4::routed_moe_layer_adapter>(
                        "moe_layer0", stream.get(),
                        [&](auto *out, std::string *detail) {
                            return q4::routed_moe_layer_adapter::load(
                                           argv[1], moe_options, stream.get(),
                                           out, detail) ==
                                   q4::moe_adapter_status::ok;
                        });

        q4::ple_adapter_config ple_config{};
        ple_config.layer_index = 1u;
        const std::uint64_t ple = measure_component<q4::ple_layer_adapter>(
                "ple_layer1", stream.get(),
                [&](auto *out, std::string *detail) {
                    return q4::ple_layer_adapter::load(
                                   *catalog, ple_config, stream.get(), out,
                                   detail) == q4::ple_adapter_status::ok;
                });

        q4::decoder_residency_config config{};
        config.clean_free_bytes = clean.free_bytes;
        std::shared_ptr<q4::expert_slot_arena> shared_expert_arena;
        std::shared_ptr<const q4::expert_checkpoint_catalog>
                shared_expert_catalog;
        error.clear();
        require(q4::expert_slot_arena::create(
                        device, q4::kMoeTopK, &shared_expert_arena, &error),
                "shared expert arena create failed: " + error);
        error.clear();
        require(q4::expert_checkpoint_catalog::open(
                        argv[1], q4::pinned_expert_checkpoint_identity(),
                        &shared_expert_catalog, &error),
                "shared expert catalog open failed: " + error);
        config.shared_expert_arena = shared_expert_arena;
        config.shared_expert_catalog = shared_expert_catalog;
        std::unique_ptr<q4::decoder_residency_manager> manager;
        error.clear();
        require_residency(q4::decoder_residency_manager::create(
                                  *catalog, config, stream.get(), &manager,
                                  &error),
                          "residency create", error);
        require(manager && manager->initialized(),
                "residency manager did not initialize");
        const q4::decoder_residency_metrics planned = manager->metrics();
        require(planned.mode == q4::decoder_residency_mode::resident_all,
                "real checkpoint unexpectedly requires decoder weight paging");
        require(planned.predicted_peak_bytes <= planned.usable_limit_bytes,
                "predicted resident-all set exceeds usable ceiling");

        error.clear();
        require_residency(manager->load_all(stream.get(), &error),
                          "load all 48 layers", error);
        require_cuda(cudaStreamSynchronize(stream.get()), "resident-all sync");
        const q4::decoder_residency_metrics resident = manager->metrics();
        require(manager->all_48_resident() && resident.resident_layers == 48u,
                "not all 48 decoder layers are resident");
        const q4::expert_slot_arena_metrics shared_arena_metrics =
                shared_expert_arena->metrics();
        require(shared_arena_metrics.attached_bridges == 48u &&
                        shared_arena_metrics.capacity_bytes == 27648160u,
                "48 decoder layers did not attach to one shared expert arena");
        require(resident.peak_observed_bytes <= resident.usable_limit_bytes &&
                        resident.peak_observed_bytes <
                                q4::kDecoderResidencyHardLimit,
                "measured decoder resident set crossed the 29 GiB contract");
        require(resident.reload_count == 0u &&
                        resident.hot_path_reload_count == 0u &&
                        resident.no_reload_decode_path,
                "resident-all path introduced a decoder reload");

        const auto layers = manager->resident_layer_array();
        for (std::size_t index = 0u; index < layers.size(); ++index) {
            require(layers[index] != nullptr && layers[index]->initialized() &&
                            layers[index]->config().layer_index == index &&
                            layers[index]->kind() ==
                                    q4::decoder_layer_kind_for(index),
                    "resident layer identity mismatch at " +
                            std::to_string(index));
        }
        for (const std::size_t index : std::array<std::size_t, 3>{0u, 1u, 3u}) {
            q4::decoder_layer_lease lease;
            error.clear();
            require_residency(manager->acquire(index, stream.get(), &lease,
                                               &error),
                              "resident hot-path acquire", error);
            require(lease.valid() && lease.get() == layers[index],
                    "resident hot-path lease changed layer identity");
        }
        const q4::decoder_residency_metrics after_acquire = manager->metrics();
        require(after_acquire.hot_path_reload_count == 0u &&
                        after_acquire.reload_count == 0u,
                "hot-path acquire reloaded immutable weights");

        std::size_t largest_layer = 0u;
        std::uint64_t largest_layer_bytes = 0u;
        for (const auto &record : resident.layer) {
            std::cout << "layer=" << record.layer_index
                      << " kind="
                      << (record.kind == q4::decoder_layer_kind::qsa ? "qsa"
                                                                    : "gdn")
                      << " checkpoint_bytes="
                      << record.checkpoint_immutable_bytes
                      << " observed_cuda_bytes=" << record.observed_cuda_bytes
                      << " adapter_overhead_bytes="
                      << record.adapter_overhead_bytes << '\n';
            if (record.observed_cuda_bytes > largest_layer_bytes) {
                largest_layer_bytes = record.observed_cuda_bytes;
                largest_layer = record.layer_index;
            }
        }

        std::cout << "qwen4exp-decoder-residency-test: PASS"
                  << " sm=" << properties.major << properties.minor
                  << " mode=" << q4::decoder_residency_mode_string(resident.mode)
                  << " layers=" << resident.resident_layers
                  << " before_bytes=" << planned.external_bytes_at_create
                  << " predicted_peak_bytes=" << planned.predicted_peak_bytes
                  << " after_peak_bytes=" << resident.peak_observed_bytes
                  << " usable_limit_bytes=" << resident.usable_limit_bytes
                  << " hard_limit_bytes=" << resident.hard_limit_bytes
                  << " margin_bytes="
                  << (resident.hard_limit_bytes - resident.peak_observed_bytes)
                  << " output_head_cuda=" << output_head_cuda
                  << " mhc_pair_layer0=" << (mhc_attention + mhc_mlp)
                  << " gdn_layer0=" << gdn
                  << " qsa_layer3=" << qsa
                  << " moe_layer0=" << moe
                  << " ple_layer1=" << ple
                  << " largest_layer=" << largest_layer
                  << " largest_layer_bytes=" << largest_layer_bytes
                  << " reloads=" << resident.reload_count
                  << " hot_path_reloads=" << resident.hot_path_reload_count
                  << " expert_arena_bytes="
                  << shared_arena_metrics.capacity_bytes
                  << " expert_arena_bridges="
                  << shared_arena_metrics.attached_bridges
                  << '\n';
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "qwen4exp-decoder-residency-test: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
