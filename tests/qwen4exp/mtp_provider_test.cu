#include "axiom/qwen4exp/mtp_provider.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void require_cuda(cudaError_t status, const std::string &where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(where + ": " + cudaGetErrorString(status));
    }
}

void require_status(q4::mtp_status actual,
                    q4::mtp_status expected,
                    const std::string &where,
                    const std::string &detail = {}) {
    if (actual != expected) {
        throw std::runtime_error(
                where + ": expected " + q4::mtp_status_string(expected) +
                ", got " + q4::mtp_status_string(actual) +
                (detail.empty() ? std::string{} : " (" + detail + ")"));
    }
}

template <typename T>
class device_buffer final {
public:
    explicit device_buffer(std::size_t elements) : elements_(elements) {
        require(elements != 0u, "zero-sized device allocation");
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                                elements * sizeof(T)),
                     "cudaMalloc");
    }
    ~device_buffer() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    T *get() noexcept { return pointer_; }
    const T *get() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return elements_; }

private:
    T *pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

class stream_guard final {
public:
    stream_guard() {
        require_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");
    }
    ~stream_guard() {
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }
    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

std::pair<std::vector<float>, std::vector<float>> make_rope(
        std::size_t positions) {
    constexpr std::size_t rotary = 64u;
    constexpr std::size_t half = rotary / 2u;
    constexpr double theta = 10000000.0;
    std::vector<float> cosine(positions * rotary);
    std::vector<float> sine(positions * rotary);
    for (std::size_t position = 0u; position < positions; ++position) {
        for (std::size_t dimension = 0u; dimension < half; ++dimension) {
            const double frequency = std::pow(
                    theta, -2.0 * static_cast<double>(dimension) /
                                   static_cast<double>(rotary));
            const float angle = static_cast<float>(
                    static_cast<double>(position) * frequency);
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            cosine[position * rotary + dimension] = c;
            cosine[position * rotary + dimension + half] = c;
            sine[position * rotary + dimension] = s;
            sine[position * rotary + dimension + half] = s;
        }
    }
    return {std::move(cosine), std::move(sine)};
}

float maximum_error(const std::vector<float> &left,
                    const std::vector<float> &right) {
    require(left.size() == right.size(), "comparison size mismatch");
    float maximum = 0.0F;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        require(std::isfinite(left[index]) && std::isfinite(right[index]),
                "non-finite MTP hidden output");
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

void validate_route(const q4::mtp_prediction &prediction) {
    std::array<bool, q4::kMtpExperts> seen{};
    float sum = 0.0F;
    for (std::size_t slot = 0u; slot < q4::kMtpTopK; ++slot) {
        require(prediction.experts[slot] < q4::kMtpExperts,
                "route contains an invalid expert id");
        require(!seen[prediction.experts[slot]],
                "route contains a duplicate expert id");
        seen[prediction.experts[slot]] = true;
        require(std::isfinite(prediction.router_weights[slot]) &&
                        prediction.router_weights[slot] > 0.0F,
                "route contains an invalid weight");
        sum += prediction.router_weights[slot];
    }
    require(std::abs(sum - 1.0F) <= 2.0e-6F,
            "top-10 router weights are not normalized");
}

}  // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: mtp-provider-test MODEL_ROOT");
        int device = -1;
        require_cuda(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        require_cuda(cudaGetDeviceProperties(&properties, device),
                     "cudaGetDeviceProperties");
        require(properties.major == 12 && properties.minor == 0,
                "real MTP gate requires SM120");

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        std::string error;
        require(q4::checkpoint_catalog::open(argv[1], &catalog, &error),
                error.empty() ? "checkpoint open failed" : error);
        require(catalog != nullptr, "checkpoint catalog is null");

        q4::mtp_contract_report contract{};
        require_status(q4::mtp_validate_checkpoint_contract(
                               *catalog, &contract, &error),
                       q4::mtp_status::ok, "MTP contract", error);
        require(contract.tensor_count == q4::kMtpTensorCount &&
                        contract.tensor_bytes == 5214301696ull &&
                        contract.exact_namespace && contract.all_bf16 &&
                        contract.single_full_attention_layer &&
                        contract.shared_lm_head,
                "MTP contract report differs from the pinned checkpoint");

        q4::mtp_provider_config invalid{};
        invalid.max_context = 0u;
        require_status(q4::mtp_validate_config(invalid),
                       q4::mtp_status::unsupported_config,
                       "invalid config gate");

        stream_guard stream;
        q4::mtp_provider_config config{};
        config.max_context = 8u;
        std::unique_ptr<q4::mtp_provider> provider;
        error.clear();
        require_status(q4::mtp_provider::load(
                               *catalog, config, stream.get(), &provider, &error),
                       q4::mtp_status::ok, "real MTP provider load", error);
        require(provider != nullptr && provider->initialized() &&
                        provider->resident_weight_bytes() == 1452535296ull,
                "resident MTP provider footprint mismatch");

        std::unique_ptr<q4::mtp_session> session;
        error.clear();
        require_status(provider->create_session(
                               8u, stream.get(), &session, &error),
                       q4::mtp_status::ok, "real MTP session create", error);
        require(session != nullptr && session->initialized(),
                "MTP session was not initialized");

        std::vector<float> embedding(q4::kMtpHidden);
        std::vector<float> hidden(q4::kMtpResidual);
        for (std::size_t index = 0u; index < embedding.size(); ++index) {
            embedding[index] =
                    0.018F * std::sin(static_cast<float>(index) * 0.013F) +
                    0.004F * std::cos(static_cast<float>(index) * 0.003F);
        }
        for (std::size_t index = 0u; index < hidden.size(); ++index) {
            hidden[index] =
                    0.015F * std::cos(static_cast<float>(index) * 0.007F) -
                    0.003F * std::sin(static_cast<float>(index) * 0.017F);
        }
        auto [cosine, sine] = make_rope(8u);

        device_buffer<float> device_embedding(embedding.size());
        device_buffer<float> device_hidden(hidden.size());
        device_buffer<float> device_output_a(hidden.size());
        device_buffer<float> device_output_b(hidden.size());
        device_buffer<float> device_cosine(cosine.size());
        device_buffer<float> device_sine(sine.size());
        require_cuda(cudaMemcpyAsync(
                             device_embedding.get(), embedding.data(),
                             embedding.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream.get()),
                     "copy embedding");
        require_cuda(cudaMemcpyAsync(
                             device_hidden.get(), hidden.data(),
                             hidden.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream.get()),
                     "copy hidden");
        require_cuda(cudaMemcpyAsync(
                             device_cosine.get(), cosine.data(),
                             cosine.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream.get()),
                     "copy cosine");
        require_cuda(cudaMemcpyAsync(
                             device_sine.get(), sine.data(),
                             sine.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream.get()),
                     "copy sine");
        require_cuda(cudaStreamSynchronize(stream.get()), "input upload");

        q4::mtp_prediction first{};
        error.clear();
        require_status(session->stage_token(
                               device_embedding.get(), device_hidden.get(),
                               device_cosine.get(), device_sine.get(), 8u,
                               device_output_a.get(), &first, stream.get(),
                               &error),
                       q4::mtp_status::ok, "first real MTP forward", error);
        validate_route(first);
        require(first.token_id < q4::kMtpVocab &&
                        std::isfinite(first.selected_logit) &&
                        first.nvme_bytes_read == 98304000ull &&
                        first.expert_cache_hits == 0u,
                "first MTP prediction/MLP paging result mismatch");
        const q4::mtp_session_state staged = session->state();
        require(staged.transaction_open && staged.staged &&
                        staged.committed_tokens == 0u && !staged.poisoned,
                "MTP stage was published before commit");

        std::vector<float> output_a(hidden.size());
        require_cuda(cudaMemcpy(output_a.data(), device_output_a.get(),
                                output_a.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy first MTP hidden");
        error.clear();
        require_status(session->rollback(stream.get(), &error),
                       q4::mtp_status::ok, "MTP rollback", error);
        require(session->state().committed_tokens == 0u &&
                        !session->state().transaction_open,
                "rollback published MTP cache state");

        q4::mtp_prediction second{};
        error.clear();
        require_status(session->stage_token(
                               device_embedding.get(), device_hidden.get(),
                               device_cosine.get(), device_sine.get(), 8u,
                               device_output_b.get(), &second, stream.get(),
                               &error),
                       q4::mtp_status::ok, "second real MTP forward", error);
        validate_route(second);
        require(second.token_id == first.token_id &&
                        second.experts == first.experts &&
                        second.router_weights == first.router_weights &&
                        second.nvme_bytes_read == 0u &&
                        second.expert_cache_hits == q4::kMtpTopK &&
                        std::abs(second.selected_logit - first.selected_logit) <=
                                1.0e-5F,
                "MTP deterministic replay/cache-hit gate failed");
        const std::uint32_t replay_cache_hits = second.expert_cache_hits;
        std::vector<float> output_b(hidden.size());
        require_cuda(cudaMemcpy(output_b.data(), device_output_b.get(),
                                output_b.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy second MTP hidden");
        const float hidden_error = maximum_error(output_a, output_b);
        require(hidden_error <= 1.0e-6F,
                "MTP hidden replay is not deterministic");

        error.clear();
        require_status(session->commit(stream.get(), &error),
                       q4::mtp_status::ok, "MTP commit", error);
        const q4::mtp_session_state committed = session->state();
        require(committed.committed_tokens == 1u &&
                        !committed.transaction_open && !committed.staged &&
                        !committed.poisoned,
                "MTP commit did not publish exactly one KV row");

        error.clear();
        require_status(session->stage_token(
                               embedding.data(), device_hidden.get(),
                               device_cosine.get(), device_sine.get(), 8u,
                               device_output_b.get(), &second, stream.get(),
                               &error),
                       q4::mtp_status::invalid_device_pointer,
                       "host pointer fail-closed gate", error);
        require(session->state().committed_tokens == 1u &&
                        !session->state().transaction_open,
                "rejected pointer changed MTP state");

        error.clear();
        require_status(session->reset(stream.get(), &error),
                       q4::mtp_status::ok, "MTP reset", error);
        require(session->state().committed_tokens == 0u &&
                        !session->state().poisoned,
                "MTP reset did not clear transactional metadata");

        std::cout << "PASS sm=" << properties.major << properties.minor
                  << " tensors=" << contract.tensor_count
                  << " tensor_bytes=" << contract.tensor_bytes
                  << " resident_bytes=" << provider->resident_weight_bytes()
                  << " token=" << first.token_id
                  << " logit=" << first.selected_logit
                  << " route0=" << first.experts[0]
                  << " nvme_first=" << first.nvme_bytes_read
                  << " cache_hits_replay=" << replay_cache_hits
                  << " hidden_abs=" << hidden_error
                  << " transaction=rollback+commit+fail_closed\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "FAIL " << exception.what() << '\n';
        return 1;
    }
}
