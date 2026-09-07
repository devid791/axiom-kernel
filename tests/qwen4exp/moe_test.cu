#include "axiom/qwen4exp/moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void require_cuda(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

template <typename T>
class device_buffer {
public:
    explicit device_buffer(std::size_t count) : count_(count) {
        require(count != 0u, "zero-sized device buffer");
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)),
                     "cudaMalloc");
    }
    ~device_buffer() { if (data_ != nullptr) (void)cudaFree(data_); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    T *data() noexcept { return data_; }
    const T *data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    void upload(const std::vector<T> &values) {
        require(values.size() == count_, "device upload size mismatch");
        require_cuda(cudaMemcpy(data_, values.data(), count_ * sizeof(T),
                                cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    void download(std::vector<T> *values) const {
        require(values != nullptr && values->size() == count_,
                "device download size mismatch");
        require_cuda(cudaMemcpy(values->data(), data_, count_ * sizeof(T),
                                cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }
private:
    T *data_ = nullptr;
    std::size_t count_ = 0;
};

// Frozen single-thread GPU oracle from 62d85d6: keep the arithmetic and
// selection order independent of the optimized kernel. CPU std::exp is not a
// bitwise oracle. Both CUDA translation units use NVCCFLAGS_PRECISE.
__global__ void scalar_router_topk_reference(const float *logits,
                                   std::uint32_t experts,
                                   std::uint32_t top_k,
                                   bool normalize_topk,
                                   std::uint32_t *indices,
                                   float *weights,
                                   std::uint32_t *status) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    *status = static_cast<std::uint32_t>(q4::moe_status::ok);
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        if (!isfinite(logits[expert])) {
            *status = static_cast<std::uint32_t>(q4::moe_status::non_finite);
            for (std::uint32_t slot = 0; slot < top_k; ++slot) {
                indices[slot] = UINT32_MAX;
                weights[slot] = 0.0f;
            }
            return;
        }
    }

    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        float best = -INFINITY;
        std::uint32_t best_index = UINT32_MAX;
        for (std::uint32_t expert = 0; expert < experts; ++expert) {
            bool used = false;
            for (std::uint32_t previous = 0; previous < slot; ++previous) {
                used = used || indices[previous] == expert;
            }
            const float value = logits[expert];
            if (!used && (value > best ||
                          (value == best && expert < best_index))) {
                best = value;
                best_index = expert;
            }
        }
        indices[slot] = best_index;
    }

    const float maximum = logits[indices[0]];
    float selected_sum = 0.0f;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        weights[slot] = expf(logits[indices[slot]] - maximum);
        selected_sum += weights[slot];
    }
    if (!isfinite(selected_sum) || selected_sum <= 0.0f) {
        *status = static_cast<std::uint32_t>(q4::moe_status::non_finite);
        return;
    }
    if (normalize_topk) {
        for (std::uint32_t slot = 0; slot < top_k; ++slot) {
            weights[slot] /= selected_sum;
        }
    } else {
        float global_sum = 0.0f;
        for (std::uint32_t expert = 0; expert < experts; ++expert) {
            global_sum += expf(logits[expert] - maximum);
        }
        if (!isfinite(global_sum) || global_sum <= 0.0f) {
            *status = static_cast<std::uint32_t>(q4::moe_status::non_finite);
            return;
        }
        for (std::uint32_t slot = 0; slot < top_k; ++slot) {
            weights[slot] /= global_sum;
        }
    }
}

float e2m1(std::uint8_t nibble) {
    static constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                        2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = values[nibble & 7u];
    return (nibble & 8u) != 0u ? -magnitude : magnitude;
}

float e4m3(std::uint8_t encoded) {
    const bool negative = (encoded & 0x80u) != 0u;
    const unsigned exponent = (encoded >> 3u) & 15u;
    const unsigned mantissa = encoded & 7u;
    const float value = exponent == 0u
            ? std::ldexp(static_cast<float>(mantissa), -9)
            : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                         static_cast<int>(exponent) - 7);
    return negative ? -value : value;
}

std::uint8_t deterministic_nibble(std::size_t seed, std::size_t index) {
    static constexpr std::uint8_t codes[] = {1u, 2u, 3u, 4u, 5u, 9u, 10u, 12u};
    return codes[(seed + index * 5u) % (sizeof(codes) / sizeof(codes[0]))];
}

void fill_weight(std::vector<std::uint8_t> *weight,
                 std::uint32_t slots,
                 std::uint32_t rows,
                 std::uint32_t cols,
                 std::size_t seed) {
    require(weight != nullptr && weight->size() ==
                    static_cast<std::size_t>(slots) * rows * (cols / 2u),
            "weight geometry mismatch");
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t column = 0; column < cols; column += 2u) {
                const std::size_t logical =
                        (static_cast<std::size_t>(slot) * rows + row) * cols + column;
                const std::uint8_t low = deterministic_nibble(seed, logical);
                const std::uint8_t high = deterministic_nibble(seed + 3u, logical + 1u);
                (*weight)[logical / 2u] = static_cast<std::uint8_t>(low | (high << 4u));
            }
        }
    }
}

float matvec_element(const std::vector<std::uint8_t> &weight,
                     const std::vector<std::uint8_t> &scales,
                     const std::vector<float> &globals,
                     std::uint32_t slots,
                     std::uint32_t rows,
                     std::uint32_t cols,
                     std::uint32_t slot,
                     std::uint32_t row,
                     const float *input) {
    (void)slots;
    const std::size_t weight_base =
            (static_cast<std::size_t>(slot) * rows + row) * (cols / 2u);
    const std::size_t scale_base =
            (static_cast<std::size_t>(slot) * rows + row) * (cols / 16u);
    float result = 0.0f;
    for (std::uint32_t column = 0; column < cols; ++column) {
        const std::uint8_t packed = weight[weight_base + column / 2u];
        const std::uint8_t nibble = (column & 1u) == 0u
                ? packed & 15u : packed >> 4u;
        result += e2m1(nibble) * e4m3(scales[scale_base + column / 16u]) *
                  globals[slot] * input[column];
    }
    return result;
}

void test_router() {
    const q4::moe_config config = q4::moe_qwen4_exp_config();
    require(q4::moe_is_qwen4_exp_contract(config), "exact MoE contract rejected");
    std::vector<float> logits(config.experts);
    for (std::uint32_t expert = 0; expert < config.experts; ++expert) {
        logits[expert] = std::sin(static_cast<float>(expert) * 0.173f) * 3.0f +
                         static_cast<float>(expert % 17u) * 0.01f;
    }
    std::vector<std::uint32_t> host_indices(config.top_k);
    std::vector<float> host_weights(config.top_k);
    require(q4::moe_router_topk_host(config, logits.data(), host_indices.data(),
                                    host_weights.data()) == q4::moe_status::ok,
            "host router failed");

    device_buffer<float> device_logits(logits.size());
    device_buffer<std::uint32_t> device_indices(config.top_k);
    device_buffer<float> device_weights(config.top_k);
    device_buffer<std::uint32_t> device_status(1u);
    device_logits.upload(logits);
    require(q4::moe_router_topk_cuda(
                    config, device_logits.data(), device_indices.data(),
                    device_weights.data(), device_status.data()) == q4::moe_status::ok,
            "CUDA router launch failed");
    require_cuda(cudaDeviceSynchronize(), "router synchronize");
    std::vector<std::uint32_t> actual_indices(config.top_k);
    std::vector<float> actual_weights(config.top_k);
    std::vector<std::uint32_t> actual_status(1u);
    device_indices.download(&actual_indices);
    device_weights.download(&actual_weights);
    device_status.download(&actual_status);
    require(actual_status[0] == static_cast<std::uint32_t>(q4::moe_status::ok),
            "CUDA router device status failed");
    require(actual_indices == host_indices, "CUDA router selected different experts");
    float weight_sum = 0.0f;
    for (std::size_t slot = 0; slot < host_weights.size(); ++slot) {
        require(std::abs(actual_weights[slot] - host_weights[slot]) < 2.0e-7f,
                "CUDA router weight mismatch");
        weight_sum += actual_weights[slot];
    }
    require(std::abs(weight_sum - 1.0f) < 2.0e-6f,
            "normalized router weights do not sum to one");

    logits[11] = std::numeric_limits<float>::quiet_NaN();
    require(q4::moe_router_topk_host(config, logits.data(), host_indices.data(),
                                    host_weights.data()) == q4::moe_status::non_finite,
            "host router accepted NaN");
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "float must be 32 bits");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

class router_comparison {
public:
    router_comparison(std::uint32_t experts, std::uint32_t top_k)
        : logits_(experts), actual_indices_(top_k), expected_indices_(top_k),
          actual_weights_(top_k), expected_weights_(top_k),
          actual_status_(1u), expected_status_(1u) {
        config_.experts = experts;
        config_.top_k = top_k;
        require(q4::moe_validate_config(config_) == q4::moe_status::ok,
                "bitwise router config rejected");
    }

    void check(const std::vector<float> &logits, const char *label) {
        logits_.upload(logits);
        const bool finite = std::all_of(logits.begin(), logits.end(),
                                       [](float x) { return std::isfinite(x); });
        for (bool normalize : {false, true}) {
            config_.normalize_topk = normalize;
            // Poison both outputs differently on every call, including status;
            // untouched/reused slots must not accidentally compare equal.
            require_cuda(cudaMemset(actual_indices_.data(), 0xa5,
                                    config_.top_k * sizeof(std::uint32_t)), "poison indices");
            require_cuda(cudaMemset(expected_indices_.data(), 0x5a,
                                    config_.top_k * sizeof(std::uint32_t)), "poison reference indices");
            require_cuda(cudaMemset(actual_weights_.data(), 0xa5,
                                    config_.top_k * sizeof(float)), "poison weights");
            require_cuda(cudaMemset(expected_weights_.data(), 0x5a,
                                    config_.top_k * sizeof(float)), "poison reference weights");
            require_cuda(cudaMemset(actual_status_.data(), 0xa5, sizeof(std::uint32_t)),
                         "poison status");
            require_cuda(cudaMemset(expected_status_.data(), 0x5a, sizeof(std::uint32_t)),
                         "poison reference status");
            scalar_router_topk_reference<<<1, 1>>>(
                    logits_.data(), config_.experts, config_.top_k, normalize,
                    expected_indices_.data(), expected_weights_.data(), expected_status_.data());
            require_cuda(cudaGetLastError(), "scalar reference launch");
            launch_parallel();
            require_cuda(cudaDeviceSynchronize(), "bitwise router synchronize");
            std::vector<std::uint32_t> actual_indices(config_.top_k), expected_indices(config_.top_k);
            std::vector<float> actual_weights(config_.top_k), expected_weights(config_.top_k);
            std::vector<std::uint32_t> actual_status(1u), expected_status(1u);
            actual_indices_.download(&actual_indices);
            expected_indices_.download(&expected_indices);
            actual_weights_.download(&actual_weights);
            expected_weights_.download(&expected_weights);
            actual_status_.download(&actual_status);
            expected_status_.download(&expected_status);
            const std::string context = std::string(label) + " experts=" +
                    std::to_string(config_.experts) + " top_k=" +
                    std::to_string(config_.top_k) + " normalize=" + std::to_string(normalize);
            require(actual_status == expected_status, context + ": status mismatch");
            require(actual_status[0] == static_cast<std::uint32_t>(
                            finite ? q4::moe_status::ok : q4::moe_status::non_finite),
                    context + ": unexpected status");
            require(actual_indices == expected_indices, context + ": index mismatch");
            for (std::uint32_t slot = 0; slot < config_.top_k; ++slot) {
                require(float_bits(actual_weights[slot]) == float_bits(expected_weights[slot]),
                        context + ": weight bits mismatch at slot=" + std::to_string(slot));
                if (!finite) {
                    require(actual_indices[slot] == UINT32_MAX &&
                                    float_bits(actual_weights[slot]) == 0u,
                            context + ": non-finite output was not cleared");
                } else if (slot != 0u) {
                    const auto previous = actual_indices[slot - 1u];
                    const auto current = actual_indices[slot];
                    require(logits[previous] > logits[current] ||
                                    (logits[previous] == logits[current] && previous < current),
                            context + ": descending value/lower-index tie order violated");
                }
            }
        }
    }

    // Opt-in, bounded kernel timing on one fixed router input (no model run).
    // CUDA events exclude uploads, comparisons, and allocation. No speed gate.
    void time_one_input(const std::vector<float> &logits) {
        check(logits, "timing input");
        config_.normalize_topk = true;
        auto launch = [&](bool scalar) {
            if (scalar) {
                scalar_router_topk_reference<<<1, 1>>>(
                        logits_.data(), config_.experts, config_.top_k, true,
                        expected_indices_.data(), expected_weights_.data(), expected_status_.data());
                require_cuda(cudaGetLastError(), "timed scalar launch");
            } else {
                launch_parallel();
            }
        };
        struct event {
            cudaEvent_t value = nullptr;
            event() { require_cuda(cudaEventCreate(&value), "cudaEventCreate"); }
            ~event() { if (value != nullptr) (void)cudaEventDestroy(value); }
        } start, stop;
        constexpr unsigned iterations = 100u;
        float times[2]{};
        for (unsigned variant = 0; variant < 2u; ++variant) {
            const bool scalar = variant == 0u;
            for (unsigned warmup = 0; warmup < 5u; ++warmup) launch(scalar);
            require_cuda(cudaEventRecord(start.value), "timing start");
            for (unsigned iteration = 0; iteration < iterations; ++iteration) launch(scalar);
            require_cuda(cudaEventRecord(stop.value), "timing stop");
            require_cuda(cudaEventSynchronize(stop.value), "timing synchronize");
            require_cuda(cudaEventElapsedTime(&times[variant], start.value, stop.value),
                         "cudaEventElapsedTime");
        }
        std::printf("qwen4exp-router timing experts=%u top_k=%u normalize=1 iterations=%u "
                    "scalar_us=%.3f parallel_us=%.3f\n", config_.experts, config_.top_k,
                    iterations, times[0] * 1000.0f / iterations, times[1] * 1000.0f / iterations);
    }

private:
    void launch_parallel() {
        require(q4::moe_router_topk_cuda(config_, logits_.data(), actual_indices_.data(),
                        actual_weights_.data(), actual_status_.data()) == q4::moe_status::ok,
                "parallel router launch failed");
    }
    q4::moe_config config_;
    device_buffer<float> logits_;
    device_buffer<std::uint32_t> actual_indices_, expected_indices_;
    device_buffer<float> actual_weights_, expected_weights_;
    device_buffer<std::uint32_t> actual_status_, expected_status_;
};

void test_router_bitwise() {
    std::mt19937 random(0x514f504bu);
    auto fill_random = [&](std::vector<float> &logits) {
        for (float &value : logits) {
            value = static_cast<float>(static_cast<int>(random() % 65537u) - 32768) / 1024.0f;
        }
    };
    // Warp/block boundaries, partial blocks, default geometry, and both limits.
    for (const std::uint32_t experts : {1u, 2u, 3u, 10u, 31u, 32u, 33u, 63u, 64u, 65u,
            127u, 128u, 129u, 255u, 256u, 257u, 511u, 512u, 513u, 1023u, 1024u,
            1025u, 2047u, 2048u, 2049u, 4095u, 4096u}) {
        std::vector<std::uint32_t> topks{1u, std::min(10u, experts), std::min(64u, experts)};
        std::sort(topks.begin(), topks.end());
        topks.erase(std::unique(topks.begin(), topks.end()), topks.end());
        for (const auto top_k : topks) {
            router_comparison comparison(experts, top_k);
            std::vector<float> logits(experts);
            fill_random(logits);
            comparison.check(logits, "seeded random");
            // Reuse buffers and move the winner across lanes/warps/strides.
            std::reverse(logits.begin(), logits.end());
            comparison.check(logits, "reversed random");
            std::fill(logits.begin(), logits.end(), 3.0f);
            comparison.check(logits, "all equal");
        }
    }
    // Every supported top_k, with both all-selected and random expert counts.
    for (std::uint32_t top_k = 1u; top_k <= 64u; ++top_k) {
        const std::uint32_t experts = top_k + random() % (4097u - top_k);
        std::vector<float> logits(experts);
        fill_random(logits);
        router_comparison(experts, top_k).check(logits, "top_k sweep");
        logits.resize(top_k);
        router_comparison(top_k, top_k).check(logits, "all selected");
    }
    for (const std::uint32_t experts : {1u, 33u, 257u, 512u, 4096u}) {
        router_comparison comparison(experts, std::min(64u, experts));
        std::vector<float> logits(experts);
        for (std::uint32_t i = 0; i < experts; ++i) logits[i] = (i & 1u) ? 0.0f : -0.0f;
        comparison.check(logits, "signed zero ties");
        for (std::uint32_t i = 0; i < experts; ++i) logits[i] = static_cast<float>(i % 7u) - 3.0f;
        comparison.check(logits, "ties across lanes and strides");
        for (std::uint32_t i = 0; i < experts; ++i) logits[i] = static_cast<float>(i);
        comparison.check(logits, "ascending");
        std::reverse(logits.begin(), logits.end());
        comparison.check(logits, "descending");
        std::fill(logits.begin(), logits.end(), -std::numeric_limits<float>::max());
        comparison.check(logits, "lowest finite ties");
        logits.back() = std::numeric_limits<float>::max();
        comparison.check(logits, "finite subtraction overflow and exp underflow");
        std::fill(logits.begin(), logits.end(), std::numeric_limits<float>::max());
        comparison.check(logits, "highest finite ties");
        const float tiny = std::numeric_limits<float>::denorm_min();
        for (std::uint32_t i = 0; i < experts; ++i) {
            logits[i] = static_cast<float>(static_cast<int>(i % 5u) - 2) * tiny;
        }
        comparison.check(logits, "subnormal logits");
        for (std::uint32_t i = 0; i < experts; ++i) {
            logits[i] = (i & 1u) ? 1.0f : std::nextafter(1.0f, 2.0f);
        }
        comparison.check(logits, "adjacent floats");
        // Raw random finite IEEE values cover exponents and mantissas beyond
        // the moderate-range random inputs; deliberately retain subnormals.
        for (float &value : logits) {
            std::uint32_t bits = random();
            if ((bits & 0x7f800000u) == 0x7f800000u) bits ^= 0x00800000u;
            std::memcpy(&value, &bits, sizeof(value));
        }
        comparison.check(logits, "random finite float bits");
        for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                   std::numeric_limits<float>::infinity(),
                                   -std::numeric_limits<float>::infinity()}) {
            for (const std::uint32_t position : {0u, std::min(31u, experts - 1u),
                    std::min(32u, experts - 1u), std::min(255u, experts - 1u),
                    std::min(256u, experts - 1u), experts - 1u}) {
                std::fill(logits.begin(), logits.end(), 0.0f);
                logits[position] = invalid;
                comparison.check(logits, "non-finite at boundary");
            }
        }
        std::fill(logits.begin(), logits.end(), std::numeric_limits<float>::quiet_NaN());
        comparison.check(logits, "all NaN");
        fill_random(logits);
        comparison.check(logits, "finite after rejection");
    }
    std::puts("qwen4exp-router bitwise scalar GPU comparisons: pass");
}

void optional_router_timing() {
    const char *enabled = std::getenv("QWEN4EXP_ROUTER_TIMING");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0) return;
    const auto config = q4::moe_qwen4_exp_config();
    std::vector<float> logits(config.experts);
    for (std::uint32_t i = 0; i < config.experts; ++i) {
        logits[i] = std::sin(static_cast<float>(i) * 0.173f) * 3.0f +
                    static_cast<float>(i % 17u) * 0.01f;
    }
    router_comparison(config.experts, config.top_k).time_one_input(logits);
}

void test_synthetic_moe() {
    q4::moe_config config;
    config.hidden = 64u;
    config.intermediate = 32u;
    config.experts = 3u;
    config.top_k = 2u;
    config.normalize_topk = true;
    require(q4::moe_validate_config(config) == q4::moe_status::ok,
            "synthetic MoE config rejected");
    constexpr std::uint32_t slots = 2u;

    const std::size_t gu_weight_size =
            static_cast<std::size_t>(slots) * config.intermediate * (config.hidden / 2u);
    const std::size_t gu_scale_size =
            static_cast<std::size_t>(slots) * config.intermediate * (config.hidden / 16u);
    const std::size_t down_weight_size =
            static_cast<std::size_t>(slots) * config.hidden * (config.intermediate / 2u);
    const std::size_t down_scale_size =
            static_cast<std::size_t>(slots) * config.hidden * (config.intermediate / 16u);
    std::vector<std::uint8_t> gate_weight(gu_weight_size);
    std::vector<std::uint8_t> up_weight(gu_weight_size);
    std::vector<std::uint8_t> down_weight(down_weight_size);
    fill_weight(&gate_weight, slots, config.intermediate, config.hidden, 1u);
    fill_weight(&up_weight, slots, config.intermediate, config.hidden, 7u);
    fill_weight(&down_weight, slots, config.hidden, config.intermediate, 13u);
    std::vector<std::uint8_t> gate_scale(gu_scale_size, 0x38u);
    std::vector<std::uint8_t> up_scale(gu_scale_size, 0x30u);
    std::vector<std::uint8_t> down_scale(down_scale_size, 0x38u);
    std::vector<float> gate_global{0.03125f, 0.046875f};
    std::vector<float> up_global{0.025f, 0.0375f};
    std::vector<float> down_global{0.02f, 0.03f};
    std::vector<float> input(config.hidden);
    for (std::uint32_t i = 0; i < config.hidden; ++i) {
        input[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.2f;
    }
    std::vector<std::uint32_t> slot_indices{0u, 1u};
    std::vector<float> router_weights{0.65f, 0.35f};

    std::vector<float> expected(config.hidden, 0.0f);
    std::vector<float> mid(static_cast<std::size_t>(slots) * config.intermediate);
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
        for (std::uint32_t row = 0; row < config.intermediate; ++row) {
            const float gate = matvec_element(
                    gate_weight, gate_scale, gate_global, slots,
                    config.intermediate, config.hidden, slot, row, input.data());
            const float up = matvec_element(
                    up_weight, up_scale, up_global, slots,
                    config.intermediate, config.hidden, slot, row, input.data());
            mid[static_cast<std::size_t>(slot) * config.intermediate + row] =
                    (gate / (1.0f + std::exp(-gate))) * up;
        }
        for (std::uint32_t row = 0; row < config.hidden; ++row) {
            expected[row] += router_weights[slot] * matvec_element(
                    down_weight, down_scale, down_global, slots,
                    config.hidden, config.intermediate, slot, row,
                    mid.data() + static_cast<std::size_t>(slot) * config.intermediate);
        }
    }

    device_buffer<std::uint8_t> d_gate_weight(gate_weight.size());
    device_buffer<std::uint8_t> d_up_weight(up_weight.size());
    device_buffer<std::uint8_t> d_down_weight(down_weight.size());
    device_buffer<std::uint8_t> d_gate_scale(gate_scale.size());
    device_buffer<std::uint8_t> d_up_scale(up_scale.size());
    device_buffer<std::uint8_t> d_down_scale(down_scale.size());
    device_buffer<float> d_gate_global(gate_global.size());
    device_buffer<float> d_up_global(up_global.size());
    device_buffer<float> d_down_global(down_global.size());
    device_buffer<float> d_input(input.size());
    device_buffer<std::uint32_t> d_indices(slot_indices.size());
    device_buffer<float> d_router(router_weights.size());
    device_buffer<float> d_mid(mid.size());
    device_buffer<float> d_output(expected.size());
    d_gate_weight.upload(gate_weight); d_up_weight.upload(up_weight);
    d_down_weight.upload(down_weight); d_gate_scale.upload(gate_scale);
    d_up_scale.upload(up_scale); d_down_scale.upload(down_scale);
    d_gate_global.upload(gate_global); d_up_global.upload(up_global);
    d_down_global.upload(down_global); d_input.upload(input);
    d_indices.upload(slot_indices); d_router.upload(router_weights);

    const q4::moe_slot_planes planes{
        d_gate_weight.data(), d_gate_scale.data(), d_gate_global.data(),
        d_up_weight.data(), d_up_scale.data(), d_up_global.data(),
        d_down_weight.data(), d_down_scale.data(), d_down_global.data()};
    require(q4::moe_forward_slots_f32_cuda(
                    0, config, slots, planes, d_indices.data(), d_router.data(),
                    d_input.data(), d_mid.data(), d_output.data()) == q4::moe_status::ok,
            "fused slot MoE failed");
    std::vector<float> actual(expected.size());
    d_output.download(&actual);
    float maximum = 0.0f;
    float rms_error = 0.0f;
    float rms_reference = 0.0f;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float error = std::abs(actual[index] - expected[index]);
        maximum = std::max(maximum, error);
        rms_error += error * error;
        rms_reference += expected[index] * expected[index];
    }
    const float relative = std::sqrt(rms_error / std::max(rms_reference, 1.0e-30f));
    require(maximum < 2.0e-5f && relative < 2.0e-5f,
            "fused slot MoE differs from host NVFP4 oracle");
    std::printf("qwen4exp-moe synthetic max_abs=%.9g relative_rms=%.9g\n",
                maximum, relative);
}

}  // namespace

int main() {
    try {
        int devices = 0;
        require_cuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        require(devices > 0, "qwen4exp MoE test requires a CUDA device");
        require_cuda(cudaSetDevice(0), "cudaSetDevice");
        test_router();
        test_router_bitwise();
        test_synthetic_moe();
        optional_router_timing();
        std::puts("qwen4exp-moe-test: pass");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "qwen4exp-moe-test: %s\n", error.what());
        return 1;
    }
}
