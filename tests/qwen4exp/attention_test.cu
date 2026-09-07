#include "axiom/qwen4exp/attention.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace attention = axiom::qwen4exp::attention;

namespace {

static_assert(std::is_standard_layout_v<attention::config>);
static_assert(std::is_trivially_copyable_v<attention::config>);
static_assert(std::is_standard_layout_v<attention::cache_state>);
static_assert(std::is_trivially_copyable_v<attention::cache_state>);
static_assert(std::is_standard_layout_v<attention::staged_batch>);
static_assert(std::is_trivially_copyable_v<attention::staged_batch>);

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

void require_status(
    attention::status actual,
    attention::status expected,
    const std::string& where) {
    if (actual != expected) {
        fail(
            where + ": expected " + attention::status_string(expected) +
            ", got " + attention::status_string(actual));
    }
}

void cuda_require(cudaError_t result, const std::string& where) {
    if (result != cudaSuccess) {
        fail(where + ": " + cudaGetErrorString(result));
    }
}

std::uint16_t bf16_bits(float value) {
    const __nv_bfloat16 converted = __float2bfloat16_rn(value);
    const __nv_bfloat16_raw raw = static_cast<__nv_bfloat16_raw>(converted);
    return raw.x;
}

float bf16_float(std::uint16_t bits) {
    __nv_bfloat16_raw raw{};
    raw.x = bits;
    const __nv_bfloat16 converted(raw);
    return __bfloat162float(converted);
}

float bf16_round(float value) {
    return bf16_float(bf16_bits(value));
}

template <typename T>
class device_buffer {
public:
    explicit device_buffer(std::size_t count) : count_(count) {
        if (count_ != 0u) {
            cuda_require(
                cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
                "cudaMalloc");
        }
    }

    ~device_buffer() {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }

    void upload(const std::vector<T>& source, cudaStream_t stream) {
        require(source.size() <= count_, "device upload exceeds capacity");
        if (!source.empty()) {
            cuda_require(
                cudaMemcpyAsync(
                    data_,
                    source.data(),
                    source.size() * sizeof(T),
                    cudaMemcpyHostToDevice,
                    stream),
                "cudaMemcpyAsync H2D");
        }
    }

    std::vector<T> download(std::size_t count, cudaStream_t stream) const {
        require(count <= count_, "device download exceeds capacity");
        std::vector<T> output(count);
        if (count != 0u) {
            cuda_require(
                cudaMemcpyAsync(
                    output.data(),
                    data_,
                    count * sizeof(T),
                    cudaMemcpyDeviceToHost,
                    stream),
                "cudaMemcpyAsync D2H");
        }
        cuda_require(cudaStreamSynchronize(stream), "cudaStreamSynchronize download");
        return output;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0u;
};

attention::config small_config() {
    attention::config value{};
    value.query_heads = 4u;
    value.kv_heads = 2u;
    value.head_dim = 8u;
    value.rotary_dim = 4u;
    value.max_context = 16u;
    value.max_selected_tokens = 8u;
    value.rms_epsilon = 1.0e-6F;
    return value;
}

struct fixture_inputs {
    std::size_t tokens = 0u;
    std::vector<std::uint16_t> q;
    std::vector<std::uint16_t> k;
    std::vector<std::uint16_t> v;
    std::vector<float> cosine;
    std::vector<float> sine;
};

fixture_inputs make_inputs(
    const attention::config& value,
    std::size_t tokens,
    float seed) {
    fixture_inputs result{};
    result.tokens = tokens;
    result.q.resize(
        tokens * value.query_heads * 2u * value.head_dim);
    result.k.resize(tokens * value.kv_heads * value.head_dim);
    result.v.resize(tokens * value.kv_heads * value.head_dim);
    result.cosine.resize(tokens * value.rotary_dim);
    result.sine.resize(tokens * value.rotary_dim);

    for (std::size_t token = 0u; token < tokens; ++token) {
        for (std::size_t head = 0u; head < value.query_heads; ++head) {
            const std::size_t base =
                (token * value.query_heads + head) * 2u * value.head_dim;
            for (std::size_t dim = 0u; dim < value.head_dim; ++dim) {
                const float query =
                    std::sin(seed + 0.17F * static_cast<float>(token + 1u) +
                             0.11F * static_cast<float>(head + 1u) +
                             0.07F * static_cast<float>(dim + 1u));
                const float gate =
                    -0.8F + 0.13F * static_cast<float>(head) +
                    0.09F * static_cast<float>(dim) +
                    0.05F * static_cast<float>(token);
                result.q[base + dim] = bf16_bits(query);
                result.q[base + value.head_dim + dim] = bf16_bits(gate);
            }
        }
        for (std::size_t head = 0u; head < value.kv_heads; ++head) {
            const std::size_t base =
                (token * value.kv_heads + head) * value.head_dim;
            for (std::size_t dim = 0u; dim < value.head_dim; ++dim) {
                result.k[base + dim] = bf16_bits(
                    std::cos(seed + 0.19F * static_cast<float>(token + 1u) +
                             0.23F * static_cast<float>(head + 1u) +
                             0.03F * static_cast<float>(dim + 1u)));
                result.v[base + dim] = bf16_bits(
                    -0.6F + 0.21F * static_cast<float>(token) +
                    0.17F * static_cast<float>(head) +
                    0.08F * static_cast<float>(dim));
            }
        }
        const std::size_t half = value.rotary_dim / 2u;
        for (std::size_t dim = 0u; dim < half; ++dim) {
            const float angle =
                seed * 0.1F + 0.2F * static_cast<float>(token) +
                0.07F * static_cast<float>(dim + 1u);
            result.cosine[token * value.rotary_dim + dim] = std::cos(angle);
            result.cosine[token * value.rotary_dim + dim + half] = std::cos(angle);
            result.sine[token * value.rotary_dim + dim] = std::sin(angle);
            result.sine[token * value.rotary_dim + dim + half] = std::sin(angle);
        }
    }
    return result;
}

std::vector<std::uint16_t> make_norm_weight(std::size_t dimensions, float phase) {
    std::vector<std::uint16_t> result(dimensions);
    for (std::size_t dim = 0u; dim < dimensions; ++dim) {
        result[dim] = bf16_bits(
            0.12F * std::sin(phase + 0.31F * static_cast<float>(dim + 1u)));
    }
    return result;
}

void oracle_norm_rope(
    const std::uint16_t* input,
    const std::vector<std::uint16_t>& weight,
    const float* cosine,
    const float* sine,
    const attention::config& value,
    std::vector<float>* output) {
    output->assign(value.head_dim, 0.0F);
    float square_sum = 0.0F;
    for (std::size_t dim = 0u; dim < value.head_dim; ++dim) {
        const float element = bf16_float(input[dim]);
        square_sum += element * element;
    }
    const float inverse = 1.0F /
                          std::sqrt(
                              square_sum / static_cast<float>(value.head_dim) +
                              value.rms_epsilon);
    for (std::size_t dim = 0u; dim < value.head_dim; ++dim) {
        (*output)[dim] = bf16_round(
            bf16_float(input[dim]) * inverse *
            (1.0F + bf16_float(weight[dim])));
    }
    const std::vector<float> normalized = *output;
    const std::size_t half = value.rotary_dim / 2u;
    for (std::size_t dim = 0u; dim < value.rotary_dim; ++dim) {
        const std::size_t pair = dim < half ? dim + half : dim - half;
        const float rotated_half =
            dim < half ? -normalized[pair] : normalized[pair];
        (*output)[dim] =
            normalized[dim] * cosine[dim] + rotated_half * sine[dim];
    }
}

struct oracle_stage_result {
    std::vector<float> prepared_queries;
};

oracle_stage_result oracle_stage(
    const attention::config& value,
    const fixture_inputs& input,
    const std::vector<std::uint16_t>& q_weight,
    const std::vector<std::uint16_t>& k_weight,
    std::size_t cache_start,
    std::vector<std::uint16_t>* key_cache,
    std::vector<std::uint16_t>* value_cache) {
    oracle_stage_result result{};
    result.prepared_queries.resize(
        input.tokens * value.query_heads * value.head_dim);
    std::vector<float> temporary;
    for (std::size_t token = 0u; token < input.tokens; ++token) {
        const float* cosine = input.cosine.data() + token * value.rotary_dim;
        const float* sine = input.sine.data() + token * value.rotary_dim;
        for (std::size_t head = 0u; head < value.query_heads; ++head) {
            const std::size_t source =
                (token * value.query_heads + head) * 2u * value.head_dim;
            oracle_norm_rope(
                input.q.data() + source,
                q_weight,
                cosine,
                sine,
                value,
                &temporary);
            std::copy(
                temporary.begin(),
                temporary.end(),
                result.prepared_queries.begin() +
                    static_cast<std::ptrdiff_t>(
                        (token * value.query_heads + head) * value.head_dim));
        }
        for (std::size_t head = 0u; head < value.kv_heads; ++head) {
            const std::size_t source =
                (token * value.kv_heads + head) * value.head_dim;
            oracle_norm_rope(
                input.k.data() + source,
                k_weight,
                cosine,
                sine,
                value,
                &temporary);
            const std::size_t destination =
                ((cache_start + token) * value.kv_heads + head) * value.head_dim;
            for (std::size_t dim = 0u; dim < value.head_dim; ++dim) {
                (*key_cache)[destination + dim] = bf16_bits(temporary[dim]);
                (*value_cache)[destination + dim] = input.v[source + dim];
            }
        }
    }
    return result;
}

std::vector<std::uint16_t> oracle_attention(
    const attention::config& value,
    const fixture_inputs& input,
    const oracle_stage_result& stage,
    const std::vector<std::uint16_t>& key_cache,
    const std::vector<std::uint16_t>& value_cache,
    const std::vector<std::int32_t>& selected,
    const std::vector<std::uint32_t>& counts,
    std::size_t stride) {
    std::vector<std::uint16_t> output(
        input.tokens * value.query_heads * value.head_dim,
        bf16_bits(0.0F));
    const std::size_t group = value.query_heads / value.kv_heads;
    const float scale =
        1.0F / std::sqrt(static_cast<float>(value.head_dim));
    for (std::size_t token = 0u; token < input.tokens; ++token) {
        for (std::size_t head = 0u; head < value.query_heads; ++head) {
            const std::size_t kv_head = head / group;
            const float* query =
                stage.prepared_queries.data() +
                (token * value.query_heads + head) * value.head_dim;
            std::vector<float> scores(counts[token], 0.0F);
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t ordinal = 0u; ordinal < counts[token]; ++ordinal) {
                const std::size_t index = static_cast<std::size_t>(
                    selected[token * stride + ordinal]);
                float dot = 0.0F;
                for (std::size_t dim = 0u; dim < value.head_dim; ++dim) {
                    const std::size_t cache_offset =
                        (index * value.kv_heads + kv_head) * value.head_dim + dim;
                    dot += query[dim] * bf16_float(key_cache[cache_offset]);
                }
                scores[ordinal] = dot * scale;
                maximum = std::max(maximum, scores[ordinal]);
            }
            float denominator = 0.0F;
            for (float score : scores) {
                denominator += std::exp(score - maximum);
            }
            for (std::size_t dim = 0u; dim < value.head_dim; ++dim) {
                float sum = 0.0F;
                for (std::size_t ordinal = 0u; ordinal < counts[token]; ++ordinal) {
                    const std::size_t index = static_cast<std::size_t>(
                        selected[token * stride + ordinal]);
                    const std::size_t cache_offset =
                        (index * value.kv_heads + kv_head) * value.head_dim + dim;
                    const float probability =
                        std::exp(scores[ordinal] - maximum) / denominator;
                    sum += probability * bf16_float(value_cache[cache_offset]);
                }
                const std::size_t q_offset =
                    (token * value.query_heads + head) * 2u * value.head_dim;
                const float gate =
                    bf16_float(input.q[q_offset + value.head_dim + dim]);
                const float sigmoid =
                    gate >= 0.0F
                        ? 1.0F / (1.0F + std::exp(-gate))
                        : std::exp(gate) / (1.0F + std::exp(gate));
                output[(token * value.query_heads + head) * value.head_dim + dim] =
                    bf16_bits(sum * sigmoid);
            }
        }
    }
    return output;
}

void compare_bf16(
    const std::vector<std::uint16_t>& actual,
    const std::vector<std::uint16_t>& expected,
    float absolute_tolerance,
    float relative_tolerance,
    const std::string& where) {
    require(actual.size() == expected.size(), where + ": size mismatch");
    float maximum_absolute = 0.0F;
    float maximum_relative = 0.0F;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        const float lhs = bf16_float(actual[index]);
        const float rhs = bf16_float(expected[index]);
        const float absolute = std::fabs(lhs - rhs);
        const float relative = absolute / std::max(1.0e-4F, std::fabs(rhs));
        maximum_absolute = std::max(maximum_absolute, absolute);
        maximum_relative = std::max(maximum_relative, relative);
        if (!std::isfinite(lhs) || absolute > absolute_tolerance + relative_tolerance * std::fabs(rhs)) {
            fail(
                where + ": mismatch at " + std::to_string(index) +
                ", actual=" + std::to_string(lhs) +
                ", expected=" + std::to_string(rhs));
        }
    }
    std::cout << where << " max_abs=" << maximum_absolute
              << " max_rel=" << maximum_relative << '\n';
}

void test_checkpoint_contract() {
    require_status(
        attention::validate_checkpoint_config(attention::checkpoint_config),
        attention::status::ok,
        "checkpoint config");
    require(attention::kCheckpointQueryHeads == 24u, "query heads must be 24");
    require(attention::kCheckpointKvHeads == 2u, "KV heads must be 2");
    require(attention::kCheckpointHeadDim == 256u, "head dim must be 256");
    require(attention::kCheckpointRotaryDim == 64u, "RoPE dim must be 64");
    require(attention::kCheckpointGqaRatio == 12u, "GQA ratio must be 12:1");
    require(attention::kCheckpointQHeadStride == 512u, "q head stride must be 512");
    require(attention::kCheckpointOutputDim == 6144u, "pre-o_proj dim must be 6144");
    require(attention::kCheckpointMaxContext == 262144u, "native context must be 262144");
    require(
        attention::kCheckpointRmsEpsilon == 1.0e-6F,
        "checkpoint RMS epsilon mismatch");
    require(
        attention::attention_scale(attention::checkpoint_config) ==
            attention::kCheckpointAttentionScale,
        "checkpoint attention scale must be exactly 1/16");

    std::size_t bytes = 0u;
    require_status(
        attention::workspace_bytes(attention::checkpoint_config, 2u, &bytes),
        attention::status::ok,
        "checkpoint workspace");
    require(
        bytes == 2u * 24u * 256u * sizeof(float) + 255u,
        "workspace formula mismatch");

    attention::config invalid = attention::checkpoint_config;
    invalid.kv_heads = 5u;
    require_status(
        attention::validate_config(invalid),
        attention::status::invalid_config,
        "invalid GQA");
    invalid = attention::checkpoint_config;
    invalid.rotary_dim = 65u;
    require_status(
        attention::validate_config(invalid),
        attention::status::invalid_config,
        "odd rotary dim");
}

struct cuda_fixture {
    explicit cuda_fixture(const attention::config& value, std::size_t capacity)
        : config(value),
          cache_elements(capacity * value.kv_heads * value.head_dim),
          key_cache(cache_elements),
          value_cache(cache_elements),
          device_status(1u) {
        cuda_require(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
        require_status(
            attention::cache_initialize(
                &cache,
                key_cache.get(),
                value_cache.get(),
                capacity),
            attention::status::ok,
            "cache_initialize");
    }

    ~cuda_fixture() {
        if (stream != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream));
        }
    }

    attention::config config{};
    std::size_t cache_elements = 0u;
    device_buffer<std::uint16_t> key_cache;
    device_buffer<std::uint16_t> value_cache;
    device_buffer<attention::status> device_status;
    attention::cache_state cache{};
    cudaStream_t stream = nullptr;
};

struct uploaded_inputs {
    explicit uploaded_inputs(const fixture_inputs& input)
        : q(input.q.size()),
          k(input.k.size()),
          v(input.v.size()),
          cosine(input.cosine.size()),
          sine(input.sine.size()) {}

    device_buffer<std::uint16_t> q;
    device_buffer<std::uint16_t> k;
    device_buffer<std::uint16_t> v;
    device_buffer<float> cosine;
    device_buffer<float> sine;
};

attention::staged_batch stage_fixture(
    cuda_fixture* fixture,
    const fixture_inputs& input,
    const std::vector<std::uint16_t>& q_weight,
    const std::vector<std::uint16_t>& k_weight,
    uploaded_inputs* uploaded,
    device_buffer<std::uint16_t>* device_q_weight,
    device_buffer<std::uint16_t>* device_k_weight,
    device_buffer<std::uint8_t>* workspace) {
    uploaded->q.upload(input.q, fixture->stream);
    uploaded->k.upload(input.k, fixture->stream);
    uploaded->v.upload(input.v, fixture->stream);
    uploaded->cosine.upload(input.cosine, fixture->stream);
    uploaded->sine.upload(input.sine, fixture->stream);
    device_q_weight->upload(q_weight, fixture->stream);
    device_k_weight->upload(k_weight, fixture->stream);
    require_status(
        attention::reset_device_status(fixture->device_status.get(), fixture->stream),
        attention::status::ok,
        "reset stage status");
    attention::staged_batch batch{};
    require_status(
        attention::stage_cuda(
            fixture->config,
            &fixture->cache,
            uploaded->q.get(),
            uploaded->k.get(),
            uploaded->v.get(),
            device_q_weight->get(),
            device_k_weight->get(),
            uploaded->cosine.get(),
            uploaded->sine.get(),
            input.tokens,
            workspace->get(),
            workspace->size(),
            &batch,
            fixture->device_status.get(),
            fixture->stream),
        attention::status::ok,
        "stage_cuda");
    attention::status asynchronous = attention::status::cuda_failure;
    require_status(
        attention::collect_device_status(
            fixture->device_status.get(),
            &asynchronous,
            fixture->stream),
        attention::status::ok,
        "collect stage status");
    require_status(asynchronous, attention::status::ok, "stage device status");
    return batch;
}

std::vector<std::uint16_t> forward_fixture(
    cuda_fixture* fixture,
    const attention::staged_batch& batch,
    const std::vector<std::int32_t>& selected,
    const std::vector<std::uint32_t>& counts,
    std::size_t stride,
    attention::status expected_device_status) {
    device_buffer<std::int32_t> device_selected(selected.size());
    device_buffer<std::uint32_t> device_counts(counts.size());
    device_buffer<std::uint16_t> output(
        batch.token_count * fixture->config.query_heads * fixture->config.head_dim);
    device_selected.upload(selected, fixture->stream);
    device_counts.upload(counts, fixture->stream);
    require_status(
        attention::reset_device_status(fixture->device_status.get(), fixture->stream),
        attention::status::ok,
        "reset forward status");
    require_status(
        attention::forward_qsa_cuda(
            fixture->config,
            &fixture->cache,
            batch,
            device_selected.get(),
            device_counts.get(),
            stride,
            output.get(),
            fixture->device_status.get(),
            fixture->stream),
        attention::status::ok,
        "forward_qsa_cuda");
    attention::status asynchronous = attention::status::cuda_failure;
    require_status(
        attention::collect_device_status(
            fixture->device_status.get(),
            &asynchronous,
            fixture->stream),
        attention::status::ok,
        "collect forward status");
    require_status(asynchronous, expected_device_status, "forward device status");
    return output.download(output.size(), fixture->stream);
}

void test_prefill_decode_and_qsa_parity() {
    const attention::config value = small_config();
    cuda_fixture fixture(value, 8u);
    const std::vector<std::uint16_t> q_weight =
        make_norm_weight(value.head_dim, 0.2F);
    const std::vector<std::uint16_t> k_weight =
        make_norm_weight(value.head_dim, 0.7F);
    device_buffer<std::uint16_t> device_q_weight(q_weight.size());
    device_buffer<std::uint16_t> device_k_weight(k_weight.size());
    std::size_t workspace_capacity = 0u;
    require_status(
        attention::workspace_bytes(value, 3u, &workspace_capacity),
        attention::status::ok,
        "prefill workspace");
    device_buffer<std::uint8_t> workspace(workspace_capacity);
    std::vector<std::uint16_t> oracle_key(fixture.cache_elements, bf16_bits(0.0F));
    std::vector<std::uint16_t> oracle_value(fixture.cache_elements, bf16_bits(0.0F));

    require_status(
        attention::begin_transaction(&fixture.cache),
        attention::status::ok,
        "prefill begin");
    const fixture_inputs prefill = make_inputs(value, 3u, 0.4F);
    uploaded_inputs device_prefill(prefill);
    const attention::staged_batch prefill_batch = stage_fixture(
        &fixture,
        prefill,
        q_weight,
        k_weight,
        &device_prefill,
        &device_q_weight,
        &device_k_weight,
        &workspace);
    require(prefill_batch.start_position == 0u, "prefill must start at zero");
    const oracle_stage_result oracle_prefill = oracle_stage(
        value,
        prefill,
        q_weight,
        k_weight,
        0u,
        &oracle_key,
        &oracle_value);
    const std::vector<std::int32_t> selected{
        0, -1, -1,
        0, 1, -1,
        0, 2, -1,
    };
    const std::vector<std::uint32_t> counts{1u, 2u, 2u};
    constexpr std::size_t stride = 3u;
    const std::vector<std::uint16_t> expected = oracle_attention(
        value,
        prefill,
        oracle_prefill,
        oracle_key,
        oracle_value,
        selected,
        counts,
        stride);
    const std::vector<std::uint16_t> actual = forward_fixture(
        &fixture,
        prefill_batch,
        selected,
        counts,
        stride,
        attention::status::ok);
    /* The oracle sums dimensions serially while CUDA uses a balanced tree;
     * after BF16 output rounding, 1/128 absolute plus 2% relative is a strict
     * bound for these deterministic values.
     */
    compare_bf16(actual, expected, 7.8125e-3F, 2.0e-2F, "prefill BF16 parity");
    compare_bf16(
        fixture.key_cache.download(3u * value.kv_heads * value.head_dim, fixture.stream),
        std::vector<std::uint16_t>(
            oracle_key.begin(),
            oracle_key.begin() +
                static_cast<std::ptrdiff_t>(3u * value.kv_heads * value.head_dim)),
        7.8125e-3F,
        1.0e-2F,
        "prefill K cache parity");
    require_status(
        attention::commit_prefix(&fixture.cache, 3u),
        attention::status::ok,
        "prefill commit");

    attention::cache_lengths lengths{};
    require_status(
        attention::cache_get_lengths(&fixture.cache, &lengths),
        attention::status::ok,
        "lengths after prefill");
    require(
        lengths.committed == 3u && lengths.staged == 0u && lengths.visible == 3u,
        "prefill lengths mismatch");

    require_status(
        attention::begin_transaction(&fixture.cache),
        attention::status::ok,
        "decode begin");
    const fixture_inputs decode = make_inputs(value, 1u, 1.1F);
    uploaded_inputs device_decode(decode);
    const attention::staged_batch decode_batch = stage_fixture(
        &fixture,
        decode,
        q_weight,
        k_weight,
        &device_decode,
        &device_q_weight,
        &device_k_weight,
        &workspace);
    require(decode_batch.start_position == 3u, "decode must append at token 3");
    const oracle_stage_result oracle_decode = oracle_stage(
        value,
        decode,
        q_weight,
        k_weight,
        3u,
        &oracle_key,
        &oracle_value);
    const std::vector<std::int32_t> decode_selected{0, 2, 3};
    const std::vector<std::uint32_t> decode_counts{3u};
    const std::vector<std::uint16_t> expected_decode = oracle_attention(
        value,
        decode,
        oracle_decode,
        oracle_key,
        oracle_value,
        decode_selected,
        decode_counts,
        3u);
    const std::vector<std::uint16_t> actual_decode = forward_fixture(
        &fixture,
        decode_batch,
        decode_selected,
        decode_counts,
        3u,
        attention::status::ok);
    compare_bf16(
        actual_decode,
        expected_decode,
        7.8125e-3F,
        2.0e-2F,
        "decode BF16 parity");
    require_status(
        attention::commit_prefix(&fixture.cache, 1u),
        attention::status::ok,
        "decode commit");
}

void test_checkpoint_shape_decode_parity() {
    const attention::config value = attention::checkpoint_config;
    cuda_fixture fixture(value, 1u);
    const std::vector<std::uint16_t> q_weight =
        make_norm_weight(value.head_dim, 0.15F);
    const std::vector<std::uint16_t> k_weight =
        make_norm_weight(value.head_dim, 0.65F);
    device_buffer<std::uint16_t> device_q_weight(q_weight.size());
    device_buffer<std::uint16_t> device_k_weight(k_weight.size());
    std::size_t workspace_capacity = 0u;
    require_status(
        attention::workspace_bytes(value, 1u, &workspace_capacity),
        attention::status::ok,
        "checkpoint-shape workspace");
    device_buffer<std::uint8_t> workspace(workspace_capacity);
    std::vector<std::uint16_t> oracle_key(fixture.cache_elements, bf16_bits(0.0F));
    std::vector<std::uint16_t> oracle_value(fixture.cache_elements, bf16_bits(0.0F));

    require_status(
        attention::begin_transaction(&fixture.cache),
        attention::status::ok,
        "checkpoint-shape begin");
    const fixture_inputs input = make_inputs(value, 1u, 0.9F);
    uploaded_inputs uploaded(input);
    const attention::staged_batch batch = stage_fixture(
        &fixture,
        input,
        q_weight,
        k_weight,
        &uploaded,
        &device_q_weight,
        &device_k_weight,
        &workspace);
    const oracle_stage_result oracle = oracle_stage(
        value,
        input,
        q_weight,
        k_weight,
        0u,
        &oracle_key,
        &oracle_value);
    const std::vector<std::int32_t> selected{0};
    const std::vector<std::uint32_t> counts{1u};
    const std::vector<std::uint16_t> expected = oracle_attention(
        value,
        input,
        oracle,
        oracle_key,
        oracle_value,
        selected,
        counts,
        1u);
    const std::vector<std::uint16_t> actual = forward_fixture(
        &fixture,
        batch,
        selected,
        counts,
        1u,
        attention::status::ok);
    require(
        actual.size() == attention::kCheckpointOutputDim,
        "checkpoint output must contain 6144 BF16 values");
    compare_bf16(
        actual,
        expected,
        7.8125e-3F,
        2.0e-2F,
        "checkpoint-shape decode parity");
    require_status(
        attention::commit_prefix(&fixture.cache, 1u),
        attention::status::ok,
        "checkpoint-shape commit");
}

void test_transaction_prefix_rollback_and_capacity() {
    const attention::config value = small_config();
    cuda_fixture fixture(value, 3u);
    const std::vector<std::uint16_t> q_weight(value.head_dim, bf16_bits(0.0F));
    const std::vector<std::uint16_t> k_weight(value.head_dim, bf16_bits(0.0F));
    device_buffer<std::uint16_t> device_q_weight(q_weight.size());
    device_buffer<std::uint16_t> device_k_weight(k_weight.size());
    std::size_t workspace_capacity = 0u;
    require_status(
        attention::workspace_bytes(value, 3u, &workspace_capacity),
        attention::status::ok,
        "transaction workspace");
    device_buffer<std::uint8_t> workspace(workspace_capacity);

    require_status(
        attention::begin_transaction(&fixture.cache),
        attention::status::ok,
        "transaction begin");
    const fixture_inputs first = make_inputs(value, 3u, 2.0F);
    uploaded_inputs device_first(first);
    static_cast<void>(stage_fixture(
        &fixture,
        first,
        q_weight,
        k_weight,
        &device_first,
        &device_q_weight,
        &device_k_weight,
        &workspace));
    attention::cache_lengths lengths{};
    require_status(
        attention::cache_get_lengths(&fixture.cache, &lengths),
        attention::status::ok,
        "staged lengths");
    require(
        lengths.committed == 0u && lengths.staged == 3u && lengths.visible == 3u,
        "staged length mismatch");
    require_status(
        attention::commit_prefix(&fixture.cache, 4u),
        attention::status::invalid_argument,
        "commit beyond staged suffix");
    require_status(
        attention::cache_get_lengths(&fixture.cache, &lengths),
        attention::status::ok,
        "lengths after rejected commit");
    require(
        lengths.committed == 0u && lengths.staged == 3u && lengths.visible == 3u,
        "rejected commit must preserve the transaction");
    require_status(
        attention::commit_prefix(&fixture.cache, 2u),
        attention::status::ok,
        "accepted prefix");
    require_status(
        attention::cache_get_lengths(&fixture.cache, &lengths),
        attention::status::ok,
        "prefix lengths");
    require(
        lengths.committed == 2u && lengths.staged == 0u && lengths.visible == 2u,
        "rejected suffix must be invisible");

    require_status(
        attention::begin_transaction(&fixture.cache),
        attention::status::ok,
        "overwrite suffix begin");
    const fixture_inputs overwrite = make_inputs(value, 1u, 4.0F);
    uploaded_inputs device_overwrite(overwrite);
    const attention::staged_batch overwrite_batch = stage_fixture(
        &fixture,
        overwrite,
        q_weight,
        k_weight,
        &device_overwrite,
        &device_q_weight,
        &device_k_weight,
        &workspace);
    require(overwrite_batch.start_position == 2u, "rejected suffix slot must be reused");
    require_status(
        attention::rollback(&fixture.cache),
        attention::status::ok,
        "rollback overwrite");
    require_status(
        attention::cache_get_lengths(&fixture.cache, &lengths),
        attention::status::ok,
        "rollback lengths");
    require(lengths.visible == 2u, "rollback must restore visible length");
    device_buffer<std::int32_t> stale_selected(1u);
    device_buffer<std::uint32_t> stale_count(1u);
    device_buffer<std::uint16_t> stale_output(
        value.query_heads * value.head_dim);
    stale_selected.upload(std::vector<std::int32_t>{0}, fixture.stream);
    stale_count.upload(std::vector<std::uint32_t>{1u}, fixture.stream);
    require_status(
        attention::forward_qsa_cuda(
            value,
            &fixture.cache,
            overwrite_batch,
            stale_selected.get(),
            stale_count.get(),
            1u,
            stale_output.get(),
            fixture.device_status.get(),
            fixture.stream),
        attention::status::invalid_state,
        "rolled-back batch must not remain visible");

    require_status(
        attention::begin_transaction(&fixture.cache),
        attention::status::ok,
        "capacity begin");
    fixture_inputs too_many = make_inputs(value, 2u, 5.0F);
    uploaded_inputs device_too_many(too_many);
    device_too_many.q.upload(too_many.q, fixture.stream);
    device_too_many.k.upload(too_many.k, fixture.stream);
    device_too_many.v.upload(too_many.v, fixture.stream);
    device_too_many.cosine.upload(too_many.cosine, fixture.stream);
    device_too_many.sine.upload(too_many.sine, fixture.stream);
    device_q_weight.upload(q_weight, fixture.stream);
    device_k_weight.upload(k_weight, fixture.stream);
    attention::staged_batch rejected{};
    require_status(
        attention::stage_cuda(
            value,
            &fixture.cache,
            device_too_many.q.get(),
            device_too_many.k.get(),
            device_too_many.v.get(),
            device_q_weight.get(),
            device_k_weight.get(),
            device_too_many.cosine.get(),
            device_too_many.sine.get(),
            too_many.tokens,
            workspace.get(),
            workspace.size(),
            &rejected,
            fixture.device_status.get(),
            fixture.stream),
        attention::status::capacity_exceeded,
        "cache overflow must fail closed");
    require_status(
        attention::rollback(&fixture.cache),
        attention::status::ok,
        "capacity rollback");
}

void test_invalid_qsa_lists_fail_closed() {
    const attention::config value = small_config();
    cuda_fixture fixture(value, 4u);
    const std::vector<std::uint16_t> q_weight(value.head_dim, bf16_bits(0.0F));
    const std::vector<std::uint16_t> k_weight(value.head_dim, bf16_bits(0.0F));
    device_buffer<std::uint16_t> device_q_weight(q_weight.size());
    device_buffer<std::uint16_t> device_k_weight(k_weight.size());
    std::size_t workspace_capacity = 0u;
    require_status(
        attention::workspace_bytes(value, 2u, &workspace_capacity),
        attention::status::ok,
        "invalid-list workspace");
    device_buffer<std::uint8_t> workspace(workspace_capacity);
    require_status(
        attention::begin_transaction(&fixture.cache),
        attention::status::ok,
        "invalid-list begin");
    const fixture_inputs input = make_inputs(value, 2u, 6.0F);
    uploaded_inputs uploaded(input);
    const attention::staged_batch batch = stage_fixture(
        &fixture,
        input,
        q_weight,
        k_weight,
        &uploaded,
        &device_q_weight,
        &device_k_weight,
        &workspace);

    const std::vector<std::uint16_t> future_output = forward_fixture(
        &fixture,
        batch,
        std::vector<std::int32_t>{1, -1, 0, 1},
        std::vector<std::uint32_t>{1u, 2u},
        2u,
        attention::status::invalid_selected_index);
    for (std::size_t index = 0u;
         index < value.query_heads * value.head_dim;
         ++index) {
        require(
            bf16_float(future_output[index]) == 0.0F,
            "future causal row must be zeroed");
    }

    const std::vector<std::uint16_t> count_output = forward_fixture(
        &fixture,
        batch,
        std::vector<std::int32_t>{0, -1, 0, 1},
        std::vector<std::uint32_t>{3u, 2u},
        2u,
        attention::status::invalid_selected_count);
    for (std::size_t index = 0u;
         index < value.query_heads * value.head_dim;
         ++index) {
        require(
            bf16_float(count_output[index]) == 0.0F,
            "invalid-count row must be zeroed");
    }
    require_status(
        attention::rollback(&fixture.cache),
        attention::status::ok,
        "invalid-list rollback");
}

void test_sm120_device() {
    int device = 0;
    cuda_require(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_require(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    require(
        properties.major == 12 && properties.minor == 0,
        "attention gate requires an actual sm_120 device");
}

}  // namespace

int main() {
    try {
        test_sm120_device();
        test_checkpoint_contract();
        test_checkpoint_shape_decode_parity();
        test_prefill_decode_and_qsa_parity();
        test_transaction_prefix_rollback_and_capacity();
        test_invalid_qsa_lists_fail_closed();
        std::cout
            << "qwen4exp-attention-test: pass sm=120 q=24 kv=2 d=256 rope=64 "
               "gqa=12:1 context=262144 prefill=parity decode=parity "
               "transaction=accepted-prefix/rollback qsa=fail-closed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen4exp-attention-test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
