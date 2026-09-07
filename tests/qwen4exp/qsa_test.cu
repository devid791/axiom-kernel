#include "axiom/qwen4exp/qsa.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qsa = axiom::qwen4exp::qsa;

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

void require_status(qsa::status actual, qsa::status expected, const std::string& where) {
    if (actual != expected) {
        fail(where + ": expected " + qsa::status_string(expected) + ", got " + qsa::status_string(actual));
    }
}

void cuda_require(cudaError_t result, const std::string& where) {
    if (result != cudaSuccess) {
        fail(where + ": " + cudaGetErrorString(result));
    }
}

void near(float actual, float expected, float tolerance, const std::string& where) {
    if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
        fail(where + ": expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

template <typename T>
class device_buffer {
public:
    explicit device_buffer(std::size_t count) : count_(count) {
        if (count_ != 0) {
            cuda_require(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)), "cudaMalloc");
        }
    }

    ~device_buffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }

    void upload(const std::vector<T>& source) {
        require(source.size() <= count_, "device upload exceeds capacity");
        if (!source.empty()) {
            cuda_require(
                cudaMemcpy(data_, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice),
                "cudaMemcpy H2D");
        }
    }

    std::vector<T> download(std::size_t count) const {
        require(count <= count_, "device download exceeds capacity");
        std::vector<T> output(count);
        if (count != 0) {
            cuda_require(
                cudaMemcpy(output.data(), data_, count * sizeof(T), cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H");
        }
        return output;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

qsa::config small_config() {
    qsa::config cfg{};
    cfg.query_heads = 2;
    cfg.kv_heads = 1;
    cfg.head_dim = 4;
    cfg.rotary_dim = 4;
    cfg.compress_ratio = 2;
    cfg.token_budget = 4;
    cfg.max_context = 4096;
    cfg.rms_epsilon = 1.0e-6F;
    return cfg;
}

void test_contract_and_workspace_gate() {
    require_status(qsa::validate_checkpoint_config(qsa::checkpoint_config), qsa::status::ok, "checkpoint config");

    std::size_t topk = 0;
    require_status(qsa::block_topk(qsa::checkpoint_config, &topk), qsa::status::ok, "checkpoint topk");
    require(topk == qsa::checkpoint_block_topk, "checkpoint topk must be 512");

    std::size_t selected = 0;
    require_status(
        qsa::selected_token_capacity(qsa::checkpoint_config, 513U * 4U + 3U, &selected),
        qsa::status::ok,
        "checkpoint selected capacity");
    require(selected == qsa::checkpoint_selected_capacity, "checkpoint selected capacity must be 2051");

    std::size_t host_bytes = 0;
    std::size_t cuda_bytes = 0;
    constexpr std::size_t maximum_blocks = 262144U / 4U;
    require_status(
        qsa::host_select_workspace_bytes(qsa::checkpoint_config, maximum_blocks, &host_bytes),
        qsa::status::ok,
        "real host workspace query");
    require_status(
        qsa::cuda_select_workspace_bytes(qsa::checkpoint_config, maximum_blocks, &cuda_bytes),
        qsa::status::ok,
        "real CUDA workspace query");
    require(
        host_bytes >= 512U * 8U && host_bytes <= 512U * 8U + 16U,
        "host top-k workspace must remain bounded to 512 aligned candidates");
    require(cuda_bytes > maximum_blocks * 12U, "CUDA workspace must expose all sort buffers");
    require(cuda_bytes < 64U * 1024U * 1024U, "real dimension gate must not request a huge workspace");

    qsa::config invalid = qsa::checkpoint_config;
    invalid.kv_heads = 2;
    require_status(qsa::validate_config(invalid), qsa::status::invalid_config, "kv_heads fail closed");
    invalid = qsa::checkpoint_config;
    invalid.rotary_dim = 65;
    require_status(qsa::validate_config(invalid), qsa::status::invalid_config, "odd rotary fail closed");
}

void test_rmsnorm_and_rope_analytic() {
    const qsa::config cfg = small_config();
    const std::vector<float> raw{
        1.0F, 2.0F, 3.0F, 4.0F,
        -1.0F, 2.0F, -3.0F, 4.0F,
    };
    const std::vector<float> weight{0.0F, 1.0F, 0.5F, -0.5F};
    const std::vector<float> cosine{0.0F, 0.0F, 0.0F, 0.0F};
    const std::vector<float> sine{1.0F, 1.0F, 1.0F, 1.0F};
    std::vector<float> output(raw.size());
    require_status(
        qsa::prepare_queries_host_f32(
            cfg, raw.data(), weight.data(), cosine.data(), sine.data(), 1, output.data()),
        qsa::status::ok,
        "analytic prepare");

    for (std::size_t head = 0; head < 2; ++head) {
        const float* input = raw.data() + head * 4;
        const float inverse = 1.0F /
                              std::sqrt((input[0] * input[0] + input[1] * input[1] +
                                         input[2] * input[2] + input[3] * input[3]) /
                                            4.0F +
                                        cfg.rms_epsilon);
        const float scaled0 = input[0] * inverse;
        const float scaled1 = input[1] * inverse * 2.0F;
        const float scaled2 = input[2] * inverse * 1.5F;
        const float scaled3 = input[3] * inverse * 0.5F;
        near(output[head * 4 + 0], -scaled2, 2.0e-6F, "RoPE first half 0");
        near(output[head * 4 + 1], -scaled3, 2.0e-6F, "RoPE first half 1");
        near(output[head * 4 + 2], scaled0, 2.0e-6F, "RoPE second half 0");
        near(output[head * 4 + 3], scaled1, 2.0e-6F, "RoPE second half 1");
    }
}

void test_selection_tail_mask_and_fail_closed() {
    qsa::config cfg = small_config();
    cfg.rotary_dim = 0;
    const std::vector<std::int32_t> visible{0, 2, 3, 5, 6, 8, 9};
    const std::vector<float> scores{1.0F, 3.0F, 2.0F};
    std::size_t workspace_bytes = 0;
    require_status(
        qsa::host_select_workspace_bytes(cfg, scores.size(), &workspace_bytes),
        qsa::status::ok,
        "small host workspace");
    std::vector<std::uint8_t> workspace(workspace_bytes);
    std::vector<std::int32_t> selected(5, -1);
    std::size_t selected_count = 0;
    require_status(
        qsa::select_tokens_host_f32(
            cfg,
            scores.data(),
            scores.size(),
            visible.data(),
            visible.size(),
            10,
            selected.data(),
            selected.size(),
            workspace.data(),
            workspace.size(),
            &selected_count),
        qsa::status::ok,
        "topk and tail");
    const std::vector<std::int32_t> expected{3, 5, 6, 8, 9};
    require(selected_count == expected.size(), "selected count with tail");
    require(selected == expected, "top-k expansion and causal tail order");

    std::vector<std::uint8_t> mask(10, 99U);
    require_status(
        qsa::build_mask_host(selected.data(), selected_count, mask.size(), mask.data()),
        qsa::status::ok,
        "build mask");
    for (std::size_t i = 0; i < mask.size(); ++i) {
        const bool expected_bit = std::find(expected.begin(), expected.end(), static_cast<std::int32_t>(i)) != expected.end();
        require(mask[i] == static_cast<std::uint8_t>(expected_bit), "mask semantic mismatch");
    }

    const std::vector<std::int32_t> unsorted{0, 2, 2, 5};
    require_status(
        qsa::select_tokens_host_f32(
            cfg,
            scores.data(),
            2,
            unsorted.data(),
            unsorted.size(),
            10,
            selected.data(),
            selected.size(),
            workspace.data(),
            workspace.size(),
            &selected_count),
        qsa::status::invalid_visible_index,
        "duplicate visible index");

    std::vector<float> invalid_scores = scores;
    invalid_scores[1] = std::numeric_limits<float>::quiet_NaN();
    require_status(
        qsa::select_tokens_host_f32(
            cfg,
            invalid_scores.data(),
            invalid_scores.size(),
            visible.data(),
            visible.size(),
            10,
            selected.data(),
            selected.size(),
            workspace.data(),
            workspace.size(),
            &selected_count),
        qsa::status::non_finite,
        "nonfinite score");
}

std::vector<std::int32_t> host_boundary_selection(std::size_t block_count) {
    const qsa::config cfg = qsa::checkpoint_config;
    const std::size_t tail = block_count == 513 ? 3 : 0;
    const std::size_t visible_count = block_count * cfg.compress_ratio + tail;
    std::vector<std::int32_t> visible(visible_count);
    for (std::size_t i = 0; i < visible.size(); ++i) {
        visible[i] = static_cast<std::int32_t>(i);
    }
    std::vector<float> scores(block_count);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        scores[i] = static_cast<float>(i);
    }
    std::size_t workspace_bytes = 0;
    require_status(
        qsa::host_select_workspace_bytes(cfg, block_count, &workspace_bytes),
        qsa::status::ok,
        "boundary host workspace");
    std::vector<std::uint8_t> workspace(workspace_bytes);
    std::size_t capacity = 0;
    require_status(
        qsa::selected_token_capacity(cfg, visible_count, &capacity),
        qsa::status::ok,
        "boundary selected capacity");
    std::vector<std::int32_t> selected(capacity, -1);
    std::size_t selected_count = 0;
    require_status(
        qsa::select_tokens_host_f32(
            cfg,
            scores.data(),
            block_count,
            visible.data(),
            visible.size(),
            visible.size(),
            selected.data(),
            selected.size(),
            workspace.data(),
            workspace.size(),
            &selected_count),
        qsa::status::ok,
        "boundary select");
    require(selected_count == capacity, "boundary selected count");
    if (block_count == 512) {
        require(selected.size() == 2048, "512 blocks retain every token");
        require(selected.front() == 2044 && selected.back() == 3, "512 block ordering");
    } else {
        require(selected.size() == 2051, "513 blocks plus tail have 2051 outputs");
        require(selected[0] == 2048 && selected[3] == 2051, "highest block selected first");
        require(selected[2044] == 4 && selected[2047] == 7, "lowest retained block is one");
        require(selected[2048] == 2052 && selected[2050] == 2054, "causal tail appended unchanged");
    }
    return selected;
}

void test_exact_512_513_boundaries_and_ties() {
    (void)host_boundary_selection(512);
    (void)host_boundary_selection(513);

    qsa::config cfg = small_config();
    cfg.compress_ratio = 1;
    cfg.token_budget = 2;
    const std::vector<std::int32_t> visible{0, 1, 2};
    const std::vector<float> tied{5.0F, 5.0F, 1.0F};
    std::size_t bytes = 0;
    require_status(qsa::host_select_workspace_bytes(cfg, 3, &bytes), qsa::status::ok, "tie workspace");
    std::vector<std::uint8_t> workspace(bytes);
    std::vector<std::int32_t> selected(2, -1);
    std::size_t count = 0;
    require_status(
        qsa::select_tokens_host_f32(
            cfg,
            tied.data(),
            tied.size(),
            visible.data(),
            visible.size(),
            visible.size(),
            selected.data(),
            selected.size(),
            workspace.data(),
            workspace.size(),
            &count),
        qsa::status::ok,
        "tie selection");
    require(selected == std::vector<std::int32_t>({0, 1}), "ties must prefer ascending block ordinal");
}

struct small_fixture {
    qsa::config cfg = small_config();
    std::vector<float> raw_query{
        1.0F, 2.0F, 3.0F, 4.0F,
        -1.0F, 2.0F, -3.0F, 4.0F,
    };
    std::vector<float> q_weight{0.0F, 0.25F, -0.125F, 0.5F};
    std::vector<float> query_cos{0.9950042F, 0.9800666F, 0.9950042F, 0.9800666F};
    std::vector<float> query_sin{0.0998334F, 0.1986693F, 0.0998334F, 0.1986693F};
    std::vector<float> raw_keys;
    std::vector<float> k_weight{0.125F, -0.25F, 0.0F, 0.375F};
    std::vector<float> full_cos;
    std::vector<float> full_sin;
    std::vector<std::int32_t> visible{0, 2, 3, 5, 6, 8, 9};

    small_fixture() : raw_keys(10 * 4), full_cos(10 * 4), full_sin(10 * 4) {
        for (std::size_t token = 0; token < 10; ++token) {
            for (std::size_t dim = 0; dim < 4; ++dim) {
                raw_keys[token * 4 + dim] =
                    static_cast<float>((static_cast<int>(token) - 3) * (static_cast<int>(dim) + 1)) * 0.125F +
                    static_cast<float>(dim) * 0.2F;
                const float angle = static_cast<float>(token) * 0.01F * static_cast<float>((dim % 2) + 1);
                full_cos[token * 4 + dim] = std::cos(angle);
                full_sin[token * 4 + dim] = std::sin(angle);
            }
        }
    }
};

void compare_vectors(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance,
    const std::string& where) {
    require(actual.size() == expected.size(), where + " size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        near(actual[i], expected[i], tolerance, where + "[" + std::to_string(i) + "]");
    }
}

void test_cuda_f32_and_bf16_parity() {
    const small_fixture fixture;
    const std::size_t blocks = fixture.visible.size() / fixture.cfg.compress_ratio;

    std::vector<float> host_query(fixture.raw_query.size());
    std::vector<float> host_pooled(blocks * fixture.cfg.head_dim);
    std::vector<float> host_scores(blocks);
    std::size_t host_blocks = 0;
    require_status(
        qsa::prepare_queries_host_f32(
            fixture.cfg,
            fixture.raw_query.data(),
            fixture.q_weight.data(),
            fixture.query_cos.data(),
            fixture.query_sin.data(),
            1,
            host_query.data()),
        qsa::status::ok,
        "host query parity oracle");
    require_status(
        qsa::pool_keys_host_f32(
            fixture.cfg,
            fixture.raw_keys.data(),
            10,
            fixture.visible.data(),
            fixture.visible.size(),
            fixture.k_weight.data(),
            fixture.full_cos.data(),
            fixture.full_sin.data(),
            10,
            host_pooled.data(),
            blocks,
            &host_blocks),
        qsa::status::ok,
        "host pooled parity oracle");
    require(host_blocks == blocks, "host pool block count");
    require_status(
        qsa::score_blocks_host_f32(
            fixture.cfg, host_query.data(), host_pooled.data(), blocks, host_scores.data()),
        qsa::status::ok,
        "host score parity oracle");

    device_buffer<float> d_raw_query(fixture.raw_query.size());
    device_buffer<float> d_q_weight(fixture.q_weight.size());
    device_buffer<float> d_query_cos(fixture.query_cos.size());
    device_buffer<float> d_query_sin(fixture.query_sin.size());
    device_buffer<float> d_query(host_query.size());
    device_buffer<float> d_raw_keys(fixture.raw_keys.size());
    device_buffer<float> d_k_weight(fixture.k_weight.size());
    device_buffer<float> d_full_cos(fixture.full_cos.size());
    device_buffer<float> d_full_sin(fixture.full_sin.size());
    device_buffer<std::int32_t> d_visible(fixture.visible.size());
    device_buffer<float> d_pooled(host_pooled.size());
    device_buffer<float> d_scores(host_scores.size());
    device_buffer<qsa::status> d_status(1);

    d_raw_query.upload(fixture.raw_query);
    d_q_weight.upload(fixture.q_weight);
    d_query_cos.upload(fixture.query_cos);
    d_query_sin.upload(fixture.query_sin);
    d_raw_keys.upload(fixture.raw_keys);
    d_k_weight.upload(fixture.k_weight);
    d_full_cos.upload(fixture.full_cos);
    d_full_sin.upload(fixture.full_sin);
    d_visible.upload(fixture.visible);

    cudaStream_t stream = nullptr;
    cuda_require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    require_status(qsa::cuda_reset_status(d_status.get(), stream), qsa::status::ok, "reset status");
    require_status(
        qsa::prepare_queries_cuda_f32(
            fixture.cfg,
            d_raw_query.get(),
            d_q_weight.get(),
            d_query_cos.get(),
            d_query_sin.get(),
            1,
            d_query.get(),
            d_status.get(),
            stream),
        qsa::status::ok,
        "CUDA prepare F32");
    std::size_t cuda_blocks = 0;
    require_status(
        qsa::pool_keys_cuda_f32(
            fixture.cfg,
            d_raw_keys.get(),
            10,
            d_visible.get(),
            fixture.visible.size(),
            d_k_weight.get(),
            d_full_cos.get(),
            d_full_sin.get(),
            10,
            d_pooled.get(),
            blocks,
            &cuda_blocks,
            d_status.get(),
            stream),
        qsa::status::ok,
        "CUDA pool F32");
    require(cuda_blocks == blocks, "CUDA pool block count");
    require_status(
        qsa::score_blocks_cuda_f32(
            fixture.cfg,
            d_query.get(),
            d_pooled.get(),
            blocks,
            d_scores.get(),
            d_status.get(),
            stream),
        qsa::status::ok,
        "CUDA score F32");
    qsa::status device_status = qsa::status::cuda_failure;
    require_status(
        qsa::cuda_collect_status(d_status.get(), stream, &device_status),
        qsa::status::ok,
        "collect F32 status");
    require_status(device_status, qsa::status::ok, "F32 device status");
    compare_vectors(d_query.download(host_query.size()), host_query, 3.0e-6F, "query CPU/CUDA parity");
    compare_vectors(d_pooled.download(host_pooled.size()), host_pooled, 4.0e-6F, "pool CPU/CUDA parity");
    compare_vectors(d_scores.download(host_scores.size()), host_scores, 6.0e-6F, "score CPU/CUDA parity");

    std::size_t select_workspace_bytes = 0;
    require_status(
        qsa::cuda_select_workspace_bytes(fixture.cfg, blocks, &select_workspace_bytes),
        qsa::status::ok,
        "CUDA select workspace");
    device_buffer<std::uint8_t> d_workspace(select_workspace_bytes);
    std::size_t selected_capacity = 0;
    require_status(
        qsa::selected_token_capacity(fixture.cfg, fixture.visible.size(), &selected_capacity),
        qsa::status::ok,
        "CUDA selected capacity");
    device_buffer<std::int32_t> d_selected(selected_capacity);
    device_buffer<std::uint8_t> d_mask(10);
    std::size_t selected_count = 0;
    require_status(qsa::cuda_reset_status(d_status.get(), stream), qsa::status::ok, "reset select status");
    require_status(
        qsa::select_tokens_cuda_f32(
            fixture.cfg,
            d_scores.get(),
            blocks,
            d_visible.get(),
            fixture.visible.size(),
            10,
            d_selected.get(),
            selected_capacity,
            d_workspace.get(),
            d_workspace.size(),
            &selected_count,
            d_status.get(),
            stream),
        qsa::status::ok,
        "CUDA select");
    require_status(
        qsa::build_mask_cuda(
            d_selected.get(), selected_count, 10, d_mask.get(), d_status.get(), stream),
        qsa::status::ok,
        "CUDA mask");
    require_status(
        qsa::cuda_collect_status(d_status.get(), stream, &device_status),
        qsa::status::ok,
        "collect select status");
    require_status(device_status, qsa::status::ok, "select device status");

    std::size_t host_select_bytes = 0;
    require_status(
        qsa::host_select_workspace_bytes(fixture.cfg, blocks, &host_select_bytes),
        qsa::status::ok,
        "host select parity workspace");
    std::vector<std::uint8_t> host_workspace(host_select_bytes);
    std::vector<std::int32_t> host_selected(selected_capacity);
    std::size_t host_selected_count = 0;
    require_status(
        qsa::select_tokens_host_f32(
            fixture.cfg,
            host_scores.data(),
            blocks,
            fixture.visible.data(),
            fixture.visible.size(),
            10,
            host_selected.data(),
            host_selected.size(),
            host_workspace.data(),
            host_workspace.size(),
            &host_selected_count),
        qsa::status::ok,
        "host select parity oracle");
    require(selected_count == host_selected_count, "CUDA selected count parity");
    require(d_selected.download(selected_count) == host_selected, "CUDA selected tokens parity");
    std::vector<std::uint8_t> host_mask(10);
    require_status(
        qsa::build_mask_host(host_selected.data(), host_selected.size(), 10, host_mask.data()),
        qsa::status::ok,
        "host mask parity oracle");
    require(d_mask.download(10) == host_mask, "CUDA bool mask parity");

    std::vector<__nv_bfloat16> bf16_query(fixture.raw_query.size());
    std::vector<__nv_bfloat16> bf16_q_weight(fixture.q_weight.size());
    std::vector<__nv_bfloat16> bf16_keys(fixture.raw_keys.size());
    std::vector<__nv_bfloat16> bf16_k_weight(fixture.k_weight.size());
    std::transform(
        fixture.raw_query.begin(), fixture.raw_query.end(), bf16_query.begin(),
        [](float value) { return __float2bfloat16(value); });
    std::transform(
        fixture.q_weight.begin(), fixture.q_weight.end(), bf16_q_weight.begin(),
        [](float value) { return __float2bfloat16(value); });
    std::transform(
        fixture.raw_keys.begin(), fixture.raw_keys.end(), bf16_keys.begin(),
        [](float value) { return __float2bfloat16(value); });
    std::transform(
        fixture.k_weight.begin(), fixture.k_weight.end(), bf16_k_weight.begin(),
        [](float value) { return __float2bfloat16(value); });
    device_buffer<__nv_bfloat16> d_bf16_query(bf16_query.size());
    device_buffer<__nv_bfloat16> d_bf16_q_weight(bf16_q_weight.size());
    device_buffer<__nv_bfloat16> d_bf16_keys(bf16_keys.size());
    device_buffer<__nv_bfloat16> d_bf16_k_weight(bf16_k_weight.size());
    d_bf16_query.upload(bf16_query);
    d_bf16_q_weight.upload(bf16_q_weight);
    d_bf16_keys.upload(bf16_keys);
    d_bf16_k_weight.upload(bf16_k_weight);
    require_status(qsa::cuda_reset_status(d_status.get(), stream), qsa::status::ok, "reset BF16 status");
    require_status(
        qsa::prepare_queries_cuda_bf16(
            fixture.cfg,
            d_bf16_query.get(),
            d_bf16_q_weight.get(),
            d_query_cos.get(),
            d_query_sin.get(),
            1,
            d_query.get(),
            d_status.get(),
            stream),
        qsa::status::ok,
        "CUDA prepare BF16");
    require_status(
        qsa::pool_keys_cuda_bf16(
            fixture.cfg,
            d_bf16_keys.get(),
            10,
            d_visible.get(),
            fixture.visible.size(),
            d_bf16_k_weight.get(),
            d_full_cos.get(),
            d_full_sin.get(),
            10,
            d_pooled.get(),
            blocks,
            &cuda_blocks,
            d_status.get(),
            stream),
        qsa::status::ok,
        "CUDA pool BF16");
    require_status(
        qsa::cuda_collect_status(d_status.get(), stream, &device_status),
        qsa::status::ok,
        "collect BF16 status");
    require_status(device_status, qsa::status::ok, "BF16 device status");
    compare_vectors(d_query.download(host_query.size()), host_query, 8.0e-3F, "BF16 query tolerance");
    compare_vectors(d_pooled.download(host_pooled.size()), host_pooled, 8.0e-3F, "BF16 pool tolerance");

    const std::vector<float> tied_scores{5.0F, 5.0F, 1.0F};
    d_scores.upload(tied_scores);
    require_status(qsa::cuda_reset_status(d_status.get(), stream), qsa::status::ok, "reset tie status");
    require_status(
        qsa::select_tokens_cuda_f32(
            fixture.cfg,
            d_scores.get(),
            blocks,
            d_visible.get(),
            fixture.visible.size(),
            10,
            d_selected.get(),
            selected_capacity,
            d_workspace.get(),
            d_workspace.size(),
            &selected_count,
            d_status.get(),
            stream),
        qsa::status::ok,
        "CUDA stable tie select");
    require_status(
        qsa::cuda_collect_status(d_status.get(), stream, &device_status),
        qsa::status::ok,
        "collect tie status");
    require_status(device_status, qsa::status::ok, "tie device status");
    require(
        d_selected.download(selected_count) == std::vector<std::int32_t>({0, 2, 3, 5, 9}),
        "CUDA ties must preserve ascending block ordinal before causal tail");

    std::vector<float> nan_scores = host_scores;
    nan_scores[0] = std::numeric_limits<float>::quiet_NaN();
    d_scores.upload(nan_scores);
    require_status(qsa::cuda_reset_status(d_status.get(), stream), qsa::status::ok, "reset invalid status");
    require_status(
        qsa::select_tokens_cuda_f32(
            fixture.cfg,
            d_scores.get(),
            blocks,
            d_visible.get(),
            fixture.visible.size(),
            10,
            d_selected.get(),
            selected_capacity,
            d_workspace.get(),
            d_workspace.size(),
            &selected_count,
            d_status.get(),
            stream),
        qsa::status::ok,
        "launch nonfinite select");
    require_status(
        qsa::cuda_collect_status(d_status.get(), stream, &device_status),
        qsa::status::ok,
        "collect nonfinite status");
    require_status(device_status, qsa::status::non_finite, "device nonfinite fail closed");

    cuda_require(cudaStreamDestroy(stream), "cudaStreamDestroy");
}

void test_cuda_513_boundary() {
    const qsa::config cfg = qsa::checkpoint_config;
    constexpr std::size_t blocks = 513;
    constexpr std::size_t visible_count = blocks * 4 + 3;
    std::vector<float> scores(blocks);
    std::vector<std::int32_t> visible(visible_count);
    for (std::size_t i = 0; i < blocks; ++i) {
        scores[i] = static_cast<float>(i);
    }
    for (std::size_t i = 0; i < visible_count; ++i) {
        visible[i] = static_cast<std::int32_t>(i);
    }
    const std::vector<std::int32_t> expected = host_boundary_selection(blocks);

    device_buffer<float> d_scores(scores.size());
    device_buffer<std::int32_t> d_visible(visible.size());
    device_buffer<std::int32_t> d_selected(expected.size());
    device_buffer<qsa::status> d_status(1);
    d_scores.upload(scores);
    d_visible.upload(visible);
    std::size_t workspace_bytes = 0;
    require_status(
        qsa::cuda_select_workspace_bytes(cfg, blocks, &workspace_bytes),
        qsa::status::ok,
        "513 CUDA workspace");
    device_buffer<std::uint8_t> d_workspace(workspace_bytes);
    cudaStream_t stream = nullptr;
    cuda_require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "513 stream create");
    require_status(qsa::cuda_reset_status(d_status.get(), stream), qsa::status::ok, "513 reset status");
    std::size_t count = 0;
    require_status(
        qsa::select_tokens_cuda_f32(
            cfg,
            d_scores.get(),
            blocks,
            d_visible.get(),
            visible.size(),
            visible.size(),
            d_selected.get(),
            expected.size(),
            d_workspace.get(),
            d_workspace.size(),
            &count,
            d_status.get(),
            stream),
        qsa::status::ok,
        "513 CUDA select");
    qsa::status device_status = qsa::status::cuda_failure;
    require_status(
        qsa::cuda_collect_status(d_status.get(), stream, &device_status),
        qsa::status::ok,
        "513 collect status");
    require_status(device_status, qsa::status::ok, "513 device status");
    require(count == expected.size(), "513 CUDA count");
    require(d_selected.download(count) == expected, "513 CPU/CUDA parity");
    cuda_require(cudaStreamDestroy(stream), "513 stream destroy");
}

}  // namespace

int main() {
    try {
        int device = -1;
        cuda_require(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        cuda_require(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        require(properties.major == 12 && properties.minor == 0, "QSA CUDA gate requires SM120");

        test_contract_and_workspace_gate();
        test_rmsnorm_and_rope_analytic();
        test_selection_tail_mask_and_fail_closed();
        test_exact_512_513_boundaries_and_ties();
        test_cuda_f32_and_bf16_parity();
        test_cuda_513_boundary();

        std::cout << "qwen4exp-qsa-test: pass"
                  << " sm=" << properties.major << properties.minor
                  << " topk=" << qsa::checkpoint_block_topk
                  << " boundary=512/513"
                  << " f32=parity bf16=input mask=oracle" << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "qwen4exp-qsa-test: FAIL: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
