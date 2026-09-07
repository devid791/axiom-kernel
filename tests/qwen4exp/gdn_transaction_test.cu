#include "axiom/qwen4exp/gdn_transaction.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

void require_cuda(cudaError_t status, const std::string& label) {
    if (status != cudaSuccess) {
        fail(label + ": " + cudaGetErrorString(status));
    }
}

void require_gdn(
    q4::GdnStatus actual,
    q4::GdnStatus expected,
    const std::string& label) {
    if (actual != expected) {
        fail(label + ": expected " + q4::gdn_status_string(expected) +
             ", got " + q4::gdn_status_string(actual));
    }
}

void require_transaction(
    q4::GdnTransactionStatus actual,
    q4::GdnTransactionStatus expected,
    const std::string& label) {
    if (actual != expected) {
        fail(label + ": expected " +
             q4::gdn_transaction_status_string(expected) + ", got " +
             q4::gdn_transaction_status_string(actual));
    }
}

float deterministic_value(
    std::uint64_t seed,
    std::size_t index,
    float scale) {
    std::uint64_t value = seed + 0x9E3779B97F4A7C15ULL * (index + 1ULL);
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    const std::int32_t signed_value =
        static_cast<std::int32_t>((value >> 32U) & 0xFFFFU) - 32768;
    return scale * static_cast<float>(signed_value) / 32768.0F;
}

void fill_deterministic(
    std::vector<float>* values,
    std::uint64_t seed,
    float scale) {
    require(values != nullptr, "fill_deterministic received null vector");
    for (std::size_t index = 0; index < values->size(); ++index) {
        (*values)[index] = deterministic_value(seed, index, scale);
    }
}

float compare_vectors(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float absolute_tolerance,
    float relative_tolerance,
    const std::string& label) {
    require(expected.size() == actual.size(), label + ": size mismatch");
    float maximum_absolute = 0.0F;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(std::isfinite(expected[index]), label + ": non-finite oracle");
        require(std::isfinite(actual[index]), label + ": non-finite CUDA value");
        const float absolute = std::abs(expected[index] - actual[index]);
        const float allowed = absolute_tolerance +
            relative_tolerance * std::abs(expected[index]);
        maximum_absolute = std::max(maximum_absolute, absolute);
        if (absolute > allowed) {
            fail(label + ": mismatch at " + std::to_string(index) +
                 ", expected=" + std::to_string(expected[index]) +
                 ", actual=" + std::to_string(actual[index]) +
                 ", absolute=" + std::to_string(absolute) +
                 ", allowed=" + std::to_string(allowed));
        }
    }
    return maximum_absolute;
}

class Stream final {
public:
    Stream() {
        require_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~Stream() {
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

    void synchronize() const {
        require_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    cudaStream_t stream_ = nullptr;
};

template <typename T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
        require(elements_ != 0u, "DeviceBuffer cannot be empty");
        require(
            elements_ <= std::numeric_limits<std::size_t>::max() / sizeof(T),
            "DeviceBuffer byte overflow");
        require_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&pointer_), elements_ * sizeof(T)),
            "cudaMalloc DeviceBuffer");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            (void)cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] T* data() noexcept { return pointer_; }
    [[nodiscard]] const T* data() const noexcept { return pointer_; }

    void copy_from(const std::vector<T>& source, cudaStream_t stream) {
        require(source.size() == elements_, "DeviceBuffer H2D size mismatch");
        require_cuda(
            cudaMemcpyAsync(
                pointer_,
                source.data(),
                elements_ * sizeof(T),
                cudaMemcpyHostToDevice,
                stream),
            "cudaMemcpyAsync H2D");
    }

    void copy_to(std::vector<T>* destination, cudaStream_t stream) const {
        require(
            destination != nullptr && destination->size() == elements_,
            "DeviceBuffer D2H size mismatch");
        require_cuda(
            cudaMemcpyAsync(
                destination->data(),
                pointer_,
                elements_ * sizeof(T),
                cudaMemcpyDeviceToHost,
                stream),
            "cudaMemcpyAsync D2H");
    }

private:
    T* pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

struct Token {
    std::vector<float> projected;
    std::vector<float> z;
    std::vector<float> a;
    std::vector<float> b;
};

Token make_token(
    const q4::GdnConfig& config,
    const q4::GdnFootprint& footprint,
    std::size_t token_index) {
    Token token{
        std::vector<float>(footprint.step_workspace_floats),
        std::vector<float>(
            config.batch * config.value_heads * config.value_head_dim),
        std::vector<float>(config.batch * config.value_heads),
        std::vector<float>(config.batch * config.value_heads)};
    fill_deterministic(&token.projected, 1000u + token_index * 13u, 0.45F);
    fill_deterministic(&token.z, 2000u + token_index * 17u, 1.2F);
    fill_deterministic(&token.a, 3000u + token_index * 19u, 0.55F);
    fill_deterministic(&token.b, 4000u + token_index * 23u, 0.8F);
    return token;
}

struct DeviceStateImage {
    std::vector<std::byte> conv;
    std::vector<std::byte> recurrent;
    std::uint32_t cursor = 0u;
    std::uint64_t tokens = 0u;
};

DeviceStateImage capture_state(
    const q4::GdnDeviceState& state,
    const q4::GdnFootprint& footprint,
    const Stream& stream) {
    DeviceStateImage image{
        std::vector<std::byte>(footprint.conv_state_floats * sizeof(float)),
        std::vector<std::byte>(footprint.recurrent_state_floats * sizeof(float)),
        state.conv_cursor,
        state.tokens_seen};
    require_cuda(
        cudaMemcpyAsync(
            image.conv.data(),
            state.conv_state,
            image.conv.size(),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "capture conv state");
    require_cuda(
        cudaMemcpyAsync(
            image.recurrent.data(),
            state.recurrent_state,
            image.recurrent.size(),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "capture recurrent state");
    stream.synchronize();
    return image;
}

void require_byte_exact(
    const DeviceStateImage& expected,
    const DeviceStateImage& actual,
    const std::string& label) {
    require(expected.cursor == actual.cursor, label + ": cursor differs");
    require(expected.tokens == actual.tokens, label + ": token count differs");
    require(expected.conv.size() == actual.conv.size(), label + ": conv size differs");
    require(
        expected.recurrent.size() == actual.recurrent.size(),
        label + ": recurrent size differs");
    require(
        std::memcmp(
            expected.conv.data(), actual.conv.data(), expected.conv.size()) == 0,
        label + ": conv bytes differ");
    require(
        std::memcmp(
            expected.recurrent.data(),
            actual.recurrent.data(),
            expected.recurrent.size()) == 0,
        label + ": recurrent bytes differ");
}

struct DeviceStepBuffers {
    explicit DeviceStepBuffers(
        const q4::GdnConfig& config,
        const q4::GdnFootprint& footprint)
        : projected(footprint.step_workspace_floats),
          z(config.batch * config.value_heads * config.value_head_dim),
          a(config.batch * config.value_heads),
          b(config.batch * config.value_heads),
          convolved(footprint.step_workspace_floats),
          output(config.batch * config.value_heads * config.value_head_dim) {}

    DeviceBuffer<float> projected;
    DeviceBuffer<float> z;
    DeviceBuffer<float> a;
    DeviceBuffer<float> b;
    DeviceBuffer<float> convolved;
    DeviceBuffer<float> output;
};

std::vector<float> stage_cuda(
    q4::GdnTransactionState* transaction,
    const q4::GdnWeightsView& weights,
    DeviceStepBuffers* buffers,
    const Token& token,
    const Stream& stream) {
    require(transaction != nullptr && buffers != nullptr, "stage_cuda null state");
    buffers->projected.copy_from(token.projected, stream.get());
    buffers->z.copy_from(token.z, stream.get());
    buffers->a.copy_from(token.a, stream.get());
    buffers->b.copy_from(token.b, stream.get());
    const q4::GdnStepInputs inputs{
        buffers->projected.data(),
        buffers->z.data(),
        buffers->a.data(),
        buffers->b.data()};
    const q4::GdnStepOutputs outputs{
        buffers->convolved.data(), buffers->output.data()};
    require_transaction(
        q4::gdn_transaction_stage(
            transaction, inputs, weights, outputs, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "transaction stage");
    std::vector<float> result(token.z.size(), 0.0F);
    buffers->output.copy_to(&result, stream.get());
    stream.synchronize();
    return result;
}

std::vector<float> step_host(
    const q4::GdnConfig& config,
    const q4::GdnWeightsView& weights,
    const q4::GdnHostStateView& state,
    const q4::GdnFootprint& footprint,
    const Token& token) {
    std::vector<float> convolved(footprint.step_workspace_floats, 0.0F);
    std::vector<float> output(token.z.size(), 0.0F);
    const q4::GdnStepInputs inputs{
        token.projected.data(), token.z.data(), token.a.data(), token.b.data()};
    const q4::GdnStepOutputs outputs{convolved.data(), output.data()};
    require_gdn(
        q4::gdn_step_host(config, inputs, weights, state, outputs),
        q4::GdnStatus::kOk,
        "host oracle step");
    return output;
}

std::size_t test_real_qwen4_exp_snapshot() {
    Stream stream;
    const q4::GdnConfig config = q4::gdn_qwen4_exp_config(1u);
    require(q4::gdn_is_qwen4_exp_contract(config), "real GDN contract rejected");
    q4::GdnFootprint footprint{};
    require_gdn(
        q4::gdn_footprint(config, &footprint),
        q4::GdnStatus::kOk,
        "real GDN footprint");

    q4::GdnDeviceState state{};
    require_gdn(
        q4::gdn_device_state_init(&state, config, stream.get()),
        q4::GdnStatus::kOk,
        "real GDN state init");
    q4::GdnTransactionState transaction{};
    require_transaction(
        q4::gdn_transaction_state_init(&transaction, &state, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "real transaction init");
    require(
        transaction.conv_state_bytes ==
            footprint.conv_state_floats * sizeof(float),
        "real conv snapshot is not exact");
    require(
        transaction.recurrent_state_bytes ==
            footprint.recurrent_state_floats * sizeof(float),
        "real recurrent snapshot is not exact");
    require(
        transaction.snapshot_allocation_bytes ==
            sizeof(q4::GdnTransactionDeviceMetadata) +
                footprint.device_state_bytes,
        "real snapshot footprint mismatch");
    require_transaction(
        q4::gdn_transaction_begin(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "real empty transaction begin");
    require_transaction(
        q4::gdn_transaction_rollback(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "real empty transaction rollback");
    stream.synchronize();
    const std::size_t snapshot_bytes = transaction.snapshot_allocation_bytes;
    require_transaction(
        q4::gdn_transaction_state_release(&transaction),
        q4::GdnTransactionStatus::kOk,
        "real transaction release");
    require_gdn(
        q4::gdn_device_state_release(&state),
        q4::GdnStatus::kOk,
        "real GDN state release");
    return snapshot_bytes;
}

float run_transaction_test() {
    int device = -1;
    cudaDeviceProp properties{};
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    require_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    require(
        properties.major == 12 && properties.minor == 0,
        "test requires an sm_120 GPU");

    q4::GdnConfig config{};
    config.batch = 1u;
    config.key_heads = 2u;
    config.value_heads = 4u;
    config.key_head_dim = 8u;
    config.value_head_dim = 8u;
    q4::GdnFootprint footprint{};
    require_gdn(
        q4::gdn_footprint(config, &footprint),
        q4::GdnStatus::kOk,
        "reduced footprint");

    std::vector<float> conv_weight(
        footprint.conv_channels * config.conv_kernel);
    std::vector<float> a_log(config.value_heads);
    std::vector<float> dt_bias(config.value_heads);
    std::vector<float> norm_weight(config.value_head_dim);
    fill_deterministic(&conv_weight, 11u, 0.12F);
    for (std::size_t channel = 0; channel < footprint.conv_channels; ++channel) {
        conv_weight[channel * config.conv_kernel + config.conv_kernel - 1u] += 0.75F;
    }
    for (std::size_t head = 0; head < config.value_heads; ++head) {
        a_log[head] = std::log(0.05F + 0.025F * static_cast<float>(head + 1u));
        dt_bias[head] = deterministic_value(17u, head, 0.25F);
    }
    for (std::size_t index = 0; index < norm_weight.size(); ++index) {
        norm_weight[index] = 1.0F + deterministic_value(19u, index, 0.15F);
    }
    const q4::GdnWeightsView host_weights{
        conv_weight.data(),
        nullptr,
        a_log.data(),
        dt_bias.data(),
        norm_weight.data()};

    std::vector<float> host_conv(footprint.conv_state_floats, 0.0F);
    std::vector<float> host_recurrent(footprint.recurrent_state_floats, 0.0F);
    std::uint32_t host_cursor = 0u;
    std::uint64_t host_tokens = 0u;
    const q4::GdnHostStateView host_state{
        host_conv.data(),
        host_conv.size(),
        host_recurrent.data(),
        host_recurrent.size(),
        &host_cursor,
        &host_tokens};
    require_gdn(
        q4::gdn_host_state_reset(config, host_state),
        q4::GdnStatus::kOk,
        "host reset");

    Stream stream;
    Stream wrong_stream;
    q4::GdnDeviceState device_state{};
    require_gdn(
        q4::gdn_device_state_init(&device_state, config, stream.get()),
        q4::GdnStatus::kOk,
        "device state init");
    stream.synchronize();

    q4::GdnDeviceState uninitialized_state{};
    q4::GdnTransactionState invalid_transaction{};
    require_transaction(
        q4::gdn_transaction_state_init(
            &invalid_transaction, &uninitialized_state, stream.get()),
        q4::GdnTransactionStatus::kUninitialized,
        "reject uninitialized GDN state");

    q4::GdnTransactionState transaction{};
    require_transaction(
        q4::gdn_transaction_state_init(
            &transaction, &device_state, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "transaction init");
    require(
        transaction.snapshot_allocation_bytes ==
            sizeof(q4::GdnTransactionDeviceMetadata) +
                footprint.device_state_bytes,
        "snapshot is not the exact metadata plus state footprint");
    require(
        transaction.snapshot_allocation_bytes <=
            q4::kGdnTransactionMaxSnapshotBytes,
        "snapshot exceeds bound");
    require_transaction(
        q4::gdn_transaction_state_init(
            &transaction, &device_state, stream.get()),
        q4::GdnTransactionStatus::kAlreadyInitialized,
        "reject duplicate transaction init");

    DeviceBuffer<float> device_conv_weight(conv_weight.size());
    DeviceBuffer<float> device_a_log(a_log.size());
    DeviceBuffer<float> device_dt_bias(dt_bias.size());
    DeviceBuffer<float> device_norm_weight(norm_weight.size());
    device_conv_weight.copy_from(conv_weight, stream.get());
    device_a_log.copy_from(a_log, stream.get());
    device_dt_bias.copy_from(dt_bias, stream.get());
    device_norm_weight.copy_from(norm_weight, stream.get());
    const q4::GdnWeightsView device_weights{
        device_conv_weight.data(),
        nullptr,
        device_a_log.data(),
        device_dt_bias.data(),
        device_norm_weight.data()};
    DeviceStepBuffers buffers(config, footprint);

    std::vector<Token> tokens;
    for (std::size_t index = 0; index < 8u; ++index) {
        tokens.push_back(make_token(config, footprint, index));
    }

    require_transaction(
        q4::gdn_transaction_stage(
            &transaction,
            q4::GdnStepInputs{},
            device_weights,
            q4::GdnStepOutputs{},
            stream.get()),
        q4::GdnTransactionStatus::kNoTransaction,
        "stage requires transaction");
    require_transaction(
        q4::gdn_transaction_commit_all(&transaction, stream.get()),
        q4::GdnTransactionStatus::kNoTransaction,
        "commit requires transaction");
    require_transaction(
        q4::gdn_transaction_rollback(&transaction, stream.get()),
        q4::GdnTransactionStatus::kNoTransaction,
        "rollback requires transaction");

    float maximum_absolute = 0.0F;
    require_transaction(
        q4::gdn_transaction_begin(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "begin initial commit");
    for (std::size_t index = 0; index < 2u; ++index) {
        const std::vector<float> expected =
            step_host(config, host_weights, host_state, footprint, tokens[index]);
        const std::vector<float> actual = stage_cuda(
            &transaction, device_weights, &buffers, tokens[index], stream);
        maximum_absolute = std::max(
            maximum_absolute,
            compare_vectors(expected, actual, 4.0e-5F, 5.0e-5F, "initial commit output"));
    }
    require_transaction(
        q4::gdn_transaction_commit_all(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "commit initial tokens");
    const DeviceStateImage committed = capture_state(device_state, footprint, stream);
    require(committed.tokens == 2u, "initial commit token count mismatch");
    require(committed.cursor == 2u, "initial commit cursor mismatch");

    require_transaction(
        q4::gdn_transaction_begin(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "begin speculative transaction");
    q4::GdnTransactionDeviceMetadata captured_metadata{};
    require_cuda(
        cudaMemcpyAsync(
            &captured_metadata,
            transaction.snapshot_metadata,
            sizeof(captured_metadata),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "copy transaction metadata");
    stream.synchronize();
    require(
        captured_metadata.tokens_seen == committed.tokens &&
            captured_metadata.conv_cursor == committed.cursor &&
            captured_metadata.reserved == 0u,
        "device metadata snapshot mismatch");
    require_transaction(
        q4::gdn_transaction_begin(&transaction, stream.get()),
        q4::GdnTransactionStatus::kTransactionOpen,
        "reject nested begin");
    require_transaction(
        q4::gdn_transaction_begin(&transaction, wrong_stream.get()),
        q4::GdnTransactionStatus::kStreamMismatch,
        "reject different transaction stream");
    require_transaction(
        q4::gdn_transaction_state_release(&transaction),
        q4::GdnTransactionStatus::kTransactionOpen,
        "reject release while transaction is open");

    for (std::size_t index = 2u; index < 5u; ++index) {
        (void)stage_cuda(
            &transaction, device_weights, &buffers, tokens[index], stream);
    }
    const DeviceStateImage speculative = capture_state(device_state, footprint, stream);
    require(
        speculative.conv != committed.conv ||
            speculative.recurrent != committed.recurrent,
        "speculative steps did not mutate state");
    require_transaction(
        q4::gdn_transaction_commit_prefix(&transaction, 1u, stream.get()),
        q4::GdnTransactionStatus::kPartialAcceptUnsupported,
        "partial accept must fail closed");
    require(
        transaction.transaction_open && transaction.staged_steps == 3u,
        "partial accept changed transaction state");
    const DeviceStateImage after_rejected_prefix =
        capture_state(device_state, footprint, stream);
    require_byte_exact(
        speculative,
        after_rejected_prefix,
        "rejected partial accept must not mutate GDN state");
    require_transaction(
        q4::gdn_transaction_commit_prefix(&transaction, 4u, stream.get()),
        q4::GdnTransactionStatus::kInvalidArgument,
        "reject prefix beyond staged count");
    require_transaction(
        q4::gdn_transaction_rollback(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "rollback speculation");
    const DeviceStateImage rolled_back = capture_state(device_state, footprint, stream);
    require_byte_exact(committed, rolled_back, "byte-exact rollback");

    // Decoder policy for partial acceptance: rollback above, then replay the
    // accepted prefix in a fresh transaction and commit all of it.
    require_transaction(
        q4::gdn_transaction_begin(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "begin accepted-prefix replay");
    for (std::size_t index = 2u; index < 4u; ++index) {
        const std::vector<float> expected =
            step_host(config, host_weights, host_state, footprint, tokens[index]);
        const std::vector<float> actual = stage_cuda(
            &transaction, device_weights, &buffers, tokens[index], stream);
        maximum_absolute = std::max(
            maximum_absolute,
            compare_vectors(expected, actual, 4.0e-5F, 5.0e-5F, "replay output"));
    }
    require_transaction(
        q4::gdn_transaction_commit_all(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "commit replayed prefix");
    const DeviceStateImage replayed = capture_state(device_state, footprint, stream);
    require(replayed.tokens == host_tokens, "replay token count differs from host oracle");
    require(replayed.cursor == host_cursor, "replay cursor differs from host oracle");
    std::vector<float> cuda_recurrent(footprint.recurrent_state_floats, 0.0F);
    require_cuda(
        cudaMemcpyAsync(
            cuda_recurrent.data(),
            device_state.recurrent_state,
            cuda_recurrent.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "copy replayed recurrent state");
    stream.synchronize();
    maximum_absolute = std::max(
        maximum_absolute,
        compare_vectors(
            host_recurrent,
            cuda_recurrent,
            5.0e-5F,
            6.0e-5F,
            "replayed recurrent state"));

    // A malformed step is rejected before GDN mutates state and rollback still
    // recovers the committed bytes.
    const DeviceStateImage before_invalid = capture_state(device_state, footprint, stream);
    require_transaction(
        q4::gdn_transaction_begin(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "begin invalid-step probe");
    const q4::GdnStepInputs invalid_inputs{
        buffers.projected.data(), nullptr, buffers.a.data(), buffers.b.data()};
    const q4::GdnStepOutputs valid_outputs{
        buffers.convolved.data(), buffers.output.data()};
    require_transaction(
        q4::gdn_transaction_stage(
            &transaction,
            invalid_inputs,
            device_weights,
            valid_outputs,
            stream.get()),
        q4::GdnTransactionStatus::kInvalidArgument,
        "reject malformed staged step");
    require(transaction.staged_steps == 0u, "invalid stage incremented staged count");
    require_transaction(
        q4::gdn_transaction_rollback(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "rollback after malformed step");
    require_byte_exact(
        before_invalid,
        capture_state(device_state, footprint, stream),
        "malformed-step rollback");

    // Host metadata corruption is detected by commit, while rollback is still
    // permitted to recover both metadata and device bytes.
    require_transaction(
        q4::gdn_transaction_begin(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "begin metadata corruption probe");
    device_state.tokens_seen += 1u;
    require_transaction(
        q4::gdn_transaction_commit_all(&transaction, stream.get()),
        q4::GdnTransactionStatus::kStateMismatch,
        "detect metadata corruption");
    require_transaction(
        q4::gdn_transaction_rollback(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "rollback metadata corruption");
    require_byte_exact(
        before_invalid,
        capture_state(device_state, footprint, stream),
        "metadata corruption recovery");

    // Counter overflow is rejected without launching GDN.  Restore the
    // committed host metadata after validating rollback of that snapshot.
    const std::uint64_t saved_tokens = device_state.tokens_seen;
    const std::uint32_t saved_cursor = device_state.conv_cursor;
    device_state.tokens_seen = std::numeric_limits<std::uint64_t>::max();
    require_transaction(
        q4::gdn_transaction_begin(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "begin overflow probe");
    const q4::GdnStepInputs valid_inputs{
        buffers.projected.data(), buffers.z.data(), buffers.a.data(), buffers.b.data()};
    require_transaction(
        q4::gdn_transaction_stage(
            &transaction,
            valid_inputs,
            device_weights,
            valid_outputs,
            stream.get()),
        q4::GdnTransactionStatus::kSizeOverflow,
        "reject token counter overflow");
    require_transaction(
        q4::gdn_transaction_rollback(&transaction, stream.get()),
        q4::GdnTransactionStatus::kOk,
        "rollback overflow probe");
    stream.synchronize();
    require(
        device_state.tokens_seen == std::numeric_limits<std::uint64_t>::max(),
        "overflow snapshot metadata was not restored");
    device_state.tokens_seen = saved_tokens;
    device_state.conv_cursor = saved_cursor;

    require_transaction(
        q4::gdn_transaction_state_release(&transaction),
        q4::GdnTransactionStatus::kOk,
        "transaction release");
    require_transaction(
        q4::gdn_transaction_state_release(&transaction),
        q4::GdnTransactionStatus::kOk,
        "idempotent transaction release");
    require_gdn(
        q4::gdn_device_state_release(&device_state),
        q4::GdnStatus::kOk,
        "device state release");
    return maximum_absolute;
}

}  // namespace

int main() {
    try {
        const std::size_t real_snapshot_bytes = test_real_qwen4_exp_snapshot();
        const float maximum_absolute = run_transaction_test();
        std::cout << "gdn_transaction_test: OK sm=120 committed=4 rollback=byte-exact "
                  << "partial_accept=rollback+replay real_snapshot_bytes="
                  << real_snapshot_bytes << " max_cuda_abs="
                  << maximum_absolute << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gdn_transaction_test: FAIL " << error.what() << '\n';
        return 1;
    }
}
