#include "axiom/qwen4exp_admission.hpp"
#include "axiom/qwen4exp/checkpoint.hpp"

#include "axiom/sha256.hpp"
#include "axiom_aliced_json.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace axiom::qwen4exp {
namespace {

constexpr std::uint64_t kMaxConfigBytes = 1024ull * 1024ull;
constexpr std::uint64_t kMaxIndexBytes = 128ull * 1024ull * 1024ull;
constexpr std::uint64_t kMaxHeaderBytes = 64ull * 1024ull * 1024ull;

constexpr std::uint64_t kExpectedGlobal = 2;
constexpr std::uint64_t kExpectedHyper = 387;
constexpr std::uint64_t kExpectedDeltaNet = 324;
constexpr std::uint64_t kExpectedQsa = 108;
constexpr std::uint64_t kExpectedExperts = 294912;
constexpr std::uint64_t kExpectedMoeControl = 240;
constexpr std::uint64_t kExpectedPle = 138;
constexpr std::uint64_t kExpectedVision = 333;
constexpr std::uint64_t kExpectedMtp = 31;

struct file_snapshot {
    dev_t device = 0;
    ino_t inode = 0;
    off_t bytes = 0;
    timespec mtime{};
    timespec ctime{};
};

struct tensor_role {
    enum kind {
        invalid = 0,
        global,
        hyper,
        deltanet,
        qsa,
        expert,
        moe_control,
        ple,
        vision,
        mtp,
    } value = invalid;
    std::uint32_t layer = 0;
    std::uint32_t expert_id = 0;
    std::string suffix;
    bool ple_payload = false;
};

void set_error(std::string *error, const std::string &message) {
    if (error) *error = "qwen4_exp admission: " + message;
}

bool fail(std::string *error, const std::string &message) {
    set_error(error, message);
    return false;
}

std::string join_path(const std::string &root, const std::string &name) {
    return !root.empty() && root.back() == '/' ? root + name : root + "/" + name;
}

bool snapshot_equal(const file_snapshot &left, const file_snapshot &right) {
    return left.device == right.device && left.inode == right.inode &&
           left.bytes == right.bytes &&
           left.mtime.tv_sec == right.mtime.tv_sec &&
           left.mtime.tv_nsec == right.mtime.tv_nsec &&
           left.ctime.tv_sec == right.ctime.tv_sec &&
           left.ctime.tv_nsec == right.ctime.tv_nsec;
}

file_snapshot snapshot_from_stat(const struct stat &status) {
    file_snapshot out;
    out.device = status.st_dev;
    out.inode = status.st_ino;
    out.bytes = status.st_size;
    out.mtime = status.st_mtim;
    out.ctime = status.st_ctim;
    return out;
}

bool read_regular(const std::string &path,
                  std::uint64_t limit,
                  std::string *content,
                  std::string *digest,
                  std::uint64_t *bytes,
                  std::string *error) {
    struct stat before{};
    if (lstat(path.c_str(), &before) != 0) {
        return fail(error, "required file is missing: " + path);
    }
    if (S_ISLNK(before.st_mode)) return fail(error, "refusing symlink: " + path);
    if (!S_ISREG(before.st_mode) || before.st_size < 0) {
        return fail(error, "not a regular file: " + path);
    }
    if (static_cast<std::uint64_t>(before.st_size) > limit) {
        return fail(error, "file exceeds admission limit: " + path);
    }

    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return fail(error, "cannot open file: " + path);
    struct stat opened{};
    bool ok = fstat(fd, &opened) == 0 && S_ISREG(opened.st_mode) &&
              opened.st_dev == before.st_dev && opened.st_ino == before.st_ino &&
              opened.st_size == before.st_size;
    if (!ok) set_error(error, "file changed before read: " + path);

    crypto::sha256 hash;
    std::array<char, 64u * 1024u> buffer{};
    content->clear();
    content->reserve(static_cast<std::size_t>(before.st_size));
    std::uint64_t total = 0;
    while (ok) {
        const ssize_t count = read(fd, buffer.data(), buffer.size());
        if (count < 0) {
            if (errno == EINTR) continue;
            set_error(error, "read failed: " + path);
            ok = false;
            break;
        }
        if (count == 0) break;
        const auto amount = static_cast<std::size_t>(count);
        hash.update(buffer.data(), amount);
        content->append(buffer.data(), amount);
        total += amount;
    }

    struct stat after{};
    if (ok && (fstat(fd, &after) != 0 || after.st_size != before.st_size ||
               after.st_mtim.tv_sec != before.st_mtim.tv_sec ||
               after.st_mtim.tv_nsec != before.st_mtim.tv_nsec)) {
        set_error(error, "file changed during read: " + path);
        ok = false;
    }
    if (close(fd) != 0 && ok) {
        set_error(error, "close failed: " + path);
        ok = false;
    }
    if (!ok) return false;
    if (total != static_cast<std::uint64_t>(before.st_size)) {
        return fail(error, "short read: " + path);
    }
    if (digest) *digest = crypto::sha256_hex(hash.finish());
    if (bytes) *bytes = total;
    return true;
}

bool parse_json_file(const std::string &path,
                     std::uint64_t limit,
                     ajson *json,
                     std::string *digest,
                     std::string *error) {
    std::string content;
    if (!read_regular(path, limit, &content, digest, nullptr, error)) return false;
    std::string parse_error;
    if (!ajson_parse(content, *json, parse_error)) {
        return fail(error, "invalid JSON in " + path + ": " + parse_error);
    }
    return true;
}

const ajson *member(const ajson &object, const char *key,
                    const std::string &where, std::string *error) {
    if (!object.is_object()) {
        fail(error, where + " is not an object");
        return nullptr;
    }
    const ajson *value = object.get(key);
    if (!value) fail(error, where + "." + key + " is missing");
    return value;
}

bool expect_string(const ajson &object, const char *key, const char *expected,
                   const std::string &where, std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_string() || value->s != expected) {
        return fail(error, where + "." + key + " does not match the pinned contract");
    }
    return true;
}

bool expect_int(const ajson &object, const char *key, long long expected,
                const std::string &where, std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_int() || value->i != expected) {
        return fail(error, where + "." + key + " does not match the pinned contract");
    }
    return true;
}

bool expect_number(const ajson &object, const char *key, double expected,
                   const std::string &where, std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_number() || std::fabs(value->number() - expected) > 1.0e-12) {
        return fail(error, where + "." + key + " does not match the pinned contract");
    }
    return true;
}

bool expect_bool(const ajson &object, const char *key, bool expected,
                 const std::string &where, std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_bool() || value->b != expected) {
        return fail(error, where + "." + key + " does not match the pinned contract");
    }
    return true;
}

bool expect_string_array(const ajson &object, const char *key,
                         const std::vector<std::string> &expected,
                         const std::string &where, std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_array() || value->arr.size() != expected.size()) {
        return fail(error, where + "." + key + " has an unexpected array shape");
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!value->arr[i].is_string() || value->arr[i].s != expected[i]) {
            return fail(error, where + "." + key + " differs at index " +
                                      std::to_string(i));
        }
    }
    return true;
}

bool expect_int_array(const ajson &object, const char *key,
                      const std::vector<long long> &expected,
                      const std::string &where, std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_array() || value->arr.size() != expected.size()) {
        return fail(error, where + "." + key + " has an unexpected array shape");
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!value->arr[i].is_int() || value->arr[i].i != expected[i]) {
            return fail(error, where + "." + key + " differs at index " +
                                      std::to_string(i));
        }
    }
    return true;
}

bool validate_quant_object(const ajson &quant, std::string *error) {
    if (!expect_string(quant, "quant_algo", "NVFP4", "quantization", error)) return false;
    if (quant.has("quant_method") &&
        !expect_string(quant, "quant_method", "modelopt", "quantization", error)) return false;
    const ajson *producer = member(quant, "producer", "quantization", error);
    if (!producer || !expect_string(*producer, "name", "modelopt", "producer", error) ||
        !expect_string(*producer, "version", "0.46.0", "producer", error)) return false;
    return true;
}

bool validate_config(const ajson &root, std::string *error) {
    if (!expect_string(root, "model_type", "qwen4_exp", "config", error)) return false;
    if (!expect_bool(root, "language_model_only", false, "config", error)) return false;
    if (!expect_bool(root, "tie_word_embeddings", false, "config", error)) return false;
    if (!expect_int(root, "image_token_id", 248056, "config", error) ||
        !expect_int(root, "video_token_id", 248057, "config", error) ||
        !expect_int(root, "vision_start_token_id", 248053, "config", error) ||
        !expect_int(root, "vision_end_token_id", 248054, "config", error)) return false;
    if (!expect_string_array(root, "architectures",
                             {"Qwen4ExpForConditionalGeneration"}, "config", error)) return false;

    const ajson *quant = member(root, "quantization_config", "config", error);
    if (!quant || !validate_quant_object(*quant, error)) return false;
    if (!expect_string(*quant, "quant_method", "modelopt", "config.quantization_config", error)) {
        return false;
    }

    const ajson *text = member(root, "text_config", "config", error);
    if (!text) return false;
    const std::array<std::pair<const char *, long long>, 25> integer_fields{{
        {"hidden_size", 2560}, {"num_hidden_layers", 48},
        {"num_attention_heads", 24}, {"num_key_value_heads", 2},
        {"head_dim", 256}, {"vocab_size", 248320},
        {"max_position_embeddings", 262144}, {"full_attention_interval", 4},
        {"hc_count", 4}, {"hc_lowrank", 320},
        {"linear_conv_kernel_dim", 4}, {"linear_key_head_dim", 128},
        {"linear_num_key_heads", 16}, {"linear_num_value_heads", 48},
        {"linear_value_head_dim", 128}, {"num_experts", 512},
        {"num_experts_per_tok", 10}, {"moe_intermediate_size", 640},
        {"shared_expert_intermediate_size", 640}, {"indexer_budget", 2048},
        {"indexer_compress_ratio", 4}, {"indexer_head_dim", 128},
        {"indexer_kv_heads", 1}, {"indexer_n_heads", 4},
        {"mtp_num_hidden_layers", 1},
    }};
    for (const auto &field : integer_fields) {
        if (!expect_int(*text, field.first, field.second, "config.text_config", error)) return false;
    }
    if (!expect_string(*text, "model_type", "qwen4_exp_text", "config.text_config", error) ||
        !expect_string(*text, "dtype", "bfloat16", "config.text_config", error) ||
        !expect_string(*text, "mamba_ssm_dtype", "float32", "config.text_config", error) ||
        !expect_string(*text, "output_gate_type", "sigmoid", "config.text_config", error) ||
        !expect_string(*text, "ple_embedding_dtype", "float8_e4m3fn",
                       "config.text_config", error) ||
        !expect_number(*text, "rms_norm_eps", 0.000001, "config.text_config", error)) return false;
    if (!expect_int_array(*text, "ple_layer_ids", {2}, "config.text_config", error)) return false;

    std::vector<std::string> layer_types;
    layer_types.reserve(48);
    for (int layer = 0; layer < 48; ++layer) {
        layer_types.push_back((layer % 4) == 3 ? "full_attention" : "linear_attention");
    }
    if (!expect_string_array(*text, "layer_types", layer_types,
                             "config.text_config", error)) return false;

    const ajson *mtp = member(*text, "mtp", "config.text_config", error);
    if (!mtp || !expect_bool(*mtp, "hybrid", true, "config.text_config.mtp", error) ||
        !expect_int(*mtp, "num_hidden_layers", 1, "config.text_config.mtp", error) ||
        !expect_int(*mtp, "rope_theta", 10000000, "config.text_config.mtp", error) ||
        !expect_string_array(*mtp, "layer_types", {"full_attention"},
                             "config.text_config.mtp", error)) return false;

    const ajson *rope = member(*text, "rope_parameters", "config.text_config", error);
    if (!rope || !expect_string(*rope, "rope_type", "default",
                                "config.text_config.rope_parameters", error) ||
        !expect_int(*rope, "rope_theta", 10000000,
                    "config.text_config.rope_parameters", error) ||
        !expect_int_array(*rope, "mrope_section", {11, 11, 10},
                          "config.text_config.rope_parameters", error)) return false;

    const ajson *vision = member(root, "vision_config", "config", error);
    if (!vision) return false;
    const std::array<std::pair<const char *, long long>, 10> vision_fields{{
        {"depth", 27}, {"hidden_size", 1152}, {"intermediate_size", 4304},
        {"num_heads", 16}, {"out_hidden_size", 2560}, {"patch_size", 16},
        {"spatial_merge_size", 2}, {"temporal_patch_size", 2},
        {"num_position_embeddings", 2304}, {"in_channels", 3},
    }};
    for (const auto &field : vision_fields) {
        if (!expect_int(*vision, field.first, field.second, "config.vision_config", error)) {
            return false;
        }
    }
    return true;
}

bool validate_hf_quant(const ajson &root, std::string *error) {
    const ajson *producer = member(root, "producer", "hf_quant_config", error);
    if (!producer || !expect_string(*producer, "name", "modelopt", "hf_quant_config.producer", error) ||
        !expect_string(*producer, "version", "0.46.0", "hf_quant_config.producer", error)) {
        return false;
    }
    const ajson *quant = member(root, "quantization", "hf_quant_config", error);
    if (!quant || !expect_string(*quant, "quant_algo", "NVFP4", "hf_quant_config.quantization", error) ||
        !expect_int(*quant, "group_size", 16, "hf_quant_config.quantization", error)) {
        return false;
    }
    return true;
}

bool starts_with(const std::string &value, const char *prefix) {
    const std::size_t length = std::strlen(prefix);
    return value.size() >= length && value.compare(0, length, prefix) == 0;
}

bool one_of(const std::string &value, const std::vector<std::string> &choices) {
    return std::find(choices.begin(), choices.end(), value) != choices.end();
}

bool parse_index(const std::string &value, std::size_t *cursor,
                 std::uint32_t limit, std::uint32_t *result) {
    std::size_t position = *cursor;
    if (position >= value.size() || value[position] < '0' || value[position] > '9') return false;
    std::uint64_t number = 0;
    while (position < value.size() && value[position] >= '0' && value[position] <= '9') {
        number = number * 10u + static_cast<unsigned>(value[position] - '0');
        if (number >= limit) return false;
        ++position;
    }
    if (position >= value.size() || value[position] != '.') return false;
    *cursor = position + 1u;
    *result = static_cast<std::uint32_t>(number);
    return true;
}

tensor_role classify_tensor(const std::string &name) {
    tensor_role role;
    if (name == "lm_head.weight" ||
        name == "model.language_model.embed_tokens.weight") {
        role.value = tensor_role::global;
        return role;
    }
    if (one_of(name, {
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"})) {
        role.value = tensor_role::hyper;
        return role;
    }

    constexpr char layer_prefix[] = "model.language_model.layers.";
    if (starts_with(name, layer_prefix)) {
        std::size_t cursor = sizeof(layer_prefix) - 1u;
        if (!parse_index(name, &cursor, 48, &role.layer)) return role;
        const std::string suffix = name.substr(cursor);
        role.suffix = suffix;

        if (one_of(suffix, {
                "attn_hyper_connection.block_inject_weight.weight",
                "attn_hyper_connection.hc_norm.weight",
                "attn_hyper_connection.input_mix_weight_down.weight",
                "attn_hyper_connection.input_mix_weight_up.weight",
                "mlp_hyper_connection.block_inject_weight.weight",
                "mlp_hyper_connection.hc_norm.weight",
                "mlp_hyper_connection.input_mix_weight_down.weight",
                "mlp_hyper_connection.input_mix_weight_up.weight"})) {
            role.value = tensor_role::hyper;
            return role;
        }
        if (one_of(suffix, {
                "linear_attn.A_log", "linear_attn.conv1d.weight",
                "linear_attn.dt_bias", "linear_attn.in_proj_a.weight",
                "linear_attn.in_proj_b.weight", "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight", "linear_attn.norm.weight",
                "linear_attn.out_proj.weight"})) {
            if ((role.layer % 4u) != 3u) role.value = tensor_role::deltanet;
            return role;
        }
        if (one_of(suffix, {
                "self_attn.indexer.index_qk_proj.weight",
                "self_attn.indexer.k_layernorm.weight",
                "self_attn.indexer.q_layernorm.weight",
                "self_attn.k_norm.weight", "self_attn.k_proj.weight",
                "self_attn.o_proj.weight", "self_attn.q_norm.weight",
                "self_attn.q_proj.weight", "self_attn.v_proj.weight"})) {
            if ((role.layer % 4u) == 3u) role.value = tensor_role::qsa;
            return role;
        }
        if (one_of(suffix, {
                "mlp.gate.weight", "mlp.shared_expert.down_proj.weight",
                "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert_gate.weight"})) {
            role.value = tensor_role::moe_control;
            return role;
        }

        constexpr char expert_prefix[] = "mlp.experts.";
        if (starts_with(suffix, expert_prefix)) {
            std::size_t expert_cursor = sizeof(expert_prefix) - 1u;
            if (!parse_index(suffix, &expert_cursor, 512, &role.expert_id)) return tensor_role{};
            role.suffix = suffix.substr(expert_cursor);
            if (one_of(role.suffix, {
                    "down_proj.input_scale", "down_proj.weight",
                    "down_proj.weight_scale", "down_proj.weight_scale_2",
                    "gate_proj.input_scale", "gate_proj.weight",
                    "gate_proj.weight_scale", "gate_proj.weight_scale_2",
                    "up_proj.input_scale", "up_proj.weight",
                    "up_proj.weight_scale", "up_proj.weight_scale_2"})) {
                role.value = tensor_role::expert;
            }
            return role;
        }

        constexpr char ple_prefix[] = "ple.";
        if (starts_with(suffix, ple_prefix) && role.layer == 1u) {
            const std::string ple_suffix = suffix.substr(sizeof(ple_prefix) - 1u);
            role.suffix = ple_suffix;
            if (one_of(ple_suffix, {
                    "conv1d.weight", "key_proj.weight", "norm_conv.weight",
                    "norm_key.weight", "norm_query.weight",
                    "ple_embedding.layer_multipliers",
                    "ple_embedding.ngram_embedding.weight_scale",
                    "ple_embedding.ngram_heads_offsets",
                    "ple_embedding.ngram_heads_vocab_sizes",
                    "value_proj.weight"})) {
                role.value = tensor_role::ple;
                role.ple_payload = ple_suffix == "ple_embedding.ngram_embedding.weight_scale";
                return role;
            }
            constexpr char shard_prefix[] = "ple_embedding.ngram_embedding.shard_";
            if (starts_with(ple_suffix, shard_prefix)) {
                std::size_t shard_cursor = sizeof(shard_prefix) - 1u;
                std::uint32_t shard = 0;
                if (parse_index(ple_suffix, &shard_cursor, 128, &shard) &&
                    ple_suffix.substr(shard_cursor) == "weight") {
                    role.value = tensor_role::ple;
                    role.ple_payload = true;
                }
            }
            return role;
        }
        return role;
    }

    constexpr char vision_block_prefix[] = "model.visual.blocks.";
    if (starts_with(name, vision_block_prefix)) {
        std::size_t cursor = sizeof(vision_block_prefix) - 1u;
        if (!parse_index(name, &cursor, 27, &role.layer)) return role;
        role.suffix = name.substr(cursor);
        if (one_of(role.suffix, {
                "attn.proj.bias", "attn.proj.weight", "attn.qkv.bias",
                "attn.qkv.weight", "mlp.linear_fc1.bias", "mlp.linear_fc1.weight",
                "mlp.linear_fc2.bias", "mlp.linear_fc2.weight", "norm1.bias",
                "norm1.weight", "norm2.bias", "norm2.weight"})) {
            role.value = tensor_role::vision;
        }
        return role;
    }
    if (one_of(name, {
            "model.visual.merger.linear_fc1.bias", "model.visual.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc2.bias", "model.visual.merger.linear_fc2.weight",
            "model.visual.merger.norm.bias", "model.visual.merger.norm.weight",
            "model.visual.patch_embed.proj.bias", "model.visual.patch_embed.proj.weight",
            "model.visual.pos_embed.weight"})) {
        role.value = tensor_role::vision;
        return role;
    }

    if (one_of(name, {
            "mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"})) {
        role.value = tensor_role::mtp;
        return role;
    }
    constexpr char mtp_layer_prefix[] = "mtp.layers.0.";
    if (starts_with(name, mtp_layer_prefix)) {
        role.suffix = name.substr(sizeof(mtp_layer_prefix) - 1u);
        if (one_of(role.suffix, {
                "attn_hyper_connection.block_inject_weight.weight",
                "attn_hyper_connection.hc_norm.weight",
                "attn_hyper_connection.input_mix_weight_down.weight",
                "attn_hyper_connection.input_mix_weight_up.weight",
                "mlp.experts.down_proj", "mlp.experts.gate_up_proj",
                "mlp.gate.weight", "mlp.shared_expert.down_proj.weight",
                "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert_gate.weight",
                "mlp_hyper_connection.block_inject_weight.weight",
                "mlp_hyper_connection.hc_norm.weight",
                "mlp_hyper_connection.input_mix_weight_down.weight",
                "mlp_hyper_connection.input_mix_weight_up.weight",
                "self_attn.indexer.index_qk_proj.weight",
                "self_attn.indexer.k_layernorm.weight",
                "self_attn.indexer.q_layernorm.weight",
                "self_attn.k_norm.weight", "self_attn.k_proj.weight",
                "self_attn.o_proj.weight", "self_attn.q_norm.weight",
                "self_attn.q_proj.weight", "self_attn.v_proj.weight"})) {
            role.value = tensor_role::mtp;
        }
    }
    return role;
}

std::string expert_shard_name(std::uint32_t layer, std::uint32_t expert) {
    const std::uint32_t first = (expert / 128u) * 128u;
    const std::uint32_t last = first + 127u;
    char name[96];
    std::snprintf(name, sizeof(name),
                  "layer-%05u-experts-%04u-%04u.safetensors",
                  layer, first, last);
    return name;
}

std::set<std::string> expected_shards() {
    std::set<std::string> names;
    for (std::uint32_t layer = 0; layer < 48u; ++layer) {
        for (std::uint32_t group = 0; group < 4u; ++group) {
            names.insert(expert_shard_name(layer, group * 128u));
        }
    }
    names.insert("model-bf16-00001.safetensors");
    names.insert("model-bf16-00010.safetensors");
    names.insert("model-bf16-00011.safetensors");
    names.insert("model-bf16-00012.safetensors");
    for (std::uint32_t shard = 0; shard < 10u; ++shard) {
        char name[64];
        std::snprintf(name, sizeof(name), "model-plefp8-%05u.safetensors", shard);
        names.insert(name);
    }
    return names;
}

bool is_bf16_shard(const std::string &name) {
    return name == "model-bf16-00001.safetensors" ||
           name == "model-bf16-00010.safetensors" ||
           name == "model-bf16-00011.safetensors" ||
           name == "model-bf16-00012.safetensors";
}

bool is_ple_shard(const std::string &name) {
    return starts_with(name, "model-plefp8-") &&
           name.size() == std::strlen("model-plefp8-00000.safetensors");
}

void count_role(const tensor_role &role, role_counts *counts) {
    switch (role.value) {
        case tensor_role::global: ++counts->global; break;
        case tensor_role::hyper: ++counts->hyper_connection; break;
        case tensor_role::deltanet: ++counts->gated_deltanet; break;
        case tensor_role::qsa: ++counts->qsa; break;
        case tensor_role::expert: ++counts->routed_expert; break;
        case tensor_role::moe_control: ++counts->moe_control; break;
        case tensor_role::ple: ++counts->ple; break;
        case tensor_role::vision: ++counts->vision; break;
        case tensor_role::mtp: ++counts->mtp; break;
        case tensor_role::invalid: break;
    }
}

bool exact_role_counts(const role_counts &counts, std::string *error) {
    const std::array<std::tuple<const char *, std::uint64_t, std::uint64_t>, 9> checks{{
        {"global", counts.global, kExpectedGlobal},
        {"hyper_connection", counts.hyper_connection, kExpectedHyper},
        {"gated_deltanet", counts.gated_deltanet, kExpectedDeltaNet},
        {"qsa", counts.qsa, kExpectedQsa},
        {"routed_expert", counts.routed_expert, kExpectedExperts},
        {"moe_control", counts.moe_control, kExpectedMoeControl},
        {"ple", counts.ple, kExpectedPle},
        {"vision", counts.vision, kExpectedVision},
        {"mtp", counts.mtp, kExpectedMtp},
    }};
    for (const auto &check : checks) {
        if (std::get<1>(check) != std::get<2>(check)) {
            return fail(error, std::string("role count mismatch for ") + std::get<0>(check) +
                                      ": got " + std::to_string(std::get<1>(check)) +
                                      ", expected " + std::to_string(std::get<2>(check)));
        }
    }
    return true;
}

class index_stream_parser {
public:
    using visitor = std::function<bool(const std::string &, const std::string &)>;

    explicit index_stream_parser(const std::string &text)
        : cursor_(text.data()), end_(text.data() + text.size()) {}

    bool parse(const visitor &visit,
               std::uint64_t *total_size,
               std::uint64_t *tensor_count,
               std::string *error) {
        error_ = error;
        whitespace();
        if (!take('{')) return diagnostic("index root is not an object");
        bool have_metadata = false;
        bool have_weight_map = false;
        *tensor_count = 0;
        whitespace();
        if (peek('}')) return diagnostic("index root is empty");
        for (;;) {
            std::string key;
            if (!string(&key) || !colon()) return false;
            if (key == "metadata") {
                if (have_metadata) return diagnostic("duplicate index.metadata");
                have_metadata = true;
                if (!metadata(total_size)) return false;
            } else if (key == "weight_map") {
                if (have_weight_map) return diagnostic("duplicate index.weight_map");
                have_weight_map = true;
                if (!weight_map(visit, tensor_count)) return false;
            } else {
                return diagnostic("unknown index root member: " + key);
            }
            whitespace();
            if (take(',')) continue;
            if (take('}')) break;
            return diagnostic("expected ',' or '}' in index root");
        }
        whitespace();
        if (cursor_ != end_) return diagnostic("trailing data after index root");
        if (!have_metadata || !have_weight_map) {
            return diagnostic("index is missing metadata or weight_map");
        }
        return true;
    }

private:
    void whitespace() {
        while (cursor_ != end_ && (*cursor_ == ' ' || *cursor_ == '\t' ||
                                   *cursor_ == '\n' || *cursor_ == '\r')) ++cursor_;
    }

    bool peek(char token) {
        whitespace();
        return cursor_ != end_ && *cursor_ == token;
    }

    bool take(char token) {
        whitespace();
        if (cursor_ == end_ || *cursor_ != token) return false;
        ++cursor_;
        return true;
    }

    bool diagnostic(const std::string &message) {
        return fail(error_, "invalid model.safetensors.index.json: " + message);
    }

    bool string(std::string *value) {
        whitespace();
        if (cursor_ == end_) return diagnostic("unexpected end before string");
        ajson_parser parser(cursor_, static_cast<std::size_t>(end_ - cursor_));
        if (!parser.parse_string(*value)) {
            return diagnostic(parser.err.empty() ? "invalid string" : parser.err);
        }
        cursor_ = parser.p;
        return true;
    }

    bool colon() {
        if (!take(':')) return diagnostic("expected ':'");
        return true;
    }

    bool unsigned_integer(std::uint64_t *value) {
        whitespace();
        if (cursor_ == end_ || *cursor_ < '0' || *cursor_ > '9') {
            return diagnostic("expected unsigned integer");
        }
        std::uint64_t result = 0;
        while (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9') {
            const unsigned digit = static_cast<unsigned>(*cursor_ - '0');
            if (result > (std::numeric_limits<std::uint64_t>::max() - digit) / 10u) {
                return diagnostic("integer overflow");
            }
            result = result * 10u + digit;
            ++cursor_;
        }
        *value = result;
        return true;
    }

    bool metadata(std::uint64_t *total_size) {
        if (!take('{')) return diagnostic("metadata is not an object");
        bool have_total_size = false;
        whitespace();
        if (take('}')) return diagnostic("metadata is empty");
        for (;;) {
            std::string key;
            if (!string(&key) || !colon()) return false;
            if (key != "total_size") return diagnostic("unknown metadata member: " + key);
            if (have_total_size) return diagnostic("duplicate metadata.total_size");
            have_total_size = true;
            if (!unsigned_integer(total_size)) return false;
            whitespace();
            if (take(',')) continue;
            if (take('}')) break;
            return diagnostic("expected ',' or '}' in metadata");
        }
        if (!have_total_size) return diagnostic("metadata.total_size is missing");
        return true;
    }

    bool weight_map(const visitor &visit, std::uint64_t *tensor_count) {
        if (!take('{')) return diagnostic("weight_map is not an object");
        whitespace();
        if (take('}')) return diagnostic("weight_map is empty");
        for (;;) {
            std::string tensor_name;
            std::string shard_name;
            if (!string(&tensor_name) || !colon() || !string(&shard_name)) return false;
            if (!visit(tensor_name, shard_name)) return false;
            ++*tensor_count;
            whitespace();
            if (take(',')) continue;
            if (take('}')) break;
            return diagnostic("expected ',' or '}' in weight_map");
        }
        return true;
    }

    const char *cursor_;
    const char *end_;
    std::string *error_ = nullptr;
};

bool validate_index(const std::string &text,
                    admission_report *report,
                    std::unordered_map<std::string, std::vector<std::string>> *by_shard,
                    std::string *error) {
    const std::set<std::string> required_shards = expected_shards();
    std::set<std::string> observed_shards;
    std::unordered_set<std::string> observed_tensors;
    role_counts counts;
    by_shard->reserve(kShardCount);
    observed_tensors.reserve(kTensorCount);

    auto visit = [&](const std::string &tensor_name, const std::string &mapped) {
        if (!observed_tensors.insert(tensor_name).second) {
            return fail(error, "duplicate tensor in index: " + tensor_name);
        }
        if (required_shards.count(mapped) == 0u) {
            return fail(error, "tensor maps to an unknown shard: " + tensor_name);
        }
        const tensor_role role = classify_tensor(tensor_name);
        if (role.value == tensor_role::invalid) {
            return fail(error, "unknown tensor role: " + tensor_name);
        }
        if (role.value == tensor_role::expert) {
            const std::string expected = expert_shard_name(role.layer, role.expert_id);
            if (mapped != expected) {
                return fail(error, "expert tensor is mapped to the wrong shard: " + tensor_name);
            }
        } else if (role.value == tensor_role::ple && role.ple_payload) {
            if (!is_ple_shard(mapped)) {
                return fail(error, "PLE FP8 payload is mapped outside the PLE shard set: " +
                                          tensor_name);
            }
        } else if (!is_bf16_shard(mapped)) {
            return fail(error, "BF16/control tensor is mapped outside the BF16 shard set: " +
                                      tensor_name);
        }
        count_role(role, &counts);
        observed_shards.insert(mapped);
        (*by_shard)[mapped].push_back(tensor_name);
        return true;
    };

    std::uint64_t total_size = 0;
    std::uint64_t tensor_count = 0;
    index_stream_parser parser(text);
    if (!parser.parse(visit, &total_size, &tensor_count, error)) return false;
    if (total_size != kSafetensorsPhysicalBytes) {
        return fail(error, "index.metadata.total_size does not match the pinned checkpoint");
    }
    if (tensor_count != kTensorCount || observed_tensors.size() != kTensorCount) {
        return fail(error, "index tensor count does not match the pinned checkpoint");
    }

    if (observed_shards != required_shards) {
        return fail(error, "index shard set does not match the 206-shard pinned layout");
    }
    if (!exact_role_counts(counts, error)) return false;
    report->roles = counts;
    report->tensor_payload_bytes = kTensorPayloadBytes;
    report->safetensors_physical_bytes = kSafetensorsPhysicalBytes;
    report->tensor_count = tensor_count;
    report->shard_count = observed_shards.size();
    return true;
}

bool read_u64_le(const unsigned char bytes[8], std::uint64_t *value) {
    std::uint64_t out = 0;
    for (unsigned i = 0; i < 8; ++i) out |= static_cast<std::uint64_t>(bytes[i]) << (8u * i);
    *value = out;
    return true;
}

bool descriptor_offsets(const ajson &descriptor,
                        std::uint64_t *begin,
                        std::uint64_t *end,
                        std::string *error) {
    if (!descriptor.is_object()) return fail(error, "safetensors descriptor is not an object");
    const ajson *dtype = descriptor.get("dtype");
    const ajson *shape = descriptor.get("shape");
    const ajson *offsets = descriptor.get("data_offsets");
    if (!dtype || !dtype->is_string() || !shape || !shape->is_array() ||
        !offsets || !offsets->is_array() || offsets->arr.size() != 2u ||
        !offsets->arr[0].is_int() || !offsets->arr[1].is_int() ||
        offsets->arr[0].i < 0 || offsets->arr[1].i < offsets->arr[0].i) {
        return fail(error, "invalid safetensors tensor descriptor");
    }
    for (const ajson &dimension : shape->arr) {
        if (!dimension.is_int() || dimension.i < 0) {
            return fail(error, "invalid safetensors tensor shape");
        }
    }
    *begin = static_cast<std::uint64_t>(offsets->arr[0].i);
    *end = static_cast<std::uint64_t>(offsets->arr[1].i);
    return true;
}

bool descriptor_matches(const ajson &descriptor,
                        const char *expected_dtype,
                        std::initializer_list<std::uint64_t> expected_shape,
                        std::string *diagnostic) {
    const ajson *dtype = descriptor.get("dtype");
    const ajson *shape = descriptor.get("shape");
    const ajson *offsets = descriptor.get("data_offsets");
    if (!dtype || !dtype->is_string() || dtype->s != expected_dtype ||
        !shape || !shape->is_array() || shape->arr.size() != expected_shape.size() ||
        !offsets || !offsets->is_array() || offsets->arr.size() != 2u ||
        !offsets->arr[0].is_int() || !offsets->arr[1].is_int()) {
        *diagnostic = "dtype/shape does not match contract";
        return false;
    }
    std::uint64_t elements = 1;
    std::size_t index = 0;
    for (const std::uint64_t expected : expected_shape) {
        const ajson &actual = shape->arr[index++];
        if (!actual.is_int() || actual.i < 0 ||
            static_cast<std::uint64_t>(actual.i) != expected) {
            *diagnostic = "dtype/shape does not match contract";
            return false;
        }
        if (expected != 0u && elements > std::numeric_limits<std::uint64_t>::max() / expected) {
            *diagnostic = "tensor element count overflow";
            return false;
        }
        elements *= expected;
    }
    std::uint64_t scalar_bytes = 0;
    if (dtype->s == "BF16") scalar_bytes = 2;
    else if (dtype->s == "F32") scalar_bytes = 4;
    else if (dtype->s == "F8_E4M3" || dtype->s == "U8") scalar_bytes = 1;
    else if (dtype->s == "I64") scalar_bytes = 8;
    else {
        *diagnostic = "unsupported dtype in pinned contract";
        return false;
    }
    const std::uint64_t begin = static_cast<std::uint64_t>(offsets->arr[0].i);
    const std::uint64_t end = static_cast<std::uint64_t>(offsets->arr[1].i);
    if (elements > std::numeric_limits<std::uint64_t>::max() / scalar_bytes ||
        end < begin || end - begin != elements * scalar_bytes) {
        *diagnostic = "tensor byte span does not match dtype and shape";
        return false;
    }
    return true;
}

bool validate_descriptor_contract(const std::string &name,
                                  const ajson &descriptor,
                                  std::string *diagnostic) {
    const tensor_role role = classify_tensor(name);
    if (role.value == tensor_role::invalid) {
        *diagnostic = "tensor has no qwen4_exp role";
        return false;
    }
    auto match = [&](const char *dtype,
                     std::initializer_list<std::uint64_t> shape) {
        return descriptor_matches(descriptor, dtype, shape, diagnostic);
    };

    if (role.value == tensor_role::global) return match("BF16", {248320, 2560});
    if (role.value == tensor_role::hyper) {
        if (name.find("block_inject_weight") != std::string::npos) {
            return match("BF16", {4, 10240});
        }
        if (name.find("hc_norm") != std::string::npos) return match("BF16", {10240});
        if (name.find("input_mix_weight_down") != std::string::npos) {
            return match("BF16", {320, 10240});
        }
        if (name.find("input_mix_weight_up") != std::string::npos) {
            return match("BF16", {10240, 320});
        }
    }
    if (role.value == tensor_role::deltanet) {
        if (role.suffix == "linear_attn.in_proj_qkv.weight") return match("BF16", {10240, 2560});
        if (role.suffix == "linear_attn.in_proj_z.weight") return match("BF16", {6144, 2560});
        if (role.suffix == "linear_attn.in_proj_a.weight" ||
            role.suffix == "linear_attn.in_proj_b.weight") return match("BF16", {48, 2560});
        if (role.suffix == "linear_attn.conv1d.weight") return match("BF16", {10240, 1, 4});
        if (role.suffix == "linear_attn.A_log" ||
            role.suffix == "linear_attn.dt_bias") return match("BF16", {48});
        if (role.suffix == "linear_attn.norm.weight") return match("BF16", {128});
        if (role.suffix == "linear_attn.out_proj.weight") return match("BF16", {2560, 6144});
    }
    auto match_qsa_suffix = [&](const std::string &suffix) {
        if (suffix == "self_attn.indexer.index_qk_proj.weight") return match("BF16", {640, 2560});
        if (suffix == "self_attn.indexer.k_layernorm.weight" ||
            suffix == "self_attn.indexer.q_layernorm.weight") return match("BF16", {128});
        if (suffix == "self_attn.k_norm.weight" || suffix == "self_attn.q_norm.weight") {
            return match("BF16", {256});
        }
        if (suffix == "self_attn.k_proj.weight" || suffix == "self_attn.v_proj.weight") {
            return match("BF16", {512, 2560});
        }
        if (suffix == "self_attn.q_proj.weight") return match("BF16", {12288, 2560});
        if (suffix == "self_attn.o_proj.weight") return match("BF16", {2560, 6144});
        return false;
    };
    if (role.value == tensor_role::qsa) return match_qsa_suffix(role.suffix);

    if (role.value == tensor_role::expert) {
        const bool down = starts_with(role.suffix, "down_proj.");
        if (role.suffix.find(".weight") != std::string::npos &&
            role.suffix.size() >= 7u &&
            role.suffix.compare(role.suffix.size() - 7u, 7u, ".weight") == 0) {
            return down ? match("U8", {2560, 320}) : match("U8", {640, 1280});
        }
        if (role.suffix.find("weight_scale_2") != std::string::npos ||
            role.suffix.find("input_scale") != std::string::npos) return match("F32", {});
        if (role.suffix.find("weight_scale") != std::string::npos) {
            return down ? match("F8_E4M3", {2560, 40})
                        : match("F8_E4M3", {640, 160});
        }
    }
    if (role.value == tensor_role::moe_control) {
        if (role.suffix == "mlp.gate.weight") return match("BF16", {512, 2560});
        if (role.suffix == "mlp.shared_expert.down_proj.weight") return match("BF16", {2560, 640});
        if (role.suffix == "mlp.shared_expert.gate_proj.weight" ||
            role.suffix == "mlp.shared_expert.up_proj.weight") return match("BF16", {640, 2560});
        if (role.suffix == "mlp.shared_expert_gate.weight") return match("BF16", {1, 2560});
    }
    if (role.value == tensor_role::ple) {
        if (starts_with(role.suffix, "ple_embedding.ngram_embedding.shard_")) {
            return match("F8_E4M3", {2500012, 160});
        }
        if (role.suffix == "ple_embedding.ngram_embedding.weight_scale") return match("BF16", {1});
        if (role.suffix == "ple_embedding.layer_multipliers") return match("I64", {3});
        if (role.suffix == "ple_embedding.ngram_heads_offsets" ||
            role.suffix == "ple_embedding.ngram_heads_vocab_sizes") return match("I64", {16});
        if (role.suffix == "key_proj.weight") return match("BF16", {10240, 2560});
        if (role.suffix == "value_proj.weight") return match("BF16", {2560, 2560});
        if (role.suffix == "conv1d.weight") return match("BF16", {10240, 1, 4});
        if (role.suffix == "norm_conv.weight" || role.suffix == "norm_key.weight" ||
            role.suffix == "norm_query.weight") return match("BF16", {10240});
    }
    if (role.value == tensor_role::vision) {
        const std::string &suffix = role.suffix;
        if (starts_with(name, "model.visual.blocks.")) {
            if (suffix == "attn.proj.bias" || suffix == "mlp.linear_fc2.bias" ||
                suffix == "norm1.bias" || suffix == "norm1.weight" ||
                suffix == "norm2.bias" || suffix == "norm2.weight") return match("BF16", {1152});
            if (suffix == "attn.proj.weight") return match("BF16", {1152, 1152});
            if (suffix == "attn.qkv.bias") return match("BF16", {3456});
            if (suffix == "attn.qkv.weight") return match("BF16", {3456, 1152});
            if (suffix == "mlp.linear_fc1.bias") return match("BF16", {4304});
            if (suffix == "mlp.linear_fc1.weight") return match("BF16", {4304, 1152});
            if (suffix == "mlp.linear_fc2.weight") return match("BF16", {1152, 4304});
        }
        if (name == "model.visual.merger.linear_fc1.bias") return match("BF16", {4608});
        if (name == "model.visual.merger.linear_fc1.weight") return match("BF16", {4608, 4608});
        if (name == "model.visual.merger.linear_fc2.bias") return match("BF16", {2560});
        if (name == "model.visual.merger.linear_fc2.weight") return match("BF16", {2560, 4608});
        if (name == "model.visual.merger.norm.bias" || name == "model.visual.merger.norm.weight" ||
            name == "model.visual.patch_embed.proj.bias") return match("BF16", {1152});
        if (name == "model.visual.patch_embed.proj.weight") return match("BF16", {1152, 3, 2, 16, 16});
        if (name == "model.visual.pos_embed.weight") return match("BF16", {2304, 1152});
    }
    if (role.value == tensor_role::mtp) {
        if (name == "mtp.fc_embedding.weight" || name == "mtp.fc_hidden.weight") {
            return match("BF16", {2560, 2560});
        }
        if (name == "mtp.pre_fc_norm_embedding.weight") return match("BF16", {2560});
        if (name == "mtp.pre_fc_norm_hidden.weight") return match("BF16", {10240});
        if (name.find("block_inject_weight") != std::string::npos) return match("BF16", {4, 10240});
        if (name.find("hc_norm") != std::string::npos) return match("BF16", {10240});
        if (name.find("input_mix_weight_down") != std::string::npos) return match("BF16", {320, 10240});
        if (name.find("input_mix_weight_up") != std::string::npos) return match("BF16", {10240, 320});
        if (role.suffix == "mlp.experts.down_proj") return match("BF16", {512, 2560, 640});
        if (role.suffix == "mlp.experts.gate_up_proj") return match("BF16", {512, 1280, 2560});
        if (role.suffix == "mlp.gate.weight") return match("BF16", {512, 2560});
        if (role.suffix == "mlp.shared_expert.down_proj.weight") return match("BF16", {2560, 640});
        if (role.suffix == "mlp.shared_expert.gate_proj.weight" ||
            role.suffix == "mlp.shared_expert.up_proj.weight") return match("BF16", {640, 2560});
        if (role.suffix == "mlp.shared_expert_gate.weight") return match("BF16", {1, 2560});
        if (starts_with(role.suffix, "self_attn.")) return match_qsa_suffix(role.suffix);
    }
    *diagnostic = "role has no exact dtype/shape contract";
    return false;
}

enum class shard_state { valid, missing, incomplete, invalid };

using shard_descriptor_callback = std::function<bool(
        const std::string &, const ajson &, std::uint64_t, std::string *)>;

shard_state validate_shard(const std::string &path,
                           const std::vector<std::string> &expected_tensors,
                           std::uint64_t *physical_bytes,
                           std::string *diagnostic,
                           const shard_descriptor_callback &callback) {
    struct stat path_status{};
    if (lstat(path.c_str(), &path_status) != 0) {
        if (errno == ENOENT || errno == ENOTDIR) return shard_state::missing;
        *diagnostic = "cannot inspect shard";
        return shard_state::invalid;
    }
    if (S_ISLNK(path_status.st_mode) || !S_ISREG(path_status.st_mode) ||
        path_status.st_size < 8) {
        *diagnostic = "shard is not a stable regular safetensors file";
        return shard_state::invalid;
    }
    const file_snapshot before = snapshot_from_stat(path_status);
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        *diagnostic = "cannot open shard";
        return shard_state::invalid;
    }
    unsigned char prefix[8]{};
    ssize_t got = pread(fd, prefix, sizeof(prefix), 0);
    if (got != static_cast<ssize_t>(sizeof(prefix))) {
        close(fd);
        *diagnostic = "short safetensors prefix";
        return shard_state::incomplete;
    }
    std::uint64_t header_bytes = 0;
    read_u64_le(prefix, &header_bytes);
    if (header_bytes == 0 || header_bytes > kMaxHeaderBytes ||
        header_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        close(fd);
        *diagnostic = "invalid safetensors header length";
        return shard_state::invalid;
    }
    if (static_cast<std::uint64_t>(path_status.st_size) < 8u + header_bytes) {
        close(fd);
        *diagnostic = "safetensors header is still incomplete";
        return shard_state::incomplete;
    }
    std::string header(static_cast<std::size_t>(header_bytes), '\0');
    std::size_t offset = 0;
    while (offset < header.size()) {
        const ssize_t count = pread(fd, header.data() + offset, header.size() - offset,
                                    static_cast<off_t>(8u + offset));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            close(fd);
            *diagnostic = "short safetensors header read";
            return shard_state::incomplete;
        }
        offset += static_cast<std::size_t>(count);
    }
    struct stat after_status{};
    const bool stable = fstat(fd, &after_status) == 0 &&
                        snapshot_equal(before, snapshot_from_stat(after_status));
    close(fd);
    if (!stable) {
        *diagnostic = "shard changed during admission";
        return shard_state::incomplete;
    }

    ajson parsed;
    std::string parse_error;
    if (!ajson_parse(header, parsed, parse_error) || !parsed.is_object()) {
        *diagnostic = "invalid safetensors header JSON: " + parse_error;
        return shard_state::invalid;
    }
    std::unordered_set<std::string> expected(expected_tensors.begin(), expected_tensors.end());
    std::uint64_t max_end = 0;
    std::size_t tensor_entries = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    spans.reserve(expected_tensors.size());
    for (std::size_t i = 0; i < parsed.keys.size(); ++i) {
        const std::string &name = parsed.keys[i];
        if (name == "__metadata__") continue;
        if (expected.erase(name) == 0u) {
            *diagnostic = "unindexed tensor in shard: " + name;
            return shard_state::invalid;
        }
        std::string contract_error;
        if (!validate_descriptor_contract(name, parsed.vals[i], &contract_error)) {
            *diagnostic = contract_error + ": " + name;
            return shard_state::invalid;
        }
        std::uint64_t begin = 0, end = 0;
        std::string descriptor_error;
        if (!descriptor_offsets(parsed.vals[i], &begin, &end, &descriptor_error)) {
            *diagnostic = descriptor_error + ": " + name;
            return shard_state::invalid;
        }
        if (callback && !callback(name, parsed.vals[i], 8u + header_bytes,
                                  &descriptor_error)) {
            *diagnostic = descriptor_error + ": " + name;
            return shard_state::invalid;
        }
        spans.emplace_back(begin, end);
        max_end = std::max(max_end, end);
        ++tensor_entries;
    }
    if (!expected.empty() || tensor_entries != expected_tensors.size()) {
        *diagnostic = "safetensors header does not match index membership";
        return shard_state::invalid;
    }
    std::sort(spans.begin(), spans.end());
    std::uint64_t cursor = 0;
    for (const auto &span : spans) {
        if (span.first != cursor) {
            *diagnostic = "safetensors data offsets are not contiguous";
            return shard_state::invalid;
        }
        cursor = span.second;
    }
    const std::uint64_t expected_file_bytes = 8u + header_bytes + max_end;
    *physical_bytes = static_cast<std::uint64_t>(path_status.st_size);
    if (*physical_bytes < expected_file_bytes) {
        *diagnostic = "shard payload is still downloading";
        return shard_state::incomplete;
    }
    if (*physical_bytes != expected_file_bytes) {
        *diagnostic = "shard size exceeds safetensors header contract";
        return shard_state::invalid;
    }
    return shard_state::valid;
}

bool hash_weight_manifest(const std::string &model_root,
                          const std::set<std::string> &shard_set,
                          std::string *manifest_sha256,
                          std::string *error) {
    struct hash_result {
        std::string name;
        std::string digest;
        std::uint64_t bytes = 0;
        std::string error;
        bool valid = false;
    };

    std::vector<hash_result> results;
    results.reserve(shard_set.size());
    for (const std::string &name : shard_set) {
        hash_result result;
        result.name = name;
        results.push_back(std::move(result));
    }

    std::atomic<std::size_t> next{0};
    const unsigned available = std::thread::hardware_concurrency();
    const unsigned workers = std::max(1u, std::min(4u, available == 0u ? 4u : available));
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned worker = 0; worker < workers; ++worker) {
        pool.emplace_back([&]() {
            for (;;) {
                const std::size_t index = next.fetch_add(1u);
                if (index >= results.size()) break;
                hash_result &result = results[index];
                result.valid = crypto::sha256_file_hex(
                        join_path(model_root, result.name), &result.digest,
                        &result.bytes, &result.error);
            }
        });
    }
    for (std::thread &worker : pool) worker.join();

    std::string canonical;
    canonical.reserve(results.size() * 128u);
    for (const hash_result &result : results) {
        if (!result.valid) {
            return fail(error, "cannot hash " + result.name + ": " + result.error);
        }
        canonical += result.name;
        canonical += '\t';
        canonical += std::to_string(result.bytes);
        canonical += '\t';
        canonical += result.digest;
        canonical += '\n';
    }
    *manifest_sha256 = crypto::sha256_string_hex(canonical);
    return true;
}

ajson string_array_json(const std::vector<std::string> &values) {
    ajson array = ajson::jarr();
    for (const std::string &value : values) array.push(ajson::jstr(value));
    return array;
}

ajson role_counts_json(const role_counts &counts) {
    ajson result = ajson::jobj();
    result.set("global", ajson::jint(static_cast<long long>(counts.global)));
    result.set("hyper_connection", ajson::jint(static_cast<long long>(counts.hyper_connection)));
    result.set("gated_deltanet", ajson::jint(static_cast<long long>(counts.gated_deltanet)));
    result.set("qsa", ajson::jint(static_cast<long long>(counts.qsa)));
    result.set("routed_expert", ajson::jint(static_cast<long long>(counts.routed_expert)));
    result.set("moe_control", ajson::jint(static_cast<long long>(counts.moe_control)));
    result.set("ple", ajson::jint(static_cast<long long>(counts.ple)));
    result.set("vision", ajson::jint(static_cast<long long>(counts.vision)));
    result.set("mtp", ajson::jint(static_cast<long long>(counts.mtp)));
    return result;
}

}  // namespace

bool inspect_checkpoint(const std::string &model_root,
                        bool metadata_only,
                        admission_report *report,
                        std::string *error) noexcept {
    if (!report) return fail(error, "null report");
    *report = admission_report{};
    report->model_root = model_root;
    try {
        struct stat root_status{};
        if (model_root.empty() || lstat(model_root.c_str(), &root_status) != 0 ||
            S_ISLNK(root_status.st_mode) || !S_ISDIR(root_status.st_mode)) {
            return fail(error, "model root is not a real directory");
        }

        ajson config, quant;
        std::string index_text;
        std::string config_hash, quant_hash, index_hash, tokenizer_hash;
        if (!parse_json_file(join_path(model_root, "config.json"), kMaxConfigBytes,
                             &config, &config_hash, error) ||
            !parse_json_file(join_path(model_root, "hf_quant_config.json"), kMaxConfigBytes,
                             &quant, &quant_hash, error) ||
            !read_regular(join_path(model_root, "model.safetensors.index.json"),
                          kMaxIndexBytes, &index_text, &index_hash, nullptr, error)) return false;
        std::uint64_t tokenizer_bytes = 0;
        if (!crypto::sha256_file_hex(join_path(model_root, "tokenizer.json"),
                                     &tokenizer_hash, &tokenizer_bytes, error)) {
            return fail(error, "cannot hash tokenizer.json: " + (error ? *error : std::string{}));
        }
        report->metadata_sha256 = {
            {"config.json", config_hash},
            {"hf_quant_config.json", quant_hash},
            {"model.safetensors.index.json", index_hash},
            {"tokenizer.json", tokenizer_hash},
        };
        report->identity_valid = config_hash == kConfigSha256 &&
                                 quant_hash == kQuantConfigSha256 &&
                                 index_hash == kIndexSha256 &&
                                 tokenizer_hash == kTokenizerSha256;
        if (!report->identity_valid) {
            return fail(error, "metadata hash differs from pinned revision " +
                                      std::string(kRevision));
        }

        if (!validate_config(config, error)) return false;
        report->config_valid = true;
        if (!validate_hf_quant(quant, error)) return false;
        report->quantization_valid = true;

        std::unordered_map<std::string, std::vector<std::string>> by_shard;
        if (!validate_index(index_text, report, &by_shard, error)) return false;
        report->tensor_contract_valid = true;
        report->metadata_admitted = true;

        const std::set<std::string> shards = expected_shards();
        for (const std::string &name : shards) {
            std::uint64_t physical_bytes = 0;
            std::string diagnostic;
            const shard_state state = validate_shard(join_path(model_root, name),
                                                     by_shard.at(name),
                                                     &physical_bytes, &diagnostic, {});
            if (state == shard_state::valid) {
                ++report->present_shards;
                report->present_shard_bytes += physical_bytes;
            } else if (state == shard_state::missing) {
                report->missing_shards.push_back(name);
            } else if (state == shard_state::incomplete) {
                report->incomplete_shards.push_back(name);
                report->diagnostics.push_back(name + ": " + diagnostic);
            } else {
                report->diagnostics.push_back(name + ": " + diagnostic);
                if (!metadata_only) return fail(error, name + ": " + diagnostic);
            }
        }
        report->checkpoint_complete = report->present_shards == kShardCount &&
                                      report->missing_shards.empty() &&
                                      report->incomplete_shards.empty() &&
                                      report->diagnostics.empty();
        if (!metadata_only && !report->checkpoint_complete) {
            return fail(error, "checkpoint is incomplete: " +
                                      std::to_string(report->present_shards) + "/" +
                                      std::to_string(kShardCount) + " shards admitted");
        }
        if (!metadata_only) {
            if (!hash_weight_manifest(model_root, shards,
                                      &report->weight_manifest_sha256, error)) return false;
            report->payload_hashes_valid =
                    report->weight_manifest_sha256 == kWeightManifestSha256;
            if (!report->payload_hashes_valid) {
                return fail(error, "weight manifest hash differs from pinned LFS objects");
            }
            report->checkpoint_admitted = report->metadata_admitted &&
                                          report->checkpoint_complete &&
                                          report->payload_hashes_valid;
        }
        return true;
    } catch (const std::exception &exception) {
        return fail(error, std::string("internal exception: ") + exception.what());
    } catch (...) {
        return fail(error, "internal exception");
    }
}

std::string report_json(const admission_report &report) {
    ajson root = ajson::jobj();
    root.set("schema", ajson::jstr("axiom.qwen4_exp.admission.v1"));
    root.set("repository", ajson::jstr(report.repository));
    root.set("revision", ajson::jstr(report.revision));
    root.set("model_root", ajson::jstr(report.model_root));
    root.set("identity_valid", ajson::jbool(report.identity_valid));
    root.set("config_valid", ajson::jbool(report.config_valid));
    root.set("quantization_valid", ajson::jbool(report.quantization_valid));
    root.set("tensor_contract_valid", ajson::jbool(report.tensor_contract_valid));
    root.set("metadata_admitted", ajson::jbool(report.metadata_admitted));
    root.set("checkpoint_complete", ajson::jbool(report.checkpoint_complete));
    root.set("payload_hashes_valid", ajson::jbool(report.payload_hashes_valid));
    root.set("checkpoint_admitted", ajson::jbool(report.checkpoint_admitted));
    root.set("native_runtime_ready", ajson::jbool(report.native_runtime_ready));
    root.set("tensor_payload_bytes", ajson::jint(static_cast<long long>(report.tensor_payload_bytes)));
    root.set("safetensors_physical_bytes",
             ajson::jint(static_cast<long long>(report.safetensors_physical_bytes)));
    root.set("tensor_count", ajson::jint(static_cast<long long>(report.tensor_count)));
    root.set("shard_count", ajson::jint(static_cast<long long>(report.shard_count)));
    root.set("present_shards", ajson::jint(static_cast<long long>(report.present_shards)));
    root.set("present_shard_bytes", ajson::jint(static_cast<long long>(report.present_shard_bytes)));
    root.set("weight_manifest_sha256", report.weight_manifest_sha256.empty()
            ? ajson::jnull() : ajson::jstr(report.weight_manifest_sha256));
    root.set("roles", role_counts_json(report.roles));
    ajson hashes = ajson::jobj();
    for (const auto &item : report.metadata_sha256) hashes.set(item.first, ajson::jstr(item.second));
    root.set("metadata_sha256", std::move(hashes));
    root.set("missing_shards", string_array_json(report.missing_shards));
    root.set("incomplete_shards", string_array_json(report.incomplete_shards));
    root.set("diagnostics", string_array_json(report.diagnostics));
    return ajson_dumps(root);
}

struct checkpoint_catalog::impl {
    std::string root;
    std::vector<tensor_span> tensors;
    std::unordered_map<std::string, std::size_t> lookup;
    std::unordered_map<std::string, file_snapshot> shard_snapshots;
};

checkpoint_catalog::checkpoint_catalog() : impl_(std::make_unique<impl>()) {}
checkpoint_catalog::~checkpoint_catalog() = default;
checkpoint_catalog::checkpoint_catalog(checkpoint_catalog &&) noexcept = default;
checkpoint_catalog &checkpoint_catalog::operator=(checkpoint_catalog &&) noexcept = default;

bool checkpoint_catalog::open(const std::string &model_root,
                              std::unique_ptr<checkpoint_catalog> *out,
                              std::string *error) noexcept {
    if (!out) return fail(error, "checkpoint catalog output is null");
    out->reset();
    try {
        admission_report gate;
        if (!inspect_checkpoint(model_root, true, &gate, error)) return false;
        if (!gate.metadata_admitted || !gate.checkpoint_complete) {
            return fail(error, "checkpoint catalog requires a complete structural admission");
        }

        std::string index_text;
        std::string index_hash;
        if (!read_regular(join_path(model_root, "model.safetensors.index.json"),
                          kMaxIndexBytes, &index_text, &index_hash, nullptr, error)) return false;
        if (index_hash != kIndexSha256) return fail(error, "checkpoint index identity changed");
        admission_report index_report;
        std::unordered_map<std::string, std::vector<std::string>> by_shard;
        if (!validate_index(index_text, &index_report, &by_shard, error)) return false;

        auto catalog = std::make_unique<checkpoint_catalog>();
        catalog->impl_->root = model_root;
        catalog->impl_->tensors.reserve(kTensorCount);
        catalog->impl_->lookup.reserve(kTensorCount);
        catalog->impl_->shard_snapshots.reserve(kShardCount);

        for (const std::string &shard : expected_shards()) {
            shard_descriptor_callback callback = [&](const std::string &name,
                                                     const ajson &descriptor,
                                                     std::uint64_t data_begin,
                                                     std::string *callback_error) {
                const ajson *dtype = descriptor.get("dtype");
                const ajson *shape = descriptor.get("shape");
                const ajson *offsets = descriptor.get("data_offsets");
                if (!dtype || !dtype->is_string() || !shape || !shape->is_array() ||
                    shape->arr.size() > 5u || !offsets || !offsets->is_array() ||
                    offsets->arr.size() != 2u || !offsets->arr[0].is_int() ||
                    !offsets->arr[1].is_int() || offsets->arr[0].i < 0 ||
                    offsets->arr[1].i < offsets->arr[0].i) {
                    if (callback_error) *callback_error = "invalid catalog descriptor";
                    return false;
                }
                tensor_span span;
                span.name = name;
                span.shard = shard;
                if (dtype->s == "BF16") span.dtype = tensor_dtype::bf16;
                else if (dtype->s == "F32") span.dtype = tensor_dtype::f32;
                else if (dtype->s == "F8_E4M3") span.dtype = tensor_dtype::f8_e4m3;
                else if (dtype->s == "U8") span.dtype = tensor_dtype::u8;
                else if (dtype->s == "I64") span.dtype = tensor_dtype::i64;
                else {
                    if (callback_error) *callback_error = "unsupported catalog dtype";
                    return false;
                }
                span.rank = static_cast<std::uint32_t>(shape->arr.size());
                for (std::size_t i = 0; i < shape->arr.size(); ++i) {
                    if (!shape->arr[i].is_int() || shape->arr[i].i < 0) {
                        if (callback_error) *callback_error = "invalid catalog shape";
                        return false;
                    }
                    span.shape[i] = static_cast<std::uint64_t>(shape->arr[i].i);
                }
                const std::uint64_t relative = static_cast<std::uint64_t>(offsets->arr[0].i);
                const std::uint64_t end = static_cast<std::uint64_t>(offsets->arr[1].i);
                if (data_begin > std::numeric_limits<std::uint64_t>::max() - relative) {
                    if (callback_error) *callback_error = "catalog file offset overflow";
                    return false;
                }
                span.file_offset = data_begin + relative;
                span.bytes = end - relative;
                const std::size_t position = catalog->impl_->tensors.size();
                if (!catalog->impl_->lookup.emplace(span.name, position).second) {
                    if (callback_error) *callback_error = "duplicate catalog tensor";
                    return false;
                }
                catalog->impl_->tensors.push_back(std::move(span));
                return true;
            };
            std::uint64_t physical_bytes = 0;
            std::string diagnostic;
            const std::string path = join_path(model_root, shard);
            if (validate_shard(path, by_shard.at(shard), &physical_bytes,
                               &diagnostic, callback) != shard_state::valid) {
                return fail(error, "cannot catalog " + shard + ": " + diagnostic);
            }
            struct stat status{};
            if (lstat(path.c_str(), &status) != 0 || S_ISLNK(status.st_mode) ||
                !S_ISREG(status.st_mode) || status.st_size < 0 ||
                static_cast<std::uint64_t>(status.st_size) != physical_bytes) {
                return fail(error, "shard changed after catalog scan: " + shard);
            }
            catalog->impl_->shard_snapshots.emplace(shard, snapshot_from_stat(status));
        }
        if (catalog->impl_->tensors.size() != kTensorCount ||
            catalog->impl_->lookup.size() != kTensorCount ||
            catalog->impl_->shard_snapshots.size() != kShardCount) {
            return fail(error, "checkpoint catalog cardinality mismatch");
        }
        *out = std::move(catalog);
        return true;
    } catch (const std::exception &exception) {
        return fail(error, std::string("checkpoint catalog exception: ") + exception.what());
    } catch (...) {
        return fail(error, "checkpoint catalog exception");
    }
}

const tensor_span *checkpoint_catalog::find(const std::string &name) const noexcept {
    if (!impl_) return nullptr;
    const auto found = impl_->lookup.find(name);
    return found == impl_->lookup.end() ? nullptr : &impl_->tensors[found->second];
}

const tensor_span *checkpoint_catalog::at(std::size_t index) const noexcept {
    return impl_ && index < impl_->tensors.size() ? &impl_->tensors[index] : nullptr;
}

std::size_t checkpoint_catalog::tensor_count() const noexcept {
    return impl_ ? impl_->tensors.size() : 0u;
}

const std::string &checkpoint_catalog::model_root() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->root : empty;
}

bool checkpoint_catalog::read_range(const std::string &name,
                                    std::uint64_t tensor_byte_offset,
                                    void *destination,
                                    std::size_t bytes,
                                    std::string *error) const noexcept {
    if (!impl_ || !destination || bytes == 0u) {
        return fail(error, "invalid checkpoint range read arguments");
    }
    const tensor_span *span = find(name);
    if (!span) return fail(error, "tensor is not in checkpoint catalog: " + name);
    const std::uint64_t amount = static_cast<std::uint64_t>(bytes);
    if (tensor_byte_offset > span->bytes || amount > span->bytes - tensor_byte_offset ||
        span->file_offset > std::numeric_limits<std::uint64_t>::max() - tensor_byte_offset) {
        return fail(error, "checkpoint range is outside tensor: " + name);
    }
    const std::uint64_t absolute = span->file_offset + tensor_byte_offset;
    if (absolute > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        return fail(error, "checkpoint range exceeds platform offset: " + name);
    }
    const auto expected_it = impl_->shard_snapshots.find(span->shard);
    if (expected_it == impl_->shard_snapshots.end()) {
        return fail(error, "catalog shard identity is missing: " + span->shard);
    }
    const std::string path = join_path(impl_->root, span->shard);
    struct stat path_status{};
    if (lstat(path.c_str(), &path_status) != 0 || S_ISLNK(path_status.st_mode) ||
        !S_ISREG(path_status.st_mode) ||
        !snapshot_equal(expected_it->second, snapshot_from_stat(path_status))) {
        return fail(error, "checkpoint shard identity changed: " + span->shard);
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return fail(error, "cannot open checkpoint shard: " + span->shard);
    std::size_t complete = 0;
    bool ok = true;
    while (complete < bytes) {
        const std::size_t chunk = std::min<std::size_t>(bytes - complete, 1u << 30u);
        const ssize_t count = pread(fd, static_cast<unsigned char *>(destination) + complete,
                                    chunk, static_cast<off_t>(absolute + complete));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            set_error(error, "short checkpoint range read: " + name);
            ok = false;
            break;
        }
        complete += static_cast<std::size_t>(count);
    }
    struct stat after{};
    if (ok && (fstat(fd, &after) != 0 ||
               !snapshot_equal(expected_it->second, snapshot_from_stat(after)))) {
        set_error(error, "checkpoint shard changed during range read: " + span->shard);
        ok = false;
    }
    if (close(fd) != 0 && ok) {
        set_error(error, "checkpoint shard close failed: " + span->shard);
        ok = false;
    }
    return ok;
}

}  // namespace axiom::qwen4exp
