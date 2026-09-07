#include "axiom/qwen38_spec_identity.hpp"

#include "axiom/sha256.hpp"
#include "axiom_aliced_json.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace axiom::qwen38::spec_identity {
namespace {

constexpr std::uint64_t kMaxJsonBytes = 256ull * 1024ull * 1024ull;
constexpr char kCanonicalMagic[] = "AXIOM-SPECFP1";

struct private_artifact {
    artifact_evidence evidence;
    dev_t device = 0;
    ino_t inode = 0;
    bool has_inode = false;
};

void set_error(std::string *error, const std::string &message) {
    if (error) *error = "spec identity: " + message;
}

bool fail(std::string *error, const std::string &message) {
    set_error(error, message);
    return false;
}

bool caught_failure(std::string *error) noexcept {
    try {
        set_error(error, "internal failure");
    } catch (...) {
        /* Preserve the noexcept/bool contract even under allocation failure. */
    }
    return false;
}

bool is_lower_hex64(const std::string &value) {
    if (value.size() != 64u) return false;
    for (const unsigned char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

bool is_git_object_id(const std::string &value) {
    if (value.size() != 40u && value.size() != 64u) return false;
    for (const unsigned char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

std::string u64_string(std::uint64_t value) {
    return std::to_string(value);
}

std::string u32_string(std::uint32_t value) {
    return std::to_string(static_cast<std::uint64_t>(value));
}

std::string join_path(const std::string &root, const std::string &name) {
    if (!root.empty() && root.back() == '/') return root + name;
    return root + "/" + name;
}

bool validate_root(const std::string &root, const char *role, std::string *error) {
    if (root.empty()) return fail(error, std::string(role) + " root is empty");
    struct stat status {};
    if (lstat(root.c_str(), &status) != 0) {
        return fail(error, std::string(role) + " root is missing");
    }
    if (S_ISLNK(status.st_mode)) {
        return fail(error, std::string(role) + " root is a symlink");
    }
    if (!S_ISDIR(status.st_mode)) {
        return fail(error, std::string(role) + " root is not a directory");
    }
    return true;
}

bool valid_shard_basename(const std::string &name) {
    if (name.empty() || name == "." || name == "..") return false;
    if (name.find('/') != std::string::npos ||
        name.find('\\') != std::string::npos ||
        name.find('\0') != std::string::npos) return false;
    constexpr const char suffix[] = ".safetensors";
    return name.size() > sizeof(suffix) - 1u &&
            name.compare(name.size() - (sizeof(suffix) - 1u),
                         sizeof(suffix) - 1u, suffix) == 0;
}

bool same_snapshot(const struct stat &left, const struct stat &right) {
    return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
            left.st_mode == right.st_mode && left.st_size == right.st_size &&
            left.st_mtim.tv_sec == right.st_mtim.tv_sec &&
            left.st_mtim.tv_nsec == right.st_mtim.tv_nsec &&
            left.st_ctim.tv_sec == right.st_ctim.tv_sec &&
            left.st_ctim.tv_nsec == right.st_ctim.tv_nsec;
}

bool secure_read_or_hash(
        const std::string &path,
        const std::string &logical_path,
        std::uint64_t content_limit,
        std::string *content,
        std::string *digest,
        std::uint64_t *bytes,
        dev_t *device,
        ino_t *inode,
        std::string *error) {
    struct stat path_before {};
    if (lstat(path.c_str(), &path_before) != 0) {
        if (errno == ENOENT || errno == ENOTDIR) {
            return fail(error, "required artifact missing: " + logical_path);
        }
        return fail(error, "cannot inspect artifact: " + logical_path);
    }
    if (S_ISLNK(path_before.st_mode)) {
        return fail(error, "artifact is a symlink: " + logical_path);
    }
    if (!S_ISREG(path_before.st_mode) || path_before.st_size < 0) {
        return fail(error, "artifact is not a regular file: " + logical_path);
    }
    if (content && static_cast<std::uint64_t>(path_before.st_size) > content_limit) {
        return fail(error, "JSON artifact exceeds size limit: " + logical_path);
    }

    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        if (errno == ELOOP) return fail(error, "artifact is a symlink: " + logical_path);
        return fail(error, "cannot open artifact: " + logical_path);
    }
    struct stat opened {};
    bool ok = fstat(fd, &opened) == 0 && S_ISREG(opened.st_mode) &&
            opened.st_size >= 0 && same_snapshot(path_before, opened);
    if (!ok) set_error(error, "artifact changed before hashing: " + logical_path);

    if (content && ok) {
        content->clear();
        content->reserve(static_cast<std::size_t>(opened.st_size));
    }
    crypto::sha256 hash;
    std::array<unsigned char, 64u * 1024u> buffer{};
    std::uint64_t total = 0u;
    while (ok) {
        const ssize_t count = read(fd, buffer.data(), buffer.size());
        if (count < 0) {
            if (errno == EINTR) continue;
            set_error(error, "cannot read artifact: " + logical_path);
            ok = false;
            break;
        }
        if (count == 0) break;
        const auto count_u = static_cast<std::size_t>(count);
        if (total > std::numeric_limits<std::uint64_t>::max() - count_u) {
            set_error(error, "artifact size overflow: " + logical_path);
            ok = false;
            break;
        }
        hash.update(buffer.data(), count_u);
        if (content) content->append(
                reinterpret_cast<const char *>(buffer.data()), count_u);
        total += static_cast<std::uint64_t>(count_u);
    }

    struct stat opened_after {};
    struct stat path_after {};
    if (ok && (fstat(fd, &opened_after) != 0 ||
               !same_snapshot(opened, opened_after))) {
        set_error(error, "artifact changed while hashing: " + logical_path);
        ok = false;
    }
    if (ok && (lstat(path.c_str(), &path_after) != 0 ||
               S_ISLNK(path_after.st_mode) ||
               !same_snapshot(opened, path_after))) {
        set_error(error, "artifact path changed while hashing: " + logical_path);
        ok = false;
    }
    if (ok && total != static_cast<std::uint64_t>(opened.st_size)) {
        set_error(error, "artifact size changed while hashing: " + logical_path);
        ok = false;
    }
    if (close(fd) != 0 && ok) {
        set_error(error, "cannot close artifact: " + logical_path);
        ok = false;
    }
    if (!ok) return false;

    if (digest) *digest = crypto::sha256_hex(hash.finish());
    if (bytes) *bytes = total;
    if (device) *device = opened.st_dev;
    if (inode) *inode = opened.st_ino;
    return true;
}

bool optional_regular_exists(
        const std::string &path,
        const std::string &logical_path,
        bool *exists,
        std::string *error) {
    struct stat status {};
    if (lstat(path.c_str(), &status) != 0) {
        if (errno == ENOENT || errno == ENOTDIR) {
            *exists = false;
            return true;
        }
        return fail(error, "cannot inspect optional artifact: " + logical_path);
    }
    if (S_ISLNK(status.st_mode)) {
        return fail(error, "artifact is a symlink: " + logical_path);
    }
    if (!S_ISREG(status.st_mode)) {
        return fail(error, "artifact is not a regular file: " + logical_path);
    }
    *exists = true;
    return true;
}

class strict_json_scanner {
public:
    strict_json_scanner(const char *begin, const char *end)
        : cursor_(begin), end_(end) {}

    bool scan(std::string *diagnostic) {
        if (!value(0u)) {
            if (diagnostic) *diagnostic = message_.empty() ? "invalid JSON" : message_;
            return false;
        }
        whitespace();
        if (cursor_ != end_) {
            if (diagnostic) *diagnostic = "trailing data after JSON value";
            return false;
        }
        return true;
    }

private:
    void whitespace() {
        while (cursor_ != end_ && (*cursor_ == ' ' || *cursor_ == '\t' ||
                                   *cursor_ == '\n' || *cursor_ == '\r')) ++cursor_;
    }

    bool error(const char *message) {
        if (message_.empty()) message_ = message;
        return false;
    }

    bool string(std::string *decoded) {
        ajson_parser parser(cursor_, static_cast<std::size_t>(end_ - cursor_));
        std::string ignored;
        std::string &output = decoded ? *decoded : ignored;
        if (!parser.parse_string(output)) {
            return error(parser.err.empty() ? "invalid JSON string" : parser.err.c_str());
        }
        cursor_ = parser.p;
        return true;
    }

    bool number() {
        ajson_parser parser(cursor_, static_cast<std::size_t>(end_ - cursor_));
        ajson ignored;
        if (!parser.parse_number(ignored)) {
            return error(parser.err.empty() ? "invalid JSON number" : parser.err.c_str());
        }
        cursor_ = parser.p;
        return true;
    }

    bool literal(const char *word, std::size_t length) {
        if (static_cast<std::size_t>(end_ - cursor_) < length ||
            std::memcmp(cursor_, word, length) != 0) return error("invalid JSON literal");
        cursor_ += length;
        return true;
    }

    bool value(std::uint32_t depth) {
        if (depth > ALICED_JSON_MAX_DEPTH) return error("JSON nesting too deep");
        whitespace();
        if (cursor_ == end_) return error("unexpected end of JSON");
        if (*cursor_ == '{') return object(depth + 1u);
        if (*cursor_ == '[') return array(depth + 1u);
        if (*cursor_ == '"') return string(nullptr);
        if (*cursor_ == 't') return literal("true", 4u);
        if (*cursor_ == 'f') return literal("false", 5u);
        if (*cursor_ == 'n') return literal("null", 4u);
        if (*cursor_ == '-' || (*cursor_ >= '0' && *cursor_ <= '9')) return number();
        return error("unexpected JSON character");
    }

    bool object(std::uint32_t depth) {
        ++cursor_;
        whitespace();
        if (cursor_ != end_ && *cursor_ == '}') {
            ++cursor_;
            return true;
        }
        std::set<std::string> keys;
        while (cursor_ != end_) {
            whitespace();
            std::string key;
            if (!string(&key)) return false;
            if (!keys.insert(key).second) return error("duplicate object key");
            whitespace();
            if (cursor_ == end_ || *cursor_ != ':') return error("expected ':'");
            ++cursor_;
            if (!value(depth)) return false;
            whitespace();
            if (cursor_ != end_ && *cursor_ == ',') {
                ++cursor_;
                continue;
            }
            if (cursor_ != end_ && *cursor_ == '}') {
                ++cursor_;
                return true;
            }
            return error("expected ',' or '}'");
        }
        return error("unterminated JSON object");
    }

    bool array(std::uint32_t depth) {
        ++cursor_;
        whitespace();
        if (cursor_ != end_ && *cursor_ == ']') {
            ++cursor_;
            return true;
        }
        while (cursor_ != end_) {
            if (!value(depth)) return false;
            whitespace();
            if (cursor_ != end_ && *cursor_ == ',') {
                ++cursor_;
                continue;
            }
            if (cursor_ != end_ && *cursor_ == ']') {
                ++cursor_;
                return true;
            }
            return error("expected ',' or ']'");
        }
        return error("unterminated JSON array");
    }

    const char *cursor_;
    const char *end_;
    std::string message_;
};

bool strict_json(
        const std::string &text,
        const std::string &logical_path,
        ajson *parsed,
        std::string *error) {
    strict_json_scanner scanner(text.data(), text.data() + text.size());
    std::string diagnostic;
    if (!scanner.scan(&diagnostic)) {
        return fail(error, "invalid JSON in " + logical_path + ": " + diagnostic);
    }
    if (parsed) {
        std::string parse_error;
        if (!ajson_parse(text, *parsed, parse_error)) {
            return fail(error, "invalid JSON in " + logical_path + ": " + parse_error);
        }
    }
    return true;
}

bool add_file_artifact(
        const std::string &root,
        const std::string &name,
        const std::string &component,
        const std::string &logical_path,
        bool read_content,
        std::string *content,
        std::vector<private_artifact> *artifacts,
        std::string *error) {
    private_artifact artifact;
    artifact.evidence.component = component;
    artifact.evidence.logical_path = logical_path;
    artifact.evidence.source_kind = "regular-file";
    if (!secure_read_or_hash(
                join_path(root, name), logical_path, kMaxJsonBytes,
                read_content ? content : nullptr,
                &artifact.evidence.sha256, &artifact.evidence.bytes,
                &artifact.device, &artifact.inode, error)) return false;
    artifact.has_inode = true;
    artifacts->push_back(std::move(artifact));
    return true;
}

bool add_json_artifact(
        const std::string &root,
        const std::string &name,
        const std::string &component,
        const std::string &logical_path,
        ajson *parsed,
        std::vector<private_artifact> *artifacts,
        std::string *error) {
    std::string content;
    if (!add_file_artifact(root, name, component, logical_path, true,
                           &content, artifacts, error)) return false;
    return strict_json(content, logical_path, parsed, error);
}

bool reject_inode_aliases(
        const std::vector<private_artifact> &artifacts,
        const std::string &component,
        std::string *error) {
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::string> seen;
    for (const auto &artifact : artifacts) {
        if (!artifact.has_inode || artifact.evidence.component != component) continue;
        const auto key = std::make_pair(
                static_cast<std::uint64_t>(artifact.device),
                static_cast<std::uint64_t>(artifact.inode));
        const auto inserted = seen.emplace(key, artifact.evidence.logical_path);
        if (!inserted.second && inserted.first->second != artifact.evidence.logical_path) {
            return fail(error, "incompatible duplicate artifact aliases: " +
                         inserted.first->second + " and " +
                         artifact.evidence.logical_path);
        }
    }
    return true;
}

const ajson *member(
        const ajson &object,
        const char *key,
        const std::string &where,
        std::string *error) {
    if (!object.is_object()) {
        fail(error, "expected JSON object at " + where);
        return nullptr;
    }
    const ajson *value = object.get(key);
    if (!value) fail(error, "missing field " + where + "." + key);
    return value;
}

bool expect_object(
        const ajson &object,
        const char *key,
        const std::string &where,
        const ajson **out,
        std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_object()) return fail(error, "field is not an object: " + where + "." + key);
    *out = value;
    return true;
}

bool expect_string(
        const ajson &object,
        const char *key,
        const std::string &where,
        const std::string &expected,
        std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_string() || value->s != expected) {
        return fail(error, "field mismatch: " + where + "." + key);
    }
    return true;
}

bool get_u64(
        const ajson &object,
        const char *key,
        const std::string &where,
        std::uint64_t *out,
        std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_int() || value->i < 0) {
        return fail(error, "field is not an unsigned integer: " + where + "." + key);
    }
    *out = static_cast<std::uint64_t>(value->i);
    return true;
}

bool expect_u64(
        const ajson &object,
        const char *key,
        const std::string &where,
        std::uint64_t expected,
        std::string *error) {
    std::uint64_t actual = 0u;
    if (!get_u64(object, key, where, &actual, error)) return false;
    if (actual != expected) return fail(error, "field mismatch: " + where + "." + key);
    return true;
}

bool expect_number(
        const ajson &object,
        const char *key,
        const std::string &where,
        double expected,
        std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_number() || value->number() != expected) {
        return fail(error, "field mismatch: " + where + "." + key);
    }
    return true;
}

bool expect_bool(
        const ajson &object,
        const char *key,
        const std::string &where,
        bool expected,
        std::string *error) {
    const ajson *value = member(object, key, where, error);
    if (!value) return false;
    if (!value->is_bool() || value->b != expected) {
        return fail(error, "field mismatch: " + where + "." + key);
    }
    return true;
}

bool validate_target_config(
        const ajson &root,
        const runtime_contract &runtime,
        std::string *error) {
    const ajson *text = nullptr;
    if (!expect_string(root, "model_type", "target.config", "qwen3_5", error) ||
        !expect_object(root, "text_config", "target.config", &text, error)) return false;
    if (!expect_string(*text, "model_type", "target.config.text_config", "qwen3_5_text", error) ||
        !expect_u64(*text, "hidden_size", "target.config.text_config", runtime.target_hidden_size, error) ||
        !expect_u64(*text, "intermediate_size", "target.config.text_config", runtime.target_intermediate_size, error) ||
        !expect_u64(*text, "num_hidden_layers", "target.config.text_config", runtime.target_layers, error) ||
        !expect_u64(*text, "num_attention_heads", "target.config.text_config", runtime.target_attention_heads, error) ||
        !expect_u64(*text, "num_key_value_heads", "target.config.text_config", runtime.target_key_value_heads, error) ||
        !expect_u64(*text, "head_dim", "target.config.text_config", runtime.target_head_dim, error) ||
        !expect_u64(*text, "vocab_size", "target.config.text_config", runtime.target_vocab_size, error) ||
        !expect_u64(*text, "max_position_embeddings", "target.config.text_config", runtime.target_native_context, error) ||
        !expect_u64(*text, "linear_num_key_heads", "target.config.text_config", runtime.target_linear_key_heads, error) ||
        !expect_u64(*text, "linear_key_head_dim", "target.config.text_config", runtime.target_linear_key_head_dim, error) ||
        !expect_u64(*text, "linear_num_value_heads", "target.config.text_config", runtime.target_linear_value_heads, error) ||
        !expect_u64(*text, "linear_value_head_dim", "target.config.text_config", runtime.target_linear_value_head_dim, error) ||
        !expect_u64(*text, "linear_conv_kernel_dim", "target.config.text_config", runtime.target_linear_conv_kernel_dim, error) ||
        !expect_number(*text, "rms_norm_eps", "target.config.text_config", 1.0e-6, error) ||
        !expect_number(*text, "partial_rotary_factor", "target.config.text_config", 0.25, error)) return false;

    const ajson *layers = member(*text, "layer_types", "target.config.text_config", error);
    if (!layers) return false;
    if (!layers->is_array() || layers->arr.size() != runtime.target_layers) {
        return fail(error, "field mismatch: target.config.text_config.layer_types");
    }
    for (std::size_t index = 0; index < layers->arr.size(); ++index) {
        const char *expected = (index % 4u == 3u) ? "full_attention" : "linear_attention";
        if (!layers->arr[index].is_string() || layers->arr[index].s != expected) {
            return fail(error, "field mismatch: target.config.text_config.layer_types");
        }
    }

    const ajson *rope = nullptr;
    if (!expect_object(*text, "rope_parameters", "target.config.text_config", &rope, error) ||
        !expect_string(*rope, "rope_type", "target.config.text_config.rope_parameters", "default", error) ||
        !expect_u64(*rope, "rope_theta", "target.config.text_config.rope_parameters", runtime.target_rope_theta, error) ||
        !expect_number(*rope, "partial_rotary_factor", "target.config.text_config.rope_parameters", 0.25, error) ||
        !expect_bool(*rope, "mrope_interleaved", "target.config.text_config.rope_parameters", true, error)) return false;
    const ajson *section = member(*rope, "mrope_section", "target.config.text_config.rope_parameters", error);
    if (!section || !section->is_array() || section->arr.size() != 3u ||
        !section->arr[0].is_int() || section->arr[0].i != 11 ||
        !section->arr[1].is_int() || section->arr[1].i != 11 ||
        !section->arr[2].is_int() || section->arr[2].i != 10) {
        return fail(error, "field mismatch: target.config.text_config.rope_parameters.mrope_section");
    }
    if (runtime.target_rope_dim != runtime.target_head_dim / 4u ||
        runtime.target_tap_count != runtime.target_tap_layer_ids.size()) {
        return fail(error, "invalid target runtime geometry");
    }
    return true;
}

bool validate_draft_config(
        const ajson &root,
        const runtime_contract &runtime,
        std::string *error) {
    if (!expect_string(root, "model_type", "draft.config", "qwen3", error) ||
        !expect_u64(root, "hidden_size", "draft.config", runtime.draft_hidden_size, error) ||
        !expect_u64(root, "intermediate_size", "draft.config", runtime.draft_intermediate_size, error) ||
        !expect_u64(root, "num_hidden_layers", "draft.config", runtime.draft_layers, error) ||
        !expect_u64(root, "num_attention_heads", "draft.config", runtime.draft_attention_heads, error) ||
        !expect_u64(root, "num_key_value_heads", "draft.config", runtime.draft_key_value_heads, error) ||
        !expect_u64(root, "head_dim", "draft.config", runtime.draft_head_dim, error) ||
        !expect_u64(root, "vocab_size", "draft.config", runtime.draft_vocab_size, error) ||
        !expect_u64(root, "max_position_embeddings", "draft.config", runtime.draft_max_context, error) ||
        !expect_u64(root, "block_size", "draft.config", runtime.draft_block_size, error) ||
        !expect_u64(root, "num_target_layers", "draft.config", runtime.target_layers, error) ||
        !expect_number(root, "rms_norm_eps", "draft.config", 1.0e-6, error) ||
        !expect_bool(root, "enable_confidence_head", "draft.config", true, error) ||
        !expect_bool(root, "confidence_head_with_markov", "draft.config", true, error)) return false;

    const ajson *layers = member(root, "layer_types", "draft.config", error);
    if (!layers || !layers->is_array() || layers->arr.size() != runtime.draft_layers) {
        return fail(error, "field mismatch: draft.config.layer_types");
    }
    for (const auto &layer : layers->arr) {
        if (!layer.is_string() || layer.s != "full_attention") {
            return fail(error, "field mismatch: draft.config.layer_types");
        }
    }

    const ajson *rope = nullptr;
    if (!expect_object(root, "rope_parameters", "draft.config", &rope, error) ||
        !expect_string(*rope, "rope_type", "draft.config.rope_parameters", "yarn", error) ||
        !expect_u64(*rope, "rope_theta", "draft.config.rope_parameters", runtime.draft_rope_theta, error) ||
        !expect_number(*rope, "factor", "draft.config.rope_parameters", runtime.draft_yarn_factor, error) ||
        !expect_number(*rope, "beta_fast", "draft.config.rope_parameters", runtime.draft_yarn_beta_fast, error) ||
        !expect_number(*rope, "beta_slow", "draft.config.rope_parameters", runtime.draft_yarn_beta_slow, error) ||
        !expect_u64(*rope, "original_max_position_embeddings", "draft.config.rope_parameters", runtime.draft_yarn_original_context, error)) return false;

    const ajson *dflash = nullptr;
    if (!expect_object(root, "dflash_config", "draft.config", &dflash, error) ||
        !expect_string(*dflash, "attention_mode", "draft.config.dflash_config", "gqa", error) ||
        !expect_string(*dflash, "projector_type", "draft.config.dflash_config", "dspark", error) ||
        !expect_string(*dflash, "markov_head_type", "draft.config.dflash_config", "vanilla", error) ||
        !expect_u64(*dflash, "mask_token_id", "draft.config.dflash_config", runtime.draft_mask_token_id, error) ||
        !expect_u64(*dflash, "markov_rank", "draft.config.dflash_config", runtime.draft_markov_rank, error) ||
        !expect_number(*dflash, "confidence_head_alpha", "draft.config.dflash_config", 1.0, error) ||
        !expect_bool(*dflash, "enable_confidence_head", "draft.config.dflash_config", true, error) ||
        !expect_bool(*dflash, "confidence_head_with_markov", "draft.config.dflash_config", true, error)) return false;
    const ajson *taps = member(*dflash, "target_layer_ids", "draft.config.dflash_config", error);
    if (!taps || !taps->is_array() || taps->arr.size() != runtime.target_tap_count) {
        return fail(error, "field mismatch: draft.config.dflash_config.target_layer_ids");
    }
    for (std::size_t index = 0; index < taps->arr.size(); ++index) {
        if (!taps->arr[index].is_int() || taps->arr[index].i < 0 ||
            static_cast<std::uint64_t>(taps->arr[index].i) !=
                    runtime.target_tap_layer_ids[index]) {
            return fail(error, "field mismatch: draft.config.dflash_config.target_layer_ids");
        }
    }
    if (runtime.draft_confidence_features !=
        runtime.draft_hidden_size + runtime.draft_markov_rank ||
        runtime.temporal_verify_width != runtime.draft_block_size + 1u) {
        return fail(error, "invalid draft runtime geometry");
    }
    return true;
}

void store_be32(unsigned char output[4], std::uint32_t value) {
    output[0] = static_cast<unsigned char>(value >> 24u);
    output[1] = static_cast<unsigned char>(value >> 16u);
    output[2] = static_cast<unsigned char>(value >> 8u);
    output[3] = static_cast<unsigned char>(value);
}

void store_be64(unsigned char output[8], std::uint64_t value) {
    for (unsigned index = 0u; index < 8u; ++index) {
        output[7u - index] = static_cast<unsigned char>(value >> (index * 8u));
    }
}

bool canonical_hash(
        const std::map<std::string, std::string> &records,
        std::string *digest,
        std::string *error) {
    if (!digest) return fail(error, "missing canonical digest output");
    if (records.size() > std::numeric_limits<std::uint32_t>::max()) {
        return fail(error, "too many canonical records");
    }
    crypto::sha256 hash;
    hash.update(kCanonicalMagic, sizeof(kCanonicalMagic)); /* includes NUL */
    unsigned char count[4];
    store_be32(count, static_cast<std::uint32_t>(records.size()));
    hash.update(count, sizeof(count));
    for (const auto &record : records) {
        if (record.first.size() > std::numeric_limits<std::uint32_t>::max()) {
            return fail(error, "canonical record key is too large");
        }
        unsigned char key_length[4];
        unsigned char value_length[8];
        store_be32(key_length, static_cast<std::uint32_t>(record.first.size()));
        store_be64(value_length, static_cast<std::uint64_t>(record.second.size()));
        hash.update(key_length, sizeof(key_length));
        hash.update(record.first.data(), record.first.size());
        hash.update(value_length, sizeof(value_length));
        hash.update(record.second.data(), record.second.size());
    }
    *digest = crypto::sha256_hex(hash.finish());
    return true;
}

bool component_hash(
        const std::vector<private_artifact> &artifacts,
        const std::string &component,
        std::string *digest,
        std::string *error) {
    std::map<std::string, std::string> records;
    std::uint64_t count = 0u;
    for (const auto &artifact : artifacts) {
        if (artifact.evidence.component != component) continue;
        const std::string base = "artifact/" + artifact.evidence.logical_path + "/";
        if (!records.emplace(base + "sha256", artifact.evidence.sha256).second ||
            !records.emplace(base + "bytes", u64_string(artifact.evidence.bytes)).second ||
            !records.emplace(base + "source_kind", artifact.evidence.source_kind).second) {
            return fail(error, "duplicate logical artifact: " +
                         artifact.evidence.logical_path);
        }
        ++count;
    }
    if (count == 0u) return fail(error, "component has no artifacts: " + component);
    records.emplace("component", component);
    records.emplace("artifact_count", u64_string(count));
    return canonical_hash(records, digest, error);
}

std::map<std::string, std::string> runtime_record_map(
        const runtime_contract &runtime) {
    std::map<std::string, std::string> records;
    const auto put_u32 = [&](const char *key, std::uint32_t value) {
        records.emplace(key, u32_string(value));
    };
    const auto put_u64 = [&](const char *key, std::uint64_t value) {
        records.emplace(key, u64_string(value));
    };
    const auto put = [&](const char *key, const char *value) {
        records.emplace(key, value);
    };

    put("contract.revision", "1");
    put("pipeline", "dspark-greedy-speculative");
    put("token.authority", "target-greedy-verification");
    put("failure.mode", "fail-closed-no-silent-speculative-fallback");
    put("sequence.layout", "single-sequence-temporal-m8");
    put("target.transaction", "verify-prefix-commit-or-exact-abort");
    put("target.tap.semantic", "post-block-hf-hidden-states-layer-plus-one");
    put("target.tap.layout", "f32-tap-major-[tap][time][hidden]");
    put("target.tap.value_semantic", "bf16-materialized-in-f32-container");
    put("target.tap.time_zero", "input-token-at-token-start-position");
    put("target.architecture", "qwen3_5_text-hybrid");
    put("target.layer_pattern", "linear-linear-linear-full-repeat-16");
    put("target.weight_quantization", "mixed-nvfp4-fp8-component-bound");
    put("target.kv_dtype", "e4m3fn-fixed-scale-one");
    put("target.activation_dtype", "f32-native-executor");
    put("target.rope.type", "yarn-runtime-override");
    put("target.rope.base_checkpoint_type", "default-mrope");
    put("target.rope.mrope_interleaved", "true");
    put("target.rope.mrope_section", "11,11,10");
    put("target.rope.partial_rotary", "1/4");
    put("target.rope.mscale", "1.138629436111989");
    put("draft.architecture", "qwen3-dspark-five-layer");
    put("draft.weight_dtype", "bf16");
    put("draft.activation_dtype", "f32-native-executor");
    put("draft.embedding", "borrowed-target-embedding");
    put("draft.lm_head", "borrowed-target-lm-head");
    put("draft.feature_fusion", "concat-five-target-taps-fc-rmsnorm");
    put("draft.attention", "noncausal-over-seven-token-draft-block");
    put("draft.kv", "injected-transactional-bf16");
    put("draft.proposal", "seven-token-greedy-with-markov-head");
    put("draft.confidence", "sigmoid-observational-not-token-authority");
    put("draft.rope.type", "yarn");
    put("draft.rope.mscale", "1.346573590279973");
    put("rms_norm.epsilon", "0.000001");
    put("rms_norm.zero_centered", "false");
    put("confidence_head.alpha", "1.0");

    put_u32("abi.axiom", runtime.axiom_abi_version);
    put_u32("abi.target_model_dspark", runtime.target_model_dspark_abi_version);
    put_u32("abi.target_capture", runtime.target_capture_abi_version);
    put_u32("abi.speculative_target", runtime.speculative_target_abi_version);
    put_u32("abi.dspark_compute_layout", runtime.dspark_compute_layout_version);
    put_u32("abi.dspark_compute_device", runtime.dspark_compute_device_abi_version);
    put_u32("target.tap.count", runtime.target_tap_count);
    for (std::size_t index = 0; index < runtime.target_tap_layer_ids.size(); ++index) {
        records.emplace("target.tap." + std::to_string(index) + ".post_block_layer",
                        u32_string(runtime.target_tap_layer_ids[index]));
    }
    put_u32("target.temporal_verify_width", runtime.temporal_verify_width);
    put_u32("target.hidden_size", runtime.target_hidden_size);
    put_u32("target.intermediate_size", runtime.target_intermediate_size);
    put_u32("target.layers", runtime.target_layers);
    put_u32("target.full_attention.heads", runtime.target_attention_heads);
    put_u32("target.full_attention.kv_heads", runtime.target_key_value_heads);
    put_u32("target.full_attention.head_dim", runtime.target_head_dim);
    put_u32("target.full_attention.rope_dim", runtime.target_rope_dim);
    put_u32("target.linear_attention.key_heads", runtime.target_linear_key_heads);
    put_u32("target.linear_attention.key_head_dim", runtime.target_linear_key_head_dim);
    put_u32("target.linear_attention.value_heads", runtime.target_linear_value_heads);
    put_u32("target.linear_attention.value_head_dim", runtime.target_linear_value_head_dim);
    put_u32("target.linear_attention.conv_kernel_dim", runtime.target_linear_conv_kernel_dim);
    put_u32("target.vocab_size", runtime.target_vocab_size);
    put_u32("target.context.native", runtime.target_native_context);
    put_u64("target.context.yarn_target",
            static_cast<std::uint64_t>(runtime.target_yarn_original_context) *
                    runtime.target_yarn_factor);
    put_u64("target.rope.theta", runtime.target_rope_theta);
    put_u32("target.rope.yarn_factor", runtime.target_yarn_factor);
    put_u32("target.rope.yarn_original_context", runtime.target_yarn_original_context);
    put_u32("target.rope.yarn_beta_fast", runtime.target_yarn_beta_fast);
    put_u32("target.rope.yarn_beta_slow", runtime.target_yarn_beta_slow);

    put_u32("draft.hidden_size", runtime.draft_hidden_size);
    put_u32("draft.intermediate_size", runtime.draft_intermediate_size);
    put_u32("draft.layers", runtime.draft_layers);
    put_u32("draft.attention.heads", runtime.draft_attention_heads);
    put_u32("draft.attention.kv_heads", runtime.draft_key_value_heads);
    put_u32("draft.attention.head_dim", runtime.draft_head_dim);
    put_u32("draft.vocab_size", runtime.draft_vocab_size);
    put_u32("draft.context.max", runtime.draft_max_context);
    put_u32("draft.block_size", runtime.draft_block_size);
    put_u32("draft.verify_width", runtime.temporal_verify_width);
    put_u64("draft.rope.theta", runtime.draft_rope_theta);
    put_u32("draft.rope.yarn_factor", runtime.draft_yarn_factor);
    put_u32("draft.rope.yarn_original_context", runtime.draft_yarn_original_context);
    put_u32("draft.rope.yarn_beta_fast", runtime.draft_yarn_beta_fast);
    put_u32("draft.rope.yarn_beta_slow", runtime.draft_yarn_beta_slow);
    put_u32("draft.mask_token_id", runtime.draft_mask_token_id);
    put_u32("draft.markov_rank", runtime.draft_markov_rank);
    put_u32("draft.confidence_features", runtime.draft_confidence_features);
    return records;
}

bool compute_fingerprint(
        const component_hashes &components,
        const std::vector<std::pair<std::string, std::string>> &runtime_records,
        std::string *fingerprint,
        std::string *error) {
    std::map<std::string, std::string> records;
    records.emplace("component.target.sha256", components.target_sha256);
    records.emplace("component.tokenizer.sha256", components.tokenizer_sha256);
    records.emplace("component.chat_template.sha256", components.chat_template_sha256);
    records.emplace("component.draft.sha256", components.draft_sha256);
    records.emplace("component.runtime.sha256", components.runtime_sha256);
    for (const auto &record : runtime_records) {
        if (!records.emplace("runtime/" + record.first, record.second).second) {
            return fail(error, "duplicate runtime record: " + record.first);
        }
    }
    std::string digest;
    if (!canonical_hash(records, &digest, error)) return false;
    *fingerprint = std::string(kFingerprintPrefix) + digest;
    return true;
}

bool result_is_consistent(const result &identity, std::string *error) {
    if (!is_lower_hex64(identity.components.target_sha256) ||
        !is_lower_hex64(identity.components.tokenizer_sha256) ||
        !is_lower_hex64(identity.components.chat_template_sha256) ||
        !is_lower_hex64(identity.components.draft_sha256) ||
        !is_lower_hex64(identity.components.runtime_sha256)) {
        return fail(error, "identity contains an invalid component hash");
    }
    std::string previous;
    std::map<std::string, std::string> runtime;
    for (const auto &record : identity.runtime_records) {
        if (record.first.empty() || !runtime.emplace(record.first, record.second).second) {
            return fail(error, "identity contains duplicate runtime records");
        }
        if (!previous.empty() && record.first <= previous) {
            return fail(error, "identity runtime records are not sorted");
        }
        previous = record.first;
    }
    std::string runtime_digest;
    if (!canonical_hash(runtime, &runtime_digest, error)) return false;
    if (runtime_digest != identity.components.runtime_sha256) {
        return fail(error, "identity runtime component mismatch");
    }
    std::string recomputed;
    if (!compute_fingerprint(identity.components, identity.runtime_records,
                             &recomputed, error)) return false;
    if (identity.fingerprint != recomputed) {
        return fail(error, "identity fingerprint mismatch");
    }
    return true;
}

bool exact_keys(
        const ajson &object,
        const std::vector<std::string> &expected,
        const std::string &where,
        std::string *error) {
    if (!object.is_object()) return fail(error, "expected object: " + where);
    std::vector<std::string> actual = object.keys;
    std::vector<std::string> wanted = expected;
    std::sort(actual.begin(), actual.end());
    std::sort(wanted.begin(), wanted.end());
    if (actual != wanted) return fail(error, "sidecar schema mismatch at " + where);
    return true;
}

bool exact_json_string(
        const ajson &object,
        const char *key,
        const std::string &expected,
        const std::string &where,
        std::string *error) {
    const ajson *value = object.get(key);
    if (!value || !value->is_string() || value->s != expected) {
        return fail(error, "sidecar value mismatch: " + where + "." + key);
    }
    return true;
}

ajson components_json(const component_hashes &components) {
    ajson object = ajson::jobj();
    object.set("target_sha256", ajson::jstr(components.target_sha256));
    object.set("tokenizer_sha256", ajson::jstr(components.tokenizer_sha256));
    object.set("chat_template_sha256", ajson::jstr(components.chat_template_sha256));
    object.set("draft_sha256", ajson::jstr(components.draft_sha256));
    object.set("runtime_sha256", ajson::jstr(components.runtime_sha256));
    return object;
}

ajson runtime_json(const result &identity) {
    ajson object = ajson::jobj();
    for (const auto &record : identity.runtime_records) {
        object.set(record.first, ajson::jstr(record.second));
    }
    return object;
}

[[maybe_unused]] ajson identity_json(const result &identity) {
    ajson root = ajson::jobj();
    root.set("fingerprint", ajson::jstr(identity.fingerprint));
    root.set("components", components_json(identity.components));
    root.set("runtime", runtime_json(identity));
    ajson artifacts = ajson::jarr();
    for (const auto &artifact : identity.artifacts) {
        ajson item = ajson::jobj();
        item.set("component", ajson::jstr(artifact.component));
        item.set("logical_path", ajson::jstr(artifact.logical_path));
        item.set("source_kind", ajson::jstr(artifact.source_kind));
        item.set("sha256", ajson::jstr(artifact.sha256));
        item.set("bytes", ajson::jint(static_cast<long long>(artifact.bytes)));
        artifacts.push(std::move(item));
    }
    root.set("artifacts", std::move(artifacts));
    return root;
}

}  // namespace

bool compute(
        const std::string &target_root,
        const std::string &draft_root,
        const runtime_contract &runtime,
        result *out,
        std::string *error) noexcept {
    if (error) error->clear();
    if (out) *out = result{};
    try {
        if (!out) return fail(error, "missing result output");
        if (!validate_root(target_root, "target", error) ||
            !validate_root(draft_root, "draft", error)) return false;

        std::vector<private_artifact> artifacts;
        ajson target_config;
        ajson target_index;
        ajson tokenizer_config;
        ajson draft_config;
        if (!add_json_artifact(target_root, "config.json", "target",
                               "target/config.json", &target_config,
                               &artifacts, error) ||
            !validate_target_config(target_config, runtime, error) ||
            !add_json_artifact(target_root, "hf_quant_config.json", "target",
                               "target/hf_quant_config.json", nullptr,
                               &artifacts, error) ||
            !add_json_artifact(target_root, "model.safetensors.index.json", "target",
                               "target/model.safetensors.index.json", &target_index,
                               &artifacts, error)) return false;

        const ajson *weight_map = nullptr;
        if (!expect_object(target_index, "weight_map", "target.index",
                           &weight_map, error)) return false;
        if (weight_map->keys.empty()) return fail(error, "target weight_map is empty");
        std::set<std::string> shard_names;
        for (std::size_t index = 0; index < weight_map->keys.size(); ++index) {
            if (weight_map->keys[index].empty() || !weight_map->vals[index].is_string()) {
                return fail(error, "target weight_map contains an invalid entry");
            }
            const std::string &shard = weight_map->vals[index].s;
            if (!valid_shard_basename(shard)) {
                return fail(error, "target shard path is not a basename: " + shard);
            }
            shard_names.insert(shard);
        }
        for (const auto &shard : shard_names) {
            if (!add_file_artifact(target_root, shard, "target", "target/" + shard,
                                   false, nullptr, &artifacts, error)) return false;
        }

        if (!add_json_artifact(target_root, "tokenizer.json", "tokenizer",
                               "tokenizer/tokenizer.json", nullptr,
                               &artifacts, error) ||
            !add_json_artifact(target_root, "tokenizer_config.json", "tokenizer",
                               "tokenizer/tokenizer_config.json", &tokenizer_config,
                               &artifacts, error)) return false;
        for (const char *optional : {"vocab.json", "merges.txt"}) {
            bool exists = false;
            const std::string logical = std::string("tokenizer/") + optional;
            if (!optional_regular_exists(join_path(target_root, optional), logical,
                                         &exists, error)) return false;
            if (!exists) continue;
            if (std::strcmp(optional, "vocab.json") == 0) {
                if (!add_json_artifact(target_root, optional, "tokenizer", logical,
                                       nullptr, &artifacts, error)) return false;
            } else if (!add_file_artifact(target_root, optional, "tokenizer", logical,
                                          false, nullptr, &artifacts, error)) return false;
        }

        bool template_file_exists = false;
        if (!optional_regular_exists(join_path(target_root, "chat_template.jinja"),
                                     "chat_template/chat_template.jinja",
                                     &template_file_exists, error)) return false;
        if (template_file_exists) {
            std::string template_content;
            if (!add_file_artifact(target_root, "chat_template.jinja", "chat_template",
                                   "chat_template/chat_template.jinja", true,
                                   &template_content, &artifacts, error)) return false;
            if (template_content.empty()) return fail(error, "chat template is empty");
        } else {
            const ajson *template_value = member(
                    tokenizer_config, "chat_template", "target.tokenizer_config", error);
            if (!template_value) return false;
            if (!template_value->is_string() || template_value->s.empty()) {
                return fail(error, "target.tokenizer_config.chat_template is not a non-empty string");
            }
            private_artifact artifact;
            artifact.evidence.component = "chat_template";
            artifact.evidence.logical_path =
                    "chat_template/tokenizer_config.json#chat_template";
            artifact.evidence.source_kind = "json-string";
            artifact.evidence.bytes = template_value->s.size();
            artifact.evidence.sha256 = crypto::sha256_string_hex(template_value->s);
            artifacts.push_back(std::move(artifact));
        }

        if (!add_json_artifact(draft_root, "config.json", "draft",
                               "draft/config.json", &draft_config,
                               &artifacts, error) ||
            !validate_draft_config(draft_config, runtime, error) ||
            !add_file_artifact(draft_root, "model.safetensors", "draft",
                               "draft/model.safetensors", false, nullptr,
                               &artifacts, error)) return false;

        for (const char *component : {"target", "tokenizer", "draft"}) {
            if (!reject_inode_aliases(artifacts, component, error)) return false;
        }

        result computed;
        if (!component_hash(artifacts, "target", &computed.components.target_sha256, error) ||
            !component_hash(artifacts, "tokenizer", &computed.components.tokenizer_sha256, error) ||
            !component_hash(artifacts, "draft", &computed.components.draft_sha256, error)) return false;
        for (const auto &artifact : artifacts) {
            if (artifact.evidence.component == "chat_template") {
                computed.components.chat_template_sha256 = artifact.evidence.sha256;
                break;
            }
        }
        if (!is_lower_hex64(computed.components.chat_template_sha256)) {
            return fail(error, "chat template component is missing");
        }

        const auto runtime_map = runtime_record_map(runtime);
        if (!canonical_hash(runtime_map, &computed.components.runtime_sha256, error)) return false;
        computed.runtime_records.assign(runtime_map.begin(), runtime_map.end());
        if (!compute_fingerprint(computed.components, computed.runtime_records,
                                 &computed.fingerprint, error)) return false;

        std::sort(artifacts.begin(), artifacts.end(),
                  [](const private_artifact &left, const private_artifact &right) {
                      if (left.evidence.component != right.evidence.component) {
                          return left.evidence.component < right.evidence.component;
                      }
                      return left.evidence.logical_path < right.evidence.logical_path;
                  });
        computed.artifacts.reserve(artifacts.size());
        for (const auto &artifact : artifacts) {
            computed.artifacts.push_back(artifact.evidence);
        }
        if (!result_is_consistent(computed, error)) return false;
        *out = std::move(computed);
        return true;
    } catch (...) {
        if (out) *out = result{};
        return caught_failure(error);
    }
}

bool validate_sidecar(
        const std::string &sidecar_path,
        const result &expected,
        qualification_evidence *qualification,
        std::string *error) noexcept {
    if (error) error->clear();
    if (qualification) *qualification = qualification_evidence{};
    try {
        if (!result_is_consistent(expected, error)) return false;
        if (sidecar_path.empty()) return fail(error, "sidecar missing");
        std::string text;
        std::string digest;
        std::uint64_t bytes = 0u;
        dev_t device = 0;
        ino_t inode = 0;
        std::string read_error;
        if (!secure_read_or_hash(sidecar_path, "sidecar", kMaxJsonBytes,
                                 &text, &digest, &bytes, &device, &inode,
                                 &read_error)) {
            if (read_error == "spec identity: required artifact missing: sidecar") {
                return fail(error, "sidecar missing");
            }
            if (error) *error = read_error;
            return false;
        }
        ajson root;
        if (!strict_json(text, "sidecar", &root, error)) return false;
        if (!exact_keys(root, {"schema", "fingerprint", "components", "runtime",
                               "qualification"}, "root", error)) return false;
        if (!exact_json_string(root, "schema", kSidecarSchema, "root", error) ||
            !exact_json_string(root, "fingerprint", expected.fingerprint,
                               "root", error)) return false;

        const ajson *components = root.get("components");
        if (!components ||
            !exact_keys(*components,
                        {"target_sha256", "tokenizer_sha256",
                         "chat_template_sha256", "draft_sha256", "runtime_sha256"},
                        "components", error) ||
            !exact_json_string(*components, "target_sha256",
                               expected.components.target_sha256, "components", error) ||
            !exact_json_string(*components, "tokenizer_sha256",
                               expected.components.tokenizer_sha256, "components", error) ||
            !exact_json_string(*components, "chat_template_sha256",
                               expected.components.chat_template_sha256, "components", error) ||
            !exact_json_string(*components, "draft_sha256",
                               expected.components.draft_sha256, "components", error) ||
            !exact_json_string(*components, "runtime_sha256",
                               expected.components.runtime_sha256, "components", error)) return false;

        const ajson *runtime = root.get("runtime");
        std::vector<std::string> runtime_keys;
        runtime_keys.reserve(expected.runtime_records.size());
        for (const auto &record : expected.runtime_records) runtime_keys.push_back(record.first);
        if (!runtime || !exact_keys(*runtime, runtime_keys, "runtime", error)) return false;
        for (const auto &record : expected.runtime_records) {
            if (!exact_json_string(*runtime, record.first.c_str(), record.second,
                                   "runtime", error)) return false;
        }

        const ajson *proof = root.get("qualification");
        if (!proof ||
            !exact_keys(*proof, {"status", "evidence_sha256", "source_commit"},
                        "qualification", error)) return false;
        const ajson *status = proof->get("status");
        const ajson *evidence = proof->get("evidence_sha256");
        const ajson *source = proof->get("source_commit");
        if (!status || !status->is_string() || status->s != "pass") {
            return fail(error, "sidecar qualification is not pass");
        }
        if (!evidence || !evidence->is_string() ||
            !is_lower_hex64(evidence->s)) {
            return fail(error, "sidecar qualification evidence_sha256 is invalid");
        }
        if (!source || !source->is_string() || !is_git_object_id(source->s)) {
            return fail(error, "sidecar qualification source_commit is invalid");
        }
        if (qualification) {
            qualification->status = status->s;
            qualification->evidence_sha256 = evidence->s;
            qualification->source_commit = source->s;
        }
        return true;
    } catch (...) {
        if (qualification) *qualification = qualification_evidence{};
        return caught_failure(error);
    }
}

bool build_sidecar_json(
        const result &identity,
        const std::string &source_commit,
        const std::string &evidence_sha256,
        std::string *json,
        std::string *error) noexcept {
    if (error) error->clear();
    if (json) json->clear();
    try {
        if (!json) return fail(error, "missing sidecar JSON output");
        if (!result_is_consistent(identity, error)) return false;
        if (!is_git_object_id(source_commit)) {
            return fail(error, "sidecar qualification source_commit is invalid");
        }
        if (!is_lower_hex64(evidence_sha256)) {
            return fail(error, "sidecar qualification evidence_sha256 is invalid");
        }
        ajson root = ajson::jobj();
        root.set("schema", ajson::jstr(kSidecarSchema));
        root.set("fingerprint", ajson::jstr(identity.fingerprint));
        root.set("components", components_json(identity.components));
        root.set("runtime", runtime_json(identity));
        ajson qualification = ajson::jobj();
        qualification.set("status", ajson::jstr("pass"));
        qualification.set("evidence_sha256", ajson::jstr(evidence_sha256));
        qualification.set("source_commit", ajson::jstr(source_commit));
        root.set("qualification", std::move(qualification));
        *json = ajson_dumps(root);
        json->push_back('\n');
        return true;
    } catch (...) {
        if (json) json->clear();
        return caught_failure(error);
    }
}

}  // namespace axiom::qwen38::spec_identity

#ifndef AXIOM_QWEN38_SPEC_IDENTITY_LIBRARY_ONLY
namespace {

void usage(const char *program) {
    std::fprintf(stderr,
                 "usage: %s --target ROOT --draft ROOT [--sidecar FILE]\n"
                 "       %s --target ROOT --draft ROOT --emit-sidecar SOURCE_COMMIT EVIDENCE_SHA256\n",
                 program, program);
}

}  // namespace

int main(int argc, char **argv) {
    using namespace axiom::qwen38::spec_identity;
    try {
        std::string target;
        std::string draft;
        std::string sidecar;
        std::string source_commit;
        std::string evidence_sha256;
        bool emit_sidecar = false;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if ((argument == "--target" || argument == "--draft" ||
                 argument == "--sidecar") && index + 1 < argc) {
                const std::string value = argv[++index];
                if (argument == "--target") target = value;
                else if (argument == "--draft") draft = value;
                else sidecar = value;
                continue;
            }
            if (argument == "--emit-sidecar" && index + 2 < argc) {
                emit_sidecar = true;
                source_commit = argv[++index];
                evidence_sha256 = argv[++index];
                continue;
            }
            if (argument == "--help" || argument == "-h") {
                usage(argv[0]);
                return 0;
            }
            usage(argv[0]);
            return 2;
        }
        if (target.empty() || draft.empty() || (emit_sidecar && !sidecar.empty())) {
            usage(argv[0]);
            return 2;
        }
        result identity;
        std::string error;
        if (!compute(target, draft, &identity, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        if (emit_sidecar) {
            std::string json;
            if (!build_sidecar_json(identity, source_commit, evidence_sha256,
                                    &json, &error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
            std::fwrite(json.data(), 1u, json.size(), stdout);
            return 0;
        }
        if (!sidecar.empty()) {
            qualification_evidence qualification;
            if (!validate_sidecar(sidecar, identity, &qualification, &error)) {
                std::fprintf(stderr, "%s\n", error.c_str());
                return 1;
            }
        }
        const std::string json = ajson_dumps(identity_json(identity)) + "\n";
        std::fwrite(json.data(), 1u, json.size(), stdout);
        return 0;
    } catch (...) {
        std::fprintf(stderr, "spec identity: internal failure\n");
        return 1;
    }
}
#endif
