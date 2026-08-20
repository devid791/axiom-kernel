#include "axiom_remote_span_wire.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct TensorInfo {
    std::string name;
    int layer = -1;
    uint64_t bytes = 0;
};

struct InventoryNode {
    std::string name;
    std::string endpoint;
    uint64_t vram_bytes = 0;
};

struct SpanPlan {
    uint32_t start = 0;
    uint32_t end = 0;
    uint64_t bytes = 0;
};

struct ModelMeta {
    std::string format;
    std::string architecture;
    std::vector<TensorInfo> tensors;
    std::vector<uint64_t> layer_bytes;
    uint64_t whole_bytes = 0;
    uint32_t layer_count = 0;
};

struct PlanNode {
    std::string name;
    std::string endpoint;
    SpanPlan span;
    std::string kv_owner;
};

struct ParsedPlan {
    std::string model_id;
    std::string model_sha;
    std::vector<PlanNode> nodes;
    std::vector<SpanPlan> local_layers;
};

struct Options {
    std::string model_id = "Qwen";
    std::string model_path;
    std::string inventory_path;
    std::string validate_plan;
    uint32_t node_index = 1;
    uint32_t node_count = 2;
};

static const char *arg_value(int argc, char **argv, const char *name, const char *fallback = nullptr) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return fallback;
}

static bool has_arg(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return true;
    }
    return false;
}

static uint32_t arg_u32(int argc, char **argv, const char *name, uint32_t fallback) {
    const char *v = arg_value(argc, argv, name);
    if (!v) return fallback;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(v, &end, 10);
    return end && *end == '\0' ? (uint32_t)parsed : fallback;
}

static std::string read_text(const fs::path &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("open_failed:" + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void json_string(const std::string &s) {
    std::putchar('"');
    for (const unsigned char ch : s) {
        switch (ch) {
        case '\\': std::fputs("\\\\", stdout); break;
        case '"': std::fputs("\\\"", stdout); break;
        case '\n': std::fputs("\\n", stdout); break;
        case '\r': std::fputs("\\r", stdout); break;
        case '\t': std::fputs("\\t", stdout); break;
        default:
            if (ch < 32) std::printf("\\u%04x", (unsigned)ch);
            else std::putchar((int)ch);
            break;
        }
    }
    std::putchar('"');
}

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return s;
}

static std::string trim_copy(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

static int extract_layer(const std::string &name) {
    struct Pattern {
        std::regex regex;
        size_t group;
    };
    static const Pattern patterns[] = {
        {std::regex("(^|\\.)blk\\.(\\d+)\\."), 2},
        {std::regex("(^|\\.)layers\\.(\\d+)\\."), 2},
        {std::regex("model\\.language_model\\.layers\\.(\\d+)\\."), 1},
        {std::regex("model\\.layers\\.(\\d+)\\."), 1},
    };
    for (const Pattern &pattern : patterns) {
        std::smatch match;
        if (std::regex_search(name, match, pattern.regex) && match.size() > pattern.group) {
            return std::atoi(match[pattern.group].str().c_str());
        }
    }
    return -1;
}

static uint32_t rd_u32(const std::vector<uint8_t> &buf, size_t &pos) {
    if (pos + 4 > buf.size()) throw std::runtime_error("truncated_u32");
    uint32_t v = 0;
    std::memcpy(&v, buf.data() + pos, 4);
    pos += 4;
    return v;
}

static uint64_t rd_u64(const std::vector<uint8_t> &buf, size_t &pos) {
    if (pos + 8 > buf.size()) throw std::runtime_error("truncated_u64");
    uint64_t v = 0;
    std::memcpy(&v, buf.data() + pos, 8);
    pos += 8;
    return v;
}

static std::string rd_str(const std::vector<uint8_t> &buf, size_t &pos) {
    const uint64_t n = rd_u64(buf, pos);
    if (n > buf.size() || pos + (size_t)n > buf.size()) throw std::runtime_error("truncated_string");
    std::string s((const char *)buf.data() + pos, (size_t)n);
    pos += (size_t)n;
    return s;
}

static void skip_gguf_value(const std::vector<uint8_t> &buf, size_t &pos, uint32_t type);

static void skip_gguf_scalar(const std::vector<uint8_t> &buf, size_t &pos, uint32_t type) {
    static const uint8_t widths[] = {
        1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8,
    };
    if (type == 8) {
        (void)rd_str(buf, pos);
        return;
    }
    if (type == 9) {
        skip_gguf_value(buf, pos, type);
        return;
    }
    if (type >= sizeof(widths) || widths[type] == 0) throw std::runtime_error("unknown_gguf_value_type");
    if (pos + widths[type] > buf.size()) throw std::runtime_error("truncated_value");
    pos += widths[type];
}

static void skip_gguf_value(const std::vector<uint8_t> &buf, size_t &pos, uint32_t type) {
    if (type != 9) {
        skip_gguf_scalar(buf, pos, type);
        return;
    }
    const uint32_t item_type = rd_u32(buf, pos);
    const uint64_t count = rd_u64(buf, pos);
    for (uint64_t i = 0; i < count; ++i) skip_gguf_scalar(buf, pos, item_type);
}

static uint64_t ggml_type_bytes(uint32_t type, const std::vector<uint64_t> &dims) {
    uint64_t elems = 1;
    for (uint64_t dim : dims) elems *= dim ? dim : 1u;
    auto blocks = [elems](uint64_t block) { return (elems + block - 1u) / block; };
    switch (type) {
    case 0: return elems * 4u;
    case 1: return elems * 2u;
    case 2: return blocks(32) * 18u;
    case 3: return blocks(32) * 20u;
    case 6: return blocks(32) * 22u;
    case 7: return blocks(32) * 24u;
    case 8: return blocks(32) * 34u;
    case 9: return blocks(32) * 36u;
    case 10: return blocks(256) * 84u;
    case 11: return blocks(256) * 110u;
    case 12: return blocks(256) * 144u;
    case 13: return blocks(256) * 176u;
    case 14: return blocks(256) * 210u;
    case 15: return blocks(256) * 292u;
    case 16: return blocks(256) * 66u;
    case 17: return blocks(256) * 74u;
    case 18: return blocks(256) * 98u;
    case 19: return blocks(256) * 50u;
    case 20: return blocks(32) * 18u;
    case 21: return blocks(256) * 110u;
    case 22: return blocks(256) * 82u;
    case 23: return blocks(256) * 136u;
    case 24: return elems;
    case 25: return elems * 2u;
    case 26: return elems * 4u;
    case 27: return elems * 8u;
    case 28: return elems * 8u;
    case 30: return elems * 2u;
    default: return 0;
    }
}

static std::vector<uint8_t> read_file_prefix(const fs::path &path, uint64_t cap = 256ull * 1024ull * 1024ull) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open_failed");
    in.seekg(0, std::ios::end);
    const uint64_t size = (uint64_t)in.tellg();
    in.seekg(0, std::ios::beg);
    const uint64_t read_size = std::min(size, cap);
    std::vector<uint8_t> out((size_t)read_size);
    if (read_size && !in.read((char *)out.data(), (std::streamsize)read_size)) {
        throw std::runtime_error("read_failed");
    }
    return out;
}

static std::vector<TensorInfo> parse_gguf(const fs::path &path, std::string *architecture) {
    std::vector<uint8_t> buf = read_file_prefix(path);
    size_t pos = 0;
    const uint32_t magic = rd_u32(buf, pos);
    if (magic != 0x46554747u) throw std::runtime_error("bad_gguf_magic");
    const uint32_t version = rd_u32(buf, pos);
    if (version != 3u) throw std::runtime_error("unsupported_gguf_version");
    const uint64_t tensor_count = rd_u64(buf, pos);
    const uint64_t kv_count = rd_u64(buf, pos);
    for (uint64_t i = 0; i < kv_count; ++i) {
        const std::string key = rd_str(buf, pos);
        const uint32_t type = rd_u32(buf, pos);
        if (key == "general.architecture" && type == 8) {
            *architecture = rd_str(buf, pos);
        } else {
            skip_gguf_value(buf, pos, type);
        }
    }
    std::vector<TensorInfo> tensors;
    tensors.reserve((size_t)std::min<uint64_t>(tensor_count, 1000000u));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo t;
        t.name = rd_str(buf, pos);
        const uint32_t ndims = rd_u32(buf, pos);
        std::vector<uint64_t> dims;
        dims.reserve(ndims);
        for (uint32_t d = 0; d < ndims; ++d) dims.push_back(rd_u64(buf, pos));
        const uint32_t type = rd_u32(buf, pos);
        (void)rd_u64(buf, pos);
        t.layer = extract_layer(t.name);
        t.bytes = ggml_type_bytes(type, dims);
        tensors.push_back(t);
    }
    return tensors;
}

static void parse_safetensors_file(const fs::path &path, std::vector<TensorInfo> *tensors) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open_failed");
    uint64_t header_len = 0;
    in.read((char *)&header_len, 8);
    if (!in || header_len == 0 || header_len > 256ull * 1024ull * 1024ull) {
        throw std::runtime_error("invalid_safetensors_header");
    }
    std::string header((size_t)header_len, '\0');
    in.read(header.data(), (std::streamsize)header_len);
    if (!in) throw std::runtime_error("read_safetensors_header_failed");
    const std::regex item("\"([^\"]+)\"\\s*:\\s*\\{[^\\{\\}]*\"data_offsets\"\\s*:\\s*\\[\\s*(\\d+)\\s*,\\s*(\\d+)\\s*\\]");
    for (std::sregex_iterator it(header.begin(), header.end(), item), end; it != end; ++it) {
        TensorInfo t;
        t.name = (*it)[1].str();
        if (t.name == "__metadata__") continue;
        t.layer = extract_layer(t.name);
        const uint64_t a = std::strtoull((*it)[2].str().c_str(), nullptr, 10);
        const uint64_t b = std::strtoull((*it)[3].str().c_str(), nullptr, 10);
        t.bytes = b >= a ? b - a : 0;
        tensors->push_back(t);
    }
}

static std::vector<TensorInfo> parse_safetensors_dir(const fs::path &path) {
    std::vector<TensorInfo> tensors;
    std::vector<fs::path> files;
    if (fs::is_regular_file(path) && path.extension() == ".safetensors") {
        files.push_back(path);
    } else {
        for (const fs::directory_entry &entry : fs::recursive_directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    for (const fs::path &file : files) parse_safetensors_file(file, &tensors);
    return tensors;
}

static ModelMeta load_model_meta(const fs::path &path) {
    if (!fs::exists(path)) throw std::runtime_error("model_path_missing");
    ModelMeta meta;
    if (fs::is_regular_file(path) && path.extension() == ".gguf") {
        meta.format = "gguf";
        meta.tensors = parse_gguf(path, &meta.architecture);
    } else if (fs::is_directory(path) || path.extension() == ".safetensors") {
        meta.format = "safetensors";
        meta.tensors = parse_safetensors_dir(path);
    } else {
        throw std::runtime_error("unsupported_model_path_format");
    }
    int max_layer = -1;
    for (const TensorInfo &t : meta.tensors) {
        if (t.layer > max_layer) max_layer = t.layer;
        meta.whole_bytes += t.bytes;
    }
    meta.layer_count = max_layer >= 0 ? (uint32_t)max_layer + 1u : 0u;
    meta.layer_bytes.assign(meta.layer_count, 0u);
    for (const TensorInfo &t : meta.tensors) {
        if (t.layer >= 0 && (uint32_t)t.layer < meta.layer_count) {
            meta.layer_bytes[(uint32_t)t.layer] += t.bytes;
        }
    }
    return meta;
}

static std::string detect_family(const Options &opts, const std::string &architecture) {
    const std::string probe = lower_copy(opts.model_id + " " + architecture + " " + opts.model_path);
    if (probe.find("deepseek") != std::string::npos) return "deepseek";
    if (probe.find("qwen") != std::string::npos) return "qwen";
    return "unsupported";
}

static int fail_json(const std::string &message) {
    std::printf("{\"schema\":\"axiom_model_split_plan_v1\",\"status\":\"fail\",\"error\":");
    json_string(message);
    std::printf("}\n");
    return 1;
}

static uint64_t span_bytes(const std::vector<uint64_t> &layer_bytes, uint32_t start, uint32_t end) {
    if (start > end || end >= layer_bytes.size()) return 0;
    uint64_t total = 0;
    for (uint32_t i = start; i <= end; ++i) total += layer_bytes[i];
    return total;
}

static std::vector<InventoryNode> parse_inventory(const fs::path &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("inventory_open_failed");
    std::vector<InventoryNode> nodes;
    std::string line;
    uint32_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::replace(line.begin(), line.end(), ',', ' ');
        std::replace(line.begin(), line.end(), '\t', ' ');
        line = trim_copy(line);
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string name, endpoint, vram;
        ss >> name >> endpoint >> vram;
        if (name == "name" || name == "node") continue;
        if (name.empty() || endpoint.empty() || vram.empty()) {
            throw std::runtime_error("inventory_bad_row:" + std::to_string(line_no));
        }
        char *end = nullptr;
        const double gb = std::strtod(vram.c_str(), &end);
        if (!end || *end != '\0' || !(gb > 0.0)) {
            throw std::runtime_error("inventory_bad_vram_gb:" + std::to_string(line_no));
        }
        const long double bytes = (long double)gb * 1024.0L * 1024.0L * 1024.0L;
        if (bytes > (long double)std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("inventory_vram_overflow:" + std::to_string(line_no));
        }
        nodes.push_back({name, endpoint, (uint64_t)std::llround(bytes)});
    }
    if (nodes.empty()) throw std::runtime_error("inventory_empty");
    return nodes;
}

static std::vector<SpanPlan> byte_balanced_spans(const std::vector<uint64_t> &layer_bytes, uint32_t parts) {
    const uint32_t layers = (uint32_t)layer_bytes.size();
    if (parts == 0 || layers < parts) throw std::runtime_error("insufficient_layers_for_parts");
    std::vector<uint64_t> prefix(layers + 1u, 0u);
    for (uint32_t i = 0; i < layers; ++i) prefix[i + 1u] = prefix[i] + layer_bytes[i];
    const uint64_t total = prefix.back();
    if (total == 0) throw std::runtime_error("no_layer_bytes_found");
    std::vector<SpanPlan> spans;
    uint32_t start = 0;
    for (uint32_t p = 0; p < parts; ++p) {
        const uint32_t remaining = parts - p;
        uint32_t end = layers - 1u;
        if (remaining > 1u) {
            const uint64_t target = (uint64_t)(((unsigned __int128)total * (p + 1u)) / parts);
            const uint32_t max_end = layers - remaining;
            end = start;
            uint64_t best_delta = std::numeric_limits<uint64_t>::max();
            for (uint32_t cand = start; cand <= max_end; ++cand) {
                const uint64_t after = prefix[cand + 1u];
                const uint64_t delta = after > target ? after - target : target - after;
                if (delta < best_delta) {
                    best_delta = delta;
                    end = cand;
                }
            }
        }
        spans.push_back({start, end, span_bytes(layer_bytes, start, end)});
        start = end + 1u;
    }
    return spans;
}

static void print_span_json(const SpanPlan &span) {
    std::printf("{\"layer_start\":%u,\"layer_end\":%u,\"resident_bytes_expected\":%llu}",
                span.start, span.end, (unsigned long long)span.bytes);
}

static int emit_plan(const Options &opts) {
    const ModelMeta meta = load_model_meta(opts.model_path);
    if (meta.tensors.empty()) return fail_json("no_tensors_found");
    if (meta.layer_count == 0) return fail_json("no_layer_tensors_found");
    const std::vector<InventoryNode> nodes = parse_inventory(opts.inventory_path);
    const std::vector<SpanPlan> spans = byte_balanced_spans(meta.layer_bytes, (uint32_t)nodes.size() + 1u);
    const uint64_t model_sha = axiom_remote_model_fingerprint(opts.model_path.c_str());
    if (!model_sha) return fail_json("model_fingerprint_unavailable");
    for (size_t i = 0; i < nodes.size(); ++i) {
        const SpanPlan &span = spans[i + 1u];
        if (span.bytes > nodes[i].vram_bytes) {
            return fail_json("vram_exceeded:" + nodes[i].name);
        }
    }

    std::printf("{\"schema\":\"axiom_model_split_plan_v1\",\"status\":\"pass\"");
    std::printf(",\"model_id\":"); json_string(opts.model_id);
    std::printf(",\"model_sha\":\"%016llx\"", (unsigned long long)model_sha);
    std::printf(",\"model_path\":"); json_string(opts.model_path);
    std::printf(",\"format\":"); json_string(meta.format);
    std::printf(",\"architecture\":"); json_string(meta.architecture);
    std::printf(",\"layer_count\":%u", meta.layer_count);
    std::printf(",\"policy\":{\"name\":\"static_byte_balanced_v1\","
                "\"basis\":\"layer_tensor_bytes\",\"dynamic_scheduling\":false,"
                "\"automatic_failover\":false,\"retry_owner\":\"client\","
                "\"span_bytes\":[");
    for (size_t i = 0; i < spans.size(); ++i) {
        if (i) std::printf(",");
        std::printf("{\"role\":");
        json_string(i == 0 ? "coordinator" : "worker");
        if (i > 0) {
            std::printf(",\"node\":");
            json_string(nodes[i - 1u].name);
        }
        std::printf(",\"layer_start\":%u,\"layer_end\":%u,\"bytes\":%llu}",
                    spans[i].start, spans[i].end, (unsigned long long)spans[i].bytes);
    }
    std::printf("]}");
    std::printf(",\"nodes\":[");
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (i) std::printf(",");
        const SpanPlan &span = spans[i + 1u];
        std::printf("{\"name\":"); json_string(nodes[i].name);
        std::printf(",\"endpoint\":"); json_string(nodes[i].endpoint);
        std::printf(",\"layer_start\":%u,\"layer_end\":%u,"
                    "\"resident_bytes_expected\":%llu,\"kv_owner\":\"worker\"}",
                    span.start, span.end, (unsigned long long)span.bytes);
    }
    std::printf("],\"coordinator\":{\"layers_local\":[");
    print_span_json(spans[0]);
    std::printf("]}}\n");
    return 0;
}

static bool json_string_at(const std::string &s, size_t pos, std::string *out, size_t *end) {
    if (pos >= s.size() || s[pos] != '"') return false;
    std::string value;
    for (size_t i = pos + 1u; i < s.size(); ++i) {
        if (s[i] == '\\') {
            if (++i >= s.size()) return false;
            value.push_back(s[i]);
        } else if (s[i] == '"') {
            if (out) *out = value;
            if (end) *end = i + 1u;
            return true;
        } else {
            value.push_back(s[i]);
        }
    }
    return false;
}

static std::string json_get_string(const std::string &obj, const std::string &key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = obj.find(needle);
    if (key_pos == std::string::npos) return "";
    size_t colon = obj.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return "";
    ++colon;
    while (colon < obj.size() && std::isspace((unsigned char)obj[colon])) ++colon;
    std::string out;
    return json_string_at(obj, colon, &out, nullptr) ? out : "";
}

static bool json_get_u64(const std::string &obj, const std::string &key, uint64_t *out) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch m;
    if (!std::regex_search(obj, m, re)) return false;
    *out = std::strtoull(m[1].str().c_str(), nullptr, 10);
    return true;
}

static size_t matching_json_bracket(const std::string &s, size_t open, char a, char b) {
    bool in_str = false, esc = false;
    int depth = 0;
    for (size_t i = open; i < s.size(); ++i) {
        const char ch = s[i];
        if (esc) { esc = false; continue; }
        if (in_str) {
            if (ch == '\\') esc = true;
            else if (ch == '"') in_str = false;
            continue;
        }
        if (ch == '"') in_str = true;
        else if (ch == a) ++depth;
        else if (ch == b && --depth == 0) return i;
    }
    return std::string::npos;
}

static std::vector<std::string> json_array_objects(const std::string &json, const std::string &key) {
    std::vector<std::string> out;
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) return out;
    const size_t open = json.find('[', key_pos + needle.size());
    if (open == std::string::npos) return out;
    const size_t close = matching_json_bracket(json, open, '[', ']');
    if (close == std::string::npos) return out;
    const std::string body = json.substr(open + 1u, close - open - 1u);
    for (size_t i = 0; i < body.size(); ++i) {
        if (body[i] != '{') continue;
        const size_t end = matching_json_bracket(body, i, '{', '}');
        if (end == std::string::npos) break;
        out.push_back(body.substr(i, end - i + 1u));
        i = end;
    }
    return out;
}

static ParsedPlan parse_plan_json(const std::string &json) {
    ParsedPlan plan;
    plan.model_id = json_get_string(json, "model_id");
    plan.model_sha = json_get_string(json, "model_sha");
    for (const std::string &obj : json_array_objects(json, "nodes")) {
        PlanNode n;
        n.name = json_get_string(obj, "name");
        n.endpoint = json_get_string(obj, "endpoint");
        n.kv_owner = json_get_string(obj, "kv_owner");
        uint64_t v = 0;
        if (!json_get_u64(obj, "layer_start", &v) || v > 0xffffffffull) throw std::runtime_error("plan_node_layer_start_missing");
        n.span.start = (uint32_t)v;
        if (!json_get_u64(obj, "layer_end", &v) || v > 0xffffffffull) throw std::runtime_error("plan_node_layer_end_missing");
        n.span.end = (uint32_t)v;
        if (!json_get_u64(obj, "resident_bytes_expected", &n.span.bytes)) throw std::runtime_error("plan_node_resident_bytes_missing");
        plan.nodes.push_back(n);
    }
    for (const std::string &obj : json_array_objects(json, "layers_local")) {
        SpanPlan s;
        uint64_t v = 0;
        if (!json_get_u64(obj, "layer_start", &v) || v > 0xffffffffull) throw std::runtime_error("plan_local_layer_start_missing");
        s.start = (uint32_t)v;
        if (!json_get_u64(obj, "layer_end", &v) || v > 0xffffffffull) throw std::runtime_error("plan_local_layer_end_missing");
        s.end = (uint32_t)v;
        if (!json_get_u64(obj, "resident_bytes_expected", &s.bytes)) throw std::runtime_error("plan_local_resident_bytes_missing");
        plan.local_layers.push_back(s);
    }
    return plan;
}

static int validate_plan(const Options &opts) {
    if (opts.model_path.empty()) return fail_json("missing_model_path");
    if (opts.inventory_path.empty()) return fail_json("missing_inventory");
    const ModelMeta meta = load_model_meta(opts.model_path);
    if (meta.layer_count == 0) return fail_json("no_layer_tensors_found");
    const ParsedPlan plan = parse_plan_json(read_text(opts.validate_plan));
    if (plan.model_sha.empty()) return fail_json("model_sha_missing");
    const uint64_t model_sha = axiom_remote_model_fingerprint(opts.model_path.c_str());
    char sha_buf[17];
    std::snprintf(sha_buf, sizeof sha_buf, "%016llx", (unsigned long long)model_sha);
    if (!model_sha || lower_copy(plan.model_sha) != sha_buf) return fail_json("model_sha_mismatch");

    const std::vector<InventoryNode> inv = parse_inventory(opts.inventory_path);
    std::map<std::string, InventoryNode> by_name;
    for (const InventoryNode &n : inv) by_name[n.name] = n;

    struct TaggedSpan {
        uint32_t start, end;
        uint64_t bytes;
        bool worker;
        std::string name;
        std::string endpoint;
        std::string kv_owner;
    };
    std::vector<TaggedSpan> spans;
    for (const SpanPlan &s : plan.local_layers) {
        spans.push_back({s.start, s.end, s.bytes, false, "coordinator", "", ""});
    }
    for (const PlanNode &n : plan.nodes) {
        spans.push_back({n.span.start, n.span.end, n.span.bytes, true, n.name, n.endpoint, n.kv_owner});
    }
    if (spans.empty()) return fail_json("plan_has_no_spans");
    std::sort(spans.begin(), spans.end(), [](const TaggedSpan &a, const TaggedSpan &b) {
        if (a.start != b.start) return a.start < b.start;
        return a.end < b.end;
    });

    uint32_t expected = 0;
    for (const TaggedSpan &s : spans) {
        if (s.start > s.end || s.end >= meta.layer_count) return fail_json("span_out_of_range");
        if (s.start < expected) return fail_json("span_overlap");
        if (s.start > expected) return fail_json("span_gap");
        const uint64_t computed = span_bytes(meta.layer_bytes, s.start, s.end);
        if (computed != s.bytes) return fail_json("resident_bytes_mismatch");
        expected = s.end + 1u;
        if (!s.worker) continue;
        if (s.kv_owner != "worker") return fail_json("kv_owner_not_worker");
        const auto it = by_name.find(s.name);
        if (it == by_name.end()) return fail_json("inventory_node_missing:" + s.name);
        if (it->second.endpoint != s.endpoint) return fail_json("inventory_endpoint_mismatch:" + s.name);
        if (s.bytes > it->second.vram_bytes) return fail_json("vram_exceeded:" + s.name);
    }
    if (expected != meta.layer_count) return fail_json("span_gap");
    std::printf("{\"schema\":\"axiom_model_split_plan_validate_v1\",\"status\":\"pass\","
                "\"model_id\":");
    json_string(plan.model_id);
    std::printf(",\"layer_count\":%u,\"worker_count\":%zu,\"checks\":{"
                "\"contiguous\":true,\"no_overlap\":true,\"coverage_total\":true,"
                "\"vram_within_inventory\":true,\"resident_bytes_match\":true}}\n",
                meta.layer_count, plan.nodes.size());
    return 0;
}

static int legacy_node_plan(const Options &opts) {
    const ModelMeta meta = load_model_meta(opts.model_path);
    const std::string family = detect_family(opts, meta.architecture);
    const uint32_t node_zero = opts.node_index - 1u;
    const uint32_t base = meta.layer_count / opts.node_count;
    const uint32_t rem = meta.layer_count % opts.node_count;
    const uint32_t start_layer = node_zero * base + std::min(node_zero, rem);
    const uint32_t count = base + (node_zero < rem ? 1u : 0u);
    const uint32_t end_layer = count ? start_layer + count - 1u : start_layer;
    const bool owns_common_tail = opts.node_index == 1u;
    uint64_t assigned_bytes = 0;
    uint64_t assigned_tensors = 0;
    for (const TensorInfo &t : meta.tensors) {
        const bool layer_match = t.layer >= 0 &&
            (uint32_t)t.layer >= start_layer &&
            (uint32_t)t.layer <= end_layer;
        const bool common_match = owns_common_tail && t.layer < 0;
        if (layer_match || common_match) {
            assigned_tensors++;
            assigned_bytes += t.bytes;
        }
    }
    const bool pass = family != "unsupported" &&
        !meta.tensors.empty() &&
        meta.layer_count > 0 &&
        assigned_tensors > 0 &&
        assigned_tensors < meta.tensors.size();

    std::printf("{\"schema\":\"axiom_model_split_plan_v1\",\"status\":");
    json_string(pass ? "pass" : "fail");
    std::printf(",\"model_id\":"); json_string(opts.model_id);
    std::printf(",\"model_family\":"); json_string(family);
    std::printf(",\"model_path\":"); json_string(opts.model_path);
    std::printf(",\"format\":"); json_string(meta.format);
    std::printf(",\"architecture\":"); json_string(meta.architecture);
    std::printf(",\"node_index\":%u,\"node_count\":%u,\"start_layer\":%u,\"end_layer\":%u,"
                "\"layer_count\":%u,\"owns_common_tail\":%s,\"tensor_count\":%llu,"
                "\"assigned_tensors\":%llu,\"assigned_bytes\":%llu,\"whole_model_bytes\":%llu,"
                "\"whole_model_registered\":false,\"assigned_tensor_span_only\":true,"
                "\"single_node_bypass_allowed\":false,\"remote_worker_required\":true",
                opts.node_index, opts.node_count, start_layer, end_layer, meta.layer_count,
                owns_common_tail ? "true" : "false",
                (unsigned long long)meta.tensors.size(),
                (unsigned long long)assigned_tensors,
                (unsigned long long)assigned_bytes,
                (unsigned long long)meta.whole_bytes);
    if (!pass) {
        std::printf(",\"reason\":");
        if (family == "unsupported") json_string("unsupported_model_family");
        else if (meta.tensors.empty()) json_string("no_tensors_found");
        else if (meta.layer_count == 0) json_string("no_layer_tensors_found");
        else if (assigned_tensors == 0) json_string("no_tensors_assigned_to_node");
        else json_string("split_plan_not_partial");
    }
    std::printf("}\n");
    return pass ? 0 : 1;
}

int main(int argc, char **argv) {
    if (has_arg(argc, argv, "--help")) {
        std::puts("usage:");
        std::puts("  axiom-model-split-plan --model-id NAME --model-path PATH --inventory nodes.txt");
        std::puts("  axiom-model-split-plan --validate-plan plan.json --model-path PATH --inventory nodes.txt");
        std::puts("  legacy: axiom-model-split-plan --model-id NAME --model-path PATH --node-index 1 --node-count 2");
        return 0;
    }
    Options opts;
    opts.model_id = arg_value(argc, argv, "--model-id", opts.model_id.c_str());
    opts.model_path = arg_value(argc, argv, "--model-path", "");
    opts.inventory_path = arg_value(argc, argv, "--inventory", "");
    opts.validate_plan = arg_value(argc, argv, "--validate-plan", arg_value(argc, argv, "--validate", ""));
    opts.node_index = arg_u32(argc, argv, "--node-index", 1);
    opts.node_count = arg_u32(argc, argv, "--node-count", 2);
    if (!opts.validate_plan.empty() && !fs::exists(opts.validate_plan)) return fail_json("plan_path_missing");
    if (opts.model_path.empty()) return fail_json("missing_model_path");
    if (opts.node_count == 0 || opts.node_index == 0 || opts.node_index > opts.node_count) {
        return fail_json("invalid_node_index_or_count");
    }
    try {
        if (!opts.validate_plan.empty()) return validate_plan(opts);
        if (!opts.inventory_path.empty()) return emit_plan(opts);
        return legacy_node_plan(opts);
    } catch (const std::exception &exc) {
        return fail_json(exc.what());
    }
}
