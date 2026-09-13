#pragma once

// Opt-in Responses wire boundary. Core remains the registry/executor; native
// validation sees only canonical names. No argument, instruction or ID rewrite.
#include "axiom_aliced_json.h"
#include "axiom_codex_schema.h"
#include <openssl/sha.h>
#include <map>
#include <stdexcept>

namespace axiom_codex {
// An unconfigured bridge must not interrupt a healthy long-context prefill at
// an invented wall-clock deadline. Zero denotes ABSENT configuration only;
// explicit operator/per-request deadlines remain finite and authoritative.
// Disconnect/cancellation and token/context limits remain independently active.
inline bool request_deadline_ms(const char *configured, uint64_t *out) {
    if (!out) return false;
    *out = 0u;
    if (!configured || !*configured) return true;
    uint64_t value = 0;
    for (const char *p = configured; *p; ++p) {
        if (*p < '0' || *p > '9' || value > 86400000u / 10u) return false;
        value = value * 10u + static_cast<unsigned>(*p - '0');
        if (value > 86400000u) return false;
    }
    if (value == 0u) return false;
    *out = value; return true;
}
inline std::string string_field(const ajson &v, const char *key) {
    const auto *p = v.get(key);
    return p && p->is_string() ? p->s : std::string();
}
inline void erase_field(ajson &v, const char *key) {
    for (size_t i = 0; i < v.keys.size(); ++i) {
        if (v.keys[i] == key) {
            v.keys.erase(v.keys.begin() + i);
            v.vals.erase(v.vals.begin() + i);
            return;
        }
    }
}
inline bool canonical(const std::string &s) {
    if (s.empty() || s.size() > 64 || s.front() < 'a' || s.front() > 'z' ||
        s.back() == '_') return false;
    bool separator = false;
    for (char c : s) {
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) separator = false;
        else if (c == '_' && !separator) separator = true;
        else return false;
    }
    return true;
}

// A Core request is a complete authoritative transcript. A stable session ID
// does not authorize replacing its changed tools/instructions with old tokens.
// This controls only the caller's approximate-tail path; exact-prefix reuse
// remains available in the existing session store.
inline bool allow_stateful_tail(bool codex_provider, bool native_eligible) {
    return !codex_provider && native_eligible;
}

// Core children share a root session_id but carry distinct thread_id values.
// Partition only the INTERNAL persistence lookup, never the wire identity.
// Otherwise each child overwrites the parent's one saved prefix and causes a
// full prefill when the parent resumes. Legacy and root-thread keys stay exact.
inline std::string cache_session_id(const std::string &root, const ajson &request) {
    const auto thread = string_field(request, "thread_id");
    if (thread.empty() || thread == root) return root;
    const std::string key = "axiom-codex-child-v1:" + std::to_string(root.size()) + ":" + root + thread;
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if (!SHA256(reinterpret_cast<const unsigned char *>(key.data()), key.size(), digest))
        throw std::runtime_error("Codex cache identity digest failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "codex-child-";
    for (const auto c : digest) { result += hex[c >> 4]; result += hex[c & 15]; }
    return result;
}

inline void apply_request_defaults(ajson &request) {
    // Core normally omits an output limit. The legacy 128-token chat default
    // truncates ordinary patch/delegation arguments. Codex needs a finite
    // operational budget, not an inferred change to reasoning or context.
    // Caller-supplied limits, including invalid values, remain authoritative
    // and go through the existing strict validator.
    if (!request.get("max_output_tokens") && !request.get("max_tokens")) {
        const auto *alias = request.get("max_completion_tokens");
        request.set("max_output_tokens", alias ? *alias : ajson::jint(4096));
    }
}

// OpenAI custom tools put their input grammar in `format`, not description.
// Qwen receives these as a native function with one string argument (`input`).
// Preserve the exact grammar in model-visible documentation; otherwise Core's
// apply_patch is indistinguishable from a conventional unified-diff tool.
// This is prompt guidance, not a claim of grammar-constrained CUDA decoding.
// Core retains the original custom tool and validates/executes its raw input.
inline void describe_custom_format(ajson &definition, const std::string &path) {
    const auto *format = definition.get("format");
    if (!format || format->is_null()) return; // genuinely unconstrained custom input
    if (!format->is_object()) throw schema_failure(schema_path(path, "format"), "format", "custom format must be an object");
    const auto type = string_field(*format, "type");
    if (type == "text") return;
    if (type != "grammar") throw schema_failure(schema_path(path, "format"), "type", "unsupported custom format type");
    const auto syntax = string_field(*format, "syntax"), grammar = string_field(*format, "definition");
    if (syntax != "lark" && syntax != "regex")
        throw schema_failure(schema_path(schema_path(path, "format"), "syntax"), "syntax", "custom grammar syntax must be lark or regex");
    if (grammar.empty()) throw schema_failure(schema_path(schema_path(path, "format"), "definition"), "definition", "custom grammar must be nonempty");
    const auto contract = "\nCore custom input contract: call this native function with a JSON object whose input field is the raw custom-tool text. "
        "The input string (not the surrounding JSON object) must follow the exact " + syntax + " grammar below. "
        "Do not substitute another patch or command format. Core validates and executes the input.\n" + grammar;
    const auto description = string_field(definition, "description");
    // Idempotent for deferred discovery/history and diagnostic projection.
    if (description.size() < contract.size() || description.compare(description.size() - contract.size(), contract.size(), contract) != 0)
        definition.set("description", ajson::jstr(description + contract));
}

class tool_wire_map {
    struct identity { std::string name; std::string scope; std::string variant; };
    std::map<std::string, identity> reverse_;
    std::map<std::string, ajson> originals_;
    std::map<std::string, ajson> branches_;
    std::map<std::string, std::vector<std::string>> variants_;
    std::map<std::string, std::string> declared_;
    std::map<std::string, ajson> prepared_definitions_;
    std::string tool_search_wire_;
    std::set<std::string> unavailable_;
    std::set<std::string> callable_;
    ajson rejected_ = ajson::jarr();
    schema_check schema_check_;
    argument_check argument_check_;
    std::function<bool(const ajson &, std::string *)> tool_check_;
    ajson response_fields_ = ajson::jobj();
    ajson original_request_ = ajson::jobj();
    std::set<std::string> selected_;
    bool require_call_ = false, forbid_calls_ = false, parallel_calls_ = true, restrict_catalog_ = false;

    void reject(const ajson &tool, const std::string &scope, const schema_failure &e) {
        const auto *f = tool.get("function");
        const std::string name = string_field(f ? *f : tool, "name");
        ajson diagnostic = ajson::jobj();
        diagnostic.set("code", ajson::jstr("codex_tool_schema_unavailable"));
        diagnostic.set("tool_name", ajson::jstr(name.empty() ? string_field(tool, "type") : name));
        diagnostic.set("namespace", ajson::jstr(scope));
        diagnostic.set("json_path", ajson::jstr(e.path));
        diagnostic.set("unsupported_keyword", ajson::jstr(e.keyword));
        diagnostic.set("reason", ajson::jstr(e.what()));
        if (!name.empty()) {
            const auto wire = bind(name, scope);
            unavailable_.insert(wire);
            for (const auto &v : variants_[wire]) unavailable_.insert(v);
            diagnostic.set("wire_name", ajson::jstr(wire));
        }
        rejected_.push(std::move(diagnostic));
    }

    // Injective byte escape: normal words and single interior underscores stay
    // readable. z is reserved (zz); every other byte is zHH. No casing loss,
    // underscore collapsing or Unicode normalization can merge identities.
    static std::string wire_component(const std::string &value) {
        constexpr char hex[] = "0123456789abcdef";
        std::string out;
        for (size_t i = 0; i < value.size(); ++i) {
            const auto c = static_cast<unsigned char>(value[i]);
            if (c == 'z') out += "zz";
            else if ((c >= 'a' && c <= 'y') || (c >= '0' && c <= '9') ||
                     (c == '_' && i > 0 && i + 1 < value.size() &&
                      value[i - 1] != '_' && value[i + 1] != '_')) out += static_cast<char>(c);
            else { out += 'z'; out += hex[c >> 4]; out += hex[c & 15]; }
        }
        return out;
    }

    std::string bind(const std::string &name, const std::string &scope, const std::string &variant = {}) {
        if (name.empty()) throw std::runtime_error("Codex tool name must be nonempty");
        std::string wire = name;
        if (!variant.empty() || !scope.empty() || !canonical(name) || name.rfind("codex_", 0) == 0) {
            const auto n = wire_component(name), s = wire_component(scope);
            // Lengths delimit components even when a tool contains _s12_, etc.
            // Ordering/discovery never changes a tool's model-visible name.
            wire = "codex_n" + std::to_string(n.size()) + "_" + n +
                   "_s" + std::to_string(s.size()) + (s.empty() ? "" : "_" + s);
            if (!variant.empty() || wire.size() > 64) {
                // Rare overlong names and exact schema branches still need a
                // bounded digest. Retain a readable stem, and reject collisions
                // across full identities INCLUDING the original schema branch.
                const std::string key = std::to_string(scope.size()) + ":" + scope +
                    std::to_string(name.size()) + ":" + name +
                    std::to_string(variant.size()) + ":" + variant;
                unsigned char digest[SHA256_DIGEST_LENGTH];
                if (!SHA256(reinterpret_cast<const unsigned char *>(key.data()), key.size(), digest))
                    throw std::runtime_error("Codex tool identity digest failed");
                auto stem = n.substr(0, 22);
                while (!stem.empty() && stem.back() == '_') stem.pop_back();
                constexpr char hex[] = "0123456789abcdef";
                wire = "codex_h_" + stem + "_";
                for (size_t i = 0; i < 16; ++i) {
                    wire += hex[digest[i] >> 4]; wire += hex[digest[i] & 15];
                }
            }
        }
        if (!canonical(wire)) throw std::runtime_error("Codex wire identity is not canonical");
        const auto old = reverse_.find(wire);
        if (old != reverse_.end() &&
            (old->second.name != name || old->second.scope != scope || old->second.variant != variant))
            throw std::runtime_error("Codex tool wire-name collision");
        reverse_[wire] = {name, scope, variant};
        return wire;
    }

    void rename(ajson &v, const std::string &scope = {}) {
        const auto *name = v.get("name");
        if (!name || !name->is_string())
            throw std::runtime_error("Codex tool requires a string name");
        const auto *ns = v.get("namespace");
        if (ns && !ns->is_null() && !ns->is_string())
            throw std::runtime_error("Codex tool namespace must be a string");
        const std::string effective_scope = scope.empty() ? string_field(v, "namespace") : scope;
        v.set("name", ajson::jstr(bind(name->s, effective_scope)));
        erase_field(v, "namespace");
    }

    ajson definitions(const ajson &list, const std::string &scope = {}, const std::string &base = "$.tools", bool replay = false) {
        if (!list.is_array()) throw std::runtime_error("Codex tools must be an array");
        ajson out = ajson::jarr();
        for (size_t index = 0; index < list.arr.size(); ++index) {
            const auto &original = list.arr[index];
            const auto *nested_original = original.get("function");
            const auto effective_scope = scope.empty() ? string_field(nested_original ? *nested_original : original, "namespace") : scope;
            const auto path = base + "[" + std::to_string(index) + "]";
            try {
            ajson tool = original;
            const std::string type = string_field(tool, "type");
            if (type == "namespace") {
                if (!scope.empty()) throw std::runtime_error("Nested Codex namespaces are unsupported");
                const std::string ns = string_field(tool, "name");
                const auto *children = tool.get("tools");
                if (ns.empty() || !children) throw std::runtime_error("Invalid Codex namespace tool");
                for (auto child : definitions(*children, ns, schema_path(path, "tools"), replay).arr) out.push(std::move(child));
                continue;
            }
            if (type == "tool_search") {
                // Core owns discovery/execution. Only translate its typed
                // client search protocol into Qwen's function-call grammar.
                if (!scope.empty() || string_field(tool, "execution") != "client" || !tool.get("parameters"))
                    throw schema_failure(path, "execution", "tool_search requires execution=client and the Core parameter schema");
                ajson function = tool; function.set("type", ajson::jstr("function"));
                function.set("name", ajson::jstr("tool_search")); erase_field(function, "execution");
                ajson one = ajson::jarr(); one.push(std::move(function));
                auto lowered = definitions(one, {}, path, replay);
                if (lowered.arr.size() != 1) throw schema_failure(path, "parameters", "client tool_search requires one exact native schema");
                tool_search_wire_ = string_field(lowered.arr.front(), "name");
                out.push(std::move(lowered.arr.front())); continue;
            }
            if (type == "function" || type == "custom" || (type.empty() && !scope.empty())) {
                const auto *nested = tool.get("function");
                const ajson definition = nested ? *nested : tool;
                const auto *name = definition.get("name");
                if (!name || !name->is_string()) throw schema_failure(path, "name", "tool requires a string name");
                const auto base_wire = bind(name->s, effective_scope);
                ajson identity_definition = original;
                // Loading status is transport metadata, not a new definition.
                erase_field(identity_definition, "defer_loading");
                const auto encoded = ajson_dumps(identity_definition);
                const auto previous = declared_.find(base_wire);
                if (previous != declared_.end()) {
                    if (previous->second == encoded) {
                        if (replay) for (auto loaded : prepared_definitions_[base_wire].arr) {
                            loaded.set("defer_loading", ajson::jbool(true)); out.push(std::move(loaded));
                        }
                        continue;
                    }
                    throw schema_failure(path, "name", "conflicting duplicate tool definitions; all instances are unavailable");
                }
                const auto *parameters = definition.get("parameters");
                const auto alternatives = parameters ? lower_schema_alternatives(*parameters,
                    schema_path(path, "parameters"), schema_check_, argument_check_) : std::vector<ajson>{ajson::jobj()};
                ajson documentation = ajson::jobj();
                if (parameters && alternatives.size() > 1) (void)hoist_schema_documentation(*parameters, &documentation);
                const auto primary_wire = alternatives.size() == 1 ? base_wire :
                    bind(name->s, effective_scope, ajson_dumps(alternatives.front()));
                ajson prepared = ajson::jarr();
                std::vector<std::string> wires;
                for (const auto &branch : alternatives) {
                    ajson candidate = tool, function = definition;
                    if (type == "custom") describe_custom_format(function, path);
                    const auto wire = alternatives.size() == 1 ? base_wire : bind(name->s, effective_scope, ajson_dumps(branch));
                    function.set("name", ajson::jstr(wire)); erase_field(function, "namespace");
                    if (parameters) function.set("parameters", alternatives.size() > 1 ? hoist_schema_documentation(branch, nullptr) : branch);
                    if (alternatives.size() > 1) {
                        std::string description = wire == primary_wire ? string_field(definition, "description") :
                            "Same tool as " + primary_wire + "; alternative argument schema. Use its shared documentation.";
                        if (wire == primary_wire && !documentation.keys.empty())
                            description += "\nShared parameter documentation (original schema paths): " + ajson_dumps(documentation);
                        function.set("description", ajson::jstr(std::move(description)));
                    }
                    if (wire != name->s) function.set("description", ajson::jstr(
                        "Core tool " + (effective_scope.empty() ? std::string() : effective_scope + ".") + name->s +
                        ". Call directly with <function=" + wire + "> inside <tool_call>, using this schema's parameters. "
                        "This is a callable tool, not a shell command. Printing its name or passing it to another tool does not invoke it. " +
                        string_field(function, "description")));
                    if (nested) candidate.set("function", std::move(function)); else candidate = std::move(function);
                    if (type.empty()) candidate.set("type", ajson::jstr("function"));
                    std::string error;
                    if (tool_check_ && !tool_check_(candidate, &error)) throw schema_failure(path, "tool", error);
                    if (parameters) { originals_[wire] = *parameters; branches_[wire] = branch; }
                    // Identical alternatives are harmless overlap, not extra tools.
                    if (std::find(wires.begin(), wires.end(), wire) == wires.end()) {
                        wires.push_back(wire); prepared.push(std::move(candidate));
                    }
                }
                variants_[base_wire] = std::move(wires); declared_[base_wire] = encoded;
                prepared_definitions_[base_wire] = prepared;
                for (auto &candidate : prepared.arr) {
                    const auto *function = candidate.get("function");
                    callable_.insert(string_field(function ? *function : candidate, "name"));
                    if (replay) candidate.set("defer_loading", ajson::jbool(true));
                    out.push(std::move(candidate));
                }
                continue;
            }
            // Built-in types are not invented as executable functions here.
            std::string error;
            if (tool_check_ && !tool_check_(tool, &error)) throw schema_failure(path, "tool", error);
            const auto *f = tool.get("function");
            const auto wire = string_field(f ? *f : tool, "name");
            if (!wire.empty()) {
                const auto encoded = ajson_dumps(tool);
                const auto previous = declared_.find(wire);
                if (previous != declared_.end()) {
                    if (previous->second == encoded) continue;
                    throw schema_failure(path, "name", "conflicting duplicate tool definitions; all instances are unavailable");
                }
                declared_[wire] = encoded;
            }
            out.push(std::move(tool));
            } catch (const schema_failure &e) { reject(original, effective_scope, e); }
              catch (const std::exception &e) { reject(original, effective_scope, schema_failure(path, "tool", e.what())); }
        }
        return out;
    }

    std::vector<std::string> reference(const ajson &v) {
        const auto *nested = v.get("function");
        const auto &f = nested ? *nested : v;
        const auto base = bind(string_field(f, "name"), string_field(f, "namespace"));
        if (unavailable_.count(base)) throw std::runtime_error("tool_choice selects an unavailable tool: " + string_field(f, "name"));
        const auto found = variants_.find(base);
        if (found == variants_.end()) throw std::runtime_error("tool_choice references an undeclared tool: " + string_field(f, "name"));
        return found->second;
    }

    void choice(ajson &v) {
        if (!v.is_object()) {
            if (!v.is_string() || (v.s != "auto" && v.s != "none" && v.s != "required"))
                throw std::runtime_error("invalid Codex tool_choice");
            require_call_ = v.s == "required"; forbid_calls_ = v.s == "none";
            return;
        }
        const std::string type = string_field(v, "type");
        if (type == "tool_search") {
            if (tool_search_wire_.empty()) throw std::runtime_error("tool_choice references undeclared client tool_search");
            selected_.insert(tool_search_wire_); require_call_ = true;
            v = ajson::jobj(); v.set("type", ajson::jstr("function")); v.set("name", ajson::jstr(tool_search_wire_));
        } else if (type == "allowed_tools") {
            restrict_catalog_ = true;
            const auto mode = string_field(v, "mode");
            if (mode != "auto" && mode != "required") throw std::runtime_error("allowed_tools.mode must be auto or required");
            const auto *tools = v.get("tools");
            if (!tools || !tools->is_array() || tools->arr.empty()) throw std::runtime_error("allowed_tools requires nonempty tool references");
            for (const auto &tool : tools->arr) for (const auto &wire : reference(tool)) selected_.insert(wire);
            require_call_ = mode == "required";
            v = ajson::jstr(mode);
        } else if (type == "function" || type == "custom") {
            const auto wires = reference(v);
            selected_.insert(wires.begin(), wires.end()); require_call_ = true;
            if (wires.size() == 1) {
                v = ajson::jobj(); v.set("type", ajson::jstr(type)); v.set("name", ajson::jstr(wires.front()));
            } else { v = ajson::jstr("required"); restrict_catalog_ = true; }
        } else throw std::runtime_error("unsupported Codex tool_choice type: " + type);
    }

    void history(ajson &item) {
        const auto base = bind(string_field(item, "name"), string_field(item, "namespace"));
        const auto found = variants_.find(base);
        if (found == variants_.end() || found->second.size() == 1) { rename(item); return; }
        ajson arguments; std::string error;
        if (!ajson_parse(string_field(item, "arguments"), arguments, error)) throw std::runtime_error("invalid tool history arguments: " + error);
        for (const auto &wire : found->second) if (argument_check_ && argument_check_(arguments, branches_.at(wire), &error)) {
            item.set("name", ajson::jstr(wire)); erase_field(item, "namespace"); return;
        }
        throw std::runtime_error("tool history arguments match no exact schema alternative");
    }

public:
    void schema_validation(schema_check schema, argument_check arguments,
                           std::function<bool(const ajson &, std::string *)> tool) {
        schema_check_ = std::move(schema); argument_check_ = std::move(arguments); tool_check_ = std::move(tool);
    }
    const ajson &rejections() const { return rejected_; }
    const ajson &original_request() const { return original_request_; }
    bool is_client_tool_search(const std::string &wire) const { return !tool_search_wire_.empty() && wire == tool_search_wire_; }
    const ajson *original_schema(const std::string &wire) const {
        const auto i = originals_.find(wire); return i == originals_.end() ? nullptr : &i->second;
    }
    bool validate_arguments(const std::string &wire, const ajson &arguments, std::string *error) const {
        if (!callable_.count(wire) || unavailable_.count(wire)) { if (error) *error = "unknown or unavailable tool: " + wire; return false; }
        const auto *schema = original_schema(wire);
        return !schema || !argument_check_ || argument_check_(arguments, *schema, error);
    }
    bool validate_calls(const std::vector<std::string> &names, std::string *error) const {
        if ((require_call_ && names.empty()) || (forbid_calls_ && !names.empty()) || (!parallel_calls_ && names.size() > 1)) {
            if (error) *error = "model output violates tool_choice or parallel_tool_calls";
            return false;
        }
        for (const auto &name : names) if (!callable_.count(name) || (!selected_.empty() && !selected_.count(name)) || unavailable_.count(name)) {
            if (error) *error = "model selected an unavailable or disallowed tool: " + name;
            return false;
        }
        return true;
    }
    bool prepare(ajson &request, std::string *error) {
        try {
            // A map can be reused by contract tests or an embedding. Never let
            // previous rejections, selections or request identities leak.
            reverse_.clear(); originals_.clear(); branches_.clear(); variants_.clear(); declared_.clear();
            prepared_definitions_.clear(); tool_search_wire_.clear();
            unavailable_.clear(); callable_.clear(); rejected_ = ajson::jarr(); response_fields_ = ajson::jobj(); selected_.clear();
            require_call_ = forbid_calls_ = restrict_catalog_ = false; parallel_calls_ = true;
            original_request_ = request;
            ajson copy = request;
            // Core's authoritative Responses metadata carries thread/session
            // identity even when it is absent from the top-level body.
            if (const auto *metadata = request.get("client_metadata"); metadata && metadata->is_object()) {
                for (const char *key : {"session_id", "thread_id", "turn_id"}) {
                    const auto *value = metadata->get(key);
                    if (!value) continue;
                    if (!value->is_string() || value->s.empty()) throw std::runtime_error(std::string("invalid Core client_metadata.") + key);
                    if (const auto *explicit_value = copy.get(key)) {
                        if (!explicit_value->is_string() || explicit_value->s != value->s)
                            throw std::runtime_error(std::string("conflicting Core identity: ") + key);
                    } else copy.set(key, *value);
                }
            }
            for (const char *key : {"metadata", "client_metadata", "session_id", "thread_id", "turn_id", "parallel_tool_calls"})
                if (const auto *value = copy.get(key)) response_fields_.set(key, *value);
            if (const auto *parallel = request.get("parallel_tool_calls")) {
                if (!parallel->is_bool()) throw std::runtime_error("parallel_tool_calls must be boolean");
                parallel_calls_ = parallel->b;
            }
            // Transactional copy: malformed catalogs never partially modify input.
            if (const auto *tools = copy.get("tools")) {
                ajson admitted = definitions(*tools), available = ajson::jarr();
                for (const auto &tool : admitted.arr) {
                    const auto *f = tool.get("function");
                    if (!unavailable_.count(string_field(f ? *f : tool, "name"))) available.push(tool);
                }
                copy.set("tools", std::move(available));
            }
            if (const auto *input = copy.get("input"); input && input->is_array()) {
                ajson items = *input;
                for (size_t index = 0; index < items.arr.size(); ++index) {
                    auto &item = items.arr[index];
                    const auto type = string_field(item, "type");
                    if (type == "function_call" || type == "custom_tool_call") history(item);
                    else if (type == "tool_search_output") {
                        if (const auto *tools = item.get("tools")) item.set("tools", definitions(*tools, {}, "$.input[" + std::to_string(index) + "].tools", true));
                    }
                    // function_call_output, call_id, content and arguments are opaque.
                }
                copy.set("input", std::move(items));
            }
            if (const auto *selected = copy.get("tool_choice")) {
                ajson c = *selected; choice(c); copy.set("tool_choice", std::move(c));
            }
            if (restrict_catalog_) {
                ajson filtered = ajson::jarr();
                if (const auto *tools = copy.get("tools")) for (const auto &tool : tools->arr) {
                    const auto *f = tool.get("function");
                    if (selected_.count(string_field(f ? *f : tool, "name"))) filtered.push(tool);
                }
                copy.set("tools", std::move(filtered));
            }
            if (string_field(copy, "tool_choice") == "required") {
                const auto *tools = copy.get("tools");
                if (!tools || tools->arr.empty()) throw std::runtime_error("tool_choice=required but no tool can be advertised safely");
            }
            request = std::move(copy);
            return true;
        } catch (const std::exception &e) {
            if (error) *error = e.what();
            return false;
        }
    }

    // Only structural Responses fields are traversed. Never rewrite model text,
    // tool arguments/results, errors, call_id, item id or session identifiers.
    void restore(ajson &v) const {
        if (!v.is_object()) return;
        const auto type = string_field(v, "type");
        if (type == "response" || string_field(v, "object") == "response" || v.get("output")) {
            if (!rejected_.arr.empty()) v.set("codex_tool_errors", rejected_);
            for (size_t i = 0; i < response_fields_.keys.size(); ++i) v.set(response_fields_.keys[i], response_fields_.vals[i]);
        }
        if (type == "function_call" || type == "custom_tool_call") {
            if (is_client_tool_search(string_field(v, "name"))) {
                ajson arguments; std::string error;
                const auto text = string_field(v, "arguments");
                if (!ajson_parse(text.empty() ? "{}" : text, arguments, error) || !arguments.is_object())
                    throw std::runtime_error("invalid client tool_search arguments: " + error);
                v.set("type", ajson::jstr("tool_search_call"));
                v.set("execution", ajson::jstr("client")); v.set("arguments", std::move(arguments));
                erase_field(v, "name"); erase_field(v, "namespace");
                return;
            }
            const auto found = reverse_.find(string_field(v, "name"));
            if (found != reverse_.end()) {
                v.set("name", ajson::jstr(found->second.name));
                if (!found->second.scope.empty()) v.set("namespace", ajson::jstr(found->second.scope));
            }
        }
        for (const char *key : {"item", "response"}) {
            if (const auto *child = v.get(key)) {
                ajson copy = *child; restore(copy); v.set(key, std::move(copy));
            }
        }
        if (const auto *output = v.get("output"); output && output->is_array()) {
            ajson copy = *output;
            for (auto &item : copy.arr) restore(item);
            v.set("output", std::move(copy));
        }
    }
};

// Codex 0.153.4 ModelInfo, alongside (not instead of) the OpenAI data list.
inline ajson catalog_model(const std::string &id, uint32_t context, uint32_t maximum,
                           const std::string &effort, bool client_tool_search = false) {
    ajson m = ajson::jobj();
    m.set("slug", ajson::jstr(id)); m.set("display_name", ajson::jstr("Axiom " + id));
    m.set("description", ajson::jstr("Native Axiom Responses provider"));
    m.set("default_reasoning_level", ajson::jstr(effort));
    ajson levels = ajson::jarr();
    for (const char *name : {"ultra-fast", "minimal", "low", "medium", "high", "xhigh", "max"}) {
        ajson level = ajson::jobj();
        level.set("effort", ajson::jstr(name)); level.set("description", ajson::jstr(name));
        levels.push(std::move(level));
    }
    m.set("supported_reasoning_levels", std::move(levels));
    m.set("shell_type", ajson::jstr("shell_command"));
    m.set("visibility", ajson::jstr("list")); m.set("supported_in_api", ajson::jbool(true));
    m.set("priority", ajson::jint(1)); m.set("context_window", ajson::jint(context));
    m.set("max_context_window", ajson::jint(maximum));
    m.set("support_verbosity", ajson::jbool(false));
    m.set("default_verbosity", ajson::jnull()); m.set("default_reasoning_summary", ajson::jstr("none"));
    m.set("availability_nux", ajson::jnull()); m.set("upgrade", ajson::jnull());
    m.set("apply_patch_tool_type", ajson::jstr("freeform"));
    m.set("experimental_supported_tools", ajson::jarr());
    if (client_tool_search) m.set("supports_search_tool", ajson::jbool(true));
    m.set("input_modalities", [] { ajson a = ajson::jarr(); a.push(ajson::jstr("text")); return a; }());
    ajson truncation = ajson::jobj();
    truncation.set("mode", ajson::jstr("tokens")); truncation.set("limit", ajson::jint(10000));
    m.set("truncation_policy", std::move(truncation));
    m.set("base_instructions", ajson::jstr("You are Axiom. Use the provided tools precisely and report verified results."));
    return m;
}
} // namespace axiom_codex
