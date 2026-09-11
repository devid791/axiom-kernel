#pragma once
#include "axiom_aliced_json.h"
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <set>
#include <vector>

namespace axiom_codex {
// These callbacks use the daemon's unchanged native validators. The argument
// callback additionally evaluates anyOf and bounds against the ORIGINAL tree.
using schema_check = std::function<bool(const ajson &, const std::string &, std::string *)>;
using argument_check = std::function<bool(const ajson &, const ajson &, std::string *)>;

struct schema_failure : std::runtime_error {
    std::string path, keyword;
    schema_failure(std::string p, std::string k, std::string reason)
        : std::runtime_error(reason), path(std::move(p)), keyword(std::move(k)) {}
};

inline std::string schema_path(const std::string &base, const std::string &key) {
    return base + "[" + ajson_dumps(ajson::jstr(key)) + "]";
}

// Conservative exact compiler, not a JSON Schema dialect converter. No unknown
// assertion is silently ignored. Unbounded unions cannot be represented in the
// native single-type vocabulary and are rejected, never approximated.
inline ajson lower_schema(const ajson &source, const std::string &path,
                          const schema_check &check, const argument_check &accept,
                          size_t depth = 0) {
    auto fail = [&](const std::string &key, const std::string &reason) -> void {
        throw schema_failure(path, key, reason);
    };
    if (depth > ALICED_JSON_MAX_DEPTH) fail("schema", "schema nesting limit exceeded");
    if (!source.is_object()) fail("schema", "schema must be an object");
    static const std::set<std::string> known = {
        "type", "enum", "const", "properties", "required", "additionalProperties", "items", "anyOf", "minItems",
        "minLength", "maxLength", "minimum", "maximum",
        "description", "title", "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly"
    };
    for (const auto &key : source.keys)
        if (!known.count(key)) fail(key, "assertion is not implemented by the native tool schema validator");
    for (const char *key : {"minLength", "maxLength", "minimum", "maximum"}) {
        if (const auto *bound = source.get(key)) {
            const bool length = std::string(key) == "minLength" || std::string(key) == "maxLength";
            if (!bound->is_number() || !std::isfinite(bound->number()) ||
                (length && (bound->number() < 0 || std::floor(bound->number()) != bound->number())))
                fail(key, std::string(key) + (length ? " must be a non-negative integer" : " must be a finite number"));
            if (!accept) fail(key, "Codex original-schema argument validation must be configured");
            // Native syntax accepts these but does not enforce arguments.
            // Retain the exact assertion/value on wire; the Codex guard owns
            // enforcement. Do not change the legacy native validator.
        }
    }
    if (const auto *minimum = source.get("minItems")) {
        if (!minimum->is_number() || !std::isfinite(minimum->number()) ||
            minimum->number() < 0 || std::floor(minimum->number()) != minimum->number())
            fail("minItems", "minItems must be a non-negative integer");
        if (!accept) fail("minItems", "Codex original-schema argument validation must be configured");
        // Native schema syntax already accepts minItems, but its argument
        // validator does not enforce it. KEEP the keyword on the wire and
        // enforce it in the Codex original-schema guard before emitting calls.
        // This is not removal or relaxation of an assertion.
    }
    ajson out = source;
    if (const auto *properties = source.get("properties")) {
        if (!properties->is_object()) fail("properties", "properties must be an object");
        ajson children = ajson::jobj();
        for (size_t i = 0; i < properties->keys.size(); ++i)
            children.set(properties->keys[i], lower_schema(properties->vals[i],
                schema_path(schema_path(path, "properties"), properties->keys[i]), check, accept, depth + 1));
        out.set("properties", std::move(children));
    }
    for (const char *key : {"items", "additionalProperties"}) {
        if (const auto *child = source.get(key)) {
            if (std::string(key) == "additionalProperties" && child->is_bool()) continue;
            out.set(key, lower_schema(*child, schema_path(path, key), check, accept, depth + 1));
        }
    }
    if (const auto *branches = source.get("anyOf")) {
        if (!branches->is_array() || branches->arr.empty()) fail("anyOf", "anyOf requires a non-empty array of schemas");
        if (!check || !accept) fail("anyOf", "original-schema validation is not configured");
        // Validate sibling syntax even when it would not apply to a finite
        // scalar candidate (e.g. malformed required on a string-valued union).
        ajson siblings = ajson::jobj();
        for (size_t i = 0; i < out.keys.size(); ++i)
            if (out.keys[i] != "anyOf") siblings.set(out.keys[i], out.vals[i]);
        std::string sibling_error;
        if (!check(siblings, path, &sibling_error)) fail("schema", sibling_error);
        ajson values = ajson::jarr();
        for (size_t i = 0; i < branches->arr.size(); ++i) {
            const auto &branch = branches->arr[i];
            const ajson lowered = lower_schema(branch, schema_path(path, "anyOf") + "[" + std::to_string(i) + "]", check, accept, depth + 1);
            ajson candidates = ajson::jarr();
            if (const auto *en = lowered.get("enum")) candidates = *en;
            else if (const auto *constant = lowered.get("const")) candidates.push(*constant);
            else if (const auto *type = lowered.get("type"); type && type->is_string() && type->s == "null") candidates.push(ajson::jnull());
            else if (type && type->is_string() && type->s == "boolean") {
                candidates.push(ajson::jbool(false)); candidates.push(ajson::jbool(true));
            } else fail("anyOf", "union has a non-finite branch; native single-type schemas cannot express it exactly");
            for (const auto &candidate : candidates.arr) {
                std::string ignored;
                if (!accept(candidate, branch, &ignored)) continue;
                // Deduplication is only a size optimization, never branch selection.
                const auto encoded = ajson_dumps(candidate);
                if (std::none_of(values.arr.begin(), values.arr.end(), [&](const ajson &v) { return ajson_dumps(v) == encoded; }))
                    values.push(candidate);
            }
        }
        // Parent assertions intersect the union. Evaluate the ORIGINAL schema,
        // including all siblings; finite enumeration makes this exact.
        ajson admitted = ajson::jarr();
        for (const auto &candidate : values.arr) {
            std::string ignored;
            if (accept(candidate, source, &ignored)) admitted.push(candidate);
        }
        if (admitted.arr.empty()) fail("anyOf", "empty intersection cannot be represented by the native non-empty enum vocabulary");
        ajson replacement = ajson::jobj();
        for (const char *key : {"description", "title", "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly",
                               "minLength", "maxLength", "minimum", "maximum"})
            if (const auto *annotation = source.get(key)) replacement.set(key, *annotation);
        replacement.set("enum", std::move(admitted));
        out = std::move(replacement);
    }
    if (check) {
        std::string error;
        if (!check(out, path, &error)) fail("schema", error);
    }
    return out;
}

// A non-finite union is not expressible in one native single-type schema.
// Compile an EXACT union of whole argument schemas instead. Each alternative
// gets a reversible wire name at the provider boundary, never an argument
// coercion. The bound rejects the whole tool, never a subset of its domain.
inline std::vector<ajson> lower_schema_alternatives(
        const ajson &source, const std::string &path, const schema_check &check,
        const argument_check &accept, size_t depth = 0, size_t limit = 64) {
    try { return {lower_schema(source, path, check, accept, depth)}; }
    catch (const schema_failure &e) {
        // Only the expressiveness failure is recoverable by distribution.
        if (e.keyword != "anyOf" || std::string(e.what()).find("non-finite branch") == std::string::npos) throw;
    }
    if (depth > ALICED_JSON_MAX_DEPTH || !source.is_object())
        throw schema_failure(path, "schema", "invalid schema nesting");
    auto bounded = [&](size_t count) {
        if (count > limit) throw schema_failure(path, "anyOf",
            "exact union exceeds the 64 whole-tool alternatives bound; no branches were advertised");
    };
    if (const auto *branches = source.get("anyOf")) {
        // Finite intersections were handled above. Non-finite intersections
        // require a separate proof/compiler, not overwriting sibling assertions.
        for (const auto &key : source.keys)
            if (key != "anyOf" && key != "description" && key != "title" && key != "default" &&
                key != "examples" && key != "$comment" && key != "deprecated" &&
                key != "readOnly" && key != "writeOnly")
                throw schema_failure(path, "anyOf", "non-finite union with sibling assertions cannot be lowered exactly");
        std::vector<ajson> out;
        for (size_t i = 0; i < branches->arr.size(); ++i) {
            auto alternatives = lower_schema_alternatives(branches->arr[i],
                schema_path(path, "anyOf") + "[" + std::to_string(i) + "]", check, accept, depth + 1, limit);
            for (auto &branch : alternatives) {
                for (size_t k = 0; k < source.keys.size(); ++k)
                    if (source.keys[k] != "anyOf") branch.set(source.keys[k], source.vals[k]);
                out.push_back(std::move(branch)); bounded(out.size());
            }
        }
        return out;
    }
    // Distribution through properties is exact, including absent optional
    // properties. Distribution through items/additionalProperties is NOT:
    // [string, null] must not become homogeneous arrays. Keep rejecting these.
    ajson base = source;
    const auto *properties = source.get("properties");
    if (properties) base.set("properties", ajson::jobj());
    base = lower_schema(base, path, [](const ajson &, const std::string &, std::string *) { return true; }, accept, depth);
    std::vector<ajson> out{base};
    if (properties) for (size_t i = 0; i < properties->keys.size(); ++i) {
        auto choices = lower_schema_alternatives(properties->vals[i],
            schema_path(schema_path(path, "properties"), properties->keys[i]), check, accept, depth + 1, limit);
        if (choices.size() > limit / out.size()) bounded(limit + 1);
        std::vector<ajson> expanded;
        for (const auto &parent : out) for (const auto &child : choices) {
            ajson copy = parent, children = *parent.get("properties");
            children.set(properties->keys[i], child); copy.set("properties", std::move(children));
            expanded.push_back(std::move(copy));
        }
        out = std::move(expanded);
    }
    for (const auto &branch : out) if (check) {
        std::string error;
        if (!check(branch, path, &error)) throw schema_failure(path, "schema", error);
    }
    return out;
}

// Hoist documentation once per original tool when distributing a union.
// Assertions and enum/const VALUES are never traversed as schema metadata.
// Notes retain their exact original schema paths, including branch-local docs.
inline ajson hoist_schema_documentation(const ajson &schema, ajson *notes,
                                        const std::string &path = "$") {
    if (!schema.is_object()) return schema;
    ajson out = ajson::jobj(), own = ajson::jobj();
    for (size_t i = 0; i < schema.keys.size(); ++i) {
        const auto &key = schema.keys[i]; const auto &value = schema.vals[i];
        if (key == "description" || key == "title" || key == "examples" || key == "default" || key == "$comment") {
            own.set(key, value); continue;
        }
        if (key == "properties" && value.is_object()) {
            ajson properties = ajson::jobj();
            for (size_t k = 0; k < value.keys.size(); ++k)
                properties.set(value.keys[k], hoist_schema_documentation(value.vals[k], notes,
                    schema_path(schema_path(path, key), value.keys[k])));
            out.set(key, std::move(properties));
        } else if ((key == "items" || key == "additionalProperties") && value.is_object()) {
            out.set(key, hoist_schema_documentation(value, notes, schema_path(path, key)));
        } else if (key == "anyOf" && value.is_array()) {
            ajson branches = ajson::jarr();
            for (size_t k = 0; k < value.arr.size(); ++k)
                branches.push(hoist_schema_documentation(value.arr[k], notes,
                    schema_path(path, key) + "[" + std::to_string(k) + "]"));
            out.set(key, std::move(branches));
        } else out.set(key, value);
    }
    if (notes && !own.keys.empty()) notes->set(path, std::move(own));
    return out;
}
} // namespace axiom_codex
