#include "axiom/qwen38_spec_identity.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using axiom::qwen38::spec_identity::build_sidecar_json;
using axiom::qwen38::spec_identity::compute;
using axiom::qwen38::spec_identity::qualification_evidence;
using axiom::qwen38::spec_identity::result;
using axiom::qwen38::spec_identity::runtime_contract;
using axiom::qwen38::spec_identity::validate_sidecar;

namespace {

struct fixture {
    fs::path root;
    fs::path target;
    fs::path draft;
};

class test_context {
public:
    void check(bool condition, const std::string &message) {
        if (condition) return;
        ++failures_;
        std::cerr << "FAIL: " << message << '\n';
    }

    void contains(
            const std::string &actual,
            const std::string &needle,
            const std::string &message) {
        check(actual.find(needle) != std::string::npos,
              message + " (actual: " + actual + ")");
    }

    int failures() const { return failures_; }

private:
    int failures_ = 0;
};

void write_file(const fs::path &path, const std::string &content) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("open failed: " + path.string());
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!stream) throw std::runtime_error("write failed: " + path.string());
}

std::string read_file(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("read open failed: " + path.string());
    return std::string(std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>());
}

std::string target_config_json() {
    std::string layers = "[";
    for (unsigned index = 0; index < 64u; ++index) {
        if (index) layers += ',';
        layers += (index % 4u == 3u) ? "\"full_attention\"" :
                                      "\"linear_attention\"";
    }
    layers += ']';
    return
        "{\"model_type\":\"qwen3_5\",\"text_config\":{"
        "\"model_type\":\"qwen3_5_text\","
        "\"hidden_size\":5120,\"intermediate_size\":17408,"
        "\"num_hidden_layers\":64,\"num_attention_heads\":24,"
        "\"num_key_value_heads\":4,\"head_dim\":256,"
        "\"vocab_size\":248320,\"max_position_embeddings\":262144,"
        "\"linear_num_key_heads\":16,\"linear_key_head_dim\":128,"
        "\"linear_num_value_heads\":48,\"linear_value_head_dim\":128,"
        "\"linear_conv_kernel_dim\":4,\"rms_norm_eps\":0.000001,"
        "\"partial_rotary_factor\":0.25,\"layer_types\":" + layers + ","
        "\"rope_parameters\":{\"rope_type\":\"default\","
        "\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,"
        "\"mrope_interleaved\":true,\"mrope_section\":[11,11,10]}}}";
}

std::string draft_config_json() {
    return
        "{\"model_type\":\"qwen3\",\"hidden_size\":5120,"
        "\"intermediate_size\":10240,\"num_hidden_layers\":5,"
        "\"num_attention_heads\":40,\"num_key_value_heads\":8,"
        "\"head_dim\":128,\"vocab_size\":248320,"
        "\"max_position_embeddings\":262144,\"block_size\":7,"
        "\"num_target_layers\":64,\"rms_norm_eps\":0.000001,"
        "\"enable_confidence_head\":true,"
        "\"confidence_head_with_markov\":true,"
        "\"layer_types\":[\"full_attention\",\"full_attention\","
        "\"full_attention\",\"full_attention\",\"full_attention\"],"
        "\"rope_parameters\":{\"rope_type\":\"yarn\","
        "\"rope_theta\":10000000,\"factor\":32.0,\"beta_fast\":32.0,"
        "\"beta_slow\":1.0,\"original_max_position_embeddings\":8192},"
        "\"dflash_config\":{\"attention_mode\":\"gqa\","
        "\"projector_type\":\"dspark\",\"markov_head_type\":\"vanilla\","
        "\"mask_token_id\":248077,\"markov_rank\":256,"
        "\"confidence_head_alpha\":1.0,\"enable_confidence_head\":true,"
        "\"confidence_head_with_markov\":true,"
        "\"target_layer_ids\":[4,16,28,40,52]}}";
}

fixture make_fixture(const fs::path &parent, const std::string &name) {
    fixture value;
    value.root = parent / name;
    value.target = value.root / "target";
    value.draft = value.root / "draft";
    fs::create_directories(value.target);
    fs::create_directories(value.draft);

    write_file(value.target / "config.json", target_config_json());
    write_file(value.target / "hf_quant_config.json",
               "{\"producer\":\"fixture\",\"quantization\":\"nvfp4\"}");
    write_file(value.target / "model.safetensors.index.json",
               "{\"metadata\":{\"total_size\":11},\"weight_map\":{"
               "\"model.a\":\"model-00001-of-00002.safetensors\","
               "\"model.b\":\"model-00002-of-00002.safetensors\","
               "\"model.c\":\"model-00001-of-00002.safetensors\"}}");
    write_file(value.target / "model-00001-of-00002.safetensors", "TARGET-A\0x");
    write_file(value.target / "model-00002-of-00002.safetensors", "TARGET-B\0y");
    write_file(value.target / "unreferenced.safetensors", "NOT-IN-IDENTITY");
    write_file(value.target / "tokenizer.json",
               "{\"version\":\"1.0\",\"model\":{\"vocab\":{\"a\":0}}}");
    write_file(value.target / "tokenizer_config.json",
               "{\"tokenizer_class\":\"Fixture\","
               "\"chat_template\":\"INLINE {{ messages }}\"}");
    write_file(value.target / "vocab.json", "{\"a\":0,\"b\":1}");
    write_file(value.target / "merges.txt", "#version: 0.2\na b\n");
    write_file(value.target / "chat_template.jinja", "FILE {{ messages }}\n");

    write_file(value.draft / "config.json", draft_config_json());
    write_file(value.draft / "model.safetensors", "DRAFT-MODEL\0z");
    return value;
}

bool compute_fixture(const fixture &value, result *identity, std::string *error) {
    return compute(value.target.string(), value.draft.string(), identity, error);
}

std::string replace_once(
        std::string value,
        const std::string &before,
        const std::string &after) {
    const std::size_t offset = value.find(before);
    if (offset == std::string::npos) {
        throw std::runtime_error("replacement token not found: " + before);
    }
    value.replace(offset, before.size(), after);
    return value;
}

const axiom::qwen38::spec_identity::artifact_evidence *find_artifact(
        const result &identity,
        const std::string &logical_path) {
    for (const auto &artifact : identity.artifacts) {
        if (artifact.logical_path == logical_path) return &artifact;
    }
    return nullptr;
}

void test_golden_and_relocation(
        test_context &test,
        const fs::path &root,
        result *golden_identity) {
    const fixture first = make_fixture(root, "golden-a");
    const fixture second = make_fixture(root, "golden-b");
    std::string error;
    result first_identity;
    result second_identity;
    test.check(compute_fixture(first, &first_identity, &error),
               "golden fixture computes: " + error);
    error.clear();
    test.check(compute_fixture(second, &second_identity, &error),
               "relocated fixture computes: " + error);
    test.check(first_identity.fingerprint == second_identity.fingerprint,
               "fingerprint is independent of absolute root path");
    test.check(first_identity.components.target_sha256 ==
                       second_identity.components.target_sha256,
               "target component is deterministic");
    test.check(first_identity.artifacts.size() == 12u,
               "artifact evidence contains only required and present bundle files");
    test.check(find_artifact(first_identity,
                             "target/unreferenced.safetensors") == nullptr,
               "unreferenced target shard is excluded");
    const auto *template_artifact = find_artifact(
            first_identity, "chat_template/chat_template.jinja");
    test.check(template_artifact && template_artifact->source_kind == "regular-file",
               "chat_template.jinja is preferred");

    /* Golden deliberately binds the canonical magic, BE lengths, component
     * record names, runtime contract and fixture bytes. */
    constexpr const char *expected =
            "specfp1:7f1211392ed9f5b048055e510271bd4d717c5a99490dc6482d8c2733e5af114d";
    test.check(first_identity.fingerprint == expected,
               "specfp1 deterministic golden (actual: " +
                       first_identity.fingerprint + ")");
    *golden_identity = std::move(first_identity);
}

void test_component_mutations(
        test_context &test,
        const fs::path &root,
        const result &golden) {
    struct mutation_case {
        const char *name;
        const char *relative;
        const char *suffix;
        const std::string axiom::qwen38::spec_identity::component_hashes::*component;
    };
    const mutation_case cases[] = {
        {"target", "target/model-00001-of-00002.safetensors", "-mutated",
         &axiom::qwen38::spec_identity::component_hashes::target_sha256},
        {"draft", "draft/model.safetensors", "-mutated",
         &axiom::qwen38::spec_identity::component_hashes::draft_sha256},
        {"tokenizer", "target/tokenizer.json", " ",
         &axiom::qwen38::spec_identity::component_hashes::tokenizer_sha256},
        {"template", "target/chat_template.jinja", "# mutation\n",
         &axiom::qwen38::spec_identity::component_hashes::chat_template_sha256},
    };
    for (const auto &item : cases) {
        const fixture value = make_fixture(root, std::string("mutation-") + item.name);
        const fs::path path = value.root / item.relative;
        write_file(path, read_file(path) + item.suffix);
        result changed;
        std::string error;
        test.check(compute_fixture(value, &changed, &error),
                   std::string(item.name) + " mutation computes: " + error);
        test.check(changed.fingerprint != golden.fingerprint,
                   std::string(item.name) + " mutation changes fingerprint");
        test.check(changed.components.*(item.component) !=
                           golden.components.*(item.component),
                   std::string(item.name) + " mutation changes its component hash");
    }

    const fixture unused = make_fixture(root, "mutation-unreferenced");
    write_file(unused.target / "unreferenced.safetensors", "MUTATED-BUT-UNREFERENCED");
    result unchanged;
    std::string error;
    test.check(compute_fixture(unused, &unchanged, &error),
               "unreferenced mutation computes: " + error);
    test.check(unchanged.fingerprint == golden.fingerprint,
               "unreferenced shard does not affect identity");

    const fixture fallback = make_fixture(root, "template-fallback");
    fs::remove(fallback.target / "chat_template.jinja");
    result fallback_identity;
    error.clear();
    test.check(compute_fixture(fallback, &fallback_identity, &error),
               "tokenizer_config template fallback computes: " + error);
    const auto *fallback_artifact = find_artifact(
            fallback_identity,
            "chat_template/tokenizer_config.json#chat_template");
    test.check(fallback_artifact && fallback_artifact->source_kind == "json-string",
               "fallback hashes decoded tokenizer_config chat_template value");
    test.check(fallback_identity.components.chat_template_sha256 !=
                       golden.components.chat_template_sha256,
               "fallback template content has its own identity");

    const fixture runtime_value = make_fixture(root, "runtime-mutation");
    runtime_contract changed_runtime;
    changed_runtime.dspark_compute_layout_version += 1u;
    result runtime_identity;
    error.clear();
    test.check(compute(runtime_value.target.string(), runtime_value.draft.string(),
                       changed_runtime, &runtime_identity, &error),
               "runtime ABI mutation computes a distinct contract: " + error);
    test.check(runtime_identity.components.runtime_sha256 !=
                       golden.components.runtime_sha256 &&
               runtime_identity.fingerprint != golden.fingerprint,
               "runtime mutation changes runtime hash and fingerprint");
}

void test_unsafe_artifacts(test_context &test, const fs::path &root) {
    {
        const fixture value = make_fixture(root, "path-traversal");
        write_file(value.target / "model.safetensors.index.json",
                   "{\"weight_map\":{\"model.a\":\"../escape.safetensors\"}}");
        result ignored;
        std::string error;
        test.check(!compute_fixture(value, &ignored, &error),
                   "path traversal is rejected");
        test.contains(error, "target shard path is not a basename",
                      "path traversal has stable diagnostic");
    }
    {
        const fixture value = make_fixture(root, "symlink-shard");
        fs::remove(value.target / "model-00002-of-00002.safetensors");
        if (symlink("model-00001-of-00002.safetensors",
                    (value.target / "model-00002-of-00002.safetensors").c_str()) != 0) {
            throw std::runtime_error("symlink fixture failed");
        }
        result ignored;
        std::string error;
        test.check(!compute_fixture(value, &ignored, &error),
                   "symlink shard is rejected");
        test.contains(error, "artifact is a symlink",
                      "symlink has stable diagnostic");
    }
    {
        const fixture value = make_fixture(root, "hardlink-alias");
        fs::remove(value.target / "model-00002-of-00002.safetensors");
        if (link((value.target / "model-00001-of-00002.safetensors").c_str(),
                 (value.target / "model-00002-of-00002.safetensors").c_str()) != 0) {
            throw std::runtime_error("hardlink fixture failed");
        }
        result ignored;
        std::string error;
        test.check(!compute_fixture(value, &ignored, &error),
                   "incompatible hard-link aliases are rejected");
        test.contains(error, "incompatible duplicate artifact aliases",
                      "hard-link alias has stable diagnostic");
    }
    {
        const fixture value = make_fixture(root, "duplicate-index-key");
        write_file(value.target / "model.safetensors.index.json",
                   "{\"weight_map\":{\"model.a\":"
                   "\"model-00001-of-00002.safetensors\",\"model.a\":"
                   "\"model-00002-of-00002.safetensors\"}}");
        result ignored;
        std::string error;
        test.check(!compute_fixture(value, &ignored, &error),
                   "duplicate JSON tensor key is rejected");
        test.contains(error, "duplicate object key",
                      "duplicate JSON key has stable diagnostic");
    }
    {
        const fixture value = make_fixture(root, "missing-shard");
        fs::remove(value.target / "model-00002-of-00002.safetensors");
        result ignored;
        std::string error;
        test.check(!compute_fixture(value, &ignored, &error),
                   "missing referenced shard is rejected");
        test.contains(error, "required artifact missing",
                      "missing shard has stable diagnostic");
    }
}

void test_sidecar(
        test_context &test,
        const fs::path &root,
        const result &identity) {
    const std::string evidence(64u, 'a');
    std::string sidecar;
    std::string error;
    const std::string source_commit(40u, 'd');
    test.check(build_sidecar_json(identity, source_commit, evidence,
                                  &sidecar, &error),
               "qualified sidecar builds: " + error);
    const fs::path valid_path = root / "valid-sidecar.json";
    write_file(valid_path, sidecar);
    qualification_evidence proof;
    error.clear();
    test.check(validate_sidecar(valid_path.string(), identity, &proof, &error),
               "qualified sidecar validates: " + error);
    test.check(proof.status == "pass" && proof.evidence_sha256 == evidence &&
                       proof.source_commit == source_commit,
               "qualification evidence is returned exactly");

    error.clear();
    test.check(!validate_sidecar((root / "missing-sidecar.json").string(),
                                 identity, nullptr, &error),
               "missing sidecar is rejected");
    test.check(error == "spec identity: sidecar missing",
               "missing sidecar diagnostic is stable");

    const fs::path unqualified_path = root / "unqualified-sidecar.json";
    write_file(unqualified_path,
               replace_once(sidecar, "\"status\":\"pass\"",
                             "\"status\":\"fail\""));
    error.clear();
    test.check(!validate_sidecar(unqualified_path.string(), identity,
                                 nullptr, &error),
               "unqualified sidecar is rejected");
    test.check(error == "spec identity: sidecar qualification is not pass",
               "unqualified diagnostic is stable");

    const fs::path runtime_path = root / "runtime-mutated-sidecar.json";
    write_file(runtime_path,
               replace_once(sidecar, "\"abi.axiom\":\"1\"",
                             "\"abi.axiom\":\"9\""));
    error.clear();
    test.check(!validate_sidecar(runtime_path.string(), identity,
                                 nullptr, &error),
               "runtime-mutated sidecar is rejected");
    test.contains(error, "sidecar value mismatch: runtime.abi.axiom",
                  "runtime mutation diagnostic is stable");

    const fs::path component_path = root / "component-mutated-sidecar.json";
    write_file(component_path,
               replace_once(sidecar, identity.components.draft_sha256,
                             std::string(64u, 'b')));
    error.clear();
    test.check(!validate_sidecar(component_path.string(), identity,
                                 nullptr, &error),
               "component-mutated sidecar is rejected");
    test.contains(error, "sidecar value mismatch: components.draft_sha256",
                  "component mutation diagnostic is stable");

    const fs::path schema_path = root / "schema-expanded-sidecar.json";
    write_file(schema_path,
               replace_once(sidecar, "{\"schema\"",
                             "{\"unknown\":true,\"schema\""));
    error.clear();
    test.check(!validate_sidecar(schema_path.string(), identity,
                                 nullptr, &error),
               "unknown sidecar field is rejected");
    test.check(error == "spec identity: sidecar schema mismatch at root",
               "exact sidecar schema diagnostic is stable");

    const fs::path duplicate_path = root / "duplicate-sidecar.json";
    write_file(duplicate_path,
               replace_once(sidecar, "{\"schema\"",
                             "{\"schema\":\"axiom-speculative-contract-v1\",\"schema\""));
    error.clear();
    test.check(!validate_sidecar(duplicate_path.string(), identity,
                                 nullptr, &error),
               "duplicate sidecar field is rejected");
    test.contains(error, "duplicate object key",
                  "duplicate sidecar field diagnostic is stable");

    std::string ignored;
    error.clear();
    test.check(!build_sidecar_json(identity, "", evidence, &ignored, &error),
               "empty source commit is rejected");
    test.check(error ==
                       "spec identity: sidecar qualification source_commit is invalid",
               "empty source commit diagnostic is stable");
    error.clear();
    test.check(!build_sidecar_json(identity, source_commit, "xyz", &ignored, &error),
               "invalid evidence hash is rejected");
}

void test_api_boundaries(test_context &test, const fs::path &root) {
    result output;
    std::string error;
    test.check(!compute("", root.string(), &output, &error),
               "empty target root fails without throwing");
    test.check(error == "spec identity: target root is empty",
               "empty target error is stable");
    error.clear();
    test.check(!compute(root.string(), root.string(), nullptr, &error),
               "null result fails without throwing");
    test.check(error == "spec identity: missing result output",
               "null result error is stable");
}

}  // namespace

int main() {
    char pattern[] = "/tmp/axiom-spec-identity-test.XXXXXX";
    char *created = mkdtemp(pattern);
    if (!created) {
        std::cerr << "mkdtemp failed: " << std::strerror(errno) << '\n';
        return 2;
    }
    const fs::path root(created);
    test_context test;
    try {
        result golden;
        test_golden_and_relocation(test, root, &golden);
        test_component_mutations(test, root, golden);
        test_unsafe_artifacts(test, root);
        test_sidecar(test, root, golden);
        test_api_boundaries(test, root);
    } catch (const std::exception &exception) {
        std::cerr << "UNCAUGHT TEST ERROR: " << exception.what() << '\n';
        fs::remove_all(root);
        return 2;
    }
    fs::remove_all(root);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " test(s) failed\n";
        return 1;
    }
    std::cout << "axiom_qwen38_spec_identity_test: PASS\n";
    return 0;
}
