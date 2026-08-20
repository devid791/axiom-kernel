/* Native text-generation entry point for unsloth/Qwen3.8-27B-NVFP4.
 *
 * This is deliberately an integrated path: tokenizer -> prompt prefill ->
 * native 64-layer decoder -> greedy token decode.  It is not an HTTP server
 * or a scheduler yet.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"

namespace {

constexpr uint32_t kDefaultMaxNew = 16u;
constexpr uint32_t kDefaultMaxContext = 64u;
constexpr uint32_t kTokenCapacity = 4096u;
constexpr uint32_t kTextCapacity = 65536u;

int fail(const char *what, int rc = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-generate: %s%s%s\n", what,
                 rc == AXIOM_OK ? "" : ": ", rc == AXIOM_OK ? "" : axiom_status_string(rc));
    return 1;
}

bool parse_u32(const char *text, uint32_t *out) {
    if (!text || !out || !*text) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

bool is_stop(uint32_t token, const axiom_tokenizer_info &info) {
    return token == info.endoftext_token_id || token == info.im_end_token_id;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: %s MODEL_DIR PROMPT [MAX_NEW] [MAX_CONTEXT]\n", argv[0]);
        return 2;
    }

    uint32_t max_new = kDefaultMaxNew;
    uint32_t max_context = kDefaultMaxContext;
    if ((argc >= 4 && (!parse_u32(argv[3], &max_new) || max_new == 0u)) ||
        (argc == 5 && (!parse_u32(argv[4], &max_context) || max_context == 0u))) {
        return fail("MAX_NEW and MAX_CONTEXT must be non-zero unsigned integers");
    }

    axiom_tokenizer_config tokenizer_config{};
    tokenizer_config.abi_version = AXIOM_ABI_VERSION;
    tokenizer_config.path = argv[1];
    tokenizer_config.name = "qwen3.8";
    tokenizer_config.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;

    axiom_tokenizer *tokenizer = nullptr;
    int rc = axiom_tokenizer_open(&tokenizer, &tokenizer_config);
    if (rc != AXIOM_OK) return fail("tokenizer open", rc);

    axiom_tokenizer_info tokenizer_info{};
    tokenizer_info.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_tokenizer_info_get(tokenizer, &tokenizer_info);
    if (rc != AXIOM_OK) {
        axiom_tokenizer_close(tokenizer);
        return fail("tokenizer info", rc);
    }

    std::vector<uint32_t> prompt_ids(kTokenCapacity);
    uint32_t prompt_count = 0u;
    rc = axiom_tokenizer_encode_text(
            tokenizer, argv[2], prompt_ids.data(), static_cast<uint32_t>(prompt_ids.size()), &prompt_count);
    if (rc != AXIOM_OK || prompt_count == 0u) {
        axiom_tokenizer_close(tokenizer);
        return fail(rc == AXIOM_OK ? "prompt tokenized to zero tokens" : "prompt tokenization", rc);
    }
    if (prompt_count >= max_context || max_new > max_context - prompt_count + 1u) {
        axiom_tokenizer_close(tokenizer);
        return fail("MAX_CONTEXT is too small for prompt plus requested generation");
    }

    const auto load_start = std::chrono::steady_clock::now();
    axiom_qwen38_model *model = nullptr;
    rc = axiom_qwen38_model_create(argv[1], 0, max_context, &model);
    const auto load_end = std::chrono::steady_clock::now();
    if (rc != AXIOM_OK) {
        axiom_tokenizer_close(tokenizer);
        return fail("model create", rc);
    }

    uint32_t next_token = 0u;
    float next_logit = 0.0f;
    const auto prefill_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0u; i < prompt_count; ++i) {
        rc = axiom_qwen38_model_forward_token(model, prompt_ids[i], &next_token, &next_logit);
        if (rc != AXIOM_OK) break;
    }
    const auto prefill_end = std::chrono::steady_clock::now();

    std::vector<uint32_t> generated;
    generated.reserve(max_new);
    const auto decode_start = std::chrono::steady_clock::now();
    while (rc == AXIOM_OK && generated.size() < max_new) {
        generated.push_back(next_token);
        if (is_stop(next_token, tokenizer_info)) break;
        if (axiom_qwen38_model_position(model) >= max_context) break;
        rc = axiom_qwen38_model_forward_token(model, next_token, &next_token, &next_logit);
    }
    const auto decode_end = std::chrono::steady_clock::now();

    std::vector<char> text(kTextCapacity, '\0');
    uint32_t text_bytes = 0u;
    if (rc == AXIOM_OK && !generated.empty()) {
        rc = axiom_tokenizer_decode_ids(
                tokenizer, generated.data(), static_cast<uint32_t>(generated.size()), text.data(),
                static_cast<uint32_t>(text.size()), &text_bytes);
    }

    const double load_seconds = std::chrono::duration<double>(load_end - load_start).count();
    const double prefill_seconds = std::chrono::duration<double>(prefill_end - prefill_start).count();
    const double decode_seconds = std::chrono::duration<double>(decode_end - decode_start).count();
    if (rc == AXIOM_OK) {
        const double decode_tok_s = decode_seconds > 0.0 ? generated.size() / decode_seconds : 0.0;
        std::printf("prompt_tokens=%u generated_tokens=%zu load_seconds=%.3f prefill_seconds=%.3f "
                    "decode_seconds=%.3f decode_tok_s=%.3f\n",
                    prompt_count, generated.size(), load_seconds, prefill_seconds, decode_seconds, decode_tok_s);
        std::printf("text=");
        if (text_bytes != 0u) std::fwrite(text.data(), 1u, text_bytes, stdout);
        std::fputc('\n', stdout);
        std::printf("ids=");
        for (size_t i = 0u; i < generated.size(); ++i) {
            std::printf("%s%u", i == 0u ? "" : ",", generated[i]);
        }
        std::fputc('\n', stdout);
    }

    axiom_qwen38_model_destroy(model);
    axiom_tokenizer_close(tokenizer);
    return rc == AXIOM_OK ? 0 : fail("native generation", rc);
}
