#include "axiom/axiom.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

[[nodiscard]] bool require(bool condition, const char *message) {
    if (!condition) std::fprintf(stderr, "qwen4exp-tokenizer-test: %s\n", message);
    return condition;
}

struct expected_token {
    const char *text;
    std::uint32_t id;
};

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_ROOT\n", argv[0]);
        return 2;
    }

    axiom_tokenizer_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.path = argv[1];
    config.name = "qwen4-exp-pinned";
    config.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;

    axiom_tokenizer *tokenizer = nullptr;
    int status = axiom_tokenizer_open(&tokenizer, &config);
    if (!require(status == AXIOM_OK && tokenizer != nullptr,
                 "cannot open pinned HF tokenizer")) {
        return 1;
    }

    axiom_tokenizer_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    status = axiom_tokenizer_info_get(tokenizer, &info);
    bool ok = require(status == AXIOM_OK, "cannot read tokenizer metadata") &&
              require(info.vocab_size == 248044u, "base vocabulary mismatch") &&
              require(info.added_tokens == 33u, "added-token count mismatch") &&
              require(info.endoftext_token_id == 248044u, "EOS mismatch") &&
              require(info.im_start_token_id == 248045u, "im_start mismatch") &&
              require(info.im_end_token_id == 248046u, "im_end mismatch") &&
              require(info.tool_call_token_id == 248058u, "tool_call mismatch") &&
              require(info.tool_call_end_token_id == 248059u,
                      "tool_call_end mismatch");

    constexpr std::array<expected_token, 9> expected{{
        {"<|vision_start|>", 248053u},
        {"<|vision_end|>", 248054u},
        {"<|vision_pad|>", 248055u},
        {"<|image_pad|>", 248056u},
        {"<|video_pad|>", 248057u},
        {"<tool_call>", 248058u},
        {"</tool_call>", 248059u},
        {"<think>", 248068u},
        {"</think>", 248069u},
    }};
    for (const expected_token &entry : expected) {
        std::uint32_t id = 0u;
        status = axiom_tokenizer_token_id(tokenizer, entry.text, &id);
        ok = require(status == AXIOM_OK && id == entry.id,
                     "special-token mapping mismatch") && ok;
    }

    constexpr char prompt[] = "Hello, Axiom — ciao!\n工具 ✓";
    std::vector<std::uint32_t> ids(512u);
    std::uint32_t count = 0u;
    status = axiom_tokenizer_encode_text(
            tokenizer, prompt, ids.data(), static_cast<std::uint32_t>(ids.size()),
            &count);
    ok = require(status == AXIOM_OK && count > 0u,
                 "UTF-8 prompt encode failed") && ok;
    ids.resize(count);
    for (const std::uint32_t id : ids) {
        ok = require(id < 248320u, "encoded token exceeds model vocabulary") && ok;
    }

    std::vector<char> decoded(4096u, '\0');
    std::uint32_t decoded_bytes = 0u;
    status = axiom_tokenizer_decode_ids(
            tokenizer, ids.data(), static_cast<std::uint32_t>(ids.size()),
            decoded.data(), static_cast<std::uint32_t>(decoded.size()),
            &decoded_bytes);
    ok = require(status == AXIOM_OK, "UTF-8 prompt decode failed") && ok;
    ok = require(decoded_bytes == std::strlen(prompt) &&
                         std::memcmp(decoded.data(), prompt, decoded_bytes) == 0,
                 "UTF-8 encode/decode is not byte-exact") && ok;

    std::vector<std::uint32_t> repeated(512u);
    std::uint32_t repeated_count = 0u;
    status = axiom_tokenizer_encode_text(
            tokenizer, prompt, repeated.data(),
            static_cast<std::uint32_t>(repeated.size()), &repeated_count);
    ok = require(status == AXIOM_OK && repeated_count == ids.size() &&
                         std::memcmp(repeated.data(), ids.data(),
                                     ids.size() * sizeof(std::uint32_t)) == 0,
                 "tokenization is not deterministic") && ok;

    axiom_tokenizer_close(tokenizer);
    if (!ok) return 1;
    std::printf(
            "qwen4exp-tokenizer-test: PASS base_vocab=%u added=%u "
            "model_vocab=248320 encoded=%u tokenizer_hash=0x%016llx "
            "chat_template_hash=0x%016llx deterministic=2/2\n",
            info.vocab_size, info.added_tokens, count,
            static_cast<unsigned long long>(info.tokenizer_hash),
            static_cast<unsigned long long>(info.chat_template_hash));
    return 0;
}
