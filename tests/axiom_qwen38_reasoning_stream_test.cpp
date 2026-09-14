// Actual tokenizer, phase handoff and SSE writer; CPU-only, no model/listener.
#define main axiom_api_main
#include "../tools/axiom_qwen38_api.cpp"
#undef main
#include <cassert>
#include <iostream>

static std::string available(int fd) {
    std::string out;
    char buffer[8192];
    ssize_t n;
    while ((n = recv(fd, buffer, sizeof(buffer), MSG_DONTWAIT)) > 0) out.append(buffer, n);
    return out;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    server_state state;
    state.max_context = 1048576u;
    state.default_context = 262144u;
    axiom_tokenizer_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.path = argv[1];
    config.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;
    config.name = "cpu-reasoning-stream";
    assert(axiom_tokenizer_open(&state.tokenizer, &config) == AXIOM_OK);
    state.tokenizer_info.abi_version = AXIOM_ABI_VERSION;
    assert(axiom_tokenizer_info_get(state.tokenizer, &state.tokenizer_info) == AXIOM_OK);
    std::vector<uint32_t> hidden;
    assert(append_text_ids(&state, "Private fixture reasoning.\n</think>", &hidden) == AXIOM_OK);
    size_t checks = 0;
    for (const std::string &text : {
            std::string("{\"solutions\":3,\"min_packs\":9,\"sixes\":3, \"nines\":5,\"twenties\":1}"),
            std::string("\n\nLeading and trailing spaces stay intact.  \n"),
            std::string("Literal </think> and <think> in visible text."),
            std::string("\nOne intentional newline."),
            std::string("  indented answer  \n"),
            std::string("café — 🌍 — 日本語")}) {
        std::vector<uint32_t> visible;
        // The native template supplies exactly two LF after the first close.
        // Remove only that protocol separator, preserving the actual answer.
        assert(append_text_ids(&state, "\n\n" + text, &visible) == AXIOM_OK);
        for (size_t split = 0; split <= visible.size(); ++split) {
            int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
            axiom_codex::tool_wire_map bridge;
            live_responses_stream stream;
            assert(start_live_responses_stream(&stream, fd[0], 209, "phase-handoff", &bridge));
            std::string wire = available(fd[1]);
            generation_stream_sink sink;
            sink.user = &stream;
            sink.emit = &live_responses_emit_token;
            sink.reasoning_separator_pending = true;
            generation_result thinking;
            thinking.ids = hidden;
            thinking.ids.insert(thinking.ids.end(), visible.begin(), visible.begin() + split);
            const auto accounted = thinking.ids;
            std::string error;
            assert(emit_thinking_visible_prefix(&state, thinking, split, &sink, &error) == AXIOM_OK);
            assert(thinking.ids == accounted);
            wire += available(fd[1]);
            generation_result second;
            for (size_t i = split; i < visible.size(); ++i) {
                assert(append_generated_token(&state, &second, visible[i], &sink, &error) == AXIOM_OK);
                wire += available(fd[1]);
            }
            assert(drain_generation_utf8(&sink, true, &error) == AXIOM_OK);
            wire += available(fd[1]);
            assert(stream.raw_text == text);
            generation_result result;
            result.text = "Private fixture reasoning.\n</think>\n\n" + text;
            result.thinking_budget = 1024u;
            result.finish_reason = "stop";
            ajson response = live_responses_stub(&stream);
            assert(finish_live_responses_stream(&stream, response, result, nullptr, nullptr));
            wire += available(fd[1]);
            assert(wire.find("stream_content_mismatch") == std::string::npos);
            assert(wire.find("Private fixture reasoning") == std::string::npos);
            assert(wire.find("event: response.completed") != std::string::npos);
            assert(stream.raw_text == text);
            assert(visible_model_text(result.text, true) == text);
            assert(reasoning_model_text(result.text, true) == "Private fixture reasoning.");
            assert(visible_model_text(text, false) == text);
            assert(reasoning_model_text(text, false).empty());
            close(fd[0]); close(fd[1]);
            ++checks;
        }
    }
    generation_result accounting;
    std::string stage;
    assert(append_text_ids(&state, "reasoning</think>\n\nLiteral </think> stays.", &accounting.ids) == AXIOM_OK);
    assert(refresh_generation_text(&state, &accounting, &stage) == AXIOM_OK);
    std::vector<uint32_t> first_reasoning;
    assert(append_text_ids(&state, "reasoning</think>", &first_reasoning) == AXIOM_OK);
    assert(accounting.thinking_tokens == first_reasoning.size());
    assert(accounting.visible_output_tokens == accounting.ids.size() - first_reasoning.size());
    ++checks;
    // A partial tool marker at the phase boundary must remain off the text
    // stream; the real writer buffers its suffix until the complete marker.
    for (size_t split = 1; split < 11; ++split) {
        const std::string call = "<tool_call>\n<function=qa>\n</function>\n</tool_call>";
        int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        live_responses_stream stream;
        assert(start_live_responses_stream(&stream, fd[0], 210, "tool-phase"));
        assert(live_responses_emit_token(&stream, call.substr(0, split)));
        assert(live_responses_emit_token(&stream, call.substr(split)));
        assert(stream.emitted_bytes == 0u);
        assert(available(fd[1]).find("response.output_text.delta") == std::string::npos);
        close(fd[0]); close(fd[1]); ++checks;
    }
    axiom_tokenizer_close(state.tokenizer);
    state.tokenizer = nullptr;
    std::cout << "REASONING_STREAM_HANDOFF PASS checks=" << checks << " all-token-splits unicode whitespace tool-marker\n";
}
