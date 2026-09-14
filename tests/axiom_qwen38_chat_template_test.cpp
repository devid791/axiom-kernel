// CPU-only prompt oracle against the native Qwen3.8 ChatML contract.
#define main axiom_api_main
#include "../tools/axiom_qwen38_api.cpp"
#undef main
#include <cassert>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    server_state state;
    state.max_context = 1048576u;
    state.default_context = 262144u;
    state.no_think = true;
    axiom_tokenizer_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.path = argv[1]; config.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;
    assert(axiom_tokenizer_open(&state.tokenizer, &config) == AXIOM_OK);
    state.tokenizer_info.abi_version = AXIOM_ABI_VERSION;
    assert(axiom_tokenizer_info_get(state.tokenizer, &state.tokenizer_info) == AXIOM_OK);
    const std::string low = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    const std::string high = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    size_t checks = 0, failures = 0;
    const auto check = [&](bool pass, const std::string &name) {
        ++checks; if (!pass) ++failures;
        std::cout << name << " status=" << (pass ? "PASS" : "FAIL") << '\n';
    };
    const auto message = [](const char *role, const char *text) {
        ajson item = ajson::jobj(); item.set("role", ajson::jstr(role));
        item.set("content", ajson::jstr(text)); return item;
    };
    const auto turn = [](const std::string &role, const std::string &text) {
        return "<|im_start|>" + role + "\n" + text + "<|im_end|>\n";
    };
    for (const std::string effort : {"ultra-fast", "minimal", "low", "medium", "high", "xhigh", "max"}) {
        const std::string conditioning = effort == "minimal" || effort == "low" ? low :
                effort == "ultra-fast" || effort == "medium" ? "" : high;
        const std::string prefix = "<|im_start|>assistant\n" + std::string(
                effort == "ultra-fast" ? "<think>\n\n</think>\n\n" : "<think>\n");
        for (const bool history : {false, true}) for (int system_kind = 0; system_kind < 3; ++system_kind) {
            ajson payload = ajson::jobj(), messages = ajson::jarr();
            payload.set("reasoning_effort", ajson::jstr(effort));
            const std::string system_text = system_kind == 0 ? "" : "Keep these instructions.";
            if (system_kind == 1) messages.push(message("system", system_text.c_str()));
            if (system_kind == 2) payload.set("system", ajson::jstr(system_text));
            messages.push(message("user", "Hello."));
            const std::string combined = conditioning +
                    (!conditioning.empty() && !system_text.empty() ? "\n\n" : "") + system_text;
            std::string expected = combined.empty() ? "" : turn("system", combined);
            expected += turn("user", "Hello.");
            if (history) {
                messages.push(message("assistant", "Hi."));
                messages.push(message("user", "Continue."));
                expected += turn("assistant", "<think>\n\n</think>\n\nHi.") + turn("user", "Continue.");
            }
            payload.set("messages", messages); expected += prefix;
            std::vector<uint32_t> actual, suffix, reference;
            std::string error; uint32_t original = 0; bool compacted = false;
            const bool built = make_chat_ids(&state, payload, false, &actual, &suffix, &error, &original, &compacted);
            assert(append_text_ids(&state, expected, &reference) == AXIOM_OK);
            check(built && error.empty() && !compacted && actual == reference,
                    effort + (history ? "/history_template/" : "/single_template/") + std::to_string(system_kind));
            std::shared_ptr<vision_request> vision;
            std::vector<uint32_t> multimodal;
            error.clear();
            const bool encoded_mm = encode_multimodal_chat_ids(&state, payload, false,
                    &multimodal, &vision, &error, &original, &compacted);
            check(encoded_mm && error.empty() && multimodal == reference && vision &&
                    vision->embedding_slots.size() == reference.size() &&
                    std::all_of(vision->embedding_slots.begin(), vision->embedding_slots.end(),
                            [](int32_t slot) { return slot == -1; }), effort + "/multimodal_text_slot_parity");
            if (history) {
                std::vector<uint32_t> expected_suffix;
                assert(append_text_ids(&state, "\n" + turn("user", "Continue.") + prefix,
                        &expected_suffix) == AXIOM_OK);
                check(suffix == expected_suffix, effort + "/native_resume_separator");
            }
        }
    }
    ajson tool_payload;
    std::string parse_error;
    assert(ajson_parse(R"JSON({"reasoning_effort":"low","system":"Top instructions.","messages":[{"role":"system","content":"Initial instructions."},{"role":"user","content":"Read both."},{"role":"assistant","content":"","reasoning_content":"Check both files.","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a"}}},{"function":{"name":"read_file","arguments":{"path":"b"}}}]},{"role":"tool","content":"one"},{"role":"tool","content":"two"}],"tools":[{"type":"function","function":{"name":"read_file","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]})JSON", tool_payload, parse_error));
    ajson tools;
    assert(normalize_tools(tool_payload.get("tools"), false, &tools, &parse_error));
    const std::string tool_expected = turn("system", low + "\n\n" + render_qwen_tools_prompt(tools, nullptr) +
            "\n\nTop instructions.\n\nInitial instructions.") + turn("user", "Read both.") +
            turn("assistant", "<think>\nCheck both files.\n</think>\n\n"
                "<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>\n"
                "<tool_call>\n<function=read_file>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>") +
            turn("user", "<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response>") +
            "<|im_start|>assistant\n<think>\n";
    std::vector<uint32_t> reference, actual, suffix, multimodal;
    assert(append_text_ids(&state, tool_expected, &reference) == AXIOM_OK);
    std::string error; uint32_t original = 0; bool compacted = false;
    check(make_chat_ids(&state, tool_payload, false, &actual, &suffix, &error, &original, &compacted) &&
            actual == reference, "tools/system_order_history_reasoning_grouped_results");
    std::shared_ptr<vision_request> vision;
    check(encode_multimodal_chat_ids(&state, tool_payload, false, &multimodal, &vision,
            &error, &original, &compacted) && multimodal == reference && vision &&
            vision->embedding_slots.size() == reference.size(), "tools/multimodal_prompt_parity");
    state.kv_config_signature = "old-config";
    const auto key = make_session_key(&state, "qa-template-contract", "ultra-fast");
    auto old_key = key; old_key.config_signature = state.kv_config_signature;
    check(key.config_signature == "old-config;chat_template=qwen38-chatml-v1" &&
            axiom::qwen38::qwen38_persistent_session_store::namespace_text(key) !=
            axiom::qwen38::qwen38_persistent_session_store::namespace_text(old_key),
            "old_native_kv_cannot_enter_stateful_tail_fallback");
    axiom_tokenizer_close(state.tokenizer); state.tokenizer = nullptr;
    std::cout << "CHAT_TEMPLATE checks=" << checks << " failures=" << failures << '\n';
    return failures ? 1 : 0;
}
