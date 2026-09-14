// CPU-only contract tests against the actual API adapter. No model is loaded,
// no server is started and no production configuration is changed.
#define main axiom_api_main
#include "../tools/axiom_qwen38_api.cpp"
#undef main
#include <iostream>

int main() {
    const char *ids[] = {"ultra-fast", "minimal", "low", "medium", "high", "xhigh", "max"};
    // These are Axiom/Qwen caps, NOT claimed OpenAI token budgets.
    const uint32_t caps[] = {0, 1024, 2048, 4096, 8192, 16384, 32768};
    size_t checks = 0, failures = 0;
    const auto check = [&](bool pass, const std::string &name) {
        ++checks;
        if (!pass) ++failures;
        std::cout << name << " status=" << (pass ? "PASS" : "FAIL") << '\n';
    };
    check(axiom::reasoning::kProfileCount == 7, "seven_profiles");
    for (size_t i = 0; i < 7; ++i) {
        const auto *profile = axiom::reasoning::find(ids[i]);
        check(profile && std::string(profile->id) == ids[i] &&
                profile->thinking == (i != 0) && profile->thinking_budget_tokens == caps[i],
                std::string(ids[i]) + "/reference_contract");
        for (int protocol = 0; protocol < 3; ++protocol) {
            ajson payload = ajson::jobj(), reasoning = ajson::jobj();
            reasoning.set("effort", ajson::jstr(ids[i]));
            std::string error;
            if (protocol == 0) payload.set("reasoning_effort", ajson::jstr(ids[i]));
            else payload.set("reasoning", reasoning);
            if (protocol == 2) {
                payload.set("input", ajson::jstr("Contract fixture, no inference."));
                ajson normalized;
                check(normalize_responses_request(payload, &normalized, &error) && error.empty(),
                        std::string(ids[i]) + "/responses_normalization");
                payload = std::move(normalized);
            }
            const bool off = resolve_no_think(nullptr, payload, &error);
            thinking_plan plan;
            sampling_params sampling;
            check(error.empty() && off == (i == 0) &&
                    effective_reasoning_effort(nullptr, payload) == ids[i] &&
                    make_thinking_plan(payload, off, 64, true, 262144, &plan, &error) &&
                    plan.enabled == (i != 0) && plan.thinking_budget == caps[i] &&
                    plan.visible_budget == 64 &&
                    parse_sampling_params(payload, off, 1, &sampling, &error) &&
                    sampling.enabled == (i != 0) && error.empty(),
                    std::string(ids[i]) + "/protocol_" + std::to_string(protocol));
        }
    }
    // Unknown names must not silently become medium/max. 'none' is an external
    // provider ID; Axiom's advertised off ID remains ultra-fast, not a new level.
    for (const char *bad : {"none", "off", "ultra", "maximum", "unsupported"}) {
        ajson payload = ajson::jobj();
        payload.set("reasoning_effort", ajson::jstr(bad));
        std::string error;
        resolve_no_think(nullptr, payload, &error);
        check(!error.empty(), std::string("reject_unadvertised/") + bad);
    }
    // A stale legacy boolean must not overwrite the explicitly selected level.
    for (const char *id : {"ultra-fast", "max"}) {
        ajson payload = ajson::jobj();
        payload.set("reasoning_effort", ajson::jstr(id));
        payload.set("think", ajson::jbool(std::string(id) == "ultra-fast"));
        std::string error;
        check(resolve_no_think(nullptr, payload, &error) == (std::string(id) == "ultra-fast") &&
                error.empty(), std::string("explicit_profile_wins/") + id);
    }
    // Qwen's prompt may already contain the opening <think> token. Only the
    // visible suffix may supply executable calls or the final user-facing text.
    ajson tools;
    std::string json_error;
    const bool tools_ok = ajson_parse(R"JSON([{"name":"qa_echo","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}}])JSON", tools, json_error);
    check(tools_ok, "thinking_tool_fixture");
    const auto hidden_call = native_tool_contract_self_test_call("qa_echo", {{"value", "HIDDEN"}});
    const auto visible_call = native_tool_contract_self_test_call("qa_echo", {{"value", "VISIBLE"}});
    for (const std::string &opening : {std::string(), std::string("<think>")}) {
        const auto prefix = opening.empty() ? "implicit_think" : "explicit_think";
        const auto no_call = parse_native_tool_calls(opening + "Private planning.\n" + hidden_call + "\n</think>\nFinal answer.", tools, true);
        check(no_call.status == "none" && no_call.calls.empty() &&
                no_call.content == "Final answer.", std::string(prefix) + "/hidden_call_not_executed");
        const auto actual = parse_native_tool_calls(opening + hidden_call + "\n</think>\n" + visible_call, tools, true);
        check(actual.status == "pass" && actual.calls.size() == 1 &&
                actual.calls[0].get("function")->get("arguments")->s == "{\"value\":\"VISIBLE\"}" &&
                actual.content.empty(), std::string(prefix) + "/only_visible_call");
        const auto plain = parse_native_tool_calls(opening + "Private planning.\n</think>\nFinal answer.", tools, true);
        check(plain.status == "none" && plain.content == "Final answer.",
                std::string(prefix) + "/final_content_not_planning");
    }
    const auto invalid_visible = parse_native_tool_calls("Planning.\n</think>\n<tool_call>broken", tools, true);
    check(invalid_visible.status == "fail", "visible_invalid_tool_still_rejected");
    const auto off_literal = parse_native_tool_calls(native_tool_contract_self_test_call("qa_echo", {{"value", "</think>"}}), tools);
    check(off_literal.status == "pass" && off_literal.calls.size() == 1 &&
            off_literal.calls[0].get("function")->get("arguments")->s == "{\"value\":\"</think>\"}",
            "off_tool_literal_closing_tag_preserved");
    const auto unfinished = parse_native_tool_calls(hidden_call, tools, true);
    check(unfinished.status == "none" && unfinished.calls.empty() && unfinished.content.empty(),
            "unfinished_reasoning_never_authorizes_tools");
    // Exercise the actual append hook and the scheduling allowance used by
    // both decode paths. No fake model/GPU inference is counted by these tests.
    server_state append_state;
    for (const uint32_t cap : {1u, 2u, 7u, 8u, 9u, 64u, 512u}) {
        for (const uint32_t batch : {1u, 8u, 264u}) {
            bool safe = true;
            for (const uint32_t prefix : {0u, 1u, 7u, 8u, 63u, 64u}) {
                for (const auto &marker : {std::vector<uint32_t>{248069u},
                                         std::vector<uint32_t>{91u, 92u}}) {
                    generation_stream_sink guarded;
                    guarded.output_budget.reasoning_end_marker = marker;
                    guarded.output_budget.visible_limit = cap;
                    generation_result generated;
                    std::string stage;
                    while (const uint32_t remaining = generation_tokens_remaining(
                            generated, 1024u, &guarded)) {
                        const uint32_t count = std::min(batch, remaining);
                        for (uint32_t i = 0u; i < count; ++i) {
                            const size_t position = generated.ids.size();
                            const uint32_t token = position >= prefix &&
                                    position < prefix + marker.size()
                                    ? marker[position - prefix] : 42u;
                            safe = safe && generation_tokens_remaining(
                                    generated, 1024u, &guarded) != 0u;
                            safe = safe && append_generated_token(&append_state,
                                    &generated, token, &guarded, &stage) == AXIOM_OK;
                        }
                    }
                    safe = safe && guarded.output_budget.reasoning_ended &&
                            guarded.output_budget.visible_count == cap &&
                            generated.ids.size() == prefix + marker.size() + cap;
                }
            }
            check(safe, "visible_limit/cap=" + std::to_string(cap) +
                    "/batch=" + std::to_string(batch));
        }
    }
    generation_result boundary;
    boundary.ids = {7u, 8u, 9u};
    generation_stream_sink unrestricted;
    check(generation_tokens_remaining(boundary, 64u, nullptr) == 61u &&
            generation_tokens_remaining(boundary, 64u, &unrestricted) == 61u,
            "off_visible_budget_unchanged");
    unrestricted.output_budget.reasoning_end_marker = {248069u};
    unrestricted.output_budget.visible_limit = UINT32_MAX;
    check(generation_tokens_remaining(boundary, 64u, &unrestricted) == 61u &&
            generation_tokens_remaining(boundary, 3u, &unrestricted) == 0u &&
            generation_tokens_remaining(boundary, 2u, &unrestricted) == 0u,
            "visible_budget_saturating_arithmetic");
    generation_stream_sink repeated;
    repeated.output_budget.reasoning_end_marker = {248069u};
    repeated.output_budget.visible_limit = 3u;
    generation_result repeated_result;
    std::string repeated_error;
    bool repeated_ok = true;
    for (uint32_t token : {42u, 248069u, 248069u, 42u, 248069u}) {
        repeated_ok = repeated_ok && append_generated_token(&append_state,
                &repeated_result, token, &repeated, &repeated_error) == AXIOM_OK;
    }
    check(repeated_ok && repeated.output_budget.visible_count == 3u &&
            generation_tokens_remaining(repeated_result, 1024u, &repeated) == 0u,
            "visible_literal_marker_does_not_reset_budget");
    std::cout << "checks=" << checks << " failures=" << failures << '\n';
    return failures ? 1 : 0;
}
