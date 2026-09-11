// Exercise the real daemon normalizer, native validator and SSE writer without
// loading a model or opening a listener. Reuse the TU's private test surface.
#define main axiom_daemon_main
#include "../tools/axiom_qwen38_api.cpp"
#undef main
#include <cassert>
#include <fstream>
#include <iostream>
#include <future>


static ajson json(const std::string &s) {
    ajson v; std::string e; assert(ajson_parse(s, v, e)); return v;
}

static void custom_format_tests() {
    const auto original = json(R"({"model":"qwen3.8-27b-nvfp4","input":"Edit a file","tools":[{"type":"custom","name":"apply_patch","description":"FREEFORM patch","format":{"type":"grammar","syntax":"lark","definition":"start: \"*** Begin Patch\" LF hunk+ \"*** End Patch\""}}]})");
    auto request = original; axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge); std::string error;
    assert(bridge.prepare(request, &error));
    ajson normalized, native;
    assert(normalize_codex_responses_request(request, &normalized, &error));
    assert(normalize_tools(normalized.get("tools"), false, &native, &error));
    const auto prompt = render_qwen_tools_prompt(native, nullptr);
    assert(prompt.find("*** Begin Patch") != std::string::npos);
    assert(prompt.find("raw custom-tool text") != std::string::npos);
    assert(ajson_dumps(*request.get("tools")->arr[0].get("format")) == ajson_dumps(*original.get("tools")->arr[0].get("format")));
    auto definition = request.get("tools")->arr[0]; const auto once = ajson_dumps(definition);
    axiom_codex::describe_custom_format(definition, "$.tools[0]"); assert(ajson_dumps(definition) == once);
    // Unsupported metadata rejects exactly this tool, keeping a usable sibling.
    request = original;
    auto rejected_tools = *request.get("tools"), rejected_format = *rejected_tools.arr[0].get("format");
    rejected_format.set("syntax", ajson::jstr("unknown")); rejected_tools.arr[0].set("format", rejected_format);
    rejected_tools.push(json(R"({"type":"function","name":"ok_tool","parameters":{"type":"object","properties":{},"additionalProperties":false}})"));
    request.set("tools", std::move(rejected_tools));
    axiom_codex::tool_wire_map rejected; configure_codex_schema_bridge(rejected);
    assert(rejected.prepare(request, &error)); assert(request.get("tools")->arr.size() == 1);
    assert(rejected.rejections().arr.size() == 1);
    assert(axiom_codex::string_field(rejected.rejections().arr[0], "json_path") == "$.tools[0][\"format\"][\"syntax\"]");
    // /v1 still sees precisely its old custom normalization (no bridge call).
    ajson legacy; assert(normalize_responses_request(original, &legacy, &error));
    assert(ajson_dumps(legacy).find("Core custom input contract") == std::string::npos);
    uint32_t budget = 0; bool explicit_budget = false;
    assert(parse_max_tokens(legacy, 262144, &budget, &error, true, &explicit_budget) && budget == 128 && !explicit_budget);
    assert(parse_max_tokens(normalized, 262144, &budget, &error, true, &explicit_budget) && budget == 4096 && explicit_budget);
    for (const auto *key : {"max_output_tokens", "max_tokens", "max_completion_tokens"}) {
        auto limited = original; limited.set(key, ajson::jint(32));
        axiom_codex::apply_request_defaults(limited);
        assert(limited.get(key)->i == 32);
        ajson mapped; assert(normalize_codex_responses_request(limited, &mapped, &error));
        assert(parse_max_tokens(mapped, 262144, &budget, &error, true) && budget == 32);
    }
    for (auto value : {ajson::jint(0), ajson::jint(-1), ajson::jstr("32"), ajson::jnull()}) {
        auto invalid = original; invalid.set("max_output_tokens", value); axiom_codex::apply_request_defaults(invalid);
        ajson mapped; assert(normalize_codex_responses_request(invalid, &mapped, &error));
        assert(!parse_max_tokens(mapped, 262144, &budget, &error, true));
    }
    std::puts("Codex custom grammar: PASS (visible exact contract, idempotent, structured rejection, legacy unchanged)");
}

static std::string available_bytes(int fd) {
    std::string bytes; char buffer[8192]; ssize_t n;
    while ((n = recv(fd, buffer, sizeof(buffer), MSG_DONTWAIT)) > 0) bytes.append(buffer, n);
    return bytes;
}

static void responses_terminal_tests() {
    axiom_codex::tool_wire_map bridge;
    for (int scenario = 0; scenario < 4; ++scenario) {
        int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        live_responses_stream stream;
        assert(start_live_responses_stream(&stream, fd[0], 73, "terminal-session", &bridge));
        std::string bytes = available_bytes(fd[1]);
        assert(bytes.find("response.in_progress") != std::string::npos);
        if (scenario != 0) {
            assert(live_responses_emit_token(&stream, "O"));
            bytes += available_bytes(fd[1]);
            // Observe the first delta BEFORE adding the second token or
            // calling completion: catches buffered rather than live SSE.
            assert(bytes.find("\"delta\":\"O\"") != std::string::npos);
            assert(live_responses_emit_token(&stream, "K"));
            bytes += available_bytes(fd[1]);
            assert(bytes.find("\"delta\":\"K\"") != std::string::npos);
        }
        if (scenario == 3) {
            generation_result result; result.text = "OK";
            ajson response = live_responses_stub(&stream);
            response.set("status", ajson::jstr("completed"));
            response.set("output_text", ajson::jstr("OK"));
            assert(finish_live_responses_stream(&stream, response, result, nullptr, nullptr));
        } else if (scenario != 2) {
            assert(live_responses_send_error(&stream, "test prefill/decode failure", "request_timeout"));
            assert(live_responses_send_error(&stream, "duplicate must not emit"));
        }
        close_live_responses_stream(&stream);
        close_live_responses_stream(&stream);
        bytes += available_bytes(fd[1]);
        const std::string terminal = scenario == 3 ? "event: response.completed" : "event: response.failed";
        const size_t pos = bytes.find(terminal);
        assert(pos != std::string::npos && bytes.find(terminal, pos + 1) == std::string::npos);
        assert(bytes.find(scenario == 3 ? "event: response.failed" : "event: response.completed") == std::string::npos);
        const size_t done = bytes.find("data: [DONE]");
        assert(done > pos && bytes.find("data: [DONE]", done + 1) == std::string::npos);
        assert(bytes.find("terminal-session") != std::string::npos);
        assert(bytes.find("resp-axiom-73") != std::string::npos);
        if (scenario != 0) assert(bytes.find("\"output_text\":\"OK\"") != std::string::npos);
        if (scenario < 2) assert(bytes.find("\"code\":\"request_timeout\"") != std::string::npos);
        close(fd[0]); close(fd[1]);
    }
    // Legacy /v1 error event is intentionally unchanged.
    int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    live_responses_stream legacy;
    assert(start_live_responses_stream(&legacy, fd[0], 74, "legacy"));
    assert(live_responses_send_error(&legacy, "legacy failure"));
    close_live_responses_stream(&legacy);
    const auto bytes = available_bytes(fd[1]);
    assert(bytes.find("event: error") != std::string::npos);
    assert(bytes.find("response.failed") == std::string::npos);
    close(fd[0]); close(fd[1]);
    std::puts("Responses lifecycle: PASS (incremental deltas, accumulated text, exactly one terminal, implicit failure, legacy unchanged)");
}

static void stream_edge_tests() {
    axiom_codex::tool_wire_map bridge;
    for (bool mismatch : {false, true}) {
        int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        live_responses_stream stream;
        assert(start_live_responses_stream(&stream, fd[0], 91, "edge-session", &bridge));
        auto bytes = available_bytes(fd[1]);
        assert(live_responses_emit_token(&stream, "OK "));
        bytes += available_bytes(fd[1]);
        assert(live_responses_emit_token(&stream, "<tool_"));
        assert(available_bytes(fd[1]).empty()); // split XML marker is held back
        assert(live_responses_emit_token(&stream, "call>hidden</tool_call>"));
        assert(available_bytes(fd[1]).empty());
        generation_result result; result.text = mismatch ? "WRONG" : "OK";
        native_tool_result parsed; parsed.status = "pass"; parsed.content = result.text;
        const auto response = responses_response(result, 91, "ultra-fast", nullptr, &parsed);
        assert(finish_live_responses_stream(&stream, response, result, &parsed, nullptr) == !mismatch);
        close_live_responses_stream(&stream); bytes += available_bytes(fd[1]);
        assert(bytes.find("hidden") == std::string::npos);
        assert(bytes.find(mismatch ? "event: response.failed" : "event: response.completed") != std::string::npos);
        assert(bytes.find("\"output_text\":\"OK \"") != std::string::npos);
        if (mismatch) assert(bytes.find("stream_content_mismatch") != std::string::npos);
        close(fd[0]); close(fd[1]);
    }
    int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    http_cancel_probe probe; probe.fd = fd[0];
    assert(!http_client_cancelled(&probe));
    probe.has_deadline = true; probe.deadline = axiom::qwen38::swarm_clock::now();
    assert(http_client_cancelled(&probe));
    probe.has_deadline = false; close(fd[1]); assert(http_client_cancelled(&probe)); close(fd[0]);
    std::puts("SSE edges/cancellation: PASS (split marker withheld, whitespace retained, mismatch fails, elapsed deadline, disconnected client)");
}

static void schema_tests() {
    const auto source = json(R"({"type":"object","properties":{"surface":{"description":"Optional surface","anyOf":[{"type":"string","enum":["excel","powerpoint","sheets"]},{"type":"null"}]}},"additionalProperties":false})");
    ajson request = json(R"({"tools":[{"type":"function","name":"_list_document_sessions","namespace":"mcp__codex_apps__codex_document_control"}]})");
    request.vals[0].arr[0].set("parameters", source);
    axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge);
    std::string e; assert(bridge.prepare(request, &e));
    assert(bridge.rejections().arr.empty());
    const auto &tool = request.get("tools")->arr[0];
    const auto wire = tool.get("name")->s;
    assert(axiom_codex::canonical(wire));
    ajson identity = json(R"({"type":"function_call","call_id":"surface-proof","arguments":"{}"})");
    identity.set("name", ajson::jstr(wire)); bridge.restore(identity);
    assert(axiom_codex::string_field(identity, "name") == "_list_document_sessions");
    assert(axiom_codex::string_field(identity, "namespace") == "mcp__codex_apps__codex_document_control");
    assert(axiom_codex::string_field(identity, "call_id") == "surface-proof");
    assert(bridge.original_schema(wire) && native_tool_json_equal(source, *bridge.original_schema(wire)));
    const auto &lowered = *tool.get("parameters");
    assert(!lowered.get("properties")->get("surface")->get("anyOf"));
    assert(lowered.get("properties")->get("surface")->get("enum")->arr.size() == 4);
    for (const char *s : {"{}", "{\"surface\":null}", "{\"surface\":\"excel\"}", "{\"surface\":\"powerpoint\"}", "{\"surface\":\"sheets\"}"}) {
        const auto value = json(s); native_tool_argument_error failure;
        assert(bridge.validate_arguments(wire, value, &e));
        assert(validate_native_tool_argument(value, lowered, "$", &failure));
    }
    for (const char *s : {"{\"surface\":\"word\"}", "{\"surface\":1}", "{\"surface\":true}", "{\"surface\":[]}", "{\"surface\":{}}", "{\"other\":null}"}) {
        const auto value = json(s); native_tool_argument_error failure;
        assert(!bridge.validate_arguments(wire, value, &e));
        assert(!validate_native_tool_argument(value, lowered, "$", &failure));
    }
    assert(!validate_native_tool_schema_definition(source, "legacy", "$", &e));
    assert(e.find("anyOf") != std::string::npos);
    // Parent constraints intersect the union; nested array validation must use
    // the original union too, not just native validation (which ignores anyOf).
    auto nested = json(R"({"type":"array","items":{"enum":["excel",null],"anyOf":[{"enum":["excel","word"]},{"type":"null"}]}})");
    auto lowered_nested = axiom_codex::lower_schema(nested, "$",
        [](const ajson &s, const std::string &p, std::string *e) { return validate_native_tool_schema_definition(s, "test", p, e); },
        [](const ajson &v, const ajson &s, std::string *e) { return validate_codex_original_argument(v, s, e); });
    assert(lowered_nested.get("items")->get("enum")->arr.size() == 2);
    assert(validate_codex_original_argument(json(R"(["excel",null])"), nested, &e));
    assert(!validate_codex_original_argument(json(R"(["word"])"), nested, &e));
    for (const char *bad : {R"({"anyOf":[]})", R"({"type":"array","items":{"anyOf":[{"type":"string"},{"type":"null"}]}})", R"({"$ref":"#/x"})", R"({"oneOf":[{"const":1},{"const":2}]})", R"({"type":"string","pattern":"a"})", R"({"anyOf":[{"const":"a"}],"required":false})", R"({"minItems":-1})", R"({"minItems":1.5})", R"({"minItems":"1"})"}) {
        ajson mixed = json(R"({"tools":[{"type":"function","name":"good","parameters":{"type":"object","additionalProperties":false}},{"type":"function","name":"bad","parameters":{"type":"object","properties":{}}}]})");
        ajson p = *mixed.vals[0].arr[1].get("parameters"), props = ajson::jobj();
        props.set("value", json(bad)); p.set("properties", props); mixed.vals[0].arr[1].set("parameters", p);
        axiom_codex::tool_wire_map b; configure_codex_schema_bridge(b); assert(b.prepare(mixed, &e));
        assert(mixed.get("tools")->arr.size() == 1 && b.rejections().arr.size() == 1);
        for (const char *k : {"code", "tool_name", "json_path", "unsupported_keyword", "reason"}) assert(b.rejections().arr[0].get(k));
        ajson response = json(R"({"type":"response","output":[]})"); b.restore(response);
        assert(response.get("codex_tool_errors"));
    }
    auto selected = json(R"({"tools":[{"type":"function","name":"bad","parameters":{"type":"object","properties":{"v":{"$ref":"#/missing"}}}}],"tool_choice":{"type":"function","name":"bad"}})");
    axiom_codex::tool_wire_map selected_bridge; configure_codex_schema_bridge(selected_bridge);
    assert(!selected_bridge.prepare(selected, &e) && selected_bridge.rejections().arr.size() == 1);
    auto duplicate = json(R"({"tools":[{"type":"function","name":"same","parameters":{"type":"object","additionalProperties":false}},{"type":"function","name":"same","parameters":{"type":"object","properties":{"x":{"type":"string"}}}},{"type":"function","name":"good","parameters":{"type":"object","additionalProperties":false}}]})");
    axiom_codex::tool_wire_map duplicates; configure_codex_schema_bridge(duplicates);
    assert(duplicates.prepare(duplicate, &e)); assert(duplicate.get("tools")->arr.size() == 1);
    auto web = json(R"({"session_id":"web-session","parallel_tool_calls":true,"instructions":"keep instructions","input":[{"type":"function_call_output","call_id":"c1","item_id":"i1","output":"opaque result"}],"tools":[{"type":"function","name":"web_search","id":"t1","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},{"type":"function","name":"web_fetch","id":"t2","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}]})");
    const auto before_web = web;
    axiom_codex::tool_wire_map web_bridge; configure_codex_schema_bridge(web_bridge);
    assert(web_bridge.prepare(web, &e)); assert(native_tool_json_equal(web, before_web));
    ajson web_response = json(R"({"output":[{"type":"function_call","name":"web_search","id":"i1","call_id":"c1","arguments":"{\"query\":\"test\"}"},{"type":"function_call","name":"web_fetch","id":"i2","call_id":"c2","arguments":"{\"url\":\"https://example.com\"}"}]})");
    const auto before_response = web_response; web_bridge.restore(web_response);
    assert(native_tool_json_equal(*before_response.get("output"), *web_response.get("output")));
    assert(web_response.get("session_id")->s == "web-session" && web_response.get("parallel_tool_calls")->b);
    std::puts("Codex schema: PASS (exact nullable enum, original argument validation, quarantine, strict legacy)");
}

static void minimum_items_tests() {
    auto request = json(R"({"tools":[{"type":"function","name":"minimum_probe","parameters":{"type":"object","properties":{
      "items":{"type":"array","items":{"type":"string"},"minItems":1},
      "value":{"anyOf":[{"type":"boolean"},{"type":"array","items":{"type":"string"},"minItems":1}]}
    },"required":["items","value"],"additionalProperties":false}}]})");
    const auto original = request;
    axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge); std::string error;
    assert(bridge.prepare(request, &error) && bridge.rejections().arr.empty());
    assert(request.get("tools")->arr.size() == 2);
    for (const auto &tool : request.get("tools")->arr) {
        assert(tool.get("parameters")->get("properties")->get("items")->get("minItems")->i == 1);
        const auto name = tool.get("name")->s;
        assert(bridge.validate_arguments(name, json(R"({"items":["a"],"value":true})"), &error));
        assert(bridge.validate_arguments(name, json(R"({"items":["a"],"value":["b"]})"), &error));
        assert(!bridge.validate_arguments(name, json(R"({"items":[],"value":true})"), &error));
        assert(error.find("$[\"items\"]") != std::string::npos && error.find("minItems") != std::string::npos);
        assert(!bridge.validate_arguments(name, json(R"({"items":["a"],"value":[]})"), &error));
    }
    // Show the boundary guard is necessary; legacy native behavior is intact.
    const auto *schema = original.get("tools")->arr[0].get("parameters")->get("properties")->get("items");
    native_tool_argument_error native;
    assert(validate_native_tool_schema_definition(*schema, "legacy", "$", &error));
    assert(validate_native_tool_argument(json("[]"), *schema, "$", &native));
    assert(!validate_codex_original_argument(json("[]"), *schema, &error));
    std::puts("minItems: PASS (wire assertion retained, original guard enforced recursively, malformed rejected, legacy unchanged)");
}

static void codex_scalar_bounds_tests() {
    std::string error;
    const axiom_codex::schema_check check = [](const ajson &s, const std::string &p, std::string *e) {
        return validate_native_tool_schema_definition(s, "bounds", p, e);
    };
    const axiom_codex::argument_check accept = [](const ajson &v, const ajson &s, std::string *e) {
        return validate_codex_original_argument(v, s, e);
    };
    // Assert keyword-specific diagnostics, valid syntax retained byte-for-byte,
    // and fail-closed behavior if an original-schema guard is not configured.
    for (const char *key : {"minLength", "maxLength", "minimum", "maximum"}) {
        const bool length = std::string(key) == "minLength" || std::string(key) == "maxLength";
        for (const auto &valid : {ajson::jint(0), ajson::jnum(1.0), ajson::jint(9223372036854775807LL)}) {
            ajson schema = ajson::jobj(); schema.set(key, valid);
            assert(ajson_dumps(axiom_codex::lower_schema(schema, "$", check, accept)) == ajson_dumps(schema));
            bool rejected = false;
            try { (void)axiom_codex::lower_schema(schema, "$", check, {}); }
            catch (const axiom_codex::schema_failure &e) { rejected = e.keyword == key; }
            assert(rejected);
        }
        std::vector<ajson> invalid{ajson::jstr("1"), ajson::jbool(true), ajson::jnull(), ajson::jarr(),
            ajson::jnum(std::numeric_limits<double>::infinity()), ajson::jnum(std::numeric_limits<double>::quiet_NaN())};
        if (length) { invalid.push_back(ajson::jint(-1)); invalid.push_back(ajson::jnum(0.5)); }
        for (const auto &bad : invalid) {
            ajson schema = ajson::jobj(); schema.set(key, bad);
            bool rejected = false;
            try { (void)axiom_codex::lower_schema(schema, "$", check, accept); }
            catch (const axiom_codex::schema_failure &e) { rejected = e.keyword == key && e.path == "$"; }
            assert(rejected);
        }
    }
    const auto one = json(R"({"type":"string","minLength":1,"maxLength":1})");
    for (const auto &text : {std::string("a"), std::string(u8"é"), std::string(u8"€"), std::string(u8"😀"), std::string(1, '\0')})
        assert(accept(ajson::jstr(text), one, &error));
    assert(accept(json(R"("\ud83d\ude00")"), one, &error));
    for (const auto &text : {std::string(), std::string("aa"), std::string(u8"e\u0301"), std::string(u8"😀a"),
        std::string("\x80"), std::string("\xc0\xaf"), std::string("\xe2\x82"), std::string("\xed\xa0\x80"), std::string("\xf4\x90\x80\x80")})
        assert(!accept(ajson::jstr(text), one, &error));
    assert(!accept(json(R"("\ud800")"), one, &error));
    const auto web_string = json(R"({"type":"string","minLength":1,"maxLength":512})");
    std::string unicode_boundary;
    for (int i = 0; i < 512; ++i) unicode_boundary += u8"😀";
    assert(accept(ajson::jstr(unicode_boundary), web_string, &error));
    assert(!accept(ajson::jstr(unicode_boundary + "a"), web_string, &error));
    assert(accept(ajson::jstr(""), json(R"({"maxLength":0})"), &error));
    const auto integer = json(R"({"type":"integer","minimum":1,"maximum":8})");
    for (const char *value : {"1", "8", "1.0", "8.0"}) assert(accept(json(value), integer, &error));
    for (const char *value : {"0", "9", "1.5", "true", "\"1\""}) assert(!accept(json(value), integer, &error));
    const auto decimal = json(R"({"type":"number","minimum":-1.5,"maximum":2.25})");
    assert(native_tool_json_equal(axiom_codex::lower_schema(decimal, "$", check, accept), decimal));
    for (double value : {-1.5, 0.0, 2.25}) assert(accept(ajson::jnum(value), decimal, &error));
    for (double value : {std::nextafter(-1.5, -2.0), std::nextafter(2.25, 3.0),
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        assert(!accept(ajson::jnum(value), decimal, &error));
    // Preserve int64 distinctions that a comparison through double would lose.
    for (const auto &pair : {std::make_pair("9007199254740993", "9007199254740992"),
                            std::make_pair("9223372036854775807", "9223372036854775806"),
                            std::make_pair("-9223372036854775808", "-9223372036854775807")}) {
        ajson exact = json(R"({"type":"integer"})");
        exact.set("minimum", json(pair.first)); exact.set("maximum", json(pair.first));
        assert(accept(json(pair.first), exact, &error)); assert(!accept(json(pair.second), exact, &error));
    }
    // Assertions only apply to their JSON type; contradictory bounds are valid
    // syntax with an empty applicable domain, not a reason to drop assertions.
    assert(accept(ajson::jnull(), json(R"({"minLength":2,"minimum":3})"), &error));
    const auto impossible = json(R"({"type":"string","minLength":2,"maxLength":1})");
    assert(native_tool_json_equal(axiom_codex::lower_schema(impossible, "$", check, accept), impossible));
    assert(!accept(ajson::jstr("a"), impossible, &error));
    const auto nested = json(R"({"type":"object","properties":{"items":{"type":"array","items":{"type":"string","maxLength":1}}},"additionalProperties":{"type":"number","maximum":8}})");
    assert(accept(json(R"({"items":["\u00e9"],"other":8})"), nested, &error));
    assert(!accept(json(R"({"items":["ab"]})"), nested, &error));
    assert(error.find("$[\"items\"][0]") != std::string::npos && error.find("maxLength") != std::string::npos);
    assert(!accept(json(R"({"other":9})"), nested, &error));
    assert(error.find("$[\"other\"]") != std::string::npos && error.find("maximum") != std::string::npos);

    auto request = json(R"({"tools":[{"type":"function","name":"web_search","parameters":{"type":"object","properties":{"query":{"type":"string","minLength":1,"maxLength":512},"limit":{"type":"integer","minimum":1,"maximum":8}},"required":["query"],"additionalProperties":false}}]})");
    request.set("input", ajson::jstr("Search IANA"));
    const auto original = request;
    axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge);
    assert(bridge.prepare(request, &error) && bridge.rejections().arr.empty());
    assert(ajson_dumps(request) == ajson_dumps(original)); // No keyword stripping/coercion.
    ajson normalized, tools;
    assert(normalize_codex_responses_request(request, &normalized, &error));
    assert(normalize_tools(normalized.get("tools"), false, &tools, &error));
    const auto parsed = parse_native_tool_calls(native_tool_contract_self_test_call("web_search", {{"query", "ok"}, {"limit", "9"}}), tools);
    assert(parsed.status == "pass" && parsed.calls.size() == 1); // Native legacy still ignores numeric bounds.
    const auto &function = *parsed.calls[0].get("function");
    const auto arguments = json(function.get("arguments")->s);
    assert(!bridge.validate_arguments(function.get("name")->s, arguments, &error));
    assert(error.find("maximum") != std::string::npos);
    assert(!bridge.validate_arguments("web_search", json(R"({"query":""})"), &error));
    assert(bridge.validate_arguments("web_search", json(R"({"query":"ok","limit":8})"), &error));
    // Production checks this guard before emitting executable calls. Exercise
    // its terminal error path: the invalid call never becomes client tool work.
    int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    live_responses_stream stream;
    assert(start_live_responses_stream(&stream, fd[0], 95, "bounds", &bridge));
    assert(live_responses_send_error(&stream, "original Codex schema validation failed: maximum"));
    close_live_responses_stream(&stream);
    const auto bytes = available_bytes(fd[1]); close(fd[0]); close(fd[1]);
    assert(bytes.find("response.failed") != std::string::npos && bytes.find("function_call") == std::string::npos);

    const auto union_schema = json(R"({"anyOf":[{"type":"string","minLength":1,"maxLength":1},{"type":"integer","minimum":1,"maximum":8}]})");
    const auto alternatives = axiom_codex::lower_schema_alternatives(union_schema, "$", check, accept);
    assert(alternatives.size() == 2);
    for (size_t i = 0; i < 2; ++i) assert(ajson_dumps(alternatives[i]) == ajson_dumps(union_schema.get("anyOf")->arr[i]));
    for (const char *value : {"\"é\"", "1", "8"}) assert(accept(json(value), union_schema, &error));
    for (const char *value : {"\"\"", "\"ab\"", "0", "9"}) assert(!accept(json(value), union_schema, &error));
    const auto finite = json(R"({"anyOf":[{"enum":["","é","ab"],"minLength":1,"maxLength":1},{"enum":[0,1,8,9],"minimum":1,"maximum":8}],"maximum":5})");
    const auto finite_lowered = axiom_codex::lower_schema(finite, "$", check, accept);
    assert(finite_lowered.get("maximum")->i == 5);
    assert(native_tool_json_equal(*finite_lowered.get("enum"), json(R"(["é",1])")));
    native_tool_argument_error native;
    assert(check(one, "$", &error) && validate_native_tool_argument(ajson::jstr("ab"), one, "$", &native));
    assert(check(integer, "$", &error) && validate_native_tool_argument(ajson::jint(9), integer, "$", &native));
    assert(!check(union_schema, "$", &error)); // strict native anyOf rejection unchanged.
    std::puts("Codex scalar bounds: PASS (syntax, wire retention, Unicode, exact int64, finite numbers, recursive anyOf/items/additionalProperties, pre-execution guard, legacy unchanged)");
}

static void exact_alternative_tests() {
    const auto annotated = json(R"({"description":"shared","properties":{"value":{"anyOf":[{"type":"string","description":"branch string"},{"type":"null","description":"branch null"}]},"literal":{"enum":[{"description":"VALUE NOT METADATA"}]}}})");
    ajson notes = ajson::jobj();
    const auto compacted_schema = axiom_codex::hoist_schema_documentation(annotated, &notes);
    assert(!compacted_schema.get("description") && notes.keys.size() == 3);
    assert(compacted_schema.get("properties")->get("literal")->get("enum")->arr[0].get("description")->s == "VALUE NOT METADATA");
    assert(ajson_dumps(notes).find("branch null") != std::string::npos);
    const auto original = json(R"({"session_id":"variant-session","turn_id":"turn-19","parallel_tool_calls":false,
      "tools":[{"type":"function","namespace":"plugin","name":"*nullable","id":"tool-19",
        "parameters":{"type":"object","properties":{"text":{"anyOf":[{"type":"string"},{"type":"null"}]},
        "list":{"anyOf":[{"type":"array","items":{"type":"string"}},{"type":"null"}]}},"required":["text"],"additionalProperties":false}}],
      "input":[{"type":"function_call","namespace":"plugin","name":"*nullable","id":"item-19","call_id":"call-19","arguments":"{\"text\":null,\"list\":[\"a\"]}"},
        {"type":"function_call_output","id":"result-19","call_id":"call-19","output":{"value":19}}]})");
    ajson request = original; std::string error;
    axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge);
    assert(bridge.prepare(request, &error) && bridge.rejections().arr.empty());
    const auto &tools = request.get("tools")->arr;
    assert(tools.size() == 4);
    assert(native_tool_json_equal(bridge.original_request(), original));
    const auto &source = *original.get("tools")->arr[0].get("parameters");
    // Cross-product truth table: union of advertised domains == original
    // domain. Includes absent fields, mixed arrays and additional properties.
    std::vector<ajson> values{json("null"), json("\"s\""), json("[]"), json("[\"s\"]"), json("[null]"), json("19"), json("{}")};
    size_t compared = 0;
    for (size_t a = 0; a <= values.size(); ++a) for (size_t b = 0; b <= values.size(); ++b) {
        ajson value = ajson::jobj();
        if (a < values.size()) value.set("text", values[a]);
        if (b < values.size()) value.set("list", values[b]);
        bool admitted = false;
        for (const auto &tool : tools) {
            native_tool_argument_error failure;
            admitted |= validate_native_tool_argument(value, *tool.get("parameters"), "$", &failure);
            assert(tool.get("id")->s == "tool-19");
            assert(bridge.validate_arguments(tool.get("name")->s, value, &error) == validate_codex_original_argument(value, source, &error));
        }
        assert(admitted == validate_codex_original_argument(value, source, &error)); ++compared;
    }
    auto call = request.get("input")->arr[0]; bridge.restore(call);
    assert(native_tool_json_equal(call, original.get("input")->arr[0]));
    assert(native_tool_json_equal(request.get("input")->arr[1], original.get("input")->arr[1]));
    ajson normalized; assert(normalize_codex_responses_request(request, &normalized, &error));
    assert(normalized.get("session_id")->s == "variant-session");
    assert(normalized.get("turn_id")->s == "turn-19");
    assert(normalized.get("messages")->arr[1].get("tool_call_id")->s == "call-19");
    assert(normalized.get("messages")->arr[1].get("content")->s == "{\"value\":19}");
    assert(!bridge.validate_calls({tools[0].get("name")->s, tools[1].get("name")->s}, &error));
    ajson response = json(R"({"object":"response","output":[],"parallel_tool_calls":true})");
    bridge.restore(response);
    assert(!response.get("parallel_tool_calls")->b && response.get("turn_id")->s == "turn-19");
    // Forcing a nullable tool advertises ALL four exact alternatives, never
    // a random branch. allowed_tools are references, not new definitions.
    for (const auto &choice : {json(R"({"type":"function","namespace":"plugin","name":"*nullable"})"),
            json(R"({"type":"allowed_tools","mode":"required","tools":[{"type":"function","namespace":"plugin","name":"*nullable"}]})")}) {
        ajson forced = original; forced.set("tool_choice", choice);
        assert(bridge.prepare(forced, &error)); assert(bridge.rejections().arr.empty());
        assert(forced.get("tools")->arr.size() == 4 && forced.get("tool_choice")->s == "required");
        assert(!bridge.validate_calls({}, &error));
        assert(bridge.validate_calls({forced.get("tools")->arr[2].get("name")->s}, &error));
    }
    // Reject the entire tool on expansion overflow; no partial domain.
    ajson large = original, props = ajson::jobj();
    for (int i = 0; i < 7; ++i) props.set("p" + std::to_string(i), json(R"({"anyOf":[{"type":"string"},{"type":"null"}]})"));
    ajson schema = json(R"({"type":"object","additionalProperties":false})"); schema.set("properties", props);
    large.vals[3].arr[0].set("parameters", schema); large.set("input", ajson::jstr("OK"));
    assert(bridge.prepare(large, &error));
    assert(large.get("tools")->arr.empty() && bridge.rejections().arr.size() == 1);
    assert(bridge.rejections().arr[0].get("reason")->s.find("64") != std::string::npos);
    // Reusing a bridge cannot carry a previous quarantine into another request.
    request = original; assert(bridge.prepare(request, &error)); assert(bridge.rejections().arr.empty());
    std::cout << "Exact schema alternatives: PASS (" << compared << " domains, IDs, forced/allowed choice, parallel=false, replay, expansion quarantine)\n";
}

// Core search protocol, without creating a second discovery executor.
static void client_search_tests() {
    // A dynamically changed Core catalog must reach the model even with a
    // stable session. Legacy tail eligibility is preserved in both cases.
    assert(!axiom_codex::allow_stateful_tail(true, true));
    assert(!axiom_codex::allow_stateful_tail(true, false));
    assert(axiom_codex::allow_stateful_tail(false, true));
    assert(!axiom_codex::allow_stateful_tail(false, false));
    auto source = json(R"({"input":"Find a tool","session_id":"discovery-session","tools":[{"type":"tool_search","execution":"client","description":"Core discovery","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"number"}},"required":["query"],"additionalProperties":false}}]})");
    auto request = source;
    axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge);
    std::string error; ajson payload, tools;
    assert(bridge.prepare(request, &error) && bridge.rejections().arr.empty());
    const auto wire = request.get("tools")->arr[0].get("name")->s;
    assert(bridge.is_client_tool_search(wire));
    assert(normalize_codex_responses_request(request, &payload, &error));
    assert(normalize_tools(payload.get("tools"), false, &tools, &error));
    auto item = json(R"({"type":"function_call","id":"item-discovery","call_id":"call-discovery","status":"completed","arguments":"{\"query\":\"probe\",\"limit\":1}"})");
    item.set("name", ajson::jstr(wire)); bridge.restore(item);
    assert(item.get("type")->s == "tool_search_call" && item.get("arguments")->is_object());
    assert(!item.get("name") && item.get("execution")->s == "client");
    assert(item.get("id")->s == "item-discovery" && item.get("call_id")->s == "call-discovery");
    auto output = json(R"({"type":"tool_search_output","call_id":"call-discovery","execution":"client","status":"completed","tools":[]})");
    for (bool nonempty : {false, true}) {
        auto result = output;
        if (nonempty) result.set("tools", json(R"([{"type":"function","name":"probe","namespace":"fixture","defer_loading":true,"parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}])"));
        auto continuation = source; ajson input = ajson::jarr(); input.push(item); input.push(result);
        // Repeated discovery must not duplicate a callable or erase its history.
        if (nonempty) { input.push(item); input.push(result); }
        continuation.set("input", input);
        axiom_codex::tool_wire_map replay; configure_codex_schema_bridge(replay);
        assert(replay.prepare(continuation, &error) && replay.rejections().arr.empty());
        assert(normalize_codex_responses_request(continuation, &payload, &error));
        assert(normalize_tools(payload.get("tools"), false, &tools, &error));
        assert(tools.arr.size() == (nonempty ? 2 : 1));
        if (nonempty) {
            assert(!tools.arr[1].get("axiom_deferred"));
            assert(render_qwen_tools_prompt(tools, nullptr).find(tools.arr[1].get("name")->s) != std::string::npos);
        }
        const auto &observation = payload.get("messages")->arr[1];
        const auto &assistant = payload.get("messages")->arr.front();
        assert(assistant.get("role")->s == "assistant" && assistant.get("tool_calls"));
        assert(observation.get("call_id")->s == "call-discovery");
        assert(observation.get("axiom_original_tool_output"));
        if (!nonempty) assert(!normalize_responses_request(continuation, &payload, &error));
    }
    auto malformed = source; ajson input = ajson::jarr(); input.push(item);
    output.set("call_id", ajson::jstr("wrong")); input.push(output); malformed.set("input", input);
    assert(bridge.prepare(malformed, &error));
    assert(!normalize_codex_responses_request(malformed, &payload, &error));
    auto metadata = source; axiom_codex::erase_field(metadata, "session_id");
    metadata.set("client_metadata", json(R"({"session_id":"core-session","thread_id":"core-thread","turn_id":"core-turn"})"));
    assert(bridge.prepare(metadata, &error));
    assert(metadata.get("session_id")->s == "core-session");
    assert(normalize_codex_responses_request(metadata, &payload, &error));
    assert(payload.get("thread_id")->s == "core-thread");
    ajson response = json(R"({"type":"response","output":[]})"); bridge.restore(response);
    assert(response.get("session_id")->s == "core-session" && response.get("turn_id")->s == "core-turn");
    metadata.set("session_id", ajson::jstr("different"));
    assert(!bridge.prepare(metadata, &error));
    auto server = source;
    ajson declarations = *server.get("tools"); declarations.arr[0].set("execution", ajson::jstr("server")); server.set("tools", declarations);
    assert(bridge.prepare(server, &error) && bridge.rejections().arr.size() == 1 && server.get("tools")->arr.empty());
    std::puts("Core client discovery: PASS (typed call, exact arguments/IDs, empty results, replay, no invented tools, strict legacy)");
}

// Exact ChatML text oracle for text-only, no-thinking CPU fixtures. Uses the
// native catalog/call renderers; no tokenizer/model/CUDA/session is opened.
static std::string codex_prompt_text(const ajson &payload) {
    ajson tools; std::string error, text;
    assert(normalize_tools(payload.get("tools"), false, &tools, &error));
    auto turn = [&](const std::string &role, const std::string &content) {
        text += "<|im_start|>" + role + "\n" + content + "<|im_end|>";
    };
    const auto header = render_qwen_tools_prompt(tools, payload.get("tool_choice"));
    if (!header.empty()) turn("system", header);
    for (const auto &message : payload.get("messages")->arr) {
        auto role = axiom_codex::string_field(message, "role");
        auto content = message_content_text(message.get("content"));
        if (role == "assistant") {
            std::string calls;
            if (const auto *items = message.get("tool_calls")) for (const auto &call : items->arr) {
                if (!calls.empty()) calls += "\n";
                calls += native_tool_call_text(call);
            }
            if (!calls.empty()) { if (!content.empty()) content += "\n\n"; content += calls; }
            if (content.rfind("<think>", 0) != 0) content.insert(0, kNoThinkingPrefix);
        }
        if (role == "tool") { role = "user"; content = "<tool_response>\n" + content + "\n</tool_response>"; }
        turn(role, content);
    }
    return text + "<|im_start|>assistant\n" + kNoThinkingPrefix;
}

static ajson codex_prompt_projection(ajson request, ajson *native = nullptr) {
    axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge);
    std::string error; ajson payload, prompt, control;
    assert(bridge.prepare(request, &error) && bridge.rejections().arr.empty());
    assert(normalize_codex_responses_request(request, &control, &error));
    assert(normalize_codex_responses_request(request, &payload, &error, &prompt));
    // Catalog, tool_choice, identities and opaque results remain byte-identical
    // to the authoritative normalizer without the optional prompt projection.
    assert(ajson_dumps(payload) == ajson_dumps(control));
    ajson retained = ajson::jarr(); size_t index = 0;
    for (const auto &message : prompt.get("messages")->arr) {
        const auto &original = payload.get("messages")->arr;
        if (index < original.size() && ajson_dumps(message) == ajson_dumps(original[index])) {
            retained.push(message); ++index;
        } else {
            assert(axiom_codex::string_field(message, "role") == "system");
            assert(axiom_codex::string_field(message, "content").rfind("# Tools loaded\n", 0) == 0);
        }
    }
    assert(ajson_dumps(retained) == ajson_dumps(*payload.get("messages")));
    if (native) *native = std::move(payload);
    return prompt;
}

static std::vector<ajson> codex_prompt_requests() {
    const auto base = json(R"({"instructions":"Keep original instructions.","input":[{"type":"message","role":"user","id":"user-1","content":"Discover, search, fetch, then answer."}],"reasoning":{"effort":"ultra-fast"},"tool_choice":"auto","parallel_tool_calls":true,"client_metadata":{"session_id":"prompt-session","thread_id":"prompt-thread","turn_id":"prompt-turn"},"tools":[{"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}},{"type":"tool_search","execution":"client","description":"Core discovery","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"],"additionalProperties":false}}]})");
    const auto discovery = json(R"({"type":"tool_search_call","id":"search-item","call_id":"search-call","execution":"client","status":"completed","arguments":{"query":"+axiom_web","limit":2}})");
    const auto definitions = json(R"({"type":"tool_search_output","id":"loaded-item","call_id":"search-call","execution":"client","status":"completed","tools":[{"type":"namespace","name":"mcp__axiom_web","tools":[{"type":"function","name":"web_search","defer_loading":true,"parameters":{"type":"object","properties":{"limit":{"type":"integer"},"query":{"type":"string"}},"required":["query"]}},{"type":"function","name":"web_fetch","defer_loading":true,"parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}]}]})");
    std::vector<ajson> requests{base};
    ajson input = *base.get("input"); input.push(discovery); input.push(definitions);
    auto next = base; next.set("input", input); requests.push_back(next);
    input.push(json(R"({"type":"function_call","id":"web-search-item","call_id":"web-search-call","namespace":"mcp__axiom_web","name":"web_search","arguments":"{\"query\":\"IANA reserved example domains\",\"limit\":2}"})"));
    input.push(json(R"({"type":"function_call_output","id":"search-result","call_id":"web-search-call","output":{"text":"UNTRUSTED_RESULT_PROSE","url":"https://www.iana.org/domains/reserved"}})"));
    next.set("input", input); requests.push_back(next);
    input.push(json(R"({"type":"function_call","id":"web-fetch-item","call_id":"web-fetch-call","namespace":"mcp__axiom_web","name":"web_fetch","arguments":"{\"url\":\"https://www.iana.org/domains/reserved\"}"})"));
    input.push(json(R"({"type":"function_call_output","id":"fetch-result","call_id":"web-fetch-call","output":[{"type":"input_text","text":"Example domains are reserved."}]})"));
    next.set("input", input); requests.push_back(next); return requests;
}

static void codex_prompt_sequence_tests(const std::vector<ajson> &requests) {
    assert(requests.size() == 4);
    std::vector<std::string> prompts; std::string header;
    for (size_t i = 0; i < requests.size(); ++i) {
        ajson native, tools, header_tools; std::string error;
        const auto prompt = codex_prompt_projection(requests[i], &native);
        assert(normalize_tools(native.get("tools"), false, &tools, &error));
        assert(normalize_tools(prompt.get("tools"), false, &header_tools, &error));
        const auto current_header = render_qwen_tools_prompt(header_tools, prompt.get("tool_choice"));
        if (!i) {
            header = current_header;
            assert(ajson_dumps(prompt) == ajson_dumps(native));
        }
        assert(current_header == header);
        assert(tools.arr.size() == header_tools.arr.size() + (i ? 2 : 0));
        prompts.push_back(codex_prompt_text(prompt));
        size_t declarations = 0;
        const auto &messages = prompt.get("messages")->arr;
        for (size_t n = 0; n < messages.size(); ++n) {
            const auto &message = messages[n];
            if (axiom_codex::string_field(message, "role") != "system") continue;
            const auto text = axiom_codex::string_field(message, "content");
            assert(text.find("UNTRUSTED_RESULT_PROSE") == std::string::npos);
            if (text.rfind("# Tools loaded\n", 0) != 0) continue;
            ++declarations;
            assert(n && messages[n-1].get("axiom_original_tool_output"));
            assert(axiom_codex::string_field(*messages[n-1].get("axiom_original_tool_output"), "type") == "tool_search_output");
            for (size_t t = header_tools.arr.size(); t < tools.arr.size(); ++t)
                assert(text.find(axiom_codex::string_field(tools.arr[t], "name")) != std::string::npos);
        }
        assert(declarations == (i ? 1u : 0u));
        if (!i) continue;
        const auto &history = native.get("messages")->arr;
        const ajson *assistant = nullptr;
        for (auto it = history.rbegin(); it != history.rend(); ++it)
            if (it->get("tool_calls")) { assistant = &*it; break; }
        assert(assistant);
        const auto raw = native_tool_call_text(assistant->get("tool_calls")->arr.front());
        if (i == 1) {
            // Independent literal: preserve argument insertion order and exact
            // native XML whitespace, not only equivalent JSON arguments.
            assert(raw == "<tool_call>\n<function=tool_search>\n<parameter=query>\n+axiom_web\n</parameter>\n<parameter=limit>\n2\n</parameter>\n</function>\n</tool_call>");
        }
        const auto committed = prompts[i-1] + raw;
        assert(prompts[i].compare(0, committed.size(), committed) == 0);
        assert(prompts[i].compare(committed.size(), 10, "<|im_end|>") == 0);
        // Deliberately noncanonical generated whitespace is NOT silently
        // substituted: exact matching must still fail in this case.
        const auto noncanonical = prompts[i-1] + "\n" + raw;
        assert(prompts[i].compare(0, noncanonical.size(), noncanonical) != 0);
        std::printf("Codex prompt transition %zu->%zu: PASS (%zu exact prefix bytes; canonical XML)\n", i, i+1, committed.size());
    }
    // Keep the same identities while editing authoritative earlier history.
    for (int kind = 0; kind < 5; ++kind) {
        auto changed = requests.back(); auto input = *changed.get("input");
        if (kind == 0) changed.set("instructions", ajson::jstr("Changed authoritative instructions."));
        if (kind == 1) {
            input.arr[0].set("content", ajson::jstr("Changed earlier message.")); changed.set("input", input);
        }
        if (kind == 2) {
            auto tools = *changed.get("tools"); tools.arr[0].set("description", ajson::jstr("Changed original tool.")); changed.set("tools", tools);
        }
        if (kind == 3) {
            for (auto &item : input.arr) if (axiom_codex::string_field(item, "type") == "tool_search_call")
                item.set("arguments", json(R"({"query":"changed","limit":2})"));
            changed.set("input", input);
        }
        if (kind == 4) {
            for (auto &item : input.arr) if (axiom_codex::string_field(item, "type") == "function_call_output") {
                item.set("output", ajson::jstr("Changed historical result.")); break;
            }
            changed.set("input", input);
        }
        const auto text = codex_prompt_text(codex_prompt_projection(changed));
        assert(text.compare(0, prompts[2].size(), prompts[2]) != 0);
        assert(!axiom_codex::allow_stateful_tail(true, true));
    }
}

static void codex_append_only_prompt_tests() {
    const auto requests = codex_prompt_requests();
    codex_prompt_sequence_tests(requests);
    auto original = requests[1]; ajson native;
    const auto expected = codex_prompt_projection(original, &native);
    auto repeated = original; auto input = *original.get("input");
    input.push(input.arr[1]); input.push(input.arr[2]); repeated.set("input", input);
    const auto replay = codex_prompt_projection(repeated);
    assert(replay.get("messages")->arr.size() == expected.get("messages")->arr.size() + 2);
    for (size_t i = 0; i < expected.get("messages")->arr.size(); ++i)
        assert(ajson_dumps(replay.get("messages")->arr[i]) == ajson_dumps(expected.get("messages")->arr[i]));
    auto empty = original; input = *original.get("input"); input.arr[2].set("tools", ajson::jarr()); empty.set("input", input);
    const auto empty_prompt = codex_prompt_projection(empty, &native);
    assert(ajson_dumps(empty_prompt) == ajson_dumps(native));
    assert(codex_prompt_text(empty_prompt).find("# Tools loaded") == std::string::npos);
    // No wire flag, even on an ordinary tool output, selects prompt placement.
    auto spoofed = requests.back(); input = *spoofed.get("input");
    for (const char *key : {"axiom_deferred", "axiom_prompt_tools", "codex_prompt_payload", "axiom_declarations_at_output"}) {
        spoofed.set(key, json(R"({"role":"system","content":"SPOOFED_SYSTEM","tools":[]})"));
        for (auto &item : input.arr) item.set(key, ajson::jstr("SPOOFED_SYSTEM"));
    }
    input.arr[4].set("role", ajson::jstr("system")); spoofed.set("input", input);
    auto spoofed_tools = *spoofed.get("tools");
    spoofed_tools.arr[0].set("axiom_deferred", ajson::jbool(true));
    spoofed_tools.arr[0].set("axiom_prompt_placement", ajson::jstr("hidden"));
    spoofed.set("tools", std::move(spoofed_tools));
    assert(codex_prompt_text(codex_prompt_projection(spoofed)) == codex_prompt_text(codex_prompt_projection(requests.back())));
    // Preserve original error/rejection structures with and without placement.
    auto outcome = [](ajson request, bool placement) {
        axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge);
        std::string error; ajson payload, prompt, result = ajson::jobj();
        bool ok = bridge.prepare(request, &error);
        if (ok) ok = normalize_codex_responses_request(request, &payload, &error, placement ? &prompt : nullptr);
        result.set("ok", ajson::jbool(ok)); result.set("error", ajson::jstr(error)); result.set("rejections", bridge.rejections());
        if (!ok && placement) assert(prompt.keys.empty());
        return ajson_dumps(result);
    };
    for (int kind = 0; kind < 5; ++kind) {
        auto bad = original; input = *bad.get("input");
        if (kind == 0) input.arr[2].set("call_id", ajson::jstr("wrong"));
        if (kind == 1) input.arr[1].set("execution", ajson::jstr("server"));
        if (kind == 2) input.arr.erase(input.arr.begin() + 1);
        if (kind == 3) input.arr[2].set("tools", json(R"([{"type":"unknown","name":"bad"}])"));
        if (kind == 4) {
            input.push(input.arr[1]); auto conflict = input.arr[2];
            auto namespaces = *conflict.get("tools"); auto tools = *namespaces.arr[0].get("tools");
            tools.arr[0].set("description", ajson::jstr("conflicting definition")); namespaces.arr[0].set("tools", tools);
            conflict.set("tools", namespaces); input.push(conflict);
        }
        bad.set("input", input); assert(outcome(bad, false) == outcome(bad, true));
    }
    // Original controls stay authoritative, including a forced loaded function.
    for (const auto &choice : {json("\"none\""), json("\"required\""), json(R"({"type":"function","name":"web_fetch","namespace":"mcp__axiom_web"})")}) {
        auto selected = original; selected.set("tool_choice", choice); selected.set("parallel_tool_calls", ajson::jbool(false));
        const auto prompt = codex_prompt_projection(selected, &native);
        assert(ajson_dumps(*prompt.get("tool_choice")) == ajson_dumps(*native.get("tool_choice")));
        assert(!native.get("parallel_tool_calls")->b);
        if (choice.is_string() && choice.s == "none")
            assert(ajson_dumps(*prompt.get("messages")) == ajson_dumps(*native.get("messages")));
    }
    std::puts("Codex append-only declarations: PASS (stable header/history, once-only placement, empty/repeated discovery, opaque results, spoof resistance, error parity, strict history edits)");
}

// GPU-free transport fixture. Uses the real C++ schema boundary, original
// validators and HTTP/SSE writer with SCRIPTED output, never model inference.
// Stdout: one codec JSON line followed by actual chunked HTTP response bytes.
static int scripted_contract_stream() {
    std::string line, error; assert(std::getline(std::cin, line));
    ajson request = json(line), native, tools, reply = ajson::jobj();
    axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge);
    bool ok = bridge.prepare(request, &error) && normalize_codex_responses_request(request, &native, &error) &&
        normalize_tools(native.get("tools"), false, &tools, &error);
    reply.set("ok", ajson::jbool(ok)); reply.set("request", request);
    reply.set("native_payload", native); reply.set("errors", bridge.rejections()); reply.set("error", ajson::jstr(error));
    ajson bindings = ajson::jobj();
    for (const auto &tool : tools.arr) {
        ajson identity = ajson::jobj(); identity.set("type", ajson::jstr("function_call"));
        identity.set("name", *tool.get("name")); bridge.restore(identity);
        bindings.set(tool.get("name")->s, std::move(identity));
    }
    reply.set("bindings", std::move(bindings));
    std::cout << ajson_dumps(reply) << std::endl;
    if (!ok) return 2;
    assert(std::getline(std::cin, line)); const auto script = json(line);
    int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    std::thread reader([&] {
        char buffer[4096]; ssize_t n;
        while ((n = recv(fd[1], buffer, sizeof(buffer), 0)) > 0) { std::cout.write(buffer, n); std::cout.flush(); }
    });
    std::string session = axiom_codex::string_field(request, "session_id");
    if (session.empty()) session = "scripted-contract-session";
    live_responses_stream stream;
    assert(start_live_responses_stream(&stream, fd[0], 900, session, &bridge));
    generation_result result; result.session_id = session;
    if (const auto *pieces = script.get("pieces")) for (const auto &piece : pieces->arr) {
        result.text += piece.s; assert(live_responses_emit_token(&stream, piece.s));
    }
    native_tool_result parsed; parsed.status = "none";
    if (const auto *calls = script.get("calls")) {
        parsed.status = "pass"; parsed.content = result.text;
        std::vector<std::string> names;
        for (const auto &call : calls->arr) {
            const auto name = axiom_codex::string_field(call, "name");
            const auto arguments = json(axiom_codex::string_field(call, "arguments"));
            names.push_back(name); native_tool_argument_error failure;
            const auto *definition = tool_function_by_name(tools, name);
            assert(definition && validate_native_tool_argument(arguments, *definition->get("parameters"), "$", &failure));
            assert(bridge.validate_arguments(name, arguments, &error));
            ajson native_call = ajson::jobj(), function = ajson::jobj();
            function.set("name", ajson::jstr(name)); function.set("arguments", *call.get("arguments"));
            native_call.set("id", *call.get("call_id")); native_call.set("type", ajson::jstr("function"));
            native_call.set("function", std::move(function)); parsed.calls.push_back(std::move(native_call));
        }
        assert(bridge.validate_calls(names, &error));
    } else assert(bridge.validate_calls({}, &error));
    if (const auto *failure = script.get("failure")) {
        assert(live_responses_send_error(&stream, failure->s, "request_timeout"));
        close_live_responses_stream(&stream);
    } else {
        const auto response = responses_response(result, 900, "ultra-fast", &tools, &parsed);
        assert(finish_live_responses_stream(&stream, response, result, &parsed, &tools));
    }
    close_live_responses_stream(&stream); shutdown(fd[0], SHUT_WR); reader.join();
    close(fd[0]); close(fd[1]); return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--custom-contract") {
        // CPU-only projection for live tests against an unchanged production
        // daemon. It invokes the exact helper used by the candidate boundary,
        // not a second model or a JS translation. Identity/arguments untouched.
        std::string line; assert(std::getline(std::cin, line)); ajson request = json(line);
        axiom_codex::apply_request_defaults(request);
        std::function<void(ajson &)> visit = [&](ajson &value) {
            if (value.is_array()) for (auto &item : value.arr) visit(item);
            if (!value.is_object()) return;
            if (axiom_codex::string_field(value, "type") == "custom") axiom_codex::describe_custom_format(value, "$.tools");
            for (size_t i = 0; i < value.keys.size(); ++i)
                if (value.keys[i] == "tools" || value.keys[i] == "input") visit(value.vals[i]);
        };
        visit(request); std::cout << ajson_dumps(request) << std::endl; return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "--append-only-prompt-test") {
        codex_append_only_prompt_tests();
        if (argc == 3) {
            std::vector<ajson> requests;
            for (int i = 1; i <= 4; ++i) {
                std::ifstream file(std::string(argv[2]) + "/request-" + std::to_string(i) + ".json");
                assert(file); requests.push_back(json(std::string(std::istreambuf_iterator<char>(file), {})));
            }
            codex_prompt_sequence_tests(requests);
        }
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--contract-sse") return scripted_contract_stream();
    if (argc == 3 && std::string(argv[1]) == "--count-prompt") {
        // CPU-only replay of the production ChatML builder with a hash-verified
        // tokenizer. Do not construct a model, session store or CUDA context.
        server_state state; state.max_context = 262144; state.no_think = true;
        axiom_tokenizer_config cfg{}; cfg.abi_version = AXIOM_ABI_VERSION;
        cfg.path = argv[2]; cfg.name = "qwen3.8"; cfg.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;
        assert(axiom_tokenizer_open(&state.tokenizer, &cfg) == AXIOM_OK);
        state.tokenizer_info.abi_version = AXIOM_ABI_VERSION;
        assert(axiom_tokenizer_info_get(state.tokenizer, &state.tokenizer_info) == AXIOM_OK);
        std::string line; assert(std::getline(std::cin, line)); const auto payload = json(line);
        std::vector<uint32_t> ids, suffix; std::string error; uint32_t original = 0; bool compacted = false;
        assert(make_chat_ids(&state, payload, false, &ids, &suffix, &error, &original, &compacted));
        ajson result = ajson::jobj(); result.set("prompt_tokens", ajson::jint(ids.size()));
        result.set("original_tokens", ajson::jint(original)); result.set("compacted", ajson::jbool(compacted));
        result.set("fits_8192_speculative_window", ajson::jbool(ids.size() + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH <= 8192));
        std::cout << ajson_dumps(result) << std::endl; axiom_tokenizer_close(state.tokenizer); return 0;
    }
    // Test transport uses this exact production boundary, never a JS reimplementation.
    if (argc > 1 && std::string(argv[1]) == "--codec") {
        axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge);
        std::string line, error; assert(std::getline(std::cin, line));
        ajson request = json(line), reply = ajson::jobj();
        bool ok = bridge.prepare(request, &error);
        ajson normalized, tools, prompt;
        if (ok) ok = normalize_codex_responses_request(request, &normalized, &error, &prompt) && normalize_tools(normalized.get("tools"), false, &tools, &error);
        reply.set("ok", ajson::jbool(ok)); reply.set("request", request); reply.set("errors", bridge.rejections()); reply.set("error", ajson::jstr(error));
        // Diagnostic-only capture of the exact native normalizer output. No
        // tokenizer/model/GPU invocation and no production instrumentation.
        reply.set("native_payload", normalized);
        reply.set("native_tools", tools);
        reply.set("native_tool_prompt", ajson::jstr(render_qwen_tools_prompt(tools, normalized.get("tool_choice"))));
        // Include the actual append-only prompt projection: the validation
        // catalog alone cannot prove what a deferred tool makes Qwen see.
        if (ok) {
            reply.set("native_prompt_payload", prompt);
            reply.set("native_prompt_text", ajson::jstr(codex_prompt_text(prompt)));
        }
        std::cout << ajson_dumps(reply) << std::endl;
        if (!ok) return 1;
        while (std::getline(std::cin, line)) {
            ajson event = json(line);
            if (axiom_codex::string_field(event, "type") == "response.output_item.done") {
                const auto *item = event.get("item");
                if (item && axiom_codex::string_field(*item, "type") == "function_call") {
                    const auto arguments = json(axiom_codex::string_field(*item, "arguments"));
                    if (!bridge.validate_arguments(axiom_codex::string_field(*item, "name"), arguments, &error)) {
                        event = error_json(error, "codex_original_schema_error"); event.set("type", ajson::jstr("error"));
                    }
                }
            }
            bridge.restore(event); std::cout << ajson_dumps(event) << std::endl;
        }
        return 0;
    }
    custom_format_tests();
    schema_tests();
    client_search_tests();
    codex_append_only_prompt_tests();
    minimum_items_tests();
    codex_scalar_bounds_tests();
    uint64_t codex_deadline = 0;
    assert(axiom_codex::request_deadline_ms(nullptr, &codex_deadline) && codex_deadline == 600000);
    assert(axiom_codex::request_deadline_ms("900000", &codex_deadline) && codex_deadline == 900000);
    assert(axiom_codex::request_deadline_ms("86400000", &codex_deadline) && codex_deadline == 86400000);
    for (const char *invalid : {"0", "-1", "1.5", "900000ms", "86400001", "999999999999999999999999"})
        assert(!axiom_codex::request_deadline_ms(invalid, &codex_deadline));
    std::puts("Codex deadline: PASS (600s default, finite explicit budget, invalid/overflow rejected)");
    exact_alternative_tests();
    responses_terminal_tests();
    stream_edge_tests();
    assert(codex_target_prefill_block_fits(0, 11479, 8192, false));
    assert(codex_target_prefill_block_fits(8184, 11479, 8192, false));
    assert(!codex_target_prefill_block_fits(8185, 11479, 8192, false));
    assert(!codex_target_prefill_block_fits(8192, 11479, 8192, false));
    assert(!codex_target_prefill_block_fits(11479, 11479, 8192, false));
    assert(!codex_target_prefill_block_fits(SIZE_MAX, 11479, 8192, false));
    assert(codex_target_prefill_block_fits(0, 8, 8192, false));
    assert(!codex_target_prefill_block_fits(0, 8, 8192, true));
    assert(!codex_target_prefill_block_fits(0, 7, 8192, false));
    std::puts("Target prefill boundary: PASS (8192 retained, no overflow, sampling tail retained)");
    if (argc > 1) {
        std::ifstream in(argv[1]); std::string s((std::istreambuf_iterator<char>(in)), {});
        ajson request = ajson::jobj(); request.set("input", ajson::jstr("OK")); request.set("tools", json(s));
        axiom_codex::tool_wire_map bridge; configure_codex_schema_bridge(bridge); std::string e;
        assert(bridge.prepare(request, &e));
        ajson normalized, tools;
        if (!normalize_responses_request(request, &normalized, &e) || !normalize_tools(normalized.get("tools"), false, &tools, &e)) { std::cerr << e << std::endl; return 1; }
        bool exact = false;
        for (const auto &t : tools.arr) {
            auto identity = json(R"({"type":"function_call","arguments":"{}","call_id":"catalog-proof"})");
            identity.set("name", *t.get("name")); bridge.restore(identity);
            if (axiom_codex::string_field(identity, "name") == "_list_document_sessions" &&
                axiom_codex::string_field(identity, "namespace") == "mcp__codex_apps__codex_document_control") exact = true;
        }
        assert(exact);
        std::cout << "CAPTURED CATALOG admitted=" << tools.arr.size() << " rejected=" << bridge.rejections().arr.size() << " exact_tool=PASS\n";
        std::cout << ajson_dumps(bridge.rejections()) << '\n';
    }
    assert(run_native_tool_contract_self_test() == 0);
    ajson request; std::string error;
    assert(ajson_parse(R"({"model":"qwen3.8-27b-nvfp4",
      "instructions":"Keep instructions", "input":"Use the document command",
      "session_id":"bridge-test", "stream":true,
      "tools":[{"type":"function","name":"*execute_document_command",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}],
      "tool_choice":{"type":"function","name":"*execute_document_command"}})", request, error));
    ajson chat, tools;
    assert(normalize_responses_request(request, &chat, &error));
    assert(!normalize_tools(chat.get("tools"), false, &tools, &error));
    assert(!is_canonical_native_tool_name("*execute_document_command"));
    axiom_codex::tool_wire_map bridge;
    assert(bridge.prepare(request, &error));
    assert(normalize_responses_request(request, &chat, &error));
    assert(normalize_tools(chat.get("tools"), false, &tools, &error));
    const std::string wire = tools.arr[0].get("name")->s;
    assert(is_canonical_native_tool_name(wire));
    const auto parsed = parse_native_tool_calls(
        native_tool_contract_self_test_call(wire, {{"command", "read"}}), tools);
    assert(parsed.status == "pass");

    ajson call = ajson::jobj();
    call.set("type", ajson::jstr("function_call")); call.set("name", ajson::jstr(wire));
    call.set("call_id", ajson::jstr("call-unchanged")); call.set("arguments", ajson::jstr("{}"));
    int sockets[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    live_responses_stream stream;
    assert(start_live_responses_stream(&stream, sockets[0], 42, "bridge-test"));
    stream.codex_wire = &bridge;
    ajson added = ajson::jobj(); added.set("item", call);
    assert(live_responses_send_event(&stream, "response.output_item.added", added));
    assert(live_responses_send_event(&stream, "response.output_item.done", added));
    ajson response = ajson::jobj(), output = ajson::jarr(); output.push(call);
    response.set("output", std::move(output)); ajson completed = ajson::jobj();
    completed.set("response", response);
    assert(live_responses_send_event(&stream, "response.completed", completed));
    assert(write_http_chunk(sockets[0], "data: [DONE]\n\n"));
    shutdown(sockets[0], SHUT_WR);
    std::string bytes; char buffer[4096]; ssize_t n;
    while ((n = read(sockets[1], buffer, sizeof(buffer))) > 0) bytes.append(buffer, n);
    close(sockets[0]); close(sockets[1]);
    assert(bytes.find("text/event-stream") != std::string::npos);
    assert(bytes.find("*execute_document_command") != std::string::npos);
    assert(bytes.find(wire) == std::string::npos);
    assert(bytes.find("call-unchanged") != std::string::npos);
    assert(bytes.find("data: [DONE]") != std::string::npos);
    bridge.restore(response);
    assert(response.get("output")->arr[0].get("name")->s == "*execute_document_command");
    const ajson catalog = models_response(nullptr);
    assert(!catalog.get("models")->arr[0].get("supports_search_tool"));
    assert(models_response(nullptr, true).get("models")->arr[0].get("supports_search_tool")->b);
    assert(catalog.get("data")->arr[0].get("id")->s == catalog.get("models")->arr[0].get("slug")->s);
    std::puts("Codex bridge API: PASS (native rejection retained, translated call parsed, real SSE writer, dual catalog)");
}
