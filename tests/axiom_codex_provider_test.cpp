#include "axiom_codex_provider.h"
#include <cassert>
#include <iostream>

static ajson parse(const char *text) {
    ajson result; std::string error;
    assert(ajson_parse(text, result, error));
    return result;
}
static void same(const ajson &a, const ajson &b) { assert(ajson_dumps(a) == ajson_dumps(b)); }
static void readable_identity_tests() {
    auto tool = [](const std::string &scope, const std::string &name) {
        ajson t = parse(R"({"type":"function","parameters":{"type":"object"}})");
        t.set("name", ajson::jstr(name));
        if (!scope.empty()) t.set("namespace", ajson::jstr(scope));
        return t;
    };
    ajson request = parse(R"({"tools":[],"input":[],"session_id":"root","parallel_tool_calls":true})");
    const std::vector<std::pair<std::string, std::string>> identities = {
        {"mcp__axiom_web", "web_search"}, {"mcp__axiom_web", "web_fetch"},
        {"alpha", "run"}, {"beta", "run"}, {"a_b", "c"}, {"a", "b_c"},
        {"", "*execute_document_command"}, {"", "execute_document_command"},
        {"", "Bash"}, {"", "bash"}, {"", "z5f"}, {"", "_"},
        {"", "zz"}, {"", "z"}, {"", "a__b"}, {"", "a_b"},
        {"", "codex_n4_bash_s0"}, {"s", "a_s1_b"}, {"b_s1_s", "a"},
        {"", std::string(100, 'a')}, {"long", std::string(99, 'a') + "b"},
        {"UTF-8", "caf\xc3\xa9"}, {"UTF-8", "cafe"}
    };
    ajson catalog = ajson::jarr();
    for (const auto &[scope, name] : identities) catalog.push(tool(scope, name));
    request.set("tools", catalog);
    const auto original = request;
    std::string error; axiom_codex::tool_wire_map bridge;
    assert(bridge.prepare(request, &error) && bridge.rejections().arr.empty());
    const auto &lowered = request.get("tools")->arr;
    assert(lowered.size() == identities.size());
    std::set<std::string> names;
    for (size_t i = 0; i < lowered.size(); ++i) {
        const auto wire = lowered[i].get("name")->s;
        assert(axiom_codex::canonical(wire) && names.insert(wire).second);
        ajson call = parse(R"({"type":"function_call","arguments":"{}","call_id":"call-fixed","id":"item-fixed"})");
        call.set("name", ajson::jstr(wire));
        assert(bridge.validate_calls({wire}, &error));
        bridge.restore(call);
        assert(call.get("name")->s == identities[i].second);
        assert(axiom_codex::string_field(call, "namespace") == identities[i].first);
        assert(call.get("call_id")->s == "call-fixed" && call.get("id")->s == "item-fixed");
    }
    // The live failure was copying a long random digest. Both web names now
    // remain readable, without a digest in the model-visible declaration.
    assert(lowered[0].get("name")->s == "codex_n10_web_search_s18_mcpz5fz5faxiom_web");
    assert(lowered[1].get("name")->s == "codex_n9_web_fetch_s18_mcpz5fz5faxiom_web");
    const auto fetch = lowered[1].get("name")->s;
    assert(!bridge.validate_calls({fetch + "x"}, &error));
    assert(!bridge.validate_arguments("codex_tool_f7ebb9c4df21fb3ed89bdd7ba164c7c64098c64098c69da9d1671f", parse("{}"), &error));
    // Translated identities must describe an actual native invocation, not
    // leave the model to interpret namespace.name as a shell/SDK expression.
    for (size_t i = 0; i < lowered.size(); ++i) {
        const auto wire = lowered[i].get("name")->s;
        if (wire == identities[i].second) continue;
        const auto description = axiom_codex::string_field(lowered[i], "description");
        assert(description.find("<function=" + wire + "> inside <tool_call>") != std::string::npos);
        assert(description.find("not a shell command") != std::string::npos);
        assert(description.find(identities[i].second) != std::string::npos);
    }
    // Ordinary unscoped tools retain their original description/schema.
    const auto original_command = parse(R"({"tools":[{"type":"function","name":"exec_command","description":"Execute a shell command.","parameters":{"type":"object"}}]})");
    auto unchanged = original_command; axiom_codex::tool_wire_map plain;
    assert(plain.prepare(unchanged, &error)); same(*unchanged.get("tools"), *original_command.get("tools"));
    ajson reversed = original;
    std::reverse(catalog.arr.begin(), catalog.arr.end()); reversed.set("tools", catalog);
    axiom_codex::tool_wire_map reordered; assert(reordered.prepare(reversed, &error));
    for (size_t i = 0; i < lowered.size(); ++i)
        assert(lowered[i].get("name")->s == reversed.get("tools")->arr[lowered.size()-1-i].get("name")->s);
    // Core's next request transports original namespace/name, not our wire.
    auto next = parse(R"({"tools":[],"input":[],"session_id":"root"})");
    auto discovered = parse(R"({"type":"tool_search_output","call_id":"discovery-fixed","tools":[]})");
    auto discovered_tools = ajson::jarr();
    discovered_tools.push(tool("mcp__axiom_web", "web_search")); discovered_tools.push(tool("mcp__axiom_web", "web_fetch"));
    discovered.set("tools", discovered_tools);
    auto input = ajson::jarr(); input.push(discovered);
    auto call = parse(R"({"type":"function_call","name":"web_search","namespace":"mcp__axiom_web","arguments":"{}","call_id":"search-fixed","id":"search-item"})");
    input.push(call); input.push(parse(R"({"type":"function_call_output","call_id":"search-fixed","output":"IANA result"})"));
    next.set("input", input); const auto before = next;
    axiom_codex::tool_wire_map continuation; assert(continuation.prepare(next, &error));
    assert(next.get("input")->arr[0].get("tools")->arr[1].get("name")->s == fetch);
    assert(axiom_codex::string_field(next.get("input")->arr[0].get("tools")->arr[1], "description") ==
           axiom_codex::string_field(lowered[1], "description"));
    call = next.get("input")->arr[1]; continuation.restore(call);
    for (const char *key : {"type", "name", "namespace", "arguments", "call_id", "id"})
        same(*call.get(key), *before.get("input")->arr[1].get(key));
    same(next.get("input")->arr[2], before.get("input")->arr[2]);
    std::cout << "Readable Codex wire: PASS (web, namespace collisions, escapes, Unicode, length, replay, IDs, strict rejection)\n";
}
int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--catalog") {
        ajson root = ajson::jobj(), models = ajson::jarr();
        models.push(axiom_codex::catalog_model("qwen3.8-27b-nvfp4", 262144, 1048576, "ultra-fast"));
        root.set("models", std::move(models));
        std::cout << ajson_dumps(root) << '\n'; return 0;
    }
    readable_identity_tests();
    {
        using axiom_codex::cache_session_id;
        const auto parent = parse(R"({"thread_id":"root","session_id":"root"})");
        auto child = parse(R"({"thread_id":"child","session_id":"root","turn_id":"turn-a"})");
        const auto before = ajson_dumps(child);
        assert(cache_session_id("root", parent) == "root");
        assert(cache_session_id("root", ajson::jobj()) == "root");
        const auto key = cache_session_id("root", child);
        assert(key.size() == 76 && key != "root");
        assert(ajson_dumps(child) == before);
        child.set("turn_id", ajson::jstr("turn-b"));
        assert(cache_session_id("root", child) == key);
        assert(cache_session_id("other-root", child) != key);
        child.set("thread_id", ajson::jstr("child-two"));
        assert(cache_session_id("root", child) != key);
        assert(cache_session_id("a", parse(R"({"thread_id":"bc"})")) !=
               cache_session_id("ab", parse(R"({"thread_id":"c"})")));
        std::cout << "Codex child cache: PASS (root unchanged, child isolated, stable across turns, wire untouched)\n";
    }
    const ajson original = parse(R"({
      "model":"qwen3.8-27b-nvfp4","instructions":"Keep *execute_document_command literally.",
      "session_id":"session-a","stream":true,"metadata":{"name":"Bash"},
      "input":[{"type":"message","role":"user","content":"*execute_document_command"},
        {"type":"function_call","name":"*execute_document_command","call_id":"call-a","arguments":"{\"name\":\"Bash\"}"},
        {"type":"function_call_output","call_id":"call-a","output":"{\"name\":\"Bash\"}"}],
      "tools":[{"type":"function","name":"*execute_document_command","parameters":{"type":"object","properties":{"name":{"type":"string"}}}},
        {"type":"function","name":"execute_document_command","parameters":{"type":"object"}},
        {"type":"custom","name":"Bash","description":"Raw input","format":{"type":"text"}}],
      "tool_choice":{"type":"function","name":"*execute_document_command"},
      "extra_field":{"do_not_touch":true}
    })");
    ajson wire = original;
    axiom_codex::tool_wire_map bridge; std::string error;
    assert(bridge.prepare(wire, &error));
    const auto &tools = wire.get("tools")->arr;
    const std::string alias = tools[0].get("name")->s;
    assert(axiom_codex::canonical(alias)); assert(alias != "execute_document_command");
    assert(tools[1].get("name")->s == "execute_document_command");
    assert(axiom_codex::canonical(tools[2].get("name")->s));
    assert(wire.get("tool_choice")->get("name")->s == alias);
    assert(wire.get("input")->arr[1].get("name")->s == alias);
    for (const char *field : {"instructions","session_id","stream","metadata","extra_field"})
        same(*wire.get(field), *original.get(field));
    same(*tools[0].get("parameters"), *original.get("tools")->arr[0].get("parameters"));
    same(wire.get("input")->arr[0], original.get("input")->arr[0]);
    same(wire.get("input")->arr[2], original.get("input")->arr[2]);

    ajson call = wire.get("input")->arr[1];
    ajson event = ajson::jobj(); event.set("type", ajson::jstr("response.output_item.added"));
    event.set("sequence_number", ajson::jint(5)); event.set("item", call);
    bridge.restore(event);
    same(*event.get("item"), original.get("input")->arr[1]);
    assert(event.get("sequence_number")->i == 5);
    ajson response = ajson::jobj(), output = ajson::jarr(); output.push(call);
    response.set("output", output); ajson completed = ajson::jobj();
    completed.set("response", response); bridge.restore(completed);
    same(completed.get("response")->get("output")->arr[0], original.get("input")->arr[1]);
    ajson failure = parse(R"({"type":"error","error":{"message":"*execute_document_command","code":"invalid_request_error"}})");
    ajson saved = failure; bridge.restore(failure); same(saved, failure);
    ajson delta = parse(R"({"type":"response.function_call_arguments.delta","call_id":"call-a","delta":"Bash"})");
    saved = delta; bridge.restore(delta); same(saved, delta);

    // Deterministic across requests/restarts and reordered catalogs.
    ajson reordered = original;
    ajson reversed = *original.get("tools"); std::swap(reversed.arr[0], reversed.arr[2]);
    reordered.set("tools", reversed); axiom_codex::tool_wire_map next;
    assert(next.prepare(reordered, &error));
    assert(reordered.get("tools")->arr[2].get("name")->s == alias);
    ajson continuation = original;
    continuation.set("tools", ajson::jarr());
    continuation.set("tool_choice", ajson::jstr("auto"));
    axiom_codex::tool_wire_map resumed;
    assert(resumed.prepare(continuation, &error));
    assert(continuation.get("input")->arr[1].get("name")->s == alias);

    ajson namespaces = parse(R"({"tools":[
      {"type":"namespace","name":"alpha","tools":[{"type":"function","name":"run","parameters":{}}]},
      {"type":"namespace","name":"beta","tools":[{"type":"function","name":"run","parameters":{}}]}],
      "tool_choice":{"type":"function","namespace":"beta","name":"run"},
      "input":[{"type":"function_call","namespace":"beta","name":"run","call_id":"ns-b","arguments":"{}"}]})");
    const ajson namespace_original = namespaces;
    axiom_codex::tool_wire_map scoped;
    assert(scoped.prepare(namespaces, &error));
    assert(namespaces.get("tools")->arr.size() == 2);
    assert(namespaces.get("tools")->arr[0].get("name")->s != namespaces.get("tools")->arr[1].get("name")->s);
    assert(namespaces.get("tool_choice")->get("name")->s == namespaces.get("tools")->arr[1].get("name")->s);
    call = namespaces.get("input")->arr[0]; scoped.restore(call);
    assert(call.get("name")->s == "run" && call.get("namespace")->s == "beta");
    assert(call.get("call_id")->s == "ns-b" && call.get("arguments")->s == "{}");

    ajson reserved = original;
    ajson reserved_tools = *reserved.get("tools");
    reserved_tools.arr[1].set("name", ajson::jstr(alias)); reserved.set("tools", reserved_tools);
    axiom_codex::tool_wire_map reserved_map; assert(reserved_map.prepare(reserved, &error));
    assert(reserved.get("tools")->arr[0].get("name")->s != reserved.get("tools")->arr[1].get("name")->s);
    ajson malformed = parse(R"({"tools":[{"type":"function","name":42}]})");
    saved = malformed; axiom_codex::tool_wire_map bad;
    assert(bad.prepare(malformed, &error));
    assert(malformed.get("tools")->arr.empty() && bad.rejections().arr.size() == 1);

    ajson custom = parse(R"({"tools":[{"type":"custom","name":"*patch"}],
      "input":[{"type":"custom_tool_call","name":"*patch","call_id":"custom-c","input":"raw\ntext"}],
      "tool_choice":"required"})");
    saved = custom; axiom_codex::tool_wire_map custom_map; assert(custom_map.prepare(custom, &error));
    call = custom.get("input")->arr[0]; custom_map.restore(call); same(call, saved.get("input")->arr[0]);
    same(*custom.get("tool_choice"), *saved.get("tool_choice"));
    ajson deferred = parse(R"({"input":[{"type":"tool_search_output","call_id":"search-a","tools":[
      {"type":"function","name":"*execute_document_command","defer_loading":true,"parameters":{}}]}]})");
    axiom_codex::tool_wire_map deferred_map; assert(deferred_map.prepare(deferred, &error));
    assert(deferred.get("input")->arr[0].get("tools")->arr[0].get("name")->s == alias);
    assert(deferred.get("input")->arr[0].get("tools")->arr[0].get("defer_loading")->b);
    std::cout << "Codex provider boundary: PASS (names, namespaces, replay, IDs, custom/deferred tools, SSE structures, errors)\n";
}
