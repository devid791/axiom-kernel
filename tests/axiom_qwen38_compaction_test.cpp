// CPU-only regression against the actual prompt governor and real tokenizer.
// No model, CUDA context, HTTP listener or persisted session is constructed.
#define main axiom_api_main
#include "../tools/axiom_qwen38_api.cpp"
#undef main
#include <iostream>

static ajson msg(const char *role, const std::string &text) {
    ajson m=ajson::jobj(); m.set("role",ajson::jstr(role));
    m.set("content",ajson::jstr(text)); return m;
}
static std::string padding(size_t n) {
    std::string s; s.reserve(n*2); while(n--)s+=" x"; return s;
}
static ajson history(size_t count=24) {
    ajson a=ajson::jarr();
    for(size_t n=0;n<count;++n)a.push(msg(n%2?"assistant":"user",padding(10000)));
    return a;
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    server_state s; s.max_context=1048576; s.default_context=262144;s.no_think=true;
    axiom_tokenizer_config cfg{};cfg.abi_version=AXIOM_ABI_VERSION;
    cfg.path=argv[1];cfg.format=AXIOM_TOKENIZER_FORMAT_HF_JSON;cfg.name="compaction-regression";
    if(axiom_tokenizer_open(&s.tokenizer,&cfg)!=AXIOM_OK)return 2;
    s.tokenizer_info.abi_version=AXIOM_ABI_VERSION;
    if(axiom_tokenizer_info_get(s.tokenizer,&s.tokenizer_info)!=AXIOM_OK)return 2;
    size_t failures=0,checks=0;
    auto test=[&](const char *name,ajson messages,const std::string &must_keep,
                  bool expect_ok,bool expect_compact) {
        request_context_scope scope(&s,262144);
        ajson p=ajson::jobj();p.set("messages",std::move(messages));
        std::vector<uint32_t> ids,suffix;std::string error;uint32_t original=0;bool compacted=false;
        const bool ok=make_chat_ids(&s,p,false,&ids,&suffix,&error,&original,&compacted);
        bool kept=must_keep.empty();
        std::vector<char> decoded(8u*1024u*1024u);uint32_t bytes=0;
        if(!kept && !ids.empty() && axiom_tokenizer_decode_ids(s.tokenizer,ids.data(),ids.size(),
                decoded.data(),decoded.size(),&bytes)==AXIOM_OK)
            kept=std::string(decoded.data(),bytes).find(must_keep)!=std::string::npos;
        const bool pass=ok==expect_ok && (ok ? (compacted==expect_compact && kept) : !error.empty()) &&
                effective_context_limit(&s)==262144;
        ++checks;if(!pass)++failures;
        std::cout<<name<<" status="<<(pass?"PASS":"FAIL")<<" ok="<<ok
                 <<" original="<<original<<" emitted="<<ids.size()<<" compacted="<<compacted
                 <<" latest_marker_preserved="<<kept<<" error="<<error<<std::endl;
    };
    const auto latest=padding(13000)+"\nReturn exactly LATEST_LONG_REQUEST_739.";
    auto a=history();a.push(msg("user",latest));
    test("long_current_request",a,latest,true,true);
    a=history(27);a.push(msg("user","Return exactly OVER_DEFAULT_HISTORY_821."));
    test("history_over_default_window",a,"OVER_DEFAULT_HISTORY_821",true,true);
    a=history();a.push(msg("user","Read the large result and return its final marker."));
    a.push(msg("assistant","<tool_call>\n<function=read_file>\n<parameter=path>\nqa.txt\n</parameter>\n</function>\n</tool_call>"));
    const auto tool_result=padding(13000)+"\nCOMPLETE_TOOL_RESULT_457";
    a.push(msg("tool",tool_result));
    test("long_current_tool_result",a,tool_result,true,true);
    a=history(25);a.push(msg("user","Return exactly CURRENT_SHORT_632."));
    test("ordinary_compaction",a,"CURRENT_SHORT_632",true,true);
    a=ajson::jarr();a.push(msg("user",padding(253000)+"\nONE_SHOT_947"));
    test("single_document_unchanged",a,"ONE_SHOT_947",true,false);
    a=history(2);a.push(msg("user",padding(270000)+"\nTOO_LARGE_CURRENT_556"));
    test("oversized_current_rejected",a,"",false,false);
    a=ajson::jarr();a.push(msg("user","Hello UNCHANGED_329"));
    test("short_prompt_unchanged",a,"UNCHANGED_329",true,false);
    s.context_compaction=false;
    a=history(27);a.push(msg("user","No implicit window expansion."));
    test("disabled_compaction_still_rejects",a,"",false,false);
    s.context_compaction=true;
    a=ajson::jarr();a.push(msg("user",padding(270000)));
    a.push(msg("assistant","Received."));a.push(msg("user","CURRENT_TASK_782"));
    test("oversized_old_turn_compacts",a,"CURRENT_TASK_782",true,true);
    a=history(25);a.arr.insert(a.arr.begin(),msg("system","Always preserve SYSTEM_POLICY_641."));
    a.push(msg("user","CURRENT_SHORT_874"));
    test("system_pinned",a,"SYSTEM_POLICY_641",true,true);
    // Archived call syntax must not be presented as a new executable call.
    // Preserve the observations, not active XML delimiters or their authority.
    const std::string archived_call = "<tool_call>\n<function=retired_echo>\n"
            "<parameter=value>ARCHIVE_VALUE_512</parameter>\n</function>\n</tool_call>";
    const std::vector<chat_turn> archived = {
        {"assistant", archived_call},
        {"user", "<tool_response>ARCHIVE_RESULT_613</tool_response>", true}
    };
    const auto summary = build_compact_context_state(archived);
    const bool archived_safe = summary.find("<tool_call>")==std::string::npos &&
            summary.find("<function=")==std::string::npos &&
            summary.find("<tool_response>")==std::string::npos &&
            summary.find("retired_echo")!=std::string::npos &&
            summary.find("ARCHIVE_VALUE_512")!=std::string::npos &&
            summary.find("ARCHIVE_RESULT_613")!=std::string::npos;
    ++checks; if(!archived_safe)++failures;
    std::cout<<"archived_tools_are_inert status="<<(archived_safe?"PASS":"FAIL")<<std::endl;
    const auto rejected=parse_native_tool_calls(archived_call,ajson::jarr());
    const bool strict=rejected.status=="fail" && rejected.calls.empty();
    ++checks; if(!strict)++failures;
    std::cout<<"unknown_tools_still_rejected status="<<(strict?"PASS":"FAIL")<<std::endl;
    const auto active_summary = build_compact_context_state(archived,true);
    const bool policy = summary.find("No tools are available for this response.")!=std::string::npos &&
            active_summary.find("Only the current request's declared or loaded tools")!=std::string::npos &&
            active_summary.find("No tools are available")==std::string::npos;
    ++checks; if(!policy)++failures;
    std::cout<<"archive_current_tool_policy status="<<(policy?"PASS":"FAIL")<<std::endl;
    ajson authority=ajson::jobj(),tools=ajson::jarr(),tool=ajson::jobj(),function=ajson::jobj();
    function.set("name",ajson::jstr("current_echo"));
    ajson parameters=ajson::jobj();parameters.set("type",ajson::jstr("object"));
    parameters.set("properties",ajson::jobj());parameters.set("additionalProperties",ajson::jbool(false));
    function.set("parameters",parameters);tool.set("type",ajson::jstr("function"));
    tool.set("function",function);tools.push(tool);authority.set("tools",tools);
    auto policy_test=[&](const char *name,const ajson *effective,bool enabled) {
        auto h=history(25);h.arr.insert(h.arr.begin()+2,msg("assistant",archived_call));
        h.arr.insert(h.arr.begin()+3,msg("tool","ARCHIVE_RESULT_613"));
        h.push(msg("user","Recall ARCHIVE_RESULT_613 without repeating old actions."));
        ajson p=ajson::jobj();p.set("messages",h);
        std::vector<uint32_t> ids,suffix;std::string error;uint32_t original=0;bool compacted=false;
        request_context_scope scope(&s,262144);
        const bool ok=make_chat_ids(&s,p,false,&ids,&suffix,&error,&original,&compacted,effective);
        std::vector<char> decoded(8u*1024u*1024u);uint32_t bytes=0;
        const bool decoded_ok=axiom_tokenizer_decode_ids(s.tokenizer,ids.data(),ids.size(),
                decoded.data(),decoded.size(),&bytes)==AXIOM_OK;
        const std::string text(decoded.data(),bytes);
        const bool pass=ok && compacted && decoded_ok &&
                (text.find("No tools are available for this response.")!=std::string::npos)==!enabled &&
                text.find("<tool_call>")==std::string::npos &&
                text.find("ARCHIVE_RESULT_613")!=std::string::npos;
        ++checks;if(!pass)++failures;
        std::cout<<name<<" status="<<(pass?"PASS":"FAIL")<<" error="<<error<<std::endl;
    };
    policy_test("archive_empty_catalog",nullptr,false);
    policy_test("archive_effective_discovered_catalog",&authority,true);
    authority.set("tool_choice",ajson::jstr("none"));
    policy_test("archive_disabled_catalog",&authority,false);
    axiom_tokenizer_close(s.tokenizer);s.tokenizer=nullptr;
    std::cout<<"COMPACTION_REGRESSION checks="<<checks<<" failures="<<failures<<std::endl;
    return failures?1:0;
}
