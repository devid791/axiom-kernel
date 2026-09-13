#define main axiom_api_main
#include "../tools/axiom_qwen38_api.cpp"
#undef main
#include <cassert>
#include <fstream>
#include <iostream>

namespace {
ajson json(const std::string &s) { ajson j; std::string e; assert(ajson_parse(s,j,e)); return j; }
ajson message(const char *role,const std::string &text) {
    ajson j=ajson::jobj();j.set("role",ajson::jstr(role));j.set("content",ajson::jstr(text));return j;
}
ajson call(const std::string &path,const std::string &id) {
    ajson j=ajson::jobj(),a=ajson::jobj();a.set("path",ajson::jstr(path));
    j.set("type",ajson::jstr("function_call"));j.set("name",ajson::jstr("read_file"));
    j.set("arguments",ajson::jstr(ajson_dumps(a)));j.set("call_id",ajson::jstr(id));return j;
}
ajson result(const std::string &text,const std::string &id) {
    ajson j=ajson::jobj();j.set("type",ajson::jstr("function_call_output"));
    j.set("output",ajson::jstr(text));j.set("call_id",ajson::jstr(id));return j;
}
std::string native_call(const std::string &path) {
    return "<tool_call>\n<function=read_file>\n<parameter=path>\n"+path+
        "\n</parameter>\n</function>\n</tool_call>";
}
struct request_tokens { std::vector<uint32_t> ids; ajson tools; };
request_tokens encode(server_state &s,ajson request) {
    ajson payload,prompt;std::string e;axiom_codex::tool_wire_map wire;
    configure_codex_schema_bridge(wire);assert(wire.prepare(request,&e));
    assert(normalize_codex_responses_request(request,&payload,&e,&prompt));
    request_tokens r;std::vector<uint32_t> suffix;uint32_t original=0;bool compacted=false;
    assert(make_chat_ids(&s,prompt,false,&r.ids,&suffix,&e,&original,&compacted));
    assert(!compacted);assert(normalize_tools(payload.get("tools"),false,&r.tools,&e));return r;
}
axiom::qwen38::qwen38_session_manifest saved(server_state &s,
        const std::vector<uint32_t> &prefix,const std::string &generated,bool closed=false) {
    axiom::qwen38::qwen38_session_manifest m;m.token_ids=prefix;
    assert(append_text_ids(&s,generated,&m.token_ids)==AXIOM_OK);
    if(closed)m.token_ids.push_back(s.tokenizer_info.im_end_token_id);
    m.committed_tokens=m.token_ids.size();return m;
}
}
int main(int argc,char **argv) {
    if(argc!=2&&argc!=4)return 2;
    server_state s;s.max_context=1048576;s.default_context=262144;s.no_think=true;
    axiom_tokenizer_config cfg{};cfg.abi_version=AXIOM_ABI_VERSION;cfg.path=argv[1];
    cfg.format=AXIOM_TOKENIZER_FORMAT_HF_JSON;cfg.name="cpu-replay-qualification";
    assert(axiom_tokenizer_open(&s.tokenizer,&cfg)==AXIOM_OK);
    s.tokenizer_info.abi_version=AXIOM_ABI_VERSION;assert(axiom_tokenizer_info_get(s.tokenizer,&s.tokenizer_info)==AXIOM_OK);
    ajson base=json(R"({"model":"qwen3.8-27b-nvfp4","instructions":"Keep user instructions and exact tool results.","reasoning":{"effort":"ultra-fast"},"tools":[{"type":"function","name":"read_file","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}],"input":[{"role":"user","content":"Read the named files."}]})");
    const auto first=encode(s,base);
    size_t passed=0;
    for(const bool closed:{false,true})for(const int count:{1,2,3})for(const bool preface:{false,true}) {
        auto next=base,input=*base.get("input");
        std::string raw;
        if(preface){raw="I will read them.\n\n";input.push(message("assistant","I will read them."));}
        for(int i=0;i<count;++i){const std::string path="file "+std::to_string(i)+" è \\\".txt";
            if(i)raw+="\n";raw+=native_call(path);input.push(call(path,"c"+std::to_string(i)));}
        for(int i=0;i<count;++i)input.push(result("original result "+std::to_string(i),"c"+std::to_string(i)));
        next.set("input",input);const auto request=encode(s,next);const auto m=saved(s,first.ids,raw,closed);
        std::vector<uint32_t> effective;bool projected=false;
        assert(session_equivalent_prompt(&s,m,request.ids,262144,&effective,&request.tools,&projected));
        assert(std::equal(m.token_ids.begin(),m.token_ids.end(),effective.begin()));
        if(!closed)assert(effective[m.committed_tokens]==s.tokenizer_info.im_end_token_id);
        if(count>1||preface){assert(projected);assert(!session_equivalent_prompt(&s,m,request.ids,262144,&effective));}
        assert(session_equivalent_prompt(&s,m,request.ids,262144,&effective,&request.tools));
        const auto limit=effective.size();
        assert(!session_equivalent_prompt(&s,m,request.ids,limit-1,&effective,&request.tools));
        // All authoritative edits fail, including changed schema/text/roles/calls.
        for(int edit=0;edit<5;++edit){auto changed=next,items=input;
            if(edit==0)changed.set("instructions",ajson::jstr("Changed instructions"));
            if(edit==1){items.arr[0]=message("user","Changed user history");changed.set("input",items);}
            if(edit==2){auto tools=*changed.get("tools");tools.arr[0].set("description",ajson::jstr("Changed schema documentation"));changed.set("tools",tools);}
            if(edit==3){items.arr[preface?2:1]=call("changed argument","c0");changed.set("input",items);}
            if(edit==4){items.arr[0]=message("system","Read the named files.");changed.set("input",items);}
            const auto altered=encode(s,changed);assert(!session_equivalent_prompt(&s,m,altered.ids,262144,&effective,&altered.tools));++passed;
        }
        if(count>1){auto reversed=next,items=input;std::swap(items.arr[preface?2:1],items.arr[preface?3:2]);reversed.set("input",items);const auto altered=encode(s,reversed);assert(!session_equivalent_prompt(&s,m,altered.ids,262144,&effective,&altered.tools));++passed;}
        // Persist another reply, then resume AGAIN: old native parallel calls
        // remain in the native history and must still match Core's projection.
        assert(session_equivalent_prompt(&s,m,request.ids,262144,&effective,&request.tools));
        const auto later=saved(s,effective,"Finished.",true);
        input.push(message("assistant","Finished."));input.push(message("user","Continue."));next.set("input",input);
        const auto resumed=encode(s,next);assert(session_equivalent_prompt(&s,later,resumed.ids,262144,&effective,&resumed.tools));
        const size_t result_index=1+(preface?1:0)+count;
        input.arr[result_index]=result("CHANGED PAST RESULT","c0");next.set("input",input);
        const auto changed_result=encode(s,next);assert(!session_equivalent_prompt(&s,later,changed_result.ids,262144,&effective,&changed_result.tools));
        passed+=4;
    }
    // Noncanonical OUTBOUND whitespace is projected exactly as the adapter
    // emitted it; arbitrary new INBOUND whitespace in arguments remains an edit.
    auto one=base,items=*base.get("input");items.push(call("file","c0"));items.push(result("done","c0"));one.set("input",items);
    const auto r=encode(s,one);const auto spaced=saved(s,first.ids,"\n"+native_call("  file  ")+"\n");std::vector<uint32_t> effective;
    assert(session_equivalent_prompt(&s,spaced,r.ids,262144,&effective,&r.tools));
    items.arr[1]=call(" file","c0");one.set("input",items);const auto edit=encode(s,one);
    assert(!session_equivalent_prompt(&s,spaced,edit.ids,262144,&effective,&edit.tools));
    const auto malformed=saved(s,first.ids,"<tool_call>broken");
    assert(!session_equivalent_prompt(&s,malformed,r.ids,262144,&effective,&r.tools));passed+=3;
    uint32_t nul_id=AXIOM_TOKEN_ID_INVALID;
    for(uint32_t id=0;id<s.tokenizer_info.vocab_size;++id){char bytes[65536];uint32_t n=0;
        if(axiom_tokenizer_decode_token(s.tokenizer,id,bytes,sizeof(bytes),&n)==AXIOM_OK&&n==1&&bytes[0]=='\0'){nul_id=id;break;}}
    assert(nul_id!=AXIOM_TOKEN_ID_INVALID);
    auto nul=saved(s,first.ids,"<tool_call>\n<function=read_file>\n<parameter=path>\nfile");
    nul.token_ids.push_back(nul_id);assert(append_text_ids(&s,"hidden\n</parameter>\n</function>\n</tool_call>",&nul.token_ids)==AXIOM_OK);nul.committed_tokens=nul.token_ids.size();
    std::vector<uint32_t> projected;
    assert(!project_codex_tool_prefix(&s,nul,r.tools,&projected));++passed;
    if(argc==4){
        std::ifstream f(argv[2]);const auto actual=encode(s,json(std::string((std::istreambuf_iterator<char>(f)),{})));
        std::ifstream mfile(argv[3],std::ios::binary);uint32_t count=0;mfile.seekg(24);mfile.read(reinterpret_cast<char*>(&count),4);assert(mfile&&count<=1048576);
        axiom::qwen38::qwen38_session_manifest m;m.committed_tokens=count;m.token_ids.resize(count);mfile.seekg(4096);mfile.read(reinterpret_cast<char*>(m.token_ids.data()),count*4);assert(mfile);
        bool projected=false;assert(!session_equivalent_prompt(&s,m,actual.ids,262144,&effective));
        assert(session_equivalent_prompt(&s,m,actual.ids,262144,&effective,&actual.tools,&projected)&&projected);
        assert(std::equal(m.token_ids.begin(),m.token_ids.end(),effective.begin()));
        std::cout<<"ACTUAL_PRIVATE_FIXTURE PASS native_prefix="<<count<<" suffix="<<effective.size()-count<<" incoming="<<actual.ids.size()<<'\n';
    }
    std::cout<<"CODEX_PREFIX_REPLAY PASS checks="<<passed<<" parallel=1,2,3 preface=both cursor=open,closed repeated_resume=pass edits=reject legacy=unchanged\n";
    axiom_tokenizer_close(s.tokenizer);s.tokenizer=nullptr;
}
