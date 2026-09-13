#include "axiom/qwen38_request_progress.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main() {
    using progress = axiom::qwen38::request_progress;
    using namespace std::chrono_literals;
    const auto zero = progress::clock::time_point{};
    unsigned checks = 0;
    auto check = [&](bool ok) { ++checks; if (!ok) { std::fprintf(stderr,"FAIL check=%u\n",checks); std::exit(1); } };
    progress p(zero);
    check(p.read(zero).stage == progress::phase::session_setup);
    check(!p.decode(1,zero));
    check(!p.prefill(20,10,30,zero));
    check(!p.prefill(10,40,30,zero));
    check(p.prefill(55503,55503,68575,zero+1s));
    auto a=p.read(zero+10s);
    check(a.cached_tokens==55503 && a.prompt_tokens==68575 && a.processed_tokens==0);
    check(a.revision==1 && a.seconds_since_advance==9);
    check(p.prefill(55503,55503,68575,zero+11s));
    check(p.read(zero+12s).revision==1 && p.read(zero+12s).seconds_since_advance==11);
    check(p.prefill(55503,65536,68575,zero+13s));
    check(p.read(zero+14s).processed_tokens==10033);
    check(!p.prefill(55503,65535,68575,zero+14s));
    check(!p.prefill(55504,65536,68575,zero+14s));
    check(!p.prefill(55503,65536,68576,zero+14s));
    check(!p.decode(1,zero+14s));
    check(p.prefill(55503,68575,68575,zero+15s));
    check(p.decode(0,zero+16s));
    check(p.decode(7,zero+17s));
    check(!p.decode(6,zero+18s));
    check(!p.prefill(55503,68575,68575,zero+18s));
    a=p.read(zero+20s);
    check(a.stage==progress::phase::decode && a.generated_tokens==7 && a.processed_tokens==13072);
    check(p.decode(7,zero+21s));
    check(p.read(zero+25s).seconds_since_advance==8);
    check(p.read(zero+25s).elapsed_seconds==25);
    check(p.next_pass(zero+26s));
    check(p.read(zero+27s).pass==2 && p.read(zero+27s).generated_tokens==7);
    check(!p.next_pass(zero+27s));
    check(p.prefill(68582,68582,68590,zero+28s));
    check(p.prefill(68582,68590,68590,zero+29s));
    check(p.decode(0,zero+30s));
    check(p.decode(5,zero+31s));
    check(p.read(zero+32s).generated_tokens==12 && p.read(zero+32s).processed_tokens==8);
    check(p.read(zero+30s).seconds_since_advance==0);
    check(p.read(zero-1s).elapsed_seconds==0);
    progress cached(zero);
    check(cached.prefill(100,100,100,zero));
    check(cached.decode(0,zero));
    progress fresh(zero);
    check(fresh.read(zero).revision==0 && fresh.read(zero).generated_tokens==0);
    progress concurrent;
    std::atomic<bool> done{false}, coherent{true};
    std::thread writer([&] {
        for(uint32_t n=0;n<=10000;++n) if(!concurrent.prefill(17,17+n,10017)) coherent=false;
        concurrent.decode(20); done=true;
    });
    do {
        const auto s=concurrent.read();
        if(s.stage!=progress::phase::session_setup &&
           (s.cached_tokens!=17 || s.prompt_tokens!=10017 || s.processed_tokens>10000 ||
            (s.stage==progress::phase::decode && s.processed_tokens!=10000))) coherent=false;
    } while(!done);
    writer.join();check(coherent);check(concurrent.read().generated_tokens==20);
    std::printf("REQUEST_PROGRESS PASS checks=%u\n",checks);
}
