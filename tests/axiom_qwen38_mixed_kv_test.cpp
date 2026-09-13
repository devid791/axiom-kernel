#include "axiom/qwen38_mixed_kv.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
int main() {
    unsigned checks=0;
    for (unsigned slots : {1u,2u,8u,256u}) {
        for(unsigned current : {0u,1u,7u,255u,256u,269u,511u,512u,1023u,4095u}) {
            std::vector<unsigned> ids(slots,~0u);
            unsigned begin=current+1>slots?current+1-slots:0;
            for(unsigned p=begin;p<=current;++p) ids[p%slots]=p;
            assert(axiom_qwen38::resident_suffix_begin(current,slots,ids.data())==begin);++checks;
            if(current>begin){ids[(current-1)%slots]=~0u;
                assert(axiom_qwen38::resident_suffix_begin(current,slots,ids.data())==current);++checks;}
            ids[current%slots]=~0u;
            assert(axiom_qwen38::resident_suffix_begin(current,slots,ids.data())==current+1);++checks;
        }
    }
    assert(axiom_qwen38::resident_suffix_begin(42,0,nullptr)==43);++checks;
    unsigned random=0x6e624eb7;
    for(unsigned trial=0;trial<10000;++trial){
        auto next=[&](){random^=random<<13;random^=random>>17;random^=random<<5;return random;};
        unsigned slots=next()%256+1,current=next()%4096;
        std::vector<unsigned> ids(slots,~0u);
        unsigned earliest=current+1>slots?current+1-slots:0;
        for(unsigned p=earliest;p<=current;++p)ids[p%slots]=next()%5==0?~0u:p;
        auto begin=axiom_qwen38::resident_suffix_begin(current,slots,ids.data());
        assert(begin>=earliest&&begin<=current+1);
        for(unsigned p=begin;p<=current;++p)assert(ids[p%slots]==p);
        if(begin>earliest)assert(ids[(begin-1)%slots]!=begin-1);
        ++checks;
    }
    std::printf("mixed KV ring plan: %u checks PASS\n",checks);
}
