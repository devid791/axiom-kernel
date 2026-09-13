#include "axiom/qwen38_prefix_reuse.hpp"
#include <cassert>
#include <iostream>
#include <map>
int main() {
    const std::map<uint32_t,std::string> pieces{{1,"Soft"},{2,"ware"},{3,"Software"},
        {4,"!"},{5,"?"},{6,"Software!"},{7,"<end>"},{8,"\xC3"},{9,"\xA8"},
        {10,"\xC3\xA8"},{11,""},{12,std::string("a\0b",3)},{13,"a"},{14,std::string("\0b",2)}};
    auto decode=[&](uint32_t id,std::string *out){auto p=pieces.find(id);if(p==pieces.end())return false;*out=p->second;return true;};
    size_t n=999;
    auto match=[&](std::vector<uint32_t>a,std::vector<uint32_t>b,size_t budget=1000){return axiom::qwen38::equivalent_prefix(a,a.size(),b,100,decode,&n,budget);};
    assert(match({100,1,2,101},{100,3,101,4})&&n==3);
    assert(match({100,3,101},{100,1,2,101,4})&&n==4);
    assert(match({100,3},{100,3,4})&&n==2);
    assert(!match({100,1,2,4},{100,3,5})&&n==0); // Changed text.
    assert(!match({100,3,101},{100,3,102})&&n==0); // Changed role/control.
    assert(!match({7},{101})&&n==0); // Literal text is never a control token.
    assert(!match({101},{7})&&n==0);
    assert(!match({1,2},{6})&&n==0); // Do not split an incoming token at the boundary.
    assert(!match({3,4},{1,2})&&n==0); // Incoming prefix truncated.
    assert(!match({11},{1})&&n==0); // Empty/failed decode fails closed.
    assert(!match({99},{1})&&n==0);
    assert(!match({1,2},{3},3)&&n==0); // Work bound.
    assert(match({8,9},{10,4})&&n==1); // UTF-8 split across model tokens.
    assert(match({12},{13,14,4})&&n==2); // No C-string truncation.
    assert(!axiom::qwen38::equivalent_prefix(std::vector<uint32_t>{1},2,
        std::vector<uint32_t>{1},100,decode,&n)&&n==0);
    assert(!match({},{}));
    std::cout<<"prefix-reuse: PASS byte equality, control identity, boundary, UTF8, bounds\n";
}
