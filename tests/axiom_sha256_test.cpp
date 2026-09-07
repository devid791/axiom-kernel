#include "axiom/sha256.hpp"

#include <cstdio>
#include <string>

namespace {

bool check(const std::string &input, const char *expected) {
    const std::string actual = axiom::crypto::sha256_string_hex(input);
    if (actual == expected) return true;
    std::fprintf(stderr, "sha256 mismatch input_bytes=%zu expected=%s actual=%s\n",
                 input.size(), expected, actual.c_str());
    return false;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= check("", "e3b0c44298fc1c149afbf4c8996fb924"
                   "27ae41e4649b934ca495991b7852b855");
    ok &= check("abc", "ba7816bf8f01cfea414140de5dae2223"
                      "b00361a396177a9cb410ff61f20015ad");
    ok &= check(std::string(1000000u, 'a'),
                "cdc76e5c9914fb9281a1c7e284d73e67"
                "f1809a48a497200e046d39ccc7112cd0");
    if (!ok) return 1;
    std::puts("axiom-sha256-test: PASS");
    return 0;
}
