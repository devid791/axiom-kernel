#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace axiom::qwen38 {
// A generated BPE sequence need not be the canonical re-encoding of its text.
// Match every byte, keeping added/control tokens as indivisible ID boundaries.
// Never accept a changed byte, changed control token, or partial incoming token.
// On success the caller retains the ORIGINAL native prefix and appends incoming
// tokens after consumed. This is not a rewind or approximate history match.
template<class Decode>
bool equivalent_prefix(const std::vector<std::uint32_t>& saved, std::size_t committed,
                       const std::vector<std::uint32_t>& incoming,
                       std::uint32_t base_vocab, Decode decode,
                       std::size_t* consumed, std::size_t byte_budget = 64u << 20) {
    if (!consumed) return false;
    *consumed = 0;
    if (!base_vocab || !committed || committed > saved.size()) return false;
    std::size_t i = 0, j = 0, a = 0, b = 0, bytes = 0;
    std::string left, right;
    while (i < committed) {
        if (j >= incoming.size()) return false;
        if (left.empty() && right.empty()) {
            if (saved[i] == incoming[j]) { ++i; ++j; continue; }
            if (saved[i] >= base_vocab || incoming[j] >= base_vocab) return false;
        }
        if (left.empty()) {
            if (saved[i] >= base_vocab || !decode(saved[i], &left) || left.empty()) return false;
            a = 0;
        }
        if (right.empty()) {
            if (incoming[j] >= base_vocab || !decode(incoming[j], &right) || right.empty()) return false;
            b = 0;
        }
        const auto n = std::min(left.size() - a, right.size() - b);
        if (n > byte_budget - bytes || std::memcmp(left.data() + a, right.data() + b, n)) return false;
        bytes += n; a += n; b += n;
        if (a == left.size()) { left.clear(); ++i; }
        if (b == right.size()) { right.clear(); ++j; }
    }
    if (!right.empty()) return false;
    *consumed = j;
    return true;
}
} // namespace axiom::qwen38
