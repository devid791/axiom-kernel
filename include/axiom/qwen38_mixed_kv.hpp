#pragma once
#include <cstdint>

namespace axiom_qwen38 {
// Only verified contiguous suffix pages may use direct GPU ring addressing.
// Everything before this suffix remains on the ordinary checked page reader.
inline uint32_t resident_suffix_begin(uint32_t current, uint32_t slots,
                                      const uint32_t *page_ids) {
    if (!slots || !page_ids || page_ids[current % slots] != current)
        return current + 1u;
    uint32_t begin = current;
    while (begin && current - begin + 1u < slots &&
           page_ids[(begin - 1u) % slots] == begin - 1u) --begin;
    return begin;
}
}
