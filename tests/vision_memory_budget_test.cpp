#include "axiom/vision_memory_budget.hpp"
#include <cassert>
#include <cstdio>
#include <cstdint>
int main() {
    using axiom::vision_memory::fits;
    constexpr uint64_t reserve = 128ull * 1024ull * 1024ull;
    assert(!fits(1, 0));
    assert(!fits(0, reserve));
    assert(fits(1, reserve + 1));
    assert(!fits(2, reserve + 1));
    assert(fits(UINT64_MAX - reserve, UINT64_MAX));
    assert(!fits(UINT64_MAX - reserve + 1, UINT64_MAX));
    assert(!axiom::vision_memory::host_allocation_fits(UINT64_MAX));
    std::puts("VISION_MEMORY_BUDGET_TEST PASS boundaries=7");
}
