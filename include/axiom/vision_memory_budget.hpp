#ifndef AXIOM_VISION_MEMORY_BUDGET_HPP
#define AXIOM_VISION_MEMORY_BUDGET_HPP
#include <cstdint>
#include <cstdio>

namespace axiom::vision_memory {
inline bool fits(uint64_t bytes, uint64_t available) {
    constexpr uint64_t reserve = 128ull * 1024ull * 1024ull;
    return available > reserve && bytes <= available - reserve;
}
// Linux native host, checked BEFORE image/metadata tensor allocation. No fixed
// image quota; MemAvailable includes reclaimable cache, unlike free RAM alone.
inline bool host_allocation_fits(uint64_t bytes) {
    FILE *file = std::fopen("/proc/meminfo", "r");
    if (!file) return false;
    char line[256];
    unsigned long long available_kib = 0u;
    while (std::fgets(line, sizeof(line), file))
        if (std::sscanf(line, "MemAvailable: %llu kB", &available_kib) == 1) break;
    std::fclose(file);
    return available_kib <= UINT64_MAX / 1024u && fits(bytes, available_kib * 1024u);
}
}
#endif
