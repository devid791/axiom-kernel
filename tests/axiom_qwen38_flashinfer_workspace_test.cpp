#include "axiom/qwen38_flashinfer.h"

#include <cstdint>
#include <cstdio>

namespace {

constexpr uint64_t kBytesPerChunk =
        8u * 24u * (256u * sizeof(uint16_t) + sizeof(float));

struct test_case {
    uint32_t context;
    uint64_t expected;
};

}  // namespace

int main() {
    const test_case cases[] = {
        {0u, 0u},
        {7u, 0u},
        {8u, kBytesPerChunk},
        {64u, kBytesPerChunk},
        {256u, kBytesPerChunk},
        {257u, 2u * kBytesPerChunk},
        {512u, 2u * kBytesPerChunk},
        {520u, 3u * kBytesPerChunk},
        {8192u, 32u * kBytesPerChunk},
    };
    for (const test_case &value : cases) {
        const uint64_t actual =
                axiom_qwen38_flashinfer_temporal8_workspace_bytes(value.context);
        if (actual != value.expected) {
            std::fprintf(stderr,
                         "FAIL context=%u expected=%llu actual=%llu\n",
                         value.context,
                         static_cast<unsigned long long>(value.expected),
                         static_cast<unsigned long long>(actual));
            return 1;
        }
    }
    std::puts("qwen38-flashinfer-workspace-test: pass");
    return 0;
}
