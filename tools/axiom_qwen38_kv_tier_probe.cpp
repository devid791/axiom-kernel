#include "axiom/qwen38_kv_tier.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int parse_u32(const char *text, uint32_t *out) {
    if (!text || !out || !text[0]) return AXIOM_ERR_INVALID_ARGUMENT;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > UINT32_MAX) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = static_cast<uint32_t>(value);
    return AXIOM_OK;
}

int round_trip(
        axiom_qwen38_kv_tier *tier,
        const uint32_t family,
        const uint32_t layer,
        const uint32_t logical_page,
        const uint64_t user_data) {
    const uint64_t bytes = axiom_qwen38_kv_tier_family_page_bytes(family);
    void *write_page = nullptr;
    void *read_page = nullptr;
    int rc = axiom_qwen38_kv_tier_host_page_alloc(bytes, &write_page);
    if (rc == AXIOM_OK) rc = axiom_qwen38_kv_tier_host_page_alloc(bytes, &read_page);
    if (rc != AXIOM_OK) {
        axiom_qwen38_kv_tier_host_page_free(write_page);
        axiom_qwen38_kv_tier_host_page_free(read_page);
        return rc;
    }
    std::memset(write_page, family == AXIOM_QWEN38_KV_TIER_TARGET ? 0x5Au : 0xA5u,
                static_cast<size_t>(bytes));
    std::memset(read_page, 0, static_cast<size_t>(bytes));

    axiom_qwen38_kv_tier_request request{};
    request.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    request.operation = AXIOM_QWEN38_KV_TIER_WRITE;
    request.family = family;
    request.layer = layer;
    request.logical_page = logical_page;
    request.host_page = write_page;
    request.host_page_bytes = bytes;
    request.user_data = user_data;
    rc = axiom_qwen38_kv_tier_submit(tier, &request);
    axiom_qwen38_kv_tier_completion completion{};
    uint32_t completed = 0u;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_kv_tier_wait(tier, 1u, &completion, 1u, &completed);
    }
    if (rc == AXIOM_OK && (completed != 1u || completion.result != AXIOM_OK)) {
        rc = AXIOM_ERR_IO;
    }

    request.operation = AXIOM_QWEN38_KV_TIER_READ;
    request.host_page = read_page;
    request.user_data = user_data + 1u;
    completed = 0u;
    if (rc == AXIOM_OK) rc = axiom_qwen38_kv_tier_submit(tier, &request);
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_kv_tier_wait(tier, 1u, &completion, 1u, &completed);
    }
    if (rc == AXIOM_OK && (completed != 1u || completion.result != AXIOM_OK)) {
        rc = AXIOM_ERR_IO;
    }
    if (rc == AXIOM_OK && std::memcmp(write_page, read_page, static_cast<size_t>(bytes)) != 0) {
        rc = AXIOM_ERR_RUNTIME;
    }
    axiom_qwen38_kv_tier_host_page_free(write_page);
    axiom_qwen38_kv_tier_host_page_free(read_page);
    return rc;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: %s PATH [MAX_CONTEXT] [HOT_PAGES]\n", argv[0]);
        return 2;
    }
    uint32_t max_context = 262144u;
    uint32_t hot_pages = 1u;
    if (argc >= 3 && parse_u32(argv[2], &max_context) != AXIOM_OK) return 2;
    if (argc >= 4 && parse_u32(argv[3], &hot_pages) != AXIOM_OK) return 2;

    axiom_qwen38_kv_tier_plan plan{};
    plan.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    int rc = axiom_qwen38_kv_tier_plan_get(max_context, hot_pages, &plan);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "plan failed: %d\n", rc);
        return 1;
    }
    axiom_qwen38_kv_tier_config config{};
    config.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    config.path = argv[1];
    config.max_context = max_context;
    config.hot_pages = hot_pages;
    config.queue_depth = 16u;
    config.flags = AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_DIRECT |
                   AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_IO_URING;

    axiom_qwen38_kv_tier *tier = nullptr;
    rc = axiom_qwen38_kv_tier_create(&config, &tier);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "create failed: %d\n", rc);
        return 1;
    }
    rc = round_trip(tier, AXIOM_QWEN38_KV_TIER_TARGET, 0u, 0u, 100u);
    if (rc == AXIOM_OK) rc = round_trip(tier, AXIOM_QWEN38_KV_TIER_DSPARK, 0u, 0u, 200u);
    if (rc == AXIOM_OK) rc = axiom_qwen38_kv_tier_commit(tier, plan.page_tokens);

    axiom_qwen38_kv_tier_info info{};
    info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    const int info_rc = axiom_qwen38_kv_tier_info_get(tier, &info);
    std::printf(
            "kv_tier rc=%d backend=%u direct=%u io_uring=%u max_context=%u "
            "file_bytes=%llu completed_bytes=%llu committed_tokens=%u\n",
            rc, info.backend, info.direct_io,
            info.backend == AXIOM_QWEN38_KV_TIER_BACKEND_IO_URING_DIRECT ? 1u : 0u,
            info.plan.max_context,
            static_cast<unsigned long long>(info.plan.file_bytes),
            static_cast<unsigned long long>(info.completed_bytes), info.committed_tokens);
    axiom_qwen38_kv_tier_destroy(tier);
    return rc == AXIOM_OK && info_rc == AXIOM_OK ? 0 : 1;
}
