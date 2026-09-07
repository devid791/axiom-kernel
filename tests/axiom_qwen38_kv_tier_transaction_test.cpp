#include "axiom/qwen38_kv_tier.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>

namespace {

bool expect(const bool condition, const char *message) {
    if (condition) return true;
    std::fprintf(stderr, "qwen38-kv-transaction-test: %s\n", message);
    return false;
}

int transfer_page(
        axiom_qwen38_kv_tier *tier,
        const uint32_t operation,
        const uint32_t family,
        const uint32_t layer,
        const uint32_t logical_page,
        void *page) {
    axiom_qwen38_kv_tier_request request{};
    request.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    request.operation = operation;
    request.family = family;
    request.layer = layer;
    request.logical_page = logical_page;
    request.host_page = page;
    request.host_page_bytes = axiom_qwen38_kv_tier_family_page_bytes(family);
    int rc = axiom_qwen38_kv_tier_submit(tier, &request);
    axiom_qwen38_kv_tier_completion completion{};
    uint32_t count = 0u;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_kv_tier_wait(tier, 1u, &completion, 1u, &count);
    }
    if (rc == AXIOM_OK &&
        (count != 1u || completion.result != AXIOM_OK ||
         completion.transferred_bytes != request.host_page_bytes)) {
        rc = AXIOM_ERR_IO;
    }
    return rc;
}

int write_pattern(
        axiom_qwen38_kv_tier *tier,
        const uint32_t family,
        const uint8_t pattern) {
    const uint64_t bytes = axiom_qwen38_kv_tier_family_page_bytes(family);
    void *page = nullptr;
    int rc = axiom_qwen38_kv_tier_host_page_alloc(bytes, &page);
    if (rc == AXIOM_OK) {
        std::memset(page, pattern, static_cast<size_t>(bytes));
        rc = transfer_page(
                tier, AXIOM_QWEN38_KV_TIER_WRITE, family, 0u, 0u, page);
    }
    axiom_qwen38_kv_tier_host_page_free(page);
    return rc;
}

bool page_is_pattern(
        axiom_qwen38_kv_tier *tier,
        const uint32_t family,
        const uint8_t pattern) {
    const uint64_t bytes = axiom_qwen38_kv_tier_family_page_bytes(family);
    void *page = nullptr;
    int rc = axiom_qwen38_kv_tier_host_page_alloc(bytes, &page);
    if (rc == AXIOM_OK) {
        std::memset(page, 0, static_cast<size_t>(bytes));
        rc = transfer_page(
                tier, AXIOM_QWEN38_KV_TIER_READ, family, 0u, 0u, page);
    }
    bool equal = rc == AXIOM_OK;
    const auto *bytes_view = static_cast<const uint8_t *>(page);
    for (size_t index = 0u; equal && index < bytes; ++index) {
        equal = bytes_view[index] == pattern;
    }
    axiom_qwen38_kv_tier_host_page_free(page);
    return equal;
}

int open_tier(const std::string &path, axiom_qwen38_kv_tier **out) {
    axiom_qwen38_kv_tier_config config{};
    config.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    config.path = path.c_str();
    config.max_context = 512u;
    config.hot_pages = 1u;
    config.queue_depth = 8u;
    config.flags = AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_DIRECT;
    return axiom_qwen38_kv_tier_create(&config, out);
}

bool committed_is(axiom_qwen38_kv_tier *tier, const uint32_t expected) {
    axiom_qwen38_kv_tier_info info{};
    info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    return axiom_qwen38_kv_tier_info_get(tier, &info) == AXIOM_OK &&
            info.committed_tokens == expected;
}

}  // namespace

int main() {
    char directory_template[] = "/tmp/axiom-qwen38-kv-txn-XXXXXX";
    char *directory = mkdtemp(directory_template);
    if (!expect(directory != nullptr, "mkdtemp failed")) return 1;
    const std::string tier_path = std::string(directory) + "/session.kv";
    const std::string journal_path = tier_path + ".txn";
    axiom_qwen38_kv_tier *tier = nullptr;
    bool ok = expect(open_tier(tier_path, &tier) == AXIOM_OK, "tier create failed");
    ok = ok && expect(write_pattern(tier, AXIOM_QWEN38_KV_TIER_TARGET, 0x11u) == AXIOM_OK,
                      "initial target page write failed");
    ok = ok && expect(write_pattern(tier, AXIOM_QWEN38_KV_TIER_DSPARK, 0x22u) == AXIOM_OK,
                      "initial speculative page write failed");
    ok = ok && expect(axiom_qwen38_kv_tier_commit(tier, 3u) == AXIOM_OK,
                      "initial watermark commit failed");

    ok = ok && expect(axiom_qwen38_kv_tier_transaction_begin(
                              tier, journal_path.c_str(), 3u) == AXIOM_OK,
                      "transaction begin failed");
    ok = ok && expect(write_pattern(tier, AXIOM_QWEN38_KV_TIER_TARGET, 0x33u) == AXIOM_OK &&
                              write_pattern(tier, AXIOM_QWEN38_KV_TIER_DSPARK, 0x44u) == AXIOM_OK &&
                              axiom_qwen38_kv_tier_commit(tier, 5u) == AXIOM_OK,
                      "transaction mutation failed");
    ok = ok && expect(axiom_qwen38_kv_tier_transaction_rollback(
                              tier, journal_path.c_str()) == AXIOM_OK,
                      "in-process rollback failed");
    ok = ok && expect(committed_is(tier, 3u) &&
                              page_is_pattern(tier, AXIOM_QWEN38_KV_TIER_TARGET, 0x11u) &&
                              page_is_pattern(tier, AXIOM_QWEN38_KV_TIER_DSPARK, 0x22u),
                      "rollback did not restore watermark and tail pages");

    ok = ok && expect(axiom_qwen38_kv_tier_transaction_begin(
                              tier, journal_path.c_str(), 3u) == AXIOM_OK &&
                              write_pattern(tier, AXIOM_QWEN38_KV_TIER_TARGET, 0x55u) == AXIOM_OK &&
                              axiom_qwen38_kv_tier_commit(tier, 5u) == AXIOM_OK,
                      "restart rollback fixture failed");
    axiom_qwen38_kv_tier_destroy(tier);
    tier = nullptr;
    ok = ok && expect(axiom_qwen38_kv_tier_transaction_recover_file(
                              tier_path.c_str(), journal_path.c_str(), 3u) == AXIOM_OK,
                      "restart rollback recovery failed");
    ok = ok && expect(open_tier(tier_path, &tier) == AXIOM_OK &&
                              committed_is(tier, 3u) &&
                              page_is_pattern(tier, AXIOM_QWEN38_KV_TIER_TARGET, 0x11u),
                      "restart recovery did not restore old state");

    ok = ok && expect(axiom_qwen38_kv_tier_transaction_begin(
                              tier, journal_path.c_str(), 3u) == AXIOM_OK &&
                              write_pattern(tier, AXIOM_QWEN38_KV_TIER_TARGET, 0x66u) == AXIOM_OK &&
                              axiom_qwen38_kv_tier_commit(tier, 5u) == AXIOM_OK,
                      "published transaction fixture failed");
    axiom_qwen38_kv_tier_destroy(tier);
    tier = nullptr;
    ok = ok && expect(axiom_qwen38_kv_tier_transaction_recover_file(
                              tier_path.c_str(), journal_path.c_str(), 5u) == AXIOM_OK,
                      "published transaction recovery failed");
    ok = ok && expect(open_tier(tier_path, &tier) == AXIOM_OK &&
                              committed_is(tier, 5u) &&
                              page_is_pattern(tier, AXIOM_QWEN38_KV_TIER_TARGET, 0x66u),
                      "published transaction was incorrectly rolled back");

    axiom_qwen38_kv_tier_destroy(tier);
    std::error_code cleanup_error;
    std::filesystem::remove_all(directory, cleanup_error);
    ok = ok && expect(!cleanup_error, "temporary fixture cleanup failed");
    if (ok) std::puts("qwen38-kv-transaction-test: pass");
    return ok ? 0 : 1;
}
