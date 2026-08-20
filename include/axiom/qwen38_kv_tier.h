#ifndef AXIOM_QWEN38_KV_TIER_H
#define AXIOM_QWEN38_KV_TIER_H

/*
 * Fixed-page backing store for the single-session Qwen3.8 target and DSpark
 * KV caches.  This component owns host/NVMe state only: GPU page residency and
 * FlashInfer dispatch stay with the target and draft executors.
 *
 * A logical page always covers 256 consecutive tokens. Every page record is K
 * followed by V. The native target's E4M3 record is directly scatterable into
 * its device planes; DSpark's device cache is BF16 and converts explicitly at
 * the bridge boundary:
 *
 *   target:  E4M3 [256,4,256] K + V = 512 KiB/layer
 *   DSpark:  E4M3 [256,8,128] K + V = 512 KiB/layer
 *
 * The on-disk payload is layer-major, then logical-page-major.  All offsets,
 * buffers, and request sizes are 4096-byte aligned for O_DIRECT.  The store
 * never truncates an existing file; an incompatible header fails closed.
 */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_KV_TIER_ABI_VERSION 1u
#define AXIOM_QWEN38_KV_TIER_PAGE_TOKENS 256u
#define AXIOM_QWEN38_KV_TIER_TARGET_LAYERS 16u
#define AXIOM_QWEN38_KV_TIER_DSPARK_LAYERS 5u
#define AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES 524288u
#define AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES 524288u
#define AXIOM_QWEN38_KV_TIER_COMBINED_PAGE_BYTES 11010048u
#define AXIOM_QWEN38_KV_TIER_HEADER_BYTES 4096u

/* Payload bytes per committed token, excluding the final partial-page slack. */
#define AXIOM_QWEN38_KV_TIER_TARGET_BYTES_PER_TOKEN 32768u
#define AXIOM_QWEN38_KV_TIER_DSPARK_BYTES_PER_TOKEN 10240u
#define AXIOM_QWEN38_KV_TIER_COMBINED_BYTES_PER_TOKEN 43008u

typedef struct axiom_qwen38_kv_tier axiom_qwen38_kv_tier;

typedef enum axiom_qwen38_kv_tier_family {
    AXIOM_QWEN38_KV_TIER_TARGET = 1,
    AXIOM_QWEN38_KV_TIER_DSPARK = 2,
} axiom_qwen38_kv_tier_family;

typedef enum axiom_qwen38_kv_tier_backend {
    AXIOM_QWEN38_KV_TIER_BACKEND_SYNC_DIRECT = 1,
    AXIOM_QWEN38_KV_TIER_BACKEND_IO_URING_DIRECT = 2,
} axiom_qwen38_kv_tier_backend;

/* Allocate every filesystem block up front.  Without this flag the file is
 * sparse and grows as sealed pages are written. */
#define AXIOM_QWEN38_KV_TIER_FLAG_PREALLOCATE (1ull << 0)
/* Require io_uring.  Creation fails instead of silently switching to pread/
 * pwrite when the kernel or seccomp policy rejects io_uring_setup. */
#define AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_IO_URING (1ull << 1)
/* Require O_DIRECT.  This is the production mode and is intentionally the
 * default behavior of this ABI.  The flag is reserved for explicit audits. */
#define AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_DIRECT (1ull << 2)

typedef struct axiom_qwen38_kv_tier_plan {
    uint32_t abi_version;
    uint32_t max_context;
    uint32_t page_tokens;
    uint32_t logical_pages;
    uint32_t hot_pages;
    uint32_t hot_tokens;
    uint32_t target_layers;
    uint32_t dspark_layers;
    uint64_t target_bytes_per_token;
    uint64_t dspark_bytes_per_token;
    uint64_t combined_bytes_per_token;
    uint64_t target_page_bytes;
    uint64_t dspark_page_bytes;
    uint64_t combined_page_bytes;
    uint64_t target_payload_bytes;
    uint64_t dspark_payload_bytes;
    uint64_t payload_bytes;
    uint64_t file_bytes;
    uint64_t hot_device_bytes;
} axiom_qwen38_kv_tier_plan;

typedef struct axiom_qwen38_kv_tier_config {
    uint32_t abi_version;
    const char *path;
    uint32_t max_context;
    /* A shared logical hot-page count. Each resident page consumes exactly
     * 10.5 MiB across all 16 target and five DSpark layers. */
    uint32_t hot_pages;
    uint32_t queue_depth;
    uint64_t flags;
} axiom_qwen38_kv_tier_config;

typedef struct axiom_qwen38_kv_tier_info {
    uint32_t abi_version;
    uint32_t backend;
    uint32_t direct_io;
    uint32_t preallocated;
    uint32_t queue_depth;
    uint32_t outstanding;
    uint32_t committed_tokens;
    uint32_t reserved0;
    axiom_qwen38_kv_tier_plan plan;
    uint64_t submitted_reads;
    uint64_t submitted_writes;
    uint64_t completed_bytes;
    uint64_t io_errors;
} axiom_qwen38_kv_tier_info;

typedef enum axiom_qwen38_kv_tier_operation {
    AXIOM_QWEN38_KV_TIER_READ = 1,
    AXIOM_QWEN38_KV_TIER_WRITE = 2,
} axiom_qwen38_kv_tier_operation;

typedef struct axiom_qwen38_kv_tier_request {
    uint32_t abi_version;
    uint32_t operation;
    uint32_t family;
    uint32_t layer;
    uint32_t logical_page;
    uint32_t reserved0;
    /* One 4096-aligned host buffer of exactly family_page_bytes.  Its layout
     * is K plane followed immediately by V plane.  It must remain alive until
     * the matching completion is returned. */
    void *host_page;
    uint64_t host_page_bytes;
    uint64_t user_data;
} axiom_qwen38_kv_tier_request;

typedef struct axiom_qwen38_kv_tier_completion {
    uint32_t abi_version;
    uint32_t operation;
    int32_t result;
    uint32_t reserved0;
    uint64_t transferred_bytes;
    uint64_t user_data;
} axiom_qwen38_kv_tier_completion;

/* Pure arithmetic; performs no allocation or I/O. */
int axiom_qwen38_kv_tier_plan_get(
        uint32_t max_context,
        uint32_t hot_pages,
        axiom_qwen38_kv_tier_plan *out);

/* Opens or creates a compatible cache file.  Existing contents are validated
 * and preserved.  Creation never falls back from O_DIRECT. */
int axiom_qwen38_kv_tier_create(
        const axiom_qwen38_kv_tier_config *config,
        axiom_qwen38_kv_tier **out);
void axiom_qwen38_kv_tier_destroy(axiom_qwen38_kv_tier *tier);
int axiom_qwen38_kv_tier_info_get(
        const axiom_qwen38_kv_tier *tier,
        axiom_qwen38_kv_tier_info *out);

uint64_t axiom_qwen38_kv_tier_family_page_bytes(uint32_t family);
int axiom_qwen38_kv_tier_page_offset(
        const axiom_qwen38_kv_tier *tier,
        uint32_t family,
        uint32_t layer,
        uint32_t logical_page,
        uint64_t *out_offset);

/* Aligned bounce-page helpers.  The byte count must equal either the target
 * or DSpark page record size. */
int axiom_qwen38_kv_tier_host_page_alloc(uint64_t bytes, void **out);
void axiom_qwen38_kv_tier_host_page_free(void *page);

/* Submit fixed-page reads/writes.  io_uring mode is asynchronous; sync mode
 * records an already-completed request which is drained by wait(). */
int axiom_qwen38_kv_tier_submit(
        axiom_qwen38_kv_tier *tier,
        const axiom_qwen38_kv_tier_request *request);
int axiom_qwen38_kv_tier_wait(
        axiom_qwen38_kv_tier *tier,
        uint32_t min_complete,
        axiom_qwen38_kv_tier_completion *completions,
        uint32_t completion_capacity,
        uint32_t *out_count);

/* Persists only the logical watermark after every earlier submitted write has
 * completed.  Speculative suffix pages are never made visible by this call. */
int axiom_qwen38_kv_tier_commit(
        axiom_qwen38_kv_tier *tier,
        uint32_t committed_tokens);
/* Starts a new logical session without clearing payload blocks.  Old data is
 * unreachable because the committed watermark returns to zero. */
int axiom_qwen38_kv_tier_reset(axiom_qwen38_kv_tier *tier);

#ifdef __cplusplus
}
#endif

#endif
