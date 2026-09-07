#ifndef AXIOM_QWEN38_TUNE_CANDIDATES_H
#define AXIOM_QWEN38_TUNE_CANDIDATES_H

#include <cublasLt.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Append only: the caller owns capacity slots and supplies the populated count.
// Existing candidates (including baseline 0) and caller-side filters are untouched.
// AlgoCheck cannot validate actual buffer alignment; retain execution/error checks.
static inline void append_narrow_algorithms(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t a_layout,
    cublasLtMatrixLayout_t b_layout,
    cublasLtMatrixLayout_t c_layout,
    cublasLtMatrixLayout_t d_layout,
    size_t workspace_limit,
    cublasLtMatmulHeuristicResult_t* candidates,
    int capacity,
    int* returned) {
    const char* enabled = getenv("AXIOM_QWEN38_AUTOTUNE_NARROW_TILES");
    if (enabled == NULL || strcmp(enabled, "1") != 0) return;
    if (candidates == NULL || returned == NULL || capacity <= 0 ||
        *returned <= 0 || *returned >= capacity) return;
    if (handle == NULL || operation_desc == NULL || a_layout == NULL ||
        b_layout == NULL || c_layout == NULL || d_layout == NULL) return;

    const uint32_t tiles[] = {
        CUBLASLT_MATMUL_TILE_64x8,
        CUBLASLT_MATMUL_TILE_64x16,
        CUBLASLT_MATMUL_TILE_64x32,
        CUBLASLT_MATMUL_TILE_128x16,
        CUBLASLT_MATMUL_TILE_128x32,
        CUBLASLT_MATMUL_TILE_128x64,
        CUBLASLT_MATMUL_TILE_256x32,
        CUBLASLT_MATMUL_TILE_256x64,
    };
    // Snapshot the seed count: appended algorithms never become new seeds.
    const char* wide_seeds = getenv("AXIOM_QWEN38_AUTOTUNE_NARROW_SEEDS");
    const int seed_limit = wide_seeds && strcmp(wide_seeds, "4") == 0 ? 4 : 2;
    const int seed_count = *returned < seed_limit ? *returned : seed_limit;
    for (int seed = 0; seed < seed_count && *returned < capacity; ++seed) {
        if (candidates[seed].state != CUBLAS_STATUS_SUCCESS) continue;
        for (size_t tile = 0; tile < sizeof(tiles) / sizeof(tiles[0]) &&
                              *returned < capacity; ++tile) {
            cublasLtMatmulAlgo_t algo = candidates[seed].algo;
            // Only change the tile. AlgoCheck rejects unsupported combinations;
            // no capability-driven enumeration or other config/math changes.
            if (cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                    &tiles[tile], sizeof(tiles[tile])) != CUBLAS_STATUS_SUCCESS) {
                continue;
            }
            bool duplicate = false;
            for (int i = 0; i < *returned; ++i) {
                if (memcmp(&algo, &candidates[i].algo, sizeof(algo)) == 0) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;

            cublasLtMatmulHeuristicResult_t checked = {};
            checked.state = CUBLAS_STATUS_NOT_SUPPORTED;
            if (cublasLtMatmulAlgoCheck(handle, operation_desc,
                    a_layout, b_layout, c_layout, d_layout, &algo, &checked) !=
                    CUBLAS_STATUS_SUCCESS ||
                checked.state != CUBLAS_STATUS_SUCCESS ||
                checked.workspaceSize > workspace_limit) {
                continue;
            }
            // AlgoCheck explicitly does not populate result.algo.
            checked.algo = algo;
            candidates[*returned] = checked;
            ++*returned;
        }
    }
}

// Caller gates this experiment by opt-in and projection shape. Only swizzling
// changes; the completed original pool is a fixed, non-recursive seed snapshot.
static inline void append_cta_swizzle_algorithms(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t a_layout,
    cublasLtMatrixLayout_t b_layout,
    cublasLtMatrixLayout_t c_layout,
    cublasLtMatrixLayout_t d_layout,
    size_t workspace_limit,
    cublasLtMatmulHeuristicResult_t* candidates,
    int capacity,
    int* returned) {
    if (candidates == NULL || returned == NULL || capacity <= 0 ||
        *returned <= 0 || *returned >= capacity) return;
    if (handle == NULL || operation_desc == NULL || a_layout == NULL ||
        b_layout == NULL || c_layout == NULL || d_layout == NULL) return;

    const int seed_count = *returned;
    for (int seed = 0; seed < seed_count && *returned < capacity; ++seed) {
        if (candidates[seed].state != CUBLAS_STATUS_SUCCESS) continue;
        // CUDA 13 cublasLt.h: capability and config are uint32_t; capability
        // 1 supports swizzle 1, 0 does not, and other values are reserved.
        uint32_t support = 0;
        size_t written = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(
                &candidates[seed].algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
                &support, sizeof(support), &written) != CUBLAS_STATUS_SUCCESS ||
            written != sizeof(support) || support > 1u) continue;
        for (uint32_t swizzle = 0; swizzle <= support && *returned < capacity;
             ++swizzle) {
            cublasLtMatmulAlgo_t algo = candidates[seed].algo;
            if (cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                    &swizzle, sizeof(swizzle)) != CUBLAS_STATUS_SUCCESS) continue;
            bool duplicate = false;
            for (int i = 0; i < *returned; ++i) {
                if (memcmp(&algo, &candidates[i].algo, sizeof(algo)) == 0) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;

            cublasLtMatmulHeuristicResult_t checked = {};
            checked.state = CUBLAS_STATUS_NOT_SUPPORTED;
            if (cublasLtMatmulAlgoCheck(handle, operation_desc,
                    a_layout, b_layout, c_layout, d_layout, &algo, &checked) !=
                    CUBLAS_STATUS_SUCCESS ||
                checked.state != CUBLAS_STATUS_SUCCESS ||
                checked.workspaceSize > workspace_limit) continue;
            // AlgoCheck explicitly does not populate result.algo.
            checked.algo = algo;
            candidates[*returned] = checked;
            ++*returned;
        }
    }
}

// Only the immutable primary heuristic prefix may seed this experiment.
static inline void append_stages_algorithms(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t a_layout,
    cublasLtMatrixLayout_t b_layout,
    cublasLtMatrixLayout_t c_layout,
    cublasLtMatrixLayout_t d_layout,
    size_t workspace_limit,
    cublasLtMatmulHeuristicResult_t* candidates,
    int primary_count,
    int capacity,
    int* returned) {
    constexpr int kMaxPrimarySeeds = 32;
    constexpr size_t kMaxStageEntries = 64;
    constexpr int kMaxAppended = 32;
    if (candidates == NULL || returned == NULL || primary_count <= 0 ||
        primary_count > kMaxPrimarySeeds || primary_count > *returned ||
        *returned >= capacity) return;
    if (handle == NULL || operation_desc == NULL || a_layout == NULL ||
        b_layout == NULL || c_layout == NULL || d_layout == NULL) return;

    uint32_t stages[kMaxPrimarySeeds][kMaxStageEntries] = {};
    size_t stage_counts[kMaxPrimarySeeds] = {};
    for (int seed = 0; seed < primary_count; ++seed) {
        if (candidates[seed].state != CUBLAS_STATUS_SUCCESS) continue;
        // cublasLt.h: CAP_STAGES_IDS is uint32_t[], CONFIG_STAGES_ID uint32_t.
        // Query bytes first; reject oversized/malformed lists before fetching.
        size_t bytes = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(
                &candidates[seed].algo, CUBLASLT_ALGO_CAP_STAGES_IDS,
                NULL, 0, &bytes) != CUBLAS_STATUS_SUCCESS || bytes == 0 ||
            bytes > sizeof(stages[seed]) || bytes % sizeof(uint32_t) != 0) continue;
        size_t written = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(
                &candidates[seed].algo, CUBLASLT_ALGO_CAP_STAGES_IDS,
                stages[seed], bytes, &written) != CUBLAS_STATUS_SUCCESS ||
            written != bytes) continue;
        stage_counts[seed] = written / sizeof(uint32_t);
    }

    const int original_count = *returned;
    // Stage-entry-major traversal gives every primary seed a turn per entry.
    for (size_t stage_index = 0; stage_index < kMaxStageEntries &&
            *returned < capacity && *returned - original_count < kMaxAppended;
         ++stage_index) {
        for (int seed = 0; seed < primary_count && *returned < capacity &&
                *returned - original_count < kMaxAppended; ++seed) {
            if (stage_index >= stage_counts[seed]) continue;
            cublasLtMatmulAlgo_t algo = candidates[seed].algo;
            const uint32_t stage = stages[seed][stage_index];
            // Change ONLY stages; narrow/appended candidates never become seeds.
            if (cublasLtMatmulAlgoConfigSetAttribute(
                    &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                    &stage, sizeof(stage)) != CUBLAS_STATUS_SUCCESS) continue;
            bool duplicate = false;
            for (int i = 0; i < *returned; ++i) {
                if (memcmp(&algo, &candidates[i].algo, sizeof(algo)) == 0) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            cublasLtMatmulHeuristicResult_t checked = {};
            checked.state = CUBLAS_STATUS_NOT_SUPPORTED;
            if (cublasLtMatmulAlgoCheck(handle, operation_desc,
                    a_layout, b_layout, c_layout, d_layout, &algo, &checked) !=
                    CUBLAS_STATUS_SUCCESS ||
                checked.state != CUBLAS_STATUS_SUCCESS ||
                checked.workspaceSize > workspace_limit) continue;
            // AlgoCheck explicitly does not populate result.algo.
            checked.algo = algo;
            candidates[*returned] = checked;
            fprintf(stderr,
                    "axiom-qwen38-nvfp4: gateup-stages appended_index=%d "
                    "seed=%d stage_index=%zu stages_id=%u\n",
                    *returned, seed, stage_index, (unsigned)stage);
            ++*returned;
        }
    }
}

// Caller isolates this opt-in to fused gate/up. Never seed from appended entries.
static inline void append_cap_tile_algorithms(
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t operation_desc,
    cublasLtMatrixLayout_t a_layout,
    cublasLtMatrixLayout_t b_layout,
    cublasLtMatrixLayout_t c_layout,
    cublasLtMatrixLayout_t d_layout,
    size_t workspace_limit,
    cublasLtMatmulHeuristicResult_t* candidates,
    int primary_count,
    int capacity,
    int* returned) {
    constexpr int kMaxPrimarySeeds = 32;
    constexpr size_t kMaxTileEntries = CUBLASLT_MATMUL_TILE_END;
    constexpr int kMaxAppended = 32;
    if (candidates == NULL || returned == NULL || primary_count <= 0 ||
        primary_count > kMaxPrimarySeeds || primary_count > *returned ||
        *returned >= capacity) return;
    if (handle == NULL || operation_desc == NULL || a_layout == NULL ||
        b_layout == NULL || c_layout == NULL || d_layout == NULL) return;

    uint32_t tiles[kMaxPrimarySeeds][kMaxTileEntries] = {};
    size_t counts[kMaxPrimarySeeds] = {};
    size_t cursors[kMaxPrimarySeeds] = {};
    for (int seed = 0; seed < primary_count; ++seed) {
        if (candidates[seed].state != CUBLAS_STATUS_SUCCESS) continue;
        // CAP_TILE_IDS is uint32_t[]. Query bytes, reject rather than truncate.
        size_t bytes = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(
                &candidates[seed].algo, CUBLASLT_ALGO_CAP_TILE_IDS,
                NULL, 0, &bytes) != CUBLAS_STATUS_SUCCESS || bytes == 0 ||
            bytes > sizeof(tiles[seed]) || bytes % sizeof(uint32_t) != 0) {
            fprintf(stderr, "axiom-qwen38-nvfp4: gateup-cap-tiles seed=%d "
                    "cap_query_rejected bytes=%zu\n", seed, bytes);
            continue;
        }
        size_t written = 0;
        if (cublasLtMatmulAlgoCapGetAttribute(
                &candidates[seed].algo, CUBLASLT_ALGO_CAP_TILE_IDS,
                tiles[seed], bytes, &written) != CUBLAS_STATUS_SUCCESS ||
            written != bytes) {
            fprintf(stderr, "axiom-qwen38-nvfp4: gateup-cap-tiles seed=%d "
                    "cap_fetch_rejected bytes=%zu written=%zu\n", seed, bytes, written);
            continue;
        }
        counts[seed] = written / sizeof(uint32_t);
        // Enum order is not a performance order. Put the three known omissions
        // first when advertised, then use deterministic ascending ID order.
        const auto tile_rank = [](uint32_t tile) -> uint32_t {
            if (tile == CUBLASLT_MATMUL_TILE_128x8) return 0u;
            if (tile == CUBLASLT_MATMUL_TILE_256x8) return 1u;
            if (tile == CUBLASLT_MATMUL_TILE_256x16) return 2u;
            return 3u;
        };
        for (size_t i = 1; i < counts[seed]; ++i) {
            const uint32_t tile = tiles[seed][i];
            size_t j = i;
            while (j > 0) {
                const uint32_t previous = tiles[seed][j - 1];
                if (tile_rank(previous) < tile_rank(tile) ||
                    (tile_rank(previous) == tile_rank(tile) && previous <= tile)) break;
                tiles[seed][j] = previous;
                --j;
            }
            tiles[seed][j] = tile;
        }
    }

    const int original_count = *returned;
    bool pending = true;
    while (pending && *returned < capacity &&
           *returned - original_count < kMaxAppended) {
        pending = false;
        for (int seed = 0; seed < primary_count && *returned < capacity &&
                *returned - original_count < kMaxAppended; ++seed) {
            // One accepted variant per seed per round; rejects consume no quota.
            while (cursors[seed] < counts[seed]) {
                const size_t tile_index = cursors[seed]++;
                const uint32_t tile = tiles[seed][tile_index];
                if (tile <= CUBLASLT_MATMUL_TILE_UNDEFINED ||
                    tile >= CUBLASLT_MATMUL_TILE_END) continue;
                switch (tile) {
                    case CUBLASLT_MATMUL_TILE_64x8:
                    case CUBLASLT_MATMUL_TILE_64x16:
                    case CUBLASLT_MATMUL_TILE_64x32:
                    case CUBLASLT_MATMUL_TILE_128x16:
                    case CUBLASLT_MATMUL_TILE_128x32:
                    case CUBLASLT_MATMUL_TILE_128x64:
                    case CUBLASLT_MATMUL_TILE_256x32:
                    case CUBLASLT_MATMUL_TILE_256x64:
                        continue;
                    default: break;
                }
                bool repeated_tile = false;
                for (size_t i = 0; i < tile_index; ++i) {
                    if (tiles[seed][i] == tile) { repeated_tile = true; break; }
                }
                if (repeated_tile) continue;
                cublasLtMatmulAlgo_t algo = candidates[seed].algo;
                // Change ONLY tile; all math and other configuration stays intact.
                if (cublasLtMatmulAlgoConfigSetAttribute(
                        &algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                        &tile, sizeof(tile)) != CUBLAS_STATUS_SUCCESS) continue;
                bool duplicate = false;
                for (int i = 0; i < *returned; ++i) {
                    if (memcmp(&algo, &candidates[i].algo, sizeof(algo)) == 0) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) continue;
                cublasLtMatmulHeuristicResult_t checked = {};
                checked.state = CUBLAS_STATUS_NOT_SUPPORTED;
                if (cublasLtMatmulAlgoCheck(handle, operation_desc,
                        a_layout, b_layout, c_layout, d_layout, &algo, &checked) !=
                        CUBLAS_STATUS_SUCCESS ||
                    checked.state != CUBLAS_STATUS_SUCCESS ||
                    checked.workspaceSize > workspace_limit) continue;
                // AlgoCheck does not populate result.algo.
                checked.algo = algo;
                candidates[*returned] = checked;
                fprintf(stderr, "axiom-qwen38-nvfp4: gateup-cap-tiles "
                        "appended_index=%d seed=%d tile_index=%zu tile_id=%u\n",
                        *returned, seed, tile_index, (unsigned)tile);
                ++*returned;
                break;
            }
            if (cursors[seed] < counts[seed]) pending = true;
        }
    }
}

#endif  // AXIOM_QWEN38_TUNE_CANDIDATES_H
