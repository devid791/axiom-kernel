#define _POSIX_C_SOURCE 200809L

#include "axiom/qwen_runner.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static std::vector<uint32_t> parse_ids(const char *csv) {
    std::vector<uint32_t> ids;
    const char *p = csv;
    while (p && *p) {
        char *end = nullptr;
        unsigned long v = std::strtoul(p, &end, 10);
        if (end == p) break;
        if (v <= 0xfffffffful) ids.push_back((uint32_t)v);
        p = end;
        while (*p == ',' || *p == ' ' || *p == '\t') ++p;
    }
    return ids;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                "usage: %s <model.gguf> [--ids CSV] [--max N] [--runs N] [--warmup N] [--profile-range]\n",
                argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *ids_csv = "1";
    int max_new = 128;
    int runs = 5;
    int warmup = 1;
    bool profile_range = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ids") && i + 1 < argc) ids_csv = argv[++i];
        else if (!std::strcmp(argv[i], "--max") && i + 1 < argc) max_new = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--profile-range")) profile_range = true;
        else {
            std::fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 2;
        }
    }
    if (max_new <= 0 || runs <= 0 || warmup < 0) return 2;
    std::vector<uint32_t> ids = parse_ids(ids_csv);
    if (ids.empty()) return 2;

    axiom_qwen_resident *h = axiom_qwen_gpu_create(model);
    if (!h) {
        std::fprintf(stderr, "resident_create_failed\n");
        return 1;
    }

    const uint32_t *in_ids[1] = {ids.data()};
    int n_in[1] = {(int)ids.size()};
    std::vector<uint32_t> out((size_t)max_new);
    uint32_t *out_ids[1] = {out.data()};
    int n_out[1] = {0};
    unsigned int seeds[1] = {0};
    std::vector<double> tps;
    tps.reserve((size_t)runs);

    int rc = 0;
    for (int i = 0; i < warmup; ++i) {
        n_out[0] = 0;
        rc = axiom_qwen_gpu_generate_batch(
                h, in_ids, n_in, 1, max_new, 0,
                0.0f, 0, 0.0f, seeds, out_ids, max_new, n_out, nullptr, 0);
        if (rc != 0) break;
    }

    double total_tokens = 0.0;
    double total_wall = 0.0;
    if (rc == 0 && profile_range) cudaProfilerStart();
    for (int i = 0; rc == 0 && i < runs; ++i) {
        n_out[0] = 0;
        const double t0 = now_sec();
        rc = axiom_qwen_gpu_generate_batch(
                h, in_ids, n_in, 1, max_new, 0,
                0.0f, 0, 0.0f, seeds, out_ids, max_new, n_out, nullptr, 0);
        const double wall = now_sec() - t0;
        const double tok = (double)n_out[0];
        const double run_tps = wall > 0.0 ? tok / wall : 0.0;
        total_tokens += tok;
        total_wall += wall;
        tps.push_back(run_tps);
        std::printf("run=%d rc=%d tokens=%d wall_s=%.6f tok_s=%.6f\n",
                i + 1, rc, n_out[0], wall, run_tps);
    }
    if (profile_range) cudaProfilerStop();
    axiom_qwen_gpu_destroy(h);
    if (rc != 0 || tps.empty()) {
        std::fprintf(stderr, "bench_failed rc=%d\n", rc);
        return 1;
    }
    std::sort(tps.begin(), tps.end());
    const double median = tps[tps.size() / 2u];
    const double mean = total_wall > 0.0 ? total_tokens / total_wall : 0.0;
    std::printf("SUMMARY model=%s ids=%s max=%d runs=%d median_tok_s=%.6f mean_tok_s=%.6f total_tokens=%.0f total_wall_s=%.6f\n",
            model, ids_csv, max_new, runs, median, mean, total_tokens, total_wall);
    return 0;
}
