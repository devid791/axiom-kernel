#include "axiom/qwen4exp_admission.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

void usage(const char *program) {
    std::fprintf(stderr,
                 "usage: %s MODEL_DIR [--metadata-only]\n"
                 "\n"
                 "Validates the exact pinned RadixArk qwen4_exp checkpoint.\n"
                 "The default requires all 206 stable shard files;\n"
                 "--metadata-only admits the immutable metadata contract and\n"
                 "reports missing/in-progress shards without claiming checkpoint readiness.\n",
                 program);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    bool metadata_only = false;
    std::string model_root;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--metadata-only") == 0) {
            metadata_only = true;
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (!model_root.empty()) {
            std::fprintf(stderr, "unexpected argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        } else {
            model_root = argv[i];
        }
    }
    if (model_root.empty()) {
        usage(argv[0]);
        return 2;
    }

    axiom::qwen4exp::admission_report report;
    std::string error;
    const bool ok = axiom::qwen4exp::inspect_checkpoint(
            model_root, metadata_only, &report, &error);
    std::puts(axiom::qwen4exp::report_json(report).c_str());
    if (!ok) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    return 0;
}
