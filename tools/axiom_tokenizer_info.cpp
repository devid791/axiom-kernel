#include "axiom/axiom.h"

#include <stdio.h>
#include <string.h>

static int open_tokenizer(const char *path, const char *name, axiom_tokenizer **out) {
    axiom_tokenizer_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.path = path;
    cfg.name = name;
    cfg.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;
    const int rc = axiom_tokenizer_open(out, &cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-tokenizer-info: open %s failed: %s\n", path, axiom_status_string(rc));
        return 0;
    }
    return 1;
}

static int print_info(axiom_tokenizer *tok) {
    axiom_tokenizer_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    const int rc = axiom_tokenizer_info_get(tok, &info);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-tokenizer-info: info failed: %s\n", axiom_status_string(rc));
        return 0;
    }
    printf("name=%s\n", info.name);
    printf("path=%s\n", info.path);
    printf("tokenizer_json_bytes=%llu\n", (unsigned long long)info.tokenizer_json_bytes);
    printf("tokenizer_hash=0x%016llx\n", (unsigned long long)info.tokenizer_hash);
    printf("chat_template_hash=0x%016llx\n", (unsigned long long)info.chat_template_hash);
    printf("vocab_size=%u\n", info.vocab_size);
    printf("added_tokens=%u\n", info.added_tokens);
    printf("endoftext=%u im_start=%u im_end=%u tool_call=%u tool_call_end=%u\n",
            info.endoftext_token_id,
            info.im_start_token_id,
            info.im_end_token_id,
            info.tool_call_token_id,
            info.tool_call_end_token_id);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s TOKENIZER_DIR [TOKENIZER_DIR]\n", argv[0]);
        return 2;
    }

    axiom_tokenizer *a = NULL;
    axiom_tokenizer *b = NULL;
    if (!open_tokenizer(argv[1], "a", &a)) return 1;
    if (!print_info(a)) {
        axiom_tokenizer_close(a);
        return 1;
    }

    if (argc == 3) {
        if (!open_tokenizer(argv[2], "b", &b)) {
            axiom_tokenizer_close(a);
            return 1;
        }
        puts("---");
        if (!print_info(b)) {
            axiom_tokenizer_close(b);
            axiom_tokenizer_close(a);
            return 1;
        }
        int same = 0;
        const int rc = axiom_tokenizer_same_identity(a, b, &same);
        if (rc != AXIOM_OK) {
            fprintf(stderr, "axiom-tokenizer-info: identity failed: %s\n", axiom_status_string(rc));
            axiom_tokenizer_close(b);
            axiom_tokenizer_close(a);
            return 1;
        }
        printf("---\nsame_identity=%d\n", same);
    }

    axiom_tokenizer_close(b);
    axiom_tokenizer_close(a);
    return 0;
}
