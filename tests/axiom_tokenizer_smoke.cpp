#include "axiom/axiom.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int write_file(const char *path, const char *data) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    const size_t n = strlen(data);
    const int ok = fwrite(data, 1, n, f) == n;
    fclose(f);
    return ok;
}

int main(void) {
    const char *dir = "/tmp/axiom-tokenizer-smoke";
    const char *json_path = "/tmp/axiom-tokenizer-smoke/tokenizer.json";
    const char *chat_path = "/tmp/axiom-tokenizer-smoke/chat_template.jinja";
    const char *json =
            "{"
            "\"model\":{\"type\":\"BPE\",\"vocab\":{\"c\":0,\"iao\":1,\"<|endoftext|>\":2},\"merges\":[]},"
            "\"added_tokens\":["
            "{\"id\":2,\"content\":\"<|endoftext|>\"},"
            "{\"id\":3,\"content\":\"<|im_start|>\"},"
            "{\"id\":4,\"content\":\"<|im_end|>\"},"
            "{\"id\":5,\"content\":\"<tool_call>\"},"
            "{\"id\":6,\"content\":\"</tool_call>\"}"
            "]"
            "}";

    unlink(json_path);
    unlink(chat_path);
    rmdir(dir);
    (void)mkdir(dir, 0700);
    if (!write_file(json_path, json) || !write_file(chat_path, "{{ messages }}\n")) {
        fprintf(stderr, "failed to create tokenizer smoke fixtures\n");
        return 1;
    }

    axiom_tokenizer *a = NULL;
    axiom_tokenizer *b = NULL;
    axiom_tokenizer_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.path = dir;
    cfg.name = "smoke-hf-bpe";
    cfg.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;

    int rc = axiom_tokenizer_open(&a, &cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom_tokenizer_open(a): %s\n", axiom_status_string(rc));
        return 1;
    }
    rc = axiom_tokenizer_open(&b, &cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom_tokenizer_open(b): %s\n", axiom_status_string(rc));
        return 1;
    }

    axiom_tokenizer_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_tokenizer_info_get(a, &info);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom_tokenizer_info_get: %s\n", axiom_status_string(rc));
        return 1;
    }
    if (info.vocab_size != 3 ||
        info.added_tokens != 5 ||
        info.endoftext_token_id != 2 ||
        info.im_start_token_id != 3 ||
        info.im_end_token_id != 4 ||
        info.tool_call_token_id != 5 ||
        info.tool_call_end_token_id != 6) {
        fprintf(stderr, "unexpected tokenizer metadata\n");
        return 1;
    }

    uint32_t ids[] = {0, 1, 3};
    char decoded[64];
    uint32_t decoded_bytes = 0;
    rc = axiom_tokenizer_decode_ids(a, ids, 3, decoded, sizeof(decoded), &decoded_bytes);
    if (rc != AXIOM_OK || strcmp(decoded, "ciao<|im_start|>") != 0 ||
        decoded_bytes != strlen(decoded)) {
        fprintf(stderr, "tokenizer decode check failed\n");
        return 1;
    }

    int same = 0;
    rc = axiom_tokenizer_same_identity(a, b, &same);
    if (rc != AXIOM_OK || !same) {
        fprintf(stderr, "tokenizer identity check failed\n");
        return 1;
    }

    axiom_tokenizer_close(b);
    axiom_tokenizer_close(a);
    printf("axiom-tokenizer-smoke: OK vocab=%u added=%u\n", info.vocab_size, info.added_tokens);
    return 0;
}
