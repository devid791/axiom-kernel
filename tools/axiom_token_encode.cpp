#include "axiom/axiom.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_escaped(const char *s, uint32_t n) {
    putchar('"');
    for (uint32_t i = 0; i < n; ++i) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\n') {
            fputs("\\n", stdout);
        } else if (c == '\r') {
            fputs("\\r", stdout);
        } else if (c == '\t') {
            fputs("\\t", stdout);
        } else if (c == '\\' || c == '"') {
            putchar('\\');
            putchar((char)c);
        } else if (isprint(c)) {
            putchar((char)c);
        } else {
            printf("\\x%02x", c);
        }
    }
    putchar('"');
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s TOKENIZER_DIR TEXT|- < stdin\n", argv[0]);
        return 2;
    }

    char *text = NULL;
    if (argc == 3 && strcmp(argv[2], "-") == 0) {
        size_t cap = 4096;
        size_t len = 0;
        text = (char *)calloc(cap, 1);
        if (!text) return 1;
        int c = 0;
        while ((c = getchar()) != EOF) {
            if (len + 2 > cap) {
                cap *= 2;
                char *next = (char *)realloc(text, cap);
                if (!next) {
                    free(text);
                    return 1;
                }
                text = next;
            }
            text[len++] = (char)c;
            text[len] = 0;
        }
    } else {
        size_t text_len = 0;
        for (int i = 2; i < argc; ++i) text_len += strlen(argv[i]) + (i + 1 < argc ? 1u : 0u);
        text = (char *)calloc(text_len + 1, 1);
        if (!text) return 1;
        for (int i = 2; i < argc; ++i) {
            if (i > 2) strcat(text, " ");
            strcat(text, argv[i]);
        }
    }

    axiom_tokenizer_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.path = argv[1];
    cfg.name = "tokenizer";
    cfg.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;

    axiom_tokenizer *tokenizer = NULL;
    int rc = axiom_tokenizer_open(&tokenizer, &cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-token-encode: tokenizer open failed: %s\n", axiom_status_string(rc));
        free(text);
        return 1;
    }

    uint32_t *ids = (uint32_t *)calloc(8192, sizeof(uint32_t));
    char *decoded = (char *)calloc(65536, 1);
    if (!ids || !decoded) {
        free(decoded); free(ids);
        axiom_tokenizer_close(tokenizer);
        free(text);
        return 1;
    }

    uint32_t count = 0;
    rc = axiom_tokenizer_encode_text(tokenizer, text, ids, 8192, &count);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-token-encode: encode failed: %s\n", axiom_status_string(rc));
        free(decoded); free(ids);
        axiom_tokenizer_close(tokenizer);
        free(text);
        return 1;
    }

    uint32_t decoded_bytes = 0;
    rc = axiom_tokenizer_decode_ids(tokenizer, ids, count, decoded, 65536, &decoded_bytes);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-token-encode: decode check failed: %s\n", axiom_status_string(rc));
        free(decoded); free(ids);
        axiom_tokenizer_close(tokenizer);
        free(text);
        return 1;
    }

    printf("text=");
    print_escaped(text, (uint32_t)strlen(text));
    printf(" ids=");
    for (uint32_t i = 0; i < count; ++i) printf("%s%u", i ? "," : "", ids[i]);
    printf(" decoded=");
    print_escaped(decoded, decoded_bytes);
    puts("");

    free(decoded);
    free(ids);
    axiom_tokenizer_close(tokenizer);
    free(text);
    return 0;
}
