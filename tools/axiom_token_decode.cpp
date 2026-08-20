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
        fprintf(stderr, "usage: %s TOKENIZER_DIR TOKEN_ID [TOKEN_ID...]\n", argv[0]);
        return 2;
    }

    const uint32_t token_count = (uint32_t)(argc - 2);
    uint32_t *tokens = (uint32_t *)calloc(token_count, sizeof(uint32_t));
    if (!tokens) return 1;
    for (uint32_t i = 0; i < token_count; ++i) {
        tokens[i] = (uint32_t)strtoul(argv[2 + i], NULL, 10);
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
        fprintf(stderr, "axiom-token-decode: tokenizer open failed: %s\n", axiom_status_string(rc));
        free(tokens);
        return 1;
    }

    char *decoded = (char *)calloc(65536, 1);
    char piece[2048];
    if (!decoded) {
        axiom_tokenizer_close(tokenizer);
        free(tokens);
        return 1;
    }

    uint32_t decoded_bytes = 0;
    rc = axiom_tokenizer_decode_ids(
            tokenizer,
            tokens,
            token_count,
            decoded,
            65536,
            &decoded_bytes);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-token-decode: decode failed: %s\n", axiom_status_string(rc));
        free(decoded);
        axiom_tokenizer_close(tokenizer);
        free(tokens);
        return 1;
    }

    printf("ids=");
    for (uint32_t i = 0; i < token_count; ++i) printf("%s%u", i ? "," : "", tokens[i]);
    printf(" decoded=");
    print_escaped(decoded, decoded_bytes);
    puts("");

    for (uint32_t i = 0; i < token_count; ++i) {
        uint32_t piece_bytes = 0;
        rc = axiom_tokenizer_decode_token(
                tokenizer,
                tokens[i],
                piece,
                sizeof(piece),
                &piece_bytes);
        if (rc != AXIOM_OK) {
            fprintf(stderr, "axiom-token-decode: token %u failed: %s\n",
                    tokens[i],
                    axiom_status_string(rc));
            free(decoded);
            axiom_tokenizer_close(tokenizer);
            free(tokens);
            return 1;
        }
        printf("token_id=%u piece=", tokens[i]);
        print_escaped(piece, piece_bytes);
        puts("");
    }

    free(decoded);
    axiom_tokenizer_close(tokenizer);
    free(tokens);
    return 0;
}
