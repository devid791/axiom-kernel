#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GGUF_UINT8 = 0,
    GGUF_INT8 = 1,
    GGUF_UINT16 = 2,
    GGUF_INT16 = 3,
    GGUF_UINT32 = 4,
    GGUF_INT32 = 5,
    GGUF_FLOAT32 = 6,
    GGUF_BOOL = 7,
    GGUF_STRING = 8,
    GGUF_ARRAY = 9,
    GGUF_UINT64 = 10,
    GGUF_INT64 = 11,
    GGUF_FLOAT64 = 12,
};

static int read_exact(FILE *f, void *p, size_t n) {
    return fread(p, 1, n, f) == n;
}

static int read_u32(FILE *f, uint32_t *v) {
    unsigned char b[4];
    if (!read_exact(f, b, sizeof(b))) return 0;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
         ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 1;
}

static int read_i32(FILE *f, int32_t *v) {
    uint32_t u = 0;
    if (!read_u32(f, &u)) return 0;
    *v = (int32_t)u;
    return 1;
}

static int read_f32(FILE *f, float *v) {
    uint32_t u = 0;
    if (!read_u32(f, &u)) return 0;
    memcpy(v, &u, sizeof(*v));
    return 1;
}

static int read_u64(FILE *f, uint64_t *v) {
    unsigned char b[8];
    if (!read_exact(f, b, sizeof(b))) return 0;
    *v = 0;
    for (uint32_t i = 0; i < 8; ++i) *v |= ((uint64_t)b[i]) << (8u * i);
    return 1;
}

static int skip_bytes(FILE *f, uint64_t n) {
    while (n > 0) {
        const long chunk = n > (uint64_t)0x40000000 ? 0x40000000L : (long)n;
        if (fseek(f, chunk, SEEK_CUR) != 0) return 0;
        n -= (uint64_t)chunk;
    }
    return 1;
}

static char *read_string(FILE *f, uint64_t *out_len) {
    uint64_t n = 0;
    if (!read_u64(f, &n)) return NULL;
    if (n > (uint64_t)1 << 30) return NULL;
    char *s = (char *)calloc((size_t)n + 1, 1);
    if (!s) return NULL;
    if (!read_exact(f, s, (size_t)n)) {
        free(s);
        return NULL;
    }
    if (out_len) *out_len = n;
    return s;
}

static int skip_value(FILE *f, uint32_t type);

static int skip_array(FILE *f) {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    if (!read_u32(f, &elem_type) || !read_u64(f, &count)) return 0;
    for (uint64_t i = 0; i < count; ++i) {
        if (!skip_value(f, elem_type)) return 0;
    }
    return 1;
}

static int skip_value(FILE *f, uint32_t type) {
    switch (type) {
        case GGUF_UINT8:
        case GGUF_INT8:
        case GGUF_BOOL:
            return skip_bytes(f, 1);
        case GGUF_UINT16:
        case GGUF_INT16:
            return skip_bytes(f, 2);
        case GGUF_UINT32:
        case GGUF_INT32:
        case GGUF_FLOAT32:
            return skip_bytes(f, 4);
        case GGUF_UINT64:
        case GGUF_INT64:
        case GGUF_FLOAT64:
            return skip_bytes(f, 8);
        case GGUF_STRING: {
            uint64_t n = 0;
            if (!read_u64(f, &n)) return 0;
            return skip_bytes(f, n);
        }
        case GGUF_ARRAY:
            return skip_array(f);
        default:
            return 0;
    }
}

static int print_value_or_skip(FILE *f, uint32_t type, const char *key) {
    if (type == GGUF_STRING) {
        char *s = read_string(f, NULL);
        if (!s) return 0;
        printf("meta.%s=%s\n", key, s);
        free(s);
        return 1;
    }
    if (type == GGUF_UINT32) {
        uint32_t v = 0;
        if (!read_u32(f, &v)) return 0;
        printf("meta.%s=%u\n", key, v);
        return 1;
    }
    if (type == GGUF_UINT64) {
        uint64_t v = 0;
        if (!read_u64(f, &v)) return 0;
        printf("meta.%s=%llu\n", key, (unsigned long long)v);
        return 1;
    }
    if (type == GGUF_ARRAY) {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        if (!read_u32(f, &elem_type) || !read_u64(f, &count)) return 0;
        printf("meta.%s=array(type=%u,count=%llu)\n",
                key,
                elem_type,
                (unsigned long long)count);
        if (count <= 128 &&
            (elem_type == GGUF_UINT32 || elem_type == GGUF_INT32 || elem_type == GGUF_FLOAT32)) {
            printf("meta.%s.values=[", key);
            for (uint64_t i = 0; i < count; ++i) {
                if (elem_type == GGUF_UINT32) {
                    uint32_t v = 0;
                    if (!read_u32(f, &v)) return 0;
                    printf("%s%u", i ? "," : "", v);
                } else if (elem_type == GGUF_INT32) {
                    int32_t v = 0;
                    if (!read_i32(f, &v)) return 0;
                    printf("%s%d", i ? "," : "", v);
                } else {
                    float v = 0.0f;
                    if (!read_f32(f, &v)) return 0;
                    printf("%s%.9g", i ? "," : "", v);
                }
            }
            puts("]");
        } else {
            for (uint64_t i = 0; i < count; ++i) {
                if (!skip_value(f, elem_type)) return 0;
            }
        }
        return 1;
    }
    return skip_value(f, type);
}

static int key_interesting(const char *key) {
    const char *needles[] = {
        "general.architecture",
        "general.name",
        "general.quantization_version",
        "general.file_type",
        "tokenizer.ggml.model",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "deepseek",
        "llama",
        "context_length",
        "embedding_length",
        "block_count",
        "attention.head_count",
        "attention.head_count_kv",
        "expert",
    };
    for (size_t i = 0; i < sizeof(needles) / sizeof(needles[0]); ++i) {
        if (strstr(key, needles[i])) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    unsigned char magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!read_exact(f, magic, sizeof(magic)) ||
        !read_u32(f, &version) ||
        !read_u64(f, &tensor_count) ||
        !read_u64(f, &kv_count)) {
        fprintf(stderr, "axiom-gguf-info: bad header\n");
        fclose(f);
        return 1;
    }
    if (memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "axiom-gguf-info: not GGUF\n");
        fclose(f);
        return 1;
    }

    printf("file=%s\n", argv[1]);
    printf("magic=GGUF version=%u tensor_count=%llu kv_count=%llu\n",
            version,
            (unsigned long long)tensor_count,
            (unsigned long long)kv_count);

    for (uint64_t i = 0; i < kv_count; ++i) {
        char *key = read_string(f, NULL);
        uint32_t type = 0;
        if (!key || !read_u32(f, &type)) {
            free(key);
            fprintf(stderr, "axiom-gguf-info: bad metadata\n");
            fclose(f);
            return 1;
        }
        const int interesting = key_interesting(key);
        if (interesting) {
            if (!print_value_or_skip(f, type, key)) {
                free(key);
                fclose(f);
                return 1;
            }
        } else if (!skip_value(f, type)) {
            free(key);
            fclose(f);
            return 1;
        }
        free(key);
    }

    for (uint64_t i = 0; i < tensor_count; ++i) {
        char *name = read_string(f, NULL);
        uint32_t dims = 0;
        uint64_t shape[8] = {0};
        uint32_t type = 0;
        uint64_t offset = 0;
        if (!name || !read_u32(f, &dims) || dims > 8) {
            free(name);
            fclose(f);
            return 1;
        }
        for (uint32_t d = 0; d < dims; ++d) {
            if (!read_u64(f, &shape[d])) {
                free(name);
                fclose(f);
                return 1;
            }
        }
        if (!read_u32(f, &type) || !read_u64(f, &offset)) {
            free(name);
            fclose(f);
            return 1;
        }
        if (getenv("AXIOM_GGUF_INFO_ALL") ||
            i < 64 || strstr(name, "token_embd") || strstr(name, "output") ||
            strstr(name, "blk.0")) {
            printf("tensor.%llu name=%s type=%u shape=[",
                    (unsigned long long)i,
                    name,
                    type);
            for (uint32_t d = 0; d < dims; ++d) {
                printf("%s%llu", d ? "," : "", (unsigned long long)shape[d]);
            }
            printf("] offset=%llu\n", (unsigned long long)offset);
        }
        free(name);
    }

    fclose(f);
    return 0;
}
