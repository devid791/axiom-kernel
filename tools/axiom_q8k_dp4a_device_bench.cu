#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define IQ2_XXS_BLOCK_BYTES 66u
#define QK_K 256u

typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16u];
} axiom_q8k_block;

__device__ __constant__ static const uint16_t axiom_iq2xxs_kgrid[256] = {
        0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
      100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
     1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
     1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
     2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
     4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
     5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
     8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
};

__device__ __constant__ static const uint8_t axiom_iq2xxs_sign_mask[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int parse_u32(const char *s, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v == 0 || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int parse_u32_zero_ok(const char *s, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_double(double *values, uint32_t n) {
    qsort(values, n, sizeof(double), cmp_double);
    return (n & 1u) ? values[n / 2u] : (values[n / 2u - 1u] + values[n / 2u]) * 0.5;
}

static void put_f16(uint8_t *p, uint16_t h) {
    p[0] = (uint8_t)(h & 0xffu);
    p[1] = (uint8_t)(h >> 8u);
}

static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8u) & 0xffu);
    p[2] = (uint8_t)((v >> 16u) & 0xffu);
    p[3] = (uint8_t)(v >> 24u);
}

static void fill_iq2(uint8_t *buf, uint32_t rows, uint32_t cols, uint32_t salt) {
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * IQ2_XXS_BLOCK_BYTES;
    for (uint32_t r = 0; r < rows; ++r) {
        uint8_t *row = buf + (uint64_t)r * row_bytes;
        for (uint32_t b = 0; b < blocks; ++b) {
            uint8_t *blk = row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES;
            put_f16(blk, 0x3c00u);
            for (uint32_t g = 0; g < 8u; ++g) {
                const uint32_t base = r * 17u + b * 29u + g * 37u + salt;
                const uint32_t aux_g =
                        ((base + 3u) & 255u) |
                        (((base + 19u) & 255u) << 8u) |
                        (((base + 47u) & 255u) << 16u) |
                        (((base + 101u) & 255u) << 24u);
                const uint32_t s0 = (base + 5u) & 127u;
                const uint32_t s1 = (base + 11u) & 127u;
                const uint32_t s2 = (base + 23u) & 127u;
                const uint32_t s3 = (base + 41u) & 127u;
                const uint32_t scale = (base >> 3u) & 7u;
                const uint32_t aux_s = s0 | (s1 << 7u) | (s2 << 14u) | (s3 << 21u) | (scale << 28u);
                put_u32(blk + 2u + g * 8u, aux_g);
                put_u32(blk + 2u + g * 8u + 4u, aux_s);
            }
        }
    }
}

__device__ static uint16_t dev_le16(const uint8_t *p) {
    return *(const uint16_t *)p;
}

__device__ static uint32_t dev_le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

__device__ static float dev_f16_to_f32(uint16_t h) {
    __half_raw raw;
    raw.x = h;
    return __half2float(__half(raw));
}

__device__ static float iq2_grid_value(uint32_t grid_idx, uint32_t lane) {
    const uint32_t v = ((uint32_t)axiom_iq2xxs_kgrid[grid_idx] >> (2u * lane)) & 3u;
    return v == 0u ? 8.0f : v == 1u ? 25.0f : 43.0f;
}

__device__ __forceinline__ static int32_t iq2_grid_i8(uint32_t grid_idx, uint32_t lane) {
    const uint32_t v = ((uint32_t)axiom_iq2xxs_kgrid[grid_idx] >> (2u * lane)) & 3u;
    return v == 0u ? 8 : v == 1u ? 25 : 43;
}

__device__ static void iq2_dual_dot_f32_warp(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const float *__restrict__ input,
        uint32_t blocks,
        float *gate_out,
        float *up_out) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t k = lane >> 3u;
    const uint32_t i = lane & 7u;
    const uint32_t grid_shift = 8u * k;
    const uint32_t sign_shift = 7u * k;
    const uint32_t sign_bit = 1u << i;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES;
        const uint8_t *up_blk = up_row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES;
        const float gate_d = dev_f16_to_f32(dev_le16(gate_blk));
        const float up_d = dev_f16_to_f32(dev_le16(up_blk));
        const uint8_t *gate_q = gate_blk + 2u;
        const uint8_t *up_q = up_blk + 2u;
        const float *x_ptr = input + (uint64_t)b * QK_K + lane;
        for (uint32_t g32 = 0; g32 < 8u; ++g32) {
            const uint32_t gate_aux_g = dev_le32(gate_q);
            const uint32_t gate_aux_s = dev_le32(gate_q + 4u);
            const uint32_t up_aux_g = dev_le32(up_q);
            const uint32_t up_aux_s = dev_le32(up_q + 4u);
            float gate_w = iq2_grid_value((gate_aux_g >> grid_shift) & 255u, i);
            float up_w = iq2_grid_value((up_aux_g >> grid_shift) & 255u, i);
            if (axiom_iq2xxs_sign_mask[(gate_aux_s >> sign_shift) & 127u] & sign_bit) gate_w = -gate_w;
            if (axiom_iq2xxs_sign_mask[(up_aux_s >> sign_shift) & 127u] & sign_bit) up_w = -up_w;
            const float x = __ldg(x_ptr);
            gate_acc += gate_d * (0.5f + (float)(gate_aux_s >> 28u)) * 0.25f * gate_w * x;
            up_acc += up_d * (0.5f + (float)(up_aux_s >> 28u)) * 0.25f * up_w * x;
            gate_q += 8u;
            up_q += 8u;
            x_ptr += 32u;
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__device__ static void iq2_dual_dot_q8k_ref_warp(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const axiom_q8k_block *__restrict__ input,
        uint32_t blocks,
        float *gate_out,
        float *up_out) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t k = lane >> 3u;
    const uint32_t i = lane & 7u;
    const uint32_t grid_shift = 8u * k;
    const uint32_t sign_shift = 7u * k;
    const uint32_t sign_bit = 1u << i;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES;
        const uint8_t *up_blk = up_row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES;
        const float gate_d = dev_f16_to_f32(dev_le16(gate_blk));
        const float up_d = dev_f16_to_f32(dev_le16(up_blk));
        const float x_d = input[b].d;
        const uint8_t *gate_q = gate_blk + 2u;
        const uint8_t *up_q = up_blk + 2u;
        const int8_t *x_q = input[b].qs + lane;
        for (uint32_t g32 = 0; g32 < 8u; ++g32) {
            const uint32_t gate_aux_g = dev_le32(gate_q);
            const uint32_t gate_aux_s = dev_le32(gate_q + 4u);
            const uint32_t up_aux_g = dev_le32(up_q);
            const uint32_t up_aux_s = dev_le32(up_q + 4u);
            float gate_w = iq2_grid_value((gate_aux_g >> grid_shift) & 255u, i);
            float up_w = iq2_grid_value((up_aux_g >> grid_shift) & 255u, i);
            if (axiom_iq2xxs_sign_mask[(gate_aux_s >> sign_shift) & 127u] & sign_bit) gate_w = -gate_w;
            if (axiom_iq2xxs_sign_mask[(up_aux_s >> sign_shift) & 127u] & sign_bit) up_w = -up_w;
            const float x = x_d * (float)__ldg(x_q);
            gate_acc += gate_d * (0.5f + (float)(gate_aux_s >> 28u)) * 0.25f * gate_w * x;
            up_acc += up_d * (0.5f + (float)(up_aux_s >> 28u)) * 0.25f * up_w * x;
            gate_q += 8u;
            up_q += 8u;
            x_q += 32u;
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__global__ static void q8k_pack_f32_kernel(const float *input, axiom_q8k_block *out, uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t lane = threadIdx.x;
    __shared__ float absmax[QK_K];
    __shared__ float maxv[QK_K];
    __shared__ float iscale;
    const float x = input[(uint64_t)b * QK_K + lane];
    absmax[lane] = fabsf(x);
    maxv[lane] = x;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && absmax[lane + stride] > absmax[lane]) {
            absmax[lane] = absmax[lane + stride];
            maxv[lane] = maxv[lane + stride];
        }
        __syncthreads();
    }
    if (absmax[0] == 0.0f) {
        out[b].qs[lane] = 0;
        if (lane < QK_K / 16u) out[b].bsums[lane] = 0;
        if (lane == 0u) out[b].d = 0.0f;
        return;
    }
    if (lane == 0u) iscale = -127.0f / maxv[0];
    __syncthreads();
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    out[b].qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < QK_K / 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += out[b].qs[lane * 16u + i];
        out[b].bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) out[b].d = 1.0f / iscale;
}

__device__ __forceinline__ static int32_t iq2_signed_pack4(uint32_t aux_g, uint32_t aux_s, uint32_t lane8) {
    int32_t packed = 0;
    #pragma unroll
    for (uint32_t k = 0; k < 4u; ++k) {
        const uint32_t grid_idx = (aux_g >> (8u * k)) & 255u;
        const uint32_t sign_idx = (aux_s >> (7u * k)) & 127u;
        int32_t w = iq2_grid_i8(grid_idx, lane8);
        if (axiom_iq2xxs_sign_mask[sign_idx] & (1u << lane8)) w = -w;
        packed |= (int32_t)((uint32_t)(uint8_t)(int8_t)w << (8u * k));
    }
    return packed;
}

__device__ __forceinline__ static int32_t q8k_pack4(const int8_t *q, uint32_t lane8) {
    return (int32_t)((uint32_t)(uint8_t)q[lane8] |
           ((uint32_t)(uint8_t)q[8u + lane8] << 8u) |
           ((uint32_t)(uint8_t)q[16u + lane8] << 16u) |
           ((uint32_t)(uint8_t)q[24u + lane8] << 24u));
}

__device__ static void iq2_dual_dot_q8k_qwarp8(
        const uint8_t *__restrict__ gate_row,
        const uint8_t *__restrict__ up_row,
        const axiom_q8k_block *__restrict__ input,
        uint32_t blocks,
        uint32_t lane8,
        float *gate_out,
        float *up_out) {
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *gate_blk = gate_row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES;
        const uint8_t *up_blk = up_row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES;
        const float gate_d = dev_f16_to_f32(dev_le16(gate_blk));
        const float up_d = dev_f16_to_f32(dev_le16(up_blk));
        const float x_d = input[b].d;
        const uint8_t *gate_q = gate_blk + 2u;
        const uint8_t *up_q = up_blk + 2u;
        for (uint32_t g32 = 0; g32 < 8u; ++g32) {
            const uint32_t gate_aux_g = dev_le32(gate_q);
            const uint32_t gate_aux_s = dev_le32(gate_q + 4u);
            const uint32_t up_aux_g = dev_le32(up_q);
            const uint32_t up_aux_s = dev_le32(up_q + 4u);
            const int32_t xpack = q8k_pack4(input[b].qs + g32 * 32u, lane8);
            const int32_t gate_sum = __dp4a(iq2_signed_pack4(gate_aux_g, gate_aux_s, lane8), xpack, 0);
            const int32_t up_sum = __dp4a(iq2_signed_pack4(up_aux_g, up_aux_s, lane8), xpack, 0);
            gate_acc += gate_d * x_d * (0.5f + (float)(gate_aux_s >> 28u)) * 0.25f * (float)gate_sum;
            up_acc += up_d * x_d * (0.5f + (float)(up_aux_s >> 28u)) * 0.25f * (float)up_sum;
            gate_q += 8u;
            up_q += 8u;
        }
    }
    for (uint32_t offset = 4u; offset > 0u; offset >>= 1u) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset, 8);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset, 8);
    }
    *gate_out = gate_acc;
    *up_out = up_acc;
}

__device__ static float swiglu(float gate, float up) {
    const float g = fminf(gate, 10.0f);
    const float u = fminf(fmaxf(up, -10.0f), 10.0f);
    return (g / (1.0f + expf(-g))) * u;
}

__global__ static void gateup_f32_kernel(
        const uint8_t *gate,
        const uint8_t *up,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * IQ2_XXS_BLOCK_BYTES;
    float g = 0.0f;
    float u = 0.0f;
    iq2_dual_dot_f32_warp(gate + (uint64_t)row * row_bytes, up + (uint64_t)row * row_bytes, input, blocks, &g, &u);
    if (lane == 0u) out[row] = swiglu(g, u);
}

__global__ static void gateup_q8k_ref_kernel(
        const uint8_t *gate,
        const uint8_t *up,
        const axiom_q8k_block *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * IQ2_XXS_BLOCK_BYTES;
    float g = 0.0f;
    float u = 0.0f;
    iq2_dual_dot_q8k_ref_warp(gate + (uint64_t)row * row_bytes, up + (uint64_t)row * row_bytes, input, blocks, &g, &u);
    if (lane == 0u) out[row] = swiglu(g, u);
}

__global__ static void gateup_q8k_kernel(
        const uint8_t *gate,
        const uint8_t *up,
        const axiom_q8k_block *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t qwarp = lane >> 3u;
    const uint32_t lane8 = lane & 7u;
    const uint32_t warps = blockDim.x >> 5u;
    const uint32_t rows_per_block = warps * 4u;
    const uint32_t row = (uint32_t)blockIdx.x * rows_per_block + warp * 4u + qwarp;
    if (row >= rows) return;
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * IQ2_XXS_BLOCK_BYTES;
    float g = 0.0f;
    float u = 0.0f;
    iq2_dual_dot_q8k_qwarp8(gate + (uint64_t)row * row_bytes, up + (uint64_t)row * row_bytes, input, blocks, lane8, &g, &u);
    if (lane8 == 0u) out[row] = swiglu(g, u);
}

static void fill_input(float *x, uint32_t cols) {
    for (uint32_t i = 0; i < cols; ++i) {
        const int v = (int)(i % 31u) - 15;
        x[i] = (float)v * 0.001f;
    }
}

int main(int argc, char **argv) {
    uint32_t rows = 2048;
    uint32_t cols = 4096;
    uint32_t reps = 100;
    uint32_t warmup = 5;
    uint32_t samples = 5;
    uint32_t reps_per_sample = 0;
    if (argc > 1 && !parse_u32(argv[1], &rows)) return 2;
    if (argc > 2 && !parse_u32(argv[2], &cols)) return 2;
    if (argc > 3 && !parse_u32(argv[3], &reps)) return 2;
    if (argc > 4 && !parse_u32_zero_ok(argv[4], &warmup)) return 2;
    if (argc > 5 && !parse_u32(argv[5], &samples)) return 2;
    if (argc > 6 && !parse_u32(argv[6], &reps_per_sample)) return 2;
    if (argc > 7 || rows == 0 || cols == 0 || (cols % QK_K) != 0) return 2;
    if (reps_per_sample == 0) reps_per_sample = reps;

    const uint32_t blocks = cols / QK_K;
    const size_t row_bytes = (size_t)blocks * IQ2_XXS_BLOCK_BYTES;
    const size_t weight_bytes = (size_t)rows * row_bytes;
    const size_t input_bytes = (size_t)cols * sizeof(float);
    const size_t out_bytes = (size_t)rows * sizeof(float);
    const size_t q8k_bytes = (size_t)blocks * sizeof(axiom_q8k_block);

    uint8_t *gate = (uint8_t *)malloc(weight_bytes);
    uint8_t *up = (uint8_t *)malloc(weight_bytes);
    float *input = (float *)malloc(input_bytes);
    float *base = (float *)malloc(out_bytes);
    float *qref = (float *)malloc(out_bytes);
    float *cand = (float *)malloc(out_bytes);
    double *base_samples = (double *)malloc((size_t)samples * sizeof(double));
    double *cand_samples = (double *)malloc((size_t)samples * sizeof(double));
    if (!gate || !up || !input || !base || !qref || !cand || !base_samples || !cand_samples) return 1;

    fill_iq2(gate, rows, cols, 7u);
    fill_iq2(up, rows, cols, 113u);
    fill_input(input, cols);

    uint8_t *gate_dev = NULL;
    uint8_t *up_dev = NULL;
    float *input_dev = NULL;
    float *base_dev = NULL;
    float *qref_dev = NULL;
    float *cand_dev = NULL;
    axiom_q8k_block *q8k_dev = NULL;
    cudaError_t err = cudaMalloc(&gate_dev, weight_bytes);
    err = err == cudaSuccess ? cudaMalloc(&up_dev, weight_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&input_dev, input_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&base_dev, out_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&qref_dev, out_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&cand_dev, out_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&q8k_dev, q8k_bytes) : err;
    err = err == cudaSuccess ? cudaMemcpy(gate_dev, gate, weight_bytes, cudaMemcpyHostToDevice) : err;
    err = err == cudaSuccess ? cudaMemcpy(up_dev, up, weight_bytes, cudaMemcpyHostToDevice) : err;
    err = err == cudaSuccess ? cudaMemcpy(input_dev, input, input_bytes, cudaMemcpyHostToDevice) : err;
    if (err != cudaSuccess) return 1;

    const int base_block = 256;
    const int cand_block = 64;
    const int base_grid = (int)((rows + 7u) / 8u);
    const int cand_rows_per_block = (cand_block >> 5) * 4;
    const int cand_grid = (int)((rows + (uint32_t)cand_rows_per_block - 1u) / (uint32_t)cand_rows_per_block);
    q8k_pack_f32_kernel<<<blocks, 256>>>(input_dev, q8k_dev, blocks);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return 1;

    for (uint32_t i = 0; i < warmup; ++i) {
        gateup_f32_kernel<<<base_grid, base_block>>>(gate_dev, up_dev, input_dev, base_dev, rows, cols);
        gateup_q8k_kernel<<<cand_grid, cand_block>>>(gate_dev, up_dev, q8k_dev, cand_dev, rows, cols);
    }
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return 1;

    for (uint32_t s = 0; s < samples; ++s) {
        double t0 = now_ms();
        for (uint32_t i = 0; i < reps_per_sample; ++i) {
            gateup_f32_kernel<<<base_grid, base_block>>>(gate_dev, up_dev, input_dev, base_dev, rows, cols);
        }
        err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return 1;
        double t1 = now_ms();
        base_samples[s] = ((double)reps_per_sample * 1000.0) / (t1 - t0);

        t0 = now_ms();
        for (uint32_t i = 0; i < reps_per_sample; ++i) {
            gateup_q8k_kernel<<<cand_grid, cand_block>>>(gate_dev, up_dev, q8k_dev, cand_dev, rows, cols);
        }
        err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();
        if (err != cudaSuccess) return 1;
        t1 = now_ms();
        cand_samples[s] = ((double)reps_per_sample * 1000.0) / (t1 - t0);
    }

    gateup_q8k_ref_kernel<<<base_grid, base_block>>>(gate_dev, up_dev, q8k_dev, qref_dev, rows, cols);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return 1;

    err = cudaMemcpy(base, base_dev, out_bytes, cudaMemcpyDeviceToHost);
    err = err == cudaSuccess ? cudaMemcpy(qref, qref_dev, out_bytes, cudaMemcpyDeviceToHost) : err;
    err = err == cudaSuccess ? cudaMemcpy(cand, cand_dev, out_bytes, cudaMemcpyDeviceToHost) : err;
    if (err != cudaSuccess) return 1;

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double base_sq = 0.0;
    double dot = 0.0;
    double cand_sq = 0.0;
    double qsum_abs = 0.0;
    double qsum_sq = 0.0;
    double qref_sq = 0.0;
    double qdot = 0.0;
    float max_abs = 0.0f;
    float qmax_abs = 0.0f;
    uint32_t max_idx = 0;
    uint32_t qmax_idx = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        const double d = (double)cand[i] - (double)base[i];
        const double ad = fabs(d);
        if (ad > max_abs) {
            max_abs = (float)ad;
            max_idx = i;
        }
        sum_abs += ad;
        sum_sq += d * d;
        base_sq += (double)base[i] * (double)base[i];
        cand_sq += (double)cand[i] * (double)cand[i];
        dot += (double)base[i] * (double)cand[i];
        const double qd = (double)cand[i] - (double)qref[i];
        const double qad = fabs(qd);
        if (qad > qmax_abs) {
            qmax_abs = (float)qad;
            qmax_idx = i;
        }
        qsum_abs += qad;
        qsum_sq += qd * qd;
        qref_sq += (double)qref[i] * (double)qref[i];
        qdot += (double)qref[i] * (double)cand[i];
    }
    const double base_med = median_double(base_samples, samples);
    const double cand_med = median_double(cand_samples, samples);
    const double mean_abs = sum_abs / (double)rows;
    const double rmse = sqrt(sum_sq / (double)rows);
    const double rel_l2 = base_sq > 0.0 ? sqrt(sum_sq / base_sq) : 0.0;
    const double cosine = (base_sq > 0.0 && cand_sq > 0.0) ? dot / (sqrt(base_sq) * sqrt(cand_sq)) : 0.0;
    const double qmean_abs = qsum_abs / (double)rows;
    const double qrmse = sqrt(qsum_sq / (double)rows);
    const double qrel_l2 = qref_sq > 0.0 ? sqrt(qsum_sq / qref_sq) : 0.0;
    const double qcosine = (qref_sq > 0.0 && cand_sq > 0.0) ? qdot / (sqrt(qref_sq) * sqrt(cand_sq)) : 0.0;
    const int fp32_ok = isfinite(cosine) && max_abs <= 0.05f && rel_l2 <= 0.005 && cosine >= 0.99995;
    const int q8_ok = isfinite(qcosine) && qmax_abs <= 0.002f && qrel_l2 <= 0.00005 && qcosine >= 0.999999f;

    printf("q8k_dp4a_gateup rows=%u cols=%u reps=%u warmup=%u samples=%u reps_per_sample=%u baseline_iter_s=%.3f candidate_iter_s=%.3f speedup=%.3f fp32_max_abs=%.9f fp32_mean_abs=%.9f fp32_rmse=%.9f fp32_rel_l2=%.9f fp32_cosine=%.9f fp32_max_idx=%u base=%.9f cand=%.9f fp32_status=%s q8_max_abs=%.9f q8_mean_abs=%.9f q8_rmse=%.9f q8_rel_l2=%.9f q8_cosine=%.9f q8_max_idx=%u qref=%.9f qcand=%.9f q8_status=%s status=%s\n",
            rows, cols, reps, warmup, samples, reps_per_sample,
            base_med, cand_med, base_med > 0.0 ? cand_med / base_med : 0.0,
            max_abs, mean_abs, rmse, rel_l2, cosine, max_idx, base[max_idx], cand[max_idx],
            fp32_ok ? "pass" : "warn",
            qmax_abs, qmean_abs, qrmse, qrel_l2, qcosine, qmax_idx, qref[qmax_idx], cand[qmax_idx],
            q8_ok ? "pass" : "warn",
            q8_ok ? "pass" : "warn");

    cudaFree(q8k_dev);
    cudaFree(cand_dev);
    cudaFree(qref_dev);
    cudaFree(base_dev);
    cudaFree(input_dev);
    cudaFree(up_dev);
    cudaFree(gate_dev);
    free(cand_samples);
    free(base_samples);
    free(cand);
    free(qref);
    free(base);
    free(input);
    free(up);
    free(gate);
    return q8_ok ? 0 : 3;
}
