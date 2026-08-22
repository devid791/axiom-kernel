/* Native, fixed-shape FlashInfer Qwen3.8 single-prefill launcher.
 * Source provenance: FlashInfer Apache-2.0 headers vendored in
 * third_party/flashinfer; this TU deliberately has no Python/TVM ABI. */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <exception>

#include "axiom/qwen38_flashinfer.h"

#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/fastdiv.cuh>
#include <flashinfer/pos_enc.cuh>

namespace {

constexpr uint32_t kQwen38TemporalWidth = 8u;
constexpr uint32_t kQwen38Heads = 24u;
constexpr uint32_t kQwen38KvHeads = 4u;
constexpr uint32_t kQwen38HeadDim = 256u;

static_assert(kQwen38Heads / kQwen38KvHeads == 6u, "Qwen3.8 GQA changed");

struct qwen38_flashinfer_params {
    using DTypeQ = nv_bfloat16;
    using DTypeKV = __nv_fp8_e4m3;
    using DTypeO = nv_bfloat16;
    using IdType = int32_t;

    DTypeQ *q = nullptr;
    DTypeKV *k = nullptr;
    DTypeKV *v = nullptr;
    DTypeO *o = nullptr;
    float *lse = nullptr;
    flashinfer::uint_fastdiv group_size{};

    uint8_t *maybe_custom_mask = nullptr;
    float *maybe_alibi_slopes = nullptr;
    uint8_t *maybe_k_cache_sf = nullptr;
    uint8_t *maybe_v_cache_sf = nullptr;
    double logits_soft_cap = 0.0;
    double sm_scale = 0.0625;  // 1 / sqrt(256)
    double rope_rcp_scale = 1.0;
    double rope_rcp_theta = 1.0;

    uint32_t qo_len = kQwen38TemporalWidth;
    uint32_t kv_len = kQwen38TemporalWidth;
    const uint32_t *kv_len_device = nullptr;
    uint32_t num_qo_heads = kQwen38Heads;
    uint32_t num_kv_heads = kQwen38KvHeads;
    uint32_t q_stride_n = kQwen38Heads * kQwen38HeadDim;
    uint32_t q_stride_h = kQwen38HeadDim;
    uint32_t k_stride_n = kQwen38KvHeads * kQwen38HeadDim;
    uint32_t k_stride_h = kQwen38HeadDim;
    uint32_t v_stride_n = kQwen38KvHeads * kQwen38HeadDim;
    uint32_t v_stride_h = kQwen38HeadDim;
    uint32_t head_dim = kQwen38HeadDim;
    int32_t window_left = -1;
    bool partition_kv = false;

    __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t) const {
        return qo_len;
    }

    __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t) const {
#if defined(__CUDA_ARCH__)
        return kv_len_device ? kv_len_device[0] : kv_len;
#else
        /* Host dispatch intentionally only consumes the static maximum. */
        return kv_len;
#endif
    }
};

int from_cuda(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

uint32_t graph_planning_kv_len(
        uint32_t kv_len_host,
        const uint32_t *kv_len_device) {
    if (!kv_len_device) return kv_len_host;
    const char *value = std::getenv("AXIOM_QWEN38_FLASHINFER_PLAN_KV");
    if (!value || value[0] == '\0') return kv_len_host;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (!end || end == value || end[0] != '\0' ||
        (parsed != 256ul && parsed != 512ul && parsed != 1024ul && parsed != 2048ul)) {
        return kv_len_host;
    }
    return static_cast<uint32_t>(parsed);
}

}  // namespace

extern "C" int axiom_qwen38_flashinfer_temporal8_bf16_e4m3_device(
        const uint16_t *q_bf16,
        const uint8_t *k_e4m3,
        const uint8_t *v_e4m3,
        uint16_t *out_bf16,
        uint32_t kv_len_host,
        const uint32_t *kv_len_device,
        uint16_t *split_kv_tmp_bf16,
        void *stream) {
    if (!q_bf16 || !k_e4m3 || !v_e4m3 || !out_bf16 ||
        kv_len_host < kQwen38TemporalWidth) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    qwen38_flashinfer_params params{};
    params.q = reinterpret_cast<nv_bfloat16 *>(const_cast<uint16_t *>(q_bf16));
    params.k = reinterpret_cast<__nv_fp8_e4m3 *>(const_cast<uint8_t *>(k_e4m3));
    params.v = reinterpret_cast<__nv_fp8_e4m3 *>(const_cast<uint8_t *>(v_e4m3));
    params.o = reinterpret_cast<nv_bfloat16 *>(out_bf16);
    params.group_size = flashinfer::uint_fastdiv(kQwen38Heads / kQwen38KvHeads);
    params.kv_len = graph_planning_kv_len(kv_len_host, kv_len_device);
    params.kv_len_device = kv_len_device;

    try {
        const cudaError_t status = flashinfer::SinglePrefillWithKVCacheDispatched<
                kQwen38HeadDim, kQwen38HeadDim, flashinfer::PosEncodingMode::kNone,
                /* use_fp16_qk_reduction = */ false, flashinfer::MaskMode::kCausal,
                flashinfer::DefaultAttention<false, false, false, false>>(
                params, reinterpret_cast<nv_bfloat16 *>(split_kv_tmp_bf16),
                static_cast<cudaStream_t>(stream));
        return from_cuda(status);
    } catch (const std::exception &) {
        return AXIOM_ERR_CUDA;
    } catch (...) {
        return AXIOM_ERR_CUDA;
    }
}
