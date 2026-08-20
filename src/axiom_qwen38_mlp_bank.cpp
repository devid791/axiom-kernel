/* Complete Qwen3.8 resident MLP dispatch.  ModelOpt uses NVFP4 for all 64
 * layers; the legacy compressed-tensors checkpoint uses NVFP4 0..55 and FP8
 * 56..63.  Both paths preserve the original checkpoint bytes. */

#include <cstdint>
#include <limits>
#include <new>

#include "axiom/qwen38_fp8_bank.h"
#include "axiom/qwen38_mlp_bank.h"
#include "axiom/qwen38_nvfp4_bank.h"

struct axiom_qwen38_mlp_bank {
    axiom_qwen38_nvfp4_bank *nvfp4 = nullptr;
    axiom_qwen38_fp8_bank *fp8 = nullptr;
    uint64_t device_bytes = 0u;
};

extern "C" int axiom_qwen38_mlp_bank_load(
        axiom_model *model,
        int device,
        axiom_qwen38_mlp_bank **out) {
    if (out) *out = nullptr;
    if (!model || !out || device < 0) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_qwen38_mlp_bank *bank = new (std::nothrow) axiom_qwen38_mlp_bank();
    if (!bank) return AXIOM_ERR_BUDGET;
    int rc = axiom_qwen38_nvfp4_bank_load(model, device, &bank->nvfp4);
    if (rc == AXIOM_OK &&
        axiom_qwen38_nvfp4_bank_layer_count(bank->nvfp4) < AXIOM_QWEN38_MLP_LAYER_COUNT) {
        rc = axiom_qwen38_fp8_bank_load(model, device, &bank->fp8);
    }
    if (rc != AXIOM_OK) {
        axiom_qwen38_mlp_bank_destroy(bank);
        return rc;
    }
    const uint64_t nvfp4_bytes = axiom_qwen38_nvfp4_bank_device_bytes(bank->nvfp4);
    const uint64_t fp8_bytes = axiom_qwen38_fp8_bank_device_bytes(bank->fp8);
    if (fp8_bytes > std::numeric_limits<uint64_t>::max() - nvfp4_bytes) {
        axiom_qwen38_mlp_bank_destroy(bank);
        return AXIOM_ERR_BUDGET;
    }
    bank->device_bytes = nvfp4_bytes + fp8_bytes;
    *out = bank;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_mlp_bank_destroy(axiom_qwen38_mlp_bank *bank) {
    if (!bank) return;
    axiom_qwen38_fp8_bank_destroy(bank->fp8);
    axiom_qwen38_nvfp4_bank_destroy(bank->nvfp4);
    delete bank;
}

extern "C" uint64_t axiom_qwen38_mlp_bank_device_bytes(const axiom_qwen38_mlp_bank *bank) {
    return bank ? bank->device_bytes : 0u;
}

extern "C" int axiom_qwen38_mlp_bank_forward_f32_device(
        axiom_qwen38_mlp_bank *bank,
        uint32_t layer,
        const float *input,
        float *out,
        void *stream) {
    if (!bank || !input || !out || layer >= AXIOM_QWEN38_MLP_LAYER_COUNT) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t nvfp4_layers = axiom_qwen38_nvfp4_bank_layer_count(bank->nvfp4);
    if (layer < nvfp4_layers) {
        return axiom_qwen38_nvfp4_bank_forward_f32_device(bank->nvfp4, layer, input, out, stream);
    }
    if (!bank->fp8) return AXIOM_ERR_INVALID_ARGUMENT;
    return axiom_qwen38_fp8_bank_forward_f32_device(bank->fp8, layer, input, out, stream);
}
