/* Resident bank for Qwen3.8 FP8 MLP layers 56..63.  The bank owns eight
 * native Axiom MLP objects; it does not convert checkpoint weights. */

#include <cstdint>
#include <limits>
#include <new>

#include "axiom/qwen38_fp8_bank.h"
#include "axiom/qwen38_fp8_mlp.h"

static_assert(AXIOM_QWEN38_FP8_BANK_FIRST_LAYER == 56u, "Qwen3.8 FP8 split changed");
static_assert(AXIOM_QWEN38_FP8_BANK_LAST_LAYER == 63u, "Qwen3.8 FP8 split changed");

struct axiom_qwen38_fp8_bank {
    axiom_qwen38_fp8_mlp *layers[AXIOM_QWEN38_FP8_BANK_LAYER_COUNT]{};
    uint64_t device_bytes = 0u;
};

extern "C" int axiom_qwen38_fp8_bank_load(
        axiom_model *model,
        int device,
        axiom_qwen38_fp8_bank **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || !out) return AXIOM_ERR_INVALID_ARGUMENT;

    axiom_qwen38_fp8_bank *bank = new (std::nothrow) axiom_qwen38_fp8_bank();
    if (!bank) return AXIOM_ERR_BUDGET;

    for (uint32_t index = 0u; index < AXIOM_QWEN38_FP8_BANK_LAYER_COUNT; ++index) {
        const uint32_t layer = AXIOM_QWEN38_FP8_BANK_FIRST_LAYER + index;
        int rc = axiom_qwen38_fp8_mlp_load(model, device, layer, &bank->layers[index]);
        if (rc != AXIOM_OK) {
            axiom_qwen38_fp8_bank_destroy(bank);
            return rc;
        }
        const uint64_t bytes = axiom_qwen38_fp8_mlp_device_bytes(bank->layers[index]);
        if (bytes > std::numeric_limits<uint64_t>::max() - bank->device_bytes) {
            axiom_qwen38_fp8_bank_destroy(bank);
            return AXIOM_ERR_BUDGET;
        }
        bank->device_bytes += bytes;
    }

    *out = bank;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_fp8_bank_destroy(axiom_qwen38_fp8_bank *bank) {
    if (!bank) return;
    for (axiom_qwen38_fp8_mlp *mlp : bank->layers) {
        axiom_qwen38_fp8_mlp_destroy(mlp);
    }
    delete bank;
}

extern "C" uint64_t axiom_qwen38_fp8_bank_device_bytes(
        const axiom_qwen38_fp8_bank *bank) {
    return bank ? bank->device_bytes : 0u;
}

extern "C" int axiom_qwen38_fp8_bank_forward_f32_device(
        axiom_qwen38_fp8_bank *bank,
        uint32_t layer,
        const float *input,
        float *out,
        void *stream) {
    if (!bank || !input || !out ||
        layer < AXIOM_QWEN38_FP8_BANK_FIRST_LAYER ||
        layer > AXIOM_QWEN38_FP8_BANK_LAST_LAYER) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t index = layer - AXIOM_QWEN38_FP8_BANK_FIRST_LAYER;
    axiom_qwen38_fp8_mlp *mlp = bank->layers[index];
    if (!mlp) return AXIOM_ERR_INVALID_ARGUMENT;
    return axiom_qwen38_fp8_mlp_forward_f32_device(mlp, input, out, stream);
}
