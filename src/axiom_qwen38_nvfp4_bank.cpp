/* Resident bank implementation.  It owns only Axiom MLP objects and makes no
 * checkpoint conversion: each member keeps the original NVFP4 bytes. */
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#include "axiom/qwen38_nvfp4_bank.h"
#include "axiom/qwen38_nvfp4_mlp.h"

struct axiom_qwen38_nvfp4_bank {
    int device = -1;
    std::vector<axiom_qwen38_nvfp4_mlp *> layers;
    uint64_t device_bytes = 0;
};

namespace {

uint32_t checkpoint_nvfp4_layer_count(axiom_model *model) {
    axiom_tensor_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    const char *tail = "model.language_model.layers.63.mlp.gate_proj.weight";
    if (axiom_model_tensor_info_get(model, tail, &info) == AXIOM_OK &&
        info.dtype == AXIOM_TENSOR_DTYPE_U8 && info.rank == 2u) {
        return AXIOM_QWEN38_NVFP4_MODEL_LAYERS;
    }
    return AXIOM_QWEN38_NVFP4_TC_LAYERS;
}

}  // namespace

extern "C" int axiom_qwen38_nvfp4_bank_load(
        axiom_model *model,
        int device,
        axiom_qwen38_nvfp4_bank **out) {
    if (out) *out = nullptr;
    if (!model || !out || device < 0) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_qwen38_nvfp4_bank *bank = nullptr;
    try {
        bank = new axiom_qwen38_nvfp4_bank();
        bank->layers.resize(checkpoint_nvfp4_layer_count(model), nullptr);
    } catch (...) {
        delete bank;
        return AXIOM_ERR_BUDGET;
    }
    bank->device = device;
    for (std::size_t index = 0; index < bank->layers.size(); ++index) {
        const uint32_t layer = static_cast<uint32_t>(index);
        int rc = axiom_qwen38_nvfp4_mlp_load(model, device, layer, &bank->layers[layer]);
        if (rc != AXIOM_OK) {
            axiom_qwen38_nvfp4_bank_destroy(bank);
            return rc;
        }
        bank->device_bytes += axiom_qwen38_nvfp4_mlp_device_bytes(bank->layers[layer]);
    }
    *out = bank;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_nvfp4_bank_destroy(axiom_qwen38_nvfp4_bank *bank) {
    if (!bank) return;
    for (axiom_qwen38_nvfp4_mlp *layer : bank->layers) {
        axiom_qwen38_nvfp4_mlp_destroy(layer);
    }
    delete bank;
}

extern "C" uint64_t axiom_qwen38_nvfp4_bank_device_bytes(const axiom_qwen38_nvfp4_bank *bank) {
    return bank ? bank->device_bytes : 0u;
}

extern "C" uint32_t axiom_qwen38_nvfp4_bank_layer_count(
        const axiom_qwen38_nvfp4_bank *bank) {
    return bank ? static_cast<uint32_t>(bank->layers.size()) : 0u;
}

extern "C" int axiom_qwen38_nvfp4_bank_forward_f32_device(
        axiom_qwen38_nvfp4_bank *bank,
        uint32_t layer,
        const float *input,
        float *out,
        void *stream) {
    if (!bank || layer >= bank->layers.size() || !bank->layers[layer]) return AXIOM_ERR_INVALID_ARGUMENT;
    return axiom_qwen38_nvfp4_mlp_forward_f32_device(bank->layers[layer], input, out, stream);
}
