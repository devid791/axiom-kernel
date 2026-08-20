#include "axiom/axiom.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;

    axiom_runtime *rt = NULL;
    int rc = axiom_runtime_create(&rt, &cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "f32-argmax-smoke: runtime rc=%d\n", rc);
        return 1;
    }

    float x[17];
    for (uint32_t i = 0; i < 17u; i++) x[i] = (float)i;
    x[7] = 123.0f;
    x[13] = 123.0f;

    axiom_device_buffer *dev = NULL;
    rc = axiom_device_buffer_create(rt, &dev, sizeof(x));
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(dev, 0, x, sizeof(x)) : rc;
    uint32_t idx = UINT32_MAX;
    float val = 0.0f;
    rc = rc == AXIOM_OK ? axiom_runtime_f32_argmax_device(rt, dev, 0, 17u, &idx, &val) : rc;

    int pass = (rc == AXIOM_OK && idx == 7u && val == 123.0f);
    float neg_inf[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(dev, 0, neg_inf, sizeof(neg_inf)) : rc;
    uint32_t inf_idx = UINT32_MAX;
    float inf_val = 0.0f;
    rc = rc == AXIOM_OK ? axiom_runtime_f32_argmax_device(rt, dev, 0, 4u, &inf_idx, &inf_val) : rc;
    pass = pass && rc == AXIOM_OK && inf_idx == 0u && isinf(inf_val) && inf_val < 0.0f;

    printf("f32_argmax_smoke=status=%s idx=%u val=%.6f neg_inf_idx=%u\n",
            pass ? "pass" : "fail",
            idx,
            val,
            inf_idx);

    axiom_device_buffer_destroy(dev);
    axiom_runtime_destroy(rt);
    return pass ? 0 : 1;
}
