#include "axiom/axiom.h"

#include <stdio.h>
#include <string.h>

static int wants_json(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--json") == 0) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    axiom_abi_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;

    int rc = axiom_abi_info_get(&info);
    const int check_rc = axiom_abi_check(AXIOM_ABI_VERSION);
    if (rc == AXIOM_OK && check_rc != AXIOM_OK) {
        rc = check_rc;
    }

    if (wants_json(argc, argv)) {
        printf("{\"status\":\"%s\",\"rc\":%d,\"header_abi_version\":%u,"
               "\"runtime_abi_version\":%u,\"struct_size\":%u,"
               "\"version\":\"%s\",\"backend\":\"%s\",\"build_target\":\"%s\"}\n",
               rc == AXIOM_OK ? "pass" : "fail",
               rc,
               AXIOM_ABI_VERSION,
               info.runtime_abi_version,
               info.struct_size,
               info.version ? info.version : "",
               info.backend ? info.backend : "",
               info.build_target ? info.build_target : "");
    } else {
        printf("axiom_abi_check status=%s rc=%d header=%u runtime=%u size=%u version=\"%s\" backend=%s build_target=%s\n",
               rc == AXIOM_OK ? "pass" : "fail",
               rc,
               AXIOM_ABI_VERSION,
               info.runtime_abi_version,
               info.struct_size,
               info.version ? info.version : "",
               info.backend ? info.backend : "",
               info.build_target ? info.build_target : "");
    }
    return rc == AXIOM_OK ? 0 : 1;
}
