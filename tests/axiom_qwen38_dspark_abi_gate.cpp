#define AXIOM_QWEN38_DSPARK_COMPUTE_IMPLEMENTATION 1
#include "axiom/qwen38_dspark_compute.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

struct legacy_config_v1 {
    uint32_t abi_version;
    uint32_t max_context;
    uint64_t flags;
};

struct legacy_device_state_v1 {
    uint32_t abi_version;
    uint32_t graph_ready;
    uint32_t device_session_active;
    uint32_t proposal_tokens;
    uint32_t verify_width;
    uint32_t reserved0;
    uint64_t resident_device_bytes;
    const uint32_t *anchor_token_device;
    const uint32_t *anchor_position_device;
    const uint32_t *proposal_tokens_device;
    const uint32_t *verify_tokens_device;
    const uint32_t *accepted_prefix_device;
    const uint32_t *target_commit_prefix_device;
    const uint32_t *continuation_token_device;
    const float *continuation_logit_device;
    const uint32_t *async_status_device;
    axiom_qwen38_dspark_device_history *history_device;
};

struct legacy_history_v1 {
    uint32_t proposal_tokens[7];
    uint32_t accepted_prefix;
    uint32_t continuation_token;
    uint32_t async_status;
    uint32_t next_position;
};

template <typename T>
struct guarded {
    uint64_t before;
    T value;
    uint64_t after;
};

constexpr uint64_t kBefore = UINT64_C(0x91d4a672fc0385be);
constexpr uint64_t kAfter = UINT64_C(0x6e2b598d03fc7a41);

bool expect_invalid(const char *name, const int rc) {
    if (rc == AXIOM_ERR_INVALID_ARGUMENT) return true;
    std::fprintf(stderr, "%s: expected AXIOM_ERR_INVALID_ARGUMENT, got %d\n", name, rc);
    return false;
}

}  // namespace

static_assert(AXIOM_QWEN38_DSPARK_COMPUTE_LAYOUT_VERSION == 2u,
              "DSpark compute layout must remain revision 2");
static_assert(AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION == 2u,
              "DSpark device ABI must reject revision 1");
static_assert(sizeof(legacy_config_v1) < sizeof(axiom_qwen38_dspark_compute_config));
static_assert(sizeof(legacy_device_state_v1) < sizeof(axiom_qwen38_dspark_compute_device_state));
static_assert(sizeof(legacy_history_v1) < sizeof(axiom_qwen38_dspark_device_history));

int main() {
    bool ok = true;

    axiom_qwen38_dspark_compute *created =
            reinterpret_cast<axiom_qwen38_dspark_compute *>(UINTPTR_MAX);
    ok &= expect_invalid(
            "legacy create symbol",
            axiom_qwen38_dspark_compute_create(
                    reinterpret_cast<const axiom_qwen38_dspark *>(uintptr_t{1}),
                    reinterpret_cast<axiom_runtime *>(uintptr_t{1}), 0,
                    reinterpret_cast<const axiom_qwen38_dspark_compute_config *>(uintptr_t{1}),
                    reinterpret_cast<const axiom_qwen38_dspark_target_binding *>(uintptr_t{1}),
                    &created));
    if (created != nullptr) {
        std::fprintf(stderr, "legacy create symbol: output was not cleared\n");
        ok = false;
    }

    created = reinterpret_cast<axiom_qwen38_dspark_compute *>(UINTPTR_MAX);
    ok &= expect_invalid(
            "v2 create with v1 config size",
            axiom_qwen38_dspark_compute_create_v2(
                    reinterpret_cast<const axiom_qwen38_dspark *>(uintptr_t{1}),
                    reinterpret_cast<axiom_runtime *>(uintptr_t{1}), 0,
                    reinterpret_cast<const axiom_qwen38_dspark_compute_config *>(uintptr_t{1}),
                    sizeof(legacy_config_v1),
                    reinterpret_cast<const axiom_qwen38_dspark_target_binding *>(uintptr_t{1}),
                    &created));
    if (created != nullptr) {
        std::fprintf(stderr, "v2 create size rejection: output was not cleared\n");
        ok = false;
    }

    guarded<legacy_device_state_v1> state{};
    state.before = kBefore;
    state.after = kAfter;
    state.value.abi_version = 1u;
    const auto state_snapshot = state;
    ok &= expect_invalid(
            "legacy device-state symbol",
            axiom_qwen38_dspark_compute_device_state_get(
                    reinterpret_cast<const axiom_qwen38_dspark_compute *>(uintptr_t{1}),
                    reinterpret_cast<axiom_qwen38_dspark_compute_device_state *>(&state.value)));
    if (std::memcmp(&state, &state_snapshot, sizeof(state)) != 0) {
        std::fprintf(stderr, "legacy device-state symbol modified the v1 buffer\n");
        ok = false;
    }
    ok &= expect_invalid(
            "v2 device-state with v1 output size",
            axiom_qwen38_dspark_compute_device_state_get_v2(
                    reinterpret_cast<const axiom_qwen38_dspark_compute *>(uintptr_t{1}),
                    reinterpret_cast<axiom_qwen38_dspark_compute_device_state *>(uintptr_t{1}),
                    sizeof(legacy_device_state_v1)));

    guarded<legacy_history_v1> history{};
    history.before = kBefore;
    history.after = kAfter;
    const auto history_snapshot = history;
    const auto fake_u32 = reinterpret_cast<const uint32_t *>(uintptr_t{1});
    ok &= expect_invalid(
            "legacy history-pack symbol",
            axiom_qwen38_dspark_compute_device_history_pack_enqueue(
                    fake_u32, fake_u32, fake_u32, fake_u32, fake_u32,
                    reinterpret_cast<axiom_qwen38_dspark_device_history *>(&history.value),
                    reinterpret_cast<void *>(uintptr_t{1})));
    if (std::memcmp(&history, &history_snapshot, sizeof(history)) != 0) {
        std::fprintf(stderr, "legacy history-pack symbol modified the v1 buffer\n");
        ok = false;
    }
    ok &= expect_invalid(
            "v2 history-pack with v1 output size",
            axiom_qwen38_dspark_compute_device_history_pack_enqueue_v2(
                    fake_u32, fake_u32, fake_u32, fake_u32, fake_u32, fake_u32,
                    reinterpret_cast<axiom_qwen38_dspark_device_history *>(uintptr_t{1}),
                    sizeof(legacy_history_v1), reinterpret_cast<void *>(uintptr_t{1})));

    if (!ok) return 1;
    std::printf(
            "axiom-qwen38-dspark-abi-gate: PASS layout=%u config=%zu>%zu "
            "state=%zu>%zu history=%zu>%zu legacy_symbols=reject\n",
            AXIOM_QWEN38_DSPARK_COMPUTE_LAYOUT_VERSION,
            sizeof(axiom_qwen38_dspark_compute_config), sizeof(legacy_config_v1),
            sizeof(axiom_qwen38_dspark_compute_device_state), sizeof(legacy_device_state_v1),
            sizeof(axiom_qwen38_dspark_device_history), sizeof(legacy_history_v1));
    return 0;
}
