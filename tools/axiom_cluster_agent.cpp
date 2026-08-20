#include "axiom/axiom.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static const char *arg_value(int argc, char **argv, const char *name, const char *fallback = nullptr) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return fallback;
}

static bool has_arg(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return true;
    }
    return false;
}

static uint32_t arg_u32(int argc, char **argv, const char *name, uint32_t fallback) {
    const char *v = arg_value(argc, argv, name);
    if (!v) return fallback;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(v, &end, 0);
    return end && *end == '\0' ? (uint32_t)parsed : fallback;
}

static uint64_t arg_u64(int argc, char **argv, const char *name, uint64_t fallback) {
    const char *v = arg_value(argc, argv, name);
    if (!v) return fallback;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(v, &end, 0);
    return end && *end == '\0' ? (uint64_t)parsed : fallback;
}

static axiom_transport_kind parse_transport(const char *s) {
    if (!s || std::strcmp(s, "tcp") == 0) return AXIOM_TRANSPORT_TCP;
    if (std::strcmp(s, "rdma") == 0) return AXIOM_TRANSPORT_RDMA;
    if (std::strcmp(s, "quic") == 0) return AXIOM_TRANSPORT_QUIC;
    return (axiom_transport_kind)0;
}

static const char *transport_name(axiom_transport_kind transport) {
    switch (transport) {
    case AXIOM_TRANSPORT_TCP: return "tcp";
    case AXIOM_TRANSPORT_RDMA: return "rdma";
    case AXIOM_TRANSPORT_QUIC: return "quic";
    default: return "unknown";
    }
}

static axiom_agent_kind parse_kind(const char *s) {
    if (!s || std::strcmp(s, "model-worker") == 0) return AXIOM_AGENT_MODEL_WORKER;
    if (std::strcmp(s, "worker") == 0) return AXIOM_AGENT_WORKER;
    if (std::strcmp(s, "coordinator") == 0) return AXIOM_AGENT_COORDINATOR;
    return (axiom_agent_kind)0;
}

static const char *kind_name(axiom_agent_kind kind) {
    switch (kind) {
    case AXIOM_AGENT_COORDINATOR: return "coordinator";
    case AXIOM_AGENT_WORKER: return "worker";
    case AXIOM_AGENT_MODEL_WORKER: return "model-worker";
    default: return "unknown";
    }
}

static const char *status_name(axiom_agent_status status) {
    switch (status) {
    case AXIOM_AGENT_CREATED: return "created";
    case AXIOM_AGENT_DISPATCHED: return "dispatched";
    case AXIOM_AGENT_RECEIVED: return "received";
    default: return "unknown";
    }
}

static void json_string(const char *s) {
    std::putchar('"');
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
            switch (*p) {
            case '\\': std::fputs("\\\\", stdout); break;
            case '"': std::fputs("\\\"", stdout); break;
            case '\n': std::fputs("\\n", stdout); break;
            case '\r': std::fputs("\\r", stdout); break;
            case '\t': std::fputs("\\t", stdout); break;
            default:
                if (*p < 32) std::printf("\\u%04x", (unsigned)*p);
                else std::putchar((int)*p);
                break;
            }
        }
    }
    std::putchar('"');
}

static void print_spawn_json(
        const char *mode,
        axiom_transport_kind transport,
        const axiom_agent_spawn_info *info) {
    std::printf("{\"schema\":\"axiom_cluster_agent_v1\",\"status\":\"pass\",\"mode\":");
    json_string(mode);
    std::printf(",\"transport\":");
    json_string(transport_name(transport));
    std::printf(",\"agent_id\":");
    json_string(info->agent_id);
    std::printf(",\"goal_id\":");
    json_string(info->goal_id);
    std::printf(",\"role\":");
    json_string(info->role);
    std::printf(",\"objective\":");
    json_string(info->objective);
    std::printf(",\"kind\":");
    json_string(kind_name(info->kind));
    std::printf(",\"agent_status\":");
    json_string(status_name(info->status));
    std::printf(",\"source_node\":%u,\"source_device\":%u,\"target_node\":%u,"
                "\"target_device\":%u,\"task_id\":%llu,\"session_id\":%llu,"
                "\"budget_tokens\":%llu,\"sequence\":%llu,\"flags\":%llu}\n",
                info->source_node,
                info->source_device,
                info->target_node,
                info->target_device,
                (unsigned long long)info->task_id,
                (unsigned long long)info->session_id,
                (unsigned long long)info->budget_tokens,
                (unsigned long long)info->sequence,
                (unsigned long long)info->flags);
}

static int fail_json(const char *message, int rc = 1) {
    std::printf("{\"schema\":\"axiom_cluster_agent_v1\",\"status\":\"fail\",\"error\":");
    json_string(message);
    std::printf("}\n");
    return rc;
}

static void usage(FILE *out) {
    std::fprintf(out,
            "usage:\n"
            "  axiom-cluster-agent --mode listen --listen IP:PORT [--transport rdma] [--node-id N]\n"
            "  axiom-cluster-agent --mode spawn --join IP:PORT [--transport rdma] [--node-id N] [--target-node N]\n");
}

int main(int argc, char **argv) {
    if (has_arg(argc, argv, "--help")) {
        usage(stdout);
        return 0;
    }
    const char *mode = arg_value(argc, argv, "--mode");
    if (!mode) {
        usage(stderr);
        return fail_json("missing --mode");
    }
    const axiom_transport_kind transport = parse_transport(arg_value(argc, argv, "--transport", "rdma"));
    if (transport == (axiom_transport_kind)0) return fail_json("invalid transport");

    const uint32_t node_id = arg_u32(argc, argv, "--node-id", std::strcmp(mode, "spawn") == 0 ? 0u : 1u);
    const uint32_t node_count = arg_u32(argc, argv, "--node-count", 2u);
    if (node_count == 0 || node_id >= node_count) return fail_json("invalid node id/count");

    if (std::strcmp(mode, "listen") == 0) {
        const char *listen = arg_value(argc, argv, "--listen");
        if (!listen) return fail_json("missing --listen");
        axiom_cluster_config cfg{};
        cfg.abi_version = AXIOM_ABI_VERSION;
        cfg.transport = transport;
        cfg.listen_addr = listen;
        cfg.node_id = node_id;
        cfg.node_count = node_count;
        axiom_cluster *server = nullptr;
        int rc = axiom_cluster_create(&server, &cfg);
        if (rc != AXIOM_OK) return fail_json(axiom_status_string(rc));
        axiom_cluster *peer = nullptr;
        rc = axiom_cluster_accept(server, &peer);
        if (rc != AXIOM_OK) {
            axiom_cluster_destroy(server);
            return fail_json(axiom_status_string(rc));
        }
        axiom_agent_spawn_info info{};
        info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_cluster_recv_agent_spawn(peer, &info);
        if (rc != AXIOM_OK) {
            axiom_cluster_destroy(peer);
            axiom_cluster_destroy(server);
            return fail_json(axiom_status_string(rc));
        }
        print_spawn_json(mode, transport, &info);
        axiom_cluster_destroy(peer);
        axiom_cluster_destroy(server);
        return 0;
    }

    if (std::strcmp(mode, "spawn") == 0) {
        const char *join = arg_value(argc, argv, "--join");
        if (!join) return fail_json("missing --join");
        const axiom_agent_kind kind = parse_kind(arg_value(argc, argv, "--kind", "model-worker"));
        if (kind == (axiom_agent_kind)0) return fail_json("invalid agent kind");

        axiom_cluster_config cfg{};
        cfg.abi_version = AXIOM_ABI_VERSION;
        cfg.transport = transport;
        cfg.join_addr = join;
        cfg.node_id = node_id;
        cfg.node_count = node_count;
        axiom_cluster *client = nullptr;
        int rc = axiom_cluster_create(&client, &cfg);
        if (rc != AXIOM_OK) return fail_json(axiom_status_string(rc));

        axiom_goal_config goal{};
        goal.abi_version = AXIOM_ABI_VERSION;
        goal.goal_id = arg_value(argc, argv, "--goal-id", "axiom-cx7-direct-goal");
        goal.objective = arg_value(argc, argv, "--objective", "activate Axiom CX7 direct multinode model-worker control plane");
        goal.budget_tokens = arg_u64(argc, argv, "--budget-tokens", 4096u);
        rc = axiom_cluster_goal_begin(client, &goal);
        if (rc != AXIOM_OK) {
            axiom_cluster_destroy(client);
            return fail_json(axiom_status_string(rc));
        }

        axiom_agent_spawn_config spawn{};
        spawn.abi_version = AXIOM_ABI_VERSION;
        spawn.kind = kind;
        spawn.agent_id = arg_value(argc, argv, "--agent-id", "axiom-remote-model-worker");
        spawn.goal_id = goal.goal_id;
        spawn.role = arg_value(argc, argv, "--role", "remote-model-worker");
        spawn.objective = goal.objective;
        spawn.placement.abi_version = AXIOM_ABI_VERSION;
        spawn.placement.kind = AXIOM_PLACEMENT_CLUSTER_NODE;
        spawn.placement.node_id = arg_u32(argc, argv, "--target-node", node_id == 0 ? 1u : 0u);
        spawn.placement.device_id = arg_u32(argc, argv, "--target-device", 0u);
        spawn.target_node = spawn.placement.node_id;
        spawn.target_device = spawn.placement.device_id;
        spawn.task_id = arg_u64(argc, argv, "--task-id", 0xA610C70001ull);
        spawn.session_id = arg_u64(argc, argv, "--session-id", 0x51A7C70001ull);
        spawn.budget_tokens = arg_u64(argc, argv, "--agent-budget-tokens", goal.budget_tokens);

        axiom_agent_spawn_info info{};
        info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_cluster_spawn_agent(client, &spawn, &info);
        if (rc != AXIOM_OK) {
            axiom_cluster_destroy(client);
            return fail_json(axiom_status_string(rc));
        }
        print_spawn_json(mode, transport, &info);
        axiom_cluster_destroy(client);
        return 0;
    }

    usage(stderr);
    return fail_json("invalid mode");
}
