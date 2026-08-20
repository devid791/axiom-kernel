#include "axiom/axiom.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static int run_stream_latent_smoke(axiom_transport_kind transport, int allow_unavailable) {
    int rc = AXIOM_OK;
    const char *failure = NULL;
    axiom_cluster *server = NULL;
    axiom_cluster *client = NULL;
    axiom_cluster *peer = NULL;
    axiom_cluster_info server_info;
    axiom_latent_shard send_shard;
    axiom_latent_shard recv_shard;
    uint64_t recv_bytes = 0;
    uint8_t payload[256];
    uint8_t recv_payload[256];

    memset(&server_info, 0, sizeof(server_info));
    memset(&send_shard, 0, sizeof(send_shard));
    memset(&recv_shard, 0, sizeof(recv_shard));
    memset(recv_payload, 0, sizeof(recv_payload));
    for (size_t i = 0; i < sizeof(payload); ++i) {
        payload[i] = (uint8_t)(i * 13u + 7u);
    }

    axiom_cluster_config server_config;
    memset(&server_config, 0, sizeof(server_config));
    server_config.abi_version = AXIOM_ABI_VERSION;
    server_config.transport = transport;
    server_config.listen_addr = "127.0.0.1:0";
    server_config.node_id = 0;
    server_config.node_count = 2;
    rc = axiom_cluster_create(&server, &server_config);
    if (allow_unavailable && (rc == AXIOM_ERR_NOT_IMPLEMENTED || rc == AXIOM_ERR_IO)) {
        return 0;
    }
    if (rc != AXIOM_OK) failure = "tcp server create";

    if (!failure) {
        server_info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_cluster_info_get(server, &server_info);
        if (rc != AXIOM_OK) failure = "tcp server info";
    }

    if (!failure) {
        axiom_cluster_config client_config;
        memset(&client_config, 0, sizeof(client_config));
        client_config.abi_version = AXIOM_ABI_VERSION;
        client_config.transport = transport;
        client_config.join_addr = server_info.listen_addr;
        client_config.node_id = 1;
        client_config.node_count = 2;
        rc = axiom_cluster_create(&client, &client_config);
        if (allow_unavailable && (rc == AXIOM_ERR_NOT_IMPLEMENTED || rc == AXIOM_ERR_IO)) {
            axiom_cluster_destroy(server);
            return 0;
        }
        if (rc != AXIOM_OK) failure = "stream client create";
    }

    if (!failure) {
        rc = axiom_cluster_accept(server, &peer);
        if (rc != AXIOM_OK) failure = "stream accept";
    }

    if (!failure) {
        send_shard.abi_version = AXIOM_ABI_VERSION;
        send_shard.dtype = AXIOM_LATENT_F32;
        send_shard.rows = 4;
        send_shard.cols = 16;
        send_shard.stride = 16;
        send_shard.source_node = 1;
        send_shard.source_device = 0;
        send_shard.shard_id = 0xdecafbad1234ull;
        send_shard.session_id = 0x51a7e0001ull;
        send_shard.payload_bytes = sizeof(payload);
        rc = axiom_cluster_send_latent(client, &send_shard, payload, sizeof(payload));
        if (rc != AXIOM_OK) failure = "stream latent send";
    }

    if (!failure) {
        recv_shard.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_cluster_recv_latent(
                peer,
                &recv_shard,
                recv_payload,
                sizeof(recv_payload),
                &recv_bytes);
        if (rc != AXIOM_OK) failure = "stream latent recv";
    }

    if (!failure &&
        (recv_bytes != sizeof(payload) ||
         recv_shard.dtype != AXIOM_LATENT_F32 ||
         recv_shard.rows != send_shard.rows ||
         recv_shard.cols != send_shard.cols ||
         recv_shard.stride != send_shard.stride ||
         recv_shard.shard_id != send_shard.shard_id ||
         recv_shard.session_id != send_shard.session_id ||
        memcmp(payload, recv_payload, sizeof(payload)) != 0)) {
        rc = AXIOM_ERR_RUNTIME;
        failure = "stream latent payload";
    }

    axiom_cluster_destroy(peer);
    axiom_cluster_destroy(client);
    axiom_cluster_destroy(server);
    if (failure) {
        fprintf(stderr, "axiom-smoke: %s failed transport=%d: %s\n",
                failure, (int)transport, axiom_status_string(rc));
        return 1;
    }
    return 0;
}

static int run_tcp_latent_smoke(void) {
    return run_stream_latent_smoke(AXIOM_TRANSPORT_TCP, 0);
}

static int run_rdma_latent_smoke(void) {
    return run_stream_latent_smoke(AXIOM_TRANSPORT_RDMA, 1);
}

static int run_quic_latent_smoke(void) {
    int rc = AXIOM_OK;
    axiom_cluster *server = NULL;
    axiom_cluster *client = NULL;
    axiom_cluster_info server_info;
    axiom_latent_shard send_shard;
    uint8_t payload[256];

    memset(&server_info, 0, sizeof(server_info));
    memset(&send_shard, 0, sizeof(send_shard));
    for (size_t i = 0; i < sizeof(payload); ++i) {
        payload[i] = (uint8_t)(i * 13u + 7u);
    }

    axiom_cluster_config server_config;
    memset(&server_config, 0, sizeof(server_config));
    server_config.abi_version = AXIOM_ABI_VERSION;
    server_config.transport = AXIOM_TRANSPORT_QUIC;
    server_config.listen_addr = "127.0.0.1:0";
    server_config.node_id = 0;
    server_config.node_count = 2;
    rc = axiom_cluster_create(&server, &server_config);
    if (rc == AXIOM_ERR_NOT_IMPLEMENTED) {
        return 0;
    }
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: quic server create failed: %s\n", axiom_status_string(rc));
        return 1;
    }

    server_info.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_cluster_info_get(server, &server_info);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: quic server info failed: %s\n", axiom_status_string(rc));
        axiom_cluster_destroy(server);
        return 1;
    }

    pid_t child = fork();
    if (child < 0) {
        fprintf(stderr, "axiom-smoke: quic fork failed\n");
        axiom_cluster_destroy(server);
        return 1;
    }
    if (child == 0) {
        axiom_cluster *peer = NULL;
        axiom_latent_shard recv_shard;
        uint64_t recv_bytes = 0;
        uint8_t recv_payload[256];
        memset(&recv_shard, 0, sizeof(recv_shard));
        memset(recv_payload, 0, sizeof(recv_payload));

        rc = axiom_cluster_accept(server, &peer);
        if (rc != AXIOM_OK) {
            fprintf(stderr, "axiom-smoke: quic accept failed: %s\n", axiom_status_string(rc));
            axiom_cluster_destroy(server);
            _exit(1);
        }
        recv_shard.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_cluster_recv_latent(
                peer,
                &recv_shard,
                recv_payload,
                sizeof(recv_payload),
                &recv_bytes);
        if (rc != AXIOM_OK ||
            recv_bytes != sizeof(payload) ||
            recv_shard.dtype != AXIOM_LATENT_F32 ||
            recv_shard.rows != 4 ||
            recv_shard.cols != 16 ||
            recv_shard.stride != 16 ||
            recv_shard.shard_id != 0xdecafbad1234ull ||
            recv_shard.session_id != 0x51a7e0001ull ||
            memcmp(payload, recv_payload, sizeof(payload)) != 0) {
            fprintf(stderr, "axiom-smoke: quic latent recv failed: %s\n", axiom_status_string(rc));
            axiom_cluster_destroy(peer);
            axiom_cluster_destroy(server);
            _exit(1);
        }
        axiom_cluster_destroy(peer);
        axiom_cluster_destroy(server);
        _exit(0);
    }

    axiom_cluster_config client_config;
    memset(&client_config, 0, sizeof(client_config));
    client_config.abi_version = AXIOM_ABI_VERSION;
    client_config.transport = AXIOM_TRANSPORT_QUIC;
    client_config.join_addr = server_info.listen_addr;
    client_config.node_id = 1;
    client_config.node_count = 2;
    rc = axiom_cluster_create(&client, &client_config);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: quic client create failed: %s\n", axiom_status_string(rc));
        kill(child, SIGTERM);
        (void)waitpid(child, NULL, 0);
        axiom_cluster_destroy(server);
        return 1;
    }

    send_shard.abi_version = AXIOM_ABI_VERSION;
    send_shard.dtype = AXIOM_LATENT_F32;
    send_shard.rows = 4;
    send_shard.cols = 16;
    send_shard.stride = 16;
    send_shard.source_node = 1;
    send_shard.source_device = 0;
    send_shard.shard_id = 0xdecafbad1234ull;
    send_shard.session_id = 0x51a7e0001ull;
    send_shard.payload_bytes = sizeof(payload);
    rc = axiom_cluster_send_latent(client, &send_shard, payload, sizeof(payload));
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: quic latent send failed: %s\n", axiom_status_string(rc));
        kill(child, SIGTERM);
        (void)waitpid(child, NULL, 0);
        axiom_cluster_destroy(client);
        axiom_cluster_destroy(server);
        return 1;
    }

    int status = 0;
    if (waitpid(child, &status, 0) < 0 ||
        !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0) {
        fprintf(stderr, "axiom-smoke: quic child failed\n");
        axiom_cluster_destroy(client);
        axiom_cluster_destroy(server);
        return 1;
    }
    axiom_cluster_destroy(client);
    axiom_cluster_destroy(server);
    return 0;
}

static int run_agent_spawn_smoke(void) {
    int rc = AXIOM_OK;
    const char *failure = NULL;
    axiom_cluster *server = NULL;
    axiom_cluster *client = NULL;
    axiom_cluster *peer = NULL;
    axiom_cluster_info server_info;
    axiom_agent_spawn_info spawn_info;

    memset(&server_info, 0, sizeof(server_info));
    memset(&spawn_info, 0, sizeof(spawn_info));

    axiom_cluster_config server_config;
    memset(&server_config, 0, sizeof(server_config));
    server_config.abi_version = AXIOM_ABI_VERSION;
    server_config.transport = AXIOM_TRANSPORT_TCP;
    server_config.listen_addr = "127.0.0.1:0";
    server_config.node_id = 0;
    server_config.node_count = 2;
    rc = axiom_cluster_create(&server, &server_config);
    if (rc != AXIOM_OK) failure = "agent server create";

    if (!failure) {
        server_info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_cluster_info_get(server, &server_info);
        if (rc != AXIOM_OK) failure = "agent server info";
    }

    if (!failure) {
        axiom_cluster_config client_config;
        memset(&client_config, 0, sizeof(client_config));
        client_config.abi_version = AXIOM_ABI_VERSION;
        client_config.transport = AXIOM_TRANSPORT_TCP;
        client_config.join_addr = server_info.listen_addr;
        client_config.node_id = 1;
        client_config.node_count = 2;
        rc = axiom_cluster_create(&client, &client_config);
        if (rc != AXIOM_OK) failure = "agent client create";
    }

    if (!failure) {
        rc = axiom_cluster_accept(server, &peer);
        if (rc != AXIOM_OK) failure = "agent accept";
    }

    if (!failure) {
        axiom_goal_config goal;
        memset(&goal, 0, sizeof(goal));
        goal.abi_version = AXIOM_ABI_VERSION;
        goal.goal_id = "goal-cluster-smoke";
        goal.objective = "coordinate latent-only model work across the mesh";
        goal.budget_tokens = 4096;
        rc = axiom_cluster_goal_begin(client, &goal);
        if (rc != AXIOM_OK) failure = "goal begin";
    }

    if (!failure) {
        axiom_agent_spawn_config spawn;
        memset(&spawn, 0, sizeof(spawn));
        spawn.abi_version = AXIOM_ABI_VERSION;
        spawn.kind = AXIOM_AGENT_MODEL_WORKER;
        spawn.agent_id = "worker-node0-device0";
        spawn.goal_id = "goal-cluster-smoke";
        spawn.role = "remote-model-worker";
        spawn.objective = "own assigned layer span and exchange latent shards";
        spawn.placement.abi_version = AXIOM_ABI_VERSION;
        spawn.placement.kind = AXIOM_PLACEMENT_CLUSTER_NODE;
        spawn.placement.node_id = 0;
        spawn.placement.device_id = 0;
        spawn.target_node = 0;
        spawn.target_device = 0;
        spawn.task_id = 0xA6100001ull;
        spawn.session_id = 0x51A7E0002ull;
        spawn.budget_tokens = 2048;
        spawn_info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_cluster_spawn_agent(client, &spawn, &spawn_info);
        if (rc != AXIOM_OK) failure = "agent spawn";
    }

    if (!failure) {
        memset(&spawn_info, 0, sizeof(spawn_info));
        spawn_info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_cluster_recv_agent_spawn(peer, &spawn_info);
        if (rc != AXIOM_OK) failure = "agent recv";
    }

    if (!failure &&
        (spawn_info.kind != AXIOM_AGENT_MODEL_WORKER ||
         spawn_info.status != AXIOM_AGENT_RECEIVED ||
         spawn_info.placement != AXIOM_PLACEMENT_CLUSTER_NODE ||
         spawn_info.source_node != 1 ||
         spawn_info.target_node != 0 ||
         spawn_info.target_device != 0 ||
         spawn_info.task_id != 0xA6100001ull ||
         spawn_info.session_id != 0x51A7E0002ull ||
         strcmp(spawn_info.agent_id, "worker-node0-device0") != 0 ||
         strcmp(spawn_info.goal_id, "goal-cluster-smoke") != 0 ||
         strcmp(spawn_info.role, "remote-model-worker") != 0)) {
        rc = AXIOM_ERR_RUNTIME;
        failure = "agent payload";
    }

    axiom_cluster_destroy(peer);
    axiom_cluster_destroy(client);
    axiom_cluster_destroy(server);
    if (failure) {
        fprintf(stderr, "axiom-smoke: %s failed: %s\n", failure, axiom_status_string(rc));
        return 1;
    }
    return 0;
}

int main(void) {
    axiom_config config;
    config.abi_version = AXIOM_ABI_VERSION;
    config.backend = AXIOM_BACKEND_CUDA;
    config.device = 0;
    config.flags = 0;

    axiom_runtime *runtime = NULL;
    int rc = axiom_runtime_create(&runtime, &config);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: create failed: %s\n", axiom_status_string(rc));
        return 1;
    }

    axiom_device_info info;
    rc = axiom_runtime_probe(runtime, &info);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: probe failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    printf("axiom-smoke: %s\n", axiom_version());
    printf("axiom-smoke: cuda device %s sm_%d%d sms=%d mem=%llu\n",
           info.name,
           info.major,
           info.minor,
           info.multi_processor_count,
           (unsigned long long)info.total_global_mem);

    const size_t n = 4096;
    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *out = (float *)malloc(n * sizeof(float));
    float *copy_out = (float *)malloc(n * sizeof(float));
    if (!a || !b || !out || !copy_out) {
        fprintf(stderr, "axiom-smoke: allocation failed\n");
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.25f;
        b[i] = 7.0f - (float)i * 0.125f;
        out[i] = 0.0f;
        copy_out[i] = 0.0f;
    }

    rc = axiom_smoke_vector_add(runtime, a, b, out, n);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: vector add failed: %s\n", axiom_status_string(rc));
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    float max_abs = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float expected = a[i] + b[i];
        const float err = fabsf(out[i] - expected);
        if (err > max_abs) max_abs = err;
    }

    uint32_t device_count = 0;
    rc = axiom_runtime_device_count(&device_count);
    if (rc != AXIOM_OK || device_count == 0) {
        fprintf(stderr, "axiom-smoke: device count failed: %s count=%u\n",
                axiom_status_string(rc), device_count);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    uint32_t runtime_device = UINT32_MAX;
    rc = axiom_runtime_device_id(runtime, &runtime_device);
    if (rc != AXIOM_OK || runtime_device != config.device) {
        fprintf(stderr, "axiom-smoke: runtime device id failed: %s device=%u\n",
                axiom_status_string(rc), runtime_device);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_device_buffer *src_dev = NULL;
    axiom_device_buffer *dst_dev = NULL;
    rc = axiom_device_buffer_create(runtime, &src_dev, n * sizeof(float));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(runtime, &dst_dev, n * sizeof(float)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(src_dev, 0, out, n * sizeof(float)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_copy(dst_dev, 0, src_dev, 0, n * sizeof(float)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(dst_dev, 0, copy_out, n * sizeof(float)) : rc;
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: device copy failed: %s\n", axiom_status_string(rc));
        axiom_device_buffer_destroy(dst_dev);
        axiom_device_buffer_destroy(src_dev);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    for (size_t i = 0; i < n; i++) {
        const float err = fabsf(copy_out[i] - out[i]);
        if (err > max_abs) max_abs = err;
    }

    uint32_t src_device = UINT32_MAX;
    rc = axiom_device_buffer_device_id(src_dev, &src_device);
    if (rc != AXIOM_OK || src_device != runtime_device) {
        fprintf(stderr, "axiom-smoke: buffer device id failed: %s device=%u\n",
                axiom_status_string(rc), src_device);
        axiom_device_buffer_destroy(dst_dev);
        axiom_device_buffer_destroy(src_dev);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_cluster_config cluster_config;
    cluster_config.abi_version = AXIOM_ABI_VERSION;
    cluster_config.transport = AXIOM_TRANSPORT_INPROC;
    cluster_config.listen_addr = NULL;
    cluster_config.join_addr = NULL;
    cluster_config.node_id = 0;
    cluster_config.node_count = 1;
    cluster_config.flags = 0;
    axiom_cluster *cluster = NULL;
    rc = axiom_cluster_create(&cluster, &cluster_config);
    rc = rc == AXIOM_OK ? axiom_runtime_attach_cluster(runtime, cluster) : rc;
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: local cluster attach failed: %s\n", axiom_status_string(rc));
        axiom_cluster_destroy(cluster);
        axiom_device_buffer_destroy(dst_dev);
        axiom_device_buffer_destroy(src_dev);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    if (run_tcp_latent_smoke() != 0) {
        axiom_cluster_destroy(cluster);
        axiom_device_buffer_destroy(dst_dev);
        axiom_device_buffer_destroy(src_dev);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    if (run_rdma_latent_smoke() != 0) {
        axiom_cluster_destroy(cluster);
        axiom_device_buffer_destroy(dst_dev);
        axiom_device_buffer_destroy(src_dev);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    if (run_quic_latent_smoke() != 0) {
        axiom_cluster_destroy(cluster);
        axiom_device_buffer_destroy(dst_dev);
        axiom_device_buffer_destroy(src_dev);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    if (run_agent_spawn_smoke() != 0) {
        axiom_cluster_destroy(cluster);
        axiom_device_buffer_destroy(dst_dev);
        axiom_device_buffer_destroy(src_dev);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_runtime *same_device_peer = NULL;
    axiom_config same_device_config = config;
    rc = axiom_runtime_create(&same_device_peer, &same_device_config);
    axiom_device_buffer *same_device_dst = NULL;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(same_device_peer, &same_device_dst, n * sizeof(float)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_copy(same_device_dst, 0, src_dev, 0, n * sizeof(float)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(same_device_dst, 0, copy_out, n * sizeof(float)) : rc;
    axiom_device_buffer_destroy(same_device_dst);
    axiom_runtime_destroy(same_device_peer);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: same-device peer runtime copy failed: %s\n",
                axiom_status_string(rc));
        axiom_cluster_destroy(cluster);
        axiom_device_buffer_destroy(dst_dev);
        axiom_device_buffer_destroy(src_dev);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    for (size_t i = 0; i < n; i++) {
        const float err = fabsf(copy_out[i] - out[i]);
        if (err > max_abs) max_abs = err;
    }

    if (device_count > 1) {
        axiom_runtime *remote_runtime = NULL;
        axiom_config remote_config = config;
        remote_config.device = 1;
        rc = axiom_runtime_create(&remote_runtime, &remote_config);
        axiom_device_buffer *remote_dst = NULL;
        rc = rc == AXIOM_OK ? axiom_device_buffer_create(remote_runtime, &remote_dst, n * sizeof(float)) : rc;
        rc = rc == AXIOM_OK ? axiom_device_buffer_copy(remote_dst, 0, src_dev, 0, n * sizeof(float)) : rc;
        rc = rc == AXIOM_OK ? axiom_device_buffer_download(remote_dst, 0, copy_out, n * sizeof(float)) : rc;
        axiom_device_buffer_destroy(remote_dst);
        axiom_runtime_destroy(remote_runtime);
        if (rc != AXIOM_OK) {
            fprintf(stderr, "axiom-smoke: cross-device peer copy failed: %s\n",
                    axiom_status_string(rc));
            axiom_cluster_destroy(cluster);
            axiom_device_buffer_destroy(dst_dev);
            axiom_device_buffer_destroy(src_dev);
            free(copy_out);
            free(out);
            free(b);
            free(a);
            axiom_runtime_destroy(runtime);
            return 1;
        }
        for (size_t i = 0; i < n; i++) {
            const float err = fabsf(copy_out[i] - out[i]);
            if (err > max_abs) max_abs = err;
        }
    }

    axiom_device_buffer_destroy(dst_dev);
    axiom_device_buffer_destroy(src_dev);

    enum { output_hidden = 8, output_stream_count = 4 * output_hidden, output_fn_count = 4 * output_stream_count };
    uint16_t output_fn[output_fn_count] = {0};
    float output_scale[1] = {0.0f};
    float output_base[4] = {0.0f};
    float output_streams[output_stream_count];
    float output_hc[output_hidden];
    for (size_t i = 0; i < output_stream_count; ++i) {
        output_streams[i] = (float)((int)(i % 9) - 4) * 0.125f;
    }
    for (size_t i = 0; i < output_hidden; ++i) {
        output_hc[i] = 0.0f;
    }

    axiom_device_buffer *output_fn_dev = NULL;
    axiom_device_buffer *output_scale_dev = NULL;
    axiom_device_buffer *output_base_dev = NULL;
    axiom_device_buffer *output_streams_dev = NULL;
    axiom_device_buffer *output_hc_dev = NULL;
    rc = axiom_device_buffer_create(runtime, &output_fn_dev, sizeof(output_fn));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(runtime, &output_scale_dev, sizeof(output_scale)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(runtime, &output_base_dev, sizeof(output_base)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(runtime, &output_streams_dev, sizeof(output_streams)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(runtime, &output_hc_dev, sizeof(output_hc)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(output_fn_dev, 0, output_fn, sizeof(output_fn)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(output_scale_dev, 0, output_scale, sizeof(output_scale)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(output_base_dev, 0, output_base, sizeof(output_base)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(output_streams_dev, 0, output_streams, sizeof(output_streams)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_output_hc_f32_device(
            runtime,
            output_fn_dev, 0,
            output_scale_dev, 0,
            output_base_dev, 0,
            output_streams_dev, 0,
            output_hc_dev, 0,
            output_hidden,
            1.0e-6f) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(output_hc_dev, 0, output_hc, sizeof(output_hc)) : rc;
    axiom_device_buffer_destroy(output_hc_dev);
    axiom_device_buffer_destroy(output_streams_dev);
    axiom_device_buffer_destroy(output_base_dev);
    axiom_device_buffer_destroy(output_scale_dev);
    axiom_device_buffer_destroy(output_fn_dev);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-smoke: output HC failed: %s\n", axiom_status_string(rc));
        axiom_cluster_destroy(cluster);
        free(copy_out);
        free(out);
        free(b);
        free(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    for (size_t i = 0; i < output_hidden; ++i) {
        float expected = 0.0f;
        for (size_t h = 0; h < 4; ++h) {
            expected += 0.500001f * output_streams[h * output_hidden + i];
        }
        const float err = fabsf(output_hc[i] - expected);
        if (err > max_abs) max_abs = err;
    }

    free(copy_out);
    free(out);
    free(b);
    free(a);
    axiom_runtime_destroy(runtime);
    axiom_cluster_destroy(cluster);

    if (max_abs > 0.0001f) {
        fprintf(stderr, "axiom-smoke: mismatch max_abs=%f\n", (double)max_abs);
        return 1;
    }

    puts("axiom-smoke: OK");
    return 0;
}
