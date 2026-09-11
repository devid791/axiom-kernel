#pragma once

// Request-local, target-only known-token prefill. No proposal/draft work and
// no new CUDA kernels. Reuses the qualified device verifier and existing
// position/history materialization. Caller holds the generation mutex.
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_dspark_compute.h"

class axiom_codex_prefill_graph {
    axiom_qwen38_model *model_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t verify_ = nullptr, commit_ = nullptr;
    uint32_t *inputs_ = nullptr; // eight tokens, position, constant commit width
    axiom_qwen38_model_dspark_device_target_verify_view view_{};
    uint32_t position_ = 0, limit_ = 0;
    bool owns_device_ = false;

    static int status(cudaError_t rc) {
        return rc == cudaSuccess ? AXIOM_OK :
            rc == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
    }
    int end_capture(cudaGraphExec_t *exec, int prior) {
        cudaGraph_t graph = nullptr;
        const auto ended = cudaStreamEndCapture(stream_, &graph);
        int rc = prior == AXIOM_OK ? status(ended) : prior;
        if (rc == AXIOM_OK && !graph) rc = AXIOM_ERR_CUDA;
        if (rc == AXIOM_OK) rc = status(cudaGraphInstantiate(exec, graph, 0));
        if (graph) (void)cudaGraphDestroy(graph);
        return rc;
    }

public:
    axiom_codex_prefill_graph() = default;
    axiom_codex_prefill_graph(const axiom_codex_prefill_graph &) = delete;
    axiom_codex_prefill_graph &operator=(const axiom_codex_prefill_graph &) = delete;
    ~axiom_codex_prefill_graph() {
        // The owner must finish on success or reset/restore the request on
        // failure. Never commit implicitly from a destructor.
        if (stream_) (void)cudaStreamSynchronize(stream_);
        if (verify_) (void)cudaGraphExecDestroy(verify_);
        if (commit_) (void)cudaGraphExecDestroy(commit_);
        if (inputs_) (void)cudaFree(inputs_);
        if (stream_) (void)cudaStreamDestroy(stream_);
    }
    bool ready() const { return verify_ && commit_; }
    uint32_t position() const { return position_; }

    int prepare(axiom_qwen38_model *model, uint32_t hot_limit) {
        if (model_ || !model || hot_limit == 0 || hot_limit > 8192u)
            return AXIOM_ERR_INVALID_ARGUMENT;
        axiom_qwen38_model_dspark_temporal_capabilities caps{};
        int rc = axiom_qwen38_model_dspark_temporal_capabilities_get(model, &caps);
        if (rc != AXIOM_OK || !caps.temporal_m8_available)
            return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
        model_ = model; position_ = axiom_qwen38_model_position(model); limit_ = hot_limit;
        if (position_ > limit_ || limit_ - position_ < 8u) return AXIOM_ERR_BUDGET;
        // Complete any work on the caller's default stream before switching
        // to this request-local nonblocking stream.
        rc = status(cudaSetDevice(axiom_qwen38_model_device(model_)));
        if (rc == AXIOM_OK) rc = status(cudaStreamSynchronize(cudaStreamPerThread));
        if (rc == AXIOM_OK) rc = status(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        if (rc == AXIOM_OK) rc = status(cudaMalloc(reinterpret_cast<void **>(&inputs_), 10u * sizeof(uint32_t)));
        if (rc != AXIOM_OK) return rc;
        rc = status(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        if (rc != AXIOM_OK) return rc;
        bool transaction_open = false;
        rc = axiom_qwen38_model_dspark_device_transaction_begin(model_, inputs_ + 8u, stream_);
        if (rc == AXIOM_OK) {
            transaction_open = true;
            rc = axiom_qwen38_model_dspark_device_transaction_verify(
                    model_, inputs_, inputs_ + 8u, 8u, stream_, &view_);
        }
        rc = end_capture(&verify_, rc);
        // Commit is a SEPARATE graph. Validate every output before installing
        // recurrent state; a bad logit cannot be committed speculatively.
        if (rc == AXIOM_OK) {
            rc = status(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            if (rc == AXIOM_OK) {
                rc = axiom_qwen38_model_dspark_device_transaction_commit(model_, inputs_ + 9u, stream_);
                transaction_open = false; // valid commit consumes its handle
                rc = end_capture(&commit_, rc);
            }
        }
        if (transaction_open) {
            const int abort_rc = axiom_qwen38_model_dspark_device_transaction_abort(model_, stream_);
            if (abort_rc != AXIOM_OK) rc = abort_rc;
        }
        if (rc == AXIOM_OK && (!view_.target_token_ids_device || !view_.target_logits_device ||
                !view_.target_taps_device || view_.target_tap_tokens != 8u ||
                view_.target_tap_hidden_size != AXIOM_QWEN38_DSPARK_HIDDEN ||
                view_.target_tap_count != AXIOM_QWEN38_DSPARK_TARGET_FEATURES))
            rc = AXIOM_ERR_RUNTIME;
        return rc;
    }

    int step(const uint32_t *tokens, uint32_t *next, float *logit,
             axiom_qwen38_dspark_compute *draft = nullptr) {
        if (!ready() || !tokens || !next || !logit || position_ > limit_ || limit_ - position_ < 8u)
            return AXIOM_ERR_INVALID_ARGUMENT;
        uint32_t host[10]{};
        for (unsigned i = 0; i < 8; ++i) {
            if (tokens[i] >= AXIOM_QWEN38_MODEL_VOCAB) return AXIOM_ERR_INVALID_ARGUMENT;
            host[i] = tokens[i];
        }
        host[8] = position_; host[9] = 8u;
        if (draft) {
            axiom_qwen38_dspark_compute_info info{};
            info.abi_version = AXIOM_ABI_VERSION;
            const int check = axiom_qwen38_dspark_compute_info_get(draft, &info);
            if (check != AXIOM_OK) return check;
            if (info.transaction_open || info.committed_position != position_) return AXIOM_ERR_RUNTIME;
        }
        int rc = status(cudaMemcpyAsync(inputs_, host, sizeof(host), cudaMemcpyHostToDevice, stream_));
        if (rc == AXIOM_OK && !owns_device_) {
            rc = axiom_qwen38_model_dspark_device_session_begin(model_);
            owns_device_ = rc == AXIOM_OK;
        }
        if (rc == AXIOM_OK) rc = status(cudaGraphLaunch(verify_, stream_));
        uint32_t ids[8]{}; float values[8]{};
        if (rc == AXIOM_OK) rc = status(cudaMemcpyAsync(ids, view_.target_token_ids_device,
                sizeof(ids), cudaMemcpyDeviceToHost, stream_));
        if (rc == AXIOM_OK) rc = status(cudaMemcpyAsync(values, view_.target_logits_device,
                sizeof(values), cudaMemcpyDeviceToHost, stream_));
        // Always drain before stack buffers die, even on a partial enqueue.
        const int sync = status(cudaStreamSynchronize(stream_));
        if (rc == AXIOM_OK) rc = sync;
        if (rc != AXIOM_OK) return rc;
        for (unsigned i = 0; i < 8; ++i)
            if (ids[i] >= AXIOM_QWEN38_MODEL_VOCAB || !std::isfinite(values[i])) return AXIOM_ERR_CUDA;
        if (draft) {
            // Exactly the existing DSpark prefill injection, with the same
            // taps/position/width, ordered before the unchanged target commit.
            axiom_qwen38_dspark_compute_inject_request inject{};
            inject.abi_version = AXIOM_ABI_VERSION;
            inject.target_taps = view_.target_taps_device;
            inject.target_position = position_; inject.columns = 8u; inject.stream = stream_;
            rc = axiom_qwen38_dspark_compute_inject_target(draft, &inject);
            axiom_qwen38_dspark_compute_info info{}; info.abi_version = AXIOM_ABI_VERSION;
            if (rc == AXIOM_OK) rc = axiom_qwen38_dspark_compute_info_get(draft, &info);
            if (rc == AXIOM_OK && (info.transaction_open || info.committed_position != position_ + 8u))
                rc = AXIOM_ERR_RUNTIME;
            if (rc != AXIOM_OK) { (void)cudaStreamSynchronize(stream_); return rc; }
        }
        rc = status(cudaGraphLaunch(commit_, stream_));
        // Commit stays ordered before the next replay or materialization.
        if (rc == AXIOM_OK) { position_ += 8u; *next = ids[7]; *logit = values[7]; }
        return rc;
    }

    int finish(const uint32_t *full_history, uint32_t count) {
        if (!owns_device_) return AXIOM_OK;
        if (count != position_ || (count && !full_history)) return AXIOM_ERR_INVALID_ARGUMENT;
        int rc = status(cudaStreamSynchronize(stream_));
        if (rc == AXIOM_OK) rc = axiom_qwen38_model_restore_position(model_, position_);
        if (rc == AXIOM_OK) rc = axiom_qwen38_model_committed_history_install(model_, full_history, count);
        if (rc == AXIOM_OK) owns_device_ = false;
        return rc;
    }
};
