/* Minimal OpenAI-compatible local API for Qwen3.8-27B-NVFP4 + DSpark.
 *
 * It deliberately has no authentication layer: bind it only to explicit
 * loopback and intended LAN addresses. The immutable target and DSpark weights are persistent. The
 * HTTP layer accepts independent client sessions, while the mutable target/KV
 * state is guarded by one serialized native generation scheduler. The decode
 * hot path is exclusively the fixed gamma-7 full-device CUDA graph whenever
 * the request profile permits it; an unavailable graph or bad device status is
 * returned to the client as an explicit server error.
 */

#include <cuda_runtime_api.h>
#include "axiom/qwen38_prefix_reuse.hpp"
#include "axiom/qwen38_request_progress.hpp"

#include <arpa/inet.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <condition_variable>
#include <deque>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_dspark.h"
#include "axiom/qwen38_dspark_compute.h"
#include "axiom/qwen38_kv_tier.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_mtp.h"
#include "axiom/qwen38_mtp_compute.h"
#include "axiom/qwen38_mtp_speculative.h"
#include "axiom/qwen38_session_store.hpp"
#include "axiom/qwen38_spec_identity.hpp"
#include "axiom/qwen38_speculative.h"
#include "axiom/qwen38_swarm_scheduler.hpp"
#include "axiom/qwen38_video_decode.hpp"
#include "axiom/qwen38_vision.h"
#include "axiom/vision_memory_budget.hpp"
#include "axiom/qwen38_vision_preprocess.hpp"
#include "axiom/reasoning_profiles.h"
#include "axiom_aliced_json.h"
#include "axiom_codex_provider.h"
#include "axiom_codex_prefill_graph.h"

namespace {

constexpr char kModelId[] = "qwen3.8-27b-nvfp4";
constexpr char kStartupSessionId[] = "axiom-system-startup";
constexpr char kSessionHeader[] = "X-Axiom-Session-ID";
/* v2 makes the durable cursor contract explicit: committed_tokens always
 * describes the exact token/recurrent/KV watermark, while next_token is the
 * predicted, not-yet-committed continuation.  Keeping the version in the
 * namespace prevents a v1 manifest from being interpreted with v2 semantics. */
constexpr char kSessionConfigVersion[] = "session-kv-v2";
// Prompt bytes are part of the recurrent/KV state contract. Never extend a
// pre-template-fix native session through the stateful-tail fallback. Existing
// files remain intact; clients replay their history once into the new namespace.
constexpr char kChatTemplateVersion[] = "qwen38-chatml-v1";
constexpr uint32_t kMaxNativeContext = 1024u * 1024u;
constexpr uint32_t kDefaultRequestContext = 262144u;
/* The process owns the full logical ceiling so an explicit request can opt
 * into 1M.  Requests without a selector stay on the proven 256K window. */
constexpr uint32_t kDefaultMaxContext = kMaxNativeContext;
constexpr uint32_t kDefaultMaxNew = 128u;
constexpr uint32_t kDefaultThinkingBudget =
        axiom::reasoning::kProfiles[3].thinking_budget_tokens;
constexpr uint32_t kDefaultVisibleBudget = 4096u;
constexpr uint32_t kDefaultQwenTopK = axiom::reasoning::kProfiles[3].top_k;
constexpr uint32_t kNonThinkingTopK = axiom::reasoning::kProfiles[0].top_k;
constexpr float kThinkingTemperature = axiom::reasoning::kProfiles[3].temperature;
constexpr float kThinkingTopP = axiom::reasoning::kProfiles[3].top_p;
/* Ultra-fast is the production default: the hard no-think path uses the
 * deterministic DSpark/M8 greedy graph.  Sampling remains available through
 * explicit per-request overrides or the operator profile controls. */
constexpr float kNonThinkingTemperature = axiom::reasoning::kProfiles[0].temperature;
constexpr float kNonThinkingTopP = axiom::reasoning::kProfiles[0].top_p;
constexpr char kNoThinkingPrefix[] = "<think>\n\n</think>\n\n";
constexpr char kThinkingEarlyStopText[] =
        "\n\n Considering the limited time by the user, I have to give the solution "
        "based on the thinking directly now.\n</think>\n\n";
constexpr size_t kMaxHeaderBytes = 64u * 1024u;
/* A 1M-token request is commonly tens of MiB after JSON encoding. The
 * tokenizer/model still enforce the real token budget below. */
constexpr size_t kMaxBodyBytes = 64u * 1024u * 1024u;
/* The logical conversation budget is 1M, while the proven Alice context
 * governor compacts old multi-turn history at the model's native 256K
 * boundary.  This is prompt/history compaction, not lossy KV-page rewriting:
 * one-shot documents can still exercise the real paged 1M path explicitly. */
constexpr uint32_t kContextCompactionBoundary = 262144u;
constexpr uint32_t kContextCompactionMaxOutputReserve = 4096u;
constexpr uint32_t kContextCompactionBuffer = 8000u;
constexpr uint32_t kContextCompactionRecentTokens = 12000u;
constexpr uint32_t kContextCompactionMaxSummaryChars = 12000u;
constexpr char kBackendId[] = "axiom-native-speculative-graph";
/* Keep the validated non-streaming graph chunk.  SSE uses the separate
 * tunable micro-batch below so first-token latency remains bounded. */
constexpr uint32_t kDeviceChunkCycles = 33u;
constexpr uint32_t kDefaultSseBatchCycles = 2u;
constexpr uint32_t kKvTierDefaultHotPages = 8u;
constexpr uint32_t kKvTierQueueDepth = 64u;
/* Bound durable chat state without turning persistence into a disk leak.
 * Recently used sessions survive restarts; inactive state ages out after one
 * week and LRU pressure caps the namespace count. Both values are operator
 * configurable, while zero explicitly disables the corresponding rule. */
constexpr uint64_t kDefaultSessionTtlSeconds = 7u * 24u * 60u * 60u;
constexpr uint32_t kDefaultSessionMaxNamespaces = 128u;
constexpr uint64_t kDefaultSessionGcIntervalSeconds = 5u * 60u;
constexpr uint32_t kDefaultVisionMaxPatchTokens = 4096u;
constexpr uint32_t kVisionMaxFrames = 64u;
constexpr uint32_t kVisionMaxDecodeWidth = 4096u;
constexpr uint32_t kVisionMaxDecodeHeight = 4096u;
constexpr uint64_t kVisionMaxFramePixels = 25165824ull;
constexpr size_t kSwarmQueueDepth = 64u;
constexpr size_t kSwarmEventHistory = 4096u;
constexpr size_t kSwarmMetadataHistory = 4096u;
constexpr size_t kMaxListeners = 8u;
constexpr char kDefaultListenSpec[] =
        "127.0.0.1:8015";
/* Default DSpark/M8 resident window.  The deployed value is operator-selected
 * through AXIOM_QWEN38_SPECULATIVE_CONTEXT_TOKENS after a VRAM/runtime gate;
 * it is never inferred from max_tokens. */
constexpr uint32_t kDefaultSpeculativeContextTokens =
        8u * AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
constexpr uint32_t kMinSpeculativeContextTokens =
        8u * AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
constexpr uint32_t kMaxSpeculativeContextTokens =
        32u * AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
constexpr uint32_t kValidationTokens[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH] = {
        271u, 248068u, 198u, 760u, 1156u, 369u, 40719u, 728u,
};

volatile sig_atomic_t g_stop = 0;

void on_signal(int) {
    g_stop = 1;
}
struct listen_spec {
    std::string host;
    uint16_t port = 0u;
};

struct http_request {
    std::string method;
    std::string path;
    std::string body;
    std::string session_id;
};

struct generation_result {
    std::string text;
    std::vector<uint32_t> ids;
    std::string finish_reason;
    uint32_t prompt_tokens = 0u;
    uint64_t graph_cycles = 0u;
    uint64_t graph_emitted_tokens = 0u;
    uint64_t canonical_tail_tokens = 0u;
    uint64_t scalar_tail_tokens = 0u;
    uint64_t accepted_draft_tokens = 0u;
    uint64_t proposed_draft_tokens = 0u;
    /* Draft slots that could legally become visible before max_tokens.  The
     * fixed graph may still compute seven proposals in its final capped
     * replay; acceptance quality must not count intentionally ineligible
     * slots as model rejections. */
    uint64_t eligible_draft_tokens = 0u;
    /* Prefix-valid calibration observations from DSpark's trained confidence
     * head.  Accepted draft slots and the first rejected slot are observable;
     * slots after the first mismatch are deliberately excluded because their
     * target rows are conditioned on an already divergent draft prefix. */
    uint64_t draft_confidence_samples = 0u;
    uint64_t draft_confidence_accepted_samples = 0u;
    uint64_t draft_confidence_rejected_samples = 0u;
    double draft_confidence_sum = 0.0;
    double draft_confidence_accepted_sum = 0.0;
    double draft_confidence_rejected_sum = 0.0;
    double draft_confidence_brier_sum = 0.0;
    double draft_confidence_min = 1.0;
    double draft_confidence_max = 0.0;
    std::array<uint64_t, 10> draft_confidence_bin_samples{};
    std::array<uint64_t, 10> draft_confidence_bin_accepted{};
    std::array<double, 10> draft_confidence_bin_sum{};
    double decode_seconds = 0.0;
    double decode_tokens_per_second = 0.0;
    double visible_tokens_per_second = 0.0;
    double acceptance = 0.0;
    double session_acquire_seconds = 0.0;
    double prefill_seconds = 0.0;
    double ttft_seconds = 0.0;
    uint32_t speculative_context_tokens = 0u;
    uint32_t prefix_hit_tokens = 0u;
    uint32_t suffix_prefill_tokens = 0u;
    uint32_t prefill_predictions_skipped = 0u;
    uint32_t paged_prefill_tokens = 0u;
    std::string decode_path;
    std::string fallback_reason;
    std::string speculative_mode_requested = "auto";
    std::string speculative_mode_effective;
    uint32_t speculative_max_commit_tokens = AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
    bool graph_executor_reused = false;
    uint32_t original_prompt_tokens = 0u;
    bool context_compacted = false;
    uint32_t thinking_tokens = 0u;
    uint32_t visible_output_tokens = 0u;
    uint32_t thinking_budget = 0u;
    std::string reasoning_effort;
    std::string session_id;
    /* Immutable target+draft+tokenizer+runtime pairing proven at process
     * startup. Copy it into each result so every wire protocol, including
     * final SSE events, carries the same attributable execution identity. */
    std::string speculative_fingerprint;
    bool speculative_qualified = false;
    std::string speculative_qualification_evidence_sha256;
    std::string speculative_qualification_source_commit;
    /* Position/token that were durable when the generation returned. These
     * are consumed by the manifest publisher before the model is reset. */
    uint32_t session_position = 0u;
    uint32_t session_next_token = 0u;
    /* Per-request proof.  Global restore counters remain useful telemetry but
     * cannot prove that this response, rather than a concurrent request, used
     * a durable prefix. */
    bool session_restored = false;
    bool session_stateful_resume = false;
    bool session_prefix_retokenized = false;
    bool session_tool_projection = false;
};

// Bound the visible suffix even when the first reasoning pass also generates
// the answer. Apply this BEFORE scheduling scalar steps or speculative batches;
// trimming returned IDs afterwards would leave recurrent/KV state ahead of the
// history that is actually returned to the client.
struct generation_output_budget {
    std::vector<uint32_t> reasoning_end_marker;
    uint32_t visible_limit = 0u;
    uint32_t visible_count = 0u;
    bool reasoning_ended = false;

    uint32_t allowance(uint32_t remaining) const {
        if (reasoning_end_marker.empty()) return remaining;
        if (reasoning_ended) {
            return std::min(remaining, visible_count >= visible_limit
                    ? 0u : visible_limit - visible_count);
        }
        // The next token could close reasoning (including a split marker).
        // At most visible_limit further tokens may be committed in this batch.
        return static_cast<uint32_t>(std::min<uint64_t>(
                remaining, static_cast<uint64_t>(visible_limit) + 1u));
    }

    void observe(const std::vector<uint32_t> &ids) {
        if (reasoning_end_marker.empty()) return;
        if (reasoning_ended) {
            ++visible_count;
        } else if (ids.size() >= reasoning_end_marker.size() &&
                std::equal(reasoning_end_marker.rbegin(), reasoning_end_marker.rend(),
                           ids.rbegin())) {
            reasoning_ended = true;
        }
    }
};

/* Optional wire sink owned by the HTTP handler.  The native decode paths
 * stay transport-agnostic: they append each committed token to the result
 * and, when present, hand only its decoded piece to this callback. */
struct generation_stream_sink {
    std::shared_ptr<axiom::qwen38::request_progress> progress;
    void *user = nullptr;
    bool (*emit)(void *user, const std::string &piece) = nullptr;
    /* Polled between prefill/decode units so a cancelled HTTP client cannot
     * leave the serialized native model busy until the socket timeout. */
    bool (*cancelled)(void *user) = nullptr;
    /* Streaming uses user for its emitter.  Cancellation may need a
     * scheduler/deadline probe instead, so keep the two callbacks independent
     * without leaking transport details into native generation. */
    void *cancel_user = nullptr;
    /* Qwen's byte-level tokenizer may split one UTF-8 code point across token
     * boundaries. Keep an incomplete suffix here so every SSE JSON event is
     * independently valid UTF-8 rather than emitting arbitrary token bytes. */
    std::string utf8_pending;
    // Only for a natural reasoning close: consume the template's exact two-LF
    // separator before emitting text, never arbitrary answer whitespace.
    bool reasoning_separator_pending = false;
    // Provider-specific optimization opt-in; legacy callers retain their
    // existing target prefill selection.
    bool codex_target_prefill = false;
    // Request-owned native catalog: permits ONLY reconstruction of the same
    // Core wire projection, never approximate tail/history replacement.
    const ajson *codex_replay_tools = nullptr;
    generation_output_budget output_budget;
};

uint32_t generation_tokens_remaining(
        const generation_result &result, uint32_t max_new,
        const generation_stream_sink *sink) {
    if (result.ids.size() >= max_new) return 0u;
    const uint32_t remaining = max_new - static_cast<uint32_t>(result.ids.size());
    return sink ? sink->output_budget.allowance(remaining) : remaining;
}

bool generation_cancelled(const generation_stream_sink *sink) {
    if (!sink || !sink->cancelled) return false;
    return sink->cancelled(sink->cancel_user ? sink->cancel_user : sink->user);
}

struct sampling_params {
    bool enabled = false;
    float temperature = 0.0f;
    uint32_t top_k = 0u;
    float top_p = 1.0f;
    uint64_t rng = 0x9e3779b97f4a7c15ull;
};

/* Request-scoped decode selection. `automatic` preserves the production
 * routing contract. The explicit modes are qualification controls and never
 * silently substitute one another, so graph/target A/B data is attributable. */
enum class speculative_mode {
    automatic,
    dspark,
    native_mtp,
    target,
};

const char *speculative_mode_name(const speculative_mode mode) {
    switch (mode) {
        case speculative_mode::automatic: return "auto";
        case speculative_mode::dspark: return "dspark";
        case speculative_mode::native_mtp: return "native_mtp";
        case speculative_mode::target: return "target";
    }
    return "invalid";
}

enum class speculative_engine {
    dspark,
    native_mtp,
};

const char *speculative_engine_name(const speculative_engine engine) {
    switch (engine) {
        case speculative_engine::dspark: return "dspark";
        case speculative_engine::native_mtp: return "native_mtp";
    }
    return "invalid";
}

struct thinking_plan {
    bool enabled = false;
    uint32_t thinking_budget = 0u;
    uint32_t visible_budget = 0u;
};

struct resource_snapshot {
    uint64_t process_rss_bytes = 0u;
    uint64_t process_peak_rss_bytes = 0u;
    uint64_t gpu_free_bytes = 0u;
    uint64_t gpu_total_bytes = 0u;
    uint64_t kv_submitted_reads = 0u;
    uint64_t kv_submitted_writes = 0u;
    uint64_t kv_completed_bytes = 0u;
    uint64_t kv_io_errors = 0u;
    uint32_t kv_committed_tokens = 0u;
    bool gpu_available = false;
    bool kv_available = false;
};

struct token_telemetry {
    bool available = false;
    uint64_t request_sequence = 0u;
    std::string session_id;
    uint32_t prompt_tokens = 0u;
    uint32_t completion_tokens = 0u;
    uint32_t thinking_tokens = 0u;
    uint32_t visible_output_tokens = 0u;
    uint32_t total_tokens = 0u;
    uint32_t context_window = 0u;
    uint32_t thinking_budget = 0u;
    std::string reasoning_effort;
    uint32_t original_prompt_tokens = 0u;
    double request_seconds = 0.0;
    double session_acquire_seconds = 0.0;
    double decode_seconds = 0.0;
    double decode_tokens_per_second = 0.0;
    double visible_tokens_per_second = 0.0;
    double prefill_seconds = 0.0;
    double ttft_seconds = 0.0;
    uint32_t speculative_context_tokens = 0u;
    uint32_t prefix_hit_tokens = 0u;
    uint32_t suffix_prefill_tokens = 0u;
    uint64_t graph_cycles = 0u;
    uint64_t graph_emitted_tokens = 0u;
    uint64_t scalar_tail_tokens = 0u;
    uint64_t accepted_draft_tokens = 0u;
    uint64_t eligible_draft_tokens = 0u;
    double speculative_acceptance = 0.0;
    uint64_t draft_confidence_samples = 0u;
    uint64_t draft_confidence_accepted_samples = 0u;
    uint64_t draft_confidence_rejected_samples = 0u;
    double draft_confidence_sum = 0.0;
    double draft_confidence_accepted_sum = 0.0;
    double draft_confidence_rejected_sum = 0.0;
    double draft_confidence_brier_sum = 0.0;
    double draft_confidence_min = 1.0;
    double draft_confidence_max = 0.0;
    std::array<uint64_t, 10> draft_confidence_bin_samples{};
    std::array<uint64_t, 10> draft_confidence_bin_accepted{};
    std::array<double, 10> draft_confidence_bin_sum{};
    std::string decode_path;
    std::string fallback_reason;
    std::string speculative_mode_requested;
    std::string speculative_mode_effective;
    bool graph_executor_reused = false;
    bool context_compacted = false;
    resource_snapshot resources_before;
    resource_snapshot resources_after;
};

struct native_tool_result {
    std::string status; /* "none", "pass", or "fail" */
    std::vector<ajson> calls;
    std::string content;
    std::string raw;
    std::string error;
};

struct swarm_request_metadata {
    bool requested = false;
    std::string wave_id;
    std::string agent_id;
    std::string parent_agent_id;
    int priority = 50;
    std::optional<std::chrono::milliseconds> queue_timeout;
    std::optional<std::chrono::milliseconds> deadline;
    std::optional<std::uint64_t> token_budget;
};

struct swarm_request_context {
    std::uint64_t sequence = 0u;
    swarm_request_metadata metadata;
    axiom::qwen38::swarm_task_id task_id = 0u;
    bool submitted = false;
    bool admitted = false;
    bool completed = false;
    axiom::qwen38::swarm_time_point submitted_at{};
    axiom::qwen38::swarm_time_point admitted_at{};
};

struct request_session;

struct server_state {
    axiom_qwen38_model *model = nullptr;
    axiom_qwen38_vision *vision = nullptr;
    axiom_qwen38_vision_info vision_info{};
    axiom_tokenizer *tokenizer = nullptr;
    axiom_tokenizer_info tokenizer_info{};
    axiom_runtime *draft_runtime = nullptr;
    axiom_model *draft_checkpoint = nullptr;
    axiom_qwen38_dspark *draft = nullptr;
    axiom_runtime *mtp_runtime = nullptr;
    axiom_qwen38_mtp *mtp = nullptr;
    axiom_qwen38_mtp_info mtp_info{};
    speculative_engine engine = speculative_engine::dspark;
    bool native_mtp_loaded = false;
    uint32_t native_mtp_fast_graph_min_position = 0u;
    axiom_qwen38_kv_tier *kv_tier = nullptr;
    std::mutex session_mutex;
    /* Durable KV publication is deliberately serialized behind the native
     * decode lock, but it must not hold the HTTP response open.  The worker
     * takes ownership of the post-generation model/session state and the
     * next request uses persistence_done_cv as an explicit handoff barrier;
     * mutex fairness is never assumed. It cannot mutate the model/tier until
     * the durable watermark is published and the model is reset. */
    std::mutex persistence_mutex;
    std::condition_variable persistence_cv;
    std::condition_variable persistence_done_cv;
    std::deque<std::function<void()>> persistence_jobs;
    std::thread persistence_worker;
    bool persistence_stop = false;
    /* Lifecycle GC has no place on the model/persistence critical path. It is
     * coalesced onto a maintenance worker and waits while native generation is
     * active. Generation-page pruning remains ordered with manifest commit. */
    std::mutex maintenance_mutex;
    std::condition_variable maintenance_cv;
    std::thread maintenance_worker;
    bool maintenance_stop = false;
    bool lifecycle_gc_requested = false;
    bool lifecycle_gc_force = false;
    std::string lifecycle_gc_protected_namespace;
    std::atomic<bool> lifecycle_gc_active{false};
    std::atomic<uint64_t> lifecycle_gc_deferred{0u};
    std::atomic<uint64_t> session_persistence_queued{0u};
    std::atomic<uint64_t> session_persistence_completed{0u};
    std::atomic<uint64_t> session_persistence_failures{0u};
    std::atomic<uint64_t> session_persistence_pending{0u};
    std::atomic<uint64_t> session_gc_runs{0u};
    std::atomic<uint64_t> session_generation_gc_runs{0u};
    std::atomic<uint64_t> session_lifecycle_gc_runs{0u};
    std::atomic<uint64_t> session_gc_removed_files{0u};
    std::atomic<uint64_t> session_gc_removed_sessions{0u};
    std::atomic<uint64_t> session_gc_reclaimed_bytes{0u};
    std::atomic<uint64_t> session_gc_scanned_sessions{0u};
    std::atomic<uint64_t> session_gc_protected_sessions{0u};
    std::atomic<uint64_t> session_gc_unsafe_namespaces{0u};
    std::atomic<uint64_t> session_gc_last_run_unix{0u};
    std::atomic<uint64_t> session_gc_last_duration_ms{0u};
    std::atomic<uint64_t> session_gc_failures{0u};
    /* A parsed HTTP request leases its durable namespace before tokenization
     * or swarm admission. Lifecycle GC holds this mutex across selection and
     * atomic rename, so a queued/in-flight request can never lose its cache. */
    mutable std::mutex session_lease_mutex;
    std::unordered_map<std::string, uint32_t> session_namespace_leases;
    std::atomic<uint64_t> session_leases_active{0u};
    std::atomic<uint64_t> session_leases_peak{0u};
    mutable std::mutex request_execution_mutex;
    mutable std::mutex telemetry_mutex;
    /* Correlate the live scheduler slot with the completed /ops/usage record.
     * The transport session id is intentionally retained: Desktop derives it
     * from its durable Harness session and can therefore reject another
     * client's last-completed sample instead of displaying stale tok/s. */
    mutable std::mutex active_request_mutex;
    uint64_t active_request_sequence = 0u;
    std::string active_request_session_id;
    std::shared_ptr<axiom::qwen38::request_progress> active_request_progress;
    axiom::qwen38::qwen38_swarm_scheduler swarm_scheduler;
    mutable std::mutex swarm_lease_mutex;
    std::unordered_map<axiom::qwen38::swarm_task_id,
                       axiom::qwen38::swarm_task_lease> swarm_leases;
    /* Logical Desktop wave/agent ids are strings; retain them beside the
     * numeric native scheduler records so /ops/swarm never loses correlation
     * after a task leaves the admission queue. */
    mutable std::mutex swarm_metadata_mutex;
    std::unordered_map<axiom::qwen38::swarm_task_id, swarm_request_metadata>
            swarm_metadata_history;
    /* Global logical ceiling owned by the loaded model/KV tier. */
    uint32_t max_context = 0u;
    /* Immutable, canonical listener inventory exposed by /health. */
    std::vector<std::string> listen_addresses;
    /* Per-request default; an explicit context_window may select max_context. */
    uint32_t default_context = kDefaultRequestContext;
    uint32_t vision_max_patch_tokens = kDefaultVisionMaxPatchTokens;
    std::mutex vision_mutex; // Multimodal preparation precedes scheduler admission.
    std::atomic<uint64_t> request_sequence{0u};
    /* HTTP transport is intentionally not admission-limited. Generation is
     * bounded by swarm_scheduler instead, so health/ops traffic can never be
     * rejected merely because generation requests are queued. The counter is
     * observability and clean-shutdown bookkeeping only. */
    std::atomic<size_t> active_http_workers{0u};
    token_telemetry last_telemetry;
    std::atomic<bool> healthy{true};
    bool streaming_kv = false;
    bool temporal_hot = false;
    uint32_t speculative_context_tokens = kDefaultSpeculativeContextTokens;
    bool no_think = false;
    bool context_compaction = true;
    uint32_t sse_batch_cycles = kDefaultSseBatchCycles;
    std::string kv_base_path;
    std::string target_identity;
    std::string dspark_identity;
    axiom::qwen38::spec_identity::result speculative_identity;
    std::string speculative_fingerprint;
    axiom::qwen38::spec_identity::qualification_evidence speculative_qualification;
    bool speculative_qualified = false;
    bool speculative_diagnostic_override = false;
    std::string speculative_contract_path;
    std::string kv_config_signature;
    uint32_t kv_hot_pages = kKvTierDefaultHotPages;
    uint64_t session_ttl_seconds = kDefaultSessionTtlSeconds;
    uint32_t session_max_namespaces = kDefaultSessionMaxNamespaces;
    uint64_t session_gc_interval_seconds = kDefaultSessionGcIntervalSeconds;
    axiom::qwen38::qwen38_persistent_session_store session_store;
    mutable std::mutex session_status_mutex;
    axiom::qwen38::qwen38_session_key active_session_key;
    axiom::qwen38::qwen38_session_paths active_session_paths;
    axiom::qwen38::qwen38_session_manifest active_session_manifest;
    bool active_session_manifest_exists = false;
    uint64_t active_session_generation = 0u;
    std::atomic<uint64_t> session_restore_count{0u};
    std::atomic<uint64_t> session_manifest_writes{0u};
    std::atomic<uint64_t> session_cold_page_reads{0u};
    mutable std::mutex persistence_profile_mutex;
    std::array<double, 7> persistence_last_ms{};
    uint64_t persistence_profile_jobs = 0u;
    uint32_t persistence_profile_tokens = 0u;
    int persistence_profile_status = AXIOM_OK;
    /* The native model is serialized, so one quiescent CUDA graph executor is
     * sufficient.  It is handed to the next greedy request only after KV
     * publication, exact device-session handoff and a verified position-zero
     * reset.  A raw pointer keeps server_state independent of the later full
     * request_session definition; destroy_server_state owns its lifetime. */
    request_session *resident_graph_session = nullptr;
    std::atomic<bool> graph_executor_resident{false};
    std::atomic<uint64_t> graph_executor_creates{0u};
    std::atomic<uint64_t> graph_executor_reuses{0u};
    std::atomic<uint64_t> graph_executor_recycles{0u};
    std::atomic<uint64_t> graph_executor_recycle_failures{0u};
    axiom::qwen38::swarm_agent_id swarm_kernel_agent_id = 0u;

    server_state()
            : swarm_scheduler([] {
                  axiom::qwen38::swarm_scheduler_config config;
                  config.max_concurrent = 1u;
                  config.max_queued_tasks = kSwarmQueueDepth;
                  config.max_event_history = kSwarmEventHistory;
                  return config;
              }()) {
        axiom::qwen38::swarm_agent_spec agent;
        agent.name = "qwen38-native-gpu-0";
        agent.max_concurrent = 1u;
        const auto registered = swarm_scheduler.register_agent(
                agent, axiom::qwen38::swarm_clock::now());
        if (registered.accepted) swarm_kernel_agent_id = registered.agent_id;
    }
};

bool speculative_graph_loaded(const server_state *state) {
    if (!state || !state->temporal_hot) return false;
    return state->engine == speculative_engine::native_mtp
            ? state->native_mtp_loaded && state->mtp
            : state->draft != nullptr;
}

struct session_status_snapshot {
    std::string session_id;
    std::string namespace_id;
    std::string tier_path;
    uint64_t generation = 0u;
    uint32_t committed_tokens = 0u;
    bool manifest_committed = false;
};

session_status_snapshot capture_session_status(const server_state *state) {
    session_status_snapshot snapshot;
    if (!state) return snapshot;
    std::lock_guard<std::mutex> lock(state->session_status_mutex);
    snapshot.session_id = state->active_session_key.session_id;
    snapshot.namespace_id = state->active_session_paths.namespace_id;
    snapshot.tier_path = state->active_session_paths.tier_path;
    snapshot.generation = state->active_session_generation;
    snapshot.committed_tokens = state->active_session_manifest.committed_tokens;
    snapshot.manifest_committed = state->active_session_manifest_exists;
    return snapshot;
}

struct session_namespace_lease {
    server_state *state = nullptr;
    std::string namespace_id;

    session_namespace_lease() = default;
    session_namespace_lease(server_state *state_value, std::string namespace_value)
            : state(state_value), namespace_id(std::move(namespace_value)) {
        if (!state || namespace_id.empty()) return;
        {
            std::lock_guard<std::mutex> lock(state->session_lease_mutex);
            ++state->session_namespace_leases[namespace_id];
        }
        const uint64_t active = state->session_leases_active.fetch_add(1u) + 1u;
        uint64_t peak = state->session_leases_peak.load();
        while (active > peak &&
               !state->session_leases_peak.compare_exchange_weak(peak, active)) {
        }
    }

    session_namespace_lease(const session_namespace_lease &) = delete;
    session_namespace_lease &operator=(const session_namespace_lease &) = delete;

    ~session_namespace_lease() {
        if (!state || namespace_id.empty()) return;
        {
            std::lock_guard<std::mutex> lock(state->session_lease_mutex);
            const auto found = state->session_namespace_leases.find(namespace_id);
            if (found != state->session_namespace_leases.end()) {
                if (found->second > 1u) {
                    --found->second;
                } else {
                    state->session_namespace_leases.erase(found);
                }
            }
        }
        state->session_leases_active.fetch_sub(1u);
    }
};

/* HTTP workers are independent, while native generation itself is serialized
 * by session_mutex. Keep the selected context limit thread-local so every
 * tokenizer, compaction, KV and generation guard in the request observes the
 * same value without mutating the global model configuration. */
thread_local const server_state *g_request_context_state = nullptr;
thread_local uint32_t g_request_context_limit = 0u;

uint32_t effective_context_limit(const server_state *state) {
    if (!state) return 0u;
    return g_request_context_state == state && g_request_context_limit != 0u
            ? g_request_context_limit : state->max_context;
}

struct request_context_scope {
    const server_state *previous_state = nullptr;
    uint32_t previous_limit = 0u;

    request_context_scope(const server_state *state, uint32_t limit)
            : previous_state(g_request_context_state),
              previous_limit(g_request_context_limit) {
        g_request_context_state = state;
        g_request_context_limit = limit;
    }

    ~request_context_scope() {
        g_request_context_state = previous_state;
        g_request_context_limit = previous_limit;
    }
};

void complete_persistence_operation(server_state *state) {
    if (!state) return;
    bool underflow = false;
    {
        std::lock_guard<std::mutex> lock(state->persistence_mutex);
        const uint64_t pending = state->session_persistence_pending.load(
                std::memory_order_acquire);
        if (pending == 0u) {
            underflow = true;
        } else {
            state->session_persistence_pending.fetch_sub(
                    1u, std::memory_order_release);
        }
    }
    if (underflow) {
        state->healthy.store(false);
        state->session_persistence_failures.fetch_add(1u);
        std::fprintf(stderr,
                     "axiom-qwen38-api: persistence pending counter underflow\n");
    }
    state->persistence_done_cv.notify_all();
}

void persistence_worker_loop(server_state *state) {
    if (!state) return;
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(state->persistence_mutex);
            state->persistence_cv.wait(lock, [state] {
                return state->persistence_stop || !state->persistence_jobs.empty();
            });
            if (state->persistence_jobs.empty()) {
                if (state->persistence_stop) return;
                continue;
            }
            job = std::move(state->persistence_jobs.front());
            state->persistence_jobs.pop_front();
        }
        try {
            job();
        } catch (const std::exception &exception) {
            std::fprintf(stderr,
                         "axiom-qwen38-api: deferred persistence exception: %s\n",
                         exception.what());
            state->healthy.store(false);
            state->session_persistence_failures.fetch_add(1u);
            complete_persistence_operation(state);
        } catch (...) {
            std::fprintf(stderr,
                         "axiom-qwen38-api: deferred persistence exception: unknown\n");
            state->healthy.store(false);
            state->session_persistence_failures.fetch_add(1u);
            complete_persistence_operation(state);
        }
    }
}

bool start_persistence_worker(server_state *state) {
    if (!state) return false;
    if (state->persistence_worker.joinable()) return true;
    try {
        state->persistence_worker = std::thread(persistence_worker_loop, state);
    } catch (...) {
        return false;
    }
    return true;
}

bool enqueue_persistence_job(
        server_state *state, std::function<void()> job) {
    if (!state || !job) return false;
    try {
        {
            std::lock_guard<std::mutex> lock(state->persistence_mutex);
            if (state->persistence_stop) return false;
            state->persistence_jobs.push_back(std::move(job));
            state->session_persistence_queued.fetch_add(1u);
            state->session_persistence_pending.fetch_add(
                    1u, std::memory_order_release);
        }
    } catch (...) {
        return false;
    }
    state->persistence_cv.notify_one();
    return true;
}

void stop_persistence_worker(server_state *state) {
    if (!state) return;
    {
        std::lock_guard<std::mutex> lock(state->persistence_mutex);
        state->persistence_stop = true;
    }
    state->persistence_cv.notify_all();
    state->persistence_done_cv.notify_all();
    if (state->persistence_worker.joinable()) state->persistence_worker.join();
}

std::unique_lock<std::mutex> acquire_generation_session_lock(server_state *state) {
    for (;;) {
        std::unique_lock<std::mutex> session_lock(state->session_mutex);
        if (state->session_persistence_pending.load(std::memory_order_acquire) == 0u) {
            return session_lock;
        }
        session_lock.unlock();
        std::unique_lock<std::mutex> persistence_lock(state->persistence_mutex);
        state->persistence_done_cv.wait(persistence_lock, [state] {
            return state->session_persistence_pending.load(
                           std::memory_order_acquire) == 0u;
        });
    }
}

bool swarm_string_field(
        const ajson &object, const char *key, std::string *out,
        std::string *error) {
    const ajson *value = object.get(key);
    if (!value) return true;
    if (!value->is_string() || value->s.size() > 256u) {
        if (error) *error = std::string("axiom_swarm.") + key +
                " must be a string of at most 256 bytes";
        return false;
    }
    *out = value->s;
    return true;
}

bool swarm_u64_field(
        const ajson &object, const char *key, std::uint64_t minimum,
        std::uint64_t maximum, std::optional<std::uint64_t> *out,
        std::string *error) {
    const ajson *value = object.get(key);
    if (!value) return true;
    if (!value->is_int() || value->i < 0 ||
        static_cast<std::uint64_t>(value->i) < minimum ||
        static_cast<std::uint64_t>(value->i) > maximum) {
        if (error) *error = std::string("axiom_swarm.") + key +
                " must be an integer in the configured range";
        return false;
    }
    *out = static_cast<std::uint64_t>(value->i);
    return true;
}

bool parse_swarm_metadata(
        const ajson &payload, swarm_request_metadata *out, std::string *error) {
    if (!out || !error) return false;
    *out = swarm_request_metadata{};
    const ajson *raw = payload.get("axiom_swarm");
    if (!raw) return true;
    out->requested = true;
    if (!raw->is_object()) {
        *error = "axiom_swarm must be an object";
        return false;
    }
    if (!swarm_string_field(*raw, "wave_id", &out->wave_id, error) ||
        !swarm_string_field(*raw, "agent_id", &out->agent_id, error) ||
        !swarm_string_field(*raw, "parent_agent_id", &out->parent_agent_id, error)) {
        return false;
    }
    const ajson *priority = raw->get("priority");
    if (priority) {
        if (!priority->is_int() || priority->i < 0 || priority->i > 100) {
            *error = "axiom_swarm.priority must be an integer in [0,100]";
            return false;
        }
        out->priority = static_cast<int>(priority->i);
    }
    std::optional<std::uint64_t> queue_timeout;
    std::optional<std::uint64_t> deadline;
    if (!swarm_u64_field(*raw, "queue_timeout_ms", 0u, 24ull * 60ull * 60ull * 1000ull,
                         &queue_timeout, error) ||
        !swarm_u64_field(*raw, "deadline_ms", 1u, 24ull * 60ull * 60ull * 1000ull,
                         &deadline, error) ||
        !swarm_u64_field(*raw, "token_budget", 1u, kMaxNativeContext,
                         &out->token_budget, error)) {
        return false;
    }
    if (queue_timeout) {
        out->queue_timeout = std::chrono::milliseconds(*queue_timeout);
    }
    if (deadline) {
        out->deadline = std::chrono::milliseconds(*deadline);
    }
    return true;
}

ajson swarm_response_metadata(
        const swarm_request_context &context,
        axiom::qwen38::swarm_time_point finished_at,
        const char *status = "succeeded") {
    ajson root = ajson::jobj();
    root.set("request_id", ajson::jint(static_cast<long long>(context.task_id)));
    root.set("request_sequence", ajson::jint(static_cast<long long>(context.sequence)));
    if (!context.metadata.wave_id.empty()) {
        root.set("wave_id", ajson::jstr(context.metadata.wave_id));
    }
    if (!context.metadata.agent_id.empty()) {
        root.set("agent_id", ajson::jstr(context.metadata.agent_id));
    }
    if (!context.metadata.parent_agent_id.empty()) {
        root.set("parent_agent_id", ajson::jstr(context.metadata.parent_agent_id));
    }
    root.set("priority", ajson::jint(context.metadata.priority));
    const auto queue_wait = context.admitted
            ? std::chrono::duration_cast<std::chrono::milliseconds>(
                    context.admitted_at - context.submitted_at).count()
            : 0ll;
    const auto execution = context.admitted
            ? std::chrono::duration_cast<std::chrono::milliseconds>(
                    finished_at - context.admitted_at).count()
            : 0ll;
    root.set("queue_wait_ms", ajson::jint(static_cast<long long>(std::max(0ll, queue_wait))));
    root.set("execution_ms", ajson::jint(static_cast<long long>(std::max(0ll, execution))));
    root.set("backend", ajson::jstr(kBackendId));
    root.set("status", ajson::jstr(status ? status : "succeeded"));
    return root;
}

void attach_swarm_response(
        ajson *response, const swarm_request_context &context,
        axiom::qwen38::swarm_time_point finished_at,
        const char *status = "succeeded") {
    if (!response || !context.metadata.requested || !response->is_object()) return;
    response->set("axiom_swarm", swarm_response_metadata(context, finished_at, status));
}

struct session_activation {
    axiom::qwen38::qwen38_session_key key;
    axiom::qwen38::qwen38_session_paths paths;
    axiom::qwen38::qwen38_session_manifest manifest;
    bool manifest_exists = false;
    bool restored = false;
    bool stateful_resume = false;
    bool prefix_retokenized = false;
    bool tool_projection = false;
    bool transaction_active = false;
    uint32_t start_position = 0u;
    uint32_t next_token = 0u;
    /* Non-empty for byte-equivalent retokenization or the separately permitted
     * legacy stateful tail. Native prefix token IDs/state always stay intact. */
    std::vector<uint32_t> effective_prompt_ids;
};

struct cycle_history {
    uint32_t proposal[AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS]{};
    float proposal_confidence[AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS]{};
    uint32_t accepted_prefix = 0u;
    uint32_t continuation_token = 0u;
    uint32_t async_status = 0u;
    uint32_t next_position = 0u;
    uint32_t committed_tokens = 0u;
};
static_assert(sizeof(cycle_history) == sizeof(axiom_qwen38_dspark_device_history),
              "device history ABI drift");

/* Native MTP publishes transient device pointers. Snapshot every replay into
 * a distinct device slot before launching the next graph, then perform one
 * bulk D2H transfer for the batch. */
struct native_cycle_history {
    uint32_t draft[AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS]{};
    uint32_t verify[AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH]{};
    uint32_t emitted[AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH]{};
    uint32_t accepted_prefix = 0u;
    uint32_t commit_count = 0u;
    uint32_t emitted_count = 0u;
    uint32_t continuation = 0u;
    uint32_t stop_detected = 0u;
    uint32_t matched_stop_token = AXIOM_TOKEN_ID_INVALID;
    uint32_t next_anchor = 0u;
    uint32_t next_position = 0u;
    uint32_t async_status = 0u;
    float continuation_logit = 0.0f;
};


struct request_session {
    speculative_engine engine = speculative_engine::dspark;
    axiom_qwen38_dspark_compute *compute = nullptr;
    axiom_qwen38_speculative *speculative = nullptr;
    axiom_qwen38_mtp_compute *mtp_compute = nullptr;
    axiom_qwen38_mtp_speculative *mtp_speculative = nullptr;
    cudaStream_t stream = nullptr;
    uint32_t *seed_anchor_device = nullptr;
    uint32_t *seed_position_device = nullptr;
    cycle_history *history_host = nullptr;
    cycle_history *history_device = nullptr;
    native_cycle_history *mtp_history_device = nullptr;
    native_cycle_history *mtp_history_host = nullptr;
    uint32_t history_capacity = 0u;
    bool stream_synchronized = true;
};

/* Owns every object that must remain alive until the durable KV watermark and
 * manifest have been published.  The HTTP handler receives the generated
 * answer before this job runs; the session mutex still serializes the job
 * with the next native request, so persistence is asynchronous to the wire
 * but never concurrent with mutable model access. */
struct deferred_persistence_job {
    server_state *state = nullptr;
    /* The HTTP request lease may end as soon as the response is delivered.
     * Keep an independent lease for the queued job until ordered persistence,
     * manifest publication and model reset have all completed. */
    std::unique_ptr<session_namespace_lease> namespace_lease;
    session_activation activation;
    std::vector<uint32_t> prompt_ids;
    /* Keep multimodal embeddings alive until a durable graph suffix has been
     * reduced to the visible session prefix.  Replaying an image placeholder
     * through the tokenizer embedding would silently corrupt the persisted
     * session, so the exact replay path receives this same device buffer. */
    std::shared_ptr<struct vision_request> vision;
    generation_result result;
    request_session session;
    uint32_t persist_start_position = 0u;
    uint32_t graph_position = 0u;
    bool persistent_session = false;
    bool scalar_path = false;
};

struct vision_request {
    std::vector<int32_t> embedding_slots; /* -1 for tokenizer rows */
    std::vector<float> patch_values;
    std::vector<axiom_qwen38_vision_grid> grids;
    std::vector<float> embedding_host;
    float *embedding_device = nullptr;
    uint32_t visual_tokens = 0u;
    uint32_t media_count = 0u;

    vision_request() = default;
    vision_request(const vision_request &) = delete;
    vision_request &operator=(const vision_request &) = delete;
    ~vision_request() {
        if (embedding_device) (void)cudaFree(embedding_device);
    }
};


int fail(const char *what, int rc = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-api: %s%s%s\n", what,
                 rc == AXIOM_OK ? "" : ": ", rc == AXIOM_OK ? "" : axiom_status_string(rc));
    return 1;
}

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}


int validate_target_session(server_state *state, std::string *stage) {
    if (!state || !state->model || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    capabilities.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    int rc = axiom_qwen38_model_dspark_temporal_capabilities_get(
            state->model, &capabilities);
    if (rc == AXIOM_OK && capabilities.temporal_m8_available == 1u &&
        capabilities.temporal_m8_implemented == 1u &&
        capabilities.temporal_m8_validated == 1u) {
        return AXIOM_OK;
    }
    if (rc != AXIOM_OK) {
        *stage = "temporal_capabilities_get";
        return rc;
    }
    if (axiom_qwen38_model_position(state->model) != 0u) {
        *stage = "target_not_reset_before_validation";
        return AXIOM_ERR_RUNTIME;
    }
    axiom_qwen38_model_dspark_temporal_validation_result validation{};
    validation.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    *stage = "temporal8_validate";
    rc = axiom_qwen38_model_dspark_temporal8_validate(
            state->model, kValidationTokens, 0.0f, &validation);
    if (rc == AXIOM_OK && validation.passed != 1u) rc = AXIOM_ERR_RUNTIME;
    if (rc == AXIOM_OK) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: temporal8 gate PASS position=%u "
                     "max_logit_error=%.9g max_tap_error=%.9g\n",
                     validation.tested_position,
                     static_cast<double>(validation.max_logit_abs_error),
                     static_cast<double>(validation.max_tap_abs_error));
    }
    return rc;
}

int destroy_request_session(request_session *session) {
    if (!session) return AXIOM_ERR_INVALID_ARGUMENT;
    int first = AXIOM_OK;
    if (session->stream && !session->stream_synchronized) {
        const int rc = cuda_status(cudaStreamSynchronize(session->stream));
        if (first == AXIOM_OK && rc != AXIOM_OK) first = rc;
        session->stream_synchronized = true;
    }
    axiom_qwen38_speculative_destroy(session->speculative);
    session->speculative = nullptr;
    axiom_qwen38_dspark_compute_destroy(session->compute);
    session->compute = nullptr;
    axiom_qwen38_mtp_speculative_destroy(session->mtp_speculative);
    session->mtp_speculative = nullptr;
    axiom_qwen38_mtp_compute_destroy(session->mtp_compute);
    session->mtp_compute = nullptr;
    if (session->seed_anchor_device) {
        const int rc = cuda_status(cudaFree(session->seed_anchor_device));
        if (first == AXIOM_OK && rc != AXIOM_OK) first = rc;
        session->seed_anchor_device = nullptr;
    }
    if (session->seed_position_device) {
        const int rc = cuda_status(cudaFree(session->seed_position_device));
        if (first == AXIOM_OK && rc != AXIOM_OK) first = rc;
        session->seed_position_device = nullptr;
    }
    if (session->history_host) {
        const int rc = cuda_status(cudaFreeHost(session->history_host));
        if (first == AXIOM_OK && rc != AXIOM_OK) first = rc;
        session->history_host = nullptr;
    }
    if (session->history_device) {
        const int rc = cuda_status(cudaFree(session->history_device));
        if (first == AXIOM_OK && rc != AXIOM_OK) first = rc;
        session->history_device = nullptr;
    }
    if (session->mtp_history_host) {
        const int rc = cuda_status(cudaFreeHost(session->mtp_history_host));
        if (first == AXIOM_OK && rc != AXIOM_OK) first = rc;
        session->mtp_history_host = nullptr;
    }
    if (session->mtp_history_device) {
        const int rc = cuda_status(cudaFree(session->mtp_history_device));
        if (first == AXIOM_OK && rc != AXIOM_OK) first = rc;
        session->mtp_history_device = nullptr;
    }
    if (session->stream) {
        const int rc = cuda_status(cudaStreamDestroy(session->stream));
        if (first == AXIOM_OK && rc != AXIOM_OK) first = rc;
        session->stream = nullptr;
    }
    session->history_capacity = 0u;
    session->engine = speculative_engine::dspark;
    return first;
}

int create_request_session(server_state *state, request_session *session, std::string *stage) {
    if (!state || !state->model || !session || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    session->engine = state->engine;
    if (state->engine == speculative_engine::native_mtp) {
        if (!state->mtp || !state->mtp_runtime) return AXIOM_ERR_INVALID_ARGUMENT;
        axiom_qwen38_mtp_target_binding target_binding{};
        target_binding.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
        target_binding.user_data = state->model;
        target_binding.embed_f32_device =
                axiom_qwen38_model_dspark_target_embed_f32_device;
        target_binding.lm_head_f32_device =
                axiom_qwen38_model_dspark_target_lm_head_f32_device;
        axiom_qwen38_mtp_compute_config_v2 compute_config{};
        compute_config.abi_version = AXIOM_QWEN38_MTP_COMPUTE_CONFIG_V2_ABI_VERSION;
        compute_config.struct_size = sizeof(compute_config);
        compute_config.max_context = state->speculative_context_tokens;
        compute_config.cache_dtype = AXIOM_TENSOR_DTYPE_BF16;
        compute_config.rope_profile = axiom_qwen38_model_rope_profile(state->model);
        *stage = "native_mtp_compute_create";
        int rc = axiom_qwen38_mtp_compute_create_v2(
                state->mtp, state->mtp_runtime, 0, &compute_config,
                sizeof(compute_config), &target_binding, &session->mtp_compute);
        axiom_qwen38_mtp_speculative_config speculative_config{};
        speculative_config.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        speculative_config.struct_size = sizeof(speculative_config);
        speculative_config.device = 0u;
        if (rc == AXIOM_OK) {
            *stage = "native_mtp_controller_create";
            rc = axiom_qwen38_mtp_speculative_create_v1(
                    state->model, session->mtp_compute, &speculative_config,
                    sizeof(speculative_config), &session->mtp_speculative);
        }
        std::array<uint32_t, 2u> stop_ids{};
        uint32_t stop_count = 0u;
        const uint32_t candidates[2] = {
                state->tokenizer_info.endoftext_token_id,
                state->tokenizer_info.im_end_token_id,
        };
        for (const uint32_t token : candidates) {
            if (token >= AXIOM_QWEN38_MODEL_VOCAB) continue;
            if (stop_count != 0u && stop_ids[0] == token) continue;
            stop_ids[stop_count++] = token;
        }
        if (rc == AXIOM_OK) {
            *stage = "native_mtp_stop_tokens";
            rc = axiom_qwen38_mtp_speculative_device_stop_tokens_set(
                    session->mtp_speculative, stop_ids.data(), stop_count);
        }
        return rc;
    }
    if (!state->draft || !state->draft_runtime) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_qwen38_dspark_target_binding target_binding{};
    target_binding.abi_version = AXIOM_ABI_VERSION;
    target_binding.user_data = state->model;
    target_binding.embed_f32_device = axiom_qwen38_model_dspark_target_embed_f32_device;
    target_binding.lm_head_f32_device = axiom_qwen38_model_dspark_target_lm_head_f32_device;
    axiom_qwen38_dspark_compute_config compute_config{};
    compute_config.abi_version = AXIOM_ABI_VERSION;
    compute_config.max_context = state->streaming_kv && state->temporal_hot
            ? state->speculative_context_tokens : effective_context_limit(state);
    const uint32_t stop_candidates[2] = {
        state->tokenizer_info.endoftext_token_id,
        state->tokenizer_info.im_end_token_id,
    };
    for (const uint32_t token : stop_candidates) {
        if (token >= AXIOM_QWEN38_DSPARK_VOCAB) continue;
        if (compute_config.stop_token_count != 0u &&
            compute_config.stop_token_ids[0] == token) {
            continue;
        }
        compute_config.stop_token_ids[compute_config.stop_token_count++] = token;
    }
    *stage = "dspark_compute_create";
    int rc = axiom_qwen38_dspark_compute_create(
            state->draft, state->draft_runtime, 0, &compute_config,
            &target_binding, &session->compute);
    axiom_qwen38_speculative_config speculative_config{};
    speculative_config.abi_version = AXIOM_ABI_VERSION;
    speculative_config.device = 0;
    if (rc == AXIOM_OK) {
        *stage = "speculative_create";
        rc = axiom_qwen38_speculative_create(
                state->model, state->draft, session->compute,
                &speculative_config, &session->speculative);
    }
    return rc;
}

bool request_session_ready(const request_session *session) {
    if (!session) return false;
    return session->engine == speculative_engine::native_mtp
            ? session->mtp_compute && session->mtp_speculative
            : session->compute && session->speculative;
}

int request_session_prefill_token(
        request_session *session, const uint32_t token_id,
        const float *embedding_device, const uint32_t expected_position,
        uint32_t *next_token, std::string *stage) {
    if (!request_session_ready(session) || !next_token || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (session->engine == speculative_engine::native_mtp) {
        if (embedding_device) {
            *stage = "native_mtp_vision_prefill_unsupported";
            return AXIOM_ERR_NOT_IMPLEMENTED;
        }
        axiom_qwen38_mtp_speculative_prefill_result result{};
        result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        result.struct_size = sizeof(result);
        *stage = "native_mtp_prefill";
        int rc = axiom_qwen38_mtp_speculative_prefill_token(
                session->mtp_speculative, token_id, &result);
        if (rc == AXIOM_OK &&
            (result.token_position != expected_position ||
             result.target_token_id >= AXIOM_QWEN38_MODEL_VOCAB ||
             !std::isfinite(result.target_logit))) {
            *stage = "native_mtp_prefill_contract";
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) *next_token = result.target_token_id;
        return rc;
    }
    axiom_qwen38_speculative_prefill_request request{};
    request.abi_version = AXIOM_ABI_VERSION;
    request.token_id = token_id;
    request.embedding_device = embedding_device;
    axiom_qwen38_speculative_prefill_result result{};
    result.abi_version = AXIOM_ABI_VERSION;
    *stage = "speculative_prefill";
    int rc = axiom_qwen38_speculative_prefill_token(
            session->speculative, &request, &result);
    if (rc == AXIOM_OK &&
        (result.token_position != expected_position ||
         result.position_after_commit != expected_position + 1u ||
         result.target_token_id >= AXIOM_QWEN38_DSPARK_VOCAB)) {
        *stage = "speculative_prefill_contract";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) *next_token = result.target_token_id;
    return rc;
}

int request_session_position_verify(
        const server_state *state, const request_session *session,
        const uint32_t expected, std::string *stage) {
    if (!state || !state->model || !request_session_ready(session) || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (session->engine == speculative_engine::native_mtp) {
        axiom_qwen38_mtp_compute_info info{};
        info.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
        *stage = "native_mtp_position_info";
        int rc = axiom_qwen38_mtp_compute_info_get(session->mtp_compute, &info);
        if (rc == AXIOM_OK &&
            (axiom_qwen38_model_position(state->model) != expected ||
             info.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
             info.committed_position != expected)) {
            *stage = "native_mtp_position_contract";
            rc = AXIOM_ERR_RUNTIME;
        }
        return rc;
    }
    axiom_qwen38_dspark_compute_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    *stage = "dspark_position_info";
    int rc = axiom_qwen38_dspark_compute_info_get(session->compute, &info);
    if (rc == AXIOM_OK &&
        (axiom_qwen38_model_position(state->model) != expected ||
         info.transaction_open != 0u || info.committed_position != expected)) {
        *stage = "dspark_position_contract";
        rc = AXIOM_ERR_RUNTIME;
    }
    return rc;
}

size_t generated_tokens_to_commit(const generation_result &result) {
    if (result.ids.empty()) return 0u;
    /* At max_tokens the last visible token is still the pending prediction.
     * At a native stop token every visible token has already been consumed and
     * the pending prediction is the (non-visible) stop marker itself. */
    return result.finish_reason == "stop" ? result.ids.size() : result.ids.size() - 1u;
}

uint32_t durable_session_position(
        const size_t prompt_tokens, const generation_result &result) {
    return static_cast<uint32_t>(prompt_tokens + generated_tokens_to_commit(result));
}

int restore_session_target(
        server_state *state,
        axiom_qwen38_kv_tier *tier,
        const axiom::qwen38::qwen38_session_manifest &manifest,
        bool temporal_graph,
        std::string *failure_stage);
int restore_session_speculative(
        server_state *state,
        request_session *session,
        const axiom::qwen38::qwen38_session_manifest &manifest,
        std::string *failure_stage);

/* A fixed-width graph can commit a speculative suffix after the public
 * max_tokens boundary has been reached.  That suffix is valid GPU work but
 * is not present in the client-visible conversation, so it cannot be used as
 * a durable session checkpoint. Rebuild only the exact visible prefix when
 * this happens; normal graph requests stay on the fast path. */
int replay_exact_session_prefix(
        server_state *state,
        const std::vector<uint32_t> &prompt_ids,
        const generation_result &result,
        request_session *session,
        const session_activation *activation,
        const vision_request *vision,
        uint32_t *out_position,
        uint32_t *out_next_token,
        std::string *failure_stage) {
    const uint32_t context_limit = effective_context_limit(state);
    if (!state || !state->model || !session || !out_position || !out_next_token ||
        !failure_stage || prompt_ids.empty()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const size_t generated_to_commit = generated_tokens_to_commit(result);
    if (prompt_ids.size() > context_limit ||
        generated_to_commit > context_limit - prompt_ids.size()) {
        *failure_stage = "session_exact_replay_budget";
        return AXIOM_ERR_BUDGET;
    }
    std::vector<uint32_t> exact_ids;
    try {
        exact_ids.reserve(prompt_ids.size() + generated_to_commit);
        exact_ids.insert(exact_ids.end(), prompt_ids.begin(), prompt_ids.end());
        exact_ids.insert(
                exact_ids.end(), result.ids.begin(),
                result.ids.begin() + static_cast<std::ptrdiff_t>(generated_to_commit));
    } catch (...) {
        *failure_stage = "session_exact_replay_alloc";
        return AXIOM_ERR_BUDGET;
    }

    uint32_t replay_start = 0u;
    if (activation && activation->restored) {
        replay_start = activation->start_position;
        if (activation->manifest.committed_tokens != replay_start ||
            replay_start > exact_ids.size() ||
            activation->manifest.token_ids.size() < replay_start ||
            !std::equal(
                    activation->manifest.token_ids.begin(),
                    activation->manifest.token_ids.begin() + replay_start,
                    exact_ids.begin())) {
            *failure_stage = "session_exact_replay_restore_prefix";
            return AXIOM_ERR_RUNTIME;
        }
    }

    *failure_stage = "session_exact_replay_destroy";
    int rc = destroy_request_session(session);
    if (rc == AXIOM_OK) {
        *failure_stage = "session_exact_replay_model_reset";
        rc = axiom_qwen38_model_reset(state->model);
    }
    if (rc == AXIOM_OK && replay_start != 0u) {
        *failure_stage = "session_exact_replay_target_restore";
        rc = restore_session_target(
                state, state->kv_tier, activation->manifest, false,
                failure_stage);
    }
    if (rc == AXIOM_OK) {
        rc = create_request_session(state, session, failure_stage);
    }
    if (rc == AXIOM_OK && replay_start != 0u) {
        *failure_stage = "session_exact_replay_speculative_restore";
        rc = restore_session_speculative(
                state, session, activation->manifest, failure_stage);
    }

    uint32_t next_token = 0u;
    if (replay_start != 0u) next_token = activation->next_token;
    for (size_t index = replay_start;
         index < exact_ids.size() && rc == AXIOM_OK; ++index) {
        if (exact_ids[index] >= AXIOM_QWEN38_DSPARK_VOCAB) {
            *failure_stage = "session_exact_replay_token_range";
            rc = AXIOM_ERR_INVALID_ARGUMENT;
            break;
        }
        const float *embedding_device = nullptr;
        if (vision && index < vision->embedding_slots.size()) {
            const int32_t slot = vision->embedding_slots[index];
            if (slot >= 0 && vision->embedding_device) {
                embedding_device = vision->embedding_device +
                        static_cast<size_t>(slot) * AXIOM_QWEN38_VISION_OUTPUT;
            }
        }
        *failure_stage = "session_exact_replay_prefill";
        rc = request_session_prefill_token(
                session, exact_ids[index], embedding_device,
                static_cast<uint32_t>(index), &next_token, failure_stage);
    }
    if (rc == AXIOM_OK) {
        *failure_stage = "session_exact_replay_sync";
        rc = cuda_status(cudaStreamSynchronize(nullptr));
    }
    if (rc == AXIOM_OK) {
        *failure_stage = "session_exact_replay_info";
        rc = request_session_position_verify(
                state, session, static_cast<uint32_t>(exact_ids.size()),
                failure_stage);
    }
    if (rc == AXIOM_OK) {
        *out_position = static_cast<uint32_t>(exact_ids.size());
        *out_next_token = next_token;
    }
    return rc;
}

int allocate_device_decode(
        server_state *state,
        request_session *session,
        uint32_t history_capacity,
        std::string *stage) {
    if (!state || !session || !request_session_ready(session) ||
        history_capacity == 0u || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    int rc = AXIOM_OK;
    if (session->engine == speculative_engine::native_mtp) {
        *stage = "native_mtp_device_prepare";
        rc = axiom_qwen38_mtp_speculative_device_prepare(
                session->mtp_speculative);
    } else {
        *stage = "device_target_bind";
        rc = axiom_qwen38_speculative_device_target_bind_qwen38_model(
                session->speculative);
    }
    if (rc == AXIOM_OK) {
        *stage = "decode_stream_create";
        rc = cuda_status(cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking));
    }
    if (rc == AXIOM_OK) {
        *stage = "seed_anchor_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&session->seed_anchor_device), sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK) {
        *stage = "seed_position_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&session->seed_position_device), sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK && session->engine == speculative_engine::dspark) {
        *stage = "history_device_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&session->history_device),
                static_cast<size_t>(history_capacity) * sizeof(cycle_history)));
    }
    if (rc == AXIOM_OK && session->engine == speculative_engine::dspark) {
        *stage = "history_host_allocate";
        rc = cuda_status(cudaHostAlloc(
                reinterpret_cast<void **>(&session->history_host),
                static_cast<size_t>(history_capacity) * sizeof(cycle_history),
                cudaHostAllocPortable));
    }
    if (rc == AXIOM_OK && session->engine == speculative_engine::native_mtp) {
        *stage = "native_mtp_history_device_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&session->mtp_history_device),
                static_cast<size_t>(history_capacity) *
                        sizeof(native_cycle_history)));
    }
    if (rc == AXIOM_OK && session->engine == speculative_engine::native_mtp) {
        *stage = "native_mtp_history_host_allocate";
        rc = cuda_status(cudaHostAlloc(
                reinterpret_cast<void **>(&session->mtp_history_host),
                static_cast<size_t>(history_capacity) *
                        sizeof(native_cycle_history),
                cudaHostAllocPortable));
    }
    if (rc == AXIOM_OK) session->history_capacity = history_capacity;
    return rc;
}

int take_resident_graph_session(
        server_state *state, request_session *session, bool *reused) {
    if (!state || !session || !reused) return AXIOM_ERR_INVALID_ARGUMENT;
    *reused = false;
    if (!state->resident_graph_session ||
        !request_session_ready(state->resident_graph_session)) return AXIOM_OK;
    if (state->resident_graph_session->engine != state->engine) {
        state->graph_executor_resident.store(false, std::memory_order_release);
        return destroy_request_session(state->resident_graph_session);
    }
    *session = *state->resident_graph_session;
    *state->resident_graph_session = request_session{};
    state->graph_executor_resident.store(false, std::memory_order_release);
    state->graph_executor_reuses.fetch_add(1u);
    *reused = true;
    return AXIOM_OK;
}

int prepare_graph_session_for_reuse(
        server_state *state, request_session *session,
        std::string *failure_stage) {
    if (!state || !state->model || !session || !request_session_ready(session) ||
        !session->stream ||
        !session->seed_anchor_device || !session->seed_position_device ||
        (session->engine == speculative_engine::dspark &&
         (!session->history_host || !session->history_device)) ||
        (session->engine == speculative_engine::native_mtp &&
         (!session->mtp_history_device || !session->mtp_history_host)) ||
        session->history_capacity < kDeviceChunkCycles ||
        !failure_stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    int rc = AXIOM_OK;
    if (!session->stream_synchronized) {
        *failure_stage = "graph_executor_recycle_sync";
        rc = cuda_status(cudaStreamSynchronize(session->stream));
        if (rc == AXIOM_OK) session->stream_synchronized = true;
    }
    if (rc == AXIOM_OK && session->engine == speculative_engine::native_mtp) {
        *failure_stage = "graph_executor_recycle_native_mtp";
        rc = axiom_qwen38_mtp_speculative_recycle_to_zero(
                session->mtp_speculative);
    } else if (rc == AXIOM_OK) {
        *failure_stage = "graph_executor_recycle_compute_reset";
        rc = axiom_qwen38_dspark_compute_restore_position(session->compute, 0u);
    }
    if (rc == AXIOM_OK) {
        *failure_stage = "graph_executor_recycle_commit_limit";
        rc = session->engine == speculative_engine::native_mtp
                ? axiom_qwen38_mtp_speculative_device_commit_limit_set(
                        session->mtp_speculative,
                        AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH,
                        session->stream)
                : axiom_qwen38_speculative_device_commit_limit_set(
                        session->speculative,
                        AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH,
                        session->stream);
        if (rc == AXIOM_OK) session->stream_synchronized = false;
    }
    if (rc == AXIOM_OK) {
        *failure_stage = "graph_executor_recycle_commit_sync";
        rc = cuda_status(cudaStreamSynchronize(session->stream));
        if (rc == AXIOM_OK) session->stream_synchronized = true;
    }
    if (rc == AXIOM_OK) {
        *failure_stage = "graph_executor_recycle_verify";
        rc = request_session_position_verify(state, session, 0u, failure_stage);
    }
    return rc;
}

int store_resident_graph_session(
        server_state *state, request_session *session) {
    if (!state || !session || !request_session_ready(session) ||
        !session->stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!state->resident_graph_session) {
        state->resident_graph_session = new (std::nothrow) request_session();
        if (!state->resident_graph_session) return AXIOM_ERR_BUDGET;
    }
    if (request_session_ready(state->resident_graph_session)) return AXIOM_ERR_RUNTIME;
    *state->resident_graph_session = *session;
    *session = request_session{};
    state->graph_executor_resident.store(true, std::memory_order_release);
    state->graph_executor_recycles.fetch_add(1u);
    return AXIOM_OK;
}

int enqueue_history_copy(
        cycle_history *history,
        const axiom_qwen38_speculative_device_step_result &result,
        cudaStream_t stream) {
    if (!history || !result.history_device || !stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const cudaError_t status = cudaMemcpyAsync(
            history, result.history_device, sizeof(*history),
            cudaMemcpyDeviceToDevice, stream);
    return cuda_status(status);
}

int enqueue_native_history_snapshot(
        native_cycle_history *history,
        const axiom_qwen38_mtp_speculative_device_step_result &result,
        cudaStream_t stream) {
    if (!history || !stream || !result.draft_token_ids_device ||
        !result.verify_token_ids_device || !result.emitted_token_ids_device ||
        !result.accepted_prefix_device || !result.target_commit_count_device ||
        !result.emitted_token_count_device || !result.continuation_token_device ||
        !result.stop_detected_device || !result.matched_stop_token_device ||
        !result.next_anchor_token_device || !result.next_anchor_position_device ||
        !result.async_status_device || !result.continuation_logit_device) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    auto copy = [stream](void *destination, const void *source,
                         const size_t bytes) -> int {
        return cuda_status(cudaMemcpyAsync(
                destination, source, bytes, cudaMemcpyDeviceToDevice, stream));
    };
    int rc = copy(history->draft, result.draft_token_ids_device,
                  sizeof(history->draft));
    if (rc == AXIOM_OK) rc = copy(
            history->verify, result.verify_token_ids_device,
            sizeof(history->verify));
    if (rc == AXIOM_OK) rc = copy(
            history->emitted, result.emitted_token_ids_device,
            sizeof(history->emitted));
    if (rc == AXIOM_OK) rc = copy(
            &history->accepted_prefix, result.accepted_prefix_device,
            sizeof(history->accepted_prefix));
    if (rc == AXIOM_OK) rc = copy(
            &history->commit_count, result.target_commit_count_device,
            sizeof(history->commit_count));
    if (rc == AXIOM_OK) rc = copy(
            &history->emitted_count, result.emitted_token_count_device,
            sizeof(history->emitted_count));
    if (rc == AXIOM_OK) rc = copy(
            &history->continuation, result.continuation_token_device,
            sizeof(history->continuation));
    if (rc == AXIOM_OK) rc = copy(
            &history->stop_detected, result.stop_detected_device,
            sizeof(history->stop_detected));
    if (rc == AXIOM_OK) rc = copy(
            &history->matched_stop_token, result.matched_stop_token_device,
            sizeof(history->matched_stop_token));
    if (rc == AXIOM_OK) rc = copy(
            &history->next_anchor, result.next_anchor_token_device,
            sizeof(history->next_anchor));
    if (rc == AXIOM_OK) rc = copy(
            &history->next_position, result.next_anchor_position_device,
            sizeof(history->next_position));
    if (rc == AXIOM_OK) rc = copy(
            &history->async_status, result.async_status_device,
            sizeof(history->async_status));
    if (rc == AXIOM_OK) rc = copy(
            &history->continuation_logit, result.continuation_logit_device,
            sizeof(history->continuation_logit));
    return rc;
}

int enqueue_native_history_host_copy(
        request_session *session, const uint32_t count) {
    if (!session || session->engine != speculative_engine::native_mtp ||
        !session->stream || !session->mtp_history_device ||
        !session->mtp_history_host || count == 0u ||
        count > session->history_capacity) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return cuda_status(cudaMemcpyAsync(
            session->mtp_history_host, session->mtp_history_device,
            static_cast<size_t>(count) * sizeof(native_cycle_history),
            cudaMemcpyDeviceToHost, session->stream));
}

int observe_draft_confidence(
        generation_result *result, const cycle_history &history,
        const uint32_t eligible_slots, std::string *failure_stage) {
    if (!result || !failure_stage ||
        eligible_slots > AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS ||
        history.accepted_prefix > eligible_slots) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t slot = 0u;
         slot < AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS; ++slot) {
        const double confidence = static_cast<double>(history.proposal_confidence[slot]);
        if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0) {
            *failure_stage = "device_confidence_range";
            return AXIOM_ERR_RUNTIME;
        }
    }
    /* Once one draft token differs, later target rows were evaluated under a
     * divergent draft prefix.  Only the accepted prefix and first mismatch
     * are valid binary labels for calibration. */
    uint32_t observed_slots = history.accepted_prefix;
    if (history.accepted_prefix < eligible_slots) ++observed_slots;
    for (uint32_t slot = 0u; slot < observed_slots; ++slot) {
        const double confidence = static_cast<double>(history.proposal_confidence[slot]);
        const bool accepted = slot < history.accepted_prefix;
        const double label = accepted ? 1.0 : 0.0;
        const double error = confidence - label;
        const uint32_t bin = std::min<uint32_t>(
                9u, static_cast<uint32_t>(confidence * 10.0));
        ++result->draft_confidence_samples;
        result->draft_confidence_sum += confidence;
        result->draft_confidence_brier_sum += error * error;
        result->draft_confidence_min = std::min(
                result->draft_confidence_min, confidence);
        result->draft_confidence_max = std::max(
                result->draft_confidence_max, confidence);
        ++result->draft_confidence_bin_samples[bin];
        result->draft_confidence_bin_sum[bin] += confidence;
        if (accepted) {
            ++result->draft_confidence_accepted_samples;
            result->draft_confidence_accepted_sum += confidence;
            ++result->draft_confidence_bin_accepted[bin];
        } else {
            ++result->draft_confidence_rejected_samples;
            result->draft_confidence_rejected_sum += confidence;
        }
    }
    return AXIOM_OK;
}

void merge_draft_confidence(
        generation_result *out, const generation_result &additional) {
    if (!out || additional.draft_confidence_samples == 0u) return;
    if (out->draft_confidence_samples == 0u) {
        out->draft_confidence_min = additional.draft_confidence_min;
        out->draft_confidence_max = additional.draft_confidence_max;
    } else {
        out->draft_confidence_min = std::min(
                out->draft_confidence_min, additional.draft_confidence_min);
        out->draft_confidence_max = std::max(
                out->draft_confidence_max, additional.draft_confidence_max);
    }
    out->draft_confidence_samples += additional.draft_confidence_samples;
    out->draft_confidence_accepted_samples +=
            additional.draft_confidence_accepted_samples;
    out->draft_confidence_rejected_samples +=
            additional.draft_confidence_rejected_samples;
    out->draft_confidence_sum += additional.draft_confidence_sum;
    out->draft_confidence_accepted_sum +=
            additional.draft_confidence_accepted_sum;
    out->draft_confidence_rejected_sum +=
            additional.draft_confidence_rejected_sum;
    out->draft_confidence_brier_sum += additional.draft_confidence_brier_sum;
    for (size_t bin = 0u; bin < out->draft_confidence_bin_samples.size(); ++bin) {
        out->draft_confidence_bin_samples[bin] +=
                additional.draft_confidence_bin_samples[bin];
        out->draft_confidence_bin_accepted[bin] +=
                additional.draft_confidence_bin_accepted[bin];
        out->draft_confidence_bin_sum[bin] +=
                additional.draft_confidence_bin_sum[bin];
    }
}

bool parse_u32(const char *text, uint32_t *out) {
    if (!text || !out || !*text) return false;
    uint64_t value = 0u;
    for (const char *cursor = text; *cursor != '\0'; ++cursor) {
        const unsigned char character = static_cast<unsigned char>(*cursor);
        if (character < '0' || character > '9') return false;
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10u) {
            return false;
        }
        value = value * 10u + digit;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

bool parse_u64(const char *text, uint64_t *out) {
    if (!text || !out || !*text) return false;
    uint64_t value = 0u;
    for (const char *cursor = text; *cursor != '\0'; ++cursor) {
        const unsigned char character = static_cast<unsigned char>(*cursor);
        if (character < '0' || character > '9') return false;
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10u) {
            return false;
        }
        value = value * 10u + digit;
    }
    *out = value;
    return true;
}

bool env_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && (value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
                     value[0] == 'y' || value[0] == 'Y');
}

bool env_disabled(const char *name) {
    const char *value = std::getenv(name);
    return value && (value[0] == '0' || value[0] == 'f' || value[0] == 'F');
}

bool session_persistence_enabled(const server_state *state) {
    return state && state->streaming_kv && state->session_store.enabled();
}

int create_kv_tier_at_path(
        server_state *state,
        const std::string &path,
        axiom_qwen38_kv_tier **out_tier,
        std::string *stage) {
    if (!state || !out_tier || !stage || path.empty()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_tier = nullptr;
    uint32_t hot_pages = state->kv_hot_pages;
    const uint32_t logical_pages =
            (state->max_context + AXIOM_QWEN38_KV_TIER_PAGE_TOKENS - 1u) /
            AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
    if (hot_pages > logical_pages) hot_pages = logical_pages;
    axiom_qwen38_kv_tier_config config{};
    config.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    config.path = path.c_str();
    config.max_context = state->max_context;
    config.hot_pages = hot_pages;
    config.queue_depth = kKvTierQueueDepth;
    config.flags = AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_DIRECT |
            AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_IO_URING;
    const char *preallocate = std::getenv("AXIOM_QWEN38_KV_TIER_PREALLOCATE");
    if (preallocate && std::strcmp(preallocate, "1") == 0) {
        config.flags |= AXIOM_QWEN38_KV_TIER_FLAG_PREALLOCATE;
    }
    *stage = "kv_tier_create";
    int rc = axiom_qwen38_kv_tier_create(&config, out_tier);
    if (rc != AXIOM_OK) return rc;
    axiom_qwen38_kv_tier_info info{};
    info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    rc = axiom_qwen38_kv_tier_info_get(*out_tier, &info);
    if (rc != AXIOM_OK) return rc;
    std::fprintf(stderr,
                 "axiom-qwen38-api: NVMe KV tier enabled path=%s backend=%u direct=%u "
                 "io_uring=1 max_context=%u pages=%u hot_pages=%u\n",
                 path.c_str(), info.backend, info.direct_io, state->max_context,
                 info.plan.logical_pages, info.plan.hot_pages);
    return AXIOM_OK;
}

int activate_ephemeral_target_tier(
        server_state *state, std::string *stage) {
    if (!state || !state->model || !stage || state->kv_base_path.empty()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const std::string path = state->kv_base_path + ".target-ephemeral";
    axiom_qwen38_kv_tier *fresh = nullptr;
    int rc = create_kv_tier_at_path(state, path, &fresh, stage);
    if (rc == AXIOM_OK) {
        *stage = "target_ephemeral_tier_reset";
        rc = axiom_qwen38_kv_tier_reset(fresh);
    }
    if (rc == AXIOM_OK) {
        *stage = "target_ephemeral_model_reset";
        rc = axiom_qwen38_model_reset(state->model);
    }
    axiom_qwen38_kv_tier *previous = state->kv_tier;
    if (rc == AXIOM_OK) {
        *stage = "target_ephemeral_tier_bind";
        rc = axiom_qwen38_model_kv_tier_bind(state->model, fresh);
    }
    if (rc != AXIOM_OK) {
        if (previous) (void)axiom_qwen38_model_kv_tier_bind(state->model, previous);
        axiom_qwen38_kv_tier_destroy(fresh);
        return rc;
    }
    state->kv_tier = fresh;
    if (previous && previous != fresh) axiom_qwen38_kv_tier_destroy(previous);
    {
        std::lock_guard<std::mutex> status_lock(state->session_status_mutex);
        state->active_session_key = {};
        state->active_session_paths = {};
        state->active_session_manifest = {};
        state->active_session_manifest_exists = false;
        state->active_session_generation = 0u;
    }
    return AXIOM_OK;
}

struct pending_kv_page {
    void *host_page = nullptr;
};

int drain_kv_writes(
        axiom_qwen38_kv_tier *tier,
        std::vector<pending_kv_page> *pending) {
    if (!tier || !pending) return AXIOM_ERR_INVALID_ARGUMENT;
    if (pending->empty()) return AXIOM_OK;
    std::vector<axiom_qwen38_kv_tier_completion> completions(pending->size());
    uint32_t completed = 0u;
    const int wait_rc = axiom_qwen38_kv_tier_wait(
            tier, static_cast<uint32_t>(pending->size()), completions.data(),
            static_cast<uint32_t>(completions.size()), &completed);
    int rc = wait_rc;
    if (rc == AXIOM_OK && completed != pending->size()) rc = AXIOM_ERR_IO;
    for (uint32_t index = 0u; index < completed && rc == AXIOM_OK; ++index) {
        if (completions[index].result != AXIOM_OK ||
            completions[index].transferred_bytes != AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES) {
            rc = AXIOM_ERR_IO;
        }
    }
    for (pending_kv_page &entry : *pending) {
        axiom_qwen38_kv_tier_host_page_free(entry.host_page);
        entry.host_page = nullptr;
    }
    pending->clear();
    return rc;
}

int persist_kv_pages(
        server_state *state,
        request_session *session,
        const uint32_t committed_tokens,
        const uint32_t start_position,
        std::string *failure_stage,
        const bool publish_watermark = true) {
    if (!state || !session || !failure_stage) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!state->kv_tier) return AXIOM_OK;
    if (!request_session_ready(session) ||
        committed_tokens > effective_context_limit(state) ||
        start_position > committed_tokens) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t pages = (committed_tokens + AXIOM_QWEN38_KV_TIER_PAGE_TOKENS - 1u) /
            AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
    std::vector<pending_kv_page> pending;
    pending.reserve(kKvTierQueueDepth);
    auto submit_page = [&](const uint32_t family, const uint32_t layer,
                           const uint32_t logical_page) -> int {
        if (pending.size() >= kKvTierQueueDepth) {
            const int drain_rc = drain_kv_writes(state->kv_tier, &pending);
            if (drain_rc != AXIOM_OK) return drain_rc;
        }
        const uint64_t page_bytes = axiom_qwen38_kv_tier_family_page_bytes(family);
        void *host_page = nullptr;
        int rc = axiom_qwen38_kv_tier_host_page_alloc(page_bytes, &host_page);
        if (rc == AXIOM_OK && family == AXIOM_QWEN38_KV_TIER_TARGET) {
            rc = axiom_qwen38_model_kv_page_export(
                    state->model, layer, logical_page, host_page, page_bytes);
        } else if (rc == AXIOM_OK &&
                   session->engine == speculative_engine::native_mtp) {
            rc = axiom_qwen38_mtp_compute_kv_page_export(
                    session->mtp_compute, layer, logical_page,
                    host_page, page_bytes);
        } else if (rc == AXIOM_OK) {
            rc = axiom_qwen38_dspark_compute_kv_page_export(
                    session->compute, layer, logical_page,
                    host_page, page_bytes);
        }
        if (rc == AXIOM_OK) {
            axiom_qwen38_kv_tier_request request{};
            request.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
            request.operation = AXIOM_QWEN38_KV_TIER_WRITE;
            request.family = family;
            request.layer = layer;
            request.logical_page = logical_page;
            request.host_page = host_page;
            request.host_page_bytes = page_bytes;
            rc = axiom_qwen38_kv_tier_submit(state->kv_tier, &request);
        }
        if (rc != AXIOM_OK) {
            axiom_qwen38_kv_tier_host_page_free(host_page);
            return rc;
        }
        pending.push_back(pending_kv_page{host_page});
        return AXIOM_OK;
    };

    int rc = AXIOM_OK;
    const uint32_t first_page = start_position / AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
    for (uint32_t logical_page = first_page;
         logical_page < pages && rc == AXIOM_OK; ++logical_page) {
        for (uint32_t layer = 0u;
             layer < AXIOM_QWEN38_KV_TIER_TARGET_LAYERS && rc == AXIOM_OK; ++layer) {
            rc = submit_page(AXIOM_QWEN38_KV_TIER_TARGET, layer, logical_page);
        }
        const uint32_t speculative_layers =
                session->engine == speculative_engine::native_mtp
                ? AXIOM_QWEN38_MTP_COMPUTE_KV_LAYERS
                : AXIOM_QWEN38_KV_TIER_DSPARK_LAYERS;
        for (uint32_t layer = 0u;
             layer < speculative_layers && rc == AXIOM_OK; ++layer) {
            rc = submit_page(AXIOM_QWEN38_KV_TIER_DSPARK, layer, logical_page);
        }
    }
    const int drain_rc = drain_kv_writes(state->kv_tier, &pending);
    if (rc == AXIOM_OK) rc = drain_rc;
    if (rc == AXIOM_OK && publish_watermark) {
        rc = axiom_qwen38_kv_tier_commit(state->kv_tier, committed_tokens);
    }
    if (rc != AXIOM_OK) *failure_stage = "kv_tier_write_or_commit";
    return rc;
}

void run_session_generation_gc(
        server_state *state,
        const axiom::qwen38::qwen38_session_key &key,
        const axiom::qwen38::qwen38_session_paths &paths,
        const char *phase) {
    if (!state || !state->session_store.enabled()) return;
    uint64_t removed_files = 0u;
    std::string gc_error;
    const int gc_rc = state->session_store.prune_obsolete_generations(
            key, paths, &removed_files, &gc_error);
    ++state->session_gc_runs;
    ++state->session_generation_gc_runs;
    state->session_gc_removed_files.fetch_add(removed_files);
    if (gc_rc != AXIOM_OK) {
        ++state->session_gc_failures;
        std::fprintf(stderr,
                     "axiom-qwen38-api: %s persistent KV generation GC failed: %s\n",
                     phase ? phase : "runtime", gc_error.c_str());
    }
}

void run_session_lifecycle_gc(
        server_state *state,
        const std::string &protected_namespace,
        const bool force) {
    if (!state || !state->session_store.enabled() ||
        (state->session_ttl_seconds == 0u && state->session_max_namespaces == 0u)) {
        return;
    }
    const std::time_t wall_time = std::time(nullptr);
    if (wall_time <= 0) {
        ++state->session_gc_failures;
        return;
    }
    const uint64_t now = static_cast<uint64_t>(wall_time);
    const uint64_t last = state->session_gc_last_run_unix.load();
    if (!force && last != 0u && now >= last &&
        now - last < state->session_gc_interval_seconds) {
        return;
    }
    state->session_gc_last_run_unix.store(now);
    const auto started = std::chrono::steady_clock::now();
    axiom::qwen38::qwen38_session_gc_policy policy;
    policy.now_unix_seconds = now;
    policy.ttl_seconds = state->session_ttl_seconds;
    policy.max_sessions = state->session_max_namespaces;
    std::unique_lock<std::mutex> lease_lock(state->session_lease_mutex);
    policy.protected_namespace_ids.reserve(
            state->session_namespace_leases.size() + 1u);
    for (const auto &lease : state->session_namespace_leases) {
        policy.protected_namespace_ids.push_back(lease.first);
    }
    if (!protected_namespace.empty()) {
        policy.protected_namespace_ids.push_back(protected_namespace);
    }
    axiom::qwen38::qwen38_session_gc_result result;
    std::string gc_error;
    const int gc_rc = state->session_store.prune_stale_sessions(
            policy, &result, &gc_error);
    lease_lock.unlock();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started).count();
    state->session_gc_last_duration_ms.store(
            elapsed > 0 ? static_cast<uint64_t>(elapsed) : 0u);
    ++state->session_gc_runs;
    ++state->session_lifecycle_gc_runs;
    state->session_gc_removed_files.fetch_add(result.removed_files);
    state->session_gc_removed_sessions.fetch_add(result.removed_sessions);
    state->session_gc_reclaimed_bytes.fetch_add(result.reclaimed_bytes);
    state->session_gc_scanned_sessions.fetch_add(result.scanned_sessions);
    state->session_gc_protected_sessions.fetch_add(result.protected_sessions);
    state->session_gc_unsafe_namespaces.fetch_add(result.unsafe_namespaces);
    if (gc_rc != AXIOM_OK) {
        ++state->session_gc_failures;
        std::fprintf(stderr,
                     "axiom-qwen38-api: persistent KV session lifecycle GC failed: %s\n",
                     gc_error.c_str());
        return;
    }
    if (result.removed_sessions != 0u || result.removed_files != 0u ||
        result.unsafe_namespaces != 0u) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: session lifecycle GC scanned=%llu retained=%llu "
                     "protected=%llu unsafe=%llu removed_sessions=%llu "
                     "removed_files=%llu reclaimed_bytes=%llu detail=%s\n",
                     static_cast<unsigned long long>(result.scanned_sessions),
                     static_cast<unsigned long long>(result.retained_sessions),
                     static_cast<unsigned long long>(result.protected_sessions),
                     static_cast<unsigned long long>(result.unsafe_namespaces),
                     static_cast<unsigned long long>(result.removed_sessions),
                     static_cast<unsigned long long>(result.removed_files),
                     static_cast<unsigned long long>(result.reclaimed_bytes),
                     gc_error.empty() ? "none" : gc_error.c_str());
    }
}

bool native_generation_active(const server_state *state) {
    if (!state) return false;
    std::lock_guard<std::mutex> lock(state->active_request_mutex);
    return state->active_request_sequence != 0u;
}

void session_maintenance_worker_loop(server_state *state) {
    if (!state) return;
    for (;;) {
        std::string protected_namespace;
        bool force = false;
        {
            std::unique_lock<std::mutex> lock(state->maintenance_mutex);
            state->maintenance_cv.wait(lock, [state] {
                return state->maintenance_stop || state->lifecycle_gc_requested;
            });
            while (!state->maintenance_stop) {
                lock.unlock();
                const bool busy = native_generation_active(state);
                lock.lock();
                if (!busy) break;
                state->lifecycle_gc_deferred.fetch_add(1u);
                state->maintenance_cv.wait_for(
                        lock, std::chrono::milliseconds(25), [state] {
                            return state->maintenance_stop;
                        });
            }
            if (state->maintenance_stop) return;
            protected_namespace = state->lifecycle_gc_protected_namespace;
            force = state->lifecycle_gc_force;
            state->lifecycle_gc_requested = false;
            state->lifecycle_gc_force = false;
            state->lifecycle_gc_protected_namespace.clear();
        }
        state->lifecycle_gc_active.store(true, std::memory_order_release);
        try {
            run_session_lifecycle_gc(state, protected_namespace, force);
        } catch (const std::exception &exception) {
            state->session_gc_failures.fetch_add(1u);
            std::fprintf(stderr,
                         "axiom-qwen38-api: lifecycle GC worker exception: %s\n",
                         exception.what());
        } catch (...) {
            state->session_gc_failures.fetch_add(1u);
            std::fprintf(stderr,
                         "axiom-qwen38-api: lifecycle GC worker exception: unknown\n");
        }
        state->lifecycle_gc_active.store(false, std::memory_order_release);
    }
}

bool start_session_maintenance_worker(server_state *state) {
    if (!state) return false;
    if (state->maintenance_worker.joinable()) return true;
    try {
        state->maintenance_worker = std::thread(
                session_maintenance_worker_loop, state);
    } catch (...) {
        return false;
    }
    return true;
}

void schedule_session_lifecycle_gc(
        server_state *state, const std::string &protected_namespace,
        const bool force = false) {
    if (!state || !state->session_store.enabled()) return;
    {
        std::lock_guard<std::mutex> lock(state->maintenance_mutex);
        if (state->maintenance_stop) return;
        state->lifecycle_gc_requested = true;
        state->lifecycle_gc_force = state->lifecycle_gc_force || force;
        if (!protected_namespace.empty()) {
            state->lifecycle_gc_protected_namespace = protected_namespace;
        }
    }
    state->maintenance_cv.notify_one();
}

void stop_session_maintenance_worker(server_state *state) {
    if (!state) return;
    {
        std::lock_guard<std::mutex> lock(state->maintenance_mutex);
        state->maintenance_stop = true;
    }
    state->maintenance_cv.notify_all();
    if (state->maintenance_worker.joinable()) state->maintenance_worker.join();
}

int rollback_session_transaction(
        server_state *state,
        session_activation *activation,
        std::string *failure_stage) {
    if (!activation || !activation->transaction_active) return AXIOM_OK;
    if (!state || !state->kv_tier || !failure_stage ||
        activation->paths.transaction_path.empty()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const int rc = axiom_qwen38_kv_tier_transaction_rollback(
            state->kv_tier, activation->paths.transaction_path.c_str());
    if (rc == AXIOM_OK) {
        activation->transaction_active = false;
    } else {
        *failure_stage = "session_transaction_rollback";
    }
    return rc;
}

int save_session_manifest(
        server_state *state,
        request_session *session,
        const session_activation &activation,
        const std::vector<uint32_t> &prompt_ids,
        const generation_result &result,
        std::string *failure_stage) {
    if (!state || !state->model || !state->kv_tier || !failure_stage ||
        (state->engine == speculative_engine::native_mtp &&
         !request_session_ready(session)) ||
        result.session_position > effective_context_limit(state)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::vector<uint32_t> history;
    try {
        history.reserve(prompt_ids.size() + result.ids.size());
        history.insert(history.end(), prompt_ids.begin(), prompt_ids.end());
        history.insert(history.end(), result.ids.begin(), result.ids.end());
    } catch (...) {
        *failure_stage = "session_manifest_history_alloc";
        return AXIOM_ERR_BUDGET;
    }
    if (result.session_position > history.size()) {
        *failure_stage = "session_manifest_history_alignment";
        return AXIOM_ERR_RUNTIME;
    }
    history.resize(result.session_position);
    std::vector<uint8_t> recurrent;
    try {
        recurrent.resize(axiom_qwen38_model_recurrent_state_bytes());
    } catch (...) {
        *failure_stage = "session_manifest_recurrent_alloc";
        return AXIOM_ERR_BUDGET;
    }
    *failure_stage = "session_manifest_recurrent_export";
    int rc = axiom_qwen38_model_recurrent_state_export(
            state->model, recurrent.data(), recurrent.size());
    if (rc != AXIOM_OK) return rc;
    std::vector<uint8_t> speculative_state;
    if (state->engine == speculative_engine::native_mtp) {
        try {
            speculative_state.resize(
                    sizeof(axiom_qwen38_mtp_speculative_persistent_state_v1));
        } catch (...) {
            *failure_stage = "session_manifest_native_mtp_state_alloc";
            return AXIOM_ERR_BUDGET;
        }
        auto *snapshot = reinterpret_cast<
                axiom_qwen38_mtp_speculative_persistent_state_v1 *>(
                        speculative_state.data());
        snapshot->abi_version =
                AXIOM_QWEN38_MTP_SPECULATIVE_PERSISTENT_STATE_ABI_VERSION;
        snapshot->struct_size = sizeof(*snapshot);
        *failure_stage = "session_manifest_native_mtp_state_export";
        rc = axiom_qwen38_mtp_speculative_persistent_state_export_v1(
                session->mtp_speculative, snapshot, speculative_state.size());
        if (rc != AXIOM_OK) return rc;
    }
    axiom::qwen38::qwen38_session_manifest manifest;
    manifest.generation = activation.paths.generation;
    manifest.committed_tokens = result.session_position;
    manifest.next_token = result.session_next_token;
    manifest.session_id = activation.key.session_id;
    manifest.namespace_id =
            axiom::qwen38::qwen38_persistent_session_store::namespace_id(activation.key);
    manifest.namespace_text =
            axiom::qwen38::qwen38_persistent_session_store::namespace_text(activation.key);
    manifest.profile = activation.key.profile;
    manifest.token_ids = std::move(history);
    manifest.recurrent_state = std::move(recurrent);
    manifest.speculative_state = std::move(speculative_state);
    *failure_stage = "session_manifest_publish";
    rc = state->session_store.save(
            activation.key, activation.paths, manifest, failure_stage);
    if (rc == AXIOM_OK) {
        {
            std::lock_guard<std::mutex> status_lock(state->session_status_mutex);
            state->active_session_manifest = std::move(manifest);
            state->active_session_manifest_exists = true;
            state->active_session_generation = manifest.generation;
        }
        ++state->session_manifest_writes;
        run_session_generation_gc(
                state, activation.key, activation.paths, "runtime");
        schedule_session_lifecycle_gc(
                state, activation.paths.namespace_id, false);
    }
    return rc;
}

void run_deferred_persistence_job_body(
        const std::shared_ptr<deferred_persistence_job> &job,
        bool lock_session = true) {
    if (!job || !job->state) return;
    server_state *state = job->state;
    std::unique_lock<std::mutex> session_lock(
            state->session_mutex, std::defer_lock);
    if (lock_session) session_lock.lock();
    const auto persistence_begin = std::chrono::steady_clock::now();
    auto phase_begin = persistence_begin;
    double replay_ms = 0.0, flush_ms = 0.0, manifest_ms = 0.0;
    double discard_ms = 0.0, reset_ms = 0.0;
    auto phase_ms = [&]() {
        const auto now = std::chrono::steady_clock::now();
        const double value = std::chrono::duration<double, std::milli>(
                now - phase_begin).count();
        phase_begin = now;
        return value;
    };
    int rc = AXIOM_OK;
    std::string failure_stage;
    bool manifest_publish_attempted = false;
    bool manifest_published = false;
    uint32_t persist_start_position = job->persist_start_position;
    const uint32_t visible_position = durable_session_position(
            job->prompt_ids.size(), job->result);

    /* Sampling/target fallback does not carry a speculative controller out of
     * the request. Rebuild the exact visible prefix through native MTP before
     * publishing anything so target KV, MTP KV and pending-hidden state share
     * one watermark. This is intentionally asynchronous and never substitutes
     * target-only state into a native namespace. */
    if (job->persistent_session && job->scalar_path &&
        state->engine == speculative_engine::native_mtp &&
        !request_session_ready(&job->session)) {
        failure_stage = "session_native_mtp_scalar_replay";
        rc = replay_exact_session_prefix(
                state, job->prompt_ids, job->result, &job->session,
                &job->activation, nullptr, &job->result.session_position,
                &job->result.session_next_token, &failure_stage);
        if (rc == AXIOM_OK) {
            persist_start_position = job->activation.restored
                    ? job->activation.start_position : 0u;
        }
    }

    if (rc == AXIOM_OK && job->persistent_session &&
        request_session_ready(&job->session) &&
        state->kv_tier &&
        job->graph_position != visible_position) {
        failure_stage = "session_exact_replay";
        rc = replay_exact_session_prefix(
                state, job->prompt_ids, job->result, &job->session,
                &job->activation, job->vision.get(),
                &job->result.session_position, &job->result.session_next_token,
                &failure_stage);
        if (rc == AXIOM_OK) {
            persist_start_position = job->activation.restored
                    ? job->activation.start_position : 0u;
            std::fprintf(stderr,
                         "axiom-qwen38-api: deferred exact replay "
                         "visible_position=%u graph_position=%u\n",
                         job->result.session_position, visible_position);
        }
    }

    replay_ms = phase_ms();
    if (rc == AXIOM_OK && state->kv_tier) {
        if (request_session_ready(&job->session)) {
            failure_stage = "kv_tier_flush";
            rc = persist_kv_pages(
                    state, &job->session, job->result.session_position,
                    persist_start_position, &failure_stage);
        } else if (job->scalar_path) {
            failure_stage = "kv_tier_flush";
            rc = axiom_qwen38_model_kv_tier_flush(
                    state->model, axiom_qwen38_model_position(state->model));
        }
    }
    flush_ms = phase_ms();
    if (rc == AXIOM_OK && state->kv_tier && job->persistent_session) {
        failure_stage = "session_manifest_publish";
        manifest_publish_attempted = true;
        rc = save_session_manifest(
                state, &job->session, job->activation,
                job->prompt_ids, job->result,
                &failure_stage);
        manifest_published = rc == AXIOM_OK;
    }
    manifest_ms = phase_ms();
    if (rc == AXIOM_OK && manifest_published &&
        job->activation.transaction_active) {
        failure_stage = "session_transaction_discard";
        rc = axiom_qwen38_kv_tier_transaction_discard(
                job->activation.paths.transaction_path.c_str());
        if (rc == AXIOM_OK) job->activation.transaction_active = false;
    }
    if (rc != AXIOM_OK && job->activation.transaction_active &&
        !manifest_publish_attempted) {
        const int rollback_rc = rollback_session_transaction(
                state, &job->activation, &failure_stage);
        if (rollback_rc != AXIOM_OK) rc = rollback_rc;
    }
    if (rc != AXIOM_OK) {
        state->healthy.store(false);
        state->session_persistence_failures.fetch_add(1u);
        std::fprintf(stderr,
                     "axiom-qwen38-api: deferred persistence failed at %s: %s\n",
                     failure_stage.empty() ? "unknown" : failure_stage.c_str(),
                     axiom_status_string(rc));
    } else {
        state->session_persistence_completed.fetch_add(1u);
    }

    discard_ms = phase_ms();
    const int reset_rc = axiom_qwen38_model_reset(state->model);
    if (reset_rc != AXIOM_OK) {
        state->healthy.store(false);
        if (rc == AXIOM_OK) {
            rc = reset_rc;
            state->session_persistence_failures.fetch_add(1u);
        }
        std::fprintf(stderr,
                     "axiom-qwen38-api: deferred persistence model reset failed: %s\n",
                     axiom_status_string(reset_rc));
    }

    reset_ms = phase_ms();
    bool executor_recycled = false;
    if (rc == AXIOM_OK && reset_rc == AXIOM_OK &&
        request_session_ready(&job->session) &&
        job->session.stream && !job->scalar_path) {
        std::string recycle_stage;
        int recycle_rc = prepare_graph_session_for_reuse(
                state, &job->session, &recycle_stage);
        if (recycle_rc == AXIOM_OK) {
            recycle_rc = store_resident_graph_session(state, &job->session);
        }
        if (recycle_rc == AXIOM_OK) {
            executor_recycled = true;
        } else {
            state->graph_executor_recycle_failures.fetch_add(1u);
            std::fprintf(stderr,
                         "axiom-qwen38-api: resident graph executor recycle "
                         "skipped at %s: %s\n",
                         recycle_stage.empty() ? "store" : recycle_stage.c_str(),
                         axiom_status_string(recycle_rc));
        }
    }

    const int destroy_rc = !executor_recycled && request_session_ready(&job->session)
            ? destroy_request_session(&job->session) : AXIOM_OK;
    if (destroy_rc != AXIOM_OK) {
        if (rc == AXIOM_OK) rc = destroy_rc;
        state->healthy.store(false);
        state->session_persistence_failures.fetch_add(1u);
        std::fprintf(stderr,
                     "axiom-qwen38-api: deferred persistence session cleanup failed: %s\n",
                     axiom_status_string(destroy_rc));
    }
    const double recycle_ms = phase_ms();
    const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - persistence_begin).count();
    {
        std::lock_guard<std::mutex> profile_lock(state->persistence_profile_mutex);
        state->persistence_last_ms = {replay_ms, flush_ms, manifest_ms, discard_ms,
                                      reset_ms, recycle_ms, total_ms};
        ++state->persistence_profile_jobs;
        state->persistence_profile_tokens = visible_position;
        state->persistence_profile_status = rc;
    }
    static const bool trace_persistence = [] {
        const char *value = std::getenv("AXIOM_QWEN38_TRACE_PERSISTENCE");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (trace_persistence) {
        std::fprintf(stderr,
                "axiom-qwen38-api: persistence_timing position=%u rc=%d "
                "replay_ms=%.3f flush_ms=%.3f manifest_ms=%.3f discard_ms=%.3f "
                "reset_ms=%.3f recycle_ms=%.3f total_ms=%.3f\n",
                visible_position, rc, replay_ms, flush_ms, manifest_ms,
                discard_ms, reset_ms, recycle_ms,
                std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - persistence_begin).count());
    }
}

void run_deferred_persistence_job(
        const std::shared_ptr<deferred_persistence_job> &job,
        const bool lock_session = true) noexcept {
    if (!job || !job->state) return;
    server_state *state = job->state;
    try {
        run_deferred_persistence_job_body(job, lock_session);
    } catch (const std::exception &exception) {
        state->healthy.store(false);
        state->session_persistence_failures.fetch_add(1u);
        std::fprintf(stderr,
                     "axiom-qwen38-api: deferred persistence exception: %s\n",
                     exception.what());
        std::unique_lock<std::mutex> recovery_lock(
                state->session_mutex, std::defer_lock);
        if (lock_session) recovery_lock.lock();
        if (request_session_ready(&job->session)) {
            (void)destroy_request_session(&job->session);
        }
        (void)axiom_qwen38_model_reset(state->model);
    } catch (...) {
        state->healthy.store(false);
        state->session_persistence_failures.fetch_add(1u);
        std::fprintf(stderr,
                     "axiom-qwen38-api: deferred persistence exception: unknown\n");
        std::unique_lock<std::mutex> recovery_lock(
                state->session_mutex, std::defer_lock);
        if (lock_session) recovery_lock.lock();
        if (request_session_ready(&job->session)) {
            (void)destroy_request_session(&job->session);
        }
        (void)axiom_qwen38_model_reset(state->model);
    }
    complete_persistence_operation(state);
}

bool enqueue_deferred_persistence(
        const std::shared_ptr<deferred_persistence_job> &job) {
    if (!job || !job->state) return false;
    return enqueue_persistence_job(job->state, [job] {
        run_deferred_persistence_job(job);
    });
}

void run_inline_deferred_persistence(
        const std::shared_ptr<deferred_persistence_job> &job) {
    if (!job || !job->state) return;
    {
        std::lock_guard<std::mutex> lock(job->state->persistence_mutex);
        job->state->session_persistence_queued.fetch_add(1u);
        job->state->session_persistence_pending.fetch_add(
                1u, std::memory_order_release);
    }
    run_deferred_persistence_job(job, false);
}

int read_session_page(
        axiom_qwen38_kv_tier *tier,
        const uint32_t family,
        const uint32_t layer,
        const uint32_t logical_page,
        void *host_page,
        const uint64_t page_bytes) {
    if (!tier || !host_page || page_bytes == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_qwen38_kv_tier_request request{};
    request.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    request.operation = AXIOM_QWEN38_KV_TIER_READ;
    request.family = family;
    request.layer = layer;
    request.logical_page = logical_page;
    request.host_page = host_page;
    request.host_page_bytes = page_bytes;
    int rc = axiom_qwen38_kv_tier_submit(tier, &request);
    if (rc != AXIOM_OK) return rc;
    axiom_qwen38_kv_tier_completion completion{};
    uint32_t completed = 0u;
    rc = axiom_qwen38_kv_tier_wait(tier, 1u, &completion, 1u, &completed);
    if (rc != AXIOM_OK || completed != 1u || completion.result != AXIOM_OK ||
        completion.transferred_bytes != page_bytes) {
        return rc == AXIOM_OK ? AXIOM_ERR_IO : rc;
    }
    return AXIOM_OK;
}

int restore_session_target(
        server_state *state,
        axiom_qwen38_kv_tier *tier,
        const axiom::qwen38::qwen38_session_manifest &manifest,
        const bool temporal_graph,
        std::string *failure_stage) {
    if (!state || !state->model || !tier || !failure_stage ||
        manifest.committed_tokens > effective_context_limit(state) ||
        manifest.recurrent_state.size() != axiom_qwen38_model_recurrent_state_bytes()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (manifest.committed_tokens == 0u) {
        *failure_stage = "session_restore_position";
        int rc = axiom_qwen38_model_restore_position(state->model, 0u);
        if (rc == AXIOM_OK) {
            *failure_stage = "session_restore_history";
            rc = axiom_qwen38_model_committed_history_install(
                    state->model, nullptr, 0u);
        }
        return rc;
    }
    const uint32_t page_count =
            (manifest.committed_tokens + AXIOM_QWEN38_KV_TIER_PAGE_TOKENS - 1u) /
            AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
    const uint32_t hot_pages = std::max<uint32_t>(1u, state->kv_hot_pages);
    /* M8 attention reads the complete resident temporal prefix. Scalar paging
     * needs only its true hot set and faults older pages through the tier. */
    const uint32_t first_page = temporal_graph
            ? 0u : (page_count > hot_pages ? page_count - hot_pages : 0u);
    void *host_page = nullptr;
    int rc = axiom_qwen38_kv_tier_host_page_alloc(
            AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES, &host_page);
    if (rc != AXIOM_OK) {
        *failure_stage = "session_restore_host_page";
        return rc;
    }
    for (uint32_t logical_page = first_page;
         logical_page < page_count && rc == AXIOM_OK; ++logical_page) {
        for (uint32_t layer = 0u;
             layer < AXIOM_QWEN38_KV_TIER_TARGET_LAYERS && rc == AXIOM_OK; ++layer) {
            rc = read_session_page(
                    tier, AXIOM_QWEN38_KV_TIER_TARGET, layer, logical_page,
                    host_page, AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES);
            if (rc == AXIOM_OK) {
                rc = axiom_qwen38_model_kv_page_import(
                        state->model, layer, logical_page, host_page,
                        AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES);
            }
            if (rc == AXIOM_OK) ++state->session_cold_page_reads;
        }
    }
    axiom_qwen38_kv_tier_host_page_free(host_page);
    if (rc == AXIOM_OK) {
        *failure_stage = "session_restore_recurrent";
        rc = axiom_qwen38_model_recurrent_state_import(
                state->model, manifest.recurrent_state.data(),
                manifest.recurrent_state.size());
    }
    if (rc == AXIOM_OK) {
        *failure_stage = "session_restore_position";
        rc = axiom_qwen38_model_restore_position(
                state->model, manifest.committed_tokens);
    }
    if (rc == AXIOM_OK) {
        if (manifest.token_ids.size() < manifest.committed_tokens) {
            rc = AXIOM_ERR_INVALID_ARGUMENT;
        } else {
            *failure_stage = "session_restore_history";
            rc = axiom_qwen38_model_committed_history_install(
                    state->model, manifest.token_ids.data(),
                    manifest.committed_tokens);
        }
    }
    if (rc != AXIOM_OK && failure_stage->empty()) *failure_stage = "session_restore_target";
    return rc;
}

int restore_session_speculative(
        server_state *state,
        request_session *session,
        const axiom::qwen38::qwen38_session_manifest &manifest,
        std::string *failure_stage) {
    if (!state || !session || !request_session_ready(session) || !state->kv_tier ||
        !failure_stage || manifest.committed_tokens > state->speculative_context_tokens) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t page_count =
            (manifest.committed_tokens + AXIOM_QWEN38_KV_TIER_PAGE_TOKENS - 1u) /
            AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
    void *host_page = nullptr;
    int rc = axiom_qwen38_kv_tier_host_page_alloc(
            AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES, &host_page);
    if (rc != AXIOM_OK) {
        *failure_stage = "session_restore_dspark_host_page";
        return rc;
    }
    for (uint32_t logical_page = 0u;
         logical_page < page_count && rc == AXIOM_OK; ++logical_page) {
        const uint32_t speculative_layers =
                session->engine == speculative_engine::native_mtp
                ? AXIOM_QWEN38_MTP_COMPUTE_KV_LAYERS
                : AXIOM_QWEN38_KV_TIER_DSPARK_LAYERS;
        for (uint32_t layer = 0u;
             layer < speculative_layers && rc == AXIOM_OK; ++layer) {
            rc = read_session_page(
                    state->kv_tier, AXIOM_QWEN38_KV_TIER_DSPARK, layer, logical_page,
                    host_page, AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES);
            if (rc == AXIOM_OK) {
                rc = session->engine == speculative_engine::native_mtp
                        ? axiom_qwen38_mtp_compute_kv_page_import(
                                session->mtp_compute, layer, logical_page,
                                host_page,
                                AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES)
                        : axiom_qwen38_dspark_compute_kv_page_import(
                                session->compute, layer, logical_page,
                                host_page,
                                AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES);
            }
            if (rc == AXIOM_OK) ++state->session_cold_page_reads;
        }
    }
    axiom_qwen38_kv_tier_host_page_free(host_page);
    if (rc == AXIOM_OK) {
        *failure_stage = session->engine == speculative_engine::native_mtp
                ? "session_restore_native_mtp_position"
                : "session_restore_dspark_position";
        rc = session->engine == speculative_engine::native_mtp
                ? axiom_qwen38_mtp_compute_restore_position(
                        session->mtp_compute, manifest.committed_tokens)
                : axiom_qwen38_dspark_compute_restore_position(
                        session->compute, manifest.committed_tokens);
    }
    if (rc == AXIOM_OK && session->engine == speculative_engine::native_mtp) {
        if (manifest.speculative_state.size() !=
            sizeof(axiom_qwen38_mtp_speculative_persistent_state_v1)) {
            *failure_stage = "session_restore_native_mtp_state_size";
            rc = AXIOM_ERR_IO;
        } else {
            const auto *snapshot = reinterpret_cast<const
                    axiom_qwen38_mtp_speculative_persistent_state_v1 *>(
                            manifest.speculative_state.data());
            *failure_stage = "session_restore_native_mtp_state";
            rc = axiom_qwen38_mtp_speculative_persistent_state_import_v1(
                    session->mtp_speculative, snapshot,
                    manifest.speculative_state.size());
        }
    }
    return rc;
}

bool session_prompt_matches(
        const axiom::qwen38::qwen38_session_manifest &manifest,
        const std::vector<uint32_t> &prompt_ids) {
    return manifest.committed_tokens != 0u &&
            manifest.committed_tokens <= prompt_ids.size() &&
            manifest.committed_tokens <= manifest.token_ids.size() &&
            std::equal(
                    manifest.token_ids.begin(),
                    manifest.token_ids.begin() + manifest.committed_tokens,
                    prompt_ids.begin());
}

bool project_codex_tool_prefix(server_state *state,
        const axiom::qwen38::qwen38_session_manifest &manifest,
        const ajson &tools, std::vector<uint32_t> *projected);

bool session_equivalent_prompt(
        server_state *state,
        const axiom::qwen38::qwen38_session_manifest &manifest,
        const std::vector<uint32_t> &prompt_ids, uint32_t limit,
        std::vector<uint32_t> *effective, const ajson *codex_tools = nullptr,
        bool *tool_projected = nullptr) {
    if (!state || !state->tokenizer || !effective) return false;
    effective->clear();
    if (tool_projected) *tool_projected = false;
    try {
        std::array<char, 65536> piece{};
        auto decode = [&](uint32_t id, std::string *out) {
            uint32_t bytes = 0;
            if (axiom_tokenizer_decode_token(state->tokenizer, id, piece.data(),
                                             piece.size(), &bytes) != AXIOM_OK) return false;
            out->assign(piece.data(), bytes);
            return true;
        };
        size_t consumed = 0;
        bool projected = false;
        if (!axiom::qwen38::equivalent_prefix(manifest.token_ids,
                manifest.committed_tokens, prompt_ids, state->tokenizer_info.vocab_size,
                decode, &consumed)) {
            std::vector<uint32_t> reference;
            if (!codex_tools || !project_codex_tool_prefix(state, manifest,
                    *codex_tools, &reference) ||
                !axiom::qwen38::equivalent_prefix(reference, reference.size(),
                    prompt_ids, state->tokenizer_info.vocab_size, decode, &consumed))
                return false;
            projected = true;
        }
        const uint64_t total = static_cast<uint64_t>(manifest.committed_tokens) +
                               prompt_ids.size() - consumed;
        if (total > limit) return false;
        effective->reserve(static_cast<size_t>(total));
        effective->insert(effective->end(), manifest.token_ids.begin(),
                          manifest.token_ids.begin() + manifest.committed_tokens);
        effective->insert(effective->end(), prompt_ids.begin() + consumed, prompt_ids.end());
        if (tool_projected) *tool_projected = projected;
        return true;
    } catch (...) {
        effective->clear();
        return false; // Cache optimization failure must not accept a weaker match.
    }
}

axiom::qwen38::qwen38_session_key make_session_key(
        const server_state *state,
        const std::string &session_id,
        const std::string &profile) {
    axiom::qwen38::qwen38_session_key key;
    if (!state) return key;
    key.session_id = session_id;
    key.model_id = kModelId;
    key.target_identity = state->target_identity;
    key.dspark_identity = state->dspark_identity;
    key.config_signature = state->kv_config_signature +
            ";chat_template=" + kChatTemplateVersion;
    key.profile = profile;
    return key;
}

int activate_session(
        server_state *state,
        const std::string &session_id,
        const std::string &profile,
        const std::vector<uint32_t> &prompt_ids,
        const std::vector<uint32_t> *session_suffix_ids,
        const bool restore_target,
        const bool allow_reuse,
        const bool allow_stateful_resume,
        const uint32_t resume_prompt_limit,
        session_activation *out,
        std::string *failure_stage, const ajson *codex_tools = nullptr) {
    if (!state || !state->model || !state->session_store.enabled() || !out ||
        !failure_stage || !state->streaming_kv) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = session_activation{};
    out->key = make_session_key(state, session_id, profile);
    if (!axiom::qwen38::qwen38_persistent_session_store::valid_session_id(session_id)) {
        *failure_stage = "session_id";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    bool exists = false;
    int rc = state->session_store.load(
            out->key, &out->manifest, &out->paths, &exists, failure_stage);
    if (rc != AXIOM_OK) return rc;
    out->manifest_exists = exists;
    if (exists) {
        /* Resolve a crash left between KV watermark commit and atomic
         * manifest publication before validating the pair.  The journal is
         * discarded only when the manifest identifies the new watermark;
         * otherwise its old tail page and header are restored. */
        *failure_stage = "session_transaction_recover";
        rc = axiom_qwen38_kv_tier_transaction_recover_file(
                out->paths.tier_path.c_str(),
                out->paths.transaction_path.c_str(),
                out->manifest.committed_tokens);
        if (rc != AXIOM_OK) return rc;
    }
    if (exists && out->manifest.next_token >= AXIOM_QWEN38_MODEL_VOCAB) {
        *failure_stage = "session_manifest_next_token";
        return AXIOM_ERR_IO;
    }
    /* A manifest currently persists token ids, recurrent state and paged KV,
     * but not the projected visual embeddings.  Reusing a multimodal prefix
     * would therefore replay image/video placeholder ids through the text
     * embedding table.  Callers disable reuse when a vision_request is
     * present; keep this guard explicit at the session boundary so a future
     * caller cannot accidentally bypass the invariant. */
    bool reuse = allow_reuse && exists && session_prompt_matches(out->manifest, prompt_ids);
    bool stateful_resume = false;
    if (exists && out->manifest.committed_tokens == 0u) reuse = false;
    if (!reuse && allow_reuse && exists && session_equivalent_prompt(
            state, out->manifest, prompt_ids, resume_prompt_limit, &out->effective_prompt_ids,
            codex_tools, &out->tool_projection)) {
        reuse = true;
        out->prefix_retokenized = true;
        std::fprintf(stderr,
            "axiom-qwen38-api: %s cache prefix=%u suffix=%zu\n",
            out->tool_projection ? "codex-wire-equivalent" : "tokenization-equivalent",
            out->manifest.committed_tokens,
            out->effective_prompt_ids.size() - out->manifest.committed_tokens);
    }
    /* A client transports only visible assistant content and normalized tool
     * calls.  The durable state contains the exact native bytes, including
     * hidden thinking and the model's original XML whitespace.  When exact
     * wire-prefix matching therefore misses, an explicit stable session id is
     * authoritative: resume the raw durable prefix and append only turns after
     * the latest historical assistant.  Auto-generated one-shot ids, vision
     * requests and compacted prompts never enter this path. */
    if (!reuse && allow_reuse && allow_stateful_resume && exists &&
        session_suffix_ids && !session_suffix_ids->empty()) {
        *failure_stage = "session_stateful_prompt";
        int resume_rc = axiom::qwen38::qwen38_build_stateful_resume_prompt(
                out->manifest, *session_suffix_ids,
                state->tokenizer_info.im_end_token_id,
                state->tokenizer_info.endoftext_token_id,
                AXIOM_QWEN38_MODEL_VOCAB, resume_prompt_limit,
                &out->effective_prompt_ids, &stateful_resume);
        if (resume_rc != AXIOM_OK) return resume_rc;
        reuse = stateful_resume;
    }
    if (reuse && out->manifest.committed_tokens > effective_context_limit(state)) {
        *failure_stage = "session_context_window";
        return AXIOM_ERR_BUDGET;
    }
    uint64_t generation = exists ? out->manifest.generation : 1u;
    if (!reuse && exists) {
        if (generation == std::numeric_limits<uint64_t>::max()) {
            *failure_stage = "session_generation_overflow";
            return AXIOM_ERR_BUDGET;
        }
        ++generation;
    }
    if (!exists || !reuse) {
        out->manifest_exists = false;
        out->manifest = axiom::qwen38::qwen38_session_manifest{};
    }
    rc = state->session_store.prepare(
            out->key, generation, &out->paths, failure_stage);
    if (rc != AXIOM_OK) return rc;

    axiom_qwen38_kv_tier *new_tier = nullptr;
    rc = create_kv_tier_at_path(state, out->paths.tier_path, &new_tier, failure_stage);
    if (rc != AXIOM_OK) return rc;
    axiom_qwen38_kv_tier_info tier_info{};
    tier_info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    rc = axiom_qwen38_kv_tier_info_get(new_tier, &tier_info);
    if (rc == AXIOM_OK && reuse &&
        tier_info.committed_tokens != out->manifest.committed_tokens) {
        *failure_stage = "session_tier_manifest_watermark";
        rc = AXIOM_ERR_IO;
    }
    if (rc == AXIOM_OK && !reuse && tier_info.committed_tokens != 0u) {
        *failure_stage = "session_tier_reset_generation";
        rc = axiom_qwen38_kv_tier_reset(new_tier);
    }
    if (rc == AXIOM_OK) {
        *failure_stage = "session_model_reset";
        rc = axiom_qwen38_model_reset(state->model);
    }
    axiom_qwen38_kv_tier *old_tier = state->kv_tier;
    if (rc == AXIOM_OK) {
        *failure_stage = "session_model_tier_rebind";
        rc = axiom_qwen38_model_kv_tier_bind(state->model, new_tier);
    }
    if (rc == AXIOM_OK && reuse && restore_target) {
        *failure_stage = "session_target_restore";
        rc = restore_session_target(
                state, new_tier, out->manifest, false, failure_stage);
    }
    if (rc == AXIOM_OK && reuse) {
        *failure_stage = "session_transaction_begin";
        rc = axiom_qwen38_kv_tier_transaction_begin(
                new_tier, out->paths.transaction_path.c_str(),
                out->manifest.committed_tokens);
        if (rc == AXIOM_OK) out->transaction_active = true;
    }
    if (rc != AXIOM_OK) {
        if (out->transaction_active) {
            (void)axiom_qwen38_kv_tier_transaction_rollback(
                    new_tier, out->paths.transaction_path.c_str());
            out->transaction_active = false;
        }
        if (old_tier) {
            (void)axiom_qwen38_model_kv_tier_bind(state->model, old_tier);
        }
        axiom_qwen38_kv_tier_destroy(new_tier);
        return rc;
    }
    if (old_tier && old_tier != new_tier) {
        axiom_qwen38_kv_tier_destroy(old_tier);
    }
    state->kv_tier = new_tier;
    {
        std::lock_guard<std::mutex> status_lock(state->session_status_mutex);
        state->active_session_key = out->key;
        state->active_session_paths = out->paths;
        state->active_session_manifest = out->manifest;
        state->active_session_manifest_exists = reuse;
        state->active_session_generation = generation;
    }
    out->restored = reuse;
    out->stateful_resume = stateful_resume;
    out->start_position = reuse ? out->manifest.committed_tokens : 0u;
    out->next_token = reuse ? out->manifest.next_token : 0u;
    if (reuse) ++state->session_restore_count;
    return AXIOM_OK;
}

bool parse_listen(const char *text, listen_spec *out) {
    if (!text || !out || !*text) return false;
    const std::string value(text);
    const size_t colon = value.rfind(':');
    if (colon == std::string::npos || colon == 0u || colon + 1u >= value.size()) return false;
    const std::string host = value.substr(0u, colon);
    const std::string port_text = value.substr(colon + 1u);
    char *end = nullptr;
    const unsigned long port = std::strtoul(port_text.c_str(), &end, 10);
    if (end == port_text.c_str() || *end != '\0' || port == 0u || port > 65535u) return false;
    struct in_addr addr {};
    if (inet_pton(AF_INET, host.c_str(), &addr) != 1) return false;
    /* A wildcard would expose this unauthenticated daemon on docker bridges
     * and on interfaces added after startup. Production uses explicit loopback
     * and LAN sockets instead. */
    if (addr.s_addr == htonl(INADDR_ANY)) return false;
    char canonical[INET_ADDRSTRLEN]{};
    if (!inet_ntop(AF_INET, &addr, canonical, sizeof(canonical))) return false;
    out->host = canonical;
    out->port = static_cast<uint16_t>(port);
    return true;
}

bool same_listener(const listen_spec &left, const listen_spec &right) {
    return left.port == right.port && left.host == right.host;
}

std::string format_listener(const listen_spec &spec) {
    return spec.host + ":" + std::to_string(static_cast<unsigned>(spec.port));
}

std::string format_listener_list(const std::vector<listen_spec> &specs) {
    std::string result;
    for (size_t index = 0u; index < specs.size(); ++index) {
        if (index != 0u) result.push_back(',');
        result += format_listener(specs[index]);
    }
    return result;
}

bool parse_listeners(const char *text, std::vector<listen_spec> *out) {
    if (!text || !out || !*text) return false;
    const std::string value(text);
    std::vector<listen_spec> parsed;
    size_t begin = 0u;
    while (begin < value.size()) {
        if (parsed.size() >= kMaxListeners) return false;
        const size_t comma = value.find(',', begin);
        const size_t end = comma == std::string::npos ? value.size() : comma;
        if (end == begin) return false;
        const std::string item = value.substr(begin, end - begin);
        listen_spec spec{};
        if (!parse_listen(item.c_str(), &spec)) return false;
        for (const listen_spec &existing : parsed) {
            if (same_listener(existing, spec)) return false;
        }
        parsed.push_back(std::move(spec));
        if (comma == std::string::npos) break;
        begin = comma + 1u;
        if (begin == value.size()) return false;
    }
    if (parsed.empty()) return false;
    *out = std::move(parsed);
    return true;
}

std::string ascii_lower(std::string value) {
    for (char &ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

std::string trim_ascii(std::string value) {
    size_t first = 0u;
    while (first < value.size() && (value[first] == ' ' || value[first] == '\t')) ++first;
    size_t last = value.size();
    while (last > first && (value[last - 1u] == ' ' || value[last - 1u] == '\t')) --last;
    return value.substr(first, last - first);
}

bool parse_size(const std::string &text, size_t *out) {
    if (!out || text.empty()) return false;
    size_t value = 0u;
    for (char ch : text) {
        const size_t digit = ch >= '0' && ch <= '9' ? static_cast<size_t>(ch - '0') : 10u;
        if (digit == 10u || value > (std::numeric_limits<size_t>::max() - digit) / 10u) return false;
        value = value * 10u + digit;
    }
    *out = value;
    return true;
}

bool write_all(int fd, const char *data, size_t size) {
    size_t offset = 0u;
    while (offset < size) {
        const ssize_t written = send(fd, data + offset, size - offset, MSG_NOSIGNAL);
        if (written > 0) {
            offset += static_cast<size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

const char *http_reason_phrase(const int status) {
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 501: return "Not Implemented";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    default: return "Internal Server Error";
    }
}

bool send_response_with_type(
        int fd, int status, const std::string &body, const char *content_type,
        const std::string &session_id = {},
        const std::string &extra_headers = {}) {
    const std::string session_header = session_id.empty()
            ? std::string()
            : std::string(kSessionHeader) + ": " + session_id + "\r\n";
    char head[1024]{};
    const int written = std::snprintf(
            head, sizeof(head),
            "HTTP/1.1 %d %s\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n"
            "Server: axiom-qwen38-native\r\n"
            "%s%s\r\n",
            status, http_reason_phrase(status),
            content_type ? content_type : "application/json; charset=utf-8",
            body.size(), session_header.c_str(), extra_headers.c_str());
    return written > 0 && static_cast<size_t>(written) < sizeof(head) &&
            write_all(fd, head, static_cast<size_t>(written)) && write_all(fd, body.data(), body.size());
}

bool send_response(int fd, int status, const std::string &body) {
    return send_response_with_type(fd, status, body, "application/json; charset=utf-8");
}

bool send_session_response(
        int fd, int status, const std::string &body, const std::string &session_id) {
    return send_response_with_type(
            fd, status, body, "application/json; charset=utf-8", session_id);
}

bool send_retry_response(int fd, const std::string &body) {
    return send_response_with_type(
            fd, 429, body, "application/json; charset=utf-8", {},
            "Retry-After: 1\r\n");
}

ajson error_json(const std::string &message, const char *type) {
    ajson error = ajson::jobj();
    error.set("message", ajson::jstr(message));
    error.set("type", ajson::jstr(type));
    ajson root = ajson::jobj();
    root.set("error", std::move(error));
    return root;
}

bool read_request(int fd, http_request *out, int *out_status) {
    if (out_status) *out_status = 400;
    if (!out) return false;
    std::string bytes;
    bytes.reserve(4096u);
    while (bytes.find("\r\n\r\n") == std::string::npos) {
        char chunk[8192];
        const ssize_t got = recv(fd, chunk, sizeof(chunk), 0);
        if (got > 0) {
            bytes.append(chunk, static_cast<size_t>(got));
            if (bytes.size() > kMaxHeaderBytes) {
                if (out_status) *out_status = 413;
                return false;
            }
            continue;
        }
        if (got < 0 && errno == EINTR) continue;
        return false;
    }
    const size_t header_end = bytes.find("\r\n\r\n");
    const std::string header = bytes.substr(0u, header_end);
    size_t line_end = header.find("\r\n");
    const std::string request_line = header.substr(0u, line_end);
    const size_t first_space = request_line.find(' ');
    const size_t second_space = first_space == std::string::npos ? std::string::npos
                                                                  : request_line.find(' ', first_space + 1u);
    if (first_space == std::string::npos || second_space == std::string::npos ||
        request_line.substr(second_space + 1u) != "HTTP/1.1") return false;
    out->method = request_line.substr(0u, first_space);
    out->path = request_line.substr(first_space + 1u, second_space - first_space - 1u);
    const size_t query = out->path.find('?');
    if (query != std::string::npos) out->path.resize(query);

    size_t content_length = 0u;
    bool have_content_length = false;
    size_t pos = line_end == std::string::npos ? header.size() : line_end + 2u;
    while (pos < header.size()) {
        const size_t end = header.find("\r\n", pos);
        const std::string line = header.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        pos = end == std::string::npos ? header.size() : end + 2u;
        const size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0u) return false;
        const std::string name = ascii_lower(line.substr(0u, colon));
        if (name == "content-length") {
            if (have_content_length || !parse_size(trim_ascii(line.substr(colon + 1u)), &content_length)) return false;
            have_content_length = true;
        } else if (name == "x-axiom-session-id") {
            if (!out->session_id.empty()) return false;
            out->session_id = trim_ascii(line.substr(colon + 1u));
            if (!axiom::qwen38::qwen38_persistent_session_store::valid_session_id(
                        out->session_id)) {
                return false;
            }
        } else if (name == "transfer-encoding") {
            return false;
        }
    }
    if (content_length > kMaxBodyBytes) {
        if (out_status) *out_status = 413;
        return false;
    }
    const size_t body_start = header_end + 4u;
    while (bytes.size() - body_start < content_length) {
        char chunk[8192];
        const ssize_t got = recv(fd, chunk, sizeof(chunk), 0);
        if (got > 0) {
            bytes.append(chunk, static_cast<size_t>(got));
            if (bytes.size() - body_start > kMaxBodyBytes) {
                if (out_status) *out_status = 413;
                return false;
            }
            continue;
        }
        if (got < 0 && errno == EINTR) continue;
        return false;
    }
    out->body.assign(bytes.data() + body_start, content_length);
    return true;
}

bool parse_max_tokens(
        const ajson &payload, uint32_t max_context, uint32_t *out, std::string *error,
        bool allow_stream = false, bool *explicit_value = nullptr) {
    if (!out || !error) return false;
    *out = kDefaultMaxNew;
    if (explicit_value) *explicit_value = false;
    const ajson *value = payload.get("max_tokens");
    if (!value) value = payload.get("max_completion_tokens");
    if (value) {
        if (!value->is_int() || value->i <= 0 ||
            static_cast<unsigned long long>(value->i) > std::numeric_limits<uint32_t>::max()) {
            *error = "max_tokens must be a positive integer";
            return false;
        }
        *out = static_cast<uint32_t>(value->i);
        if (explicit_value) *explicit_value = true;
    }
    if (*out > max_context) {
        *error = "max_tokens exceeds configured context";
        return false;
    }
    const ajson *stream = payload.get("stream");
    if (!allow_stream && stream && stream->is_bool() && stream->b) {
        *error = "streaming is not implemented in the native Qwen3.8 endpoint yet";
        return false;
    }
    return true;
}

bool validate_model(const ajson &payload, std::string *error) {
    if (!error) return false;
    const ajson *model = payload.get("model");
    if (!model) return true;
    if (!model->is_string() || (model->s != kModelId && model->s != "qwen3.8-27b" &&
                                model->s != "axiom/qwen3.8-27b-nvfp4")) {
        *error = "this endpoint serves only qwen3.8-27b-nvfp4";
        return false;
    }
    return true;
}

bool parse_request_context_window(
        const server_state *state, const ajson &payload, uint32_t *out,
        std::string *error) {
    if (!state || !out || !error) return false;
    *out = state->default_context;
    const ajson *value = payload.get("context_window");
    if (!value) value = payload.get("context_length");
    if (!value) value = payload.get("max_context");
    if (!value) value = payload.get("axiom_context_window");
    if (!value) return true;
    if (!value->is_int() || value->i < static_cast<int64_t>(AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH) ||
        static_cast<unsigned long long>(value->i) > state->max_context) {
        *error = "context_window must be an integer in [8, configured maximum]";
        return false;
    }
    *out = static_cast<uint32_t>(value->i);
    return true;
}

bool parse_speculative_mode(
        const ajson &payload, speculative_mode *out, std::string *error) {
    if (!out || !error) return false;
    *out = speculative_mode::automatic;
    const ajson *canonical = payload.get("axiom_speculative_mode");
    const ajson *alias = payload.get("speculative_mode");
    if (canonical && alias) {
        if (!canonical->is_string() || !alias->is_string() ||
            canonical->s != alias->s) {
            *error = "axiom_speculative_mode conflicts with speculative_mode";
            return false;
        }
    }
    const ajson *value = canonical ? canonical : alias;
    if (!value) return true;
    if (!value->is_string()) {
        *error = "axiom_speculative_mode must be one of auto, dspark, native_mtp, target";
        return false;
    }
    if (value->s == "auto") {
        *out = speculative_mode::automatic;
    } else if (value->s == "dspark") {
        *out = speculative_mode::dspark;
    } else if (value->s == "native_mtp") {
        *out = speculative_mode::native_mtp;
    } else if (value->s == "target") {
        *out = speculative_mode::target;
    } else {
        *error = "axiom_speculative_mode must be one of auto, dspark, native_mtp, target";
        return false;
    }
    return true;
}

bool parse_speculative_max_commit_tokens(
        const ajson &payload, const speculative_mode mode, uint32_t *out,
        std::string *error) {
    if (!out || !error) return false;
    *out = AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
    const ajson *value = payload.get("axiom_speculative_max_commit_tokens");
    if (!value) return true;
    if (mode != speculative_mode::dspark && mode != speculative_mode::native_mtp) {
        *error = "axiom_speculative_max_commit_tokens requires an explicit speculative graph mode";
        return false;
    }
    if (!value->is_int() || value->i < 1 ||
        value->i > AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH) {
        *error = "axiom_speculative_max_commit_tokens must be an integer in [1, 8]";
        return false;
    }
    *out = static_cast<uint32_t>(value->i);
    return true;
}

int append_text_ids(server_state *state, const std::string &text, std::vector<uint32_t> *out) {
    const uint32_t context_limit = effective_context_limit(state);
    if (!state || !state->tokenizer || !out || out->size() >= context_limit) return AXIOM_ERR_BUDGET;
    std::vector<uint32_t> encoded(context_limit - static_cast<uint32_t>(out->size()));
    uint32_t count = 0u;
    const int rc = axiom_tokenizer_encode_text(
            state->tokenizer, text.c_str(), encoded.data(), static_cast<uint32_t>(encoded.size()), &count);
    if (rc != AXIOM_OK) return rc;
    out->insert(out->end(), encoded.begin(), encoded.begin() + count);
    return AXIOM_OK;
}

std::string message_content_text(const ajson *content) {
    if (!content || content->is_null()) return {};
    if (content->is_string()) return content->s;
    if (content->is_array()) {
        std::string text;
        for (const ajson &item : content->arr) {
            if (item.is_string()) {
                text += item.s;
            } else if (item.is_object()) {
                const ajson *type = item.get("type");
                if (type && type->is_string() && type->s == "tool_result") {
                    text += "<tool_response>\n";
                    text += message_content_text(item.get("content"));
                    text += "\n</tool_response>";
                    continue;
                }
                const ajson *value = item.get("text");
                if (!value || !value->is_string()) value = item.get("content");
                if (value && value->is_string()) text += value->s;
                else if (value && value->is_array()) text += message_content_text(value);
            }
        }
        return text;
    }
    return ajson_dumps(*content);
}

std::string trim_text(std::string value) {
    size_t first = 0u;
    while (first < value.size() &&
           (value[first] == ' ' || value[first] == '\t' || value[first] == '\r' ||
            value[first] == '\n')) {
        ++first;
    }
    size_t last = value.size();
    while (last > first &&
           (value[last - 1u] == ' ' || value[last - 1u] == '\t' ||
            value[last - 1u] == '\r' || value[last - 1u] == '\n')) {
        --last;
    }
    return value.substr(first, last - first);
}

bool tools_are_enabled(const ajson *tool_choice) {
    if (!tool_choice) return true;
    return !(tool_choice->is_string() && tool_choice->s == "none");
}

bool supported_reasoning_effort(const std::string &effort) {
    return axiom::reasoning::find(effort) != nullptr;
}

std::string requested_reasoning_effort(const ajson &payload) {
    const ajson *effort = payload.get("reasoning_effort");
    if (!effort) {
        const ajson *reasoning = payload.get("reasoning");
        if (reasoning && reasoning->is_object()) effort = reasoning->get("effort");
    }
    return effort && effort->is_string()
            ? axiom::reasoning::canonical_name(effort->s) : std::string();
}

bool resolve_no_think(
        const server_state *state, const ajson &payload, std::string *error) {
    if (!error) return false;
    bool no_think = state ? state->no_think : true;
    bool explicit_switch = false;
    const ajson *think = payload.get("think");
    if (think) {
        if (!think->is_bool()) {
            *error = "think must be a boolean";
            return false;
        }
        no_think = !think->b;
        explicit_switch = true;
    }
    if (!explicit_switch) {
        const ajson *template_kwargs = payload.get("chat_template_kwargs");
        const ajson *enable_thinking = template_kwargs && template_kwargs->is_object()
                ? template_kwargs->get("enable_thinking") : nullptr;
        if (enable_thinking) {
            if (!enable_thinking->is_bool()) {
                *error = "chat_template_kwargs.enable_thinking must be a boolean";
                return false;
            }
            no_think = !enable_thinking->b;
        }
    }
    const std::string effort = requested_reasoning_effort(payload);
    if (effort.empty()) return no_think;
    if (!supported_reasoning_effort(effort)) {
        *error = "reasoning effort must be one of ultra-fast, minimal, low, medium, high, xhigh, max";
        return false;
    }
    /* Qwen's native switch is binary; the shared profile supplies the exact
     * hidden-token cap used by make_thinking_plan below. */
    const axiom::reasoning::profile *selected = axiom::reasoning::find(effort);
    return !selected->thinking;
}

std::string effective_reasoning_effort(const server_state *state, const ajson &payload) {
    const std::string requested = requested_reasoning_effort(payload);
    if (!requested.empty()) return requested;
    const ajson *think = payload.get("think");
    if (think && think->is_bool()) return think->b ? "medium" : "ultra-fast";
    const ajson *template_kwargs = payload.get("chat_template_kwargs");
    const ajson *enable_thinking = template_kwargs && template_kwargs->is_object()
            ? template_kwargs->get("enable_thinking") : nullptr;
    if (enable_thinking && enable_thinking->is_bool()) {
        return enable_thinking->b ? "medium" : "ultra-fast";
    }
    return state && state->no_think ? "ultra-fast" : "medium";
}

// Public effort IDs/budgets remain independent from the three instruction
// families actually supported by this model's chat_template.jinja.
const char *native_reasoning_conditioning(const std::string &effort) {
    if (effort == "ultra-fast") return "off";
    if (effort == "minimal" || effort == "low") return "low";
    if (effort == "medium") return "medium";
    return "xhigh";
}

std::string native_reasoning_instruction(const std::string &effort) {
    const std::string family = native_reasoning_conditioning(effort);
    if (family == "low") return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
    if (family == "xhigh") return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    return {};
}

bool resolve_session_id(
        const ajson &payload,
        const std::string &header_session_id,
        std::string *out,
        bool *generated,
        std::string *error) {
    if (!out || !generated || !error) return false;
    *generated = false;
    const ajson *value = payload.get("session_id");
    if (!value) value = payload.get("axiom_session_id");
    if (value && (!value->is_string() ||
        !axiom::qwen38::qwen38_persistent_session_store::valid_session_id(value->s))) {
        *error = "session_id must be a non-empty safe string of at most 127 characters";
        return false;
    }
    if (value && !header_session_id.empty() && value->s != header_session_id) {
        *error = "session_id conflicts with X-Axiom-Session-ID";
        return false;
    }
    if (value) {
        *out = value->s;
        return true;
    }
    if (!header_session_id.empty()) {
        *out = header_session_id;
        return true;
    }
    *out = axiom::qwen38::qwen38_persistent_session_store::generate_session_id();
    if (out->empty()) {
        *error = "could not obtain cryptographic entropy for a new session_id";
        return false;
    }
    *generated = true;
    return true;
}

uint32_t thinking_budget_for_effort(const std::string &effort) {
    const axiom::reasoning::profile *selected = axiom::reasoning::find(effort);
    return selected ? selected->thinking_budget_tokens : 0u;
}

const ajson *request_value(const ajson &payload, const char *key) {
    const ajson *value = payload.get(key);
    if (value) return value;
    const ajson *reasoning = payload.get("reasoning");
    return reasoning && reasoning->is_object() ? reasoning->get(key) : nullptr;
}

bool parse_sampling_params(
        const ajson &payload, bool no_think, uint64_t request_sequence,
        sampling_params *out, std::string *error) {
    if (!out || !error) return false;
    *out = sampling_params{};
    out->temperature = no_think ? kNonThinkingTemperature : kThinkingTemperature;
    out->top_p = no_think ? kNonThinkingTopP : kThinkingTopP;
    out->top_k = no_think ? kNonThinkingTopK : kDefaultQwenTopK;
    out->rng = 0x9e3779b97f4a7c15ull ^
            (request_sequence * 0xd1b54a32d192ed03ull);
    if (out->rng == 0u) out->rng = 0x6a09e667f3bcc909ull;

    const ajson *temperature = payload.get("temperature");
    if (temperature) {
        if (!temperature->is_number() || !std::isfinite(temperature->number()) ||
            temperature->number() < 0.0 || temperature->number() > 2.0) {
            *error = "temperature must be a finite number in [0, 2]";
            return false;
        }
        out->temperature = static_cast<float>(temperature->number());
    }
    const ajson *top_p = payload.get("top_p");
    if (top_p) {
        if (!top_p->is_number() || !std::isfinite(top_p->number()) ||
            top_p->number() <= 0.0 || top_p->number() > 1.0) {
            *error = "top_p must be a finite number in (0, 1]";
            return false;
        }
        out->top_p = static_cast<float>(top_p->number());
    }
    const ajson *top_k = payload.get("top_k");
    if (top_k) {
        if (!top_k->is_int() || top_k->i <= 0 || top_k->i > 256) {
            *error = "top_k must be an integer in [1, 256]";
            return false;
        }
        out->top_k = static_cast<uint32_t>(top_k->i);
    }
    const ajson *min_p = payload.get("min_p");
    if (min_p && (!min_p->is_number() || min_p->number() != 0.0)) {
        *error = "min_p is fixed at 0 by the native Qwen sampler";
        return false;
    }
    const ajson *seed = payload.get("seed");
    if (!seed) seed = payload.get("sampling_seed");
    if (seed) {
        if (!seed->is_int() || seed->i < 0) {
            *error = "seed must be a non-negative integer";
            return false;
        }
        out->rng = static_cast<uint64_t>(seed->i);
        if (out->rng == 0u) out->rng = 0x6a09e667f3bcc909ull;
    }
    out->enabled = out->temperature > 0.0f;
    return true;
}

bool make_thinking_plan(
        const ajson &payload, bool no_think, uint32_t parsed_visible_budget,
        bool explicit_visible_budget, uint32_t max_context,
        thinking_plan *out, std::string *error) {
    if (!out || !error) return false;
    *out = thinking_plan{};
    out->visible_budget = explicit_visible_budget
            ? parsed_visible_budget : kDefaultMaxNew;
    if (no_think) return true;
    const std::string effort = effective_reasoning_effort(nullptr, payload);
    const std::string selected_effort = effort.empty() ? "medium" : effort;
    uint32_t thinking_budget = thinking_budget_for_effort(selected_effort);
    const ajson *requested_budget = request_value(payload, "thinking_budget");
    if (requested_budget) {
        if (!requested_budget->is_int() || requested_budget->i < 0 ||
            static_cast<unsigned long long>(requested_budget->i) > max_context) {
            *error = "thinking_budget must be an integer in [0, configured context]";
            return false;
        }
        thinking_budget = static_cast<uint32_t>(requested_budget->i);
    }
    if (thinking_budget == 0u) return true;
    out->enabled = true;
    out->thinking_budget = thinking_budget;
    out->visible_budget = explicit_visible_budget
            ? parsed_visible_budget : kDefaultVisibleBudget;
    return true;
}

bool validate_native_tool_catalog(const ajson &tools, std::string *error);

bool responses_builtin_tool_type(const std::string &type) {
    return type == "web_search" || type == "web_search_preview" ||
            type == "shell" || type == "local_shell" ||
            type == "computer_use_preview";
}

bool append_normalized_responses_tool(
        const ajson &tool, bool deferred, ajson *normalized,
        std::string *error) {
    if (!normalized || !error || !tool.is_object()) {
        if (error) *error = "each Responses tool must be an object";
        return false;
    }
    const ajson *type = tool.get("type");
    if (!type || !type->is_string()) {
        *error = "Responses tools require a string type";
        return false;
    }
    if (responses_builtin_tool_type(type->s)) {
        if (deferred) {
            *error = "tool_search_output may load only function, custom or namespace tools";
            return false;
        }
        return true;
    }
    if (type->s == "namespace") {
        const ajson *nested_tools = tool.get("tools");
        if (!nested_tools || !nested_tools->is_array()) {
            *error = "Responses namespace tools must contain an array";
            return false;
        }
        for (const ajson &nested_tool : nested_tools->arr) {
            ajson flattened = nested_tool;
            if (flattened.is_object() && !flattened.get("type")) {
                flattened.set("type", ajson::jstr("function"));
            }
            if (!append_normalized_responses_tool(
                    flattened, deferred, normalized, error)) {
                return false;
            }
        }
        return true;
    }
    if (type->s != "function" && type->s != "custom") {
        *error = deferred
                ? "tool_search_output may load only function, custom or namespace tools"
                : "Axiom Responses supports function, custom and namespace tools";
        return false;
    }
    if (deferred) {
        const ajson *defer_loading = tool.get("defer_loading");
        if (!defer_loading || !defer_loading->is_bool() || !defer_loading->b) {
            *error = "tool_search_output tools require defer_loading=true";
            return false;
        }
    }

    const bool custom = type->s == "custom";
    const ajson *name = tool.get("name");
    const ajson *description = tool.get("description");
    const ajson *parameters = tool.get("parameters");
    const ajson *nested = tool.get("function");
    if (nested && nested->is_object()) {
        if (!name) name = nested->get("name");
        if (!description) description = nested->get("description");
        if (!parameters) parameters = nested->get("parameters");
    }
    if (!name || !name->is_string() || name->s.empty() ||
        (parameters && !parameters->is_object())) {
        *error = "Responses function tools require a name and object parameters";
        return false;
    }

    ajson function = ajson::jobj();
    function.set("name", *name);
    if (description && description->is_string()) {
        function.set("description", *description);
    }
    if (custom && !parameters) {
        ajson custom_parameters = ajson::jobj();
        custom_parameters.set("type", ajson::jstr("object"));
        ajson properties = ajson::jobj();
        ajson input_schema = ajson::jobj();
        input_schema.set("type", ajson::jstr("string"));
        input_schema.set("description", ajson::jstr("Unconstrained custom-tool input"));
        properties.set("input", std::move(input_schema));
        custom_parameters.set("properties", std::move(properties));
        ajson required = ajson::jarr();
        required.push(ajson::jstr("input"));
        custom_parameters.set("required", std::move(required));
        function.set("parameters", std::move(custom_parameters));
    } else {
        function.set("parameters", parameters ? *parameters : ajson::jobj());
    }
    function.set("axiom_responses_type", ajson::jstr(custom ? "custom" : "function"));
    if (deferred) function.set("axiom_deferred", ajson::jbool(true));
    ajson wrapper = ajson::jobj();
    wrapper.set("type", ajson::jstr("function"));
    wrapper.set("function", std::move(function));
    normalized->push(std::move(wrapper));
    return true;
}

ajson native_tool_prompt_definition(const ajson &function) {
    ajson prompt = ajson::jobj();
    if (!function.is_object()) return prompt;
    for (size_t index = 0u; index < function.keys.size(); ++index) {
        if (function.keys[index].rfind("axiom_", 0u) == 0u) continue;
        prompt.set(function.keys[index], function.vals[index]);
    }
    return prompt;
}

std::string render_qwen_deferred_tools_prompt(const ajson &tools) {
    if (!tools.is_array() || tools.arr.empty()) return {};
    std::string out =
            "# Tools loaded\n\nClient tool search loaded these additional functions. "
            "They are available now and use the native function-call format already specified.\n\n<tools>";
    for (const ajson &function : tools.arr) {
        out += "\n";
        out += ajson_dumps(native_tool_prompt_definition(function));
    }
    out += "\n</tools>";
    return out;
}

bool append_responses_input_item(
        const ajson &item, ajson *messages, ajson *deferred_tools,
        std::string *pending_tool_search_call_id, std::string *error) {
    if (!messages || !deferred_tools || !pending_tool_search_call_id ||
        !error || !item.is_object()) {
        if (error) *error = "each Responses input item must be an object";
        return false;
    }
    const ajson *type = item.get("type");
    const std::string item_type = type && type->is_string() ? type->s : "message";
    if (item_type == "tool_search_call") {
        const ajson *call_id = item.get("call_id");
        const ajson *execution = item.get("execution");
        const ajson *status = item.get("status");
        const ajson *arguments = item.get("arguments");
        if (!pending_tool_search_call_id->empty()) {
            *error = "tool_search_call must be followed immediately by its tool_search_output";
            return false;
        }
        if (!call_id || !call_id->is_string() || call_id->s.empty() ||
            !execution || !execution->is_string() || execution->s != "client" ||
            !status || !status->is_string() || status->s != "completed" ||
            !arguments || !arguments->is_object()) {
            *error = "tool_search_call requires call_id, execution=client, status=completed and object arguments";
            return false;
        }
        *pending_tool_search_call_id = call_id->s;
        return true;
    }
    if (item_type == "tool_search_output") {
        const ajson *call_id = item.get("call_id");
        const ajson *execution = item.get("execution");
        const ajson *status = item.get("status");
        const ajson *tools = item.get("tools");
        if (pending_tool_search_call_id->empty() || !call_id ||
            !call_id->is_string() || call_id->s != *pending_tool_search_call_id ||
            !execution || !execution->is_string() || execution->s != "client" ||
            !status || !status->is_string() || status->s != "completed" ||
            !tools || !tools->is_array() || tools->arr.empty()) {
            *error = "tool_search_output requires the matching call_id, execution=client, status=completed and non-empty tools";
            return false;
        }
        ajson loaded_wrappers = ajson::jarr();
        for (const ajson &tool : tools->arr) {
            if (!append_normalized_responses_tool(
                    tool, true, &loaded_wrappers, error)) {
                return false;
            }
        }
        ajson loaded_native = ajson::jarr();
        for (const ajson &wrapper : loaded_wrappers.arr) {
            const ajson *function = wrapper.get("function");
            if (!function || !function->is_object()) {
                *error = "tool_search_output produced an invalid function definition";
                return false;
            }
            loaded_native.push(*function);
        }
        if (!validate_native_tool_catalog(loaded_native, error)) return false;
        for (const ajson &wrapper : loaded_wrappers.arr) deferred_tools->push(wrapper);
        ajson message = ajson::jobj();
        message.set("role", ajson::jstr("system"));
        message.set("content", ajson::jstr(
                render_qwen_deferred_tools_prompt(loaded_native)));
        messages->push(std::move(message));
        pending_tool_search_call_id->clear();
        return true;
    }
    if (!pending_tool_search_call_id->empty()) {
        *error = "tool_search_call must be followed immediately by its tool_search_output";
        return false;
    }
    if (item_type == "reasoning" || item_type == "item_reference") return true;
    if (item_type == "function_call_output" || item_type == "custom_tool_call_output") {
        const ajson *output = item.get("output");
        ajson message = ajson::jobj();
        message.set("role", ajson::jstr("tool"));
        message.set("content", ajson::jstr(message_content_text(output)));
        messages->push(std::move(message));
        return true;
    }
    if (item_type == "function_call") {
        const ajson *name = item.get("name");
        const ajson *arguments = item.get("arguments");
        if (!name || !name->is_string() || name->s.empty() ||
            !arguments || !arguments->is_string()) {
            *error = "Responses function_call items require name and string arguments";
            return false;
        }
        ajson function = ajson::jobj();
        function.set("name", *name);
        function.set("arguments", *arguments);
        ajson call = ajson::jobj();
        const ajson *call_id = item.get("call_id");
        const ajson *id = item.get("id");
        call.set("id", call_id ? *call_id : (id ? *id : ajson::jstr("call_axiom")));
        call.set("type", ajson::jstr("function"));
        call.set("function", std::move(function));
        ajson calls = ajson::jarr();
        calls.push(std::move(call));
        ajson message = ajson::jobj();
        message.set("role", ajson::jstr("assistant"));
        message.set("content", ajson::jnull());
        message.set("tool_calls", std::move(calls));
        messages->push(std::move(message));
        return true;
    }
    if (item_type == "custom_tool_call") {
        const ajson *name = item.get("name");
        const ajson *input = item.get("input");
        if (!name || !name->is_string() || name->s.empty() ||
            !input || !input->is_string()) {
            *error = "Responses custom_tool_call items require name and string input";
            return false;
        }
        ajson function = ajson::jobj();
        function.set("name", *name);
        ajson arguments = ajson::jobj();
        arguments.set("input", *input);
        function.set("arguments", ajson::jstr(ajson_dumps(arguments)));
        ajson call = ajson::jobj();
        const ajson *call_id = item.get("call_id");
        const ajson *id = item.get("id");
        call.set("id", call_id ? *call_id : (id ? *id : ajson::jstr("call_axiom")));
        call.set("type", ajson::jstr("function"));
        call.set("function", std::move(function));
        ajson calls = ajson::jarr();
        calls.push(std::move(call));
        ajson message = ajson::jobj();
        message.set("role", ajson::jstr("assistant"));
        message.set("content", ajson::jnull());
        message.set("tool_calls", std::move(calls));
        messages->push(std::move(message));
        return true;
    }
    if (item_type != "message") {
        *error = "Axiom Responses supports message, function_call, custom_tool_call, function_call_output, tool_search_call and tool_search_output input items";
        return false;
    }
    const ajson *role = item.get("role");
    if (!role || !role->is_string()) {
        *error = "Responses message items require a role";
        return false;
    }
    std::string normalized_role = role->s == "developer" ? "system" : role->s;
    if (normalized_role != "system" && normalized_role != "user" &&
        normalized_role != "assistant" && normalized_role != "tool") {
        *error = "Responses message role must be system, developer, user, assistant or tool";
        return false;
    }
    ajson message = ajson::jobj();
    message.set("role", ajson::jstr(normalized_role));
    const ajson *content = item.get("content");
    /* Preserve structured input_text/input_image/input_video parts.  The
     * text-only normalizer used to flatten them here, making Responses vision
     * impossible before the multimodal prompt builder even ran. */
    ajson normalized_content = content ? *content : ajson::jnull();
    // Responses clients replay our own assistant output as output_text parts.
    // The multimodal chat builder accepts text/input_text, so canonicalize this
    // assistant-only wire alias here while preserving every media part and
    // annotation. Do not flatten the whole message (that would discard images).
    if (normalized_role == "assistant" && normalized_content.is_array()) {
        for (auto &part : normalized_content.arr) {
            const ajson *type = part.get("type");
            if (type && type->is_string() && type->s == "output_text") {
                const ajson *text = part.get("text");
                if (!text || !text->is_string()) {
                    *error = "Responses output_text parts require string text";
                    return false;
                }
                part.set("type", ajson::jstr("text"));
            }
        }
    }
    message.set("content", std::move(normalized_content));
    messages->push(std::move(message));
    return true;
}

bool normalize_responses_request(
        const ajson &request, ajson *chat_payload, std::string *error) {
    if (!chat_payload || !error) return false;
    *chat_payload = ajson::jobj();
    const ajson *model = request.get("model");
    if (model) chat_payload->set("model", *model);

    ajson messages = ajson::jarr();
    ajson deferred_tools = ajson::jarr();
    std::string pending_tool_search_call_id;
    const ajson *instructions = request.get("instructions");
    if (instructions) {
        ajson system = ajson::jobj();
        system.set("role", ajson::jstr("system"));
        system.set("content", *instructions);
        messages.push(std::move(system));
    }
    const ajson *input = request.get("input");
    if (!input) {
        *error = "Responses requests require input";
        return false;
    }
    if (input->is_string()) {
        ajson user = ajson::jobj();
        user.set("role", ajson::jstr("user"));
        user.set("content", *input);
        messages.push(std::move(user));
    } else if (input->is_array()) {
        for (const ajson &item : input->arr) {
            if (!append_responses_input_item(
                    item, &messages, &deferred_tools,
                    &pending_tool_search_call_id, error)) {
                return false;
            }
        }
    } else {
        *error = "Responses input must be a string or an array of input items";
        return false;
    }
    if (!pending_tool_search_call_id.empty()) {
        *error = "tool_search_call must be followed immediately by its tool_search_output";
        return false;
    }
    if (messages.arr.empty()) {
        *error = "Responses input did not contain a usable message";
        return false;
    }
    chat_payload->set("messages", std::move(messages));

    ajson normalized = ajson::jarr();
    const ajson *tools = request.get("tools");
    if (tools) {
        if (!tools->is_array()) {
            *error = "Responses tools must be an array";
            return false;
        }
        for (const ajson &tool : tools->arr) {
            if (!append_normalized_responses_tool(
                    tool, false, &normalized, error)) {
                return false;
            }
        }
    }
    for (const ajson &tool : deferred_tools.arr) normalized.push(tool);
    if (!normalized.arr.empty()) chat_payload->set("tools", std::move(normalized));

    const ajson *tool_choice = request.get("tool_choice");
    if (tool_choice) {
        if (tool_choice->is_object()) {
            const ajson *type = tool_choice->get("type");
            const ajson *name = tool_choice->get("name");
            if (type && type->is_string() &&
                (type->s == "function" || type->s == "custom") &&
                name && name->is_string()) {
                ajson function = ajson::jobj();
                function.set("name", *name);
                ajson choice = ajson::jobj();
                choice.set("type", ajson::jstr("function"));
                choice.set("function", std::move(function));
                chat_payload->set("tool_choice", std::move(choice));
            } else {
                *error = "Responses tool_choice object must name a function";
                return false;
            }
        } else {
            chat_payload->set("tool_choice", *tool_choice);
        }
    }
    const ajson *reasoning = request.get("reasoning");
    if (reasoning && reasoning->is_object()) {
        const ajson *effort = reasoning->get("effort");
        if (effort) chat_payload->set("reasoning_effort", *effort);
        const ajson *thinking_budget = reasoning->get("thinking_budget");
        if (thinking_budget) chat_payload->set("thinking_budget", *thinking_budget);
    }
    const char *copy_keys[] = {
        "stream", "temperature", "top_p", "top_k", "min_p", "seed",
        "sampling_seed", "think", "steer", "chat_template_kwargs", "thinking_budget",
        "context_window", "context_length", "max_context", "axiom_context_window",
        "axiom_speculative_mode", "speculative_mode",
        "axiom_speculative_max_commit_tokens"
    };
    for (const char *key : copy_keys) {
        const ajson *value = request.get(key);
        if (value) chat_payload->set(key, *value);
    }
    const ajson *max_output_tokens = request.get("max_output_tokens");
    if (max_output_tokens) chat_payload->set("max_tokens", *max_output_tokens);
    const ajson *max_tokens = request.get("max_tokens");
    if (max_tokens && !max_output_tokens) chat_payload->set("max_tokens", *max_tokens);
    return true;
}

/* Normalize OpenAI and Anthropic tool envelopes to the function objects used
 * by Qwen's native chat template. The template intentionally receives one
 * JSON object per function inside <tools>, not an OpenAI wrapper. */
bool normalize_tools(const ajson *tools, bool anthropic, ajson *out, std::string *error) {
    if (!out || !error) return false;
    *out = ajson::jarr();
    if (!tools) return true;
    if (!tools->is_array()) {
        *error = "tools must be an array";
        return false;
    }
    for (const ajson &tool : tools->arr) {
        if (!tool.is_object()) {
            *error = "each tool must be an object";
            return false;
        }
        ajson function = ajson::jobj();
        if (anthropic) {
            const ajson *name = tool.get("name");
            const ajson *description = tool.get("description");
            const ajson *schema = tool.get("input_schema");
            if (!name || !name->is_string() || name->s.empty() ||
                !schema || !schema->is_object()) {
                *error = "Anthropic tools require name and object input_schema";
                return false;
            }
            function.set("name", *name);
            if (description && description->is_string()) function.set("description", *description);
            function.set("parameters", *schema);
        } else {
            const ajson *type = tool.get("type");
            const ajson *candidate = tool.get("function");
            if (!type || !type->is_string() || type->s != "function" ||
                !candidate || !candidate->is_object()) {
                *error = "OpenAI tools must have type=function and a function object";
                return false;
            }
            const ajson *name = candidate->get("name");
            const ajson *parameters = candidate->get("parameters");
            if (!name || !name->is_string() || name->s.empty() ||
                (parameters && !parameters->is_object())) {
                *error = "function tools require a name and object parameters";
                return false;
            }
            function = *candidate;
            if (!parameters) function.set("parameters", ajson::jobj());
        }
        out->push(std::move(function));
    }
    return validate_native_tool_catalog(*out, error);
}

const ajson *tool_function_by_name(const ajson &tools, const std::string &name) {
    if (!tools.is_array()) return nullptr;
    for (const ajson &function : tools.arr) {
        const ajson *candidate = function.get("name");
        if (candidate && candidate->is_string() && candidate->s == name) return &function;
    }
    return nullptr;
}

std::string render_qwen_tools_prompt(const ajson &tools, const ajson *tool_choice) {
    if (!tools.is_array() || tools.arr.empty() || !tools_are_enabled(tool_choice)) return {};
    ajson immediate = ajson::jarr();
    for (const ajson &function : tools.arr) {
        const ajson *deferred = function.get("axiom_deferred");
        if (deferred && deferred->is_bool() && deferred->b) continue;
        immediate.push(native_tool_prompt_definition(function));
    }
    if (immediate.arr.empty()) return {};
    std::string out;
    out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const ajson &function : immediate.arr) {
        out += "\n";
        out += ajson_dumps(function);
    }
    out += "\n</tools>\n\n";
    out += "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n";
    out += "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n";
    out += "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n";
    out += "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n";
    out += "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n";
    out += "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n";
    out += "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
    if (tool_choice && tool_choice->is_object()) {
        const ajson *type = tool_choice->get("type");
        const ajson *function = tool_choice->get("function");
        const ajson *name = function && function->is_object() ? function->get("name") : nullptr;
        if (type && type->is_string() && type->s == "function" && name && name->is_string()) {
            out += "\n\nYou must call the function named `" + name->s + "`.";
        }
    }
    return out;
}

std::string native_tool_call_text(const ajson &call) {
    const ajson *function = call.get("function");
    if (!function || !function->is_object()) return {};
    const ajson *name = function->get("name");
    if (!name || !name->is_string() || name->s.empty()) return {};
    ajson args = ajson::jobj();
    const ajson *arguments = function->get("arguments");
    if (arguments && arguments->is_string()) {
        std::string parse_error;
        ajson parsed;
        if (ajson_parse(arguments->s, parsed, parse_error) && parsed.is_object()) args = std::move(parsed);
    } else if (arguments && arguments->is_object()) {
        args = *arguments;
    }
    std::string out = "<tool_call>\n<function=" + name->s + ">\n";
    for (size_t n = 0u; n < args.keys.size(); ++n) {
        out += "<parameter=" + args.keys[n] + ">\n";
        const ajson &value = args.vals[n];
        out += value.is_string() ? value.s : ajson_dumps(value);
        out += "\n</parameter>\n";
    }
    out += "</function>\n</tool_call>";
    return out;
}

std::string native_tool_calls_from_content(const ajson *content) {
    if (!content || !content->is_array()) return {};
    std::string out;
    for (const ajson &item : content->arr) {
        if (!item.is_object()) continue;
        const ajson *type = item.get("type");
        if (!type || !type->is_string() || type->s != "tool_use") continue;
        ajson function_call = ajson::jobj();
        const ajson *name = item.get("name");
        const ajson *input = item.get("input");
        if (!name || !name->is_string()) continue;
        ajson fn = ajson::jobj();
        fn.set("name", *name);
        fn.set("arguments", input ? ajson::jstr(ajson_dumps(*input)) : ajson::jstr("{}"));
        function_call.set("function", std::move(fn));
        const std::string call_text = native_tool_call_text(function_call);
        if (!call_text.empty()) {
            if (!out.empty()) out += "\n";
            out += call_text;
        }
    }
    return out;
}

bool tool_parameter_is_string(const ajson *function, const std::string &name) {
    if (!function || !function->is_object()) return false;
    const ajson *parameters = function->get("parameters");
    const ajson *properties = parameters && parameters->is_object()
            ? parameters->get("properties") : nullptr;
    const ajson *property = properties && properties->is_object()
            ? properties->get(name.c_str()) : nullptr;
    const ajson *type = property && property->is_object() ? property->get("type") : nullptr;
    return type && type->is_string() && type->s == "string";
}

ajson parse_native_parameter(const std::string &body, bool string_parameter) {
    const std::string value = trim_text(body);
    if (string_parameter) return ajson::jstr(value);
    ajson parsed;
    std::string error;
    if (ajson_parse(value, parsed, error)) return parsed;
    return ajson::jstr(value);
}

enum class native_tool_argument_error_code {
    required,
    type,
    enum_value,
    const_value,
    additional_property,
    schema,
};

struct native_tool_argument_error {
    native_tool_argument_error_code code = native_tool_argument_error_code::schema;
    std::string path;
    std::string detail;
};

const char *native_tool_argument_error_code_name(
        native_tool_argument_error_code code) {
    switch (code) {
    case native_tool_argument_error_code::required: return "required";
    case native_tool_argument_error_code::type: return "type";
    case native_tool_argument_error_code::enum_value: return "enum";
    case native_tool_argument_error_code::const_value: return "const";
    case native_tool_argument_error_code::additional_property:
        return "additional_property";
    case native_tool_argument_error_code::schema: return "schema";
    }
    return "schema";
}

const char *native_tool_argument_kind_name(const ajson &value) {
    switch (value.kind) {
    case ajson::NUL: return "null";
    case ajson::BOOL: return "boolean";
    case ajson::INT: return "integer";
    case ajson::NUM: return "number";
    case ajson::STR: return "string";
    case ajson::ARR: return "array";
    case ajson::OBJ: return "object";
    }
    return "unknown";
}

bool native_tool_json_equal(const ajson &left, const ajson &right) {
    if (left.is_number() && right.is_number()) {
        if (left.is_int() && right.is_int()) return left.i == right.i;
        if (left.is_int() && right.kind == ajson::NUM) {
            return static_cast<long double>(left.i) ==
                    static_cast<long double>(right.d);
        }
        if (left.kind == ajson::NUM && right.is_int()) {
            return static_cast<long double>(left.d) ==
                    static_cast<long double>(right.i);
        }
        return left.d == right.d;
    }
    if (left.kind != right.kind) return false;
    switch (left.kind) {
    case ajson::NUL:
        return true;
    case ajson::BOOL:
        return left.b == right.b;
    case ajson::INT:
        return left.i == right.i;
    case ajson::NUM:
        return left.d == right.d;
    case ajson::STR:
        return left.s == right.s;
    case ajson::ARR:
        if (left.arr.size() != right.arr.size()) return false;
        for (size_t index = 0u; index < left.arr.size(); ++index) {
            if (!native_tool_json_equal(left.arr[index], right.arr[index])) return false;
        }
        return true;
    case ajson::OBJ:
        if (left.keys.size() != right.keys.size()) return false;
        for (size_t index = 0u; index < left.keys.size(); ++index) {
            const ajson *candidate = right.get(left.keys[index].c_str());
            if (!candidate ||
                !native_tool_json_equal(left.vals[index], *candidate)) {
                return false;
            }
        }
        return true;
    }
    return false;
}

std::string native_tool_object_path(
        const std::string &parent, const std::string &property) {
    return parent + "[" + ajson_dumps(ajson::jstr(property)) + "]";
}

std::string native_tool_array_path(const std::string &parent, size_t index) {
    return parent + "[" + std::to_string(index) + "]";
}

bool native_tool_schema_type_matches(
        const ajson &value, const std::string &type) {
    if (type == "object") return value.is_object();
    if (type == "array") return value.is_array();
    if (type == "string") return value.is_string();
    if (type == "integer") {
        return value.is_int() ||
                (value.kind == ajson::NUM && std::trunc(value.d) == value.d);
    }
    if (type == "number") return value.is_number();
    if (type == "boolean") return value.is_bool();
    if (type == "null") return value.is_null();
    return false;
}

bool native_tool_schema_type_supported(const std::string &type) {
    return type == "object" || type == "array" || type == "string" ||
            type == "integer" || type == "number" || type == "boolean" ||
            type == "null";
}

bool fail_native_tool_argument_validation(
        native_tool_argument_error *error,
        native_tool_argument_error_code code,
        const std::string &path,
        const std::string &detail) {
    if (error) {
        error->code = code;
        error->path = path;
        error->detail = detail;
    }
    return false;
}

bool validate_native_tool_argument(
        const ajson &value, const ajson &schema, const std::string &path,
        native_tool_argument_error *error, size_t depth = 0u) {
    if (!schema.is_object()) {
        return fail_native_tool_argument_validation(
                error, native_tool_argument_error_code::schema, path,
                "schema node must be an object");
    }
    if (depth > static_cast<size_t>(ALICED_JSON_MAX_DEPTH)) {
        return fail_native_tool_argument_validation(
                error, native_tool_argument_error_code::schema, path,
                "schema nesting exceeds the supported depth");
    }

    const ajson *type = schema.get("type");
    if (type) {
        if (!type->is_string() || !native_tool_schema_type_supported(type->s)) {
            return fail_native_tool_argument_validation(
                    error, native_tool_argument_error_code::schema, path,
                    "type must be one of object, array, string, integer, number, boolean or null");
        }
        if (!native_tool_schema_type_matches(value, type->s)) {
            return fail_native_tool_argument_validation(
                    error, native_tool_argument_error_code::type, path,
                    "expected " + type->s + ", got " +
                    native_tool_argument_kind_name(value));
        }
    }

    const ajson *enum_values = schema.get("enum");
    if (enum_values) {
        if (!enum_values->is_array()) {
            return fail_native_tool_argument_validation(
                    error, native_tool_argument_error_code::schema, path,
                    "enum must be an array");
        }
        bool matched = false;
        for (const ajson &candidate : enum_values->arr) {
            if (native_tool_json_equal(value, candidate)) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            return fail_native_tool_argument_validation(
                    error, native_tool_argument_error_code::enum_value, path,
                    "value is not a declared enum member");
        }
    }

    const ajson *constant = schema.get("const");
    if (constant && !native_tool_json_equal(value, *constant)) {
        return fail_native_tool_argument_validation(
                error, native_tool_argument_error_code::const_value, path,
                "value does not equal the declared const");
    }

    if (value.is_object()) {
        const ajson *properties = schema.get("properties");
        if (properties && !properties->is_object()) {
            return fail_native_tool_argument_validation(
                    error, native_tool_argument_error_code::schema, path,
                    "properties must be an object");
        }

        const ajson *required = schema.get("required");
        if (required) {
            if (!required->is_array()) {
                return fail_native_tool_argument_validation(
                        error, native_tool_argument_error_code::schema, path,
                        "required must be an array of strings");
            }
            std::vector<std::string> required_names;
            required_names.reserve(required->arr.size());
            for (const ajson &entry : required->arr) {
                if (!entry.is_string()) {
                    return fail_native_tool_argument_validation(
                            error, native_tool_argument_error_code::schema, path,
                            "required must be an array of strings");
                }
                required_names.push_back(entry.s);
            }
            std::sort(required_names.begin(), required_names.end());
            required_names.erase(
                    std::unique(required_names.begin(), required_names.end()),
                    required_names.end());
            for (const std::string &required_name : required_names) {
                if (!value.get(required_name.c_str())) {
                    return fail_native_tool_argument_validation(
                            error, native_tool_argument_error_code::required,
                            native_tool_object_path(path, required_name),
                            "missing required property");
                }
            }
        }

        const ajson *additional = schema.get("additionalProperties");
        if (additional && !additional->is_bool() && !additional->is_object()) {
            return fail_native_tool_argument_validation(
                    error, native_tool_argument_error_code::schema, path,
                    "additionalProperties must be a boolean or object schema");
        }
        std::vector<std::string> unknown_names;
        for (const std::string &key : value.keys) {
            if (!properties || !properties->get(key.c_str())) {
                unknown_names.push_back(key);
            }
        }
        std::sort(unknown_names.begin(), unknown_names.end());
        for (const std::string &unknown_name : unknown_names) {
            const std::string unknown_path =
                    native_tool_object_path(path, unknown_name);
            if (additional && additional->is_bool() && !additional->b) {
                return fail_native_tool_argument_validation(
                        error,
                        native_tool_argument_error_code::additional_property,
                        unknown_path, "property is not declared");
            }
            if (additional && additional->is_object()) {
                const ajson *unknown_value = value.get(unknown_name.c_str());
                if (!unknown_value) {
                    return fail_native_tool_argument_validation(
                            error, native_tool_argument_error_code::schema,
                            unknown_path,
                            "argument object lookup failed during validation");
                }
                if (!validate_native_tool_argument(
                        *unknown_value, *additional, unknown_path, error,
                        depth + 1u)) {
                    return false;
                }
            }
        }

        if (properties) {
            std::vector<std::string> property_names = properties->keys;
            std::sort(property_names.begin(), property_names.end());
            for (const std::string &property_name : property_names) {
                const ajson *property_value = value.get(property_name.c_str());
                if (!property_value) continue;
                const ajson *property_schema = properties->get(property_name.c_str());
                if (!property_schema) {
                    return fail_native_tool_argument_validation(
                            error, native_tool_argument_error_code::schema,
                            native_tool_object_path(path, property_name),
                            "property schema lookup failed during validation");
                }
                if (!validate_native_tool_argument(
                        *property_value, *property_schema,
                        native_tool_object_path(path, property_name), error,
                        depth + 1u)) {
                    return false;
                }
            }
        }
    }

    if (value.is_array()) {
        const ajson *items = schema.get("items");
        if (items && !items->is_object()) {
            return fail_native_tool_argument_validation(
                    error, native_tool_argument_error_code::schema, path,
                    "items must be an object schema");
        }
        if (items) {
            for (size_t index = 0u; index < value.arr.size(); ++index) {
                if (!validate_native_tool_argument(
                        value.arr[index], *items,
                        native_tool_array_path(path, index), error,
                        depth + 1u)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool is_canonical_native_tool_name(const std::string &name) {
    if (name.empty() || name.size() > 64u || name.front() < 'a' ||
        name.front() > 'z' || name.back() == '_') {
        return false;
    }
    bool separator = false;
    for (const char ch : name) {
        const bool lower = ch >= 'a' && ch <= 'z';
        const bool digit = ch >= '0' && ch <= '9';
        if (lower || digit) {
            separator = false;
            continue;
        }
        if (ch != '_' || separator) return false;
        separator = true;
    }
    return true;
}

bool fail_native_tool_catalog_validation(
        std::string *error, const std::string &tool_name,
        const std::string &path, const std::string &detail) {
    if (error) {
        *error = "tool catalog validation failed: tool=" +
                ajson_dumps(ajson::jstr(tool_name)) + " path=" + path +
                " detail=" + detail;
    }
    return false;
}

bool validate_native_tool_schema_definition(
        const ajson &schema, const std::string &tool_name,
        const std::string &path, std::string *error, size_t depth = 0u) {
    if (!schema.is_object()) {
        return fail_native_tool_catalog_validation(
                error, tool_name, path, "schema node must be an object");
    }
    if (depth > static_cast<size_t>(ALICED_JSON_MAX_DEPTH)) {
        return fail_native_tool_catalog_validation(
                error, tool_name, path,
                "schema nesting exceeds the supported depth");
    }
    constexpr const char *unsupported[] = {
        "$ref", "allOf", "anyOf", "not", "oneOf",
    };
    for (const char *keyword : unsupported) {
        if (schema.get(keyword)) {
            return fail_native_tool_catalog_validation(
                    error, tool_name, path,
                    std::string("unsupported schema keyword ") + keyword);
        }
    }

    const ajson *type = schema.get("type");
    if (type &&
        (!type->is_string() || !native_tool_schema_type_supported(type->s))) {
        return fail_native_tool_catalog_validation(
                error, tool_name, path,
                "type must be one supported JSON type string");
    }
    const ajson *enum_values = schema.get("enum");
    if (enum_values) {
        if (!enum_values->is_array() || enum_values->arr.empty()) {
            return fail_native_tool_catalog_validation(
                    error, tool_name, path, "enum must be a non-empty array");
        }
        if (type) {
            for (const ajson &candidate : enum_values->arr) {
                if (!native_tool_schema_type_matches(candidate, type->s)) {
                    return fail_native_tool_catalog_validation(
                            error, tool_name, path,
                            "enum member does not match the declared type");
                }
            }
        }
    }
    const ajson *constant = schema.get("const");
    if (constant && type &&
        !native_tool_schema_type_matches(*constant, type->s)) {
        return fail_native_tool_catalog_validation(
                error, tool_name, path,
                "const does not match the declared type");
    }

    const ajson *properties = schema.get("properties");
    if (properties && !properties->is_object()) {
        return fail_native_tool_catalog_validation(
                error, tool_name, path, "properties must be an object");
    }
    if (properties && type && type->s != "object") {
        return fail_native_tool_catalog_validation(
                error, tool_name, path,
                "properties require type=object");
    }
    const ajson *required = schema.get("required");
    if (required) {
        if (!required->is_array()) {
            return fail_native_tool_catalog_validation(
                    error, tool_name, path,
                    "required must be an array of unique strings");
        }
        std::vector<std::string> names;
        names.reserve(required->arr.size());
        for (const ajson &entry : required->arr) {
            if (!entry.is_string() || entry.s.empty() ||
                !properties || !properties->get(entry.s.c_str())) {
                return fail_native_tool_catalog_validation(
                        error, tool_name, path,
                        "required must name declared properties");
            }
            if (std::find(names.begin(), names.end(), entry.s) != names.end()) {
                return fail_native_tool_catalog_validation(
                        error, tool_name, path,
                        "required must not contain duplicates");
            }
            names.push_back(entry.s);
        }
    }
    if (properties) {
        for (size_t index = 0u; index < properties->keys.size(); ++index) {
            if (!validate_native_tool_schema_definition(
                    properties->vals[index], tool_name,
                    native_tool_object_path(path, properties->keys[index]),
                    error, depth + 1u)) {
                return false;
            }
        }
    }

    const ajson *additional = schema.get("additionalProperties");
    if (additional && !additional->is_bool() && !additional->is_object()) {
        return fail_native_tool_catalog_validation(
                error, tool_name, path,
                "additionalProperties must be a boolean or object schema");
    }
    if (additional && additional->is_object() &&
        !validate_native_tool_schema_definition(
                *additional, tool_name, path + "[*]", error, depth + 1u)) {
        return false;
    }
    if (type && type->s == "object" &&
        (!properties || properties->keys.empty()) &&
        (!additional || (additional->is_bool() && additional->b))) {
        return fail_native_tool_catalog_validation(
                error, tool_name, path,
                "unbounded object schemas must be encoded before transport");
    }

    const ajson *items = schema.get("items");
    if (items && !items->is_object()) {
        return fail_native_tool_catalog_validation(
                error, tool_name, path, "items must be an object schema");
    }
    if (items && type && type->s != "array") {
        return fail_native_tool_catalog_validation(
                error, tool_name, path, "items require type=array");
    }
    if (type && type->s == "array" && !items) {
        return fail_native_tool_catalog_validation(
                error, tool_name, path, "array schemas require items");
    }
    if (items && !validate_native_tool_schema_definition(
            *items, tool_name, path + "[]", error, depth + 1u)) {
        return false;
    }
    return true;
}

bool validate_native_tool_catalog(const ajson &tools, std::string *error) {
    if (!error || !tools.is_array()) return false;
    std::vector<std::string> names;
    names.reserve(tools.arr.size());
    for (size_t index = 0u; index < tools.arr.size(); ++index) {
        const ajson &function = tools.arr[index];
        const std::string path = "$[" + std::to_string(index) + "]";
        if (!function.is_object()) {
            return fail_native_tool_catalog_validation(
                    error, "<invalid>", path,
                    "function must be an object");
        }
        const ajson *name = function.get("name");
        if (!name || !name->is_string()) {
            return fail_native_tool_catalog_validation(
                    error, "<invalid>", path + "[\"name\"]",
                    "name must be a string");
        }
        if (!is_canonical_native_tool_name(name->s)) {
            return fail_native_tool_catalog_validation(
                    error, name->s, path + "[\"name\"]",
                    "name must match ^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$");
        }
        if (std::find(names.begin(), names.end(), name->s) != names.end()) {
            return fail_native_tool_catalog_validation(
                    error, name->s, path + "[\"name\"]",
                    "duplicate tool name");
        }
        names.push_back(name->s);
        const ajson *parameters = function.get("parameters");
        const ajson *root_type = parameters && parameters->is_object()
                ? parameters->get("type") : nullptr;
        if (!parameters || !parameters->is_object() || !root_type ||
            !root_type->is_string() || root_type->s != "object") {
            return fail_native_tool_catalog_validation(
                    error, name->s, path + "[\"parameters\"]",
                    "parameters must be an object schema with type=object");
        }
        if (!validate_native_tool_schema_definition(
                *parameters, name->s, path + "[\"parameters\"]", error)) {
            return false;
        }
    }
    return true;
}

std::string format_native_tool_argument_error(
        const std::string &tool_name,
        const native_tool_argument_error &error) {
    return "tool argument validation failed: tool=" +
            ajson_dumps(ajson::jstr(tool_name)) + " code=" +
            native_tool_argument_error_code_name(error.code) + " path=" +
            error.path + " detail=" + error.detail;
}

// Codex boundary only. Native validation deliberately remains strict/unchanged.
// JSON Schema string lengths count Unicode codepoints, not UTF-8 bytes or
// grapheme clusters. Reject malformed UTF-8 (including parser WTF-8 surrogates)
// rather than letting invalid byte sequences evade a bound.
bool codex_string_codepoint_length(const std::string &text, size_t *length) {
    *length = 0;
    for (size_t offset = 0; offset < text.size(); ++*length) {
        const auto lead = static_cast<unsigned char>(text[offset]);
        const size_t width = lead < 0x80 ? 1 : lead >= 0xc2 && lead <= 0xdf ? 2 :
            lead >= 0xe0 && lead <= 0xef ? 3 : lead >= 0xf0 && lead <= 0xf4 ? 4 : 0;
        if (!width || width > text.size() - offset) return false;
        unsigned codepoint = width == 1 ? lead : lead & ((1u << (7 - width)) - 1u);
        for (size_t i = 1; i < width; ++i) {
            const auto next = static_cast<unsigned char>(text[offset + i]);
            if ((next & 0xc0) != 0x80) return false;
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        if ((width == 2 && codepoint < 0x80) || (width == 3 && codepoint < 0x800) ||
            (width == 4 && codepoint < 0x10000) || codepoint > 0x10ffff ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff)) return false;
        offset += width;
    }
    return true;
}

// Validate all base assertions, then anyOf at EVERY instance location against
// the retained original schema. Parameter schemas do not constrain tool results.
bool validate_codex_original_argument(const ajson &value, const ajson &schema,
                                     std::string *error, size_t depth = 0, const std::string &path = "$") {
    native_tool_argument_error native_error;
    if (depth > ALICED_JSON_MAX_DEPTH) {
        if (error) *error = path + ": original-schema nesting limit exceeded";
        return false;
    }
    if (!validate_native_tool_argument(value, schema, path, &native_error)) {
        if (error) *error = native_error.path + ": " + native_error.detail;
        return false;
    }
    if (const auto *minimum = schema.get("minItems"); value.is_array() && minimum) {
        if (static_cast<long double>(value.arr.size()) < (minimum->is_int() ?
            static_cast<long double>(minimum->i) : static_cast<long double>(minimum->number()))) {
            if (error) *error = path + ": array violates original minItems=" + ajson_dumps(*minimum);
            return false;
        }
    }
    const auto precise_number = [](const ajson &number) {
        return number.is_int() ? static_cast<long double>(number.i) : static_cast<long double>(number.number());
    };
    if (value.is_string() && (schema.has("minLength") || schema.has("maxLength"))) {
        size_t length = 0;
        if (!codex_string_codepoint_length(value.s, &length)) {
            if (error) *error = path + ": string is not valid Unicode UTF-8";
            return false;
        }
        for (const char *key : {"minLength", "maxLength"}) if (const auto *bound = schema.get(key)) {
            const bool minimum = std::string(key) == "minLength";
            if ((minimum && static_cast<long double>(length) < precise_number(*bound)) ||
                (!minimum && static_cast<long double>(length) > precise_number(*bound))) {
                if (error) *error = path + ": string violates original " + key + "=" + ajson_dumps(*bound);
                return false;
            }
        }
    }
    if (value.is_number() && (schema.has("minimum") || schema.has("maximum"))) {
        if (!std::isfinite(value.number())) {
            if (error) *error = path + ": bounded numeric argument must be finite";
            return false;
        }
        for (const char *key : {"minimum", "maximum"}) if (const auto *bound = schema.get(key)) {
            const bool minimum = std::string(key) == "minimum";
            if ((minimum && precise_number(value) < precise_number(*bound)) ||
                (!minimum && precise_number(value) > precise_number(*bound))) {
                if (error) *error = path + ": number violates original " + key + "=" + ajson_dumps(*bound);
                return false;
            }
        }
    }
    if (const auto *branches = schema.get("anyOf")) {
        bool matched = false;
        if (branches->is_array()) for (const auto &branch : branches->arr) {
            std::string ignored;
            if (validate_codex_original_argument(value, branch, &ignored, depth + 1, path)) { matched = true; break; }
        }
        if (!matched) { if (error) *error = path + ": argument does not satisfy the original anyOf schema"; return false; }
    }
    if (value.is_object()) {
        const auto *properties = schema.get("properties");
        const auto *additional = schema.get("additionalProperties");
        for (size_t i = 0; i < value.keys.size(); ++i) {
            const auto *child = properties ? properties->get(value.keys[i].c_str()) : nullptr;
            if (!child && additional && additional->is_object()) child = additional;
            if (child && !validate_codex_original_argument(value.vals[i], *child, error, depth + 1,
                axiom_codex::schema_path(path, value.keys[i]))) return false;
        }
    }
    if (value.is_array()) if (const auto *items = schema.get("items"))
        for (size_t i = 0; i < value.arr.size(); ++i)
            if (!validate_codex_original_argument(value.arr[i], *items, error, depth + 1,
                path + "[" + std::to_string(i) + "]")) return false;
    return true;
}

void configure_codex_schema_bridge(axiom_codex::tool_wire_map &bridge) {
    bridge.schema_validation(
        [](const ajson &schema, const std::string &path, std::string *error) {
            return validate_native_tool_schema_definition(schema, "codex_schema", path, error);
        },
        [](const ajson &value, const ajson &schema, std::string *error) {
            return validate_codex_original_argument(value, schema, error);
        },
        [](const ajson &tool, std::string *error) {
            if (responses_builtin_tool_type(axiom_codex::string_field(tool, "type"))) {
                *error = "server-hosted builtin has no executor in this bridge; supply a client function tool";
                return false;
            }
            ajson wrappers = ajson::jarr(), functions = ajson::jarr();
            if (!append_normalized_responses_tool(tool, false, &wrappers, error)) return false;
            for (const auto &wrapper : wrappers.arr) {
                const auto *function = wrapper.get("function");
                if (function) functions.push(*function);
            }
            return validate_native_tool_catalog(functions, error);
        });
}

// Only /codex/v1 calls this wrapper. The legacy Responses normalizer and
// global Qwen validators stay unchanged. Keep identities/opaque results in
// the native request, including parallel tool-result history.
bool normalize_codex_responses_request(
        const ajson &request, ajson *payload, std::string *error,
        ajson *prompt_payload = nullptr) {
    // Separate prompt placement from the authoritative validation payload.
    // This out-of-band output is never populated from wire placement flags.
    if (prompt_payload) *prompt_payload = ajson::jobj();
    // Project both halves of Core discovery into Qwen's tool history. Omitting
    // the assistant call makes the model repeat the user's search instruction.
    ajson compatible = request;
    axiom_codex::apply_request_defaults(compatible);
    ajson loaded_tools = ajson::jarr();
    std::map<size_t, ajson> declarations_at_output;
    if (const auto *input = request.get("input"); input && input->is_array()) {
        ajson items = *input;
        for (size_t i = 0; i < items.arr.size(); ++i) {
            const auto &call = input->arr[i];
            if (axiom_codex::string_field(call, "type") != "tool_search_call") continue;
            ajson observations = ajson::jarr();
            std::string pending;
            if (i + 1 >= items.arr.size() ||
                !append_responses_input_item(call, &observations, &loaded_tools, &pending, error)) {
                *error = "Codex tool_search_call requires a following client result"; return false;
            }
            const auto &output = input->arr[i + 1];
            const auto *found = output.get("tools");
            if (axiom_codex::string_field(output, "type") != "tool_search_output" ||
                axiom_codex::string_field(output, "call_id") != pending ||
                axiom_codex::string_field(output, "execution") != "client" ||
                axiom_codex::string_field(output, "status") != "completed" || !found || !found->is_array()) {
                *error = "Codex tool_search_output requires matching call_id, completed client status and tools array"; return false;
            }
            std::string result = "Client tool search completed with no matching tools. No additional tool was loaded.";
            if (!found->arr.empty()) {
                const size_t first_loaded = loaded_tools.arr.size();
                if (!append_responses_input_item(output, &observations, &loaded_tools, &pending, error)) return false;
                result = observations.arr.front().get("content")->s;
                if (prompt_payload) {
                    ajson declarations = ajson::jarr();
                    for (size_t n = first_loaded; n < loaded_tools.arr.size(); ++n)
                        declarations.push(*loaded_tools.arr[n].get("function"));
                    declarations_at_output.emplace(i + 1, std::move(declarations));
                }
            }
            ajson projected_call = call;
            projected_call.set("type", ajson::jstr("function_call"));
            projected_call.set("name", ajson::jstr("tool_search"));
            projected_call.set("arguments", ajson::jstr(ajson_dumps(*call.get("arguments"))));
            ajson projected_output = output;
            projected_output.set("type", ajson::jstr("function_call_output"));
            projected_output.set("output", ajson::jstr(result));
            items.arr[i] = std::move(projected_call); items.arr[++i] = std::move(projected_output);
        }
        compatible.set("input", std::move(items));
    }
    if (!normalize_responses_request(compatible, payload, error)) return false;
    std::set<std::string> immediate_names;
    if (prompt_payload) if (const auto *tools = payload->get("tools"))
        for (const auto &tool : tools->arr)
            immediate_names.insert(axiom_codex::string_field(*tool.get("function"), "name"));
    if (!loaded_tools.arr.empty()) {
        ajson all = payload->get("tools") ? *payload->get("tools") : ajson::jarr();
        for (const auto &tool : loaded_tools.arr) all.push(tool);
        payload->set("tools", std::move(all));
    }
    // Core may rediscover an identical tool. Keep every historical observation
    // but register the callable schema once; conflicts remain hard errors.
    if (const auto *tools = payload->get("tools")) {
        ajson unique = ajson::jarr(); std::map<std::string, std::string> seen;
        for (const auto &tool : tools->arr) {
            const auto *function = tool.get("function");
            if (!function) { *error = "Codex normalized tool has no function"; return false; }
            ajson identity = *function; axiom_codex::erase_field(identity, "axiom_deferred");
            const auto name = axiom_codex::string_field(identity, "name"), encoded = ajson_dumps(identity);
            const auto previous = seen.find(name);
            if (previous != seen.end()) {
                if (previous->second != encoded) { *error = "Codex conflicting loaded tool: " + name; return false; }
                continue;
            }
            // Discovery has completed: Qwen must see the function in its
            // actual advertised catalog, not only as text in a tool result.
            // Core still chooses what to load; no undiscovered tool is added.
            ajson loaded = tool; loaded.set("function", identity);
            seen[name] = encoded; unique.push(std::move(loaded));
        }
        payload->set("tools", std::move(unique));
    }
    for (const char *key : {"session_id", "axiom_session_id", "thread_id", "turn_id", "metadata", "client_metadata", "parallel_tool_calls"})
        if (const auto *value = request.get(key)) payload->set(key, *value);
    const auto *input = request.get("input");
    if (!input || !input->is_array()) {
        if (prompt_payload) *prompt_payload = *payload;
        return true;
    }
    ajson messages = *payload->get("messages");
    size_t index = request.get("instructions") ? 1 : 0;
    for (const auto &item : input->arr) {
        const auto type = axiom_codex::string_field(item, "type");
        if (type == "reasoning" || type == "item_reference") continue;
        if (index >= messages.arr.size()) { *error = "Codex input identity mapping mismatch"; return false; }
        auto &message = messages.arr[index++];
        for (const char *key : {"id", "item_id", "call_id", "tool_id"})
            if (const auto *value = item.get(key)) message.set(key, *value);
        if (type == "tool_search_output") {
            message.set("axiom_original_tool_output", item);
            if (const auto *value = item.get("call_id")) message.set("tool_call_id", *value);
        }
        if (type == "function_call_output" || type == "custom_tool_call_output") {
            if (const auto *value = item.get("call_id")) message.set("tool_call_id", *value);
            if (const auto *value = item.get("output")) {
                message.set("axiom_original_tool_output", *value);
                // Preserve JSON results as JSON, not an empty text extraction.
                if (value->is_object()) message.set("content", ajson::jstr(ajson_dumps(*value)));
            }
        }
    }
    payload->set("messages", std::move(messages));
    if (prompt_payload) {
        ajson prompt = *payload, header_tools = ajson::jarr();
        if (const auto *tools = payload->get("tools")) {
            for (const auto &tool : tools->arr)
                if (immediate_names.count(axiom_codex::string_field(*tool.get("function"), "name")))
                    header_tools.push(tool);
            prompt.set("tools", std::move(header_tools));
        }
        // Anchor each newly loaded schema after its first validated discovery
        // result. Rebuild these positions from the full authoritative history;
        // never move old declarations to the newest tail or hoist result prose.
        ajson prompt_messages = ajson::jarr();
        const auto &original_messages = payload->get("messages")->arr;
        size_t message_index = 0;
        if (request.get("instructions")) prompt_messages.push(original_messages[message_index++]);
        for (size_t i = 0; i < input->arr.size(); ++i) {
            const auto type = axiom_codex::string_field(input->arr[i], "type");
            if (type == "reasoning" || type == "item_reference") continue;
            prompt_messages.push(original_messages[message_index++]);
            const auto found = declarations_at_output.find(i);
            if (found == declarations_at_output.end() || !tools_are_enabled(payload->get("tool_choice"))) continue;
            ajson additions = ajson::jarr();
            for (const auto &function : found->second.arr)
                if (immediate_names.insert(axiom_codex::string_field(function, "name")).second)
                    additions.push(function);
            if (additions.arr.empty()) continue;
            ajson declaration = ajson::jobj();
            declaration.set("role", ajson::jstr("system"));
            declaration.set("content", ajson::jstr(render_qwen_deferred_tools_prompt(additions)));
            prompt_messages.push(std::move(declaration));
        }
        prompt.set("messages", std::move(prompt_messages));
        *prompt_payload = std::move(prompt);
    }
    return true;
}

std::string safe_tool_name(const std::string &name) {
    std::string out;
    out.reserve(std::min<size_t>(name.size(), 48u));
    for (char ch : name) {
        const bool safe = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
        if (safe && out.size() < 48u) out.push_back(ch);
        else if (out.size() < 48u) out.push_back('_');
    }
    return out.empty() ? "tool" : out;
}

native_tool_result parse_native_tool_calls(
        const std::string &text, const ajson &tools,
        bool implicit_thinking = false) {
    native_tool_result result;
    result.status = "none";
    const std::string thinkless = [&]() {
        std::string out;
        size_t pos = 0u;
        // A reasoning-enabled prompt already supplies <think>. Its generated
        // continuation need not repeat that opening tag. Only the real visible
        // suffix may authorize a call; planning examples are inert. Use the
        // generation mode, not a guessed closing tag inside an off-mode tool.
        if (implicit_thinking) {
            const size_t end = text.find("</think>");
            if (end == std::string::npos) return out;
            pos = end + 8u;
        }
        while (pos < text.size()) {
            const size_t start = text.find("<think>", pos);
            if (start == std::string::npos) {
                out.append(text, pos, std::string::npos);
                break;
            }
            out.append(text, pos, start - pos);
            const size_t end = text.find("</think>", start + 7u);
            if (end == std::string::npos) break;
            pos = end + 8u;
        }
        return out;
    }();
    result.raw = thinkless;
    result.content = trim_text(thinkless);

    size_t scan = 0u;
    size_t first_call = std::string::npos;
    size_t after_calls = 0u;
    while (true) {
        const size_t start = thinkless.find("<tool_call>", scan);
        if (start == std::string::npos) break;
        const size_t end_tag = thinkless.find("</tool_call>", start + 11u);
        if (end_tag == std::string::npos) {
            result.status = "fail";
            result.error = "unterminated <tool_call> block";
            return result;
        }
        const size_t end = end_tag + 12u;
        if (first_call == std::string::npos) first_call = start;
        after_calls = end;
        const std::string block = thinkless.substr(start + 11u, end_tag - (start + 11u));
        const size_t function_start = block.find("<function=");
        if (function_start == std::string::npos) {
            result.status = "fail";
            result.error = "tool_call is missing <function=...>";
            return result;
        }
        const size_t name_start = function_start + 10u;
        const size_t name_end = block.find('>', name_start);
        if (name_end == std::string::npos || name_end == name_start) {
            result.status = "fail";
            result.error = "tool_call has an invalid function name";
            return result;
        }
        const std::string name = block.substr(name_start, name_end - name_start);
        const ajson *function = tool_function_by_name(tools, name);
        if (!function) {
            result.status = "fail";
            result.error = "model emitted an unknown tool: " + name;
            return result;
        }
        const size_t function_end = block.find("</function>", name_end + 1u);
        if (function_end == std::string::npos) {
            result.status = "fail";
            result.error = "tool_call is missing </function>";
            return result;
        }
        ajson args = ajson::jobj();
        size_t parameter_scan = name_end + 1u;
        while (true) {
            const size_t parameter_start = block.find("<parameter=", parameter_scan);
            if (parameter_start == std::string::npos || parameter_start >= function_end) break;
            const size_t parameter_name_start = parameter_start + 11u;
            const size_t parameter_name_end = block.find('>', parameter_name_start);
            if (parameter_name_end == std::string::npos || parameter_name_end >= function_end) {
                result.status = "fail";
                result.error = "tool_call has an invalid parameter header";
                return result;
            }
            const std::string parameter_name = block.substr(
                    parameter_name_start, parameter_name_end - parameter_name_start);
            const std::string close = "</parameter>";
            const size_t parameter_end = block.find(close, parameter_name_end + 1u);
            if (parameter_end == std::string::npos || parameter_end > function_end) {
                result.status = "fail";
                result.error = "tool_call has an unterminated parameter";
                return result;
            }
            const std::string body = block.substr(
                    parameter_name_end + 1u, parameter_end - (parameter_name_end + 1u));
            args.set(parameter_name, parse_native_parameter(
                    body, tool_parameter_is_string(function, parameter_name)));
            parameter_scan = parameter_end + close.size();
        }
        const ajson *parameters = function->get("parameters");
        native_tool_argument_error argument_error;
        if (parameters && !validate_native_tool_argument(
                args, *parameters, "$", &argument_error)) {
            result.status = "fail";
            result.error = format_native_tool_argument_error(name, argument_error);
            return result;
        }
        ajson fn = ajson::jobj();
        fn.set("name", ajson::jstr(name));
        fn.set("arguments", ajson::jstr(ajson_dumps(args)));
        ajson call = ajson::jobj();
        const std::string id = "call_axiom_qwen_" + safe_tool_name(name) + "_" +
                std::to_string(static_cast<unsigned long long>(std::time(nullptr))) + "_" +
                std::to_string(result.calls.size());
        call.set("id", ajson::jstr(id));
        call.set("type", ajson::jstr("function"));
        call.set("function", std::move(fn));
        result.calls.push_back(std::move(call));
        scan = end;
    }
    if (result.calls.empty()) return result;
    result.status = "pass";
    result.content = trim_text(
            thinkless.substr(0u, first_call) + thinkless.substr(after_calls));
    return result;
}

std::string native_tool_contract_self_test_call(
        const std::string &name,
        const std::vector<std::pair<std::string, std::string>> &parameters) {
    std::string text = "<tool_call>\n<function=" + name + ">\n";
    for (const auto &parameter : parameters) {
        text += "<parameter=" + parameter.first + ">\n";
        text += parameter.second;
        text += "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

int run_native_tool_contract_self_test() {
    constexpr const char *schema_text = R"AXIOM_JSON(
{
  "type":"object",
  "properties":{
    "command":{"type":"string"},
    "mode":{"type":"string","enum":["read","write"]},
    "version":{"type":"integer","const":2},
    "ratio":{"type":"number"},
    "dry_run":{"type":"boolean"},
    "marker":{"type":"null"},
    "payload":{
      "type":"object",
      "properties":{
        "items":{
          "type":"array",
          "items":{
            "type":"object",
            "properties":{
              "id":{"type":"integer"},
              "enabled":{"type":"boolean"},
              "label":{"type":"string"}
            },
            "required":["id","enabled"],
            "additionalProperties":false
          }
        }
      },
      "required":["items"],
      "additionalProperties":false
    }
  },
  "required":["command","mode","version","ratio","dry_run","marker","payload"],
  "additionalProperties":false
}
)AXIOM_JSON";
    ajson parameter_schema;
    std::string error;
    if (!ajson_parse(schema_text, parameter_schema, error)) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=schema_parse error=%s\n",
                     error.c_str());
        return 1;
    }

    ajson function = ajson::jobj();
    function.set("name", ajson::jstr("schema_tool"));
    function.set("parameters", parameter_schema);
    ajson wrapper = ajson::jobj();
    wrapper.set("type", ajson::jstr("function"));
    wrapper.set("function", std::move(function));
    ajson chat_request_tools = ajson::jarr();
    chat_request_tools.push(std::move(wrapper));
    ajson chat_tools;
    if (!normalize_tools(&chat_request_tools, false, &chat_tools, &error)) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=chat_normalize error=%s\n",
                     error.c_str());
        return 1;
    }

    ajson responses_tool = ajson::jobj();
    responses_tool.set("type", ajson::jstr("function"));
    responses_tool.set("name", ajson::jstr("schema_tool"));
    responses_tool.set("parameters", parameter_schema);
    ajson responses_tools = ajson::jarr();
    responses_tools.push(std::move(responses_tool));
    ajson responses_request = ajson::jobj();
    responses_request.set("model", ajson::jstr(kModelId));
    responses_request.set("input", ajson::jstr("tool contract probe"));
    responses_request.set("tools", std::move(responses_tools));
    ajson responses_chat_request;
    error.clear();
    if (!normalize_responses_request(
            responses_request, &responses_chat_request, &error)) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=responses_ingest error=%s\n",
                     error.c_str());
        return 1;
    }
    ajson responses_normalized_tools;
    error.clear();
    if (!normalize_tools(
            responses_chat_request.get("tools"), false,
            &responses_normalized_tools, &error)) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=responses_normalize error=%s\n",
                     error.c_str());
        return 1;
    }

    const auto make_chat_tool = [&](const std::string &name, const ajson &schema) {
        ajson candidate = ajson::jobj();
        candidate.set("name", ajson::jstr(name));
        candidate.set("parameters", schema);
        ajson candidate_wrapper = ajson::jobj();
        candidate_wrapper.set("type", ajson::jstr("function"));
        candidate_wrapper.set("function", std::move(candidate));
        return candidate_wrapper;
    };
    const auto make_deferred_responses_request = [
            &parameter_schema](
            const std::string &immediate_name,
            const std::string &loaded_name,
            bool include_search_call,
            bool defer_loading) {
        ajson empty_schema = ajson::jobj();
        empty_schema.set("type", ajson::jstr("object"));
        empty_schema.set("properties", ajson::jobj());
        empty_schema.set("additionalProperties", ajson::jbool(false));

        ajson immediate = ajson::jobj();
        immediate.set("type", ajson::jstr("function"));
        immediate.set("name", ajson::jstr(immediate_name));
        immediate.set("description", ajson::jstr("Search the mounted tool registry"));
        immediate.set("parameters", std::move(empty_schema));
        ajson top_level_tools = ajson::jarr();
        top_level_tools.push(std::move(immediate));

        ajson input = ajson::jarr();
        ajson user = ajson::jobj();
        user.set("type", ajson::jstr("message"));
        user.set("role", ajson::jstr("user"));
        user.set("content", ajson::jstr("load the exact tool"));
        input.push(std::move(user));
        if (include_search_call) {
            ajson search_call = ajson::jobj();
            search_call.set("type", ajson::jstr("tool_search_call"));
            search_call.set("call_id", ajson::jstr("pi_tool_load_contract"));
            search_call.set("execution", ajson::jstr("client"));
            search_call.set("status", ajson::jstr("completed"));
            ajson arguments = ajson::jobj();
            arguments.set("query", ajson::jstr(loaded_name));
            arguments.set("limit", ajson::jint(1));
            search_call.set("arguments", std::move(arguments));
            input.push(std::move(search_call));
        }
        ajson loaded = ajson::jobj();
        loaded.set("type", ajson::jstr("function"));
        loaded.set("name", ajson::jstr(loaded_name));
        loaded.set("description", ajson::jstr("Deferred contract probe"));
        loaded.set("parameters", parameter_schema);
        if (defer_loading) loaded.set("defer_loading", ajson::jbool(true));
        ajson loaded_tools = ajson::jarr();
        loaded_tools.push(std::move(loaded));
        ajson search_output = ajson::jobj();
        search_output.set("type", ajson::jstr("tool_search_output"));
        search_output.set("call_id", ajson::jstr("pi_tool_load_contract"));
        search_output.set("execution", ajson::jstr("client"));
        search_output.set("status", ajson::jstr("completed"));
        search_output.set("tools", std::move(loaded_tools));
        input.push(std::move(search_output));

        ajson request = ajson::jobj();
        request.set("model", ajson::jstr(kModelId));
        request.set("input", std::move(input));
        request.set("tools", std::move(top_level_tools));
        return request;
    };

    ajson deferred_request = make_deferred_responses_request(
            "tool_search", "deferred_schema_tool", true, true);
    ajson deferred_chat_request;
    error.clear();
    if (!normalize_responses_request(
            deferred_request, &deferred_chat_request, &error)) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=deferred_ingest error=%s\n",
                     error.c_str());
        return 1;
    }
    ajson deferred_normalized_tools;
    error.clear();
    if (!normalize_tools(
            deferred_chat_request.get("tools"), false,
            &deferred_normalized_tools, &error)) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=deferred_normalize error=%s\n",
                     error.c_str());
        return 1;
    }
    const ajson *deferred_definition = tool_function_by_name(
            deferred_normalized_tools, "deferred_schema_tool");
    const ajson *deferred_marker = deferred_definition
            ? deferred_definition->get("axiom_deferred") : nullptr;
    const std::string immediate_prompt = render_qwen_tools_prompt(
            deferred_normalized_tools, nullptr);
    bool history_has_loaded_definition = false;
    const ajson *deferred_messages = deferred_chat_request.get("messages");
    if (deferred_messages && deferred_messages->is_array()) {
        for (const ajson &message : deferred_messages->arr) {
            const ajson *role = message.get("role");
            const ajson *content = message.get("content");
            if (role && role->is_string() && role->s == "system" &&
                content && content->is_string() &&
                content->s.find("deferred_schema_tool") != std::string::npos) {
                history_has_loaded_definition = true;
            }
        }
    }
    if (!deferred_marker || !deferred_marker->is_bool() || !deferred_marker->b ||
        immediate_prompt.find("tool_search") == std::string::npos ||
        immediate_prompt.find("deferred_schema_tool") != std::string::npos ||
        !history_has_loaded_definition) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=deferred_placement\n");
        return 1;
    }

    ajson ignored_tools;
    ajson malformed_deferred = make_deferred_responses_request(
            "tool_search", "deferred_schema_tool", true, false);
    error.clear();
    if (normalize_responses_request(
            malformed_deferred, &ignored_tools, &error) ||
        error.find("defer_loading=true") == std::string::npos) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=deferred_marker error=%s\n",
                     error.c_str());
        return 1;
    }
    ajson orphaned_deferred = make_deferred_responses_request(
            "tool_search", "deferred_schema_tool", false, true);
    error.clear();
    if (normalize_responses_request(
            orphaned_deferred, &ignored_tools, &error) ||
        error.find("matching call_id") == std::string::npos) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=deferred_orphan error=%s\n",
                     error.c_str());
        return 1;
    }
    ajson duplicate_deferred = make_deferred_responses_request(
            "deferred_schema_tool", "deferred_schema_tool", true, true);
    ajson duplicate_deferred_chat;
    error.clear();
    if (!normalize_responses_request(
            duplicate_deferred, &duplicate_deferred_chat, &error) ||
        normalize_tools(
            duplicate_deferred_chat.get("tools"), false,
            &ignored_tools, &error) ||
        error.find("duplicate tool name") == std::string::npos) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=deferred_duplicate error=%s\n",
                     error.c_str());
        return 1;
    }
    ajson invalid_catalog = ajson::jarr();
    invalid_catalog.push(make_chat_tool("Bash", parameter_schema));
    error.clear();
    if (normalize_tools(&invalid_catalog, false, &ignored_tools, &error) ||
        error.find("name must match") == std::string::npos) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=canonical_name error=%s\n",
                     error.c_str());
        return 1;
    }
    invalid_catalog = ajson::jarr();
    invalid_catalog.push(make_chat_tool("schema_tool", parameter_schema));
    invalid_catalog.push(make_chat_tool("schema_tool", parameter_schema));
    error.clear();
    if (normalize_tools(&invalid_catalog, false, &ignored_tools, &error) ||
        error.find("duplicate tool name") == std::string::npos) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=duplicate_name error=%s\n",
                     error.c_str());
        return 1;
    }
    ajson unsupported_schema = parameter_schema;
    unsupported_schema.set("oneOf", ajson::jarr());
    invalid_catalog = ajson::jarr();
    invalid_catalog.push(make_chat_tool("schema_tool", unsupported_schema));
    error.clear();
    if (normalize_tools(&invalid_catalog, false, &ignored_tools, &error) ||
        error.find("unsupported schema keyword oneOf") == std::string::npos) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: tool contract self-test FAIL case=unsupported_schema error=%s\n",
                     error.c_str());
        return 1;
    }

    using self_test_parameter = std::pair<std::string, std::string>;
    const std::vector<self_test_parameter> valid_parameters = {
        {"command", "run"},
        {"mode", "read"},
        {"version", "2"},
        {"ratio", "1.25"},
        {"dry_run", "false"},
        {"marker", "null"},
        {"payload", R"AXIOM_JSON({"items":[{"id":7,"enabled":true,"label":"ok"}]})AXIOM_JSON"},
    };
    size_t cases = 0u;
    const auto expect = [&](
            const char *case_name, const ajson &tools,
            const std::string &tool_name,
            const std::vector<self_test_parameter> &parameters,
            const char *expected_status,
            const char *expected_error) -> bool {
        ++cases;
        const native_tool_result result = parse_native_tool_calls(
                native_tool_contract_self_test_call(tool_name, parameters), tools);
        const bool error_matches = expected_error
                ? result.error == expected_error : result.error.empty();
        const bool calls_match = std::strcmp(expected_status, "pass") != 0 ||
                result.calls.size() == 1u;
        if (result.status != expected_status || !error_matches || !calls_match) {
            std::fprintf(stderr,
                         "axiom-qwen38-api: tool contract self-test FAIL case=%s status=%s calls=%zu error=%s\n",
                         case_name, result.status.c_str(), result.calls.size(),
                         result.error.c_str());
            return false;
        }
        return true;
    };

    if (!expect("chat_valid", chat_tools, "schema_tool", valid_parameters,
                "pass", nullptr) ||
        !expect("responses_valid", responses_normalized_tools, "schema_tool",
                valid_parameters, "pass", nullptr) ||
        !expect("responses_deferred_valid", deferred_normalized_tools,
                "deferred_schema_tool", valid_parameters, "pass", nullptr)) {
        return 1;
    }

    std::vector<self_test_parameter> changed = valid_parameters;
    changed.erase(changed.begin());
    constexpr const char *missing_command =
            "tool argument validation failed: tool=\"schema_tool\" code=required path=$[\"command\"] detail=missing required property";
    if (!expect("chat_required", chat_tools, "schema_tool", changed,
                "fail", missing_command) ||
        !expect("responses_required", responses_normalized_tools, "schema_tool",
                changed, "fail", missing_command)) {
        return 1;
    }

    changed = valid_parameters;
    changed.push_back({"rogue", "true"});
    if (!expect(
            "additional_property", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=additional_property path=$[\"rogue\"] detail=property is not declared")) {
        return 1;
    }

    changed = valid_parameters;
    changed[2].second = "\"2\"";
    if (!expect(
            "no_integer_coercion", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=type path=$[\"version\"] detail=expected integer, got string")) {
        return 1;
    }

    changed = valid_parameters;
    changed[1].second = "delete";
    if (!expect(
            "enum", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=enum path=$[\"mode\"] detail=value is not a declared enum member")) {
        return 1;
    }

    changed = valid_parameters;
    changed[2].second = "3";
    if (!expect(
            "const", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=const path=$[\"version\"] detail=value does not equal the declared const")) {
        return 1;
    }

    changed = valid_parameters;
    changed[3].second = "\"1.25\"";
    if (!expect(
            "number_type", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=type path=$[\"ratio\"] detail=expected number, got string")) {
        return 1;
    }

    changed = valid_parameters;
    changed[4].second = "\"false\"";
    if (!expect(
            "boolean_type", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=type path=$[\"dry_run\"] detail=expected boolean, got string")) {
        return 1;
    }

    changed = valid_parameters;
    changed[5].second = "\"null\"";
    if (!expect(
            "null_type", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=type path=$[\"marker\"] detail=expected null, got string")) {
        return 1;
    }

    changed = valid_parameters;
    changed[6].second =
            R"AXIOM_JSON({"items":[{"id":"7","enabled":true}]})AXIOM_JSON";
    if (!expect(
            "nested_type", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=type path=$[\"payload\"][\"items\"][0][\"id\"] detail=expected integer, got string")) {
        return 1;
    }

    changed = valid_parameters;
    changed[6].second =
            R"AXIOM_JSON({"items":[{"enabled":true}]})AXIOM_JSON";
    if (!expect(
            "nested_required", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=required path=$[\"payload\"][\"items\"][0][\"id\"] detail=missing required property")) {
        return 1;
    }

    changed = valid_parameters;
    changed[6].second =
            R"AXIOM_JSON({"items":[{"id":7,"enabled":true,"rogue":1}]})AXIOM_JSON";
    if (!expect(
            "nested_additional_property", chat_tools, "schema_tool", changed,
            "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=additional_property path=$[\"payload\"][\"items\"][0][\"rogue\"] detail=property is not declared")) {
        return 1;
    }

    changed = valid_parameters;
    changed[6].second = R"AXIOM_JSON({"items":[1]})AXIOM_JSON";
    if (!expect(
            "recursive_array_item", chat_tools, "schema_tool", changed, "fail",
            "tool argument validation failed: tool=\"schema_tool\" code=type path=$[\"payload\"][\"items\"][0] detail=expected object, got integer")) {
        return 1;
    }

    if (!expect(
            "case_sensitive_name", chat_tools, "SchemaTool", valid_parameters,
            "fail", "model emitted an unknown tool: SchemaTool")) {
        return 1;
    }

    std::printf(
            "axiom-qwen38-api: tool contract self-test PASS cases=%zu chat=pass responses=pass catalog=pass recursive=pass strict=pass\n",
            cases);
    return 0;
}

std::string visible_model_text(const std::string &text, bool reasoning_enabled = true) {
    if (!reasoning_enabled) return text;
    const size_t end = text.find("</think>");
    if (end != std::string::npos) {
        const size_t first = end + 8u;
        return text.substr(first + (text.compare(first, 2u, "\n\n") == 0 ? 2u : 0u));
    }
    const size_t start = text.find("<think>");
    if (start != std::string::npos) return trim_text(text.substr(0u, start));
    return text;
}

std::string reasoning_model_text(const std::string &text, bool reasoning_enabled = true) {
    if (!reasoning_enabled) return {};
    const size_t end = text.find("</think>");
    if (end == std::string::npos) return {};
    const size_t start = text.find("<think>");
    const size_t first = start != std::string::npos && start < end
            ? start + std::strlen("<think>") : 0u;
    return trim_text(text.substr(first, end - first));
}

struct chat_turn {
    std::string role;
    std::string content;
    // Tool replies use Qwen's user role, but do not start a new user request.
    bool tool_response = false;
};

struct encoded_chat_turn {
    chat_turn turn;
    std::vector<uint32_t> ids;
};

// Shared by text and vision: conditioning, tools, then the initial system
// message, in one native system turn. Keep content parts/bytes intact so vision
// slots and authoritative replay edits are not lost to text normalization.
ajson initial_chat_system(const server_state *state, const ajson &payload,
        const ajson &messages, const ajson &tools, bool no_think,
        size_t *first_message) {
    ajson parts = ajson::jarr();
    const auto append = [&](const ajson &content) {
        if (content.is_null() || (content.is_string() && content.s.empty()) ||
            (content.is_array() && content.arr.empty())) return;
        if (!parts.arr.empty()) parts.push(ajson::jstr("\n\n"));
        if (content.is_array()) {
            for (const auto &part : content.arr) parts.push(part);
        } else parts.push(content);
    };
    if (!no_think) append(ajson::jstr(native_reasoning_instruction(
            effective_reasoning_effort(state, payload))));
    append(ajson::jstr(render_qwen_tools_prompt(tools, payload.get("tool_choice"))));
    if (const auto *system = payload.get("system")) append(*system);
    *first_message = 0u;
    if (!messages.arr.empty()) {
        const auto &first = messages.arr.front();
        const auto *role = first.get("role");
        if (role && role->is_string() && role->s == "system" && first.get("content")) {
            append(*first.get("content"));
            *first_message = 1u;
        }
    }
    return parts;
}

std::string assistant_history_prefix(const ajson &message) {
    const auto *reasoning = message.get("reasoning_content");
    if (reasoning && reasoning->is_string()) {
        return "<think>\n" + trim_text(reasoning->s) + "\n</think>\n\n";
    }
    // Preserve the native full-text representation accepted by older clients.
    if (message_content_text(message.get("content")).rfind("<think>", 0u) == 0u) return {};
    return kNoThinkingPrefix;
}

std::string compact_snippet(const std::string &text, const size_t limit = 260u) {
    std::string compact;
    compact.reserve(std::min(text.size(), limit));
    bool separated = false;
    for (const char ch : text) {
        const bool whitespace = ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
        if (whitespace) {
            separated = !compact.empty();
            continue;
        }
        if (separated && !compact.empty()) compact.push_back(' ');
        separated = false;
        compact.push_back(ch);
        if (compact.size() >= limit) break;
    }
    return compact;
}

bool compact_important(const std::string &text) {
    const std::string lower = ascii_lower(text);
    static constexpr const char *kMarkers[] = {
            "todo", "decision", "deciso", "errore", "bug", "fix", "path",
            "file", "porta", "endpoint", "token", "chiave", "server", "vm",
            "deploy", "commit", "branch", "test", "fail", "pass", "memoria",
            "regola", "vincolo", "ricorda", "importante",
    };
    for (const char *marker : kMarkers) {
        if (lower.find(marker) != std::string::npos) return true;
    }
    return false;
}

std::string compact_role_label(const std::string &role) {
    return role == "user" ? "Utente" : role == "assistant" ? "Axiom" : role;
}

bool compact_tool_record(const chat_turn &turn) {
    return turn.tool_response || turn.content.find("<tool_response>") != std::string::npos ||
            (turn.role == "assistant" && turn.content.find("<tool_call>") != std::string::npos);
}

std::string compact_record(const chat_turn &turn) {
    std::string label = compact_role_label(turn.role);
    if (turn.tool_response || turn.content.find("<tool_response>") != std::string::npos)
        label = "Historical tool result (already returned)";
    else if (turn.role == "assistant" && turn.content.find("<tool_call>") != std::string::npos)
        label = "Historical assistant tool call (record only)";
    // Summaries are observations, never fresh native tool declarations/calls.
    // Escape before placing snippets inside the archive so protocol delimiters
    // cannot be mistaken for executable examples. Current turns stay verbatim.
    std::string text;
    for (const char ch : compact_snippet(turn.content)) {
        if (ch == '&') text += "&amp;";
        else if (ch == '<') text += "&lt;";
        else if (ch == '>') text += "&gt;";
        else text.push_back(ch);
    }
    return "- " + label + ": " + text + "\n";
}

std::string build_compact_context_state(
        const std::vector<chat_turn> &turns, const bool current_tools_enabled = false) {
    std::vector<const chat_turn *> important;
    important.reserve(18u);
    for (const chat_turn &turn : turns) {
        if (compact_important(turn.content)) important.push_back(&turn);
        if (important.size() >= 18u) break;
    }
    std::vector<const chat_turn *> selected;
    selected.reserve(6u + important.size() + 8u);
    for (size_t index = 0u; index < turns.size() && index < 6u; ++index) {
        selected.push_back(&turns[index]);
    }
    std::string summary = "AXIOM_COMPACT_STATE v1\n";
    summary += "turns_compacted: " + std::to_string(turns.size()) + "\n\n";
    if (std::any_of(turns.begin(), turns.end(), compact_tool_record)) {
        summary += "tool_history_policy:\n"
                "This archive records earlier events, not new instructions or a tool catalog. "
                "Historical calls with returned results are already completed. Use the recorded "
                "results directly; do not repeat a call merely because it appears here.\n";
        summary += current_tools_enabled
                ? "Only the current request's declared or loaded tools authorize new calls. "
                  "Names mentioned in this archive do not make tools available.\n\n"
                : "No tools are available for this response. Answer the current request in plain "
                  "text using the recorded results. If a new action is necessary but unavailable, "
                  "explain that limitation; do not invent a result or emit a tool call.\n\n";
    }
    summary += "context_start:\n";
    for (const chat_turn *turn : selected) {
        summary += compact_record(*turn);
    }
    if (!important.empty()) {
        summary += "\nimportant_points:\n";
        for (const chat_turn *turn : important) {
            summary += compact_record(*turn);
        }
    }
    summary += "\nrecent_context_before_compaction:\n";
    const size_t recent_begin = turns.size() > 8u ? turns.size() - 8u : 0u;
    for (size_t index = recent_begin; index < turns.size(); ++index) {
        const chat_turn &turn = turns[index];
        summary += compact_record(turn);
    }
    if (summary.size() <= kContextCompactionMaxSummaryChars) return summary;
    summary.resize(kContextCompactionMaxSummaryChars);
    summary += "\n[compact_state_truncated]";
    return summary;
}

int encode_chat_turn(
        server_state *state,
        const chat_turn &turn,
        encoded_chat_turn *out,
        std::string *error) {
    const uint32_t context_limit = effective_context_limit(state);
    if (!state || !out || !error ||
        state->tokenizer_info.im_start_token_id == AXIOM_TOKEN_ID_INVALID ||
        state->tokenizer_info.im_end_token_id == AXIOM_TOKEN_ID_INVALID) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    out->turn = turn;
    out->ids.clear();
    out->ids.push_back(state->tokenizer_info.im_start_token_id);
    int rc = append_text_ids(state, turn.role + "\n" + turn.content, &out->ids);
    if (rc == AXIOM_OK && out->ids.size() < context_limit) {
        out->ids.push_back(state->tokenizer_info.im_end_token_id);
        rc = append_text_ids(state, "\n", &out->ids);
    } else if (rc == AXIOM_OK) {
        rc = AXIOM_ERR_BUDGET;
    }
    if (rc != AXIOM_OK) *error = "prompt exceeds configured context";
    return rc;
}

int encode_assistant_prefix(
        server_state *state,
        bool no_think,
        std::vector<uint32_t> *out,
        std::string *error) {
    if (!state || !out || !error) return AXIOM_ERR_INVALID_ARGUMENT;
    out->clear();
    if (state->tokenizer_info.im_start_token_id == AXIOM_TOKEN_ID_INVALID) {
        *error = "Qwen ChatML tokens are missing from the tokenizer";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    out->push_back(state->tokenizer_info.im_start_token_id);
    int rc = append_text_ids(state, "assistant\n", out);
    if (rc == AXIOM_OK) {
        rc = append_text_ids(state, no_think ? kNoThinkingPrefix : "<think>\n", out);
    }
    if (rc != AXIOM_OK) *error = "prompt exceeds configured context";
    return rc;
}

void append_encoded_turns(
        const std::vector<encoded_chat_turn> &turns,
        const std::vector<size_t> &indices,
        std::vector<uint32_t> *out) {
    for (const size_t index : indices) {
        out->insert(out->end(), turns[index].ids.begin(), turns[index].ids.end());
    }
}

bool make_chat_ids(
        server_state *state,
        const ajson &payload,
        bool anthropic,
        std::vector<uint32_t> *out,
        std::vector<uint32_t> *session_suffix_ids,
        std::string *error,
        uint32_t *original_tokens,
        bool *compacted,
        const ajson *tool_authority = nullptr) {
    if (!state || !out || !session_suffix_ids || !error || !original_tokens || !compacted) {
        return false;
    }
    const uint32_t context_limit = effective_context_limit(state);
    *original_tokens = 0u;
    *compacted = false;
    session_suffix_ids->clear();
    const ajson *messages = payload.get("messages");
    if (!messages || !messages->is_array() || messages->arr.empty()) {
        *error = "messages must be a non-empty array";
        return false;
    }
    if (state->tokenizer_info.im_start_token_id == AXIOM_TOKEN_ID_INVALID ||
        state->tokenizer_info.im_end_token_id == AXIOM_TOKEN_ID_INVALID) {
        *error = "Qwen ChatML tokens are missing from the tokenizer";
        return false;
    }
    const bool no_think = resolve_no_think(state, payload, error);
    if (!error->empty() && !no_think) return false;
    out->clear();
    std::vector<chat_turn> all_turns;
    ajson normalized_tools;
    if (!normalize_tools(payload.get("tools"), anthropic, &normalized_tools, error)) return false;
    const ajson *tool_choice = payload.get("tool_choice");
    const bool tools_enabled = !normalized_tools.arr.empty() && tools_are_enabled(tool_choice);
    size_t first_message = 0u;
    const ajson system = initial_chat_system(state, payload, *messages,
            normalized_tools, no_think, &first_message);
    const std::string system_text = message_content_text(&system);
    if (!system_text.empty()) all_turns.push_back(chat_turn{"system", system_text});
    bool previous_tool = false;
    for (size_t message_index = first_message; message_index < messages->arr.size(); ++message_index) {
        const ajson &message = messages->arr[message_index];
        const ajson *role = message.get("role");
        const ajson *content = message.get("content");
        if (!message.is_object() || !role || !role->is_string() ||
            (!content && role->s != "assistant") ||
            (role->s != "system" && role->s != "user" && role->s != "assistant" &&
             role->s != "tool")) {
            *error = "each message must have role system/user/assistant/tool and content";
            return false;
        }
        std::string content_text = message_content_text(content);
        std::string tool_call_text;
        if (role->s == "assistant") {
            const ajson *tool_calls = message.get("tool_calls");
            if (tool_calls && tool_calls->is_array()) {
                for (const ajson &call : tool_calls->arr) {
                    const std::string current = native_tool_call_text(call);
                    if (!current.empty()) {
                        if (!tool_call_text.empty()) tool_call_text += "\n";
                        tool_call_text += current;
                    }
                }
            } else {
                tool_call_text = native_tool_calls_from_content(content);
            }
            if (!tool_call_text.empty()) {
                if (!content_text.empty()) content_text += "\n\n";
                content_text += tool_call_text;
            }
            content_text.insert(0u, assistant_history_prefix(message));
        }
        if (role->s == "tool") {
            content_text = "<tool_response>\n" + content_text + "\n</tool_response>";
            if (previous_tool) all_turns.back().content += "\n" + content_text;
            else all_turns.push_back(chat_turn{"user", content_text, true});
        } else {
            const bool tool_response = role->s == "user" &&
                    content_text.find("<tool_response>") != std::string::npos;
            all_turns.push_back(chat_turn{role->s, content_text, tool_response});
        }
        previous_tool = role->s == "tool";
    }

    std::vector<encoded_chat_turn> encoded;
    encoded.reserve(all_turns.size());
    uint64_t raw_tokens = 0u;
    // Historical replay may cross the selected window precisely when the
    // governor is needed. Bound tokenization by the loaded model's ceiling,
    // then enforce the selected window on the actual compacted prompt. This
    // never extends the generation/KV budget or truncates a one-shot document.
    const uint32_t history_limit = state->context_compaction
            ? std::max(context_limit, state->max_context) : context_limit;
    {
        request_context_scope history_scope(state, history_limit);
        for (const chat_turn &turn : all_turns) {
            encoded_chat_turn item;
            if (encode_chat_turn(state, turn, &item, error) != AXIOM_OK) return false;
            raw_tokens += item.ids.size();
            if (raw_tokens > history_limit) {
                *error = "prompt exceeds configured context";
                return false;
            }
            encoded.push_back(std::move(item));
        }
    }

    std::vector<uint32_t> assistant_prefix;
    if (encode_assistant_prefix(state, no_think, &assistant_prefix, error) != AXIOM_OK) return false;
    raw_tokens += assistant_prefix.size();
    if (raw_tokens > history_limit) {
        *error = "prompt exceeds configured context";
        return false;
    }
    *original_tokens = static_cast<uint32_t>(raw_tokens);

    /* The durable model state already owns every token through the most recent
     * assistant response.  Preserve just the subsequent tool/user turns and
     * the new assistant prefix so activate_session can extend exact native
     * state even when the client's historical assistant representation is
     * normalized or omits hidden thinking. */
    for (size_t index = encoded.size(); index > 0u; --index) {
        const size_t assistant_index = index - 1u;
        if (encoded[assistant_index].turn.role != "assistant" ||
            assistant_index + 1u >= encoded.size()) {
            continue;
        }
        try {
            // The saved cursor consumes the assistant's im_end exactly once;
            // its trailing ChatML newline belongs to the new replay suffix.
            if (append_text_ids(state, "\n", session_suffix_ids) != AXIOM_OK) {
                *error = "session resume tail exceeds configured context";
                return false;
            }
            for (size_t tail = assistant_index + 1u; tail < encoded.size(); ++tail) {
                session_suffix_ids->insert(
                        session_suffix_ids->end(),
                        encoded[tail].ids.begin(), encoded[tail].ids.end());
            }
            session_suffix_ids->insert(
                    session_suffix_ids->end(), assistant_prefix.begin(), assistant_prefix.end());
        } catch (...) {
            session_suffix_ids->clear();
            *error = "session resume tail allocation failed";
            return false;
        }
        break;
    }

    std::vector<size_t> raw_indices;
    raw_indices.reserve(encoded.size());
    for (size_t index = 0u; index < encoded.size(); ++index) raw_indices.push_back(index);
    auto emit_raw = [&]() {
        out->clear();
        append_encoded_turns(encoded, raw_indices, out);
        out->insert(out->end(), assistant_prefix.begin(), assistant_prefix.end());
        if (out->empty() || out->size() > context_limit) {
            *error = "prompt exceeds configured context";
            return false;
        }
        return true;
    };

    const uint32_t active_limit = kContextCompactionBoundary >
                    kContextCompactionMaxOutputReserve + kContextCompactionBuffer
            ? kContextCompactionBoundary - kContextCompactionMaxOutputReserve -
                    kContextCompactionBuffer
            : kContextCompactionBoundary;
    if (!state->context_compaction || raw_tokens <= active_limit || encoded.empty()) {
        return emit_raw();
    }

    std::vector<size_t> pinned;
    std::vector<size_t> history;
    uint64_t pinned_tokens = 0u;
    for (size_t index = 0u; index < encoded.size(); ++index) {
        if (encoded[index].turn.role == "system") {
            pinned.push_back(index);
            pinned_tokens += encoded[index].ids.size();
        } else {
            history.push_back(index);
        }
    }
    const uint64_t prefix_tokens = pinned_tokens + assistant_prefix.size();
    /* A single non-system turn is the raw long-document path. The governor
     * is for multi-turn history, where summarising older turns preserves the
     * current request while keeping the logical context ceiling at 1M. */
    if (history.size() < 2u || prefix_tokens >= active_limit) {
        return emit_raw();
    }

    uint32_t available = static_cast<uint32_t>(active_limit - prefix_tokens);
    const uint32_t keep_budget = std::min<uint32_t>(available * 3u / 5u,
                                                    kContextCompactionRecentTokens);
    // Always retain the entire latest user request and its assistant/tool
    // continuation. The recent-history target is a soft budget for OLDER
    // exchanges, not permission to summarize away the current task. Keep
    // complete exchanges so a retained tool result cannot lose its call.
    std::vector<size_t> exchange_starts{0u};
    for (size_t position = 1u; position < history.size(); ++position) {
        const chat_turn &turn = encoded[history[position]].turn;
        if (turn.role == "user" && !turn.tool_response)
            exchange_starts.push_back(position);
    }
    size_t first_exchange = exchange_starts.size() - 1u;
    size_t keep_begin = exchange_starts[first_exchange];
    uint64_t keep_tokens = 0u;
    for (size_t position = keep_begin; position < history.size(); ++position)
        keep_tokens += encoded[history[position]].ids.size();
    while (first_exchange > 0u) {
        const size_t begin = exchange_starts[first_exchange - 1u];
        uint64_t count = 0u;
        for (size_t position = begin; position < keep_begin; ++position)
            count += encoded[history[position]].ids.size();
        if (keep_tokens + count > keep_budget) break;
        keep_tokens += count;
        keep_begin = begin;
        --first_exchange;
    }
    std::vector<size_t> keep_recent(history.begin() + keep_begin, history.end());

    const size_t compact_count = history.size() - keep_recent.size();
    if (compact_count == 0u) {
        return emit_raw();
    }
    std::vector<chat_turn> compacted_turns;
    compacted_turns.reserve(compact_count);
    for (size_t index = 0u; index < compact_count; ++index) {
        compacted_turns.push_back(encoded[history[index]].turn);
    }
    // Codex's append-only prompt catalog may omit tools discovered later in
    // the history. Availability must come from the effective validated request,
    // not from that prefix-preserving projection or old call names.
    bool current_tools_enabled = tools_enabled;
    if (tool_authority) {
        ajson current_tools;
        if (!normalize_tools(tool_authority->get("tools"), anthropic, &current_tools, error))
            return false;
        current_tools_enabled = !current_tools.arr.empty() &&
                tools_are_enabled(tool_authority->get("tool_choice"));
    }
    const std::string summary = "[Contesto precedente compatto]\n" +
            build_compact_context_state(compacted_turns, current_tools_enabled);
    encoded_chat_turn summary_turn;
    if (encode_chat_turn(state, chat_turn{"user", summary}, &summary_turn, error) != AXIOM_OK) {
        return false;
    }
    encoded_chat_turn ack_turn;
    if (encode_chat_turn(
            state, chat_turn{"assistant", std::string(kNoThinkingPrefix) +
                    "Ho caricato il contesto compatto. Continuo dal punto corrente."},
            &ack_turn, error) != AXIOM_OK) {
        return false;
    }

    out->clear();
    append_encoded_turns(encoded, pinned, out);
    out->insert(out->end(), summary_turn.ids.begin(), summary_turn.ids.end());
    out->insert(out->end(), ack_turn.ids.begin(), ack_turn.ids.end());
    append_encoded_turns(encoded, keep_recent, out);
    out->insert(out->end(), assistant_prefix.begin(), assistant_prefix.end());
    while ((out->size() > active_limit || out->size() > context_limit) &&
            first_exchange + 1u < exchange_starts.size()) {
        ++first_exchange;
        keep_recent.assign(history.begin() + exchange_starts[first_exchange], history.end());
        out->clear();
        append_encoded_turns(encoded, pinned, out);
        out->insert(out->end(), summary_turn.ids.begin(), summary_turn.ids.end());
        out->insert(out->end(), ack_turn.ids.begin(), ack_turn.ids.end());
        append_encoded_turns(encoded, keep_recent, out);
        out->insert(out->end(), assistant_prefix.begin(), assistant_prefix.end());
    }
    if (out->size() > active_limit || out->size() > context_limit) {
        return emit_raw();
    }
    *compacted = true;
    return !out->empty();
}

// Reconstruct exactly what the Responses adapter exposes to Core: a native
// assistant turn may emit a text item and multiple separate function_call
// items. Core replays each as its own assistant turn. Transform ONLY the saved
// reference for comparison; all native token IDs/KV/recurrent state stay intact.
// Incoming history is never canonicalized, trimmed, reordered or skipped.
bool project_codex_tool_prefix(server_state *state,
        const axiom::qwen38::qwen38_session_manifest &manifest,
        const ajson &tools, std::vector<uint32_t> *projected) {
    if (!state || !state->tokenizer || !projected || !tools.is_array() ||
        tools.arr.empty() || !manifest.committed_tokens ||
        manifest.committed_tokens > manifest.token_ids.size()) return false;
    projected->clear();
    const auto &ids = manifest.token_ids;
    const size_t committed = manifest.committed_tokens;
    const auto start_id = state->tokenizer_info.im_start_token_id;
    const auto end_id = state->tokenizer_info.im_end_token_id;
    std::vector<uint32_t> newline;
    if (append_text_ids(state, "\n", &newline) != AXIOM_OK || newline.empty()) return false;
    const std::string assistant_header = std::string("assistant\n") + kNoThinkingPrefix;
    std::string decoded(4u << 20, '\0');
    size_t total_bytes = 0;
    bool changed = false;
    for (size_t start = 0; start < committed;) {
        if (ids[start] != start_id) return false;
        size_t end = start + 1;
        while (end < committed && ids[end] != end_id) {
            if (ids[end] == start_id) return false; // Not an actual ChatML turn.
            ++end;
        }
        const bool closed = end < committed;
        size_t next = end + (closed ? 1 : 0);
        bool has_newline = false;
        if (closed && next < committed) {
            if (committed - next < newline.size() ||
                !std::equal(newline.begin(), newline.end(), ids.begin() + next)) return false;
            next += newline.size();
            has_newline = true;
        }
        uint32_t bytes = 0;
        if (axiom_tokenizer_decode_ids(state->tokenizer, ids.data() + start + 1,
                end - start - 1, decoded.data(), decoded.size(), &bytes) != AXIOM_OK)
            return false;
        total_bytes += bytes;
        if (total_bytes > (64u << 20)) return false;
        const std::string body(decoded.data(), bytes);
        const auto retain = [&] {
            projected->insert(projected->end(), ids.begin() + start, ids.begin() + next);
        };
        if (body.compare(0, assistant_header.size(), assistant_header) != 0 ||
            body.find("<tool_call>", assistant_header.size()) == std::string::npos) {
            retain(); start = next; continue;
        }
        // The encoder accepts C strings. Never let a NUL truncate a projected
        // tool argument into an apparently matching shorter wire reference.
        if (body.find('\0') != std::string::npos) return false;
        const auto parsed = parse_native_tool_calls(body.substr(assistant_header.size()), tools);
        if (parsed.status != "pass" || parsed.calls.empty()) return false;
        // These are the same parser/renderer used on the outbound/inbound
        // protocol paths. String parameter bytes remain opaque after that
        // outbound parse; a later incoming edit cannot pass the prefix proof.
        std::vector<std::string> parts;
        if (!parsed.content.empty()) parts.push_back(parsed.content);
        for (const auto &call : parsed.calls) {
            const auto text = native_tool_call_text(call);
            if (text.empty()) return false;
            parts.push_back(text);
        }
        std::vector<uint32_t> projected_turn;
        for (size_t part = 0; part < parts.size(); ++part) {
            projected_turn.push_back(start_id);
            if (append_text_ids(state, assistant_header + parts[part], &projected_turn) != AXIOM_OK)
                return false;
            // A manifest may end BEFORE the final im_end (the predicted EOS
            // is not committed). Leave it in the incoming suffix so the real
            // native state consumes it exactly once on resumption.
            if (part + 1 < parts.size() || closed) projected_turn.push_back(end_id);
            if (part + 1 < parts.size() || has_newline) {
                projected_turn.insert(projected_turn.end(), newline.begin(), newline.end());
            }
        }
        changed |= projected_turn.size() != next - start ||
                !std::equal(projected_turn.begin(), projected_turn.end(), ids.begin() + start);
        projected->insert(projected->end(), projected_turn.begin(), projected_turn.end());
        if (projected->size() > state->max_context) return false;
        start = next;
    }
    return changed;
}

bool vision_part_type(const std::string &type) {
    return type == "image" || type == "image_url" || type == "input_image" ||
            type == "video" || type == "video_url" || type == "input_video";
}

bool json_contains_vision(const ajson &value) {
    if (value.is_array()) {
        for (const ajson &item : value.arr) {
            if (json_contains_vision(item)) return true;
        }
        return false;
    }
    if (!value.is_object()) return false;
    const ajson *type = value.get("type");
    if (type && type->is_string() && vision_part_type(type->s)) return true;
    for (const ajson &item : value.vals) {
        if (json_contains_vision(item)) return true;
    }
    return false;
}

bool payload_contains_vision(const ajson &payload) {
    return json_contains_vision(payload);
}

bool append_prompt_text(
        server_state *state, const std::string &text,
        std::vector<uint32_t> *ids, std::vector<int32_t> *slots,
        std::string *error) {
    if (!state || !ids || !slots || !error) return false;
    if (text.empty()) return true;
    std::vector<uint32_t> encoded;
    const int rc = append_text_ids(state, text, &encoded);
    if (rc != AXIOM_OK) {
        *error = "multimodal prompt exceeds configured context";
        return false;
    }
    try {
        ids->insert(ids->end(), encoded.begin(), encoded.end());
        slots->insert(slots->end(), encoded.size(), -1);
    } catch (...) {
        *error = "multimodal prompt allocation failed";
        return false;
    }
    return true;
}

bool append_prompt_token(
        uint32_t token, int32_t slot, std::vector<uint32_t> *ids,
        std::vector<int32_t> *slots, std::string *error) {
    if (!ids || !slots || !error) return false;
    try {
        ids->push_back(token);
        slots->push_back(slot);
    } catch (...) {
        *error = "multimodal prompt allocation failed";
        return false;
    }
    return true;
}

bool vision_token_id(
        server_state *state, const char *token, uint32_t *out,
        std::string *error) {
    if (!state || !state->tokenizer || !token || !out || !error) return false;
    *out = AXIOM_TOKEN_ID_INVALID;
    const int rc = axiom_tokenizer_token_id(state->tokenizer, token, out);
    if (rc != AXIOM_OK || *out == AXIOM_TOKEN_ID_INVALID) {
        *error = std::string("Qwen vision special token is missing: ") + token;
        return false;
    }
    return true;
}

bool data_url_from_source(
        const ajson &source, std::string *out, std::string *error) {
    if (!out || !error || !source.is_object()) {
        if (error) *error = "media source must be an object";
        return false;
    }
    const ajson *data = source.get("data");
    const ajson *mime = source.get("media_type");
    if (!data || !data->is_string() || data->s.empty() ||
        !mime || !mime->is_string() || mime->s.empty()) {
        *error = "base64 media source requires data and media_type";
        return false;
    }
    *out = "data:" + mime->s + ";base64," + data->s;
    return true;
}

bool media_url_from_part(
        const ajson &part, bool video, std::string *out,
        const ajson **frames, std::string *error) {
    if (!out || !frames || !error || !part.is_object()) return false;
    *out = {};
    *frames = nullptr;
    const ajson *frame_list = part.get("frames");
    if (frame_list && frame_list->is_array()) {
        *frames = frame_list;
        return true;
    }
    const char *primary = video ? "video_url" : "image_url";
    const ajson *value = part.get(primary);
    if (!value) value = part.get("url");
    if (!value && !video) value = part.get("source");
    if (!value && video) value = part.get("source");
    if (value && value->is_string()) {
        *out = value->s;
    } else if (value && value->is_object()) {
        const ajson *url = value->get("url");
        if (!url) url = value->get("data_url");
        if (url && url->is_string()) *out = url->s;
        else if (!video && value->get("data")) {
            return data_url_from_source(*value, out, error);
        } else if (video && value->get("data")) {
            return data_url_from_source(*value, out, error);
        }
    }
    if (out->empty()) {
        *error = video
                ? "video part requires a base64 data URL or frames array"
                : "image part requires a base64 data URL or Anthropic base64 source";
        return false;
    }
    if (out->compare(0u, 5u, "data:") != 0) {
        *error = "remote/file media URLs are disabled; provide a base64 data URL";
        return false;
    }
    return true;
}

bool decode_frame_part(
        const ajson &frame, axiom::qwen38::vision::rgb_image *out,
        std::string *error) {
    if (!out || !error) return false;
    if (frame.is_string()) {
        if (frame.s.compare(0u, 5u, "data:") != 0) {
            *error = "video frame must be a base64 data URL";
            return false;
        }
        return axiom::qwen38::vision::decode_data_url(frame.s, out, error);
    }
    if (!frame.is_object()) {
        *error = "video frames must be image data URLs or image parts";
        return false;
    }
    const ajson *type = frame.get("type");
    const std::string kind = type && type->is_string() ? type->s : "image_url";
    if (kind == "image" && frame.get("source")) {
        std::string url;
        const ajson *ignored = nullptr;
        if (!media_url_from_part(frame, false, &url, &ignored, error)) return false;
        return axiom::qwen38::vision::decode_data_url(url, out, error);
    }
    std::string url;
    const ajson *ignored = nullptr;
    if (!media_url_from_part(frame, false, &url, &ignored, error)) return false;
    return axiom::qwen38::vision::decode_data_url(url, out, error);
}

bool vision_host_allocation_fits(uint64_t bytes) {
    return axiom::vision_memory::host_allocation_fits(bytes);
}

bool append_visual_media(
        server_state *state, const ajson &part, bool video,
        const std::shared_ptr<vision_request> &request,
        std::vector<uint32_t> *ids, std::vector<int32_t> *slots,
        std::string *error) {
    if (!state || !request || !ids || !slots || !error || !state->vision) {
        if (error) *error = "native Qwen vision runtime is not available";
        return false;
    }
    std::string url;
    const ajson *frames_part = nullptr;
    if (!media_url_from_part(part, video, &url, &frames_part, error)) return false;

    axiom::qwen38::vision::preprocessed_media media;
    if (!video) {
        axiom::qwen38::vision::rgb_image image;
        if (!axiom::qwen38::vision::decode_data_url(url, &image, error) ||
            !axiom::qwen38::vision::preprocess_image(
                    image, axiom::qwen38::vision::kImageMinPixels,
                    axiom::qwen38::vision::kImageMaxPixels, &media, error)) {
            return false;
        }
    } else {
        std::vector<axiom::qwen38::vision::rgb_image> frames;
        if (frames_part) {
            if (frames_part->arr.empty()) {
                *error = "video frames array must not be empty";
                return false;
            }
            try {
                frames.reserve(frames_part->arr.size());
                for (const ajson &frame : frames_part->arr) {
                    axiom::qwen38::vision::rgb_image decoded;
                    if (!decode_frame_part(frame, &decoded, error)) return false;
                    frames.push_back(std::move(decoded));
                }
            } catch (...) {
                *error = "video frame allocation exceeded the native budget";
                return false;
            }
        } else {
            axiom::qwen38::vision::video_decode_limits limits;
            limits.max_frames = kVisionMaxFrames;
            limits.max_width = kVisionMaxDecodeWidth;
            limits.max_height = kVisionMaxDecodeHeight;
            limits.max_frame_pixels = kVisionMaxFramePixels;
            if (!axiom::qwen38::vision::decode_video_data_url(url, limits, &frames, error)) {
                return false;
            }
        }
        if (frames.size() > kVisionMaxFrames) {
            *error = "video exceeds the native 64-frame limit";
            return false;
        }
        if (!axiom::qwen38::vision::preprocess_video_frames(
                frames, axiom::qwen38::vision::kVideoMinPixels,
                axiom::qwen38::vision::kVideoMaxPixels, &media, error)) {
            return false;
        }
    }

    const uint64_t old_patch_count = request->patch_values.size() /
            AXIOM_QWEN38_VISION_PATCH_FEATURES;
    const uint64_t new_patch_count = media.patch_count();
    // The resident workspace size is not a combined history/input limit.
    // Preserve every image at the existing native processor resolution.
    if (new_patch_count == 0u || old_patch_count > UINT32_MAX ||
        new_patch_count > static_cast<uint64_t>(UINT32_MAX) - old_patch_count) {
        *error = "visual patch count exceeds the native index range";
        return false;
    }
    const uint64_t merged_tokens = static_cast<uint64_t>(media.grid.temporal) *
            (media.grid.height / AXIOM_QWEN38_VISION_MERGE_SIZE) *
            (media.grid.width / AXIOM_QWEN38_VISION_MERGE_SIZE);
    if (merged_tokens == 0u || merged_tokens > INT32_MAX ||
        request->visual_tokens > static_cast<uint32_t>(INT32_MAX - merged_tokens)) {
        *error = "visual token count exceeds the native prompt index budget";
        return false;
    }
    uint32_t start_id = AXIOM_TOKEN_ID_INVALID;
    uint32_t end_id = AXIOM_TOKEN_ID_INVALID;
    uint32_t pad_id = AXIOM_TOKEN_ID_INVALID;
    if (!vision_token_id(state, "<|vision_start|>", &start_id, error) ||
        !vision_token_id(state, "<|vision_end|>", &end_id, error) ||
        !vision_token_id(state, video ? "<|video_pad|>" : "<|image_pad|>", &pad_id, error)) {
        return false;
    }
    try {
        const uint64_t combined_values = (old_patch_count + new_patch_count) * AXIOM_QWEN38_VISION_PATCH_FEATURES;
        if (combined_values > SIZE_MAX / sizeof(float) ||
            (combined_values > request->patch_values.capacity() &&
             !vision_host_allocation_fits(combined_values * sizeof(float)))) {
            *error = "insufficient available host memory for visual history at native resolution";
            return false;
        }
        request->patch_values.reserve(static_cast<size_t>(combined_values));
        request->patch_values.insert(
                request->patch_values.end(), media.patches.begin(), media.patches.end());
        request->grids.push_back(media.grid);
    } catch (...) {
        *error = "visual patch allocation failed";
        return false;
    }
    if (!append_prompt_token(start_id, -1, ids, slots, error)) return false;
    for (uint64_t index = 0u; index < merged_tokens; ++index) {
        const int32_t slot = static_cast<int32_t>(request->visual_tokens++);
        if (!append_prompt_token(pad_id, slot, ids, slots, error)) return false;
    }
    if (!append_prompt_token(end_id, -1, ids, slots, error)) return false;
    ++request->media_count;
    return true;
}

bool append_multimodal_content(
        server_state *state, const ajson *content,
        const std::shared_ptr<vision_request> &request,
        std::vector<uint32_t> *ids, std::vector<int32_t> *slots,
        std::string *error) {
    if (!content || content->is_null()) return true;
    if (content->is_string()) {
        return append_prompt_text(state, content->s, ids, slots, error);
    }
    if (!content->is_array()) {
        return append_prompt_text(state, message_content_text(content), ids, slots, error);
    }
    for (const ajson &part : content->arr) {
        if (part.is_string()) {
            if (!append_prompt_text(state, part.s, ids, slots, error)) return false;
            continue;
        }
        if (!part.is_object()) {
            *error = "multimodal content parts must be objects or strings";
            return false;
        }
        const ajson *type = part.get("type");
        const std::string kind = type && type->is_string() ? type->s : "text";
        if (kind == "image" || kind == "image_url" || kind == "input_image") {
            if (!append_visual_media(state, part, false, request, ids, slots, error)) return false;
        } else if (kind == "video" || kind == "video_url" || kind == "input_video") {
            if (!append_visual_media(state, part, true, request, ids, slots, error)) return false;
        } else if (kind == "text" || kind == "input_text") {
            const ajson *text = part.get("text");
            if (!text || !text->is_string()) {
                *error = "text content part requires a string text field";
                return false;
            }
            if (!append_prompt_text(state, text->s, ids, slots, error)) return false;
        } else if (kind == "tool_result") {
            if (!append_prompt_text(state, "<tool_response>\n", ids, slots, error) ||
                !append_multimodal_content(
                        state, part.get("content"), request, ids, slots, error) ||
                !append_prompt_text(state, "\n</tool_response>", ids, slots, error)) {
                return false;
            }
        } else {
            const ajson *nested = part.get("content");
            if (nested) {
                if (!append_multimodal_content(state, nested, request, ids, slots, error)) return false;
            } else {
                *error = "unsupported multimodal content part type: " + kind;
                return false;
            }
        }
    }
    return true;
}

bool append_multimodal_turn(
        server_state *state, const std::string &role, const ajson *content,
        const std::shared_ptr<vision_request> &request,
        std::vector<uint32_t> *ids, std::vector<int32_t> *slots,
        std::string *error, const std::string &suffix = {}, const std::string &prefix = {}) {
    if (!state || !ids || !slots || !error) return false;
    if (!append_prompt_token(state->tokenizer_info.im_start_token_id, -1, ids, slots, error) ||
        !append_prompt_text(state, role + "\n", ids, slots, error) ||
        !append_prompt_text(state, prefix, ids, slots, error) ||
        !append_multimodal_content(state, content, request, ids, slots, error) ||
        !append_prompt_text(state, suffix, ids, slots, error) ||
        !append_prompt_token(state->tokenizer_info.im_end_token_id, -1, ids, slots, error) ||
        !append_prompt_text(state, "\n", ids, slots, error)) {
        return false;
    }
    return ids->size() <= effective_context_limit(state);
}

const float *vision_embedding_for(
        const vision_request *request, size_t position) {
    if (!request || !request->embedding_device || position >= request->embedding_slots.size()) {
        return nullptr;
    }
    const int32_t slot = request->embedding_slots[position];
    if (slot < 0 || static_cast<uint32_t>(slot) >= request->visual_tokens) return nullptr;
    return request->embedding_device + static_cast<size_t>(slot) * AXIOM_QWEN38_VISION_OUTPUT;
}

bool finalize_vision_request(
        server_state *state, const std::shared_ptr<vision_request> &request,
        std::string *error) {
    if (!state || !request || !error || request->media_count == 0u ||
        request->patch_values.empty() || request->grids.empty() || request->visual_tokens == 0u) {
        if (error) *error = "multimodal request contains no usable visual tokens";
        return false;
    }
    const uint64_t output_values = static_cast<uint64_t>(request->visual_tokens) *
            AXIOM_QWEN38_VISION_OUTPUT;
    if (output_values > SIZE_MAX / sizeof(float) || !vision_host_allocation_fits(output_values * sizeof(float))) {
        *error = "insufficient available host memory for vision embeddings";
        return false;
    }
    try {
        request->embedding_host.resize(static_cast<size_t>(output_values));
    } catch (...) {
        *error = "vision embedding host allocation exceeded the native budget";
        return false;
    }
    // Vision attention has independent segments for each temporal group. Run
    // complete spatial grids sequentially, never tile/crop or pool separate
    // images. This removes the cumulative GPU scratch requirement while keeping
    // the original embedding/slot order. The tower grows scratch per call using
    // actual available device memory, then releases it before the next segment.
    size_t patch_offset = 0u, output_offset = 0u;
    if (request->patch_values.size() / AXIOM_QWEN38_VISION_PATCH_FEATURES <= state->vision_max_patch_tokens) {
        // Preserve the existing batch shape/numerics for requests fitting the
        // resident workspace. Larger histories are accepted via full segments.
        uint32_t produced = 0u;
        const int rc = axiom_qwen38_vision_forward_patches(state->vision,
                request->patch_values.data(), request->patch_values.size(), request->grids.data(),
                static_cast<uint32_t>(request->grids.size()), request->embedding_host.data(),
                request->embedding_host.size(), &produced);
        if (rc != AXIOM_OK || produced != request->visual_tokens) {
            *error = std::string("native Qwen vision forward failed: ") + axiom_status_string(rc);
            return false;
        }
        patch_offset = request->patch_values.size();
        output_offset = request->embedding_host.size();
    } else {
    for (const auto &grid : request->grids) {
        const axiom_qwen38_vision_grid frame{1u, grid.height, grid.width};
        const uint64_t patch_values = static_cast<uint64_t>(grid.height) * grid.width * AXIOM_QWEN38_VISION_PATCH_FEATURES;
        const uint32_t tokens = (grid.height / AXIOM_QWEN38_VISION_MERGE_SIZE) * (grid.width / AXIOM_QWEN38_VISION_MERGE_SIZE);
        const uint64_t values = static_cast<uint64_t>(tokens) * AXIOM_QWEN38_VISION_OUTPUT;
        for (uint32_t t = 0u; t < grid.temporal; ++t) {
            if (patch_offset > request->patch_values.size() || patch_values > request->patch_values.size() - patch_offset ||
                output_offset > request->embedding_host.size() || values > request->embedding_host.size() - output_offset) {
                *error = "invalid visual segment layout";
                return false;
            }
            uint32_t produced = 0u;
            const int rc = axiom_qwen38_vision_forward_patches(state->vision,
                    request->patch_values.data() + patch_offset, patch_values, &frame, 1u,
                    request->embedding_host.data() + output_offset, values, &produced);
            if (rc != AXIOM_OK || produced != tokens) {
                *error = rc == AXIOM_ERR_BUDGET
                        ? "insufficient available GPU memory for this image at native resolution"
                        : std::string("native Qwen vision forward failed: ") + axiom_status_string(rc);
                return false;
            }
            patch_offset += patch_values;
            output_offset += values;
        }
    }
    }
    if (patch_offset != request->patch_values.size() || output_offset != request->embedding_host.size()) {
        *error = "visual segment output count mismatch";
        return false;
    }
    const cudaError_t alloc = cudaMalloc(
            reinterpret_cast<void **>(&request->embedding_device),
            output_values * sizeof(float));
    if (alloc != cudaSuccess) {
        *error = "vision embedding CUDA allocation failed";
        return false;
    }
    const cudaError_t copy = cudaMemcpy(
            request->embedding_device, request->embedding_host.data(),
            output_values * sizeof(float), cudaMemcpyHostToDevice);
    if (copy != cudaSuccess) {
        *error = "vision embedding CUDA upload failed";
        (void)cudaFree(request->embedding_device);
        request->embedding_device = nullptr;
        return false;
    }
    return true;
}

bool encode_multimodal_chat_ids(
        server_state *state, const ajson &payload, bool anthropic,
        std::vector<uint32_t> *out, std::shared_ptr<vision_request> *out_request,
        std::string *error, uint32_t *original_tokens, bool *compacted) {
    if (!state || !out || !out_request || !error || !original_tokens || !compacted) return false;
    const uint32_t context_limit = effective_context_limit(state);
    *out_request = nullptr;
    *original_tokens = 0u;
    *compacted = false;
    const ajson *messages = payload.get("messages");
    if (!messages || !messages->is_array() || messages->arr.empty()) {
        *error = "messages must be a non-empty array";
        return false;
    }
    if (state->tokenizer_info.im_start_token_id == AXIOM_TOKEN_ID_INVALID ||
        state->tokenizer_info.im_end_token_id == AXIOM_TOKEN_ID_INVALID) {
        *error = "Qwen ChatML tokens are missing from the tokenizer";
        return false;
    }
    auto request = std::make_shared<vision_request>();
    const bool no_think = resolve_no_think(state, payload, error);
    if (!error->empty() && !no_think) return false;
    ajson normalized_tools;
    if (!normalize_tools(payload.get("tools"), anthropic, &normalized_tools, error)) return false;
    size_t first_message = 0u;
    const ajson system = initial_chat_system(state, payload, *messages,
            normalized_tools, no_think, &first_message);
    out->clear();
    if (!system.arr.empty()) {
        if (!append_multimodal_turn(state, "system", &system, request, out,
                                    &request->embedding_slots, error)) return false;
    }
    bool previous_tool = false;
    for (size_t message_index = first_message; message_index < messages->arr.size(); ++message_index) {
        const ajson &message = messages->arr[message_index];
        const ajson *role = message.get("role");
        const ajson *content = message.get("content");
        if (!message.is_object() || !role || !role->is_string() ||
            (!content && role->s != "assistant") ||
            (role->s != "system" && role->s != "user" && role->s != "assistant" &&
             role->s != "tool")) {
            *error = "each message must have role system/user/assistant/tool and content";
            return false;
        }
        std::string tool_call_text;
        if (role->s == "assistant") {
            const ajson *tool_calls = message.get("tool_calls");
            if (tool_calls && tool_calls->is_array()) {
                for (const ajson &call : tool_calls->arr) {
                    const std::string current = native_tool_call_text(call);
                    if (!current.empty()) {
                        if (!tool_call_text.empty()) tool_call_text += "\n";
                        tool_call_text += current;
                    }
                }
            } else {
                tool_call_text = native_tool_calls_from_content(content);
            }
        }
        if (role->s == "tool") {
            if (!previous_tool &&
                (!append_prompt_token(state->tokenizer_info.im_start_token_id, -1, out,
                                      &request->embedding_slots, error) ||
                 !append_prompt_text(state, "user", out, &request->embedding_slots, error))) return false;
            if (!append_prompt_text(state, "\n<tool_response>\n", out,
                                    &request->embedding_slots, error) ||
                !append_multimodal_content(state, content, request, out,
                                            &request->embedding_slots, error) ||
                !append_prompt_text(state, "\n</tool_response>", out,
                                    &request->embedding_slots, error)) {
                return false;
            }
            const ajson *next_role = message_index + 1u < messages->arr.size()
                    ? messages->arr[message_index + 1u].get("role") : nullptr;
            if (!next_role || !next_role->is_string() || next_role->s != "tool") {
                if (!append_prompt_token(state->tokenizer_info.im_end_token_id, -1, out,
                                         &request->embedding_slots, error) ||
                    !append_prompt_text(state, "\n", out, &request->embedding_slots, error)) return false;
            }
        } else if (!append_multimodal_turn(
                state, role->s, content, request, out,
                &request->embedding_slots, error,
                tool_call_text.empty() ? std::string{} :
                    (message_content_text(content).empty() ? "" : "\n\n") + tool_call_text,
                role->s == "assistant" ? assistant_history_prefix(message) : std::string{})) {
            return false;
        }
        previous_tool = role->s == "tool";
    }
    std::vector<uint32_t> assistant_prefix;
    if (encode_assistant_prefix(state, no_think, &assistant_prefix, error) != AXIOM_OK) return false;
    try {
        out->insert(out->end(), assistant_prefix.begin(), assistant_prefix.end());
        request->embedding_slots.insert(request->embedding_slots.end(), assistant_prefix.size(), -1);
    } catch (...) {
        *error = "multimodal prompt allocation failed";
        return false;
    }
    if (out->size() > context_limit) {
        *error = "multimodal prompt exceeds configured context";
        return false;
    }
    *original_tokens = static_cast<uint32_t>(out->size());
    *out_request = std::move(request);
    return !out->empty();
}

bool make_multimodal_chat_ids(
        server_state *state, const ajson &payload, bool anthropic,
        std::vector<uint32_t> *out, std::shared_ptr<vision_request> *out_request,
        std::string *error, uint32_t *original_tokens, bool *compacted) {
    if (!encode_multimodal_chat_ids(state, payload, anthropic, out, out_request,
            error, original_tokens, compacted)) return false;
    if (!finalize_vision_request(state, *out_request, error)) {
        *out_request = nullptr;
        return false;
    }
    return true;
}

bool make_completion_ids(server_state *state, const ajson &payload, std::vector<uint32_t> *out, std::string *error) {
    if (!state || !out || !error) return false;
    const ajson *prompt = payload.get("prompt");
    if (!prompt || !prompt->is_string()) {
        *error = "prompt must be a string";
        return false;
    }
    out->clear();
    const int rc = append_text_ids(state, prompt->s, out);
    if (rc != AXIOM_OK || out->empty()) {
        *error = "prompt exceeds configured context";
        return false;
    }
    return true;
}

bool stop_token(uint32_t token, const axiom_tokenizer_info &info) {
    return token == info.endoftext_token_id || token == info.im_end_token_id;
}

enum class utf8_unit_status {
    complete,
    incomplete,
    invalid,
};

utf8_unit_status inspect_utf8_unit(
        const std::string &bytes, const size_t offset, size_t *unit_bytes) {
    if (!unit_bytes || offset >= bytes.size()) return utf8_unit_status::invalid;
    *unit_bytes = 1u;
    const auto byte = [&](const size_t index) {
        return static_cast<uint8_t>(bytes[index]);
    };
    const uint8_t first = byte(offset);
    if (first <= 0x7fu) {
        *unit_bytes = 1u;
        return utf8_unit_status::complete;
    }
    size_t width = 0u;
    if (first >= 0xc2u && first <= 0xdfu) width = 2u;
    else if (first >= 0xe0u && first <= 0xefu) width = 3u;
    else if (first >= 0xf0u && first <= 0xf4u) width = 4u;
    else return utf8_unit_status::invalid;
    const size_t available = bytes.size() - offset;
    const size_t inspect_bytes = std::min(width, available);
    for (size_t index = 1u; index < inspect_bytes; ++index) {
        if ((byte(offset + index) & 0xc0u) != 0x80u) {
            return utf8_unit_status::invalid;
        }
    }
    if (inspect_bytes >= 2u) {
        const uint8_t second = byte(offset + 1u);
        if ((first == 0xe0u && second < 0xa0u) ||
            (first == 0xedu && second > 0x9fu) ||
            (first == 0xf0u && second < 0x90u) ||
            (first == 0xf4u && second > 0x8fu)) {
            return utf8_unit_status::invalid;
        }
    }
    /* An available non-continuation byte proves the leading byte invalid even
     * when the nominal unit width has not arrived yet. Only a suffix whose
     * bytes are all valid prefixes may remain pending across stream chunks. */
    if (available < width) return utf8_unit_status::incomplete;
    *unit_bytes = width;
    return utf8_unit_status::complete;
}

int normalize_final_utf8(
        std::string *text, std::string *failure_stage) {
    if (!text || !failure_stage) return AXIOM_ERR_INVALID_ARGUMENT;
    std::string normalized;
    try {
        normalized.reserve(text->size() + 3u);
        size_t consumed = 0u;
        while (consumed < text->size()) {
            size_t unit_bytes = 0u;
            const utf8_unit_status status = inspect_utf8_unit(
                    *text, consumed, &unit_bytes);
            if (status == utf8_unit_status::incomplete) {
                /* Match drain_generation_utf8(final=true): one replacement
                 * character represents the entire truncated final suffix. */
                normalized.append("\xef\xbf\xbd", 3u);
                consumed = text->size();
                break;
            }
            if (status == utf8_unit_status::invalid) {
                normalized.append("\xef\xbf\xbd", 3u);
                ++consumed;
                continue;
            }
            normalized.append(*text, consumed, unit_bytes);
            consumed += unit_bytes;
        }
    } catch (...) {
        *failure_stage = "tokenizer_decode_utf8_alloc";
        return AXIOM_ERR_BUDGET;
    }
    text->swap(normalized);
    return AXIOM_OK;
}

int drain_generation_utf8(
        generation_stream_sink *sink, const bool final,
        std::string *failure_stage) {
    if (!sink || !sink->emit || !failure_stage) return AXIOM_OK;
    if (sink->reasoning_separator_pending) {
        if (!final && (sink->utf8_pending.empty() || sink->utf8_pending == "\n")) return AXIOM_OK;
        if (sink->utf8_pending.compare(0u, 2u, "\n\n") == 0) sink->utf8_pending.erase(0u, 2u);
        sink->reasoning_separator_pending = false;
    }
    std::string ready;
    size_t consumed = 0u;
    try {
        ready.reserve(sink->utf8_pending.size() + 3u);
        while (consumed < sink->utf8_pending.size()) {
            size_t unit_bytes = 0u;
            const utf8_unit_status status = inspect_utf8_unit(
                    sink->utf8_pending, consumed, &unit_bytes);
            if (status == utf8_unit_status::incomplete) break;
            if (status == utf8_unit_status::invalid) {
                ready.append("\xef\xbf\xbd", 3u);
                ++consumed;
                continue;
            }
            ready.append(sink->utf8_pending, consumed, unit_bytes);
            consumed += unit_bytes;
        }
        if (final && consumed < sink->utf8_pending.size()) {
            /* A length stop may bisect a byte-fallback code point. The wire
             * must remain valid JSON/UTF-8, so terminate that suffix with one
             * replacement character instead of leaking malformed bytes. */
            ready.append("\xef\xbf\xbd", 3u);
            consumed = sink->utf8_pending.size();
        }
        sink->utf8_pending.erase(0u, consumed);
    } catch (...) {
        *failure_stage = "tokenizer_stream_utf8_alloc";
        return AXIOM_ERR_BUDGET;
    }
    if (!ready.empty() && !sink->emit(sink->user, ready)) {
        *failure_stage = "stream_write";
        return AXIOM_ERR_IO;
    }
    return AXIOM_OK;
}

int decode_generated_ids(
        server_state *state, const std::vector<uint32_t> &ids,
        std::string *text, std::string *failure_stage) {
    if (!state || !state->tokenizer || !text || !failure_stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    text->clear();
    if (ids.empty()) return AXIOM_OK;
    std::vector<char> piece;
    try {
        piece.assign(65536u, '\0');
        text->reserve(ids.size() * 4u);
    } catch (...) {
        *failure_stage = "tokenizer_decode_alloc";
        return AXIOM_ERR_BUDGET;
    }
    for (const uint32_t token : ids) {
        uint32_t piece_bytes = 0u;
        const int rc = axiom_tokenizer_decode_token(
                state->tokenizer, token, piece.data(),
                static_cast<uint32_t>(piece.size()), &piece_bytes);
        if (rc != AXIOM_OK) {
            *failure_stage = "tokenizer_decode";
            return rc;
        }
        try {
            text->append(piece.data(), static_cast<size_t>(piece_bytes));
        } catch (...) {
            *failure_stage = "tokenizer_decode_alloc";
            return AXIOM_ERR_BUDGET;
        }
    }
    return normalize_final_utf8(text, failure_stage);
}

bool utf8_self_test_emit(void *user, const std::string &piece) {
    if (!user) return false;
    try {
        static_cast<std::string *>(user)->append(piece);
    } catch (...) {
        return false;
    }
    return true;
}

int run_utf8_normalization_self_test() {
    struct test_case {
        const char *name;
        std::string input;
        std::string expected;
    };
    const std::string replacement("\xef\xbf\xbd", 3u);
    const std::vector<test_case> cases = {
        {"valid", std::string(u8"ASCII € 😀 漢字"),
                  std::string(u8"ASCII € 😀 漢字")},
        {"truncated_three_byte", std::string("A\xe2\x82", 3u),
                                  std::string("A") + replacement},
        {"truncated_four_byte", std::string("\xf0\x9f\x98", 3u), replacement},
        {"invalid_bytes", std::string("\xff\x80" "B", 3u),
                          replacement + replacement + "B"},
        {"overlong", std::string("\xc0\xaf", 2u),
                     replacement + replacement},
        {"three_byte_lead_then_ascii", std::string("\xe2" "A", 2u),
                                         replacement + "A"},
        {"four_byte_lead_then_ascii", std::string("\xf0" "A", 2u),
                                        replacement + "A"},
        {"late_invalid_continuation", std::string("\xe2\x82" "A", 3u),
                                        replacement + replacement + "A"},
    };
    for (const test_case &entry : cases) {
        std::string normalized = entry.input;
        std::string stage;
        const int rc = normalize_final_utf8(&normalized, &stage);
        if (rc != AXIOM_OK || normalized != entry.expected) {
            std::fprintf(stderr,
                         "axiom-qwen38-api: UTF-8 self-test FAIL case=%s rc=%d stage=%s\n",
                         entry.name, rc, stage.c_str());
            return 1;
        }
    }

    std::string streamed;
    generation_stream_sink sink;
    sink.user = &streamed;
    sink.emit = &utf8_self_test_emit;
    std::string stage;
    sink.utf8_pending.assign("\xe2", 1u);
    if (drain_generation_utf8(&sink, false, &stage) != AXIOM_OK ||
        !streamed.empty() || sink.utf8_pending.size() != 1u) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: UTF-8 self-test FAIL case=stream_hold stage=%s\n",
                     stage.c_str());
        return 1;
    }
    sink.utf8_pending.append("\x82\xac", 2u);
    if (drain_generation_utf8(&sink, false, &stage) != AXIOM_OK ||
        streamed != std::string(u8"€") || !sink.utf8_pending.empty()) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: UTF-8 self-test FAIL case=stream_complete stage=%s\n",
                     stage.c_str());
        return 1;
    }
    sink.utf8_pending.assign("\xf0\x9f", 2u);
    if (drain_generation_utf8(&sink, true, &stage) != AXIOM_OK ||
        streamed != std::string(u8"€") + replacement ||
        !sink.utf8_pending.empty()) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: UTF-8 self-test FAIL case=stream_truncated stage=%s\n",
                     stage.c_str());
        return 1;
    }

    /* Final and streamed normalization must agree for every split point. This
     * includes malformed sequences where a later ASCII byte proves that an
     * otherwise incomplete leading sequence is already invalid. */
    for (const test_case &entry : cases) {
        for (size_t split = 0u; split <= entry.input.size(); ++split) {
            streamed.clear();
            sink.utf8_pending.clear();
            stage.clear();
            sink.utf8_pending.append(entry.input, 0u, split);
            if (drain_generation_utf8(&sink, false, &stage) != AXIOM_OK) {
                std::fprintf(stderr,
                             "axiom-qwen38-api: UTF-8 self-test FAIL case=%s split=%zu stage=%s\n",
                             entry.name, split, stage.c_str());
                return 1;
            }
            sink.utf8_pending.append(entry.input, split, std::string::npos);
            if (drain_generation_utf8(&sink, false, &stage) != AXIOM_OK ||
                drain_generation_utf8(&sink, true, &stage) != AXIOM_OK ||
                streamed != entry.expected || !sink.utf8_pending.empty()) {
                std::fprintf(stderr,
                             "axiom-qwen38-api: UTF-8 self-test FAIL case=%s split=%zu stream-final stage=%s\n",
                             entry.name, split, stage.c_str());
                return 1;
            }
        }
    }
    struct thinking_case {
        const char *input;
        const char *reasoning;
        const char *visible;
    };
    constexpr thinking_case thinking_cases[] = {
        {"hidden work</think>visible answer", "hidden work", "visible answer"},
        {"<think>hidden work</think>\n\nvisible answer", "hidden work", "visible answer"},
        {"hidden</think>\n\n  answer  \n", "hidden", "  answer  \n"},
        {"hidden</think>\n\nLiteral </think> stays", "hidden", "Literal </think> stays"},
        {"plain answer", "", "plain answer"},
    };
    for (const thinking_case &entry : thinking_cases) {
        if (reasoning_model_text(entry.input) != entry.reasoning ||
            visible_model_text(entry.input) != entry.visible) {
            std::fprintf(stderr,
                         "axiom-qwen38-api: UTF-8 self-test FAIL case=reasoning-split\n");
            return 1;
        }
    }
    std::printf("axiom-qwen38-api: UTF-8 self-test PASS cases=%zu stream=pass reasoning=pass\n",
                cases.size());
    return 0;
}

int run_speculative_routing_self_test() {
    struct mode_case {
        const char *value;
        speculative_mode expected;
    };
    constexpr mode_case cases[] = {
        {"auto", speculative_mode::automatic},
        {"dspark", speculative_mode::dspark},
        {"native_mtp", speculative_mode::native_mtp},
        {"target", speculative_mode::target},
    };
    std::string error;
    speculative_mode parsed = speculative_mode::target;
    ajson empty = ajson::jobj();
    if (!parse_speculative_mode(empty, &parsed, &error) ||
        parsed != speculative_mode::automatic || !error.empty()) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative routing self-test FAIL default\n");
        return 1;
    }
    for (const mode_case &entry : cases) {
        ajson payload = ajson::jobj();
        payload.set("axiom_speculative_mode", ajson::jstr(entry.value));
        error.clear();
        if (!parse_speculative_mode(payload, &parsed, &error) ||
            parsed != entry.expected || !error.empty()) {
            std::fprintf(stderr,
                         "axiom-qwen38-api: speculative routing self-test FAIL mode=%s\n",
                         entry.value);
            return 1;
        }
    }
    ajson conflict = ajson::jobj();
    conflict.set("axiom_speculative_mode", ajson::jstr("dspark"));
    conflict.set("speculative_mode", ajson::jstr("target"));
    error.clear();
    if (parse_speculative_mode(conflict, &parsed, &error) || error.empty()) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative routing self-test FAIL conflict\n");
        return 1;
    }
    ajson responses = ajson::jobj();
    responses.set("model", ajson::jstr(kModelId));
    responses.set("input", ajson::jstr("routing probe"));
    responses.set("axiom_speculative_mode", ajson::jstr("target"));
    ajson normalized;
    error.clear();
    if (!normalize_responses_request(responses, &normalized, &error) ||
        !parse_speculative_mode(normalized, &parsed, &error) ||
        parsed != speculative_mode::target) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative routing self-test FAIL responses\n");
        return 1;
    }
    ajson unknown = ajson::jobj();
    unknown.set("axiom_speculative_mode", ajson::jstr("mtp"));
    error.clear();
    if (parse_speculative_mode(unknown, &parsed, &error) || error.empty()) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative routing self-test FAIL unknown\n");
        return 1;
    }
    uint32_t max_commit = 0u;
    ajson diagnostic = ajson::jobj();
    diagnostic.set("axiom_speculative_max_commit_tokens", ajson::jint(1));
    error.clear();
    if (parse_speculative_max_commit_tokens(
                diagnostic, speculative_mode::automatic, &max_commit, &error) ||
        error.empty()) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative routing self-test FAIL diagnostic-auto\n");
        return 1;
    }
    error.clear();
    if (!parse_speculative_max_commit_tokens(
                diagnostic, speculative_mode::dspark, &max_commit, &error) ||
        max_commit != 1u || !error.empty()) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative routing self-test FAIL diagnostic-dspark\n");
        return 1;
    }
    diagnostic.set("axiom_speculative_max_commit_tokens", ajson::jint(9));
    error.clear();
    if (parse_speculative_max_commit_tokens(
                diagnostic, speculative_mode::dspark, &max_commit, &error) ||
        error.empty()) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative routing self-test FAIL diagnostic-range\n");
        return 1;
    }
    ajson responses_diagnostic = ajson::jobj();
    responses_diagnostic.set("model", ajson::jstr(kModelId));
    responses_diagnostic.set("input", ajson::jstr("diagnostic routing probe"));
    responses_diagnostic.set("axiom_speculative_mode", ajson::jstr("dspark"));
    responses_diagnostic.set("axiom_speculative_max_commit_tokens", ajson::jint(1));
    error.clear();
    if (!normalize_responses_request(responses_diagnostic, &normalized, &error) ||
        !parse_speculative_mode(normalized, &parsed, &error) ||
        !parse_speculative_max_commit_tokens(normalized, parsed, &max_commit, &error) ||
        parsed != speculative_mode::dspark || max_commit != 1u) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative routing self-test FAIL diagnostic-responses\n");
        return 1;
    }
    std::printf(
            "axiom-qwen38-api: speculative routing self-test PASS modes=4 diagnostic_commit=pass fail_closed=pass\n");
    return 0;
}

int emit_generated_token_piece(
        server_state *state,
        uint32_t token,
        generation_stream_sink *sink,
        std::string *failure_stage) {
    if (!state || !failure_stage) return AXIOM_ERR_INVALID_ARGUMENT;
    if (generation_cancelled(sink)) {
        *failure_stage = "client_disconnect";
        return AXIOM_ERR_IO;
    }
    if (!sink || !sink->emit) return AXIOM_OK;

    char piece[65536]{};
    uint32_t piece_bytes = 0u;
    const int rc = axiom_tokenizer_decode_token(
            state->tokenizer, token, piece, static_cast<uint32_t>(sizeof(piece)), &piece_bytes);
    if (rc != AXIOM_OK) {
        *failure_stage = "tokenizer_stream_decode";
        return rc;
    }
    if (piece_bytes != 0u) {
        try {
            sink->utf8_pending.append(piece, static_cast<size_t>(piece_bytes));
        } catch (...) {
            *failure_stage = "tokenizer_stream_utf8_alloc";
            return AXIOM_ERR_BUDGET;
        }
    }
    return drain_generation_utf8(sink, false, failure_stage);
}

int append_generated_token(
        server_state *state, generation_result *out, uint32_t token,
        generation_stream_sink *sink, std::string *failure_stage) {
    if (!state || !out || !failure_stage) return AXIOM_ERR_INVALID_ARGUMENT;
    if (generation_cancelled(sink)) {
        *failure_stage = "client_disconnect";
        return AXIOM_ERR_IO;
    }
    out->ids.push_back(token);
    if (sink) sink->output_budget.observe(out->ids);
    if (sink && sink->progress) sink->progress->decode(static_cast<uint32_t>(out->ids.size()));
    return emit_generated_token_piece(state, token, sink, failure_stage);
}

int emit_thinking_visible_prefix(
        server_state *state, const generation_result &thinking,
        uint32_t visible_tokens, generation_stream_sink *sink,
        std::string *failure_stage) {
    if (visible_tokens > thinking.ids.size()) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!sink || !sink->emit) return AXIOM_OK;
    // The first pass can already contain the start of the visible answer.
    // Publish that prefix BEFORE the second pass streams its continuation.
    // Decode original token pieces, not trimmed/finalized text: whitespace,
    // partial UTF-8 and split tool markers must carry across the boundary.
    // These tokens are already committed/accounted; do not append them again.
    for (size_t index = thinking.ids.size() - visible_tokens;
         index < thinking.ids.size(); ++index) {
        const int rc = emit_generated_token_piece(
                state, thinking.ids[index], sink, failure_stage);
        if (rc != AXIOM_OK) return rc;
    }
    return AXIOM_OK;
}

uint64_t next_sampling_random(uint64_t *state) {
    if (!state) return 0u;
    uint64_t value = *state;
    value ^= value << 13u;
    value ^= value >> 7u;
    value ^= value << 17u;
    *state = value == 0u ? 0x6a09e667f3bcc909ull : value;
    return *state;
}

bool sample_qwen_logits(
        const float *logits, sampling_params &params,
        uint32_t *out_token, float *out_logit) {
    if (!logits || !out_token || !out_logit || params.temperature <= 0.0f ||
        params.top_k == 0u || params.top_k > 256u || params.top_p <= 0.0f ||
        params.top_p > 1.0f) return false;
    constexpr uint32_t kMaxCandidates = 256u;
    const uint32_t candidate_count = std::min<uint32_t>(params.top_k, kMaxCandidates);
    uint32_t ids[kMaxCandidates]{};
    float values[kMaxCandidates]{};
    float maximum = -std::numeric_limits<float>::infinity();
    for (uint32_t token = 0u; token < AXIOM_QWEN38_MODEL_VOCAB; ++token) {
        const float value = logits[token];
        if (!std::isfinite(value)) return false;
        if (value > maximum) maximum = value;
        if (token < candidate_count || value > values[candidate_count - 1u]) {
            uint32_t length = token < candidate_count ? token + 1u : candidate_count;
            if (length > candidate_count) length = candidate_count;
            uint32_t position = length - 1u;
            while (position > 0u && values[position - 1u] < value) {
                values[position] = values[position - 1u];
                ids[position] = ids[position - 1u];
                --position;
            }
            values[position] = value;
            ids[position] = token;
        }
    }
    double probabilities[kMaxCandidates]{};
    double sum = 0.0;
    for (uint32_t index = 0u; index < candidate_count; ++index) {
        probabilities[index] = std::exp(
                static_cast<double>(values[index] - maximum) /
                static_cast<double>(params.temperature));
        sum += probabilities[index];
    }
    if (!std::isfinite(sum) || sum <= 0.0) return false;
    uint32_t nucleus_count = candidate_count;
    if (params.top_p < 1.0f) {
        double cumulative = 0.0;
        for (uint32_t index = 0u; index < candidate_count; ++index) {
            cumulative += probabilities[index] / sum;
            if (cumulative >= params.top_p) {
                nucleus_count = index + 1u;
                break;
            }
        }
    }
    double nucleus_sum = 0.0;
    for (uint32_t index = 0u; index < nucleus_count; ++index) {
        nucleus_sum += probabilities[index];
    }
    if (!std::isfinite(nucleus_sum) || nucleus_sum <= 0.0) return false;
    const uint64_t random_bits = next_sampling_random(&params.rng);
    const double unit = static_cast<double>(random_bits >> 11u) /
            9007199254740992.0;
    const double target = unit * nucleus_sum;
    double accumulated = 0.0;
    uint32_t selected = nucleus_count - 1u;
    for (uint32_t index = 0u; index < nucleus_count; ++index) {
        accumulated += probabilities[index];
        if (target <= accumulated) {
            selected = index;
            break;
        }
    }
    *out_token = ids[selected];
    *out_logit = values[selected];
    return true;
}

int forward_next_token(
        server_state *state, uint32_t input_token, sampling_params *sampling,
        std::vector<float> *logits, uint32_t *out_token, float *out_logit,
        std::string *failure_stage) {
    if (!state || !out_token || !out_logit || !failure_stage) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!sampling || !sampling->enabled) {
        return axiom_qwen38_model_forward_token(
                state->model, input_token, out_token, out_logit);
    }
    if (!logits) return AXIOM_ERR_INVALID_ARGUMENT;
    if (logits->size() != AXIOM_QWEN38_MODEL_VOCAB) {
        try {
            logits->resize(AXIOM_QWEN38_MODEL_VOCAB);
        } catch (...) {
            *failure_stage = "sampling_logits_alloc";
            return AXIOM_ERR_BUDGET;
        }
    }
    uint32_t greedy_token = 0u;
    float greedy_logit = 0.0f;
    const int rc = axiom_qwen38_model_forward_token_logits(
            state->model, input_token, logits->data(),
            static_cast<uint32_t>(logits->size()), &greedy_token, &greedy_logit);
    if (rc != AXIOM_OK) {
        *failure_stage = "native_sampling_forward";
        return rc;
    }
    if (!sample_qwen_logits(logits->data(), *sampling, out_token, out_logit)) {
        *failure_stage = "native_sampling";
        return AXIOM_ERR_RUNTIME;
    }
    (void)greedy_token;
    (void)greedy_logit;
    return AXIOM_OK;
}

bool codex_target_prefill_block_fits(
        size_t position, size_t prompt_tokens, size_t hot_tokens, bool sampling) {
    constexpr size_t width = AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
    if (position > prompt_tokens || width > prompt_tokens - position ||
        position > hot_tokens || width > hot_tokens - position) return false;
    return !sampling || position + width != prompt_tokens;
}

int native_generate_streaming(
        server_state *state,
        const std::vector<uint32_t> &prompt_ids,
        const uint32_t max_new,
        const std::string &session_id,
        const std::string &profile,
        const std::vector<uint32_t> *session_suffix_ids,
        const bool allow_stateful_resume,
        std::vector<uint32_t> *effective_prompt_out,
        const std::shared_ptr<vision_request> &vision_owner,
        generation_result *out,
        std::string *failure_stage,
        sampling_params *sampling,
        generation_stream_sink *sink,
        const char *decode_path,
        const char *fallback_reason,
        const speculative_mode requested_mode) {
    const auto generation_begin = std::chrono::steady_clock::now();
    const uint32_t context_limit = effective_context_limit(state);
    if (!state || !state->model || !state->tokenizer || !out || !failure_stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = generation_result{};
    out->decode_path = decode_path ? decode_path : "scalar_context";
    out->fallback_reason = fallback_reason ? fallback_reason : "";
    out->speculative_mode_requested = speculative_mode_name(requested_mode);
    out->speculative_mode_effective = "target";
    out->speculative_context_tokens = state->speculative_context_tokens;
    failure_stage->clear();
    const vision_request *vision = vision_owner.get();
    if (prompt_ids.empty() || prompt_ids.size() > context_limit || max_new == 0u ||
        static_cast<uint64_t>(prompt_ids.size()) + max_new - 1u > context_limit) {
        *failure_stage = "context_budget";
        return AXIOM_ERR_BUDGET;
    }
    const bool ephemeral_native_vision =
            state->engine == speculative_engine::native_mtp && vision_owner &&
            session_persistence_enabled(state);

    const auto acquire_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock = acquire_generation_session_lock(state);
    out->session_acquire_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - acquire_begin).count();
    if (!state->healthy.load()) {
        *failure_stage = "target_unhealthy";
        return AXIOM_ERR_RUNTIME;
    }
    const auto prefill_begin = std::chrono::steady_clock::now();
    out->prompt_tokens = static_cast<uint32_t>(prompt_ids.size());
    int rc = AXIOM_OK;
    session_activation activation;
    const bool persistent_session =
            session_persistence_enabled(state) && !ephemeral_native_vision;
    if (ephemeral_native_vision) {
        /* Native MTP has no projected-vision embedding input. Keep vision
         * fully functional on the target decoder, but isolate it in one fixed
         * throw-away tier so it cannot overwrite or impersonate a persistent
         * native-MTP namespace. Clients may resend multimodal history; no
         * false KV-resume claim is made. */
        *failure_stage = "target_ephemeral_activate";
        rc = activate_ephemeral_target_tier(state, failure_stage);
    } else if (persistent_session) {
        *failure_stage = "session_activate";
        const uint32_t resume_prompt_limit = context_limit - (max_new - 1u);
        rc = activate_session(
                state, session_id, profile, prompt_ids, session_suffix_ids,
                true, !vision_owner, allow_stateful_resume, resume_prompt_limit,
                &activation, failure_stage, sink ? sink->codex_replay_tools : nullptr);
    } else {
        *failure_stage = "target_reset";
        if (state->kv_tier) rc = axiom_qwen38_kv_tier_reset(state->kv_tier);
        if (rc == AXIOM_OK) rc = axiom_qwen38_model_reset(state->model);
    }

    const std::vector<uint32_t> &model_prompt_ids =
            activation.effective_prompt_ids.empty()
            ? prompt_ids : activation.effective_prompt_ids;
    if (rc == AXIOM_OK && effective_prompt_out) {
        try {
            *effective_prompt_out = model_prompt_ids;
        } catch (...) {
            *failure_stage = "session_effective_prompt_alloc";
            rc = AXIOM_ERR_BUDGET;
        }
    }
    out->session_restored = persistent_session && activation.restored;
    out->session_stateful_resume = persistent_session && activation.stateful_resume;
    out->session_prefix_retokenized = persistent_session && activation.prefix_retokenized;
    out->session_tool_projection = persistent_session && activation.tool_projection;

    uint32_t anchor = 0u;
    float anchor_logit = 0.0f;
    const size_t prefill_start = persistent_session ? activation.start_position : 0u;
    out->prefix_hit_tokens = out->session_restored
            ? static_cast<uint32_t>(prefill_start) : 0u;
    out->suffix_prefill_tokens = static_cast<uint32_t>(
            model_prompt_ids.size() - std::min(prefill_start, model_prompt_ids.size()));
    const bool session_hit = persistent_session && activation.restored;
    if (rc == AXIOM_OK && session_hit && prefill_start == model_prompt_ids.size()) {
        anchor = activation.next_token;
    }
    std::vector<float> sampling_logits;
    if (rc == AXIOM_OK) {
        *failure_stage = "native_prefill";
        // Consume known prompt tokens, never draft tokens. The target M8
        // transaction is qualified only inside the existing hot window;
        // beyond it keep the paged scalar target path unchanged.
        axiom_qwen38_model_dspark_temporal_capabilities prefill_caps{};
        const bool target_m8_prefill = sink && sink->codex_target_prefill && !vision &&
                env_enabled("AXIOM_CODEX_TARGET_M8_PREFILL") &&
                axiom_qwen38_model_dspark_temporal_capabilities_get(
                        state->model, &prefill_caps) == AXIOM_OK &&
                prefill_caps.temporal_m8_available;
        const bool trace_prefill = sink && sink->codex_target_prefill &&
                std::getenv("AXIOM_CODEX_PREFILL_TRACE") != nullptr;
        const bool skip_unused_predictions = sink && sink->codex_target_prefill && !vision &&
                env_enabled("AXIOM_CODEX_PREFILL_SKIP_UNUSED_PREDICTION");
        const bool paged_known_prefill = sink && sink->codex_target_prefill && !vision &&
                env_enabled("AXIOM_CODEX_PAGED_KNOWN_PREFILL");
        size_t next_trace = prefill_start;
        size_t m8_prompt_tokens = 0u;
        for (size_t index = prefill_start; index < model_prompt_ids.size(); ++index) {
            if (sink && sink->progress) sink->progress->prefill(
                    static_cast<uint32_t>(prefill_start), static_cast<uint32_t>(index),
                    static_cast<uint32_t>(model_prompt_ids.size()));
            if (trace_prefill && index >= next_trace) {
                std::fprintf(stderr, "axiom-target-prefill: session=%s path=%s position=%zu total=%zu m8=%zu elapsed_s=%.6f\n",
                        session_id.c_str(), out->decode_path.c_str(), index,
                        model_prompt_ids.size(), m8_prompt_tokens,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_begin).count());
                next_trace = index + 1024u;
            }
            if (generation_cancelled(sink)) {
                *failure_stage = "client_disconnect";
                rc = AXIOM_ERR_IO;
                break;
            }
            // Keep the final prompt token scalar when sampling requires the
            // complete vocabulary logits, not the M8 greedy reduction.
            const size_t m8_end = index + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
            if (target_m8_prefill && codex_target_prefill_block_fits(
                    index, model_prompt_ids.size(), state->speculative_context_tokens,
                    sampling && sampling->enabled)) {
                axiom_qwen38_model_transaction *transaction = nullptr;
                rc = axiom_qwen38_model_transaction_begin(state->model, &transaction);
                if (rc == AXIOM_OK) {
                    axiom_qwen38_model_dspark_verify_block8_result block{};
                    rc = axiom_qwen38_model_transaction_verify_block8(
                            transaction, model_prompt_ids.data() + index, nullptr, &block);
                    if (rc == AXIOM_OK) {
                        rc = axiom_qwen38_model_transaction_commit_prefix(
                                transaction, AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH);
                        // A valid commit consumes the transaction even when
                        // a lower-layer commit fails. Never abort it twice.
                        transaction = nullptr;
                        if (rc == AXIOM_OK) {
                            anchor = block.target_token_ids[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH - 1u];
                            anchor_logit = block.target_logits[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH - 1u];
                            m8_prompt_tokens += AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
                            index = m8_end - 1u;
                            continue;
                        }
                    }
                    if (transaction) {
                        const int abort_rc = axiom_qwen38_model_transaction_abort(transaction);
                        if (abort_rc != AXIOM_OK) rc = abort_rc;
                    }
                }
                *failure_stage = "target_m8_prefill";
                break;
            }
            const bool sample_last = sampling && sampling->enabled &&
                    index + 1u == model_prompt_ids.size();
            // Do not batch the final prompt token (sampling needs its logits).
            // Resident/page-boundary eligibility is checked before any write;
            // an execution error rolls back, never retries a partially written block.
            if (paged_known_prefill && index >= state->speculative_context_tokens &&
                model_prompt_ids.size() - index > AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH &&
                axiom_qwen38_model_can_prefill_paged8(state->model)) {
                uint32_t ids[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH]{};
                float logits[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH]{};
                rc = axiom_qwen38_model_prefill_paged8(state->model,
                        model_prompt_ids.data() + index, ids, logits);
                if (rc != AXIOM_OK) { *failure_stage = "native_prefill_paged8"; break; }
                out->paged_prefill_tokens += AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
                index += AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH - 1u;
                continue;
            }
            // The last prompt token still produces the full normal prediction.
            // This opt-in does not widen temporal/speculative execution, change
            // prompt content, or touch vision and legacy request paths.
            if (skip_unused_predictions && index >= state->speculative_context_tokens &&
                index + 1u < model_prompt_ids.size()) {
                rc = axiom_qwen38_model_prefill_known_token(state->model, model_prompt_ids[index]);
                if (rc != AXIOM_OK) { *failure_stage = "native_prefill_known_token"; break; }
                ++out->prefill_predictions_skipped;
                continue;
            }
            const float *embedding = vision_embedding_for(vision, index);
            if (embedding) {
                if (sample_last) {
                    if (sampling_logits.size() != AXIOM_QWEN38_MODEL_VOCAB) {
                        sampling_logits.resize(AXIOM_QWEN38_MODEL_VOCAB);
                    }
                    rc = axiom_qwen38_model_forward_embedding_logits(
                            state->model, model_prompt_ids[index], embedding,
                            sampling_logits.data(),
                            static_cast<uint32_t>(sampling_logits.size()),
                            &anchor, &anchor_logit);
                    if (rc == AXIOM_OK && !sample_qwen_logits(
                            sampling_logits.data(), *sampling, &anchor, &anchor_logit)) {
                        *failure_stage = "native_vision_sampling";
                        rc = AXIOM_ERR_RUNTIME;
                    }
                } else {
                    rc = axiom_qwen38_model_forward_embedding(
                            state->model, model_prompt_ids[index], embedding,
                            &anchor, &anchor_logit);
                }
                if (rc != AXIOM_OK) *failure_stage = "native_vision_prefill";
            } else {
                rc = sample_last
                        ? forward_next_token(state, model_prompt_ids[index], sampling,
                                             &sampling_logits, &anchor, &anchor_logit,
                                             failure_stage)
                                : axiom_qwen38_model_forward_token(
                                state->model, model_prompt_ids[index], &anchor, &anchor_logit);
            }
            if (rc != AXIOM_OK) break;
        }
        if (rc == AXIOM_OK && model_prompt_ids.empty()) {
            *failure_stage = "native_prefill";
            rc = AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    if (rc != AXIOM_OK) {
        const int rollback_rc = rollback_session_transaction(
                state, &activation, failure_stage);
        if (rollback_rc != AXIOM_OK) rc = rollback_rc;
        (void)axiom_qwen38_model_reset(state->model);
        return rc;
    }
        if (sink && sink->progress) {
            sink->progress->prefill(static_cast<uint32_t>(prefill_start),
                    static_cast<uint32_t>(model_prompt_ids.size()),
                    static_cast<uint32_t>(model_prompt_ids.size()));
            sink->progress->decode(0u);
        }
        const auto prefill_end = std::chrono::steady_clock::now();
        out->prefill_seconds =
                std::chrono::duration<double>(prefill_end - prefill_begin).count();
        auto append_token = [&](const uint32_t token) -> int {
            const int append_rc = append_generated_token(
                    state, out, token, sink, failure_stage);
            if (append_rc == AXIOM_OK && out->ids.size() == 1u) {
                out->ttft_seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - generation_begin).count();
            }
            return append_rc;
        };
        const auto decode_begin = std::chrono::steady_clock::now();
        bool stopped = false;
        if (rc == AXIOM_OK) {
            if (stop_token(anchor, state->tokenizer_info)) {
                stopped = true;
                out->finish_reason = "stop";
            } else {
                rc = append_token(anchor);
            }
        }
        while (rc == AXIOM_OK && !stopped &&
               generation_tokens_remaining(*out, max_new, sink) != 0u) {
            if (generation_cancelled(sink)) {
                *failure_stage = "client_disconnect";
                rc = AXIOM_ERR_IO;
                break;
            }
            if (axiom_qwen38_model_position(state->model) >= effective_context_limit(state)) {
                out->finish_reason = "length";
                break;
            }
            *failure_stage = "native_decode";
            uint32_t next = 0u;
            float next_logit = 0.0f;
            rc = forward_next_token(
                    state, anchor, sampling, &sampling_logits, &next, &next_logit,
                    failure_stage);
            if (rc != AXIOM_OK) break;
            anchor = next;
            anchor_logit = next_logit;
            if (stop_token(anchor, state->tokenizer_info)) {
                stopped = true;
                out->finish_reason = "stop";
            } else {
                rc = append_token(anchor);
            }
        }
        (void)anchor_logit;
        const auto decode_end = std::chrono::steady_clock::now();
        out->decode_seconds = std::chrono::duration<double>(decode_end - decode_begin).count();
        out->scalar_tail_tokens = out->ids.size();
        out->decode_tokens_per_second = out->decode_seconds > 0.0
                ? static_cast<double>(out->ids.size()) / out->decode_seconds : 0.0;
        out->visible_tokens_per_second = out->decode_tokens_per_second;
        out->thinking_tokens = 0u;
        out->visible_output_tokens = static_cast<uint32_t>(out->ids.size());
        if (rc == AXIOM_OK && generation_cancelled(sink)) {
            *failure_stage = "client_disconnect";
            rc = AXIOM_ERR_IO;
        }
        if (rc == AXIOM_OK && out->finish_reason.empty()) out->finish_reason = "length";

        if (rc == AXIOM_OK) {
            rc = decode_generated_ids(state, out->ids, &out->text, failure_stage);
        }
        if (rc == AXIOM_OK) {
            rc = drain_generation_utf8(sink, true, failure_stage);
        }
    if (rc == AXIOM_OK && state->kv_tier) {
        out->session_position = axiom_qwen38_model_position(state->model);
        out->session_next_token = anchor;
        if (out->session_position != durable_session_position(
                    model_prompt_ids.size(), *out)) {
            *failure_stage = "session_scalar_watermark";
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK && state->kv_tier) {
        auto job = std::make_shared<deferred_persistence_job>();
        job->state = state;
        if (persistent_session && !activation.paths.namespace_id.empty()) {
            job->namespace_lease = std::make_unique<session_namespace_lease>(
                    state, activation.paths.namespace_id);
        }
        job->prompt_ids = model_prompt_ids;
        job->vision = vision_owner;
        job->result = *out;
        job->persist_start_position = activation.start_position;
        job->graph_position = job->result.session_position;
        job->persistent_session = persistent_session;
        job->scalar_path = true;
        job->activation = std::move(activation);
        activation.transaction_active = false;
        std::vector<uint32_t>().swap(job->activation.effective_prompt_ids);
        if (!enqueue_deferred_persistence(job)) {
            run_inline_deferred_persistence(job);
        }
    } else {
        const int rollback_rc = rollback_session_transaction(
                state, &activation, failure_stage);
        if (rollback_rc != AXIOM_OK) rc = rollback_rc;
        const int reset_rc = axiom_qwen38_model_reset(state->model);
        if (reset_rc != AXIOM_OK) {
            state->healthy.store(false);
            if (rc == AXIOM_OK) {
                *failure_stage = "target_reset";
                rc = reset_rc;
            }
        }
    }
    return rc;
}

int native_generate(
        server_state *state,
        const std::vector<uint32_t> &prompt_ids,
        uint32_t max_new,
        const std::string &session_id,
        const std::string &profile,
        const std::vector<uint32_t> *session_suffix_ids,
        const bool allow_stateful_resume,
        std::vector<uint32_t> *effective_prompt_out,
        const std::shared_ptr<vision_request> &vision_owner,
        generation_result *out,
        std::string *failure_stage,
        sampling_params *sampling,
        generation_stream_sink *sink,
        const speculative_mode requested_mode,
        const uint32_t speculative_max_commit_tokens) {
    const auto generation_begin = std::chrono::steady_clock::now();
    const uint32_t context_limit = effective_context_limit(state);
    if (!state || !state->model || !state->tokenizer || !out || !failure_stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const auto run_target = [&](sampling_params *target_sampling,
                                const char *decode_path,
                                const char *fallback_reason) {
        const int target_rc = native_generate_streaming(
                state, prompt_ids, max_new, session_id, profile,
                session_suffix_ids, allow_stateful_resume, effective_prompt_out,
                vision_owner,
                out, failure_stage, target_sampling, sink,
                decode_path, fallback_reason, requested_mode);
        out->speculative_max_commit_tokens = speculative_max_commit_tokens;
        return target_rc;
    };
    if (speculative_max_commit_tokens == 0u ||
        speculative_max_commit_tokens > AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
        (requested_mode != speculative_mode::dspark &&
         requested_mode != speculative_mode::native_mtp &&
         speculative_max_commit_tokens != AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH)) {
        *failure_stage = "speculative_commit_limit_contract";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (requested_mode == speculative_mode::target) {
        return run_target(
                sampling, sampling && sampling->enabled
                        ? "target_only_sampling" : "target_only",
                nullptr);
    }
    if ((requested_mode == speculative_mode::dspark &&
         state->engine != speculative_engine::dspark) ||
        (requested_mode == speculative_mode::native_mtp &&
         state->engine != speculative_engine::native_mtp)) {
        *out = generation_result{};
        out->speculative_mode_requested = speculative_mode_name(requested_mode);
        *failure_stage = "requested_speculative_engine_not_loaded";
        return AXIOM_ERR_NOT_IMPLEMENTED;
    }
    /* Speculative engines are exact greedy decoders. Positive-temperature
     * sampling is valid only on the target. Auto routes there; an explicit
     * engine request fails closed instead of changing the requested engine. */
    if (sampling && sampling->enabled) {
        if (requested_mode == speculative_mode::dspark ||
            requested_mode == speculative_mode::native_mtp) {
            *out = generation_result{};
            out->speculative_mode_requested = speculative_mode_name(requested_mode);
            *failure_stage = "speculative_graph_requires_greedy_sampling";
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        return run_target(sampling, "scalar_sampling", "sampling_enabled");
    }
    if (vision_owner && state->engine == speculative_engine::native_mtp) {
        if (requested_mode == speculative_mode::native_mtp) {
            *out = generation_result{};
            out->speculative_mode_requested = "native_mtp";
            *failure_stage = "native_mtp_vision_prefill_unsupported";
            return AXIOM_ERR_NOT_IMPLEMENTED;
        }
        return run_target(nullptr, "target_only_vision", "vision_prefill");
    }
    const bool configured_engine_available =
            state->engine == speculative_engine::native_mtp
            ? state->native_mtp_loaded && state->mtp
            : state->draft != nullptr;
    const bool streaming_graph_candidate =
            state->streaming_kv && state->temporal_hot &&
            configured_engine_available &&
            !prompt_ids.empty() && max_new != 0u &&
            static_cast<uint64_t>(prompt_ids.size()) +
                    AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH <=
                    state->speculative_context_tokens;
    const bool full_request_fits_graph = !prompt_ids.empty() && max_new != 0u &&
            static_cast<uint64_t>(prompt_ids.size()) + max_new - 1u <=
                    state->speculative_context_tokens;
    const bool explicit_speculative =
            requested_mode == speculative_mode::dspark ||
            requested_mode == speculative_mode::native_mtp;
    if (explicit_speculative) {
        if (!configured_engine_available ||
            (state->streaming_kv && !state->temporal_hot)) {
            *out = generation_result{};
            out->speculative_mode_requested = speculative_mode_name(requested_mode);
            *failure_stage = "speculative_engine_not_available";
            return AXIOM_ERR_NOT_IMPLEMENTED;
        }
        if (state->streaming_kv &&
            (!streaming_graph_candidate || !full_request_fits_graph)) {
            *out = generation_result{};
            out->speculative_mode_requested = speculative_mode_name(requested_mode);
            *failure_stage = "speculative_context_window";
            return AXIOM_ERR_BUDGET;
        }
    } else if (state->streaming_kv && !streaming_graph_candidate) {
        return run_target(nullptr, "scalar_context", "prompt_exceeds_speculative_window");
    }
    if (!configured_engine_available) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = generation_result{};
    out->decode_path = state->engine == speculative_engine::native_mtp
            ? "native_mtp_graph" : "speculative_graph";
    out->speculative_mode_requested = speculative_mode_name(requested_mode);
    out->speculative_mode_effective = speculative_engine_name(state->engine);
    out->speculative_max_commit_tokens = speculative_max_commit_tokens;
    out->speculative_context_tokens = state->speculative_context_tokens;
    failure_stage->clear();
    const vision_request *vision = vision_owner.get();
    if (prompt_ids.empty() || prompt_ids.size() > context_limit || max_new == 0u ||
        static_cast<uint64_t>(prompt_ids.size()) + max_new - 1u > context_limit) {
        *failure_stage = "context_budget";
        return AXIOM_ERR_BUDGET;
    }
    /* max_tokens is deliberately absent from graph eligibility. The graph
     * consumes the validated resident prefix and hands an exact anchor to the
     * paged target decoder when a long response crosses that boundary. */
    const uint32_t graph_context = streaming_graph_candidate
            ? state->speculative_context_tokens : context_limit;
    const uint32_t session_context = graph_context;

    const auto acquire_begin = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock = acquire_generation_session_lock(state);
    out->session_acquire_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - acquire_begin).count();
    if (!state->healthy.load()) {
        *failure_stage = "target_unhealthy";
        return AXIOM_ERR_RUNTIME;
    }
    const auto prefill_begin = std::chrono::steady_clock::now();

    session_activation activation;
    const bool persistent_session =
            session_persistence_enabled(state);
    int rc = validate_target_session(state, failure_stage);
    if (rc == AXIOM_OK && persistent_session) {
        const uint32_t resume_prompt_limit = context_limit - (max_new - 1u);
        rc = activate_session(
                state, session_id, profile, prompt_ids, session_suffix_ids,
                false, !vision_owner, allow_stateful_resume, resume_prompt_limit,
                &activation, failure_stage, sink ? sink->codex_replay_tools : nullptr);
    } else if (rc == AXIOM_OK) {
        *failure_stage = "target_reset";
        if (state->kv_tier) rc = axiom_qwen38_kv_tier_reset(state->kv_tier);
        if (rc == AXIOM_OK) rc = axiom_qwen38_model_reset(state->model);
    }
    const std::vector<uint32_t> &model_prompt_ids =
            activation.effective_prompt_ids.empty()
            ? prompt_ids : activation.effective_prompt_ids;
    if (rc == AXIOM_OK && effective_prompt_out) {
        try {
            *effective_prompt_out = model_prompt_ids;
        } catch (...) {
            *failure_stage = "session_effective_prompt_alloc";
            rc = AXIOM_ERR_BUDGET;
        }
    }
    out->session_restored = persistent_session && activation.restored;
    out->session_stateful_resume = persistent_session && activation.stateful_resume;
    out->session_prefix_retokenized = persistent_session && activation.prefix_retokenized;
    out->session_tool_projection = persistent_session && activation.tool_projection;
    if (rc == AXIOM_OK &&
        (activation.start_position > graph_context ||
         static_cast<uint64_t>(model_prompt_ids.size()) +
                 AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH > graph_context)) {
        if (explicit_speculative) {
            /* The explicit control is used for attributable A/B runs. A
             * restored prefix that no longer fits must be reported, not
             * converted into an unlabelled target-only measurement. */
            *failure_stage = "speculative_restored_context_window";
            const int rollback_rc = rollback_session_transaction(
                    state, &activation, failure_stage);
            if (rollback_rc != AXIOM_OK) return rollback_rc;
            return AXIOM_ERR_BUDGET;
        }
        /* Stateful resume can make the exact native prefix longer than the
         * wire prompt used for the preliminary route. Re-enter the scalar
         * path only after releasing the serialized generation lock; the next
         * activation restores the same durable target namespace. */
        const int rollback_rc = rollback_session_transaction(
                state, &activation, failure_stage);
        if (rollback_rc != AXIOM_OK) return rollback_rc;
        lock.unlock();
        return run_target(
                nullptr, "scalar_context",
                "effective_context_exceeds_speculative_window");
    }
    out->prefix_hit_tokens = out->session_restored
            ? activation.start_position : 0u;
    out->suffix_prefill_tokens = static_cast<uint32_t>(
            model_prompt_ids.size() -
            std::min<size_t>(activation.start_position, model_prompt_ids.size()));
    request_session session{};
    if (rc == AXIOM_OK) {
        *failure_stage = "resident_graph_executor_take";
        rc = take_resident_graph_session(
                state, &session, &out->graph_executor_reused);
        if (rc == AXIOM_OK && !out->graph_executor_reused) {
            rc = create_request_session(state, &session, failure_stage);
        }
    }
    if (rc == AXIOM_OK && activation.restored) {
        *failure_stage = "session_restore_target";
        rc = restore_session_target(
                state, state->kv_tier, activation.manifest, true, failure_stage);
    }
    if (rc == AXIOM_OK && activation.restored) {
        *failure_stage = state->engine == speculative_engine::native_mtp
                ? "session_restore_native_mtp" : "session_restore_dspark";
        rc = restore_session_speculative(
                state, &session, activation.manifest, failure_stage);
    }

    uint32_t anchor = activation.restored && activation.start_position == model_prompt_ids.size()
            ? activation.next_token : 0u;
    if (rc == AXIOM_OK) *failure_stage = "speculative_prefill";
    const bool use_prompt_graph = sink && sink->codex_target_prefill && !vision &&
            session.engine == speculative_engine::dspark && speculative_max_commit_tokens != 1u &&
            env_enabled("AXIOM_CODEX_TARGET_PREFILL_GRAPH") &&
            env_enabled("AXIOM_QWEN38_HOST_M8_EXACT_TILED") &&
            env_enabled("AXIOM_QWEN38_DEVICE_TEMPORAL_EXACT") &&
            !env_enabled("AXIOM_QWEN38_KV_FP8_PARITY") &&
            !env_enabled("AXIOM_QWEN38_DEVICE_TEMPORAL_EXACT_REFERENCE") && graph_context <= 8192u;
    // Destroy the request-local graph before decode/persistence can transfer
    // model ownership to another worker; retain only native committed state.
    {
    axiom_codex_prefill_graph prompt_graph;
    uint32_t graph_prompt_tokens = 0u;
    for (uint32_t index = activation.start_position;
         index < model_prompt_ids.size() && rc == AXIOM_OK;) {
        if (sink && sink->progress) sink->progress->prefill(activation.start_position,
                index, static_cast<uint32_t>(model_prompt_ids.size()));
        if (generation_cancelled(sink)) {
            *failure_stage = "client_disconnect";
            rc = AXIOM_ERR_IO;
            break;
        }
        bool block8_eligible = session.engine == speculative_engine::dspark &&
                speculative_max_commit_tokens != 1u &&
                static_cast<uint64_t>(index) + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH <=
                model_prompt_ids.size();
        if (block8_eligible && vision) {
            for (uint32_t time = 0u;
                 time < AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH; ++time) {
                if (vision_embedding_for(vision, index + time)) {
                    block8_eligible = false;
                    break;
                }
            }
        }
        if (block8_eligible) {
            if (use_prompt_graph) {
                *failure_stage = "speculative_target_prefill_graph";
                if (!prompt_graph.ready()) rc = prompt_graph.prepare(state->model, graph_context);
                float logit = 0.0f;
                if (rc == AXIOM_OK) rc = prompt_graph.step(
                        model_prompt_ids.data() + index, &anchor, &logit, session.compute);
                if (rc != AXIOM_OK) break;
                graph_prompt_tokens += AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
                index += AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
                continue;
            }
            axiom_qwen38_speculative_prefill_block8_request request{};
            request.abi_version = AXIOM_ABI_VERSION;
            for (uint32_t time = 0u;
                 time < AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH; ++time) {
                request.token_ids[time] = model_prompt_ids[index + time];
            }
            axiom_qwen38_speculative_prefill_block8_result result{};
            result.abi_version = AXIOM_ABI_VERSION;
            *failure_stage = "speculative_prefill_block8";
            rc = axiom_qwen38_speculative_prefill_block8(
                    session.speculative, &request, &result);
            if (rc == AXIOM_OK &&
                (result.token_start_position != index ||
                 result.position_after_commit !=
                         index + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
                 result.target_token_ids[
                         AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH - 1u] >=
                         AXIOM_QWEN38_DSPARK_VOCAB)) {
                rc = AXIOM_ERR_RUNTIME;
            }
            if (rc == AXIOM_OK) {
                anchor = result.target_token_ids[
                        AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH - 1u];
                index += AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
            }
            continue;
        }

        if (prompt_graph.ready()) {
            rc = prompt_graph.finish(model_prompt_ids.data(), index);
            if (rc != AXIOM_OK) { *failure_stage = "speculative_prefill_graph_materialize"; break; }
        }
        *failure_stage = session.engine == speculative_engine::native_mtp
                ? "native_mtp_prefill" : "speculative_prefill";
        rc = request_session_prefill_token(
                &session, model_prompt_ids[index],
                vision_embedding_for(vision, index), index,
                &anchor, failure_stage);
        if (rc == AXIOM_OK) {
            ++index;
        }
    }
    if (rc == AXIOM_OK && prompt_graph.ready()) {
        rc = prompt_graph.finish(model_prompt_ids.data(), static_cast<uint32_t>(model_prompt_ids.size()));
        if (rc != AXIOM_OK) *failure_stage = "speculative_prefill_graph_materialize";
    }
    if (graph_prompt_tokens) {
        std::fprintf(stderr, "axiom-speculative-prefill-graph: session=%s tokens=%u status=%d elapsed_s=%.6f\n",
                session_id.c_str(), graph_prompt_tokens, rc,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - prefill_begin).count());
    }
    }
    out->prompt_tokens = static_cast<uint32_t>(prompt_ids.size());

    /* Qualification-only localization gate. A commit limit of one asks the
     * graph to expose only target row zero, so first prove that the target
     * state produced by speculative prefill is scalar-equivalent before any
     * decode cycle can mutate it. The validator restores the exact committed
     * history and leaves no open transaction. */
    if (rc == AXIOM_OK && speculative_max_commit_tokens == 1u) {
        uint32_t probe[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH]{};
        for (uint32_t index = 0u;
             index < AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH; ++index) {
            probe[index] = anchor;
        }
        axiom_qwen38_model_dspark_temporal_validation_result validation{};
        validation.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
        const uint64_t recurrent_bytes = axiom_qwen38_model_recurrent_state_bytes();
        std::vector<float> recurrent_before;
        std::vector<float> recurrent_after;
        try {
            if (recurrent_bytes % sizeof(float) != 0u) throw std::bad_alloc();
            recurrent_before.resize(static_cast<size_t>(recurrent_bytes / sizeof(float)));
            recurrent_after.resize(static_cast<size_t>(recurrent_bytes / sizeof(float)));
        } catch (...) {
            *failure_stage = "diagnostic_prefill_recurrent_alloc";
            rc = AXIOM_ERR_BUDGET;
        }
        if (rc == AXIOM_OK) {
            *failure_stage = "diagnostic_prefill_recurrent_export_before";
            rc = axiom_qwen38_model_recurrent_state_export(
                    state->model, recurrent_before.data(), recurrent_bytes);
        }
        *failure_stage = "diagnostic_prefill_temporal_parity";
        if (rc == AXIOM_OK) {
            rc = axiom_qwen38_model_dspark_temporal8_validate(
                    state->model, probe, 0.0f, &validation);
        }
        if (rc == AXIOM_OK) {
            *failure_stage = "diagnostic_prefill_recurrent_export_after";
            rc = axiom_qwen38_model_recurrent_state_export(
                    state->model, recurrent_after.data(), recurrent_bytes);
        }
        uint64_t recurrent_mismatch_count = 0u;
        uint64_t recurrent_max_error_index = 0u;
        float recurrent_max_error = 0.0f;
        if (rc == AXIOM_OK) {
            const float *before = recurrent_before.data();
            const float *after = recurrent_after.data();
            const uint64_t count = recurrent_bytes / sizeof(float);
            for (uint64_t index = 0u; index < count; ++index) {
                const float error = std::fabs(before[index] - after[index]);
                if (error != 0.0f) ++recurrent_mismatch_count;
                if (!std::isfinite(error) || error > recurrent_max_error) {
                    recurrent_max_error = error;
                    recurrent_max_error_index = index;
                }
            }
        }
        std::fprintf(
                stderr,
                "axiom-qwen38-api: diagnostic prefill temporal8 position=%u "
                "passed=%u max_logit_error=%.9g max_tap_error=%.9g "
                "recurrent_mismatches=%llu recurrent_max_error=%.9g "
                "recurrent_max_error_index=%llu\n",
                validation.tested_position, validation.passed,
                static_cast<double>(validation.max_logit_abs_error),
                static_cast<double>(validation.max_tap_abs_error),
                static_cast<unsigned long long>(recurrent_mismatch_count),
                static_cast<double>(recurrent_max_error),
                static_cast<unsigned long long>(recurrent_max_error_index));
        if (rc == AXIOM_OK && validation.passed != 1u) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }

    /* Prefill uses the ABI-v1 default stream. The decode graph runs on a
     * nonblocking stream, so make the ownership handoff explicit once, before
     * capture/replay; there is no synchronization inside the decode chunks. */
    if (rc == AXIOM_OK) {
        *failure_stage = "prefill_stream_sync";
        rc = cuda_status(cudaStreamSynchronize(nullptr));
    }

    if (rc == AXIOM_OK) {
        *failure_stage = "prefill_watermark";
        rc = request_session_position_verify(
                state, &session,
                static_cast<uint32_t>(model_prompt_ids.size()), failure_stage);
    }

    if (rc == AXIOM_OK && sink && sink->progress) {
        sink->progress->prefill(activation.start_position,
                static_cast<uint32_t>(model_prompt_ids.size()),
                static_cast<uint32_t>(model_prompt_ids.size()));
        sink->progress->decode(0u);
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    out->prefill_seconds =
            std::chrono::duration<double>(prefill_end - prefill_begin).count();
    auto append_token = [&](const uint32_t token) -> int {
        const int append_rc = append_generated_token(
                state, out, token, sink, failure_stage);
        if (append_rc == AXIOM_OK && out->ids.size() == 1u) {
            out->ttft_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - generation_begin).count();
        }
        return append_rc;
    };

    bool stopped = false;
    if (rc == AXIOM_OK) {
        if (stop_token(anchor, state->tokenizer_info)) {
            stopped = true;
            out->finish_reason = "stop";
        } else {
            rc = append_token(anchor);
        }
    }

    if (rc == AXIOM_OK && !stopped &&
        generation_tokens_remaining(*out, max_new, sink) != 0u) {
        if (!session.stream) {
            rc = allocate_device_decode(
                    state, &session, kDeviceChunkCycles, failure_stage);
            if (rc == AXIOM_OK) state->graph_executor_creates.fetch_add(1u);
        } else if (session.history_capacity < kDeviceChunkCycles ||
                   !session.seed_anchor_device || !session.seed_position_device ||
                   (session.engine == speculative_engine::dspark &&
                    (!session.history_host || !session.history_device)) ||
                   (session.engine == speculative_engine::native_mtp &&
                    (!session.mtp_history_device || !session.mtp_history_host))) {
            *failure_stage = "resident_graph_executor_incomplete";
            rc = AXIOM_ERR_RUNTIME;
        }
    }

    const uint32_t initial_position = static_cast<uint32_t>(model_prompt_ids.size());
    if (rc == AXIOM_OK && session.stream) {
        *failure_stage = "device_seed";
        cudaError_t status = cudaMemcpyAsync(
                session.seed_anchor_device, &anchor, sizeof(anchor),
                cudaMemcpyHostToDevice, session.stream);
        if (status == cudaSuccess) {
            status = cudaMemcpyAsync(
                    session.seed_position_device, &initial_position, sizeof(initial_position),
                    cudaMemcpyHostToDevice, session.stream);
        }
        rc = cuda_status(status);
        if (rc == AXIOM_OK) session.stream_synchronized = false;
    }

    const uint32_t *anchor_device = session.seed_anchor_device;
    const uint32_t *position_device = session.seed_position_device;
    uint32_t committed_position = initial_position;
    bool device_session_handed_off = false;
    bool device_session_active = false;
    const auto decode_begin = std::chrono::steady_clock::now();
    while (rc == AXIOM_OK && !stopped &&
           generation_tokens_remaining(*out, max_new, sink) != 0u &&
           committed_position < session_context) {
        if (generation_cancelled(sink)) {
            *failure_stage = "client_disconnect";
            rc = AXIOM_ERR_IO;
            break;
        }
        if (!session.stream || committed_position > session_context) {
            *failure_stage = "device_context_suffix";
            rc = AXIOM_ERR_BUDGET;
            break;
        }
        const uint32_t needed = generation_tokens_remaining(*out, max_new, sink);
        /* A graph cycle commits between one and VERIFY_WIDTH tokens. Never
         * enqueue a cycle unless its worst-case output fits inside the public
         * max_tokens boundary. The final zero-to-seven slots are completed by
         * the exact one-token path below. */
        const uint32_t cycle_commit_limit = std::min(
                needed, speculative_max_commit_tokens);
        const bool limited_commit_cycle =
                cycle_commit_limit < AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
        const uint32_t boundary_safe_cycles = limited_commit_cycle
                ? 1u : needed / AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
        if (AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH >
                session_context - committed_position) {
            break;
        }
        /* A normal request amortizes the graph synchronization over 33
         * cycles. A live SSE request trades that batching for one cycle so
         * committed tokens can cross the socket immediately. */
        uint32_t batch_cycles = limited_commit_cycle
                ? 1u
                : (sink && sink->emit ? state->sse_batch_cycles : kDeviceChunkCycles);
        if (batch_cycles > boundary_safe_cycles) batch_cycles = boundary_safe_cycles;
        const uint32_t safe_cycles =
                (session_context - committed_position) /
                AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
        if (batch_cycles > safe_cycles) batch_cycles = safe_cycles;
        if (batch_cycles == 0u) {
            *failure_stage = "device_batch_capacity";
            rc = AXIOM_ERR_BUDGET;
            break;
        }
        if (limited_commit_cycle) {
            *failure_stage = "device_commit_limit";
            rc = session.engine == speculative_engine::native_mtp
                    ? axiom_qwen38_mtp_speculative_device_commit_limit_set(
                            session.mtp_speculative, cycle_commit_limit,
                            session.stream)
                    : axiom_qwen38_speculative_device_commit_limit_set(
                            session.speculative, cycle_commit_limit,
                            session.stream);
            if (rc != AXIOM_OK) break;
        }

        for (uint32_t cycle = 0u; cycle < batch_cycles && rc == AXIOM_OK; ++cycle) {
            if (generation_cancelled(sink)) {
                *failure_stage = "client_disconnect";
                rc = AXIOM_ERR_IO;
                break;
            }
            *failure_stage = "device_graph_replay";
            session.stream_synchronized = false;
            if (session.engine == speculative_engine::native_mtp) {
                axiom_qwen38_mtp_speculative_device_step_request request{};
                request.abi_version =
                        AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION;
                request.struct_size = sizeof(request);
                request.anchor_token_device = anchor_device;
                request.anchor_position_device = position_device;
                request.stream = session.stream;
                axiom_qwen38_mtp_speculative_device_step_result result{};
                result.abi_version =
                        AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION;
                result.struct_size = sizeof(result);
                rc = axiom_qwen38_mtp_speculative_device_step_enqueue(
                        session.mtp_speculative, &request, &result);
                if (rc == AXIOM_OK &&
                    (result.graph_replayed != 1u ||
                     result.draft_tokens !=
                             AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS ||
                     result.verify_width !=
                             AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH ||
                     !result.next_anchor_token_device ||
                     !result.next_anchor_position_device)) {
                    rc = AXIOM_ERR_RUNTIME;
                }
                if (rc == AXIOM_OK) device_session_active = true;
                if (rc == AXIOM_OK) {
                    *failure_stage = "native_mtp_history_snapshot";
                    rc = enqueue_native_history_snapshot(
                            &session.mtp_history_device[cycle], result,
                            session.stream);
                }
                if (rc == AXIOM_OK) {
                    anchor_device = result.next_anchor_token_device;
                    position_device = result.next_anchor_position_device;
                }
            } else {
                axiom_qwen38_speculative_device_step_request request{};
                request.abi_version =
                        AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION;
                request.anchor_token_device = anchor_device;
                request.anchor_position_device = position_device;
                request.stream = session.stream;
                axiom_qwen38_speculative_device_step_result result{};
                result.abi_version =
                        AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION;
                rc = axiom_qwen38_speculative_device_step_enqueue(
                        session.speculative, &request, &result);
                if (rc == AXIOM_OK &&
                    (result.graph_replayed != 1u ||
                     result.proposal_tokens !=
                             AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS ||
                     result.verify_width !=
                             AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
                     !result.next_anchor_token_device ||
                     !result.next_anchor_position_device)) {
                    rc = AXIOM_ERR_RUNTIME;
                }
                if (rc == AXIOM_OK) device_session_active = true;
                if (rc == AXIOM_OK) {
                    *failure_stage = "device_history_copy";
                    rc = enqueue_history_copy(
                            &session.history_device[cycle], result,
                            session.stream);
                }
                if (rc == AXIOM_OK) {
                    anchor_device = result.next_anchor_token_device;
                    position_device = result.next_anchor_position_device;
                }
            }
            if (rc == AXIOM_OK) session.stream_synchronized = false;
        }
        if (rc != AXIOM_OK) break;

        if (session.engine == speculative_engine::dspark) {
            // Snapshot each graph result before its storage is reused, then
            // transfer the contiguous batch once. SSE keeps its batch limit.
            *failure_stage = "dspark_history_host_copy";
            rc = cuda_status(cudaMemcpyAsync(
                    session.history_host, session.history_device,
                    static_cast<size_t>(batch_cycles) * sizeof(cycle_history),
                    cudaMemcpyDeviceToHost, session.stream));
            if (rc != AXIOM_OK) break;
        }
        if (session.engine == speculative_engine::native_mtp) {
            *failure_stage = "native_mtp_history_host_copy";
            rc = enqueue_native_history_host_copy(&session, batch_cycles);
            if (rc != AXIOM_OK) break;
        }

        *failure_stage = "device_chunk_sync";
        rc = cuda_status(cudaStreamSynchronize(session.stream));
        session.stream_synchronized = true;
        for (uint32_t cycle = 0u; cycle < batch_cycles && rc == AXIOM_OK; ++cycle) {
            cycle_history native_translated{};
            const cycle_history *history_pointer = nullptr;
            if (session.engine == speculative_engine::native_mtp) {
                const native_cycle_history &native =
                        session.mtp_history_host[cycle];
                const bool terminal_noop = native.commit_count == 0u;
                if (terminal_noop) {
                    /* Replays already queued after a stop observe the stop
                     * token as their anchor and deliberately publish no
                     * continuation/logit. Validate that ABI before applying
                     * the ordinary committed-cycle contract. */
                    if (native.async_status != static_cast<uint32_t>(AXIOM_OK) ||
                        native.accepted_prefix != 0u || native.emitted_count != 0u ||
                        native.stop_detected != 1u ||
                        native.matched_stop_token != anchor ||
                        native.next_anchor != anchor ||
                        native.next_position != committed_position ||
                        native.verify[0] != anchor ||
                        !stop_token(anchor, state->tokenizer_info)) {
                        *failure_stage = "native_mtp_terminal_noop_contract";
                        rc = AXIOM_ERR_RUNTIME;
                        break;
                    }
                    native_translated.accepted_prefix = 0u;
                    native_translated.continuation_token =
                            native.matched_stop_token;
                    native_translated.async_status = native.async_status;
                    native_translated.next_position = native.next_position;
                    native_translated.committed_tokens = 0u;
                    history_pointer = &native_translated;
                }
                const bool stop_contract =
                        (native.stop_detected == 0u &&
                         native.matched_stop_token == AXIOM_TOKEN_ID_INVALID) ||
                        (native.stop_detected == 1u &&
                         native.matched_stop_token == native.continuation &&
                         stop_token(native.continuation, state->tokenizer_info));
                if (!terminal_noop &&
                    (native.async_status != static_cast<uint32_t>(AXIOM_OK) ||
                    native.accepted_prefix >
                            AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS ||
                    native.commit_count >
                            AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH ||
                    native.emitted_count != native.commit_count ||
                    (native.commit_count != 0u &&
                     native.commit_count != native.accepted_prefix + 1u) ||
                    native.continuation >= AXIOM_QWEN38_MODEL_VOCAB ||
                    !std::isfinite(native.continuation_logit) ||
                    !stop_contract || native.verify[0] != anchor)) {
                    *failure_stage = "native_mtp_device_result_contract";
                    rc = AXIOM_ERR_RUNTIME;
                    break;
                }
                for (uint32_t token = 0u; !terminal_noop &&
                     token < AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS;
                     ++token) {
                    if (native.draft[token] >= AXIOM_QWEN38_MODEL_VOCAB ||
                        native.verify[token + 1u] != native.draft[token]) {
                        *failure_stage = "native_mtp_draft_verify_contract";
                        rc = AXIOM_ERR_RUNTIME;
                        break;
                    }
                    native_translated.proposal[token] = native.draft[token];
                }
                if (rc != AXIOM_OK) break;
                for (uint32_t token = 0u; !terminal_noop &&
                     token < native.accepted_prefix; ++token) {
                    if (native.emitted[token] != native.draft[token]) {
                        *failure_stage = "native_mtp_emitted_prefix_contract";
                        rc = AXIOM_ERR_RUNTIME;
                        break;
                    }
                }
                if (rc != AXIOM_OK) break;
                if (!terminal_noop && native.commit_count != 0u &&
                    native.emitted[native.accepted_prefix] !=
                            native.continuation) {
                    *failure_stage = "native_mtp_continuation_contract";
                    rc = AXIOM_ERR_RUNTIME;
                    break;
                }
                if (!terminal_noop) {
                    native_translated.accepted_prefix = native.accepted_prefix;
                    native_translated.continuation_token = native.continuation;
                    native_translated.async_status = native.async_status;
                    native_translated.next_position = native.next_position;
                    native_translated.committed_tokens = native.commit_count;
                    history_pointer = &native_translated;
                }
            } else {
                history_pointer = &session.history_host[cycle];
            }
            const cycle_history &history = *history_pointer;
            if (history.async_status != static_cast<uint32_t>(AXIOM_OK)) {
                *failure_stage = "device_async_status_" + std::to_string(history.async_status);
                rc = AXIOM_ERR_RUNTIME;
                break;
            }
            if (history.accepted_prefix > AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS ||
                history.continuation_token >= AXIOM_QWEN38_DSPARK_VOCAB ||
                history.committed_tokens > AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
                (history.committed_tokens != 0u &&
                 history.committed_tokens != history.accepted_prefix + 1u) ||
                static_cast<uint64_t>(committed_position) + history.committed_tokens >
                        session_context) {
                *failure_stage = "device_result_range";
                rc = AXIOM_ERR_RUNTIME;
                break;
            }
            for (uint32_t token = 0u; token < AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS; ++token) {
                if (history.proposal[token] >= AXIOM_QWEN38_DSPARK_VOCAB) {
                    *failure_stage = "device_proposal_range";
                    rc = AXIOM_ERR_RUNTIME;
                    break;
                }
            }
            if (rc != AXIOM_OK) break;
            if (history.committed_tokens == 0u) {
                if (history.next_position != committed_position ||
                    !stop_token(history.continuation_token, state->tokenizer_info)) {
                    *failure_stage = "device_terminal_noop";
                    rc = AXIOM_ERR_RUNTIME;
                    break;
                }
                anchor = history.continuation_token;
                stopped = true;
                out->finish_reason = "stop";
                continue;
            }
            committed_position += history.committed_tokens;
            if (history.next_position != committed_position) {
                *failure_stage = "device_watermark";
                rc = AXIOM_ERR_RUNTIME;
                break;
            }
            const uint32_t eligible_draft_slots = limited_commit_cycle
                    ? cycle_commit_limit - 1u
                    : AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS;
            if (session.engine == speculative_engine::dspark) {
                rc = observe_draft_confidence(
                        out, history, eligible_draft_slots, failure_stage);
                if (rc != AXIOM_OK) break;
            }
            ++out->graph_cycles;
            out->accepted_draft_tokens += history.accepted_prefix;
            out->proposed_draft_tokens += AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS;
            out->eligible_draft_tokens += eligible_draft_slots;
            out->graph_emitted_tokens += history.committed_tokens;
            anchor = history.continuation_token;

            if (stopped || out->ids.size() >= max_new) continue;
            for (uint32_t token = 0u;
                 token < history.accepted_prefix && out->ids.size() < max_new;
                 ++token) {
                if (stop_token(history.proposal[token], state->tokenizer_info)) {
                    /* Stop-aware acceptance must clamp before EOS and expose
                     * it as continuation, never as an accepted proposal. */
                    *failure_stage = "device_stop_commit_invariant";
                    rc = AXIOM_ERR_RUNTIME;
                    break;
                }
                rc = append_token(history.proposal[token]);
                if (rc != AXIOM_OK) break;
            }
            if (rc == AXIOM_OK && !stopped && out->ids.size() < max_new) {
                if (stop_token(history.continuation_token, state->tokenizer_info)) {
                    stopped = true;
                    out->finish_reason = "stop";
                } else {
                    rc = append_token(history.continuation_token);
                }
            }
        }
    }

    auto end_device_session_exact = [&](const uint32_t exact_position) -> int {
        if (!request_session_ready(&session) || !session.stream ||
            exact_position < model_prompt_ids.size()) {
            *failure_stage = "device_session_handoff_arguments";
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (exact_position != committed_position) {
            *failure_stage = "device_session_physical_durable_mismatch";
            return AXIOM_ERR_RUNTIME;
        }
        const size_t generated_to_commit =
                static_cast<size_t>(exact_position) - model_prompt_ids.size();
        if (generated_to_commit > out->ids.size()) {
            *failure_stage = "device_session_handoff_history_alignment";
            return AXIOM_ERR_RUNTIME;
        }
        std::vector<uint32_t> committed_history;
        try {
            committed_history.reserve(exact_position);
            committed_history.insert(
                    committed_history.end(),
                    model_prompt_ids.begin(), model_prompt_ids.end());
            committed_history.insert(
                    committed_history.end(), out->ids.begin(),
                    out->ids.begin() +
                            static_cast<std::ptrdiff_t>(generated_to_commit));
        } catch (...) {
            *failure_stage = "device_session_handoff_history_alloc";
            return AXIOM_ERR_BUDGET;
        }
        if (committed_history.size() != exact_position) {
            *failure_stage = "device_session_handoff_history_alignment";
            return AXIOM_ERR_RUNTIME;
        }
        *failure_stage = "device_session_handoff";
        int handoff_rc = session.engine == speculative_engine::native_mtp
                ? axiom_qwen38_mtp_speculative_device_session_end(
                        session.mtp_speculative, session.stream,
                        committed_history.data(), exact_position)
                : axiom_qwen38_speculative_device_session_end(
                        session.speculative, session.stream,
                        committed_history.data(), exact_position);
        uint32_t device_anchor = 0u;
        if (handoff_rc == AXIOM_OK) {
            const cudaError_t anchor_status = cudaMemcpy(
                    &device_anchor, anchor_device, sizeof(device_anchor),
                    cudaMemcpyDeviceToHost);
            handoff_rc = cuda_status(anchor_status);
            if (handoff_rc == AXIOM_OK && device_anchor != anchor) {
                *failure_stage = "device_session_anchor_mismatch";
                handoff_rc = AXIOM_ERR_RUNTIME;
            }
        }
        if (handoff_rc == AXIOM_OK) {
            handoff_rc = request_session_position_verify(
                    state, &session, exact_position, failure_stage);
        }
        if (handoff_rc == AXIOM_OK) {
            session.stream_synchronized = true;
            device_session_handed_off = true;
        }
        return handoff_rc;
    };

    /* The graph deliberately stops when fewer than one fixed verify block
     * remains in its resident window. End device-control ownership at that
     * exact prefix; the paged target can consume the pending anchor directly.
     * The one-token speculative prefill ABI is not usable here because it is
     * implemented by an eight-token target verification internally. */
    if (rc == AXIOM_OK && device_session_active && !stopped &&
        generation_tokens_remaining(*out, max_new, sink) != 0u) {
        rc = end_device_session_exact(committed_position);
    }
    bool hybrid_scalar = false;
    if (rc == AXIOM_OK && !stopped &&
        generation_tokens_remaining(*out, max_new, sink) != 0u &&
        explicit_speculative) {
        *failure_stage = "explicit_speculative_unexpected_target_handoff";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK && !stopped &&
        generation_tokens_remaining(*out, max_new, sink) != 0u) {
        if (committed_position > session_context ||
            session_context - committed_position >=
                    AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
            axiom_qwen38_model_position(state->model) != committed_position) {
            *failure_stage = "hybrid_handoff_watermark";
            rc = AXIOM_ERR_RUNTIME;
        }
        /* Materialize the exact GRAPH prefix in the generation-owned tier
         * before scalar paging may recycle any resident page. Do not advance
         * the durable watermark yet: a crash must still expose the previous
         * manifest/tier pair until the full response is committed. */
        if (rc == AXIOM_OK && state->kv_tier) {
            *failure_stage = "hybrid_graph_prefix_stage";
            rc = persist_kv_pages(
                    state, &session, committed_position,
                    activation.start_position, failure_stage, false);
        }
        if (rc == AXIOM_OK) {
            *failure_stage = "hybrid_graph_session_destroy";
            rc = destroy_request_session(&session);
            if (rc == AXIOM_OK) session = request_session{};
        }
        if (rc == AXIOM_OK) {
            hybrid_scalar = true;
            out->decode_path = state->engine == speculative_engine::native_mtp
                    ? "native_mtp_graph_scalar" : "hybrid_graph_scalar";
            out->fallback_reason = "speculative_window_exhausted";
            out->speculative_mode_effective =
                    std::string(speculative_engine_name(state->engine)) + "+target";
        }
    }
    while (rc == AXIOM_OK && hybrid_scalar && !stopped &&
           generation_tokens_remaining(*out, max_new, sink) != 0u) {
        if (generation_cancelled(sink)) {
            *failure_stage = "client_disconnect";
            rc = AXIOM_ERR_IO;
            break;
        }
        if (committed_position >= context_limit) {
            out->finish_reason = "length";
            break;
        }
        uint32_t next = 0u;
        float next_logit = 0.0f;
        *failure_stage = "hybrid_scalar_decode";
        rc = forward_next_token(
                state, anchor, nullptr, nullptr, &next, &next_logit,
                failure_stage);
        if (rc != AXIOM_OK) break;
        ++committed_position;
        anchor = next;
        if (stop_token(anchor, state->tokenizer_info)) {
            stopped = true;
            out->finish_reason = "stop";
        } else {
            rc = append_token(anchor);
            if (rc == AXIOM_OK) ++out->scalar_tail_tokens;
        }
    }
    if (rc == AXIOM_OK && session.stream && device_session_active &&
        !device_session_handed_off && !hybrid_scalar) {
        /* Stop-aware device acceptance guarantees that physical target/GDN
         * state already equals the client-visible durable frontier. Never
         * attempt a metadata-only rewind of recurrent state. */
        const uint32_t exact_position = durable_session_position(
                model_prompt_ids.size(), *out);
        /* A stop token is physically consumed but is not part of visible
         * history. Materialize the graph watermark now; the deferred worker
         * rebuilds the exact visible prefix before publishing KV/manifest. */
        rc = end_device_session_exact(
                exact_position == committed_position
                ? exact_position : committed_position);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    out->decode_seconds = std::chrono::duration<double>(decode_end - decode_begin).count();
    if (out->decode_seconds > 0.0) {
        out->decode_tokens_per_second =
                static_cast<double>(out->graph_emitted_tokens +
                                    out->canonical_tail_tokens +
                                    out->scalar_tail_tokens) /
                out->decode_seconds;
        const uint64_t visible_graph_tokens = out->ids.size();
        out->visible_tokens_per_second =
                static_cast<double>(visible_graph_tokens) / out->decode_seconds;
    }
    if (out->eligible_draft_tokens != 0u) {
        out->acceptance = static_cast<double>(out->accepted_draft_tokens) /
                static_cast<double>(out->eligible_draft_tokens);
    }

    if (rc == AXIOM_OK && out->finish_reason.empty()) out->finish_reason = "length";
    if (rc == AXIOM_OK) {
        rc = decode_generated_ids(state, out->ids, &out->text, failure_stage);
    }
    if (rc == AXIOM_OK) {
        rc = drain_generation_utf8(sink, true, failure_stage);
    }
    out->thinking_tokens = 0u;
    out->visible_output_tokens = static_cast<uint32_t>(out->ids.size());

    if (rc == AXIOM_OK && generation_cancelled(sink)) {
        *failure_stage = "client_disconnect";
        rc = AXIOM_ERR_IO;
    }
    uint32_t persist_start_position = activation.start_position;
    if (rc == AXIOM_OK && state->kv_tier) {
        out->session_position = committed_position;
        if (!device_session_handed_off && session.stream && anchor_device) {
            const cudaError_t anchor_status = cudaMemcpy(
                    &out->session_next_token, anchor_device, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost);
            if (anchor_status != cudaSuccess) {
                *failure_stage = "session_next_token_copy";
                rc = cuda_status(anchor_status);
            }
        } else {
            out->session_next_token = anchor;
        }
    }
    if (rc == AXIOM_OK && state->kv_tier) {
        if (persistent_session) {
            const uint32_t durable_position = durable_session_position(
                    model_prompt_ids.size(), *out);
            if (durable_position > committed_position ||
                axiom_qwen38_model_position(state->model) != committed_position) {
                *failure_stage = "session_durable_watermark_mismatch";
                rc = AXIOM_ERR_RUNTIME;
            } else {
                out->session_position = committed_position;
            }
        }
    }
    if (rc == AXIOM_OK && state->kv_tier) {
        auto job = std::make_shared<deferred_persistence_job>();
        job->state = state;
        if (persistent_session && !activation.paths.namespace_id.empty()) {
            job->namespace_lease = std::make_unique<session_namespace_lease>(
                    state, activation.paths.namespace_id);
        }
        job->prompt_ids = model_prompt_ids;
        job->vision = vision_owner;
        job->result = *out;
        job->session = session;
        job->persist_start_position = persist_start_position;
        job->graph_position = committed_position;
        job->persistent_session = persistent_session;
        job->scalar_path = hybrid_scalar;
        job->activation = std::move(activation);
        activation.transaction_active = false;
        std::vector<uint32_t>().swap(job->activation.effective_prompt_ids);
        if (!enqueue_deferred_persistence(job)) {
            run_inline_deferred_persistence(job);
        }
        /* Ownership has moved to the job.  Do not destroy the CUDA stream or
         * reset the model here: the worker performs both after KV commit and
         * manifest publication. */
        session = request_session{};
    } else {
        const int rollback_rc = rollback_session_transaction(
                state, &activation, failure_stage);
        if (rollback_rc != AXIOM_OK) rc = rollback_rc;
        const int destroy_rc = destroy_request_session(&session);
        if (rc == AXIOM_OK && destroy_rc != AXIOM_OK) {
            *failure_stage = "request_session_destroy";
            rc = destroy_rc;
        }
        const int reset_rc = axiom_qwen38_model_reset(state->model);
        if (reset_rc != AXIOM_OK) {
            state->healthy.store(false);
            if (rc == AXIOM_OK) {
                *failure_stage = "target_reset";
                rc = reset_rc;
            }
        }
    }
    return rc;
}

size_t find_first_token_sequence(
        const std::vector<uint32_t> &ids, const std::vector<uint32_t> &needle) {
    if (needle.empty() || ids.size() < needle.size()) return std::string::npos;
    for (size_t start = 0u; start <= ids.size() - needle.size(); ++start) {
        if (std::equal(needle.begin(), needle.end(), ids.begin() + start)) return start;
    }
    return std::string::npos;
}

int refresh_generation_text(
        server_state *state, generation_result *result, std::string *failure_stage) {
    if (!state || !result || !failure_stage) return AXIOM_ERR_INVALID_ARGUMENT;
    result->text.clear();
    result->thinking_tokens = 0u;
    result->visible_output_tokens = static_cast<uint32_t>(result->ids.size());
    const int decode_rc = decode_generated_ids(
            state, result->ids, &result->text, failure_stage);
    if (decode_rc != AXIOM_OK) return decode_rc;
    std::vector<uint32_t> end_marker;
    if (append_text_ids(state, "</think>", &end_marker) == AXIOM_OK) {
        const size_t marker_start = find_first_token_sequence(result->ids, end_marker);
        if (marker_start != std::string::npos) {
            result->thinking_tokens = static_cast<uint32_t>(marker_start + end_marker.size());
            result->visible_output_tokens = static_cast<uint32_t>(
                    result->ids.size() - result->thinking_tokens);
        }
    }
    result->visible_tokens_per_second = result->decode_seconds > 0.0
            ? static_cast<double>(result->visible_output_tokens) / result->decode_seconds : 0.0;
    return AXIOM_OK;
}

int merge_generation_results(
        server_state *state, const generation_result &thinking,
        const generation_result *visible, uint32_t original_prompt_tokens,
        uint32_t thinking_budget, generation_result *out,
        std::string *failure_stage) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = thinking;
    if (visible) {
        out->ids.insert(out->ids.end(), visible->ids.begin(), visible->ids.end());
        out->graph_cycles += visible->graph_cycles;
        out->graph_emitted_tokens += visible->graph_emitted_tokens;
        out->canonical_tail_tokens += visible->canonical_tail_tokens;
        out->scalar_tail_tokens += visible->scalar_tail_tokens;
        out->accepted_draft_tokens += visible->accepted_draft_tokens;
        out->proposed_draft_tokens += visible->proposed_draft_tokens;
        out->eligible_draft_tokens += visible->eligible_draft_tokens;
        merge_draft_confidence(out, *visible);
        out->decode_seconds += visible->decode_seconds;
        out->prefill_seconds += visible->prefill_seconds;
        out->session_acquire_seconds += visible->session_acquire_seconds;
        out->prefix_hit_tokens = visible->prefix_hit_tokens;
        out->suffix_prefill_tokens = visible->suffix_prefill_tokens;
        if (!visible->decode_path.empty()) out->decode_path = visible->decode_path;
        if (!visible->speculative_mode_effective.empty() &&
            visible->speculative_mode_effective != out->speculative_mode_effective) {
            const std::string configured = state
                    ? speculative_engine_name(state->engine) : "dspark";
            const std::string mixed = configured + "+target";
            const bool mixed_target_speculative =
                    (out->speculative_mode_effective == configured &&
                     visible->speculative_mode_effective == "target") ||
                    (out->speculative_mode_effective == "target" &&
                     visible->speculative_mode_effective == configured) ||
                    out->speculative_mode_effective == mixed ||
                    visible->speculative_mode_effective == mixed;
            out->speculative_mode_effective = mixed_target_speculative
                    ? mixed : visible->speculative_mode_effective;
        }
        if (!visible->fallback_reason.empty()) {
            out->fallback_reason = visible->fallback_reason;
        }
        out->session_position = visible->session_position;
        out->session_next_token = visible->session_next_token;
        out->graph_executor_reused =
                out->graph_executor_reused || visible->graph_executor_reused;
        if (!visible->finish_reason.empty()) out->finish_reason = visible->finish_reason;
    }
    out->prompt_tokens = original_prompt_tokens;
    out->thinking_budget = thinking_budget;
    if (out->eligible_draft_tokens != 0u) {
        out->acceptance = static_cast<double>(out->accepted_draft_tokens) /
                static_cast<double>(out->eligible_draft_tokens);
    }
    const int refresh_rc = refresh_generation_text(state, out, failure_stage);
    if (refresh_rc != AXIOM_OK) return refresh_rc;
    /* When the thinking budget is reached, the early-stop marker is part of
     * the second phase prompt, not of either generated token sequence.  Keep
     * it in the decoded representation so the public adapters can separate
     * hidden reasoning from visible output, without counting prompt marker
     * tokens as generated completion tokens. */
    if (visible && thinking.text.find("</think>") == std::string::npos) {
        out->text = thinking.text;
        out->text += kThinkingEarlyStopText;
        out->text += visible->text;
        out->thinking_tokens = static_cast<uint32_t>(thinking.ids.size());
        out->visible_output_tokens = static_cast<uint32_t>(visible->ids.size());
    }
    out->decode_tokens_per_second = out->decode_seconds > 0.0
            ? static_cast<double>(
                    out->graph_emitted_tokens + out->canonical_tail_tokens +
                    out->scalar_tail_tokens) /
                    out->decode_seconds
            : 0.0;
    out->visible_tokens_per_second = out->decode_seconds > 0.0
            ? static_cast<double>(out->visible_output_tokens) / out->decode_seconds : 0.0;
    return AXIOM_OK;
}

int native_generate_with_thinking(
        server_state *state, const std::vector<uint32_t> &prompt_ids,
        const thinking_plan &plan, uint32_t original_prompt_tokens,
        const std::string &session_id, const std::string &profile,
        const std::vector<uint32_t> *session_suffix_ids,
        const bool allow_stateful_resume,
        const std::shared_ptr<vision_request> &vision_owner,
        sampling_params *sampling, generation_result *out,
        std::string *failure_stage, generation_stream_sink *sink,
        const speculative_mode requested_mode,
        const uint32_t speculative_max_commit_tokens) {
    if (!state || !out || !failure_stage) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!plan.enabled) {
        const int rc = native_generate(
                state, prompt_ids, plan.visible_budget, session_id, profile,
                session_suffix_ids, allow_stateful_resume, nullptr,
                vision_owner,
                out, failure_stage,
                sampling, sink, requested_mode, speculative_max_commit_tokens);
        if (rc == AXIOM_OK) {
            out->thinking_budget = 0u;
            out->thinking_tokens = 0u;
            out->visible_output_tokens = static_cast<uint32_t>(out->ids.size());
        }
        return rc;
    }
    generation_result thinking;
    std::vector<uint32_t> effective_thinking_prompt;
    generation_stream_sink thinking_sink;
    if (sink) {
        thinking_sink = *sink;
        /* Thinking remains hidden from the wire, but keeps the same
         * cancellation/deadline probe so a cancelled agent does not occupy
         * the serialized native model for its full reasoning budget. */
        thinking_sink.emit = nullptr;
        thinking_sink.user = nullptr;
    }
    thinking_sink.output_budget = generation_output_budget{};
    thinking_sink.output_budget.visible_limit = plan.visible_budget;
    int rc = append_text_ids(state, "</think>",
                            &thinking_sink.output_budget.reasoning_end_marker);
    if (rc != AXIOM_OK || thinking_sink.output_budget.reasoning_end_marker.empty()) {
        *failure_stage = "thinking_end_marker_tokenize";
        return rc != AXIOM_OK ? rc : AXIOM_ERR_RUNTIME;
    }
    rc = native_generate(
            state, prompt_ids, plan.thinking_budget, session_id, profile,
            session_suffix_ids, allow_stateful_resume, &effective_thinking_prompt,
            vision_owner,
            &thinking, failure_stage,
            sampling, &thinking_sink, requested_mode,
            speculative_max_commit_tokens);
    if (rc != AXIOM_OK) return rc;

    const bool has_end_marker = thinking_sink.output_budget.reasoning_ended;
    const bool naturally_finished = has_end_marker && thinking.finish_reason == "stop";
    if (naturally_finished) {
        return merge_generation_results(
                state, thinking, nullptr, original_prompt_tokens,
                plan.thinking_budget, out, failure_stage);
    }

    std::vector<uint32_t> continuation = effective_thinking_prompt.empty()
            ? prompt_ids : std::move(effective_thinking_prompt);
    continuation.insert(continuation.end(), thinking.ids.begin(), thinking.ids.end());
    if (!has_end_marker) {
        rc = append_text_ids(state, kThinkingEarlyStopText, &continuation);
        if (rc != AXIOM_OK) {
            *failure_stage = "thinking_early_stop_context";
            return rc;
        }
    }
    // Reuse the generation-time count. A literal closing tag in the visible
    // answer must not reset the allowance and authorize another visible pass.
    const uint32_t visible_already = thinking_sink.output_budget.visible_count;
    if (visible_already >= plan.visible_budget) {
        return merge_generation_results(
                state, thinking, nullptr, original_prompt_tokens,
                plan.thinking_budget, out, failure_stage);
    }
    const uint32_t visible_budget = plan.visible_budget - visible_already;
    if (sink) sink->reasoning_separator_pending = has_end_marker;
    rc = emit_thinking_visible_prefix(
            state, thinking, visible_already, sink, failure_stage);
    if (rc != AXIOM_OK) return rc;
    generation_result visible;
    if (sink && sink->progress) sink->progress->next_pass();
    rc = native_generate(
            state, continuation, visible_budget, session_id, profile,
            nullptr, false, nullptr,
            vision_owner,
            &visible, failure_stage,
            sampling, sink, requested_mode, speculative_max_commit_tokens);
    if (rc != AXIOM_OK) return rc;
    return merge_generation_results(
            state, thinking, &visible, original_prompt_tokens,
            plan.thinking_budget, out, failure_stage);
}

ajson reasoning_efforts_json() {
    ajson efforts = ajson::jarr();
    for (size_t index = 0u; index < axiom::reasoning::kProfileCount; ++index) {
        efforts.push(ajson::jstr(axiom::reasoning::kProfiles[index].id));
    }
    return efforts;
}

ajson speculative_modes_json() {
    ajson modes = ajson::jarr();
    modes.push(ajson::jstr("auto"));
    modes.push(ajson::jstr("dspark"));
    modes.push(ajson::jstr("native_mtp"));
    modes.push(ajson::jstr("target"));
    return modes;
}

ajson reasoning_profiles_json() {
    ajson profiles = ajson::jarr();
    for (size_t index = 0u; index < axiom::reasoning::kProfileCount; ++index) {
        const axiom::reasoning::profile &selected = axiom::reasoning::kProfiles[index];
        ajson item = ajson::jobj();
        const double ratio = static_cast<double>(selected.thinking_budget_tokens) /
                static_cast<double>(axiom::reasoning::kMaxThinkingBudgetTokens);
        item.set("id", ajson::jstr(selected.id));
        item.set("label", ajson::jstr(selected.label));
        item.set("thinking", ajson::jbool(selected.thinking));
        item.set("thinking_budget_tokens", ajson::jint(selected.thinking_budget_tokens));
        item.set("native_conditioning", ajson::jstr(native_reasoning_conditioning(selected.id)));
        item.set("max_thinking_budget_tokens",
                 ajson::jint(axiom::reasoning::kMaxThinkingBudgetTokens));
        item.set("budget_fraction_of_max", ajson::jnum(ratio));
        item.set("budget_percent_of_max", ajson::jnum(ratio * 100.0));
        item.set("temperature", ajson::jnum(selected.temperature));
        item.set("top_p", ajson::jnum(selected.top_p));
        item.set("top_k", ajson::jint(selected.top_k));
        item.set("deterministic", ajson::jbool(
                selected.temperature == 0.0f && selected.top_p == 1.0f &&
                selected.top_k == 1u));
        profiles.push(std::move(item));
    }
    return profiles;
}

ajson vision_input_modalities_json(const server_state *state) {
    ajson modalities = ajson::jarr();
    modalities.push(ajson::jstr("text"));
    if (state && state->vision) {
        modalities.push(ajson::jstr("image"));
        modalities.push(ajson::jstr("video"));
    }
    return modalities;
}

ajson vision_capabilities_json(const server_state *state) {
    const bool enabled = state && state->vision;
    ajson capabilities = ajson::jobj();
    capabilities.set("enabled", ajson::jbool(enabled));
    capabilities.set("input_modalities", vision_input_modalities_json(state));
    capabilities.set("output_modalities", [] {
        ajson modalities = ajson::jarr();
        modalities.push(ajson::jstr("text"));
        return modalities;
    }());
    capabilities.set("depth", ajson::jint(enabled ? state->vision_info.depth : 0u));
    capabilities.set("output_size", ajson::jint(
            enabled ? state->vision_info.output_size : 0u));
    capabilities.set("max_patch_tokens", ajson::jnull());
    capabilities.set("patch_budget_mode", ajson::jstr("dynamic_device_memory_per_segment"));
    capabilities.set("memory_policy_schema", ajson::jstr("axiom-vision-memory-v2"));
    capabilities.set("resident_patch_capacity", ajson::jint(enabled ? state->vision_max_patch_tokens : 0u));
    capabilities.set("image_formats", [] {
        ajson formats = ajson::jarr();
        formats.push(ajson::jstr("image/png"));
        formats.push(ajson::jstr("image/jpeg"));
        formats.push(ajson::jstr("image/webp"));
        return formats;
    }());
    capabilities.set("video_container_decoder", ajson::jbool(enabled));
    capabilities.set("video_frame_list", ajson::jbool(enabled));
    capabilities.set("video_max_frames", ajson::jint(
            enabled ? kVisionMaxFrames : 0u));
    capabilities.set("video_max_width", ajson::jint(
            enabled ? kVisionMaxDecodeWidth : 0u));
    capabilities.set("video_max_height", ajson::jint(
            enabled ? kVisionMaxDecodeHeight : 0u));
    capabilities.set("video_max_frame_pixels", ajson::jint(
            enabled ? static_cast<long long>(kVisionMaxFramePixels) : 0ll));
    capabilities.set("remote_urls", ajson::jbool(false));
    capabilities.set("file_urls", ajson::jbool(false));
    return capabilities;
}

void attach_speculative_contract(ajson *object, const server_state *state) {
    if (!object || !object->is_object()) return;
    object->set("speculative_contract_schema", ajson::jstr(
            state && state->engine == speculative_engine::native_mtp
            ? "axiom_native_mtp_runtime_v1"
            : axiom::qwen38::spec_identity::kSidecarSchema));
    object->set("speculative_fingerprint", state &&
            !state->speculative_fingerprint.empty()
            ? ajson::jstr(state->speculative_fingerprint)
            : ajson::jnull());
    object->set("speculative_qualified", ajson::jbool(
            state && state->speculative_qualified));
    object->set("speculative_diagnostic_override", ajson::jbool(
            state && state->speculative_diagnostic_override));
    object->set("speculative_qualification_evidence_sha256",
            state && state->speculative_qualified
            ? ajson::jstr(state->speculative_qualification.evidence_sha256)
            : ajson::jnull());
    object->set("speculative_qualification_source_commit",
            state && state->speculative_qualified
            ? ajson::jstr(state->speculative_qualification.source_commit)
            : ajson::jnull());
}

void attach_speculative_engine_status(
        ajson *object, const server_state *state) {
    if (!object || !object->is_object()) return;
    const bool native = state &&
            state->engine == speculative_engine::native_mtp;
    object->set("speculative_engine_configured", ajson::jstr(
            state ? speculative_engine_name(state->engine) : "unavailable"));
    object->set("speculative_engine_loaded", ajson::jbool(
            state && speculative_graph_loaded(state)));
    object->set("native_mtp_loaded", ajson::jbool(
            native && state->native_mtp_loaded));
    object->set("native_mtp_fail_closed", ajson::jbool(native));
    object->set("native_mtp_tensor_count", ajson::jint(
            native ? state->mtp_info.tensor_count : 0u));
    object->set("native_mtp_cache_layout", ajson::jstr(
            native ? "bf16_hbm_e4m3_nvme" : "unavailable"));
    object->set("native_mtp_context_tokens", ajson::jint(
            native ? state->speculative_context_tokens : 0u));
    object->set("native_mtp_fast_graph_min_position", ajson::jint(
            native ? state->native_mtp_fast_graph_min_position : 0u));
    object->set("native_mtp_temporal_policy", ajson::jstr(
            native && state->native_mtp_fast_graph_min_position != 0u
            ? "exact_then_native_splitk" : native ? "exact" : "unavailable"));
    object->set("native_mtp_persistent_kv_restore", ajson::jbool(
            native && session_persistence_enabled(state)));
    object->set("native_mtp_vision_prefill", ajson::jbool(false));
}

void attach_speculative_contract(ajson *object, const generation_result &result) {
    if (!object || !object->is_object()) return;
    object->set("speculative_contract_schema", ajson::jstr(
            axiom::qwen38::spec_identity::kSidecarSchema));
    object->set("speculative_fingerprint",
            result.speculative_fingerprint.empty()
            ? ajson::jnull() : ajson::jstr(result.speculative_fingerprint));
    object->set("speculative_qualified", ajson::jbool(
            result.speculative_qualified));
    object->set("speculative_qualification_evidence_sha256",
            result.speculative_qualified
            ? ajson::jstr(result.speculative_qualification_evidence_sha256)
            : ajson::jnull());
    object->set("speculative_qualification_source_commit",
            result.speculative_qualified
            ? ajson::jstr(result.speculative_qualification_source_commit)
            : ajson::jnull());
}

ajson models_response(const server_state *state, bool client_tool_search = false) {
    ajson model = ajson::jobj();
    const uint32_t context_window = state ? state->default_context : kDefaultRequestContext;
    const uint32_t max_context_window = state ? state->max_context : kMaxNativeContext;
    model.set("id", ajson::jstr(kModelId));
    model.set("object", ajson::jstr("model"));
    model.set("owned_by", ajson::jstr("axiom"));
    model.set("backend", ajson::jstr(kBackendId));
    model.set("context_window", ajson::jint(context_window));
    model.set("max_context_window", ajson::jint(max_context_window));
    model.set("default_context_window", ajson::jint(context_window));
    model.set("context_window_options", [&] {
        ajson options = ajson::jarr();
        options.push(ajson::jint(context_window));
        if (max_context_window != context_window) {
            options.push(ajson::jint(max_context_window));
        }
        return options;
    }());
    model.set("default_profile", ajson::jstr(
            state && !state->no_think ? "medium" : "ultra-fast"));
    model.set("default_reasoning_effort", ajson::jstr(
            state && !state->no_think ? "medium" : "ultra-fast"));
    model.set("default_thinking_budget", ajson::jint(
            state && !state->no_think ? kDefaultThinkingBudget : 0u));
    model.set("max_thinking_budget", ajson::jint(
            axiom::reasoning::kMaxThinkingBudgetTokens));
    model.set("qwen_thinking_temperature", ajson::jnum(kThinkingTemperature));
    model.set("qwen_thinking_top_p", ajson::jnum(kThinkingTopP));
    model.set("qwen_thinking_top_k", ajson::jint(kDefaultQwenTopK));
    model.set("reasoning_efforts", reasoning_efforts_json());
    model.set("reasoning_profiles", reasoning_profiles_json());
    model.set("wire_api", ajson::jstr("responses"));
    model.set("tool_calling", ajson::jbool(true));
    model.set("supports_parallel_tool_calls", ajson::jbool(true));
    model.set("supports_tool_search", ajson::jbool(true));
    model.set("deferred_tool_loading", ajson::jbool(true));
    model.set("speculative_graph", ajson::jbool(
            speculative_graph_loaded(state)));
    model.set("speculative_mode_default", ajson::jstr("auto"));
    model.set("speculative_modes", speculative_modes_json());
    model.set("speculative_mode_request_field", ajson::jstr(
            "axiom_speculative_mode"));
    model.set("hybrid_graph_scalar", ajson::jbool(
            state && state->streaming_kv && speculative_graph_loaded(state)));
    model.set("speculative_context_tokens", ajson::jint(
            state ? state->speculative_context_tokens : 0u));
    model.set("resident_graph_executor", ajson::jbool(true));
    attach_speculative_contract(&model, state);
    attach_speculative_engine_status(&model, state);
    model.set("input_modalities", vision_input_modalities_json(state));
    model.set("output_modalities", [] {
        ajson modalities = ajson::jarr();
        modalities.push(ajson::jstr("text"));
        return modalities;
    }());
    model.set("vision", vision_capabilities_json(state));
    ajson data = ajson::jarr();
    data.push(std::move(model));
    ajson root = ajson::jobj();
    root.set("object", ajson::jstr("list"));
    root.set("data", std::move(data));
    ajson codex_models = ajson::jarr();
    codex_models.push(axiom_codex::catalog_model(
            kModelId, context_window, max_context_window,
            state && !state->no_think ? "medium" : "ultra-fast", client_tool_search));
    root.set("models", std::move(codex_models));
    return root;
}

const char *session_resume_mode(const generation_result &result) {
    if (!result.session_restored) return "none";
    if (result.session_tool_projection) return "codex_wire_equivalent_prefix";
    if (result.session_prefix_retokenized) return "tokenization_equivalent_prefix";
    return result.session_stateful_resume ? "stateful_tail" : "exact_prefix";
}

std::string token_sequence_fingerprint_v1(const std::vector<uint32_t> &ids) {
    /* Two independent 64-bit lanes make accidental A/B collisions
     * negligible without adding a crypto dependency to the wire daemon. The
     * length and little-endian token bytes are part of the stable contract. */
    uint64_t fnv = 1469598103934665603ull;
    uint64_t mix = 0x6a09e667f3bcc909ull ^ static_cast<uint64_t>(ids.size());
    for (const uint32_t token : ids) {
        for (uint32_t shift = 0u; shift < 32u; shift += 8u) {
            fnv ^= static_cast<uint8_t>(token >> shift);
            fnv *= 1099511628211ull;
        }
        uint64_t value = static_cast<uint64_t>(token) + 0x9e3779b97f4a7c15ull;
        value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
        mix ^= value ^ (value >> 31u);
        mix = (mix << 17u) | (mix >> 47u);
        mix *= 0x9e3779b185ebca87ull;
    }
    char text[80]{};
    std::snprintf(
            text, sizeof(text), "tokfp1:%016llx%016llx:%llu",
            static_cast<unsigned long long>(fnv),
            static_cast<unsigned long long>(mix),
            static_cast<unsigned long long>(ids.size()));
    return text;
}

void attach_session_proof(ajson *metadata, const generation_result &result) {
    if (!metadata || !metadata->is_object()) return;
    metadata->set("session_id", ajson::jstr(result.session_id));
    metadata->set("session_restored", ajson::jbool(result.session_restored));
    metadata->set("session_resume_mode", ajson::jstr(session_resume_mode(result)));
    attach_speculative_contract(metadata, result);
}

ajson draft_confidence_json(const generation_result &result) {
    ajson root = ajson::jobj();
    root.set("schema", ajson::jstr("dspark_confidence_calibration_v1"));
    root.set("label_contract", ajson::jstr(
            "accepted_prefix_plus_first_mismatch"));
    root.set("samples", ajson::jint(static_cast<long long>(
            result.draft_confidence_samples)));
    root.set("accepted_samples", ajson::jint(static_cast<long long>(
            result.draft_confidence_accepted_samples)));
    root.set("rejected_samples", ajson::jint(static_cast<long long>(
            result.draft_confidence_rejected_samples)));
    if (result.draft_confidence_samples == 0u) {
        root.set("mean", ajson::jnull());
        root.set("min", ajson::jnull());
        root.set("max", ajson::jnull());
        root.set("brier_score", ajson::jnull());
    } else {
        root.set("mean", ajson::jnum(
                result.draft_confidence_sum /
                static_cast<double>(result.draft_confidence_samples)));
        root.set("min", ajson::jnum(result.draft_confidence_min));
        root.set("max", ajson::jnum(result.draft_confidence_max));
        root.set("brier_score", ajson::jnum(
                result.draft_confidence_brier_sum /
                static_cast<double>(result.draft_confidence_samples)));
    }
    root.set("accepted_mean", result.draft_confidence_accepted_samples == 0u
            ? ajson::jnull()
            : ajson::jnum(result.draft_confidence_accepted_sum /
                    static_cast<double>(result.draft_confidence_accepted_samples)));
    root.set("rejected_mean", result.draft_confidence_rejected_samples == 0u
            ? ajson::jnull()
            : ajson::jnum(result.draft_confidence_rejected_sum /
                    static_cast<double>(result.draft_confidence_rejected_samples)));
    ajson bins = ajson::jarr();
    for (size_t bin = 0u; bin < result.draft_confidence_bin_samples.size(); ++bin) {
        const uint64_t samples = result.draft_confidence_bin_samples[bin];
        ajson item = ajson::jobj();
        item.set("lower", ajson::jnum(static_cast<double>(bin) / 10.0));
        item.set("upper", ajson::jnum(static_cast<double>(bin + 1u) / 10.0));
        item.set("samples", ajson::jint(static_cast<long long>(samples)));
        item.set("accepted", ajson::jint(static_cast<long long>(
                result.draft_confidence_bin_accepted[bin])));
        item.set("mean_confidence", samples == 0u ? ajson::jnull() : ajson::jnum(
                result.draft_confidence_bin_sum[bin] / static_cast<double>(samples)));
        item.set("empirical_acceptance", samples == 0u ? ajson::jnull() : ajson::jnum(
                static_cast<double>(result.draft_confidence_bin_accepted[bin]) /
                static_cast<double>(samples)));
        bins.push(std::move(item));
    }
    root.set("bins", std::move(bins));
    return root;
}

void attach_decode_proof(ajson *metadata, const generation_result &result) {
    if (!metadata || !metadata->is_object()) return;
    const bool target_used = result.scalar_tail_tokens != 0u ||
            result.decode_path.rfind("scalar_", 0u) == 0u ||
            result.decode_path.rfind("target_only", 0u) == 0u;
    const bool scalar_fallback = target_used &&
            result.speculative_mode_requested != "target";
    metadata->set("decode_path", ajson::jstr(result.decode_path));
    metadata->set("speculative_mode_requested", ajson::jstr(
            result.speculative_mode_requested));
    metadata->set("speculative_mode_effective", ajson::jstr(
            result.speculative_mode_effective));
    metadata->set("speculative_max_commit_tokens", ajson::jint(
            result.speculative_max_commit_tokens));
    metadata->set("output_token_fingerprint", ajson::jstr(
            token_sequence_fingerprint_v1(result.ids)));
    metadata->set("fallback_reason", result.fallback_reason.empty()
            ? ajson::jnull() : ajson::jstr(result.fallback_reason));
    metadata->set("target_decoder_used", ajson::jbool(target_used));
    metadata->set("scalar_fallback", ajson::jbool(scalar_fallback));
    if (result.decode_path == "speculative_graph" ||
        result.decode_path == "hybrid_graph_scalar" ||
        result.decode_path == "native_mtp_graph" ||
        result.decode_path == "native_mtp_graph_scalar") {
        metadata->set("graph_executor_reused", ajson::jbool(
                result.graph_executor_reused));
    }
    metadata->set("speculative_context_tokens", ajson::jint(
            result.speculative_context_tokens));
    metadata->set("graph_cycles", ajson::jint(
            static_cast<long long>(result.graph_cycles)));
    metadata->set("graph_emitted_tokens", ajson::jint(
            static_cast<long long>(result.graph_emitted_tokens)));
    metadata->set("scalar_tail_tokens", ajson::jint(
            static_cast<long long>(result.scalar_tail_tokens)));
    metadata->set("accepted_draft_tokens", ajson::jint(
            static_cast<long long>(result.accepted_draft_tokens)));
    metadata->set("eligible_draft_tokens", ajson::jint(
            static_cast<long long>(result.eligible_draft_tokens)));
    metadata->set("speculative_acceptance", ajson::jnum(result.acceptance));
    metadata->set("draft_confidence", draft_confidence_json(result));
    metadata->set("prefix_hit_tokens", ajson::jint(result.prefix_hit_tokens));
    metadata->set("suffix_prefill_tokens", ajson::jint(result.suffix_prefill_tokens));
    if (result.prefill_predictions_skipped)
        metadata->set("prefill_predictions_skipped", ajson::jint(result.prefill_predictions_skipped));
    if (result.paged_prefill_tokens)
        metadata->set("paged_prefill_tokens", ajson::jint(result.paged_prefill_tokens));
    metadata->set("prefill_seconds", ajson::jnum(result.prefill_seconds));
    metadata->set("session_acquire_seconds", ajson::jnum(result.session_acquire_seconds));
    metadata->set("ttft_seconds", ajson::jnum(result.ttft_seconds));
}

ajson session_proof_json(const generation_result &result) {
    ajson metadata = ajson::jobj();
    attach_session_proof(&metadata, result);
    return metadata;
}

ajson completion_response(
        const generation_result &result, bool chat, uint64_t sequence,
        const native_tool_result *tool_result = nullptr) {
    const std::string visible_text = visible_model_text(result.text, result.thinking_budget != 0u);
    const std::string reasoning_text = reasoning_model_text(result.text, result.thinking_budget != 0u);
    ajson choice = ajson::jobj();
    choice.set("index", ajson::jint(0));
    const bool have_tool_calls = chat && tool_result && tool_result->status == "pass" &&
            !tool_result->calls.empty();
    if (chat && have_tool_calls) {
        ajson message = ajson::jobj();
        message.set("role", ajson::jstr("assistant"));
        message.set("content", tool_result->content.empty()
                                   ? ajson::jstr(visible_text) : ajson::jstr(tool_result->content));
        if (!reasoning_text.empty()) {
            message.set("reasoning_content", ajson::jstr(reasoning_text));
        }
        ajson calls = ajson::jarr();
        for (const ajson &call : tool_result->calls) calls.push(call);
        message.set("tool_calls", std::move(calls));
        choice.set("message", std::move(message));
        choice.set("finish_reason", ajson::jstr("tool_calls"));
    } else if (chat) {
        ajson message = ajson::jobj();
        message.set("role", ajson::jstr("assistant"));
        message.set("content", ajson::jstr(
                tool_result && !tool_result->content.empty() ? tool_result->content : visible_text));
        if (!reasoning_text.empty()) {
            message.set("reasoning_content", ajson::jstr(reasoning_text));
        }
        choice.set("message", std::move(message));
        choice.set("finish_reason", ajson::jstr(result.finish_reason));
    } else {
        choice.set("text", ajson::jstr(result.text));
        choice.set("finish_reason", ajson::jstr(result.finish_reason));
    }
    ajson choices = ajson::jarr();
    choices.push(std::move(choice));
    ajson usage = ajson::jobj();
    usage.set("prompt_tokens", ajson::jint(result.prompt_tokens));
    usage.set("completion_tokens", ajson::jint(static_cast<long long>(result.ids.size())));
    usage.set("total_tokens", ajson::jint(result.prompt_tokens + static_cast<uint32_t>(result.ids.size())));
    char id[64]{};
    std::snprintf(id, sizeof(id), "%s-%llu", chat ? "chatcmpl-axiom" : "cmpl-axiom",
                  static_cast<unsigned long long>(sequence));
    ajson root = ajson::jobj();
    root.set("id", ajson::jstr(id));
    root.set("object", ajson::jstr(chat ? "chat.completion" : "text_completion"));
    root.set("created", ajson::jint(static_cast<long long>(std::time(nullptr))));
    root.set("model", ajson::jstr(kModelId));
    root.set("choices", std::move(choices));
    root.set("usage", std::move(usage));
    ajson backend = ajson::jobj();
    backend.set("name", ajson::jstr(kBackendId));
    attach_session_proof(&backend, result);
    attach_decode_proof(&backend, result);
    backend.set("graph_cycles", ajson::jint(static_cast<long long>(result.graph_cycles)));
    backend.set("graph_emitted_tokens", ajson::jint(
            static_cast<long long>(result.graph_emitted_tokens)));
    backend.set("canonical_tail_tokens", ajson::jint(
            static_cast<long long>(result.canonical_tail_tokens)));
    backend.set("proposed_draft_tokens", ajson::jint(
            static_cast<long long>(result.proposed_draft_tokens)));
    backend.set("eligible_draft_tokens", ajson::jint(
            static_cast<long long>(result.eligible_draft_tokens)));
    backend.set("accepted_draft_tokens", ajson::jint(
            static_cast<long long>(result.accepted_draft_tokens)));
    backend.set("acceptance", ajson::jnum(result.acceptance));
    backend.set("decode_seconds", ajson::jnum(result.decode_seconds));
    backend.set("decode_tokens_per_second", ajson::jnum(result.decode_tokens_per_second));
    backend.set("visible_tokens_per_second", ajson::jnum(result.visible_tokens_per_second));
    backend.set("thinking_tokens", ajson::jint(result.thinking_tokens));
    backend.set("visible_output_tokens", ajson::jint(result.visible_output_tokens));
    backend.set("thinking_budget", ajson::jint(result.thinking_budget));
    backend.set("reasoning_effort", ajson::jstr(result.reasoning_effort));
    backend.set("context_compacted", ajson::jbool(result.context_compacted));
    backend.set("original_prompt_tokens", ajson::jint(result.original_prompt_tokens));
    backend.set("tool_calls", ajson::jint(
            static_cast<long long>(have_tool_calls ? tool_result->calls.size() : 0u)));
    if (tool_result && !tool_result->error.empty()) {
        backend.set("tool_parse_error", ajson::jstr(tool_result->error));
    }
    root.set("axiom", std::move(backend));
    return root;
}

ajson chat_stream_chunk(
        const ajson &response, ajson delta, const char *finish_reason,
        bool include_usage = false) {
    ajson choice = ajson::jobj();
    choice.set("index", ajson::jint(0));
    choice.set("delta", std::move(delta));
    choice.set("finish_reason", finish_reason
                                     ? ajson::jstr(finish_reason)
                                     : ajson::jnull());
    ajson choices = ajson::jarr();
    choices.push(std::move(choice));

    ajson chunk = ajson::jobj();
    const ajson *id = response.get("id");
    const ajson *created = response.get("created");
    const ajson *model = response.get("model");
    chunk.set("id", id ? *id : ajson::jstr("chatcmpl-axiom"));
    chunk.set("object", ajson::jstr("chat.completion.chunk"));
    chunk.set("created", created ? *created : ajson::jint(
            static_cast<long long>(std::time(nullptr))));
    chunk.set("model", model ? *model : ajson::jstr(kModelId));
    chunk.set("choices", std::move(choices));
    const ajson *axiom = response.get("axiom");
    if (axiom) chunk.set("axiom", *axiom);
    const ajson *swarm = response.get("axiom_swarm");
    if (swarm) chunk.set("axiom_swarm", *swarm);
    if (include_usage) {
        const ajson *usage = response.get("usage");
        if (usage) chunk.set("usage", *usage);
    }
    return chunk;
}

void append_chat_sse_event(std::string *body, const ajson &chunk) {
    if (!body) return;
    *body += "data: ";
    *body += ajson_dumps(chunk);
    *body += "\n\n";
}

void append_chat_text_deltas(
        std::string *body, const ajson &response, const std::string &text) {
    /* Keep chunks small enough for interactive clients without cutting a
     * UTF-8 code point. Generation is complete before tool-call parsing, so
     * this wire streaming never leaks native <tool_call> markup. */
    constexpr size_t kTextChunkBytes = 192u;
    size_t offset = 0u;
    while (offset < text.size()) {
        size_t end = std::min(text.size(), offset + kTextChunkBytes);
        while (end < text.size() && end > offset &&
               (static_cast<unsigned char>(text[end]) & 0xc0u) == 0x80u) {
            --end;
        }
        if (end == offset) end = std::min(text.size(), offset + kTextChunkBytes);
        ajson delta = ajson::jobj();
        delta.set("content", ajson::jstr(text.substr(offset, end - offset)));
        append_chat_sse_event(body, chat_stream_chunk(response, std::move(delta), nullptr));
        offset = end;
    }
}

void append_chat_tool_call_deltas(
        std::string *body, const ajson &response, const ajson &calls) {
    if (!body || !calls.is_array()) return;
    constexpr size_t kArgumentChunkBytes = 192u;
    for (size_t index = 0u; index < calls.arr.size(); ++index) {
        const ajson &call = calls.arr[index];
        const ajson *id = call.get("id");
        const ajson *function = call.get("function");
        const ajson *name = function && function->is_object()
                ? function->get("name") : nullptr;
        const ajson *arguments = function && function->is_object()
                ? function->get("arguments") : nullptr;

        ajson identity_function = ajson::jobj();
        identity_function.set("name", name ? *name : ajson::jstr("tool"));
        identity_function.set("arguments", ajson::jstr(""));
        ajson identity_call = ajson::jobj();
        identity_call.set("index", ajson::jint(static_cast<long long>(index)));
        identity_call.set("id", id ? *id : ajson::jstr("call_axiom"));
        identity_call.set("type", ajson::jstr("function"));
        identity_call.set("function", std::move(identity_function));
        ajson identity_calls = ajson::jarr();
        identity_calls.push(std::move(identity_call));
        ajson identity_delta = ajson::jobj();
        identity_delta.set("tool_calls", std::move(identity_calls));
        append_chat_sse_event(
                body, chat_stream_chunk(response, std::move(identity_delta), nullptr));

        const std::string argument_text = arguments && arguments->is_string()
                ? arguments->s : std::string("{}");
        size_t offset = 0u;
        while (offset < argument_text.size()) {
            size_t end = std::min(argument_text.size(), offset + kArgumentChunkBytes);
            while (end < argument_text.size() && end > offset &&
                   (static_cast<unsigned char>(argument_text[end]) & 0xc0u) == 0x80u) {
                --end;
            }
            if (end == offset) {
                end = std::min(argument_text.size(), offset + kArgumentChunkBytes);
            }
            ajson argument_function = ajson::jobj();
            argument_function.set(
                    "arguments", ajson::jstr(argument_text.substr(offset, end - offset)));
            ajson argument_call = ajson::jobj();
            argument_call.set("index", ajson::jint(static_cast<long long>(index)));
            argument_call.set("function", std::move(argument_function));
            ajson argument_calls = ajson::jarr();
            argument_calls.push(std::move(argument_call));
            ajson argument_delta = ajson::jobj();
            argument_delta.set("tool_calls", std::move(argument_calls));
            append_chat_sse_event(
                    body, chat_stream_chunk(response, std::move(argument_delta), nullptr));
            offset = end;
        }
    }
}

[[maybe_unused]] std::string chat_stream_body(const ajson &response) {
    std::string body;
    ajson role = ajson::jobj();
    role.set("role", ajson::jstr("assistant"));
    role.set("content", ajson::jstr(""));
    append_chat_sse_event(&body, chat_stream_chunk(response, std::move(role), nullptr));

    const ajson *choices = response.get("choices");
    const ajson *choice = choices && choices->is_array() && !choices->arr.empty()
            ? &choices->arr[0] : nullptr;
    const ajson *message = choice && choice->is_object() ? choice->get("message") : nullptr;
    const ajson *content = message && message->is_object() ? message->get("content") : nullptr;
    if (content && content->is_string() && !content->s.empty()) {
        append_chat_text_deltas(&body, response, content->s);
    }
    const ajson *tool_calls = message && message->is_object()
            ? message->get("tool_calls") : nullptr;
    if (tool_calls && tool_calls->is_array() && !tool_calls->arr.empty()) {
        append_chat_tool_call_deltas(&body, response, *tool_calls);
    }

    const ajson *finish = choice && choice->is_object()
            ? choice->get("finish_reason") : nullptr;
    const std::string finish_reason = finish && finish->is_string()
            ? finish->s : std::string("stop");
    append_chat_sse_event(
            &body, chat_stream_chunk(response, ajson::jobj(), finish_reason.c_str(), true));
    body += "data: [DONE]\n\n";
    return body;
}

bool socket_peer_disconnected(int fd) {
    if (fd < 0) return true;
    char probe = 0;
    const ssize_t result = recv(fd, &probe, sizeof(probe), MSG_PEEK | MSG_DONTWAIT);
    if (result == 0) return true;
    if (result > 0) return false;
    return errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK;
}

struct http_cancel_probe {
    int fd = -1;
    server_state *state = nullptr;
    axiom::qwen38::swarm_task_id task_id = 0u;
    bool has_deadline = false;
    axiom::qwen38::swarm_time_point deadline{};
};

bool http_client_cancelled_at(const http_cancel_probe *probe,
        axiom::qwen38::swarm_time_point now) {
    if (!probe || socket_peer_disconnected(probe->fd)) return true;
    if (probe->state && probe->task_id != 0u &&
        probe->state->swarm_scheduler.is_cancellation_requested(probe->task_id)) {
        return true;
    }
    return probe->has_deadline && now >= probe->deadline;
}

bool http_client_cancelled(void *user) {
    return http_client_cancelled_at(static_cast<const http_cancel_probe *>(user),
            axiom::qwen38::swarm_clock::now());
}

/* Streaming generation needs two independent liveness checks: the provider
 * stream must still be open, and the HTTP/scheduler probe must not report a
 * disconnected client, cancellation, or deadline.  Keep those checks in a
 * typed bridge instead of passing an HTTP probe to a provider-specific
 * callback (the two structures have different layouts). */
struct live_generation_cancel_probe {
    const bool *stream_open = nullptr;
    http_cancel_probe *http = nullptr;
    void *progress_user = nullptr;
    bool (*progress)(void *) = nullptr;
    std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();
    bool progress_failed = false;
};

bool live_generation_cancelled_at(live_generation_cancel_probe *probe,
        std::chrono::steady_clock::time_point now) {
    if (!probe || !probe->stream_open || !*probe->stream_open) return true;
    if (!probe->http || http_client_cancelled_at(probe->http, now) || probe->progress_failed) return true;
    // The generation owner polls this between prefill/decode units. Keep all
    // writes on that same thread; no heartbeat races with deltas or terminals.
    // Core times out parsed SSE events, so comments alone are insufficient.
    if (probe->progress && now - probe->last_progress >= std::chrono::seconds(15)) {
        probe->last_progress = now;
        probe->progress_failed = !probe->progress(probe->progress_user);
    }
    return probe->progress_failed;
}

bool live_generation_cancelled(void *user) {
    return live_generation_cancelled_at(static_cast<live_generation_cancel_probe *>(user),
            std::chrono::steady_clock::now());
}

bool swarm_take_parked_lease(
        server_state *state, axiom::qwen38::swarm_task_id task_id,
        axiom::qwen38::swarm_task_lease *out) {
    if (!state || !out) return false;
    std::lock_guard<std::mutex> lock(state->swarm_lease_mutex);
    const auto found = state->swarm_leases.find(task_id);
    if (found == state->swarm_leases.end()) return false;
    const auto task = state->swarm_scheduler.task_snapshot(task_id);
    if (!task || task->state != axiom::qwen38::swarm_task_state::running ||
        task->cancellation_requested) {
        state->swarm_leases.erase(found);
        return false;
    }
    *out = found->second;
    state->swarm_leases.erase(found);
    return true;
}

void remember_swarm_metadata(
        server_state *state, axiom::qwen38::swarm_task_id task_id,
        const swarm_request_metadata &metadata) {
    if (!state || task_id == 0u) return;
    std::lock_guard<std::mutex> lock(state->swarm_metadata_mutex);
    if (state->swarm_metadata_history.size() >= kSwarmMetadataHistory) {
        state->swarm_metadata_history.erase(state->swarm_metadata_history.begin());
    }
    state->swarm_metadata_history[task_id] = metadata;
}

bool swarm_metadata_for(
        const server_state *state, axiom::qwen38::swarm_task_id task_id,
        swarm_request_metadata *out) {
    if (!state || !out || task_id == 0u) return false;
    std::lock_guard<std::mutex> lock(state->swarm_metadata_mutex);
    const auto found = state->swarm_metadata_history.find(task_id);
    if (found == state->swarm_metadata_history.end()) return false;
    *out = found->second;
    return true;
}

void swarm_park_lease(
        server_state *state, const axiom::qwen38::swarm_task_lease &lease) {
    if (!state) return;
    std::lock_guard<std::mutex> lock(state->swarm_lease_mutex);
    // Cancellation may have completed after admit_next() returned to another
    // worker but before that worker publishes this handoff. Never resurrect a
    // terminal/cancelled lease, including after a successor owns the slot.
    const auto task = state->swarm_scheduler.task_snapshot(lease.task_id);
    if (!task || task->state != axiom::qwen38::swarm_task_state::running ||
        task->cancellation_requested) return;
    state->swarm_leases.emplace(lease.task_id, lease);
}

void swarm_cancel_queued_request(
        server_state *state, swarm_request_context *context,
        const std::string &reason) {
    if (!state || !context || context->task_id == 0u || context->completed ||
        context->admitted) return;
    // Only the owner may abandon a request before entering native generation.
    // Another worker can already have admitted it and parked its lease, so a
    // cooperative cancel alone does NOT release that scheduler slot. Serialize
    // cleanup with publication and acknowledge precisely this unowned attempt.
    // Already-owned generation is completed only by its normal request guard.
    std::lock_guard<std::mutex> lock(state->swarm_lease_mutex);
    const auto cancelled = state->swarm_scheduler.cancel_task(
            context->task_id, reason, axiom::qwen38::swarm_clock::now());
    state->swarm_leases.erase(context->task_id);
    if (cancelled.disposition == axiom::qwen38::swarm_cancel_disposition::cancellation_requested) {
        axiom::qwen38::swarm_task_completion completion;
        completion.outcome = axiom::qwen38::swarm_task_outcome::cancelled;
        completion.message = reason;
        (void)state->swarm_scheduler.complete(
                context->task_id, completion, axiom::qwen38::swarm_clock::now());
    }
    context->completed = true;
}

bool admit_swarm_request(
        int fd, server_state *state, swarm_request_context *context,
        std::string *error) {
    if (!state || !context || !error) return false;
    const auto now = axiom::qwen38::swarm_clock::now();
    axiom::qwen38::swarm_task_spec spec;
    spec.name = "http-" + std::to_string(
            static_cast<unsigned long long>(context->sequence));
    if (!context->metadata.wave_id.empty()) spec.name += "/" + context->metadata.wave_id;
    if (!context->metadata.agent_id.empty()) spec.name += "/" + context->metadata.agent_id;
    spec.priority = context->metadata.priority;
    if (context->metadata.token_budget) {
        spec.budget.max_tokens = context->metadata.token_budget;
    }
    if (context->metadata.deadline) {
        spec.budget.max_duration = context->metadata.deadline;
    }
    const axiom::qwen38::swarm_task_result submitted =
            state->swarm_scheduler.submit(spec, now);
    if (!submitted.accepted) {
        *error = submitted.error.empty() ? "swarm admission queue is full" : submitted.error;
        return false;
    }
    context->task_id = submitted.task_id;
    context->submitted = true;
    context->submitted_at = now;
    remember_swarm_metadata(state, context->task_id, context->metadata);

    for (;;) {
        const auto current = axiom::qwen38::swarm_clock::now();
        if (socket_peer_disconnected(fd)) {
            swarm_cancel_queued_request(state, context, "client_disconnect");
            *error = "client disconnected before native admission";
            return false;
        }
        if (state->swarm_scheduler.is_cancellation_requested(context->task_id)) {
            swarm_cancel_queued_request(state, context, "cancelled_before_admission");
            *error = "swarm request cancelled before native admission";
            return false;
        }
        if (context->metadata.queue_timeout &&
            current - context->submitted_at >= *context->metadata.queue_timeout) {
            swarm_cancel_queued_request(state, context, "queue_timeout");
            *error = "swarm queue timeout";
            return false;
        }
        if (context->metadata.deadline &&
            current - context->submitted_at >= *context->metadata.deadline) {
            swarm_cancel_queued_request(state, context, "deadline");
            *error = "swarm deadline exceeded while queued";
            return false;
        }
        (void)state->swarm_scheduler.tick(current);
        axiom::qwen38::swarm_task_lease lease;
        bool have_lease = swarm_take_parked_lease(state, context->task_id, &lease);
        if (!have_lease) {
            const std::optional<axiom::qwen38::swarm_task_lease> next =
                    state->swarm_scheduler.admit_next(current);
            if (next) {
                if (next->task_id == context->task_id) {
                    lease = *next;
                    have_lease = true;
                } else {
                    /* Any HTTP worker may wake the cooperative scheduler. Park
                     * leases for their owning request so admission remains
                     * priority/FIFO ordered without a dedicated dispatcher. */
                    swarm_park_lease(state, *next);
                }
            }
        }
        if (have_lease) {
            context->admitted = true;
            context->admitted_at = current;
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

void complete_swarm_request(
        server_state *state, swarm_request_context *context,
        axiom::qwen38::swarm_task_outcome outcome,
        std::uint64_t tokens_used, const std::string &message) {
    if (!state || !context || !context->admitted || context->completed) return;
    axiom::qwen38::swarm_task_completion completion;
    completion.outcome = outcome;
    completion.tokens_used = tokens_used;
    completion.message = message;
    (void)state->swarm_scheduler.complete(
            context->task_id, completion, axiom::qwen38::swarm_clock::now());
    context->completed = true;
}

struct swarm_request_guard {
    server_state *state = nullptr;
    swarm_request_context *context = nullptr;
    axiom::qwen38::swarm_task_outcome outcome =
            axiom::qwen38::swarm_task_outcome::failed;
    std::uint64_t tokens_used = 0u;
    std::string message = "request ended before swarm completion";

    ~swarm_request_guard() {
        complete_swarm_request(state, context, outcome, tokens_used, message);
    }

    void finish(
            axiom::qwen38::swarm_task_outcome selected,
            std::uint64_t tokens, const std::string &reason) {
        outcome = selected;
        tokens_used = tokens;
        message = reason;
        complete_swarm_request(state, context, outcome, tokens_used, message);
    }
};

struct active_request_guard {
    server_state *state = nullptr;
    uint64_t sequence = 0u;
    std::shared_ptr<axiom::qwen38::request_progress> progress =
            std::make_shared<axiom::qwen38::request_progress>();

    active_request_guard(
            server_state *selected_state, const uint64_t selected_sequence,
            const std::string &session_id)
            : state(selected_state), sequence(selected_sequence) {
        if (!state) return;
        std::lock_guard<std::mutex> lock(state->active_request_mutex);
        state->active_request_sequence = sequence;
        state->active_request_session_id = session_id;
        state->active_request_progress = progress;
    }

    ~active_request_guard() {
        if (!state) return;
        bool cleared = false;
        {
            std::lock_guard<std::mutex> lock(state->active_request_mutex);
            /* A just-admitted successor may already own the single scheduler
             * slot after this request published completion. Never clear it. */
            if (state->active_request_sequence == sequence) {
                state->active_request_sequence = 0u;
                state->active_request_session_id.clear();
                state->active_request_progress.reset();
                cleared = true;
            }
        }
        if (cleared) state->maintenance_cv.notify_one();
    }
};

struct live_chat_stream {
    int fd = -1;
    ajson metadata;
    std::string raw_text;
    size_t emitted_bytes = 0u;
    bool open = false;
};

bool write_http_chunk(int fd, const std::string &body) {
    char prefix[64]{};
    const int prefix_bytes = std::snprintf(prefix, sizeof(prefix), "%zx\r\n", body.size());
    return prefix_bytes > 0 && static_cast<size_t>(prefix_bytes) < sizeof(prefix) &&
            write_all(fd, prefix, static_cast<size_t>(prefix_bytes)) &&
            write_all(fd, body.data(), body.size()) && write_all(fd, "\r\n", 2u);
}

bool write_sse_headers(int fd, const std::string &session_id) {
    const std::string headers =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream; charset=utf-8\r\n"
            "Cache-Control: no-cache, no-transform\r\n"
            "X-Accel-Buffering: no\r\n"
            + std::string(kSessionHeader) + ": " + session_id + "\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Connection: close\r\n"
            "Server: axiom-qwen38-native\r\n\r\n";
    if (!write_all(fd, headers.data(), headers.size())) return false;
    const int no_delay = 1;
    (void)setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
    return true;
}

bool live_chat_send_data(live_chat_stream *stream, const std::string &data) {
    if (!stream || !stream->open) return false;
    std::string event = "data: ";
    event += data;
    event += "\n\n";
    return write_http_chunk(stream->fd, event);
}

bool live_chat_send_chunk(
        live_chat_stream *stream, ajson delta, const char *finish_reason = nullptr,
        bool include_usage = false) {
    if (!stream || !stream->open) return false;
    return live_chat_send_data(
            stream, ajson_dumps(chat_stream_chunk(
                    stream->metadata, std::move(delta), finish_reason, include_usage)));
}

size_t live_chat_safe_text_end(const std::string &text, bool final) {
    static constexpr char kToolMarker[] = "<tool_call>";
    const std::string marker(kToolMarker);
    const size_t tool_start = text.find(marker);
    if (tool_start != std::string::npos) return tool_start;
    if (final || text.empty()) return text.size();

    /* Hold back a possible partial marker at the edge of the current token.
     * This keeps native XML tool syntax entirely off the content stream even
     * when BPE pieces split the marker across several callbacks. */
    const size_t max_suffix = std::min(text.size(), marker.size() - 1u);
    for (size_t suffix = max_suffix; suffix > 0u; --suffix) {
        if (text.compare(text.size() - suffix, suffix, marker, 0u, suffix) == 0) {
            return text.size() - suffix;
        }
    }
    return text.size();
}

bool live_chat_flush_text(live_chat_stream *stream, bool final) {
    if (!stream) return false;
    const size_t safe_end = live_chat_safe_text_end(stream->raw_text, final);
    if (safe_end <= stream->emitted_bytes) return true;
    ajson delta = ajson::jobj();
    delta.set("content", ajson::jstr(stream->raw_text.substr(
            stream->emitted_bytes, safe_end - stream->emitted_bytes)));
    if (!live_chat_send_chunk(stream, std::move(delta))) return false;
    stream->emitted_bytes = safe_end;
    return true;
}

bool live_chat_emit_token(void *user, const std::string &piece) {
    live_chat_stream *stream = static_cast<live_chat_stream *>(user);
    if (!stream || !stream->open) return false;
    stream->raw_text += piece;
    return live_chat_flush_text(stream, false);
}

bool start_live_chat_stream(
        live_chat_stream *stream, int fd, uint64_t sequence,
        const std::string &session_id) {
    if (!stream) return false;
    *stream = live_chat_stream{};
    stream->fd = fd;
    stream->metadata = ajson::jobj();
    char id[64]{};
    std::snprintf(id, sizeof(id), "chatcmpl-axiom-%llu",
                  static_cast<unsigned long long>(sequence));
    stream->metadata.set("id", ajson::jstr(id));
    stream->metadata.set("object", ajson::jstr("chat.completion.chunk"));
    stream->metadata.set("created", ajson::jint(static_cast<long long>(std::time(nullptr))));
    stream->metadata.set("model", ajson::jstr(kModelId));
    ajson axiom = ajson::jobj();
    axiom.set("session_id", ajson::jstr(session_id));
    stream->metadata.set("axiom", std::move(axiom));

    if (!write_sse_headers(fd, session_id)) return false;
    stream->open = true;

    ajson role = ajson::jobj();
    role.set("role", ajson::jstr("assistant"));
    role.set("content", ajson::jstr(""));
    return live_chat_send_chunk(stream, std::move(role));
}

bool live_chat_send_tool_calls(live_chat_stream *stream, const std::vector<ajson> &calls) {
    if (!stream) return false;
    constexpr size_t kArgumentChunkBytes = 192u;
    for (size_t index = 0u; index < calls.size(); ++index) {
        const ajson &call = calls[index];
        const ajson *id = call.get("id");
        const ajson *function = call.get("function");
        const ajson *name = function && function->is_object()
                ? function->get("name") : nullptr;
        const ajson *arguments = function && function->is_object()
                ? function->get("arguments") : nullptr;

        ajson identity_function = ajson::jobj();
        identity_function.set("name", name ? *name : ajson::jstr("tool"));
        identity_function.set("arguments", ajson::jstr(""));
        ajson identity_call = ajson::jobj();
        identity_call.set("index", ajson::jint(static_cast<long long>(index)));
        identity_call.set("id", id ? *id : ajson::jstr("call_axiom"));
        identity_call.set("type", ajson::jstr("function"));
        identity_call.set("function", std::move(identity_function));
        ajson identity_calls = ajson::jarr();
        identity_calls.push(std::move(identity_call));
        ajson identity_delta = ajson::jobj();
        identity_delta.set("tool_calls", std::move(identity_calls));
        if (!live_chat_send_chunk(stream, std::move(identity_delta))) return false;

        const std::string argument_text = arguments && arguments->is_string()
                ? arguments->s : std::string("{}");
        size_t offset = 0u;
        while (offset < argument_text.size()) {
            size_t end = std::min(argument_text.size(), offset + kArgumentChunkBytes);
            while (end < argument_text.size() && end > offset &&
                   (static_cast<unsigned char>(argument_text[end]) & 0xc0u) == 0x80u) {
                --end;
            }
            if (end == offset) end = std::min(argument_text.size(), offset + kArgumentChunkBytes);
            ajson argument_function = ajson::jobj();
            argument_function.set(
                    "arguments", ajson::jstr(argument_text.substr(offset, end - offset)));
            ajson argument_call = ajson::jobj();
            argument_call.set("index", ajson::jint(static_cast<long long>(index)));
            argument_call.set("function", std::move(argument_function));
            ajson argument_calls = ajson::jarr();
            argument_calls.push(std::move(argument_call));
            ajson argument_delta = ajson::jobj();
            argument_delta.set("tool_calls", std::move(argument_calls));
            if (!live_chat_send_chunk(stream, std::move(argument_delta))) return false;
            offset = end;
        }
    }
    return true;
}

bool live_chat_send_error(live_chat_stream *stream, const std::string &message) {
    if (!stream || !stream->open) return false;
    return live_chat_send_data(
            stream, ajson_dumps(error_json(message, "server_error")));
}

bool finish_live_chat_stream(
        live_chat_stream *stream,
        const generation_result &result,
        const native_tool_result *tool_result) {
    if (!stream || !stream->open) return false;
    const bool have_tool_calls = tool_result && tool_result->status == "pass" &&
            !tool_result->calls.empty();
    stream->raw_text = tool_result && !tool_result->content.empty()
            ? tool_result->content : visible_model_text(result.text, result.thinking_budget != 0u);
    if (have_tool_calls) {
        const size_t tool_start = stream->raw_text.find("<tool_call>");
        if (tool_start != std::string::npos) {
            stream->raw_text = trim_text(stream->raw_text.substr(0u, tool_start));
        }
    }
    if (!live_chat_flush_text(stream, true)) return false;
    if (have_tool_calls && !live_chat_send_tool_calls(stream, tool_result->calls)) return false;

    stream->metadata.set("axiom", session_proof_json(result));
    ajson usage = ajson::jobj();
    usage.set("prompt_tokens", ajson::jint(result.prompt_tokens));
    usage.set("completion_tokens", ajson::jint(static_cast<long long>(result.ids.size())));
    usage.set("total_tokens", ajson::jint(
            result.prompt_tokens + static_cast<uint32_t>(result.ids.size())));
    stream->metadata.set("usage", std::move(usage));
    const char *finish_reason = have_tool_calls ? "tool_calls" : result.finish_reason.c_str();
    if (!live_chat_send_chunk(stream, ajson::jobj(), finish_reason, true)) return false;
    if (!live_chat_send_data(stream, "[DONE]")) return false;
    const bool closed = write_all(stream->fd, "0\r\n\r\n", 5u);
    stream->open = false;
    return closed;
}

void close_live_chat_stream(live_chat_stream *stream) {
    if (!stream || !stream->open) return;
    (void)write_all(stream->fd, "0\r\n\r\n", 5u);
    stream->open = false;
}

struct live_responses_stream {
    std::shared_ptr<axiom::qwen38::request_progress> progress;
    const axiom_codex::tool_wire_map *codex_wire = nullptr;
    int fd = -1;
    uint64_t sequence_number = 0u;
    long long created_at = static_cast<long long>(std::time(nullptr));
    std::string response_id;
    std::string message_id;
    std::string session_id;
    std::string raw_text;
    size_t emitted_bytes = 0u;
    bool open = false;
    bool message_started = false;
    bool terminal_sent = false;
    bool io_failed = false;
};

ajson request_progress_json(const std::shared_ptr<axiom::qwen38::request_progress> &progress) {
    if (!progress) return ajson::jnull();
    const auto value = progress->read();
    using phase = axiom::qwen38::request_progress::phase;
    ajson result = ajson::jobj();
    result.set("scope", ajson::jstr("current_request_observed_work_not_durable_commit"));
    result.set("phase", ajson::jstr(value.stage == phase::prefill ? "prefill" :
            value.stage == phase::decode ? "decode" : "session_setup"));
    result.set("cached_tokens", ajson::jint(value.cached_tokens));
    result.set("prompt_tokens", ajson::jint(value.prompt_tokens));
    result.set("prefill_tokens_processed", ajson::jint(value.processed_tokens));
    result.set("prefill_tokens_total", ajson::jint(value.prompt_tokens - value.cached_tokens));
    result.set("generated_tokens", ajson::jint(value.generated_tokens));
    result.set("pass", ajson::jint(value.pass));
    result.set("revision", ajson::jint(static_cast<long long>(value.revision)));
    result.set("elapsed_seconds", ajson::jnum(value.elapsed_seconds));
    result.set("seconds_since_advance", ajson::jnum(value.seconds_since_advance));
    return result;
}

ajson live_responses_stub(const live_responses_stream *stream) {
    ajson response = ajson::jobj();
    response.set("id", ajson::jstr(stream ? stream->response_id : "resp-axiom"));
    response.set("object", ajson::jstr("response"));
    response.set("created_at", ajson::jint(stream ? stream->created_at : 0));
    response.set("status", ajson::jstr("in_progress"));
    response.set("model", ajson::jstr(kModelId));
    response.set("output", ajson::jarr());
    response.set("output_text", ajson::jstr(""));
    response.set("parallel_tool_calls", ajson::jbool(true));
    ajson axiom = ajson::jobj();
    axiom.set("session_id", ajson::jstr(stream ? stream->session_id : ""));
    if (stream && stream->progress) axiom.set("progress", request_progress_json(stream->progress));
    response.set("axiom", std::move(axiom));
    return response;
}

bool live_responses_send_event(
        live_responses_stream *stream, const char *event_type, ajson data) {
    if (!stream || !stream->open || !event_type) return false;
    if (stream->codex_wire && (stream->terminal_sent || stream->io_failed)) return false;
    if (stream->codex_wire) stream->codex_wire->restore(data);
    data.set("type", ajson::jstr(event_type));
    data.set("sequence_number", ajson::jint(
            static_cast<long long>(stream->sequence_number++)));
    std::string body = "event: ";
    body += event_type;
    body += "\ndata: ";
    body += ajson_dumps(data);
    body += "\n\n";
    const bool sent = write_http_chunk(stream->fd, body);
    if (!sent) stream->io_failed = true;
    if (sent && (std::strcmp(event_type, "response.completed") == 0 ||
                 std::strcmp(event_type, "response.failed") == 0 ||
                 std::strcmp(event_type, "response.incomplete") == 0)) {
        stream->terminal_sent = true;
    }
    return sent;
}

bool start_live_responses_stream(
        live_responses_stream *stream, int fd, uint64_t sequence,
        const std::string &session_id, const axiom_codex::tool_wire_map *codex_wire = nullptr) {
    if (!stream) return false;
    *stream = live_responses_stream{};
    stream->codex_wire = codex_wire;
    stream->fd = fd;
    stream->response_id = "resp-axiom-" + std::to_string(
            static_cast<unsigned long long>(sequence));
    stream->message_id = "msg-axiom-" + std::to_string(
            static_cast<unsigned long long>(sequence));
    stream->session_id = session_id;
    if (!write_sse_headers(fd, session_id)) return false;
    stream->open = true;

    ajson created = ajson::jobj();
    created.set("response", live_responses_stub(stream));
    if (!live_responses_send_event(stream, "response.created", std::move(created))) return false;
    ajson in_progress = ajson::jobj();
    in_progress.set("response", live_responses_stub(stream));
    return live_responses_send_event(stream, "response.in_progress", std::move(in_progress));
}

bool live_responses_start_message(live_responses_stream *stream) {
    if (!stream || !stream->open) return false;
    if (stream->message_started) return true;
    stream->message_started = true;

    ajson item = ajson::jobj();
    item.set("id", ajson::jstr(stream->message_id));
    item.set("type", ajson::jstr("message"));
    item.set("status", ajson::jstr("in_progress"));
    item.set("role", ajson::jstr("assistant"));
    item.set("content", ajson::jarr());
    ajson added = ajson::jobj();
    added.set("output_index", ajson::jint(0));
    added.set("item", std::move(item));
    if (!live_responses_send_event(
            stream, "response.output_item.added", std::move(added))) return false;

    ajson part = ajson::jobj();
    part.set("type", ajson::jstr("output_text"));
    part.set("text", ajson::jstr(""));
    part.set("annotations", ajson::jarr());
    ajson part_added = ajson::jobj();
    part_added.set("item_id", ajson::jstr(stream->message_id));
    part_added.set("output_index", ajson::jint(0));
    part_added.set("content_index", ajson::jint(0));
    part_added.set("part", std::move(part));
    return live_responses_send_event(
            stream, "response.content_part.added", std::move(part_added));
}

bool live_responses_flush_text(live_responses_stream *stream, bool final) {
    if (!stream || !stream->open) return false;
    const size_t safe_end = live_chat_safe_text_end(stream->raw_text, final);
    if (safe_end <= stream->emitted_bytes) return true;
    if (!live_responses_start_message(stream)) return false;
    ajson delta = ajson::jobj();
    delta.set("item_id", ajson::jstr(stream->message_id));
    delta.set("output_index", ajson::jint(0));
    delta.set("content_index", ajson::jint(0));
    delta.set("delta", ajson::jstr(stream->raw_text.substr(
            stream->emitted_bytes, safe_end - stream->emitted_bytes)));
    if (!live_responses_send_event(
            stream, "response.output_text.delta", std::move(delta))) return false;
    stream->emitted_bytes = safe_end;
    return true;
}

bool live_responses_emit_token(void *user, const std::string &piece) {
    live_responses_stream *stream = static_cast<live_responses_stream *>(user);
    if (!stream || !stream->open) return false;
    stream->raw_text += piece;
    return live_responses_flush_text(stream, false);
}

bool live_responses_progress(void *user) {
    auto *stream = static_cast<live_responses_stream *>(user);
    if (!stream || stream->terminal_sent || stream->io_failed) return false;
    ajson event = ajson::jobj();
    event.set("response", live_responses_stub(stream));
    return live_responses_send_event(stream, "response.in_progress", std::move(event));
}

bool live_responses_finish_message(
        live_responses_stream *stream, const std::string &text, bool incomplete = false) {
    if (!stream || !stream->open || !stream->message_started) return true;
    ajson done = ajson::jobj();
    done.set("item_id", ajson::jstr(stream->message_id));
    done.set("output_index", ajson::jint(0));
    done.set("content_index", ajson::jint(0));
    done.set("text", ajson::jstr(text));
    if (!live_responses_send_event(
            stream, "response.output_text.done", std::move(done))) return false;

    ajson part = ajson::jobj();
    part.set("type", ajson::jstr("output_text"));
    part.set("text", ajson::jstr(text));
    part.set("annotations", ajson::jarr());
    ajson part_done = ajson::jobj();
    part_done.set("item_id", ajson::jstr(stream->message_id));
    part_done.set("output_index", ajson::jint(0));
    part_done.set("content_index", ajson::jint(0));
    part_done.set("part", std::move(part));
    if (!live_responses_send_event(
            stream, "response.content_part.done", std::move(part_done))) return false;

    ajson item = ajson::jobj();
    item.set("id", ajson::jstr(stream->message_id));
    item.set("type", ajson::jstr("message"));
    item.set("status", ajson::jstr(incomplete ? "incomplete" : "completed"));
    item.set("role", ajson::jstr("assistant"));
    ajson content = ajson::jarr();
    ajson final_part = ajson::jobj();
    final_part.set("type", ajson::jstr("output_text"));
    final_part.set("text", ajson::jstr(text));
    final_part.set("annotations", ajson::jarr());
    content.push(std::move(final_part));
    item.set("content", std::move(content));
    ajson item_done = ajson::jobj();
    item_done.set("output_index", ajson::jint(0));
    item_done.set("item", std::move(item));
    return live_responses_send_event(
            stream, "response.output_item.done", std::move(item_done));
}

bool live_responses_send_tool_calls(
        live_responses_stream *stream, const std::vector<ajson> &calls,
        const ajson *normalized_tools, uint32_t output_index_start) {
    if (!stream || !stream->open) return false;
    for (size_t index = 0u; index < calls.size(); ++index) {
        const ajson &call = calls[index];
        const ajson *id = call.get("id");
        const ajson *function = call.get("function");
        const ajson *name = function && function->is_object()
                ? function->get("name") : nullptr;
        const ajson *arguments = function && function->is_object()
                ? function->get("arguments") : nullptr;
        const std::string name_text = name && name->is_string() ? name->s : "tool";
        const std::string argument_text = arguments && arguments->is_string()
                ? arguments->s : std::string("{}");
        const ajson *definition = normalized_tools
                ? tool_function_by_name(*normalized_tools, name_text) : nullptr;
        const ajson *response_type = definition ? definition->get("axiom_responses_type") : nullptr;
        const bool custom = response_type && response_type->is_string() &&
                response_type->s == "custom";
        const uint32_t output_index = output_index_start + static_cast<uint32_t>(index);
        const std::string call_id = id && id->is_string() ? id->s : "call_axiom";

        if (stream->codex_wire && stream->codex_wire->is_client_tool_search(name_text)) {
            // ToolSearchCall has object arguments and is executed by Core.
            // Do not emit FunctionCall-specific deltas for this typed item.
            for (const char *status : {"in_progress", "completed"}) {
                ajson item = ajson::jobj(), event = ajson::jobj();
                item.set("type", ajson::jstr("function_call"));
                item.set("name", ajson::jstr(name_text)); item.set("id", ajson::jstr(call_id));
                item.set("call_id", ajson::jstr(call_id)); item.set("status", ajson::jstr(status));
                item.set("arguments", ajson::jstr(std::strcmp(status, "completed") == 0 ? argument_text : "{}"));
                event.set("output_index", ajson::jint(output_index)); event.set("item", std::move(item));
                if (!live_responses_send_event(stream, std::strcmp(status, "completed") == 0
                        ? "response.output_item.done" : "response.output_item.added", std::move(event))) return false;
            }
            continue;
        }

        ajson item = ajson::jobj();
        item.set("id", ajson::jstr(call_id));
        item.set("type", ajson::jstr(custom ? "custom_tool_call" : "function_call"));
        item.set("status", ajson::jstr("in_progress"));
        item.set("call_id", ajson::jstr(call_id));
        item.set("name", ajson::jstr(name_text));
        if (custom) item.set("input", ajson::jstr(""));
        else item.set("arguments", ajson::jstr(""));
        ajson added = ajson::jobj();
        added.set("output_index", ajson::jint(output_index));
        added.set("item", std::move(item));
        if (!live_responses_send_event(
                stream, "response.output_item.added", std::move(added))) return false;

        const char *delta_event = custom
                ? "response.custom_tool_call_input.delta"
                : "response.function_call_arguments.delta";
        const char *done_event = custom
                ? "response.custom_tool_call_input.done"
                : "response.function_call_arguments.done";
        const std::string value = custom ? [&]() {
            ajson parsed;
            std::string parse_error;
            if (ajson_parse(argument_text, parsed, parse_error) && parsed.is_object()) {
                const ajson *input = parsed.get("input");
                if (input && input->is_string()) return input->s;
            }
            return argument_text;
        }() : argument_text;

        ajson delta = ajson::jobj();
        delta.set("item_id", ajson::jstr(call_id));
        delta.set("output_index", ajson::jint(output_index));
        delta.set("delta", ajson::jstr(value));
        if (!live_responses_send_event(stream, delta_event, std::move(delta))) return false;
        ajson arguments_done = ajson::jobj();
        arguments_done.set("item_id", ajson::jstr(call_id));
        arguments_done.set("output_index", ajson::jint(output_index));
        if (custom) arguments_done.set("input", ajson::jstr(value));
        else arguments_done.set("arguments", ajson::jstr(value));
        if (!live_responses_send_event(stream, done_event, std::move(arguments_done))) return false;

        ajson completed_item = ajson::jobj();
        completed_item.set("id", ajson::jstr(call_id));
        completed_item.set("type", ajson::jstr(custom ? "custom_tool_call" : "function_call"));
        completed_item.set("status", ajson::jstr("completed"));
        completed_item.set("call_id", ajson::jstr(call_id));
        completed_item.set("name", ajson::jstr(name_text));
        if (custom) completed_item.set("input", ajson::jstr(value));
        else completed_item.set("arguments", ajson::jstr(value));
        ajson item_done = ajson::jobj();
        item_done.set("output_index", ajson::jint(output_index));
        item_done.set("item", std::move(completed_item));
        if (!live_responses_send_event(
                stream, "response.output_item.done", std::move(item_done))) return false;
    }
    return true;
}

bool live_responses_send_error(
        live_responses_stream *stream, const std::string &message,
        const char *code = "server_error") {
    if (!stream || !stream->open) return false;
    if (stream->codex_wire) {
        if (stream->terminal_sent) return true;
        // Only bytes already emitted are part of a failed response. Never
        // expose a partial tool envelope or label incomplete output completed.
        ajson response = live_responses_stub(stream);
        response.set("status", ajson::jstr("failed"));
        ajson error = ajson::jobj();
        error.set("code", ajson::jstr(code));
        error.set("message", ajson::jstr(message));
        response.set("error", std::move(error));
        const std::string text = stream->raw_text.substr(0u, stream->emitted_bytes);
        response.set("output_text", ajson::jstr(text));
        if (stream->message_started) {
            ajson part = ajson::jobj();
            part.set("type", ajson::jstr("output_text"));
            part.set("text", ajson::jstr(text));
            part.set("annotations", ajson::jarr());
            ajson content = ajson::jarr(); content.push(std::move(part));
            ajson item = ajson::jobj();
            item.set("id", ajson::jstr(stream->message_id));
            item.set("type", ajson::jstr("message"));
            item.set("role", ajson::jstr("assistant"));
            item.set("status", ajson::jstr("incomplete"));
            item.set("content", std::move(content));
            ajson output = ajson::jarr(); output.push(std::move(item));
            response.set("output", std::move(output));
        }
        ajson failed = ajson::jobj(); failed.set("response", std::move(response));
        return live_responses_send_event(stream, "response.failed", std::move(failed));
    }
    return live_responses_send_event(
            stream, "error", error_json(message, "server_error"));
}

const char *codex_response_status(const generation_result &result,
        bool have_tool_calls, const std::string &visible) {
    if (result.finish_reason == "length") return "incomplete";
    return !have_tool_calls && trim_text(visible).empty() ? "failed" : "completed";
}

void apply_codex_response_status(ajson *response, const char *status) {
    response->set("status", ajson::jstr(status));
    if (std::strcmp(status, "incomplete") == 0) {
        ajson details = ajson::jobj();
        details.set("reason", ajson::jstr("max_output_tokens"));
        response->set("incomplete_details", std::move(details));
    } else if (std::strcmp(status, "failed") == 0) {
        ajson error = ajson::jobj();
        error.set("code", ajson::jstr("empty_model_response"));
        error.set("message", ajson::jstr("The model stopped without visible text or a tool call"));
        response->set("error", std::move(error));
    }
}

bool finish_live_responses_stream(
        live_responses_stream *stream,
        const ajson &response,
        const generation_result &result,
        const native_tool_result *tool_result,
        const ajson *normalized_tools) {
    if (!stream || !stream->open) return false;
    const bool have_tool_calls = tool_result && tool_result->status == "pass" &&
            !tool_result->calls.empty();
    const std::string emitted_prefix = stream->raw_text.substr(0, stream->emitted_bytes);
    const std::string accumulated_visible = stream->raw_text.substr(0, live_chat_safe_text_end(stream->raw_text, true));
    stream->raw_text = tool_result && !tool_result->content.empty()
            ? tool_result->content : visible_model_text(result.text, result.thinking_budget != 0u);
    if (have_tool_calls) {
        const size_t tool_start = stream->raw_text.find("<tool_call>");
        if (tool_start != std::string::npos) {
            stream->raw_text = trim_text(stream->raw_text.substr(0u, tool_start));
        }
    }
    if (stream->codex_wire && stream->raw_text.compare(0, emitted_prefix.size(), emitted_prefix) != 0) {
        // The tool/think parser may trim whitespace already delivered live.
        // Keep those bytes in the final result; never retract or repeat them.
        if (trim_text(accumulated_visible) == trim_text(stream->raw_text)) stream->raw_text = accumulated_visible;
        else {
            stream->raw_text = emitted_prefix;
            (void)live_responses_send_error(stream, "final text disagrees with already emitted deltas", "stream_content_mismatch");
            return false;
        }
    }
    const char *terminal_status = stream->codex_wire
            ? codex_response_status(result, have_tool_calls, stream->raw_text) : "completed";
    const bool successful = std::strcmp(terminal_status, "completed") == 0;
    if (!live_responses_flush_text(stream, true)) return false;
    if (!live_responses_finish_message(stream, stream->raw_text, !successful)) return false;
    const uint32_t output_index_start = stream->message_started ? 1u : 0u;
    if (have_tool_calls && successful && !live_responses_send_tool_calls(
            stream, tool_result->calls, normalized_tools, output_index_start)) return false;

    ajson completed = ajson::jobj();
    ajson final_response = response;
    final_response.set("created_at", ajson::jint(stream->created_at));
    if (stream->codex_wire) {
        apply_codex_response_status(&final_response, terminal_status);
        // Construct final text from the same accumulator that supplied deltas.
        // Tool items remain opaque; item IDs/call IDs are not reconstructed.
        final_response.set("output_text", ajson::jstr(stream->raw_text));
        ajson output = ajson::jarr();
        if (stream->message_started) {
            ajson part = ajson::jobj(), message = ajson::jobj(), content = ajson::jarr();
            part.set("type", ajson::jstr("output_text")); part.set("text", ajson::jstr(stream->raw_text));
            part.set("annotations", ajson::jarr()); content.push(std::move(part));
            message.set("id", ajson::jstr(stream->message_id)); message.set("type", ajson::jstr("message"));
            message.set("status", ajson::jstr(successful ? "completed" : "incomplete")); message.set("role", ajson::jstr("assistant"));
            message.set("content", std::move(content)); output.push(std::move(message));
        }
        if (const auto *items = response.get("output"); successful && items) for (const auto &item : items->arr)
            if (axiom_codex::string_field(item, "type") != "message") output.push(item);
        final_response.set("output", std::move(output));
    }
    completed.set("response", std::move(final_response));
    const char *terminal_event = successful ? "response.completed" :
            std::strcmp(terminal_status, "incomplete") == 0 ? "response.incomplete" : "response.failed";
    if (!live_responses_send_event(stream, terminal_event, std::move(completed))) return false;
    if (!write_http_chunk(stream->fd, "data: [DONE]\n\n")) {
        stream->io_failed = true;
        return false;
    }
    const bool closed = write_all(stream->fd, "0\r\n\r\n", 5u);
    stream->open = false;
    return closed;
}

void close_live_responses_stream(live_responses_stream *stream) {
    if (!stream || !stream->open) return;
    if (stream->codex_wire && !stream->io_failed) {
        if (!stream->terminal_sent) {
            (void)live_responses_send_error(stream, "response stream ended before completion");
        }
        if (!stream->io_failed) (void)write_http_chunk(stream->fd, "data: [DONE]\n\n");
    }
    (void)write_all(stream->fd, "0\r\n\r\n", 5u);
    stream->open = false;
}

struct live_anthropic_stream {
    int fd = -1;
    std::string message_id;
    ajson metadata;
    std::string raw_text;
    size_t emitted_bytes = 0u;
    uint32_t input_tokens = 0u;
    bool open = false;
    bool text_block_started = false;
    bool text_block_closed = false;
};

bool live_anthropic_send_event(
        live_anthropic_stream *stream, const char *event_type, ajson data) {
    if (!stream || !stream->open || !event_type) return false;
    data.set("type", ajson::jstr(event_type));
    std::string body = "event: ";
    body += event_type;
    body += "\ndata: ";
    body += ajson_dumps(data);
    body += "\n\n";
    return write_http_chunk(stream->fd, body);
}

bool start_live_anthropic_stream(
        live_anthropic_stream *stream,
        int fd,
        uint64_t sequence,
        uint32_t input_tokens,
        const std::string &session_id) {
    if (!stream) return false;
    *stream = live_anthropic_stream{};
    stream->fd = fd;
    stream->message_id = "msg-axiom-" + std::to_string(
            static_cast<unsigned long long>(sequence));
    stream->input_tokens = input_tokens;
    if (!write_sse_headers(fd, session_id)) return false;
    stream->open = true;

    ajson usage = ajson::jobj();
    usage.set("input_tokens", ajson::jint(input_tokens));
    usage.set("output_tokens", ajson::jint(0));
    ajson message = ajson::jobj();
    message.set("id", ajson::jstr(stream->message_id));
    message.set("type", ajson::jstr("message"));
    message.set("role", ajson::jstr("assistant"));
    message.set("model", ajson::jstr(kModelId));
    message.set("content", ajson::jarr());
    message.set("stop_reason", ajson::jnull());
    message.set("stop_sequence", ajson::jnull());
    message.set("usage", std::move(usage));
    ajson axiom = ajson::jobj();
    axiom.set("session_id", ajson::jstr(session_id));
    message.set("axiom", std::move(axiom));
    ajson start = ajson::jobj();
    start.set("message", std::move(message));
    return live_anthropic_send_event(stream, "message_start", std::move(start));
}

bool live_anthropic_start_text_block(live_anthropic_stream *stream) {
    if (!stream || !stream->open) return false;
    if (stream->text_block_started) return true;
    ajson block = ajson::jobj();
    block.set("type", ajson::jstr("text"));
    block.set("text", ajson::jstr(""));
    ajson data = ajson::jobj();
    data.set("index", ajson::jint(0));
    data.set("content_block", std::move(block));
    stream->text_block_started = true;
    return live_anthropic_send_event(stream, "content_block_start", std::move(data));
}

bool live_anthropic_flush_text(live_anthropic_stream *stream, bool final) {
    if (!stream || !stream->open) return false;
    const size_t safe_end = live_chat_safe_text_end(stream->raw_text, final);
    if (safe_end <= stream->emitted_bytes) return true;
    if (!live_anthropic_start_text_block(stream)) return false;
    ajson delta = ajson::jobj();
    delta.set("type", ajson::jstr("text_delta"));
    delta.set("text", ajson::jstr(stream->raw_text.substr(
            stream->emitted_bytes, safe_end - stream->emitted_bytes)));
    ajson data = ajson::jobj();
    data.set("index", ajson::jint(0));
    data.set("delta", std::move(delta));
    if (!live_anthropic_send_event(stream, "content_block_delta", std::move(data))) return false;
    stream->emitted_bytes = safe_end;
    return true;
}

bool live_anthropic_emit_token(void *user, const std::string &piece) {
    live_anthropic_stream *stream = static_cast<live_anthropic_stream *>(user);
    if (!stream || !stream->open) return false;
    stream->raw_text += piece;
    return live_anthropic_flush_text(stream, false);
}

bool live_anthropic_close_text_block(live_anthropic_stream *stream) {
    if (!stream || !stream->open || !stream->text_block_started ||
        stream->text_block_closed) return true;
    ajson data = ajson::jobj();
    data.set("index", ajson::jint(0));
    if (!live_anthropic_send_event(stream, "content_block_stop", std::move(data))) return false;
    stream->text_block_closed = true;
    return true;
}

bool live_anthropic_send_tool_calls(
        live_anthropic_stream *stream, const std::vector<ajson> &calls) {
    if (!stream || !stream->open) return false;
    constexpr size_t kArgumentChunkBytes = 192u;
    for (size_t index = 0u; index < calls.size(); ++index) {
        const ajson &call = calls[index];
        const ajson *id = call.get("id");
        const ajson *function = call.get("function");
        const ajson *name = function && function->is_object()
                ? function->get("name") : nullptr;
        const ajson *arguments = function && function->is_object()
                ? function->get("arguments") : nullptr;
        const uint32_t block_index = static_cast<uint32_t>(
                (stream->text_block_started ? 1u : 0u) + index);
        ajson block = ajson::jobj();
        block.set("type", ajson::jstr("tool_use"));
        block.set("id", id ? *id : ajson::jstr("call_axiom"));
        block.set("name", name ? *name : ajson::jstr("tool"));
        block.set("input", ajson::jobj());
        ajson start = ajson::jobj();
        start.set("index", ajson::jint(block_index));
        start.set("content_block", std::move(block));
        if (!live_anthropic_send_event(stream, "content_block_start", std::move(start))) return false;

        const std::string argument_text = arguments && arguments->is_string()
                ? arguments->s : std::string("{}");
        size_t offset = 0u;
        while (offset < argument_text.size()) {
            size_t end = std::min(argument_text.size(), offset + kArgumentChunkBytes);
            while (end < argument_text.size() && end > offset &&
                   (static_cast<unsigned char>(argument_text[end]) & 0xc0u) == 0x80u) {
                --end;
            }
            if (end == offset) end = std::min(argument_text.size(), offset + kArgumentChunkBytes);
            ajson delta = ajson::jobj();
            delta.set("type", ajson::jstr("input_json_delta"));
            delta.set("partial_json", ajson::jstr(argument_text.substr(offset, end - offset)));
            ajson data = ajson::jobj();
            data.set("index", ajson::jint(block_index));
            data.set("delta", std::move(delta));
            if (!live_anthropic_send_event(stream, "content_block_delta", std::move(data))) return false;
            offset = end;
        }
        ajson stop = ajson::jobj();
        stop.set("index", ajson::jint(block_index));
        if (!live_anthropic_send_event(stream, "content_block_stop", std::move(stop))) return false;
    }
    return true;
}

bool live_anthropic_send_error(
        live_anthropic_stream *stream, const std::string &message) {
    if (!stream || !stream->open) return false;
    return live_anthropic_send_event(
            stream, "error", error_json(message, "server_error"));
}

bool finish_live_anthropic_stream(
        live_anthropic_stream *stream,
        const generation_result &result,
        const native_tool_result *tool_result) {
    if (!stream || !stream->open) return false;
    const bool have_tool_calls = tool_result && tool_result->status == "pass" &&
            !tool_result->calls.empty();
    stream->raw_text = tool_result && !tool_result->content.empty()
            ? tool_result->content : visible_model_text(result.text, result.thinking_budget != 0u);
    if (have_tool_calls) {
        const size_t tool_start = stream->raw_text.find("<tool_call>");
        if (tool_start != std::string::npos) {
            stream->raw_text = trim_text(stream->raw_text.substr(0u, tool_start));
        }
    }
    if (!live_anthropic_flush_text(stream, true)) return false;
    if (!live_anthropic_close_text_block(stream)) return false;
    if (have_tool_calls && !live_anthropic_send_tool_calls(stream, tool_result->calls)) return false;

    ajson delta = ajson::jobj();
    delta.set("stop_reason", ajson::jstr(have_tool_calls ? "tool_use" :
                                          (result.finish_reason == "length"
                                                   ? "max_tokens" : "end_turn")));
    delta.set("stop_sequence", ajson::jnull());
    ajson usage = ajson::jobj();
    usage.set("output_tokens", ajson::jint(static_cast<long long>(result.ids.size())));
    ajson message_delta = ajson::jobj();
    message_delta.set("delta", std::move(delta));
    message_delta.set("usage", std::move(usage));
    message_delta.set("axiom", session_proof_json(result));
    const ajson *swarm = stream->metadata.get("axiom_swarm");
    if (swarm) message_delta.set("axiom_swarm", *swarm);
    if (!live_anthropic_send_event(stream, "message_delta", std::move(message_delta))) return false;
    if (!live_anthropic_send_event(stream, "message_stop", ajson::jobj())) return false;
    const bool closed = write_all(stream->fd, "0\r\n\r\n", 5u);
    stream->open = false;
    return closed;
}

void close_live_anthropic_stream(live_anthropic_stream *stream) {
    if (!stream || !stream->open) return;
    (void)write_all(stream->fd, "0\r\n\r\n", 5u);
    stream->open = false;
}

ajson anthropic_response(
        const generation_result &result, uint64_t sequence,
        const native_tool_result *tool_result = nullptr) {
    ajson content = ajson::jarr();
    const bool have_tool_calls = tool_result && tool_result->status == "pass" &&
            !tool_result->calls.empty();
    std::string text = tool_result && !tool_result->content.empty()
            ? tool_result->content : visible_model_text(result.text, result.thinking_budget != 0u);
    if (have_tool_calls) {
        const size_t tool_start = text.find("<tool_call>");
        if (tool_start != std::string::npos) text = trim_text(text.substr(0u, tool_start));
    }
    if (!text.empty()) {
        ajson text_block = ajson::jobj();
        text_block.set("type", ajson::jstr("text"));
        text_block.set("text", ajson::jstr(text));
        content.push(std::move(text_block));
    }
    if (have_tool_calls) {
        for (const ajson &call : tool_result->calls) {
            const ajson *function = call.get("function");
            const ajson *name = function && function->is_object() ? function->get("name") : nullptr;
            const ajson *arguments = function && function->is_object()
                    ? function->get("arguments") : nullptr;
            ajson input = ajson::jobj();
            if (arguments && arguments->is_string()) {
                std::string parse_error;
                ajson parsed;
                if (ajson_parse(arguments->s, parsed, parse_error) && parsed.is_object()) {
                    input = std::move(parsed);
                }
            }
            ajson block = ajson::jobj();
            block.set("type", ajson::jstr("tool_use"));
            const ajson *id = call.get("id");
            block.set("id", id ? *id : ajson::jstr("call_axiom_qwen"));
            block.set("name", name ? *name : ajson::jstr("tool"));
            block.set("input", std::move(input));
            content.push(std::move(block));
        }
    }

    ajson usage = ajson::jobj();
    usage.set("input_tokens", ajson::jint(result.prompt_tokens));
    usage.set("output_tokens", ajson::jint(static_cast<long long>(result.ids.size())));

    char id[64]{};
    std::snprintf(id, sizeof(id), "msg-axiom-%llu", static_cast<unsigned long long>(sequence));
    ajson root = ajson::jobj();
    root.set("id", ajson::jstr(id));
    root.set("type", ajson::jstr("message"));
    root.set("role", ajson::jstr("assistant"));
    root.set("model", ajson::jstr(kModelId));
    root.set("content", std::move(content));
    root.set("stop_reason", ajson::jstr(have_tool_calls ? "tool_use" :
                                         (result.finish_reason == "length" ?
                                          "max_tokens" : "end_turn")));
    root.set("stop_sequence", ajson::jnull());
    root.set("usage", std::move(usage));
    ajson axiom = ajson::jobj();
    attach_session_proof(&axiom, result);
    attach_decode_proof(&axiom, result);
    axiom.set("backend", ajson::jstr(kBackendId));
    axiom.set("session_persistence", ajson::jbool(true));
    root.set("axiom", std::move(axiom));
    return root;
}

ajson responses_response(
        const generation_result &result, uint64_t sequence,
        const std::string &reasoning_effort,
        const ajson *normalized_tools,
        const native_tool_result *tool_result = nullptr, bool codex_provider = false) {
    const bool have_tool_calls = tool_result && tool_result->status == "pass" &&
            !tool_result->calls.empty();
    const std::string text = tool_result && tool_result->status == "pass"
            ? tool_result->content : visible_model_text(result.text, result.thinking_budget != 0u);
    const char *terminal_status = codex_provider
            ? codex_response_status(result, have_tool_calls, text) : "completed";
    const bool successful = std::strcmp(terminal_status, "completed") == 0;
    char message_id[64]{};
    std::snprintf(message_id, sizeof(message_id), "msg-axiom-%llu",
                  static_cast<unsigned long long>(sequence));
    ajson output = ajson::jarr();
    if (!text.empty()) {
        ajson text_content = ajson::jobj();
        text_content.set("type", ajson::jstr("output_text"));
        text_content.set("text", ajson::jstr(text));
        text_content.set("annotations", ajson::jarr());
        ajson message = ajson::jobj();
        message.set("id", ajson::jstr(message_id));
        message.set("type", ajson::jstr("message"));
        message.set("status", ajson::jstr(successful ? "completed" : "incomplete"));
        message.set("role", ajson::jstr("assistant"));
        ajson content = ajson::jarr();
        content.push(std::move(text_content));
        message.set("content", std::move(content));
        output.push(std::move(message));
    }
    if (have_tool_calls && successful) {
        for (const ajson &call : tool_result->calls) {
            const ajson *function = call.get("function");
            const ajson *name = function && function->is_object() ? function->get("name") : nullptr;
            const ajson *arguments = function && function->is_object()
                    ? function->get("arguments") : nullptr;
            const ajson *id = call.get("id");
            const ajson *definition = name && normalized_tools
                    ? tool_function_by_name(*normalized_tools, name->s) : nullptr;
            const ajson *response_type = definition ? definition->get("axiom_responses_type") : nullptr;
            const bool custom = response_type && response_type->is_string() &&
                    response_type->s == "custom";
            ajson item = ajson::jobj();
            item.set("id", id ? *id : ajson::jstr("fc-axiom"));
            item.set("status", ajson::jstr("completed"));
            item.set("call_id", id ? *id : ajson::jstr("call_axiom"));
            item.set("name", name ? *name : ajson::jstr("tool"));
            if (custom) {
                item.set("type", ajson::jstr("custom_tool_call"));
                std::string input;
                if (arguments && arguments->is_string()) {
                    ajson parsed;
                    std::string parse_error;
                    if (ajson_parse(arguments->s, parsed, parse_error) && parsed.is_object()) {
                        const ajson *value = parsed.get("input");
                        input = value && value->is_string() ? value->s : arguments->s;
                    } else {
                        input = arguments->s;
                    }
                }
                item.set("input", ajson::jstr(std::move(input)));
            } else {
                item.set("type", ajson::jstr("function_call"));
                item.set("arguments", arguments && arguments->is_string()
                                         ? *arguments : ajson::jstr("{}"));
            }
            output.push(std::move(item));
        }
    }

    ajson usage = ajson::jobj();
    usage.set("input_tokens", ajson::jint(result.prompt_tokens));
    usage.set("output_tokens", ajson::jint(static_cast<long long>(result.ids.size())));
    usage.set("total_tokens", ajson::jint(
            result.prompt_tokens + static_cast<uint32_t>(result.ids.size())));
    ajson reasoning = ajson::jobj();
    reasoning.set("effort", ajson::jstr(reasoning_effort));

    char id[64]{};
    std::snprintf(id, sizeof(id), "resp-axiom-%llu",
                  static_cast<unsigned long long>(sequence));
    ajson root = ajson::jobj();
    root.set("id", ajson::jstr(id));
    root.set("object", ajson::jstr("response"));
    root.set("created_at", ajson::jint(static_cast<long long>(std::time(nullptr))));
    root.set("status", ajson::jstr("completed"));
    if (codex_provider) apply_codex_response_status(&root, terminal_status);
    root.set("model", ajson::jstr(kModelId));
    root.set("output", std::move(output));
    root.set("output_text", ajson::jstr(text));
    root.set("usage", std::move(usage));
    root.set("reasoning", std::move(reasoning));
    root.set("parallel_tool_calls", ajson::jbool(true));
    ajson backend = ajson::jobj();
    backend.set("name", ajson::jstr(kBackendId));
    attach_session_proof(&backend, result);
    attach_decode_proof(&backend, result);
    backend.set("graph_cycles", ajson::jint(static_cast<long long>(result.graph_cycles)));
    backend.set("graph_emitted_tokens", ajson::jint(
            static_cast<long long>(result.graph_emitted_tokens)));
    backend.set("canonical_tail_tokens", ajson::jint(
            static_cast<long long>(result.canonical_tail_tokens)));
    backend.set("proposed_draft_tokens", ajson::jint(
            static_cast<long long>(result.proposed_draft_tokens)));
    backend.set("eligible_draft_tokens", ajson::jint(
            static_cast<long long>(result.eligible_draft_tokens)));
    backend.set("accepted_draft_tokens", ajson::jint(
            static_cast<long long>(result.accepted_draft_tokens)));
    backend.set("acceptance", ajson::jnum(result.acceptance));
    backend.set("decode_seconds", ajson::jnum(result.decode_seconds));
    backend.set("decode_tokens_per_second", ajson::jnum(result.decode_tokens_per_second));
    backend.set("visible_tokens_per_second", ajson::jnum(result.visible_tokens_per_second));
    backend.set("thinking_tokens", ajson::jint(result.thinking_tokens));
    backend.set("visible_output_tokens", ajson::jint(result.visible_output_tokens));
    backend.set("thinking_budget", ajson::jint(result.thinking_budget));
    backend.set("reasoning_effort", ajson::jstr(reasoning_effort));
    backend.set("context_compacted", ajson::jbool(result.context_compacted));
    backend.set("original_prompt_tokens", ajson::jint(result.original_prompt_tokens));
    backend.set("tool_calls", ajson::jint(
            static_cast<long long>(have_tool_calls ? tool_result->calls.size() : 0u)));
    root.set("axiom", std::move(backend));
    return root;
}

[[maybe_unused]] std::string responses_stream_body(const ajson &response) {
    std::string body;
    uint64_t sequence = 0u;
    auto event = [&body, &sequence](const char *type, ajson data) {
        data.set("type", ajson::jstr(type));
        data.set("sequence_number", ajson::jint(static_cast<long long>(sequence++)));
        body += "event: ";
        body += type;
        body += "\ndata: ";
        body += ajson_dumps(data);
        body += "\n\n";
    };

    ajson created = ajson::jobj();
    ajson created_response = response;
    created_response.set("status", ajson::jstr("in_progress"));
    created.set("response", std::move(created_response));
    event("response.created", std::move(created));

    const ajson *output = response.get("output");
    if (output && output->is_array()) {
        uint32_t output_index = 0u;
        for (const ajson &item : output->arr) {
            const ajson *type = item.get("type");
            const std::string item_type = type && type->is_string() ? type->s : "";
            const ajson *item_id = item.get("id");
            if (item_type == "message") {
                const ajson *content = item.get("content");
                const ajson *text = content && content->is_array() && !content->arr.empty()
                        ? content->arr[0].get("text") : nullptr;
                if (text && text->is_string()) {
                    ajson delta = ajson::jobj();
                    delta.set("item_id", item_id ? *item_id : ajson::jstr("msg-axiom"));
                    delta.set("output_index", ajson::jint(output_index));
                    delta.set("content_index", ajson::jint(0));
                    delta.set("delta", *text);
                    event("response.output_text.delta", std::move(delta));
                    ajson done = ajson::jobj();
                    done.set("item_id", item_id ? *item_id : ajson::jstr("msg-axiom"));
                    done.set("output_index", ajson::jint(output_index));
                    done.set("content_index", ajson::jint(0));
                    done.set("text", *text);
                    event("response.output_text.done", std::move(done));
                }
            } else if (item_type == "function_call" || item_type == "custom_tool_call") {
                ajson done = ajson::jobj();
                done.set("output_index", ajson::jint(output_index));
                done.set("item", item);
                event(item_type == "custom_tool_call"
                              ? "response.output_item.done" : "response.output_item.done",
                      std::move(done));
            }
            ++output_index;
        }
    }
    ajson completed = ajson::jobj();
    completed.set("response", response);
    event("response.completed", std::move(completed));
    body += "data: [DONE]\n\n";
    return body;
}


uint64_t process_rss_bytes() {
    FILE *file = std::fopen("/proc/self/statm", "r");
    if (!file) return 0u;
    unsigned long total_pages = 0u;
    unsigned long resident_pages = 0u;
    const int scanned = std::fscanf(file, "%lu %lu", &total_pages, &resident_pages);
    std::fclose(file);
    if (scanned != 2) return 0u;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 0u;
    return static_cast<uint64_t>(resident_pages) * static_cast<uint64_t>(page_size);
}

uint64_t process_peak_rss_bytes() {
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0u;
    /* Linux reports ru_maxrss in KiB. Axiom's production target is Linux. */
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024u;
}

resource_snapshot capture_resource_snapshot(const server_state *state) {
    resource_snapshot snapshot;
    snapshot.process_rss_bytes = process_rss_bytes();
    snapshot.process_peak_rss_bytes = process_peak_rss_bytes();
    size_t free_bytes = 0u;
    size_t total_bytes = 0u;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        snapshot.gpu_available = true;
        snapshot.gpu_free_bytes = static_cast<uint64_t>(free_bytes);
        snapshot.gpu_total_bytes = static_cast<uint64_t>(total_bytes);
    }
    if (state && state->kv_tier) {
        axiom_qwen38_kv_tier_info info{};
        info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
        if (axiom_qwen38_kv_tier_info_get(state->kv_tier, &info) == AXIOM_OK) {
            snapshot.kv_available = true;
            snapshot.kv_submitted_reads = info.submitted_reads;
            snapshot.kv_submitted_writes = info.submitted_writes;
            snapshot.kv_completed_bytes = info.completed_bytes;
            snapshot.kv_io_errors = info.io_errors;
            snapshot.kv_committed_tokens = info.committed_tokens;
        }
    }
    return snapshot;
}

uint64_t counter_delta(uint64_t after, uint64_t before) {
    return after >= before ? after - before : 0u;
}

ajson resource_snapshot_json(const resource_snapshot &snapshot) {
    ajson root = ajson::jobj();
    root.set("process_rss_bytes", ajson::jint(
            static_cast<long long>(snapshot.process_rss_bytes)));
    root.set("process_peak_rss_bytes", ajson::jint(
            static_cast<long long>(snapshot.process_peak_rss_bytes)));
    root.set("gpu_available", ajson::jbool(snapshot.gpu_available));
    if (snapshot.gpu_available) {
        root.set("gpu_total_bytes", ajson::jint(
                static_cast<long long>(snapshot.gpu_total_bytes)));
        root.set("gpu_free_bytes", ajson::jint(
                static_cast<long long>(snapshot.gpu_free_bytes)));
        root.set("gpu_used_bytes", ajson::jint(static_cast<long long>(
                snapshot.gpu_total_bytes >= snapshot.gpu_free_bytes
                        ? snapshot.gpu_total_bytes - snapshot.gpu_free_bytes : 0u)));
    } else {
        root.set("gpu_total_bytes", ajson::jnull());
        root.set("gpu_free_bytes", ajson::jnull());
        root.set("gpu_used_bytes", ajson::jnull());
    }
    root.set("kv_available", ajson::jbool(snapshot.kv_available));
    root.set("kv_submitted_reads", ajson::jint(
            static_cast<long long>(snapshot.kv_submitted_reads)));
    root.set("kv_submitted_writes", ajson::jint(
            static_cast<long long>(snapshot.kv_submitted_writes)));
    root.set("kv_completed_bytes", ajson::jint(
            static_cast<long long>(snapshot.kv_completed_bytes)));
    root.set("kv_io_errors", ajson::jint(static_cast<long long>(snapshot.kv_io_errors)));
    root.set("kv_committed_tokens", ajson::jint(snapshot.kv_committed_tokens));
    return root;
}

void record_token_telemetry(
        server_state *state, uint64_t sequence, const std::string &session_id,
        const generation_result &result,
        uint32_t context_window, double request_seconds, const resource_snapshot &before,
        const resource_snapshot &after) {
    if (!state) return;
    std::lock_guard<std::mutex> lock(state->telemetry_mutex);
    token_telemetry telemetry;
    telemetry.available = true;
    telemetry.request_sequence = sequence;
    telemetry.session_id = session_id;
    telemetry.prompt_tokens = result.prompt_tokens;
    telemetry.completion_tokens = static_cast<uint32_t>(result.ids.size());
    telemetry.thinking_tokens = result.thinking_tokens;
    telemetry.visible_output_tokens = result.visible_output_tokens;
    telemetry.total_tokens = telemetry.prompt_tokens + telemetry.completion_tokens;
    telemetry.context_window = context_window;
    telemetry.thinking_budget = result.thinking_budget;
    telemetry.reasoning_effort = result.reasoning_effort;
    telemetry.original_prompt_tokens = result.original_prompt_tokens;
    telemetry.request_seconds = request_seconds;
    telemetry.decode_seconds = result.decode_seconds;
    telemetry.decode_tokens_per_second = result.decode_tokens_per_second;
    telemetry.visible_tokens_per_second = result.visible_tokens_per_second;
    telemetry.prefill_seconds = result.prefill_seconds;
    telemetry.session_acquire_seconds = result.session_acquire_seconds;
    telemetry.ttft_seconds = result.ttft_seconds;
    telemetry.speculative_context_tokens = result.speculative_context_tokens;
    telemetry.prefix_hit_tokens = result.prefix_hit_tokens;
    telemetry.suffix_prefill_tokens = result.suffix_prefill_tokens;
    telemetry.graph_cycles = result.graph_cycles;
    telemetry.graph_emitted_tokens = result.graph_emitted_tokens;
    telemetry.scalar_tail_tokens = result.scalar_tail_tokens;
    telemetry.accepted_draft_tokens = result.accepted_draft_tokens;
    telemetry.eligible_draft_tokens = result.eligible_draft_tokens;
    telemetry.speculative_acceptance = result.acceptance;
    telemetry.draft_confidence_samples = result.draft_confidence_samples;
    telemetry.draft_confidence_accepted_samples =
            result.draft_confidence_accepted_samples;
    telemetry.draft_confidence_rejected_samples =
            result.draft_confidence_rejected_samples;
    telemetry.draft_confidence_sum = result.draft_confidence_sum;
    telemetry.draft_confidence_accepted_sum =
            result.draft_confidence_accepted_sum;
    telemetry.draft_confidence_rejected_sum =
            result.draft_confidence_rejected_sum;
    telemetry.draft_confidence_brier_sum = result.draft_confidence_brier_sum;
    telemetry.draft_confidence_min = result.draft_confidence_min;
    telemetry.draft_confidence_max = result.draft_confidence_max;
    telemetry.draft_confidence_bin_samples =
            result.draft_confidence_bin_samples;
    telemetry.draft_confidence_bin_accepted =
            result.draft_confidence_bin_accepted;
    telemetry.draft_confidence_bin_sum = result.draft_confidence_bin_sum;
    telemetry.decode_path = result.decode_path;
    telemetry.fallback_reason = result.fallback_reason;
    telemetry.speculative_mode_requested = result.speculative_mode_requested;
    telemetry.speculative_mode_effective = result.speculative_mode_effective;
    telemetry.graph_executor_reused = result.graph_executor_reused;
    telemetry.context_compacted = result.context_compacted;
    telemetry.resources_before = before;
    telemetry.resources_after = after;
    state->last_telemetry = std::move(telemetry);
}

ajson ops_usage_status(const server_state *state) {
    ajson root = ajson::jobj();
    root.set("schema", ajson::jstr("axiom_usage_status_v1"));
    root.set("status", ajson::jstr("waiting"));
    root.set("model", ajson::jstr(kModelId));
    if (!state) return root;
    std::lock_guard<std::mutex> lock(state->telemetry_mutex);
    if (!state->last_telemetry.available) return root;

    const token_telemetry &telemetry = state->last_telemetry;
    root.set("status", ajson::jstr("pass"));
    root.set("request_sequence", ajson::jint(
            static_cast<long long>(telemetry.request_sequence)));
    root.set("session_id", ajson::jstr(telemetry.session_id));
    root.set("prompt_tokens", ajson::jint(telemetry.prompt_tokens));
    root.set("completion_tokens", ajson::jint(telemetry.completion_tokens));
    root.set("thinking_tokens", ajson::jint(telemetry.thinking_tokens));
    root.set("visible_output_tokens", ajson::jint(telemetry.visible_output_tokens));
    root.set("total_tokens", ajson::jint(telemetry.total_tokens));
    root.set("context_window", ajson::jint(telemetry.context_window));
    root.set("thinking_budget", ajson::jint(telemetry.thinking_budget));
    root.set("reasoning_effort", ajson::jstr(telemetry.reasoning_effort));
    root.set("original_prompt_tokens", ajson::jint(telemetry.original_prompt_tokens));
    root.set("request_seconds", ajson::jnum(telemetry.request_seconds));
    root.set("decode_seconds", ajson::jnum(telemetry.decode_seconds));
    root.set("decode_tokens_per_second", ajson::jnum(
            telemetry.decode_tokens_per_second));
    root.set("visible_tokens_per_second", ajson::jnum(
            telemetry.visible_tokens_per_second));
    root.set("decode_path", ajson::jstr(telemetry.decode_path));
    root.set("speculative_mode_requested", ajson::jstr(
            telemetry.speculative_mode_requested));
    root.set("speculative_mode_effective", ajson::jstr(
            telemetry.speculative_mode_effective));
    root.set("fallback_reason", telemetry.fallback_reason.empty()
            ? ajson::jnull() : ajson::jstr(telemetry.fallback_reason));
    if (telemetry.decode_path == "speculative_graph" ||
        telemetry.decode_path == "hybrid_graph_scalar" ||
        telemetry.decode_path == "native_mtp_graph" ||
        telemetry.decode_path == "native_mtp_graph_scalar") {
        root.set("graph_executor_reused", ajson::jbool(
                telemetry.graph_executor_reused));
    }
    root.set("speculative_context_tokens", ajson::jint(
            telemetry.speculative_context_tokens));
    root.set("graph_cycles", ajson::jint(
            static_cast<long long>(telemetry.graph_cycles)));
    root.set("graph_emitted_tokens", ajson::jint(
            static_cast<long long>(telemetry.graph_emitted_tokens)));
    root.set("scalar_tail_tokens", ajson::jint(
            static_cast<long long>(telemetry.scalar_tail_tokens)));
    root.set("accepted_draft_tokens", ajson::jint(
            static_cast<long long>(telemetry.accepted_draft_tokens)));
    root.set("eligible_draft_tokens", ajson::jint(
            static_cast<long long>(telemetry.eligible_draft_tokens)));
    root.set("speculative_acceptance", ajson::jnum(
            telemetry.speculative_acceptance));
    generation_result confidence_view;
    confidence_view.draft_confidence_samples = telemetry.draft_confidence_samples;
    confidence_view.draft_confidence_accepted_samples =
            telemetry.draft_confidence_accepted_samples;
    confidence_view.draft_confidence_rejected_samples =
            telemetry.draft_confidence_rejected_samples;
    confidence_view.draft_confidence_sum = telemetry.draft_confidence_sum;
    confidence_view.draft_confidence_accepted_sum =
            telemetry.draft_confidence_accepted_sum;
    confidence_view.draft_confidence_rejected_sum =
            telemetry.draft_confidence_rejected_sum;
    confidence_view.draft_confidence_brier_sum =
            telemetry.draft_confidence_brier_sum;
    confidence_view.draft_confidence_min = telemetry.draft_confidence_min;
    confidence_view.draft_confidence_max = telemetry.draft_confidence_max;
    confidence_view.draft_confidence_bin_samples =
            telemetry.draft_confidence_bin_samples;
    confidence_view.draft_confidence_bin_accepted =
            telemetry.draft_confidence_bin_accepted;
    confidence_view.draft_confidence_bin_sum = telemetry.draft_confidence_bin_sum;
    root.set("draft_confidence", draft_confidence_json(confidence_view));
    root.set("prefix_hit_tokens", ajson::jint(telemetry.prefix_hit_tokens));
    root.set("suffix_prefill_tokens", ajson::jint(telemetry.suffix_prefill_tokens));
    root.set("prefill_seconds", ajson::jnum(telemetry.prefill_seconds));
    root.set("session_acquire_seconds", ajson::jnum(telemetry.session_acquire_seconds));
    root.set("ttft_seconds", ajson::jnum(telemetry.ttft_seconds));
    root.set("context_compacted", ajson::jbool(telemetry.context_compacted));
    root.set("resources_after", resource_snapshot_json(telemetry.resources_after));

    ajson delta = ajson::jobj();
    delta.set("kv_submitted_reads", ajson::jint(static_cast<long long>(counter_delta(
            telemetry.resources_after.kv_submitted_reads,
            telemetry.resources_before.kv_submitted_reads))));
    delta.set("kv_submitted_writes", ajson::jint(static_cast<long long>(counter_delta(
            telemetry.resources_after.kv_submitted_writes,
            telemetry.resources_before.kv_submitted_writes))));
    delta.set("kv_completed_bytes", ajson::jint(static_cast<long long>(counter_delta(
            telemetry.resources_after.kv_completed_bytes,
            telemetry.resources_before.kv_completed_bytes))));
    delta.set("kv_io_errors", ajson::jint(static_cast<long long>(counter_delta(
            telemetry.resources_after.kv_io_errors,
            telemetry.resources_before.kv_io_errors))));
    root.set("resources_delta", std::move(delta));
    return root;
}

ajson persistence_profile_snapshot(const server_state *state) {
    if (!state) return ajson::jnull();
    std::lock_guard<std::mutex> lock(state->persistence_profile_mutex);
    if (state->persistence_profile_jobs == 0u) return ajson::jnull();
    ajson result = ajson::jobj();
    result.set("scope", ajson::jstr("last_completed_deferred_job_not_current_request"));
    result.set("jobs_observed", ajson::jint(state->persistence_profile_jobs));
    result.set("committed_tokens", ajson::jint(state->persistence_profile_tokens));
    result.set("status_code", ajson::jint(state->persistence_profile_status));
    const char *names[] = {"replay_ms", "kv_flush_ms", "manifest_ms", "transaction_cleanup_ms",
                           "model_reset_ms", "graph_recycle_ms", "total_ms"};
    for (size_t index = 0; index < state->persistence_last_ms.size(); ++index) {
        result.set(names[index], ajson::jnum(state->persistence_last_ms[index]));
    }
    return result;
}

ajson ops_runtime_status(const server_state *state) {
    ajson root = ajson::jobj();
    const session_status_snapshot session = capture_session_status(state);
    uint64_t active_request_sequence = 0u;
    std::string active_request_session_id;
    std::shared_ptr<axiom::qwen38::request_progress> progress;
    if (state) {
        std::lock_guard<std::mutex> lock(state->active_request_mutex);
        active_request_sequence = state->active_request_sequence;
        active_request_session_id = state->active_request_session_id;
        progress = state->active_request_progress;
    }
    root.set("schema", ajson::jstr("axiom_runtime_status_v1"));
    root.set("persistence_last_job", persistence_profile_snapshot(state));
    root.set("status", ajson::jstr(state && state->healthy.load() ? "pass" : "fail"));
    root.set("model", ajson::jstr(kModelId));
    root.set("backend", ajson::jstr(kBackendId));
    root.set("loaded", ajson::jbool(state && state->healthy.load()));
    root.set("single_session", ajson::jbool(false));
    root.set("http_multi_session", ajson::jbool(true));
    root.set("max_http_sessions", ajson::jint(0));
    root.set("http_sessions_unlimited", ajson::jbool(true));
    root.set("http_workers_active", ajson::jint(state
            ? static_cast<long long>(state->active_http_workers.load()) : 0ll));
    root.set("http_worker_admission", ajson::jstr("unbounded"));
    root.set("generation_queue_capacity", ajson::jint(kSwarmQueueDepth));
    root.set("session_id_auto_generated", ajson::jbool(true));
    root.set("session_id_shared_default", ajson::jbool(false));
    root.set("session_id_header", ajson::jstr(kSessionHeader));
    root.set("session_leases_active", ajson::jint(state
            ? static_cast<long long>(state->session_leases_active.load()) : 0ll));
    root.set("session_leases_peak", ajson::jint(state
            ? static_cast<long long>(state->session_leases_peak.load()) : 0ll));
    root.set("generation_scheduler", ajson::jstr("swarm_queue_serialized_native_model"));
    root.set("generation_busy", ajson::jbool(active_request_sequence != 0u));
    root.set("active_request_progress", request_progress_json(progress));
    if (active_request_sequence != 0u) {
        root.set("active_request_sequence", ajson::jint(
                static_cast<long long>(active_request_sequence)));
        root.set("active_session_id", ajson::jstr(active_request_session_id));
    } else {
        root.set("active_request_sequence", ajson::jnull());
        root.set("active_session_id", ajson::jnull());
    }
    root.set("decode_chunk_cycles", ajson::jint(kDeviceChunkCycles));
    root.set("sse_batch_cycles", ajson::jint(
            state ? static_cast<long long>(state->sse_batch_cycles) : 0ll));
    root.set("speculative_graph", ajson::jbool(
            speculative_graph_loaded(state)));
    root.set("speculative_mode_default", ajson::jstr("auto"));
    root.set("speculative_modes", speculative_modes_json());
    root.set("hybrid_graph_scalar", ajson::jbool(
            state && state->streaming_kv && speculative_graph_loaded(state)));
    root.set("speculative_context_tokens", ajson::jint(
            state ? state->speculative_context_tokens : 0u));
    attach_speculative_contract(&root, state);
    attach_speculative_engine_status(&root, state);
    root.set("graph_executor_resident", ajson::jbool(
            state && state->graph_executor_resident.load(std::memory_order_acquire)));
    root.set("graph_executor_creates", ajson::jint(state
            ? static_cast<long long>(state->graph_executor_creates.load()) : 0ll));
    root.set("graph_executor_reuses", ajson::jint(state
            ? static_cast<long long>(state->graph_executor_reuses.load()) : 0ll));
    root.set("graph_executor_recycles", ajson::jint(state
            ? static_cast<long long>(state->graph_executor_recycles.load()) : 0ll));
    root.set("graph_executor_recycle_failures", ajson::jint(state
            ? static_cast<long long>(state->graph_executor_recycle_failures.load()) : 0ll));
    root.set("swarm_enabled", ajson::jbool(true));
    root.set("swarm_execution_width", ajson::jint(1));
    root.set("swarm_queue_capacity", ajson::jint(kSwarmQueueDepth));
    root.set("session_persistence", ajson::jbool(
            session_persistence_enabled(state)));
    root.set("session_manifest_format", ajson::jstr(
            state && state->session_store.manifest_version() == 2u
            ? "AXQ38SM1/v2-sha256" : "AXQ38SM1/v1"));
    root.set("session_id", ajson::jstr(session.session_id));
    root.set("session_namespace", ajson::jstr(session.namespace_id));
    root.set("session_generation", ajson::jint(
            static_cast<long long>(session.generation)));
    root.set("session_restore_count", ajson::jint(
            state ? static_cast<long long>(state->session_restore_count.load()) : 0ll));
    root.set("session_manifest_writes", ajson::jint(
            state ? static_cast<long long>(state->session_manifest_writes.load()) : 0ll));
    root.set("session_persistence_mode", ajson::jstr("async_ordered_commit"));
    root.set("session_persistence_state", ajson::jstr(
            state && state->session_persistence_failures.load() != 0u ? "error" :
            state && state->session_persistence_pending.load() != 0u ? "pending" :
            "committed"));
    root.set("session_persistence_pending", ajson::jint(state
            ? static_cast<long long>(state->session_persistence_pending.load()) : 0ll));
    root.set("session_persistence_queued", ajson::jint(state
            ? static_cast<long long>(state->session_persistence_queued.load()) : 0ll));
    root.set("session_persistence_completed", ajson::jint(state
            ? static_cast<long long>(state->session_persistence_completed.load()) : 0ll));
    root.set("session_persistence_failures", ajson::jint(state
            ? static_cast<long long>(state->session_persistence_failures.load()) : 0ll));
    root.set("session_gc_runs", ajson::jint(state
            ? static_cast<long long>(state->session_gc_runs.load()) : 0ll));
    root.set("session_generation_gc_runs", ajson::jint(state
            ? static_cast<long long>(state->session_generation_gc_runs.load()) : 0ll));
    root.set("session_lifecycle_gc_runs", ajson::jint(state
            ? static_cast<long long>(state->session_lifecycle_gc_runs.load()) : 0ll));
    root.set("session_gc_removed_files", ajson::jint(state
            ? static_cast<long long>(state->session_gc_removed_files.load()) : 0ll));
    root.set("session_gc_removed_sessions", ajson::jint(state
            ? static_cast<long long>(state->session_gc_removed_sessions.load()) : 0ll));
    root.set("session_gc_reclaimed_bytes", ajson::jint(state
            ? static_cast<long long>(state->session_gc_reclaimed_bytes.load()) : 0ll));
    root.set("session_gc_scanned_sessions", ajson::jint(state
            ? static_cast<long long>(state->session_gc_scanned_sessions.load()) : 0ll));
    root.set("session_gc_protected_sessions", ajson::jint(state
            ? static_cast<long long>(state->session_gc_protected_sessions.load()) : 0ll));
    root.set("session_gc_unsafe_namespaces", ajson::jint(state
            ? static_cast<long long>(state->session_gc_unsafe_namespaces.load()) : 0ll));
    root.set("session_gc_last_run_unix", ajson::jint(state
            ? static_cast<long long>(state->session_gc_last_run_unix.load()) : 0ll));
    root.set("session_gc_last_duration_ms", ajson::jint(state
            ? static_cast<long long>(state->session_gc_last_duration_ms.load()) : 0ll));
    root.set("session_gc_ttl_seconds", ajson::jint(state
            ? static_cast<long long>(state->session_ttl_seconds) : 0ll));
    root.set("session_gc_max_namespaces", ajson::jint(state
            ? static_cast<long long>(state->session_max_namespaces) : 0ll));
    root.set("session_gc_interval_seconds", ajson::jint(state
            ? static_cast<long long>(state->session_gc_interval_seconds) : 0ll));
    root.set("session_gc_failures", ajson::jint(state
            ? static_cast<long long>(state->session_gc_failures.load()) : 0ll));
    root.set("session_gc_execution", ajson::jstr("async_maintenance_worker"));
    root.set("session_gc_active", ajson::jbool(
            state && state->lifecycle_gc_active.load(std::memory_order_acquire)));
    root.set("session_gc_deferred", ajson::jint(state
            ? static_cast<long long>(state->lifecycle_gc_deferred.load()) : 0ll));
    root.set("session_gc_state", ajson::jstr(
            state && (state->session_gc_failures.load() != 0u ||
                      state->session_gc_unsafe_namespaces.load() != 0u)
                    ? "degraded" : "pass"));
    root.set("auth_disabled", ajson::jbool(true));
    root.set("no_think_default", ajson::jbool(state && state->no_think));
    root.set("default_profile", ajson::jstr(
            state && state->no_think ? "ultra-fast" : "medium"));
    root.set("default_reasoning_effort", ajson::jstr(
            state && state->no_think ? "ultra-fast" : "medium"));
    root.set("max_thinking_budget", ajson::jint(
            axiom::reasoning::kMaxThinkingBudgetTokens));
    root.set("reasoning_efforts", reasoning_efforts_json());
    root.set("reasoning_profiles", reasoning_profiles_json());
    root.set("yarn_enabled", ajson::jbool(!env_disabled("AXIOM_QWEN38_YARN")));
    root.set("yarn_factor", ajson::jnum(4.0));
    root.set("yarn_original_context", ajson::jint(262144));
    root.set("yarn_target_context", ajson::jint(state ? state->max_context : 0u));
    root.set("context_window_default", ajson::jint(
            state ? state->default_context : kDefaultRequestContext));
    root.set("context_window_max", ajson::jint(
            state ? state->max_context : kMaxNativeContext));
    root.set("max_context", ajson::jint(state ? state->max_context : 0u));
    root.set("context_global_max", ajson::jint(state ? state->max_context : 0u));
    root.set("context_compaction_enabled", ajson::jbool(state && state->context_compaction));
    root.set("context_compaction_boundary", ajson::jint(kContextCompactionBoundary));
    root.set("context_compaction_active_limit", ajson::jint(
            kContextCompactionBoundary - kContextCompactionMaxOutputReserve -
                    kContextCompactionBuffer));
    root.set("native_qwen_tool_calls", ajson::jbool(true));
    root.set("native_deferred_tools", ajson::jbool(true));
    root.set("responses_tool_search", ajson::jbool(true));
    root.set("openai_tools", ajson::jbool(true));
    root.set("anthropic_tool_use", ajson::jbool(true));
    root.set("kv_mode", ajson::jstr(state && state->streaming_kv ? "paged_runtime" : "disabled"));
    root.set("session_cold_page_reads", ajson::jint(
            state ? static_cast<long long>(state->session_cold_page_reads.load()) : 0ll));
    return root;
}

ajson ops_kv_status(const server_state *state) {
    ajson root = ajson::jobj();
    const session_status_snapshot session = capture_session_status(state);
    const bool tier_enabled = state && (state->streaming_kv || state->kv_tier);
    root.set("schema", ajson::jstr("axiom_kv_status_v1"));
    root.set("status", ajson::jstr(tier_enabled ? "pass" : "disabled"));
    root.set("enabled", ajson::jbool(tier_enabled));
    root.set("mode", ajson::jstr(state && state->streaming_kv ? "paged_runtime" :
                                (state && state->kv_tier ? "durable_mirror" : "disabled")));
    root.set("paged_runtime", ajson::jbool(state && state->streaming_kv));
    root.set("nvme_tier_object", ajson::jbool(state && state->kv_tier != nullptr));
    root.set("page_tokens", ajson::jint(AXIOM_QWEN38_KV_TIER_PAGE_TOKENS));
    root.set("max_context", ajson::jint(state ? state->max_context : 0u));
    root.set("logical_pages", ajson::jint(state
            ? (state->max_context + AXIOM_QWEN38_KV_TIER_PAGE_TOKENS - 1u) /
                    AXIOM_QWEN38_KV_TIER_PAGE_TOKENS
            : 0u));
    root.set("temporal_hot_path", ajson::jbool(state && state->temporal_hot));
    root.set("speculative_context_tokens", ajson::jint(
            state ? state->speculative_context_tokens : 0u));
    root.set("speculative_context_pages", ajson::jint(state
            ? state->speculative_context_tokens /
                    AXIOM_QWEN38_KV_TIER_PAGE_TOKENS
            : 0u));
    root.set("session_persistence", ajson::jbool(
            session_persistence_enabled(state)));
    root.set("session_leases_active", ajson::jint(state
            ? static_cast<long long>(state->session_leases_active.load()) : 0ll));
    root.set("session_leases_peak", ajson::jint(state
            ? static_cast<long long>(state->session_leases_peak.load()) : 0ll));
    root.set("session_id", ajson::jstr(session.session_id));
    root.set("session_namespace", ajson::jstr(session.namespace_id));
    root.set("session_generation", ajson::jint(
            static_cast<long long>(session.generation)));
    root.set("session_manifest_committed", ajson::jbool(
            session.manifest_committed));
    root.set("session_manifest_committed_tokens", ajson::jint(
            session.committed_tokens));
    root.set("session_tier_path", ajson::jstr(session.tier_path));
    root.set("session_restore_count", ajson::jint(
            state ? static_cast<long long>(state->session_restore_count.load()) : 0ll));
    root.set("session_manifest_writes", ajson::jint(
            state ? static_cast<long long>(state->session_manifest_writes.load()) : 0ll));
    root.set("session_persistence_mode", ajson::jstr("async_ordered_commit"));
    root.set("session_persistence_state", ajson::jstr(
            state && state->session_persistence_failures.load() != 0u ? "error" :
            state && state->session_persistence_pending.load() != 0u ? "pending" :
            "committed"));
    root.set("session_persistence_pending", ajson::jint(state
            ? static_cast<long long>(state->session_persistence_pending.load()) : 0ll));
    root.set("session_persistence_queued", ajson::jint(state
            ? static_cast<long long>(state->session_persistence_queued.load()) : 0ll));
    root.set("session_persistence_completed", ajson::jint(state
            ? static_cast<long long>(state->session_persistence_completed.load()) : 0ll));
    root.set("session_persistence_failures", ajson::jint(state
            ? static_cast<long long>(state->session_persistence_failures.load()) : 0ll));
    root.set("session_gc_runs", ajson::jint(state
            ? static_cast<long long>(state->session_gc_runs.load()) : 0ll));
    root.set("session_generation_gc_runs", ajson::jint(state
            ? static_cast<long long>(state->session_generation_gc_runs.load()) : 0ll));
    root.set("session_lifecycle_gc_runs", ajson::jint(state
            ? static_cast<long long>(state->session_lifecycle_gc_runs.load()) : 0ll));
    root.set("session_gc_removed_files", ajson::jint(state
            ? static_cast<long long>(state->session_gc_removed_files.load()) : 0ll));
    root.set("session_gc_removed_sessions", ajson::jint(state
            ? static_cast<long long>(state->session_gc_removed_sessions.load()) : 0ll));
    root.set("session_gc_reclaimed_bytes", ajson::jint(state
            ? static_cast<long long>(state->session_gc_reclaimed_bytes.load()) : 0ll));
    root.set("session_gc_scanned_sessions", ajson::jint(state
            ? static_cast<long long>(state->session_gc_scanned_sessions.load()) : 0ll));
    root.set("session_gc_protected_sessions", ajson::jint(state
            ? static_cast<long long>(state->session_gc_protected_sessions.load()) : 0ll));
    root.set("session_gc_unsafe_namespaces", ajson::jint(state
            ? static_cast<long long>(state->session_gc_unsafe_namespaces.load()) : 0ll));
    root.set("session_gc_last_run_unix", ajson::jint(state
            ? static_cast<long long>(state->session_gc_last_run_unix.load()) : 0ll));
    root.set("session_gc_last_duration_ms", ajson::jint(state
            ? static_cast<long long>(state->session_gc_last_duration_ms.load()) : 0ll));
    root.set("session_gc_ttl_seconds", ajson::jint(state
            ? static_cast<long long>(state->session_ttl_seconds) : 0ll));
    root.set("session_gc_max_namespaces", ajson::jint(state
            ? static_cast<long long>(state->session_max_namespaces) : 0ll));
    root.set("session_gc_interval_seconds", ajson::jint(state
            ? static_cast<long long>(state->session_gc_interval_seconds) : 0ll));
    root.set("session_gc_failures", ajson::jint(state
            ? static_cast<long long>(state->session_gc_failures.load()) : 0ll));
    root.set("session_gc_execution", ajson::jstr("async_maintenance_worker"));
    root.set("session_gc_active", ajson::jbool(
            state && state->lifecycle_gc_active.load(std::memory_order_acquire)));
    root.set("session_gc_deferred", ajson::jint(state
            ? static_cast<long long>(state->lifecycle_gc_deferred.load()) : 0ll));
    root.set("session_gc_state", ajson::jstr(
            state && (state->session_gc_failures.load() != 0u ||
                      state->session_gc_unsafe_namespaces.load() != 0u)
                    ? "degraded" : "pass"));
    root.set("session_cold_page_reads", ajson::jint(
            state ? static_cast<long long>(state->session_cold_page_reads.load()) : 0ll));
    if (state && state->kv_tier) {
        axiom_qwen38_kv_tier_info info{};
        info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
        if (axiom_qwen38_kv_tier_info_get(state->kv_tier, &info) == AXIOM_OK) {
            root.set("backend", ajson::jint(info.backend));
            root.set("direct_io", ajson::jbool(info.direct_io != 0u));
            root.set("committed_tokens", ajson::jint(info.committed_tokens));
            root.set("hot_pages", ajson::jint(info.plan.hot_pages));
        }
    }
    return root;
}

ajson ops_memory_status(const server_state *state) {
    ajson root = ajson::jobj();
    root.set("schema", ajson::jstr("axiom_memory_recovery_v1"));
    root.set("status", ajson::jstr("pass"));
    root.set("global_context", ajson::jint(state ? state->max_context : 0u));
    root.set("compaction_enabled", ajson::jbool(state && state->context_compaction));
    root.set("compaction_boundary", ajson::jint(kContextCompactionBoundary));
    root.set("compaction_active_limit", ajson::jint(
            kContextCompactionBoundary - kContextCompactionMaxOutputReserve -
                    kContextCompactionBuffer));
    root.set("kv_mode", ajson::jstr(state && state->streaming_kv ? "paged_runtime" : "disabled"));
    root.set("free_memory_gib", ajson::jnull());
    root.set("free_memory_available", ajson::jbool(false));
    root.set("note", ajson::jstr(
            "GPU/host free-memory telemetry is not exposed by the native API; "
            "the plan reports only enforced context and KV limits."));
    return root;
}

ajson ops_scaleout_status(const server_state *state) {
    ajson root = ajson::jobj();
    root.set("schema", ajson::jstr("axiom_scaleout_status_v1"));
    root.set("status", ajson::jstr("pass"));
    root.set("mode", ajson::jstr("single_device"));
    root.set("nodes", ajson::jint(1));
    root.set("device", ajson::jint(0));
    root.set("request_serialization", ajson::jbool(state != nullptr));
    root.set("scaleout_enabled", ajson::jbool(false));
    root.set("note", ajson::jstr("Qwen3.8 runtime is configured for one local GPU."));
    return root;
}

ajson ops_logs_status() {
    ajson root = ajson::jobj();
    root.set("schema", ajson::jstr("axiom_logs_status_v1"));
    root.set("status", ajson::jstr("pass"));
    root.set("available", ajson::jbool(false));
    root.set("source", ajson::jstr("stderr"));
    root.set("note", ajson::jstr(
            "The API does not retain an in-memory log tail; use the service journal or stderr."));
    return root;
}

const char *swarm_task_state_name(axiom::qwen38::swarm_task_state state) {
    using axiom::qwen38::swarm_task_state;
    switch (state) {
    case swarm_task_state::queued: return "queued";
    case swarm_task_state::running: return "running";
    case swarm_task_state::retry_wait: return "retry_wait";
    case swarm_task_state::succeeded: return "succeeded";
    case swarm_task_state::failed: return "failed";
    case swarm_task_state::cancelled: return "cancelled";
    case swarm_task_state::timed_out: return "timed_out";
    case swarm_task_state::budget_exhausted: return "budget_exhausted";
    }
    return "unknown";
}

const char *swarm_wave_state_name(axiom::qwen38::swarm_wave_state state) {
    using axiom::qwen38::swarm_wave_state;
    switch (state) {
    case swarm_wave_state::pending: return "pending";
    case swarm_wave_state::running: return "running";
    case swarm_wave_state::cancelling: return "cancelling";
    case swarm_wave_state::succeeded: return "succeeded";
    case swarm_wave_state::failed: return "failed";
    case swarm_wave_state::cancelled: return "cancelled";
    case swarm_wave_state::timed_out: return "timed_out";
    case swarm_wave_state::budget_exhausted: return "budget_exhausted";
    }
    return "unknown";
}

const char *swarm_agent_state_name(axiom::qwen38::swarm_agent_state state) {
    using axiom::qwen38::swarm_agent_state;
    switch (state) {
    case swarm_agent_state::idle: return "idle";
    case swarm_agent_state::busy: return "busy";
    case swarm_agent_state::draining: return "draining";
    case swarm_agent_state::offline: return "offline";
    case swarm_agent_state::failed: return "failed";
    }
    return "unknown";
}

const char *swarm_cancel_disposition_name(
        axiom::qwen38::swarm_cancel_disposition disposition) {
    using axiom::qwen38::swarm_cancel_disposition;
    switch (disposition) {
    case swarm_cancel_disposition::not_found: return "not_found";
    case swarm_cancel_disposition::already_terminal: return "already_terminal";
    case swarm_cancel_disposition::cancelled_before_run: return "cancelled_before_run";
    case swarm_cancel_disposition::cancellation_requested: return "cancellation_requested";
    }
    return "unknown";
}

const char *swarm_event_type_name(axiom::qwen38::swarm_event_type type) {
    using axiom::qwen38::swarm_event_type;
    switch (type) {
    case swarm_event_type::wave_created: return "wave_created";
    case swarm_event_type::wave_state_changed: return "wave_state_changed";
    case swarm_event_type::agent_registered: return "agent_registered";
    case swarm_event_type::agent_state_changed: return "agent_state_changed";
    case swarm_event_type::task_submitted: return "task_submitted";
    case swarm_event_type::task_rejected: return "task_rejected";
    case swarm_event_type::task_admitted: return "task_admitted";
    case swarm_event_type::task_started: return "task_started";
    case swarm_event_type::task_progress: return "task_progress";
    case swarm_event_type::task_retry_scheduled: return "task_retry_scheduled";
    case swarm_event_type::task_succeeded: return "task_succeeded";
    case swarm_event_type::task_failed: return "task_failed";
    case swarm_event_type::task_cancel_requested: return "task_cancel_requested";
    case swarm_event_type::task_cancelled: return "task_cancelled";
    case swarm_event_type::task_timed_out: return "task_timed_out";
    case swarm_event_type::task_budget_exhausted: return "task_budget_exhausted";
    }
    return "unknown";
}

ajson ops_swarm_status(const server_state *state) {
    ajson root = ajson::jobj();
    root.set("schema", ajson::jstr("axiom_swarm_status_v1"));
    root.set("status", ajson::jstr(state ? "pass" : "fail"));
    root.set("enabled", ajson::jbool(state != nullptr));
    root.set("execution_width", ajson::jint(1));
    root.set("backend", ajson::jstr("serialized_native_model"));
    if (!state) return root;
    const axiom::qwen38::swarm_scheduler_snapshot snapshot =
            state->swarm_scheduler.snapshot();
    root.set("revision", ajson::jint(static_cast<long long>(snapshot.revision)));
    const auto &telemetry = snapshot.telemetry;
    ajson counters = ajson::jobj();
    counters.set("events_emitted", ajson::jint(static_cast<long long>(telemetry.events_emitted)));
    counters.set("tasks_submitted", ajson::jint(static_cast<long long>(telemetry.tasks_submitted)));
    counters.set("tasks_rejected", ajson::jint(static_cast<long long>(telemetry.tasks_rejected)));
    counters.set("attempts_admitted", ajson::jint(static_cast<long long>(telemetry.attempts_admitted)));
    counters.set("attempts_started", ajson::jint(static_cast<long long>(telemetry.attempts_started)));
    counters.set("retries_scheduled", ajson::jint(static_cast<long long>(telemetry.retries_scheduled)));
    counters.set("progress_reports", ajson::jint(static_cast<long long>(telemetry.progress_reports)));
    counters.set("final_succeeded", ajson::jint(static_cast<long long>(telemetry.final_succeeded)));
    counters.set("final_failed", ajson::jint(static_cast<long long>(telemetry.final_failed)));
    counters.set("final_cancelled", ajson::jint(static_cast<long long>(telemetry.final_cancelled)));
    counters.set("final_timed_out", ajson::jint(static_cast<long long>(telemetry.final_timed_out)));
    counters.set("final_budget_exhausted", ajson::jint(static_cast<long long>(telemetry.final_budget_exhausted)));
    counters.set("active_tasks", ajson::jint(static_cast<long long>(telemetry.active_tasks)));
    counters.set("queued_tasks", ajson::jint(static_cast<long long>(telemetry.queued_tasks)));
    counters.set("registered_agents", ajson::jint(static_cast<long long>(telemetry.registered_agents)));
    counters.set("available_agents", ajson::jint(static_cast<long long>(telemetry.available_agents)));
    root.set("counters", std::move(counters));

    ajson tasks = ajson::jarr();
    for (const auto &task : snapshot.tasks) {
        ajson item = ajson::jobj();
        item.set("request_id", ajson::jint(static_cast<long long>(task.task_id)));
        item.set("name", ajson::jstr(task.name));
        item.set("wave_id", ajson::jint(static_cast<long long>(task.wave_id)));
        item.set("priority", ajson::jint(task.priority));
        item.set("state", ajson::jstr(swarm_task_state_name(task.state)));
        item.set("attempts_started", ajson::jint(task.attempts_started));
        item.set("current_attempt", ajson::jint(task.current_attempt));
        item.set("max_attempts", ajson::jint(task.max_attempts));
        item.set("cancellation_requested", ajson::jbool(task.cancellation_requested));
        item.set("tokens_used", ajson::jint(static_cast<long long>(task.tokens_used)));
        item.set("tokens_used_total", ajson::jint(static_cast<long long>(task.tokens_used_total)));
        item.set("terminal_reason", ajson::jstr(task.terminal_reason));
        if (task.agent_id) item.set("agent_id", ajson::jint(static_cast<long long>(*task.agent_id)));
        else item.set("agent_id", ajson::jnull());
        swarm_request_metadata metadata;
        if (swarm_metadata_for(state, task.task_id, &metadata)) {
            ajson logical = ajson::jobj();
            if (!metadata.wave_id.empty()) logical.set("wave_id", ajson::jstr(metadata.wave_id));
            if (!metadata.agent_id.empty()) logical.set("agent_id", ajson::jstr(metadata.agent_id));
            if (!metadata.parent_agent_id.empty()) {
                logical.set("parent_agent_id", ajson::jstr(metadata.parent_agent_id));
            }
            logical.set("priority", ajson::jint(metadata.priority));
            item.set("logical", std::move(logical));
        }
        tasks.push(std::move(item));
    }
    root.set("tasks", std::move(tasks));

    ajson waves = ajson::jarr();
    for (const auto &wave : snapshot.waves) {
        ajson item = ajson::jobj();
        item.set("wave_id", ajson::jint(static_cast<long long>(wave.wave_id)));
        item.set("name", ajson::jstr(wave.name));
        item.set("state", ajson::jstr(swarm_wave_state_name(wave.state)));
        item.set("cancellation_requested", ajson::jbool(wave.cancellation_requested));
        item.set("task_count", ajson::jint(static_cast<long long>(wave.task_count)));
        item.set("active_tasks", ajson::jint(static_cast<long long>(wave.active_tasks)));
        item.set("terminal_tasks", ajson::jint(static_cast<long long>(wave.terminal_tasks)));
        waves.push(std::move(item));
    }
    root.set("waves", std::move(waves));

    ajson agents = ajson::jarr();
    for (const auto &agent : snapshot.agents) {
        ajson item = ajson::jobj();
        item.set("agent_id", ajson::jint(static_cast<long long>(agent.agent_id)));
        item.set("name", ajson::jstr(agent.name));
        item.set("state", ajson::jstr(swarm_agent_state_name(agent.state)));
        item.set("max_concurrent", ajson::jint(static_cast<long long>(agent.max_concurrent)));
        item.set("active_tasks", ajson::jint(static_cast<long long>(agent.active_tasks)));
        agents.push(std::move(item));
    }
    root.set("agents", std::move(agents));

    ajson events = ajson::jarr();
    const size_t first = snapshot.events.size() > 64u ? snapshot.events.size() - 64u : 0u;
    for (size_t index = first; index < snapshot.events.size(); ++index) {
        const auto &event = snapshot.events[index];
        ajson item = ajson::jobj();
        item.set("sequence", ajson::jint(static_cast<long long>(event.sequence)));
        item.set("type", ajson::jstr(swarm_event_type_name(event.type)));
        item.set("task_id", ajson::jint(static_cast<long long>(event.task_id)));
        item.set("wave_id", ajson::jint(static_cast<long long>(event.wave_id)));
        item.set("agent_id", ajson::jint(static_cast<long long>(event.agent_id)));
        item.set("attempt", ajson::jint(event.attempt));
        item.set("tokens_used", ajson::jint(static_cast<long long>(event.tokens_used)));
        item.set("message", ajson::jstr(event.message));
        swarm_request_metadata metadata;
        if (swarm_metadata_for(state, event.task_id, &metadata)) {
            ajson logical = ajson::jobj();
            if (!metadata.wave_id.empty()) logical.set("wave_id", ajson::jstr(metadata.wave_id));
            if (!metadata.agent_id.empty()) logical.set("agent_id", ajson::jstr(metadata.agent_id));
            if (!metadata.parent_agent_id.empty()) {
                logical.set("parent_agent_id", ajson::jstr(metadata.parent_agent_id));
            }
            item.set("logical", std::move(logical));
        }
        events.push(std::move(item));
    }
    root.set("recent_events", std::move(events));
    return root;
}

ajson ops_agent_status() {
    ajson root = ajson::jobj();
    root.set("schema", ajson::jstr("axiom_agent_plane_v1"));
    root.set("status", ajson::jstr("pass"));
    root.set("model", ajson::jstr(kModelId));
    root.set("native_tool_calls", ajson::jbool(true));
    root.set("openai_tools", ajson::jbool(true));
    root.set("anthropic_tool_use", ajson::jbool(true));
    root.set("harness", ajson::jstr("axiom-harness"));
    root.set("harness_language", ajson::jstr("C++17"));
    root.set("harness_embedded", ajson::jbool(false));
    root.set("note", ajson::jstr(
            "The capability plane is the separate axiom-harness C++17 process."));
    return root;
}

ajson ops_sampling_status(const server_state *state) {
    ajson root = ajson::jobj();
    root.set("schema", ajson::jstr("axiom_sampling_profile_v1"));
    root.set("status", ajson::jstr(state && state->healthy.load() ? "pass" : "fail"));
    root.set("default_thinking", ajson::jbool(state && !state->no_think));
    root.set("default_profile", ajson::jstr(
            state && state->no_think ? "ultra-fast" : "medium"));
    root.set("default_reasoning_effort", ajson::jstr(
            state && !state->no_think ? "medium" : "ultra-fast"));
    root.set("default_thinking_budget", ajson::jint(
            state && !state->no_think ? kDefaultThinkingBudget : 0u));
    root.set("max_thinking_budget", ajson::jint(
            axiom::reasoning::kMaxThinkingBudgetTokens));
    root.set("reasoning_efforts", reasoning_efforts_json());
    root.set("reasoning_profiles", reasoning_profiles_json());
    root.set("thinking_temperature", ajson::jnum(kThinkingTemperature));
    root.set("thinking_top_p", ajson::jnum(kThinkingTopP));
    root.set("thinking_top_k", ajson::jint(kDefaultQwenTopK));
    root.set("non_thinking_temperature", ajson::jnum(kNonThinkingTemperature));
    root.set("non_thinking_top_p", ajson::jnum(kNonThinkingTopP));
    root.set("non_thinking_top_k", ajson::jint(kNonThinkingTopK));
    root.set("thinking_budget_supported", ajson::jbool(true));
    root.set("greedy_override", ajson::jstr(
            "temperature=0, top_p=1; preserves the configured Ultra-fast graph"));
    root.set("request_overrides", ajson::jstr(
            "think, reasoning.effort, thinking_budget, temperature, top_p, top_k, seed, "
            "axiom_speculative_mode"));
    return root;
}

void handle_client(int fd, server_state *state) {
    http_request request;
    int status = 400;
    if (!read_request(fd, &request, &status)) {
        send_response(fd, status, ajson_dumps(error_json("invalid HTTP request", "invalid_request_error")));
        return;
    }
    if (request.method == "GET" && request.path == "/health/live") {
        ajson root = ajson::jobj();
        root.set("schema", ajson::jstr("axiom_liveness_v1"));
        root.set("status", ajson::jstr("pass"));
        root.set("alive", ajson::jbool(true));
        root.set("process_id", ajson::jint(static_cast<long long>(::getpid())));
        root.set("control_plane", ajson::jstr("independent_http_worker"));
        root.set("http_worker_admission", ajson::jstr("unbounded"));
        send_response(fd, 200, ajson_dumps(root));
        return;
    }
    if (request.method == "GET" && request.path == "/health/ready") {
        const bool ready = state && state->healthy.load() && state->model &&
                state->tokenizer;
        ajson root = ajson::jobj();
        root.set("schema", ajson::jstr("axiom_readiness_v1"));
        root.set("status", ajson::jstr(ready ? "pass" : "fail"));
        root.set("ready", ajson::jbool(ready));
        root.set("model", ajson::jstr(kModelId));
        root.set("backend", ajson::jstr(kBackendId));
        root.set("generation_queue_capacity", ajson::jint(kSwarmQueueDepth));
        send_response(fd, ready ? 200 : 503, ajson_dumps(root));
        return;
    }
    if (request.method == "GET" && request.path == "/health") {
        ajson root = ajson::jobj();
        const session_status_snapshot session = capture_session_status(state);
        root.set("ok", ajson::jbool(state->healthy.load()));
        root.set("process_id", ajson::jint(static_cast<long long>(::getpid())));
        root.set("model", ajson::jstr(kModelId));
        root.set("loaded", ajson::jbool(state->healthy.load()));
        root.set("input_modalities", vision_input_modalities_json(state));
        root.set("output_modalities", [] {
            ajson modalities = ajson::jarr();
            modalities.push(ajson::jstr("text"));
            return modalities;
        }());
        root.set("vision", vision_capabilities_json(state));
        root.set("vision_enabled", ajson::jbool(state->vision != nullptr));
        root.set("api_key_required", ajson::jbool(false));
        root.set("liveness_endpoint", ajson::jstr("/health/live"));
        root.set("readiness_endpoint", ajson::jstr("/health/ready"));
        root.set("listen_addresses", [&] {
            ajson addresses = ajson::jarr();
            for (const std::string &address : state->listen_addresses) {
                addresses.push(ajson::jstr(address));
            }
            return addresses;
        }());
        root.set("max_context", ajson::jint(state->max_context));
        root.set("context_window_default", ajson::jint(state->default_context));
        root.set("context_window_max", ajson::jint(state->max_context));
        root.set("context_window_options", [&] {
            ajson options = ajson::jarr();
            options.push(ajson::jint(state->default_context));
            if (state->max_context != state->default_context) {
                options.push(ajson::jint(state->max_context));
            }
            return options;
        }());
        root.set("backend", ajson::jstr(kBackendId));
        root.set("backend_available", ajson::jbool(
                state->streaming_kv || state->native_mtp_loaded ||
                axiom_qwen38_speculative_backend_available() == 1));
        root.set("scalar_fallback", ajson::jbool(false));
        root.set("single_session", ajson::jbool(false));
        root.set("http_multi_session", ajson::jbool(true));
        root.set("max_http_sessions", ajson::jint(0));
        root.set("http_sessions_unlimited", ajson::jbool(true));
        root.set("http_workers_active", ajson::jint(
                static_cast<long long>(state->active_http_workers.load())));
        root.set("http_worker_admission", ajson::jstr("unbounded"));
        root.set("generation_queue_capacity", ajson::jint(kSwarmQueueDepth));
        root.set("session_id_auto_generated", ajson::jbool(true));
        root.set("session_id_shared_default", ajson::jbool(false));
        root.set("session_id_header", ajson::jstr(kSessionHeader));
        root.set("session_leases_active", ajson::jint(
                static_cast<long long>(state->session_leases_active.load())));
        root.set("session_leases_peak", ajson::jint(
                static_cast<long long>(state->session_leases_peak.load())));
        root.set("generation_scheduler", ajson::jstr("swarm_queue_serialized_native_model"));
        {
            std::lock_guard<std::mutex> lock(state->active_request_mutex);
            root.set("generation_busy", ajson::jbool(
                    state->active_request_sequence != 0u));
            if (state->active_request_sequence != 0u) {
                root.set("active_request_sequence", ajson::jint(
                        static_cast<long long>(state->active_request_sequence)));
                root.set("active_session_id", ajson::jstr(
                        state->active_request_session_id));
            } else {
                root.set("active_request_sequence", ajson::jnull());
                root.set("active_session_id", ajson::jnull());
            }
        }
        root.set("decode_chunk_cycles", ajson::jint(kDeviceChunkCycles));
        root.set("sse_batch_cycles", ajson::jint(
                static_cast<long long>(state->sse_batch_cycles)));
        root.set("speculative_graph", ajson::jbool(
                speculative_graph_loaded(state)));
        root.set("speculative_mode_default", ajson::jstr("auto"));
        root.set("speculative_modes", speculative_modes_json());
        root.set("speculative_mode_request_field", ajson::jstr(
                "axiom_speculative_mode"));
        root.set("hybrid_graph_scalar", ajson::jbool(
                state->streaming_kv && speculative_graph_loaded(state)));
        root.set("speculative_context_tokens", ajson::jint(
                state->speculative_context_tokens));
        attach_speculative_contract(&root, state);
        attach_speculative_engine_status(&root, state);
        root.set("graph_executor_resident", ajson::jbool(
                state->graph_executor_resident.load(std::memory_order_acquire)));
        root.set("graph_executor_creates", ajson::jint(
                static_cast<long long>(state->graph_executor_creates.load())));
        root.set("graph_executor_reuses", ajson::jint(
                static_cast<long long>(state->graph_executor_reuses.load())));
        root.set("graph_executor_recycles", ajson::jint(
                static_cast<long long>(state->graph_executor_recycles.load())));
        root.set("graph_executor_recycle_failures", ajson::jint(
                static_cast<long long>(state->graph_executor_recycle_failures.load())));
        root.set("swarm_enabled", ajson::jbool(true));
        root.set("swarm_execution_width", ajson::jint(1));
        root.set("swarm_queue_capacity", ajson::jint(kSwarmQueueDepth));
        root.set("session_persistence", ajson::jbool(
                session_persistence_enabled(state)));
        root.set("session_manifest_format", ajson::jstr(
                state && state->session_store.manifest_version() == 2u
                ? "AXQ38SM1/v2-sha256" : "AXQ38SM1/v1"));
        root.set("session_id", ajson::jstr(session.session_id));
        root.set("session_namespace", ajson::jstr(session.namespace_id));
        root.set("session_generation", ajson::jint(
                static_cast<long long>(session.generation)));
        root.set("session_manifest_committed", ajson::jbool(
                session.manifest_committed));
        root.set("session_manifest_committed_tokens", ajson::jint(
                session.committed_tokens));
        root.set("session_tier_path", ajson::jstr(session.tier_path));
        root.set("session_restore_count", ajson::jint(
                static_cast<long long>(state->session_restore_count.load())));
        root.set("session_manifest_writes", ajson::jint(
                static_cast<long long>(state->session_manifest_writes.load())));
        root.set("session_persistence_mode", ajson::jstr("async_ordered_commit"));
        root.set("session_persistence_state", ajson::jstr(
                state->session_persistence_failures.load() != 0u ? "error" :
                state->session_persistence_pending.load() != 0u ? "pending" :
                "committed"));
        root.set("session_persistence_pending", ajson::jint(
                static_cast<long long>(state->session_persistence_pending.load())));
        root.set("session_persistence_queued", ajson::jint(
                static_cast<long long>(state->session_persistence_queued.load())));
        root.set("session_persistence_completed", ajson::jint(
                static_cast<long long>(state->session_persistence_completed.load())));
        root.set("session_persistence_failures", ajson::jint(
                static_cast<long long>(state->session_persistence_failures.load())));
        root.set("session_gc_runs", ajson::jint(
                static_cast<long long>(state->session_gc_runs.load())));
        root.set("session_generation_gc_runs", ajson::jint(
                static_cast<long long>(state->session_generation_gc_runs.load())));
        root.set("session_lifecycle_gc_runs", ajson::jint(
                static_cast<long long>(state->session_lifecycle_gc_runs.load())));
        root.set("session_gc_removed_files", ajson::jint(
                static_cast<long long>(state->session_gc_removed_files.load())));
        root.set("session_gc_removed_sessions", ajson::jint(
                static_cast<long long>(state->session_gc_removed_sessions.load())));
        root.set("session_gc_reclaimed_bytes", ajson::jint(
                static_cast<long long>(state->session_gc_reclaimed_bytes.load())));
        root.set("session_gc_scanned_sessions", ajson::jint(
                static_cast<long long>(state->session_gc_scanned_sessions.load())));
        root.set("session_gc_protected_sessions", ajson::jint(
                static_cast<long long>(state->session_gc_protected_sessions.load())));
        root.set("session_gc_unsafe_namespaces", ajson::jint(
                static_cast<long long>(state->session_gc_unsafe_namespaces.load())));
        root.set("session_gc_last_run_unix", ajson::jint(
                static_cast<long long>(state->session_gc_last_run_unix.load())));
        root.set("session_gc_last_duration_ms", ajson::jint(
                static_cast<long long>(state->session_gc_last_duration_ms.load())));
        root.set("session_gc_ttl_seconds", ajson::jint(
                static_cast<long long>(state->session_ttl_seconds)));
        root.set("session_gc_max_namespaces", ajson::jint(
                static_cast<long long>(state->session_max_namespaces)));
        root.set("session_gc_interval_seconds", ajson::jint(
                static_cast<long long>(state->session_gc_interval_seconds)));
        root.set("session_gc_failures", ajson::jint(
                static_cast<long long>(state->session_gc_failures.load())));
        root.set("session_gc_execution", ajson::jstr("async_maintenance_worker"));
        root.set("session_gc_active", ajson::jbool(
                state->lifecycle_gc_active.load(std::memory_order_acquire)));
        root.set("session_gc_deferred", ajson::jint(
                static_cast<long long>(state->lifecycle_gc_deferred.load())));
        root.set("session_gc_state", ajson::jstr(
                state->session_gc_failures.load() != 0u ||
                        state->session_gc_unsafe_namespaces.load() != 0u
                        ? "degraded" : "pass"));
        root.set("session_cold_page_reads", ajson::jint(
                static_cast<long long>(state->session_cold_page_reads.load())));
        root.set("openai_chat_completions", ajson::jbool(true));
        root.set("openai_chat_completions_streaming", ajson::jbool(true));
        root.set("openai_completions", ajson::jbool(true));
        root.set("openai_responses", ajson::jbool(true));
        root.set("openai_responses_streaming", ajson::jbool(true));
        root.set("anthropic_messages", ajson::jbool(true));
        root.set("native_qwen_tool_calls", ajson::jbool(true));
        root.set("native_deferred_tools", ajson::jbool(true));
        root.set("responses_tool_search", ajson::jbool(true));
        root.set("openai_tools", ajson::jbool(true));
        root.set("anthropic_tool_use", ajson::jbool(true));
        root.set("no_think_default", ajson::jbool(state->no_think));
        root.set("default_profile", ajson::jstr(
                state->no_think ? "ultra-fast" : "medium"));
        root.set("default_reasoning_effort", ajson::jstr(
                state->no_think ? "ultra-fast" : "medium"));
        root.set("thinking_default_budget", ajson::jint(
                state->no_think ? 0u : kDefaultThinkingBudget));
        root.set("max_thinking_budget", ajson::jint(
                axiom::reasoning::kMaxThinkingBudgetTokens));
        root.set("reasoning_efforts", reasoning_efforts_json());
        root.set("reasoning_profiles", reasoning_profiles_json());
        root.set("thinking_sampling", ajson::jbool(true));
        root.set("thinking_temperature", ajson::jnum(kThinkingTemperature));
        root.set("thinking_top_p", ajson::jnum(kThinkingTopP));
        root.set("thinking_top_k", ajson::jint(kDefaultQwenTopK));
        root.set("yarn_enabled", ajson::jbool(!env_disabled("AXIOM_QWEN38_YARN")));
        root.set("yarn_factor", ajson::jnum(4.0));
        root.set("yarn_original_context", ajson::jint(262144));
        root.set("yarn_target_context", ajson::jint(state->max_context));
        root.set("context_global_max", ajson::jint(state->max_context));
        root.set("context_compaction_enabled", ajson::jbool(state->context_compaction));
        root.set("context_compaction_boundary", ajson::jint(kContextCompactionBoundary));
        root.set("context_compaction_active_limit", ajson::jint(
                kContextCompactionBoundary - kContextCompactionMaxOutputReserve -
                        kContextCompactionBuffer));
        root.set("kv_tier_enabled", ajson::jbool(state->kv_tier != nullptr));
        /* Runtime paging is the production profile. */
        root.set("kv_tier_mode", ajson::jstr(
                state->streaming_kv ? "paged_runtime" :
                (state->kv_tier ? "durable_mirror" : "disabled")));
        root.set("paged_kv_runtime", ajson::jbool(state->streaming_kv));
        root.set("anthropic_messages_streaming", ajson::jbool(true));
        if (state->kv_tier) {
            axiom_qwen38_kv_tier_info tier_info{};
            tier_info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
            if (axiom_qwen38_kv_tier_info_get(state->kv_tier, &tier_info) == AXIOM_OK) {
                root.set("kv_tier_backend", ajson::jint(tier_info.backend));
                root.set("kv_tier_direct_io", ajson::jbool(tier_info.direct_io != 0u));
                root.set("kv_logical_pages", ajson::jint(tier_info.plan.logical_pages));
                root.set("kv_hot_pages", ajson::jint(tier_info.plan.hot_pages));
                root.set("kv_tier_committed_tokens", ajson::jint(tier_info.committed_tokens));
            }
        }
        send_response(fd, 200, ajson_dumps(root));
        return;
    }
    if (request.method == "GET" && request.path == "/ops/runtime") {
        send_response(fd, 200, ajson_dumps(ops_runtime_status(state)));
        return;
    }
    if (request.method == "GET" && request.path == "/ops/kv") {
        send_response(fd, 200, ajson_dumps(ops_kv_status(state)));
        return;
    }
    if (request.method == "GET" && request.path == "/ops/memory/recovery-plan") {
        send_response(fd, 200, ajson_dumps(ops_memory_status(state)));
        return;
    }
    if (request.method == "GET" && request.path == "/ops/scaleout") {
        send_response(fd, 200, ajson_dumps(ops_scaleout_status(state)));
        return;
    }
    if (request.method == "GET" && request.path.rfind("/ops/logs", 0u) == 0u) {
        send_response(fd, 200, ajson_dumps(ops_logs_status()));
        return;
    }
    if (request.method == "GET" && request.path == "/ops/agent") {
        send_response(fd, 200, ajson_dumps(ops_agent_status()));
        return;
    }
    if (request.method == "GET" &&
        (request.path == "/ops/sampling" || request.path == "/ops/thinking")) {
        send_response(fd, 200, ajson_dumps(ops_sampling_status(state)));
        return;
    }
    if (request.method == "GET" &&
        (request.path == "/ops/usage" || request.path == "/ops/tokens")) {
        send_response(fd, 200, ajson_dumps(ops_usage_status(state)));
        return;
    }
    if (request.method == "GET" && request.path == "/ops/swarm") {
        send_response(fd, 200, ajson_dumps(ops_swarm_status(state)));
        return;
    }
    if (request.method == "POST" && request.path == "/ops/swarm/cancel") {
        ajson payload;
        std::string parse_error;
        if (!ajson_parse(request.body, payload, parse_error) || !payload.is_object()) {
            send_response(fd, 400, ajson_dumps(error_json(
                    parse_error.empty() ? "request body must be a JSON object" : parse_error,
                    "invalid_request_error")));
            return;
        }
        const ajson *request_id = payload.get("request_id");
        if (!request_id || !request_id->is_int() || request_id->i <= 0) {
            send_response(fd, 400, ajson_dumps(error_json(
                    "request_id must be a positive integer", "invalid_request_error")));
            return;
        }
        const auto cancelled = state->swarm_scheduler.cancel_task(
                static_cast<axiom::qwen38::swarm_task_id>(request_id->i),
                "operator_cancel", axiom::qwen38::swarm_clock::now());
        ajson response = ajson::jobj();
        response.set("status", ajson::jstr("pass"));
        response.set("request_id", *request_id);
        response.set("changed", ajson::jbool(cancelled.changed));
        response.set("disposition", ajson::jstr(
                swarm_cancel_disposition_name(cancelled.disposition)));
        send_response(fd, 200, ajson_dumps(response));
        return;
    }
    if (request.method == "GET" &&
        (request.path == "/v1/models" || request.path == "/codex/v1/models")) {
        send_response(fd, 200, ajson_dumps(models_response(state, request.path == "/codex/v1/models")));
        return;
    }
    if (request.method != "POST" ||
        (request.path != "/v1/chat/completions" && request.path != "/v1/completions" &&
         request.path != "/v1/messages" && request.path != "/v1/responses" &&
         request.path != "/codex/v1/responses")) {
        const int response_status = request.method == "GET" || request.method == "POST" ? 404 : 405;
        send_response(fd, response_status, ajson_dumps(error_json("route not found", "invalid_request_error")));
        return;
    }

    ajson payload;
    std::string parse_error;
    if (!ajson_parse(request.body, payload, parse_error) || !payload.is_object()) {
        send_response(fd, 400, ajson_dumps(error_json(
                parse_error.empty() ? "request body must be a JSON object" : parse_error,
                "invalid_request_error")));
        return;
    }
    swarm_request_metadata swarm_metadata;
    if (!parse_swarm_metadata(payload, &swarm_metadata, &parse_error)) {
        send_response(fd, 400, ajson_dumps(error_json(parse_error, "invalid_request_error")));
        return;
    }
    std::string error;
    if (!validate_model(payload, &error)) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    std::string session_id;
    bool session_id_generated = false;
    if (!resolve_session_id(
            payload, request.session_id, &session_id, &session_id_generated, &error)) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    const bool codex_provider = request.path == "/codex/v1/responses";
    // Only an explicit request/operator policy installs a total deadline.
    // Slow but advancing prefill must not be cancelled by an implicit 600s cap.
    // Cancellation remains cooperative between native steps, NOT GPU preemption.
    if (codex_provider && !swarm_metadata.deadline) {
        uint64_t deadline_ms = 0;
        if (!axiom_codex::request_deadline_ms(std::getenv("AXIOM_CODEX_REQUEST_DEADLINE_MS"), &deadline_ms)) {
            send_response(fd, 500, ajson_dumps(error_json(
                    "invalid AXIOM_CODEX_REQUEST_DEADLINE_MS; expected integer 1..86400000",
                    "invalid_codex_bridge_configuration")));
            return;
        }
        if (deadline_ms != 0u) swarm_metadata.deadline = std::chrono::milliseconds(deadline_ms);
    }
    const bool responses_api = request.path == "/v1/responses" || codex_provider;
    axiom_codex::tool_wire_map codex_wire;
    ajson codex_prompt_payload;
    bool has_codex_prompt_payload = false;
    if (codex_provider) configure_codex_schema_bridge(codex_wire);
    if (codex_provider && !codex_wire.prepare(payload, &error)) {
        ajson failure = error_json(error, "invalid_request_error");
        failure.set("codex_tool_errors", codex_wire.rejections());
        send_response(fd, 400, ajson_dumps(failure));
        return;
    }
    if (codex_provider && !codex_wire.rejections().arr.empty())
        std::fprintf(stderr, "[codex-schema] rejected_tools=%s\n", ajson_dumps(codex_wire.rejections()).c_str());
    if (responses_api) {
        ajson normalized;
        if (!(codex_provider ? normalize_codex_responses_request(payload, &normalized, &error, &codex_prompt_payload)
                             : normalize_responses_request(payload, &normalized, &error))) {
            send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
            return;
        }
        payload = std::move(normalized);
        has_codex_prompt_payload = codex_provider;
    }
    /* The Responses normalizer intentionally keeps only wire input fields.
     * Re-read the explicit session id when a provider put it inside the
     * normalized object, otherwise retain the original/default identity. */
    if (payload.get("session_id") || payload.get("axiom_session_id")) {
        bool normalized_generated = false;
        if (!resolve_session_id(
                payload, request.session_id, &session_id,
                &normalized_generated, &error)) {
            send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
            return;
        }
        session_id_generated = normalized_generated;
    }
    uint32_t selected_context = 0u;
    if (!parse_request_context_window(state, payload, &selected_context, &error)) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    speculative_mode requested_speculative_mode = speculative_mode::automatic;
    if (!parse_speculative_mode(payload, &requested_speculative_mode, &error)) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    uint32_t speculative_max_commit_tokens = AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
    if (!parse_speculative_max_commit_tokens(
                payload, requested_speculative_mode,
                &speculative_max_commit_tokens, &error)) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    request_context_scope context_scope(state, selected_context);
    const bool chat = request.path == "/v1/chat/completions";
    const bool anthropic = request.path == "/v1/messages";
    const bool chat_api = chat || responses_api;
    const bool no_think = chat_api || anthropic
            ? resolve_no_think(state, payload, &error) : true;
    if (!error.empty()) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    const std::string profile = effective_reasoning_effort(state, payload);
    std::string session_profile = requested_speculative_mode == speculative_mode::automatic
            ? profile
            : profile + "|spec=" + speculative_mode_name(requested_speculative_mode);
    if (speculative_max_commit_tokens != AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH) {
        /* Persistent manifest profiles are fixed at 31 visible bytes. Keep the
         * diagnostic namespace distinct without overflowing that contract. */
        session_profile += "|m=" + std::to_string(speculative_max_commit_tokens);
    }
    // Internal child partition only. All HTTP/SSE/result IDs remain the Core
    // root identity below, while existing per-namespace GC/leases still apply.
    std::string cache_session_id = session_id;
    try {
        if (codex_provider) cache_session_id = axiom_codex::cache_session_id(session_id, payload);
    } catch (const std::exception &failure) {
        send_response(fd, 500, ajson_dumps(error_json(failure.what(), "codex_cache_identity_error")));
        return;
    }
    const axiom::qwen38::qwen38_session_key leased_session_key =
            make_session_key(state, cache_session_id, session_profile);
    session_namespace_lease session_lease(
            state,
            axiom::qwen38::qwen38_persistent_session_store::namespace_id(
                    leased_session_key));
    uint32_t max_new = 0u;
    bool explicit_visible_budget = false;
    const ajson *stream_value = payload.get("stream");
    const bool stream_requested = stream_value && stream_value->is_bool() && stream_value->b;
    const bool live_chat_stream_requested = chat && stream_requested;
    const bool live_responses_stream_requested = responses_api && stream_requested;
    const bool live_anthropic_stream_requested = anthropic && stream_requested;
    if (!parse_max_tokens(
            payload, effective_context_limit(state), &max_new, &error,
            responses_api || chat || anthropic,
            &explicit_visible_budget)) {
        const int response_status = error.find("streaming") != std::string::npos ? 501 : 400;
        send_response(fd, response_status, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    if (swarm_metadata.token_budget &&
        *swarm_metadata.token_budget < static_cast<std::uint64_t>(max_new)) {
        max_new = static_cast<uint32_t>(*swarm_metadata.token_budget);
        if (max_new == 0u) {
            send_response(fd, 400, ajson_dumps(error_json(
                    "axiom_swarm.token_budget must be greater than zero",
                    "invalid_request_error")));
            return;
        }
    }
    sampling_params sampling;
    const uint64_t response_sequence = state->request_sequence.fetch_add(1u) + 1u;
    swarm_request_context swarm_context;
    swarm_context.sequence = response_sequence;
    swarm_context.metadata = swarm_metadata;
    if (!parse_sampling_params(
            payload, no_think, response_sequence, &sampling, &error)) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    thinking_plan plan;
    if (!make_thinking_plan(
            payload, no_think, max_new, explicit_visible_budget,
            effective_context_limit(state), &plan, &error)) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    if (!chat_api && !anthropic) {
        plan.enabled = false;
        plan.visible_budget = max_new;
    }
    std::vector<uint32_t> prompt_ids;
    std::vector<uint32_t> session_suffix_ids;
    std::shared_ptr<vision_request> media_request;
    uint32_t original_prompt_tokens = 0u;
    bool context_compacted = false;
    bool prompt_ok = false;
    const ajson &chat_prompt_payload = has_codex_prompt_payload ? codex_prompt_payload : payload;
    if (!chat_api && !anthropic) {
        prompt_ok = make_completion_ids(state, payload, &prompt_ids, &error);
    } else if (payload_contains_vision(payload)) {
        std::lock_guard<std::mutex> vision_lock(state->vision_mutex);
        prompt_ok = make_multimodal_chat_ids(
                state, chat_prompt_payload, anthropic, &prompt_ids, &media_request, &error,
                &original_prompt_tokens, &context_compacted);
    } else {
        prompt_ok = make_chat_ids(
                                  state, chat_prompt_payload, anthropic, &prompt_ids,
                                  &session_suffix_ids, &error,
                                  &original_prompt_tokens, &context_compacted, &payload);
    }

    const bool allow_stateful_session_resume = axiom_codex::allow_stateful_tail(codex_provider,
            !session_id_generated && (chat_api || anthropic) && !media_request &&
            !context_compacted && !session_suffix_ids.empty());
    if (!prompt_ok) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    if (original_prompt_tokens == 0u) {
        original_prompt_tokens = static_cast<uint32_t>(prompt_ids.size());
    }

    /* Validate and normalize tools before sending SSE headers. The native
     * model emits its XML tool grammar, while the live wire adapter below
     * converts the completed call into OpenAI tool_call deltas. */
    native_tool_result tool_result;
    ajson normalized_tools;
    if ((chat_api || anthropic) && !normalize_tools(
            payload.get("tools"), anthropic, &normalized_tools, &error)) {
        send_response(fd, 400, ajson_dumps(error_json(error, "invalid_request_error")));
        return;
    }
    const bool tools_enabled = (chat_api || anthropic) && !normalized_tools.arr.empty() &&
            tools_are_enabled(payload.get("tool_choice"));

    generation_result result;
    std::string failure_stage;
    if (!admit_swarm_request(fd, state, &swarm_context, &error)) {
        if (!socket_peer_disconnected(fd)) {
            const bool timed_out = error == "swarm queue timeout" ||
                    error == "swarm deadline exceeded while queued";
            const std::string body = ajson_dumps(error_json(
                    error, "swarm_admission_error"));
            if (timed_out) {
                send_response(fd, 408, body);
            } else {
                send_retry_response(fd, body);
            }
        }
        return;
    }
    swarm_request_guard swarm_guard;
    swarm_guard.state = state;
    swarm_guard.context = &swarm_context;
    active_request_guard active_guard(state, response_sequence, session_id);
    std::unique_lock<std::mutex> request_execution_lock(state->request_execution_mutex);
    if (payload.has("steer")) {
        send_response(fd, 400, ajson_dumps(error_json(
                "Activation overrides are not available in this public build", "invalid_request_error")));
        return;
    }

    const auto request_begin = std::chrono::steady_clock::now();
    const resource_snapshot resources_before = capture_resource_snapshot(state);
    live_chat_stream live_stream;
    live_responses_stream responses_stream;
    live_anthropic_stream anthropic_stream;
    http_cancel_probe cancel_probe;
    live_generation_cancel_probe stream_cancel;
    cancel_probe.fd = fd;
    cancel_probe.state = state;
    cancel_probe.task_id = swarm_context.task_id;
    if (swarm_metadata.deadline) {
        cancel_probe.has_deadline = true;
        cancel_probe.deadline = swarm_context.submitted_at + *swarm_metadata.deadline;
    }
    generation_stream_sink stream_sink;
    stream_sink.progress = active_guard.progress;
    stream_sink.codex_target_prefill = codex_provider;
    if (codex_provider && no_think && !context_compacted)
        stream_sink.codex_replay_tools = &normalized_tools;
    stream_sink.user = &cancel_probe;
    stream_sink.cancelled = &http_client_cancelled;
    stream_sink.cancel_user = &cancel_probe;
    if (live_chat_stream_requested) {
        if (!start_live_chat_stream(
                &live_stream, fd, response_sequence, session_id)) {
            close_live_chat_stream(&live_stream);
            return;
        }
        stream_sink.user = &live_stream;
        stream_sink.emit = &live_chat_emit_token;
        stream_cancel.stream_open = &live_stream.open;
        stream_cancel.http = &cancel_probe;
        stream_cancel.progress_user = &live_stream;
        stream_cancel.progress = [](void *p) {
            return live_chat_send_chunk(static_cast<live_chat_stream *>(p), ajson::jobj());
        };
        stream_sink.cancelled = &live_generation_cancelled;
        stream_sink.cancel_user = &stream_cancel;
    } else if (live_responses_stream_requested) {
        if (!start_live_responses_stream(
                &responses_stream, fd, response_sequence, session_id, codex_provider ? &codex_wire : nullptr)) {
            close_live_responses_stream(&responses_stream);
            return;
        }
        stream_sink.user = &responses_stream;
        responses_stream.progress = active_guard.progress;
        stream_sink.emit = &live_responses_emit_token;
        stream_cancel.stream_open = &responses_stream.open;
        stream_cancel.http = &cancel_probe;
        stream_cancel.progress_user = &responses_stream;
        stream_cancel.progress = &live_responses_progress;
        stream_sink.cancelled = &live_generation_cancelled;
        stream_sink.cancel_user = &stream_cancel;
    } else if (live_anthropic_stream_requested) {
        if (!start_live_anthropic_stream(
                &anthropic_stream, fd, response_sequence, original_prompt_tokens,
                session_id)) {
            close_live_anthropic_stream(&anthropic_stream);
            return;
        }
        stream_sink.user = &anthropic_stream;
        stream_sink.emit = &live_anthropic_emit_token;
        stream_cancel.stream_open = &anthropic_stream.open;
        stream_cancel.http = &cancel_probe;
        stream_cancel.progress_user = &anthropic_stream;
        stream_cancel.progress = [](void *p) {
            return live_anthropic_send_event(static_cast<live_anthropic_stream *>(p), "ping", ajson::jobj());
        };
        stream_sink.cancelled = &live_generation_cancelled;
        stream_sink.cancel_user = &stream_cancel;
    }
    const int rc = native_generate_with_thinking(
            state, prompt_ids, plan, original_prompt_tokens, cache_session_id, session_profile,
            &session_suffix_ids, allow_stateful_session_resume,
            media_request,
            &sampling, &result, &failure_stage,
            &stream_sink, requested_speculative_mode, speculative_max_commit_tokens);
    if (stream_cancel.progress_failed) {
        failure_stage = "stream_progress_io";
        // No second error/terminal event after the transport failed.
        live_stream.open = false;
        responses_stream.io_failed = true;
        responses_stream.open = false;
        anthropic_stream.open = false;
    }
    const resource_snapshot resources_after = capture_resource_snapshot(state);
    const double request_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - request_begin).count();
    request_execution_lock.unlock();
    if (rc != AXIOM_OK) {
        const std::string message =
                std::string(state->streaming_kv ? "native paged runtime failed at " :
                                                   "native speculative graph failed at ") +
                (failure_stage.empty() ? "request_validation" : failure_stage) + ": " +
                axiom_status_string(rc) + "; scalar fallback is disabled";
        if (http_client_cancelled(&cancel_probe)) {
            swarm_guard.outcome = cancel_probe.has_deadline &&
                    axiom::qwen38::swarm_clock::now() >= cancel_probe.deadline
                    ? axiom::qwen38::swarm_task_outcome::timed_out
                    : axiom::qwen38::swarm_task_outcome::cancelled;
            swarm_guard.message = failure_stage.empty()
                    ? "swarm cancellation" : failure_stage;
        } else {
            swarm_guard.outcome = axiom::qwen38::swarm_task_outcome::failed;
            swarm_guard.message = message;
        }
        if (live_chat_stream_requested) {
            (void)live_chat_send_error(&live_stream, message);
            (void)live_chat_send_data(&live_stream, "[DONE]");
            close_live_chat_stream(&live_stream);
            return;
        }
        if (live_responses_stream_requested) {
            const bool codex_deadline = codex_provider && cancel_probe.has_deadline &&
                    axiom::qwen38::swarm_clock::now() >= cancel_probe.deadline;
            (void)live_responses_send_error(&responses_stream,
                    codex_deadline ? "Codex request deadline exceeded before generation completed" : message,
                    codex_deadline ? "request_timeout" : "server_error");
            if (!codex_provider && responses_stream.open && !responses_stream.io_failed)
                (void)write_http_chunk(responses_stream.fd, "data: [DONE]\n\n");
            close_live_responses_stream(&responses_stream);
            return;
        }
        if (live_anthropic_stream_requested) {
            (void)live_anthropic_send_error(&anthropic_stream, message);
            close_live_anthropic_stream(&anthropic_stream);
            return;
        }
        const int response_status = rc == AXIOM_ERR_BUDGET ? 400 : 500;
        send_response(fd, response_status, ajson_dumps(error_json(message, "server_error")));
        return;
    }
    result.session_id = session_id;
    result.speculative_fingerprint = state->speculative_fingerprint;
    result.speculative_qualified = state->speculative_qualified;
    if (state->speculative_qualified) {
        result.speculative_qualification_evidence_sha256 =
                state->speculative_qualification.evidence_sha256;
        result.speculative_qualification_source_commit =
                state->speculative_qualification.source_commit;
    }
    result.reasoning_effort = effective_reasoning_effort(state, payload);
    result.prompt_tokens = original_prompt_tokens;
    result.original_prompt_tokens = original_prompt_tokens;
    result.context_compacted = context_compacted;
    if (tools_enabled || codex_provider) {
        tool_result = parse_native_tool_calls(result.text, normalized_tools, plan.enabled);
        if (codex_provider && tool_result.status != "fail") {
            std::vector<std::string> call_names;
            for (const auto &call : tool_result.calls) {
                const auto *function = call.get("function");
                if (!function) continue;
                call_names.push_back(axiom_codex::string_field(*function, "name"));
                ajson arguments; std::string validation_error;
                if (!ajson_parse(axiom_codex::string_field(*function, "arguments"), arguments, validation_error) ||
                    !codex_wire.validate_arguments(axiom_codex::string_field(*function, "name"), arguments, &validation_error)) {
                    tool_result.status = "fail";
                    tool_result.error = "original Codex schema validation failed: " + validation_error;
                    break;
                }
            }
            if (tool_result.status != "fail" && !codex_wire.validate_calls(call_names, &tool_result.error))
                tool_result.status = "fail";
        }
        if (tool_result.status == "fail") {
            const std::string message =
                    "Qwen emitted an invalid native tool call: " + tool_result.error;
            if (live_chat_stream_requested) {
                (void)live_chat_send_error(&live_stream, message);
                (void)live_chat_send_data(&live_stream, "[DONE]");
                close_live_chat_stream(&live_stream);
                return;
            }
            if (live_responses_stream_requested) {
                (void)live_responses_send_error(&responses_stream, message);
                if (!codex_provider && responses_stream.open && !responses_stream.io_failed)
                    (void)write_http_chunk(responses_stream.fd, "data: [DONE]\n\n");
                close_live_responses_stream(&responses_stream);
                return;
            }
            if (live_anthropic_stream_requested) {
                (void)live_anthropic_send_error(&anthropic_stream, message);
                close_live_anthropic_stream(&anthropic_stream);
                return;
            }
            send_response(fd, 502, ajson_dumps(error_json(
                    message, "model_tool_call_error")));
            return;
        }
    }
    record_token_telemetry(
            state, response_sequence, session_id, result, selected_context, request_seconds,
            resources_before, resources_after);
    std::fprintf(stderr,
                 "axiom-qwen38-api: request=%llu prompt=%u completion=%zu cycles=%llu "
                 "accepted=%llu proposed=%llu acceptance=%.6f decode_tok_s=%.3f "
                 "visible_tok_s=%.3f mode=%s/%s path=%s compacted=%u original_prompt=%u\n",
                 static_cast<unsigned long long>(response_sequence), result.prompt_tokens,
                 result.ids.size(), static_cast<unsigned long long>(result.graph_cycles),
                 static_cast<unsigned long long>(result.accepted_draft_tokens),
                 static_cast<unsigned long long>(result.proposed_draft_tokens),
                 result.acceptance, result.decode_tokens_per_second,
                 result.visible_tokens_per_second,
                 result.speculative_mode_requested.c_str(),
                 result.speculative_mode_effective.c_str(),
                 result.decode_path.c_str(), result.context_compacted ? 1u : 0u,
                 result.original_prompt_tokens);
    if (live_chat_stream_requested) {
        attach_swarm_response(
                &live_stream.metadata, swarm_context,
                axiom::qwen38::swarm_clock::now());
        const bool sent = finish_live_chat_stream(
                &live_stream, result, tools_enabled ? &tool_result : nullptr);
        swarm_guard.finish(
                sent ? axiom::qwen38::swarm_task_outcome::succeeded
                     : axiom::qwen38::swarm_task_outcome::cancelled,
                result.ids.size(), sent ? "completed" : "client_disconnect");
        if (!sent) {
            close_live_chat_stream(&live_stream);
        }
        return;
    }
    if (live_anthropic_stream_requested) {
        attach_swarm_response(
                &anthropic_stream.metadata, swarm_context,
                axiom::qwen38::swarm_clock::now());
        const bool sent = finish_live_anthropic_stream(
                &anthropic_stream, result,
                tools_enabled ? &tool_result : nullptr);
        swarm_guard.finish(
                sent ? axiom::qwen38::swarm_task_outcome::succeeded
                     : axiom::qwen38::swarm_task_outcome::cancelled,
                result.ids.size(), sent ? "completed" : "client_disconnect");
        if (!sent) {
            close_live_anthropic_stream(&anthropic_stream);
        }
        return;
    }
    ajson response = responses_api
            ? responses_response(result, response_sequence,
                                 effective_reasoning_effort(state, payload),
                                 &normalized_tools,
                                 tools_enabled ? &tool_result : nullptr, codex_provider)
            : anthropic
            ? anthropic_response(result, response_sequence,
                                 tools_enabled ? &tool_result : nullptr)
            : completion_response(result, chat, response_sequence,
                                  tools_enabled ? &tool_result : nullptr);
    attach_swarm_response(
            &response, swarm_context, axiom::qwen38::swarm_clock::now());
    const std::string response_status = codex_provider
            ? axiom_codex::string_field(response, "status") : "completed";
    const auto response_outcome = response_status == "incomplete"
            ? axiom::qwen38::swarm_task_outcome::budget_exhausted
            : response_status == "failed" ? axiom::qwen38::swarm_task_outcome::failed
            : axiom::qwen38::swarm_task_outcome::succeeded;
    if (live_responses_stream_requested) {
        const bool sent = finish_live_responses_stream(
                &responses_stream, response, result,
                tools_enabled ? &tool_result : nullptr, &normalized_tools);
        swarm_guard.finish(
                sent ? response_outcome
                     : axiom::qwen38::swarm_task_outcome::cancelled,
                result.ids.size(), sent ? response_status.c_str() : "client_disconnect");
        if (!sent) {
            close_live_responses_stream(&responses_stream);
        }
        return;
    }
    if (codex_provider) codex_wire.restore(response);
    send_session_response(fd, 200, ajson_dumps(response), session_id);
    swarm_guard.finish(
            response_outcome, result.ids.size(), response_status.c_str());
}

int open_listener(const listen_spec &spec) {
    struct in_addr address {};
    if (inet_pton(AF_INET, spec.host.c_str(), &address) != 1) return -1;
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
    const int yes = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) != 0) {
        close(fd);
        return -1;
    }
    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr = address;
    addr.sin_port = htons(spec.port);
    if (bind(fd, reinterpret_cast<const struct sockaddr *>(&addr), sizeof(addr)) != 0 || listen(fd, 16) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

void close_listeners(std::vector<int> *listeners) {
    if (!listeners) return;
    for (const int fd : *listeners) {
        if (fd >= 0) close(fd);
    }
    listeners->clear();
}

bool open_listeners_atomic(
        const std::vector<listen_spec> &specs, std::vector<int> *out) {
    if (!out || !out->empty() || specs.empty() || specs.size() > kMaxListeners) {
        return false;
    }
    std::vector<int> opened;
    try {
        opened.reserve(specs.size());
    } catch (...) {
        return false;
    }
    for (const listen_spec &spec : specs) {
        const int fd = open_listener(spec);
        if (fd < 0) {
            close_listeners(&opened);
            return false;
        }
        opened.push_back(fd);
    }
    *out = std::move(opened);
    return true;
}

int run_stream_progress_self_test() {
    int pair[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) return 1;
    const auto finish = [&](const char *failure) {
        if (pair[0] >= 0) close(pair[0]);
        if (pair[1] >= 0) close(pair[1]);
        std::printf("SSE_PROGRESS_TEST %s%s\n", failure ? "FAIL " : "PASS cases=13 ", failure ? failure : "");
        return failure ? 1 : 0;
    };
    const auto read_wire = [&]() {
        std::string wire;
        char bytes[4096];
        for (;;) {
            const ssize_t n = recv(pair[1], bytes, sizeof(bytes), MSG_DONTWAIT);
            if (n <= 0) break;
            wire.append(bytes, static_cast<size_t>(n));
        }
        return wire;
    };
    live_responses_stream stream;
    if (!start_live_responses_stream(&stream, pair[0], 42, "qa-progress")) return finish("start");
    const auto initial_wire = read_wire();
    stream.progress = std::make_shared<axiom::qwen38::request_progress>();
    if (!stream.progress->prefill(55503u, 65536u, 68575u)) return finish("observed-progress-fixture");
    const auto observed = stream.progress->read();
    const std::string stamp = "\"created_at\":" + std::to_string(stream.created_at);
    if (initial_wire.find(stamp) == std::string::npos) return finish("initial-timestamp");
    http_cancel_probe http;
    http.fd = pair[0];
    live_generation_cancel_probe probe;
    probe.stream_open = &stream.open;
    probe.http = &http;
    probe.progress_user = &stream;
    probe.progress = &live_responses_progress;
    const auto zero = std::chrono::steady_clock::time_point{};
    probe.last_progress = zero;
    if (live_generation_cancelled_at(&probe, zero + std::chrono::seconds(14)) || !read_wire().empty()) return finish("early-write");
    if (live_generation_cancelled_at(&probe, zero + std::chrono::seconds(15))) return finish("progress");
    const auto wire = read_wire();
    if (wire.find("\"prefill_tokens_processed\":10033") == std::string::npos ||
        wire.find("\"prefill_tokens_total\":13072") == std::string::npos ||
        wire.find("\"cached_tokens\":55503") == std::string::npos ||
        stream.progress->read().revision != observed.revision) return finish("heartbeat-must-not-invent-progress");
    if (wire.find("event: response.in_progress\n") == std::string::npos ||
        wire.find("resp-axiom-42") == std::string::npos || wire.find(stamp) == std::string::npos || wire.find("\"sequence_number\":2") == std::string::npos ||
        wire.find("output_text.delta") != std::string::npos || stream.message_started || !stream.raw_text.empty()) return finish("identity-or-fake-output");
    if (live_generation_cancelled_at(&probe, zero + std::chrono::seconds(29)) || !read_wire().empty()) return finish("rate");
    if (live_generation_cancelled_at(&probe, zero + std::chrono::seconds(30)) || read_wire().empty()) return finish("repeat");
    stream.terminal_sent = true;
    if (!live_generation_cancelled_at(&probe, zero + std::chrono::seconds(45)) || !read_wire().empty()) return finish("after-terminal");
    stream.terminal_sent = false;
    if (!live_generation_cancelled_at(&probe, zero + std::chrono::seconds(60)) || !read_wire().empty()) return finish("failure-latch");
    probe.progress_failed = false;
    http.has_deadline = true;
    http.deadline = zero + std::chrono::seconds(70);
    if (!live_generation_cancelled_at(&probe, zero + std::chrono::seconds(75)) || !read_wire().empty()) return finish("deadline");
    http.has_deadline = false;
    if (shutdown(pair[0], SHUT_WR) != 0 || http_client_cancelled_at(&http, zero + std::chrono::seconds(90))) return finish("half-close-fixture");
    if (!live_generation_cancelled_at(&probe, zero + std::chrono::seconds(90)) || !stream.io_failed || !probe.progress_failed || !read_wire().empty()) return finish("real-write-failure");
    if (!live_generation_cancelled_at(&probe, zero + std::chrono::seconds(105)) || !read_wire().empty()) return finish("real-write-failure-latched");
    close(pair[1]); pair[1] = -1;
    if (!live_generation_cancelled_at(&probe, zero + std::chrono::seconds(120))) return finish("disconnect");
    server_state state;
    auto old = std::make_unique<active_request_guard>(&state, 100u, "old");
    old->progress->prefill(10, 15, 20);
    auto successor = std::make_unique<active_request_guard>(&state, 101u, "new");
    old.reset();
    if (state.active_request_sequence != 101u || state.active_request_progress != successor->progress ||
        state.active_request_progress->read().revision != 0u) return finish("successor-isolation");
    successor.reset();
    if (state.active_request_progress || state.active_request_sequence != 0u) return finish("idle-clears-progress");
    return finish(nullptr);
}

int run_listener_self_test() {
    const auto fail_case = [](const char *name) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: listener self-test FAIL case=%s\n", name);
        return 1;
    };

    std::vector<listen_spec> parsed;
    if (!parse_listeners("127.0.0.1:8015", &parsed) || parsed.size() != 1u ||
        format_listener_list(parsed) != "127.0.0.1:8015") {
        return fail_case("single_backward_compatible");
    }
    if (!parse_listeners(kDefaultListenSpec, &parsed) || parsed.size() != 1u ||
        format_listener_list(parsed) != kDefaultListenSpec) {
        return fail_case("loopback_default");
    }
    constexpr const char *valid_eight =
            "127.0.0.1:8101,127.0.0.1:8102,127.0.0.1:8103,127.0.0.1:8104,"
            "127.0.0.1:8105,127.0.0.1:8106,127.0.0.1:8107,127.0.0.1:8108";
    if (!parse_listeners(valid_eight, &parsed) || parsed.size() != kMaxListeners) {
        return fail_case("maximum_eight");
    }
    constexpr const char *invalid_cases[] = {
        "",
        ",127.0.0.1:8015",
        "127.0.0.1:8015,",
        "127.0.0.1:8015,,127.0.0.2:8015",
        "127.0.0.1:8015,127.0.0.1:8015",
        "127.0.0.1:0",
        "localhost:8015",
        "0.0.0.0:8015",
        "127.0.0.1:8101,127.0.0.1:8102,127.0.0.1:8103,127.0.0.1:8104,"
        "127.0.0.1:8105,127.0.0.1:8106,127.0.0.1:8107,127.0.0.1:8108,"
        "127.0.0.1:8109",
    };
    for (const char *invalid : invalid_cases) {
        const std::vector<listen_spec> before = parsed;
        if (parse_listeners(invalid, &parsed) || parsed.size() != before.size()) {
            return fail_case("invalid_or_transactional_parse");
        }
    }

    /* Exercise successful multi-fd polling without loading CUDA or a model. */
    const std::vector<listen_spec> ephemeral_specs = {
        {"127.0.0.1", 0u}, {"127.0.0.2", 0u},
    };
    std::vector<int> listeners;
    std::vector<int> clients;
    std::vector<int> accepted;
    const auto cleanup = [&] {
        close_listeners(&accepted);
        close_listeners(&clients);
        close_listeners(&listeners);
    };
    if (!open_listeners_atomic(ephemeral_specs, &listeners) ||
        listeners.size() != ephemeral_specs.size()) {
        cleanup();
        return fail_case("atomic_open_success");
    }
    for (const int listener : listeners) {
        struct sockaddr_in address {};
        socklen_t address_size = sizeof(address);
        if (getsockname(listener, reinterpret_cast<struct sockaddr *>(&address),
                        &address_size) != 0) {
            cleanup();
            return fail_case("getsockname");
        }
        const int client = socket(AF_INET, SOCK_STREAM, 0);
        if (client < 0 ||
            connect(client, reinterpret_cast<const struct sockaddr *>(&address),
                    sizeof(address)) != 0) {
            if (client >= 0) close(client);
            cleanup();
            return fail_case("connect_each_listener");
        }
        clients.push_back(client);
    }
    std::vector<struct pollfd> ready;
    ready.reserve(listeners.size());
    for (const int listener : listeners) ready.push_back({listener, POLLIN, 0});
    const int poll_rc = poll(ready.data(), static_cast<nfds_t>(ready.size()), 1000);
    if (poll_rc != static_cast<int>(ready.size())) {
        cleanup();
        return fail_case("poll_all_listeners");
    }
    for (const struct pollfd &entry : ready) {
        if ((entry.revents & POLLIN) == 0) {
            cleanup();
            return fail_case("poll_missing_readiness");
        }
        const int peer = accept(entry.fd, nullptr, nullptr);
        if (peer < 0) {
            cleanup();
            return fail_case("accept_each_listener");
        }
        accepted.push_back(peer);
    }
    cleanup();

    /* Force the second bind to fail and prove no partially-open set escapes. */
    const int blocker = open_listener({"127.0.0.1", 0u});
    if (blocker < 0) return fail_case("atomic_failure_blocker");
    struct sockaddr_in blocked_address {};
    socklen_t blocked_size = sizeof(blocked_address);
    if (getsockname(blocker, reinterpret_cast<struct sockaddr *>(&blocked_address),
                    &blocked_size) != 0) {
        close(blocker);
        return fail_case("atomic_failure_getsockname");
    }
    const std::vector<listen_spec> failure_specs = {
        {"127.0.0.2", 0u},
        {"127.0.0.1", ntohs(blocked_address.sin_port)},
    };
    std::vector<int> failed_open;
    const bool unexpected_success = open_listeners_atomic(failure_specs, &failed_open);
    close(blocker);
    if (unexpected_success || !failed_open.empty()) {
        close_listeners(&failed_open);
        return fail_case("atomic_failure_cleanup");
    }

    std::printf(
            "axiom-qwen38-api: listener self-test PASS parser=pass max=8 "
            "duplicates=reject wildcard=reject atomic=pass poll=pass\n");
    return 0;
}

void destroy_server_state(server_state *state) {
    if (!state) return;
    /* Drain deferred KV jobs before destroying the model, draft or tier they
     * reference.  This preserves the KV-commit -> manifest-publish order even
     * during a clean service shutdown. */
    stop_persistence_worker(state);
    stop_session_maintenance_worker(state);
    if (state->resident_graph_session) {
        (void)destroy_request_session(state->resident_graph_session);
        delete state->resident_graph_session;
        state->resident_graph_session = nullptr;
    }
    axiom_qwen38_dspark_destroy(state->draft);
    state->draft = nullptr;
    axiom_model_close(state->draft_checkpoint);
    state->draft_checkpoint = nullptr;
    axiom_runtime_destroy(state->draft_runtime);
    state->draft_runtime = nullptr;
    axiom_qwen38_mtp_destroy(state->mtp);
    state->mtp = nullptr;
    axiom_runtime_destroy(state->mtp_runtime);
    state->mtp_runtime = nullptr;
    state->native_mtp_loaded = false;
    axiom_qwen38_vision_destroy(state->vision);
    state->vision = nullptr;
    axiom_qwen38_model_destroy(state->model);
    state->model = nullptr;
    axiom_qwen38_kv_tier_destroy(state->kv_tier);
    state->kv_tier = nullptr;
    axiom_tokenizer_close(state->tokenizer);
    state->tokenizer = nullptr;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test-stream-progress") == 0) {
        return run_stream_progress_self_test();
    }
    if (argc == 2 && std::strcmp(argv[1], "--self-test-utf8") == 0) {
        return run_utf8_normalization_self_test();
    }
    if (argc == 2 && std::strcmp(argv[1], "--self-test-speculative-routing") == 0) {
        return run_speculative_routing_self_test();
    }
    if (argc == 2 && std::strcmp(argv[1], "--self-test-listeners") == 0) {
        return run_listener_self_test();
    }
    if (argc == 2 && std::strcmp(argv[1], "--self-test-tool-contract") == 0) {
        return run_native_tool_contract_self_test();
    }
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr,
                     "usage: %s TARGET_DIR DSPARK_DIR [IP:PORT[,IP:PORT...]] [MAX_CONTEXT]\n"
                     "       %s --self-test-utf8\n"
                     "       %s --self-test-speculative-routing\n"
                     "       %s --self-test-listeners\n"
                     "       %s --self-test-tool-contract\n",
                     argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    const char *listen_text = argc >= 4 ? argv[3] : kDefaultListenSpec;
    std::vector<listen_spec> listen_specs;
    if (!parse_listeners(listen_text, &listen_specs)) {
        return fail(
                "listen addresses must be 1-8 unique explicit IPv4 IP:PORT values");
    }
    uint32_t max_context = kDefaultMaxContext;
    if (argc == 5 && (!parse_u32(argv[4], &max_context) ||
                      max_context < AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
                      max_context > kMaxNativeContext)) {
        return fail("MAX_CONTEXT must be in [8, 1048576]");
    }
    speculative_engine configured_engine = speculative_engine::dspark;
    const char *native_mtp_text = std::getenv("AXIOM_QWEN38_NATIVE_MTP");
    if (native_mtp_text && native_mtp_text[0] != '\0') {
        if (std::strcmp(native_mtp_text, "1") == 0) {
            configured_engine = speculative_engine::native_mtp;
        } else if (std::strcmp(native_mtp_text, "0") != 0) {
            return fail("AXIOM_QWEN38_NATIVE_MTP must be exactly 0 or 1",
                        AXIOM_ERR_INVALID_ARGUMENT);
        }
    }
    const bool streaming_kv = env_enabled("AXIOM_QWEN38_KV_STREAMING");
    const bool temporal_hot = streaming_kv && env_enabled("AXIOM_QWEN38_KV_TEMPORAL8");
    if (configured_engine == speculative_engine::native_mtp &&
        streaming_kv && !temporal_hot) {
        return fail("native MTP requires AXIOM_QWEN38_KV_TEMPORAL8=1",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    if (configured_engine == speculative_engine::dspark &&
        (!streaming_kv || temporal_hot) &&
        axiom_qwen38_speculative_backend_available() != 1) {
        return fail("DSpark full-device backend unavailable; scalar fallback is disabled",
                    AXIOM_ERR_NOT_IMPLEMENTED);
    }
    if (setenv("AXIOM_MODEL_NO_RESIDENT", "1", 1) != 0) {
        return fail("disable duplicate generic residency", AXIOM_ERR_IO);
    }

    server_state state{};
    uint32_t manifest_version = 1u;
    const char *manifest_version_text = std::getenv("AXIOM_QWEN38_MANIFEST_VERSION");
    if ((manifest_version_text && !parse_u32(manifest_version_text, &manifest_version)) ||
        !state.session_store.set_manifest_version(manifest_version)) {
        return fail("AXIOM_QWEN38_MANIFEST_VERSION must be 1 or 2",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    state.engine = configured_engine;
    state.max_context = max_context;
    state.listen_addresses.reserve(listen_specs.size());
    for (const listen_spec &spec : listen_specs) {
        state.listen_addresses.push_back(format_listener(spec));
    }
    state.default_context = kDefaultRequestContext;
    const char *vision_patch_text = std::getenv("AXIOM_QWEN38_VISION_MAX_PATCH_TOKENS");
    if (vision_patch_text && vision_patch_text[0] &&
        (!parse_u32(vision_patch_text, &state.vision_max_patch_tokens) ||
         state.vision_max_patch_tokens == 0u || state.vision_max_patch_tokens > 16384u)) {
        return fail("AXIOM_QWEN38_VISION_MAX_PATCH_TOKENS must be in [1,16384]",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    state.streaming_kv = streaming_kv;
    state.temporal_hot = temporal_hot;
    const char *speculative_context_text =
            std::getenv("AXIOM_QWEN38_SPECULATIVE_CONTEXT_TOKENS");
    if (speculative_context_text && speculative_context_text[0] &&
        (!parse_u32(speculative_context_text, &state.speculative_context_tokens) ||
         state.speculative_context_tokens < kMinSpeculativeContextTokens ||
         state.speculative_context_tokens > kMaxSpeculativeContextTokens ||
         state.speculative_context_tokens % AXIOM_QWEN38_KV_TIER_PAGE_TOKENS != 0u)) {
        return fail(
                "AXIOM_QWEN38_SPECULATIVE_CONTEXT_TOKENS must be page-aligned in [2048,8192]",
                AXIOM_ERR_INVALID_ARGUMENT);
    }
    if (temporal_hot && state.speculative_context_tokens > max_context) {
        return fail(
                "speculative context cannot exceed MAX_CONTEXT",
                AXIOM_ERR_INVALID_ARGUMENT);
    }
    const char *fast_graph_text =
            std::getenv("AXIOM_QWEN38_MTP_FAST_GRAPH_MIN_POSITION");
    if (fast_graph_text && fast_graph_text[0] &&
        (!parse_u32(fast_graph_text,
                    &state.native_mtp_fast_graph_min_position) ||
         state.native_mtp_fast_graph_min_position >
                 state.speculative_context_tokens)) {
        return fail(
                "AXIOM_QWEN38_MTP_FAST_GRAPH_MIN_POSITION must be in the speculative context",
                AXIOM_ERR_INVALID_ARGUMENT);
    }
    state.no_think = env_enabled("AXIOM_QWEN38_NO_THINK");
    state.context_compaction = !env_disabled("AXIOM_QWEN38_CONTEXT_COMPACTION");
    const char *kv_path = std::getenv("AXIOM_QWEN38_KV_TIER_PATH");
    state.kv_base_path = kv_path && kv_path[0] ? kv_path : "";
    if (streaming_kv && state.kv_base_path.empty()) {
        return fail("AXIOM_QWEN38_KV_TIER_PATH is required for paged runtime",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    axiom::qwen38::spec_identity::runtime_contract speculative_runtime;
    speculative_runtime.axiom_abi_version = AXIOM_ABI_VERSION;
    speculative_runtime.target_model_dspark_abi_version =
            AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    speculative_runtime.target_capture_abi_version =
            AXIOM_QWEN38_MODEL_DSPARK_CAPTURE_ABI_VERSION;
    speculative_runtime.speculative_target_abi_version =
            AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION;
    speculative_runtime.dspark_compute_layout_version =
            AXIOM_QWEN38_DSPARK_COMPUTE_LAYOUT_VERSION;
    speculative_runtime.dspark_compute_device_abi_version =
            AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION;
    speculative_runtime.target_tap_count =
            AXIOM_QWEN38_MODEL_DSPARK_TARGET_TAP_COUNT;
    speculative_runtime.temporal_verify_width =
            AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
    speculative_runtime.draft_block_size = AXIOM_QWEN38_DSPARK_BLOCK_SIZE;
    std::string speculative_identity_error;
    if (!axiom::qwen38::spec_identity::compute(
            argv[1], argv[2], speculative_runtime,
            &state.speculative_identity, &speculative_identity_error)) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative identity failed: %s\n",
                     speculative_identity_error.c_str());
        return fail("speculative_identity", AXIOM_ERR_IO);
    }
    if (state.engine == speculative_engine::native_mtp) {
        state.speculative_fingerprint =
                "mtp1:" + state.speculative_identity.components.target_sha256 +
                ":compute" + std::to_string(
                        AXIOM_QWEN38_MTP_COMPUTE_DEVICE_ABI_VERSION) +
                ":controller" + std::to_string(
                        AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION);
    } else {
        state.speculative_fingerprint = state.speculative_identity.fingerprint;
    }
    const char *contract_path_env =
            std::getenv("AXIOM_QWEN38_SPECULATIVE_CONTRACT_PATH");
    if (contract_path_env && contract_path_env[0]) {
        state.speculative_contract_path = contract_path_env;
    } else {
        state.speculative_contract_path = argv[2];
        if (!state.speculative_contract_path.empty() &&
            state.speculative_contract_path.back() != '/') {
            state.speculative_contract_path.push_back('/');
        }
        state.speculative_contract_path +=
                axiom::qwen38::spec_identity::kSidecarFilename;
    }
    state.speculative_diagnostic_override = env_enabled(
            "AXIOM_QWEN38_DIAGNOSTIC_ALLOW_UNQUALIFIED_SPECULATIVE");
    if (state.engine == speculative_engine::native_mtp) {
        /* Native-MTP qualification is carried by the release graph/perf
         * evidence, never by the incompatible DSpark pairing sidecar. */
        state.speculative_qualified = false;
    } else if (axiom::qwen38::spec_identity::validate_sidecar(
            state.speculative_contract_path, state.speculative_identity,
            &state.speculative_qualification, &speculative_identity_error)) {
        state.speculative_qualified = true;
    } else if (!state.speculative_diagnostic_override) {
        std::fprintf(stderr,
                     "axiom-qwen38-api: speculative qualification failed: %s\n",
                     speculative_identity_error.c_str());
        return fail("speculative_qualification", AXIOM_ERR_IO);
    } else {
        std::fprintf(stderr,
                     "axiom-qwen38-api: WARNING unqualified speculative pairing "
                     "allowed only by diagnostic override: %s fingerprint=%s\n",
                     speculative_identity_error.c_str(),
                     state.speculative_fingerprint.c_str());
    }
    /* Durable session/KV namespaces are derived from measured bytes, never
     * operator-provided labels or mutable paths. A new target, tokenizer,
     * draft, template or native ABI therefore cannot reuse incompatible KV. */
    state.target_identity = "sha256:" +
            state.speculative_identity.components.target_sha256;
    state.dspark_identity = state.engine == speculative_engine::native_mtp
            ? state.speculative_fingerprint
            : "sha256:" + state.speculative_identity.components.draft_sha256;
    const char *hot_text = std::getenv("AXIOM_QWEN38_KV_HOT_PAGES");
    if (hot_text && hot_text[0] &&
        (!parse_u32(hot_text, &state.kv_hot_pages) ||
         state.kv_hot_pages == 0u || state.kv_hot_pages > 256u)) {
        return fail("AXIOM_QWEN38_KV_HOT_PAGES must be in [1,256]",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    if ((!hot_text || !hot_text[0]) && streaming_kv) {
        const std::string canonical_hot_pages = std::to_string(state.kv_hot_pages);
        if (setenv("AXIOM_QWEN38_KV_HOT_PAGES",
                   canonical_hot_pages.c_str(), 1) != 0) {
            return fail("set canonical KV hot-page count", AXIOM_ERR_IO);
        }
    }
    const char *session_ttl_text =
            std::getenv("AXIOM_QWEN38_SESSION_TTL_SECONDS");
    if (session_ttl_text && session_ttl_text[0] &&
        (!parse_u64(session_ttl_text, &state.session_ttl_seconds) ||
         state.session_ttl_seconds >
                 static_cast<uint64_t>(std::numeric_limits<long long>::max()))) {
        return fail("AXIOM_QWEN38_SESSION_TTL_SECONDS must be in [0,LLONG_MAX]",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    const char *session_max_text =
            std::getenv("AXIOM_QWEN38_SESSION_MAX_NAMESPACES");
    if (session_max_text && session_max_text[0] &&
        !parse_u32(session_max_text, &state.session_max_namespaces)) {
        return fail("AXIOM_QWEN38_SESSION_MAX_NAMESPACES must be a non-negative integer",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    const char *session_gc_interval_text =
            std::getenv("AXIOM_QWEN38_SESSION_GC_INTERVAL_SECONDS");
    if (session_gc_interval_text && session_gc_interval_text[0] &&
        (!parse_u64(session_gc_interval_text, &state.session_gc_interval_seconds) ||
         state.session_gc_interval_seconds >
                 static_cast<uint64_t>(std::numeric_limits<long long>::max()))) {
        return fail("AXIOM_QWEN38_SESSION_GC_INTERVAL_SECONDS must be in [0,LLONG_MAX]",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    const char *sse_batch_text = std::getenv("AXIOM_QWEN38_SSE_BATCH_CYCLES");
    if (sse_batch_text && sse_batch_text[0] &&
        !parse_u32(sse_batch_text, &state.sse_batch_cycles)) {
        return fail("AXIOM_QWEN38_SSE_BATCH_CYCLES must be an integer",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    if (state.sse_batch_cycles == 0u || state.sse_batch_cycles > kDeviceChunkCycles) {
        return fail("AXIOM_QWEN38_SSE_BATCH_CYCLES must be in [1,33]",
                    AXIOM_ERR_INVALID_ARGUMENT);
    }
    const uint32_t logical_pages =
            (max_context + AXIOM_QWEN38_KV_TIER_PAGE_TOKENS - 1u) /
            AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
    if (state.kv_hot_pages == 0u) state.kv_hot_pages = 1u;
    if (state.kv_hot_pages > logical_pages) state.kv_hot_pages = logical_pages;
    state.kv_config_signature = std::string(kSessionConfigVersion) +
            ";model=" + kModelId + ";max=" + std::to_string(max_context) +
            ";page=" + std::to_string(AXIOM_QWEN38_KV_TIER_PAGE_TOKENS) +
            ";hot=" + std::to_string(state.kv_hot_pages) +
            ";speculative=" + std::to_string(state.speculative_context_tokens);
    /* Preserve the byte-for-byte legacy DSpark namespace. Native MTP needs a
     * discriminator because its layer count and persistent frontier differ,
     * but adding `engine=dspark` would silently orphan every ABI-v1 session. */
    if (state.engine == speculative_engine::native_mtp) {
        state.kv_config_signature += ";engine=native_mtp";
    }
    state.kv_config_signature +=
            ";specfp=" + state.speculative_fingerprint +
            ";yarn=" + (env_disabled("AXIOM_QWEN38_YARN") ? "0" : "1") +
            ";compact=" + (env_disabled("AXIOM_QWEN38_COMPACT_KV") ? "0" : "1");
    if (!state.kv_base_path.empty()) {
        state.session_store.configure(state.kv_base_path, max_context, 0u);
    }
    const std::string startup_profile = state.no_think ? "ultra-fast" : "medium";
    const auto startup_key = make_session_key(&state, kStartupSessionId, startup_profile);
    axiom::qwen38::qwen38_session_paths startup_paths;
    bool startup_manifest_exists = false;
    axiom::qwen38::qwen38_session_manifest startup_manifest;
    std::string startup_session_stage;
    int startup_session_rc = AXIOM_OK;
    if (!state.kv_base_path.empty()) {
        startup_session_rc = state.session_store.load(
                startup_key, &startup_manifest, &startup_paths,
                &startup_manifest_exists, &startup_session_stage);
        if (startup_session_rc == AXIOM_OK && !startup_manifest_exists) {
            startup_session_rc = state.session_store.prepare(
                    startup_key, 1u, &startup_paths, &startup_session_stage);
        }
    }
    if (startup_session_rc != AXIOM_OK) {
        return fail(
                startup_session_stage.empty() ? "session_manifest_load" :
                                                 startup_session_stage.c_str(),
                startup_session_rc);
    }
    std::string kv_tier_stage;
    int rc = state.kv_base_path.empty()
            ? AXIOM_OK
            : create_kv_tier_at_path(
                    &state, startup_paths.tier_path, &state.kv_tier, &kv_tier_stage);
    if (rc != AXIOM_OK) {
        destroy_server_state(&state);
        return fail(kv_tier_stage.empty() ? "kv_tier_create" : kv_tier_stage.c_str(), rc);
    }
    axiom_tokenizer_config tokenizer_config{};
    tokenizer_config.abi_version = AXIOM_ABI_VERSION;
    tokenizer_config.path = argv[1];
    tokenizer_config.name = "qwen3.8";
    tokenizer_config.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;
    const char *init_stage = "tokenizer_open";
    rc = axiom_tokenizer_open(&state.tokenizer, &tokenizer_config);
    if (rc == AXIOM_OK) {
        init_stage = "tokenizer_info";
        state.tokenizer_info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_tokenizer_info_get(state.tokenizer, &state.tokenizer_info);
    }
    if (rc == AXIOM_OK) {
        init_stage = "target_create";
        rc = axiom_qwen38_model_create(argv[1], 0, max_context, &state.model);
    }
    if (rc == AXIOM_OK) {
        init_stage = "vision_create";
        rc = axiom_qwen38_vision_create(
                axiom_qwen38_model_checkpoint(state.model), 0,
                state.vision_max_patch_tokens, &state.vision);
    }
    if (rc == AXIOM_OK) {
        init_stage = "vision_info";
        state.vision_info.abi_version = AXIOM_QWEN38_VISION_ABI_VERSION;
        rc = axiom_qwen38_vision_info_get(state.vision, &state.vision_info);
    }
    if (rc == AXIOM_OK && state.streaming_kv) {
        init_stage = "target_kv_tier_bind";
        rc = axiom_qwen38_model_kv_tier_bind(state.model, state.kv_tier);
    }
    if (rc == AXIOM_OK && !state.kv_base_path.empty()) {
        state.session_store.configure(
                state.kv_base_path, max_context,
                axiom_qwen38_model_recurrent_state_bytes(),
                state.engine == speculative_engine::native_mtp
                ? sizeof(axiom_qwen38_mtp_speculative_persistent_state_v1)
                : 0u);
        bool verified_manifest_exists = false;
        axiom::qwen38::qwen38_session_manifest verified_manifest;
        axiom::qwen38::qwen38_session_paths verified_paths;
        std::string verify_stage;
        rc = state.session_store.load(
                startup_key, &verified_manifest, &verified_paths,
                &verified_manifest_exists, &verify_stage);
        if (rc == AXIOM_OK && verified_manifest_exists) {
            axiom_qwen38_kv_tier_info tier_info{};
            tier_info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
            rc = axiom_qwen38_kv_tier_info_get(state.kv_tier, &tier_info);
            if (rc == AXIOM_OK &&
                (verified_paths.tier_path != startup_paths.tier_path ||
                 tier_info.committed_tokens != verified_manifest.committed_tokens)) {
                rc = AXIOM_ERR_IO;
                verify_stage = "startup_session_watermark";
            }
            if (rc == AXIOM_OK) {
                {
                    std::lock_guard<std::mutex> status_lock(
                            state.session_status_mutex);
                    state.active_session_manifest = verified_manifest;
                    state.active_session_manifest_exists = true;
                    state.active_session_generation = verified_manifest.generation;
                }
                run_session_generation_gc(
                        &state, startup_key, verified_paths, "startup");
            }
        }
        if (rc != AXIOM_OK) {
            init_stage = verify_stage.empty() ? "session_manifest_verify" : verify_stage.c_str();
        }
        if (rc == AXIOM_OK) {
            {
                std::lock_guard<std::mutex> status_lock(
                        state.session_status_mutex);
                state.active_session_key = startup_key;
                state.active_session_paths = startup_paths;
                state.active_session_manifest_exists = verified_manifest_exists;
                state.active_session_generation = startup_paths.generation;
            }
            run_session_lifecycle_gc(
                    &state, startup_paths.namespace_id, true);
        }
    }
    std::string validation_stage;
    if (rc == AXIOM_OK && (!state.streaming_kv || state.temporal_hot)) {
        init_stage = "temporal8_validate";
        rc = validate_target_session(&state, &validation_stage);
    }
    axiom_config runtime_config{};
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = 0u;
    if (rc == AXIOM_OK && (!state.streaming_kv || state.temporal_hot)) {
        init_stage = state.engine == speculative_engine::native_mtp
                ? "native_mtp_runtime_create" : "draft_runtime_create";
        rc = axiom_runtime_create(&state.draft_runtime, &runtime_config);
    }
    axiom_model_config checkpoint_config{};
    checkpoint_config.abi_version = AXIOM_ABI_VERSION;
    checkpoint_config.path = argv[2];
    checkpoint_config.name = "qwen3.8-dspark";
    checkpoint_config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    checkpoint_config.placement.abi_version = AXIOM_ABI_VERSION;
    checkpoint_config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    if (rc == AXIOM_OK && (!state.streaming_kv || state.temporal_hot) &&
        state.engine == speculative_engine::dspark) {
        init_stage = "draft_model_open";
        rc = axiom_model_open(state.draft_runtime, &state.draft_checkpoint, &checkpoint_config);
    }
    if (rc == AXIOM_OK && (!state.streaming_kv || state.temporal_hot) &&
        state.engine == speculative_engine::dspark) {
        init_stage = "draft_load";
        rc = axiom_qwen38_dspark_load(
                state.draft_checkpoint, state.draft_runtime, 0, &state.draft);
    }
    if (rc == AXIOM_OK && (!state.streaming_kv || state.temporal_hot) &&
        state.engine == speculative_engine::native_mtp) {
        state.mtp_runtime = state.draft_runtime;
        state.draft_runtime = nullptr;
        init_stage = "native_mtp_load";
        rc = axiom_qwen38_mtp_load(
                axiom_qwen38_model_checkpoint(state.model), state.mtp_runtime,
                0, &state.mtp);
    }
    if (rc == AXIOM_OK && state.engine == speculative_engine::native_mtp) {
        init_stage = "native_mtp_info";
        state.mtp_info = axiom_qwen38_mtp_info{};
        state.mtp_info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_qwen38_mtp_info_get(state.mtp, &state.mtp_info);
        if (rc == AXIOM_OK &&
            (state.mtp_info.tensor_count !=
                     AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT ||
             state.mtp_info.resident_dtype != AXIOM_TENSOR_DTYPE_BF16 ||
             state.mtp_info.device_bytes == 0u)) {
            rc = AXIOM_ERR_RUNTIME;
        }
        state.native_mtp_loaded = rc == AXIOM_OK;
    }
    if (rc != AXIOM_OK) {
        destroy_server_state(&state);
        return fail(init_stage, rc);
    }
    if (!start_session_maintenance_worker(&state)) {
        destroy_server_state(&state);
        return fail("session_maintenance_worker_start", AXIOM_ERR_RUNTIME);
    }
    if (!start_persistence_worker(&state)) {
        destroy_server_state(&state);
        return fail("persistence_worker_start", AXIOM_ERR_RUNTIME);
    }
    std::vector<int> listeners;
    if (!open_listeners_atomic(listen_specs, &listeners)) {
        destroy_server_state(&state);
        return fail("atomic bind/listen", AXIOM_ERR_IO);
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);
    const std::string active_listeners = format_listener_list(listen_specs);
    // Prime before accepting requests; busy health reads use this snapshot.
    std::fprintf(stderr,
                 "axiom-qwen38-api: listening on %s without API key; "
                 "backend=%s engine=%s scalar_fallback=disabled max_context=%u kv_streaming=%u "
                 "no_think=%u speculative_qualified=%u fingerprint=%s\n",
                 active_listeners.c_str(),
                 kBackendId, speculative_engine_name(state.engine), max_context,
                 state.streaming_kv ? 1u : 0u,
                 state.no_think ? 1u : 0u,
                 state.speculative_qualified ? 1u : 0u,
                 state.speculative_fingerprint.c_str());
    std::vector<struct pollfd> ready;
    ready.reserve(listeners.size());
    for (const int listener : listeners) ready.push_back({listener, POLLIN, 0});
    while (!g_stop) {
        for (struct pollfd &entry : ready) entry.revents = 0;
        const int poll_rc = poll(
                ready.data(), static_cast<nfds_t>(ready.size()), 1000);
        if (poll_rc < 0) {
            if (errno == EINTR) continue;
            state.healthy.store(false);
            break;
        }
        if (poll_rc == 0) continue;
        bool listener_failure = false;
        for (const struct pollfd &entry : ready) {
            if (entry.revents == 0) continue;
            if ((entry.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
                listener_failure = true;
                break;
            }
            if ((entry.revents & POLLIN) == 0) continue;
            const int client = accept(entry.fd, nullptr, nullptr);
            if (client < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
                listener_failure = true;
                break;
            }
            state.active_http_workers.fetch_add(1u);
            (void)fcntl(client, F_SETFD, FD_CLOEXEC);
            struct timeval timeout {};
            timeout.tv_sec = 60;
            setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            try {
                std::thread([client, &state]() {
                    handle_client(client, &state);
                    close(client);
                    state.active_http_workers.fetch_sub(1u);
                }).detach();
            } catch (...) {
                state.active_http_workers.fetch_sub(1u);
                send_retry_response(client, ajson_dumps(error_json(
                        "HTTP worker allocation temporarily unavailable",
                        "transport_overload")));
                close(client);
            }
        }
        if (listener_failure) {
            state.healthy.store(false);
            break;
        }
    }
    close_listeners(&listeners);
    while (state.active_http_workers.load() != 0u) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    destroy_server_state(&state);
    return 0;
}
