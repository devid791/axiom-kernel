#ifndef AXIOM_QWEN38_SWARM_SCHEDULER_HPP
#define AXIOM_QWEN38_SWARM_SCHEDULER_HPP

/*
 * Deterministic, dependency-free scheduler primitives for the Qwen3.8 swarm.
 *
 * The scheduler is deliberately cooperative: it admits work and returns a
 * lease to an external worker, while the worker reports progress and
 * completion.  No worker thread, clock thread, network client or model
 * runtime is created here.  Callers pass the current time explicitly so the
 * same sequence of inputs produces the same ordering and state transitions.
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace axiom {
namespace qwen38 {

using swarm_task_id = std::uint64_t;
using swarm_wave_id = std::uint64_t;
using swarm_agent_id = std::uint64_t;
using swarm_clock = std::chrono::steady_clock;
using swarm_time_point = swarm_clock::time_point;

enum class swarm_agent_state {
    idle,
    busy,
    draining,
    offline,
    failed,
};

enum class swarm_wave_state {
    pending,
    running,
    cancelling,
    succeeded,
    failed,
    cancelled,
    timed_out,
    budget_exhausted,
};

enum class swarm_task_state {
    queued,
    running,
    retry_wait,
    succeeded,
    failed,
    cancelled,
    timed_out,
    budget_exhausted,
};

enum class swarm_task_outcome {
    succeeded,
    failed,
    cancelled,
    timed_out,
    budget_exhausted,
};

enum class swarm_retry_backoff_strategy {
    fixed,
    exponential,
};

enum class swarm_event_type {
    wave_created,
    wave_state_changed,
    agent_registered,
    agent_state_changed,
    task_submitted,
    task_rejected,
    task_admitted,
    task_started,
    task_progress,
    task_retry_scheduled,
    task_succeeded,
    task_failed,
    task_cancel_requested,
    task_cancelled,
    task_timed_out,
    task_budget_exhausted,
};

enum class swarm_cancel_disposition {
    not_found,
    already_terminal,
    cancelled_before_run,
    cancellation_requested,
};

enum class swarm_completion_disposition {
    rejected_unknown_task,
    rejected_not_running,
    rejected_non_monotonic_tokens,
    accepted,
    retry_scheduled,
    cancelled,
    timed_out,
    budget_exhausted,
};

enum class swarm_progress_disposition {
    rejected_unknown_task,
    rejected_not_running,
    rejected_non_monotonic_tokens,
    accepted,
    cancellation_requested,
    timed_out,
    budget_exhausted,
};

struct swarm_budget {
    /* Budgets apply independently to each attempt. */
    std::optional<std::uint64_t> max_tokens;
    std::optional<std::chrono::milliseconds> max_duration;
};

struct swarm_retry_policy {
    /* The first execution is attempt 1; max_attempts includes it. */
    std::uint32_t max_attempts = 1u;
    bool retry_on_failure = true;
    bool retry_on_timeout = false;
    bool retry_on_budget_exhausted = false;
    swarm_retry_backoff_strategy backoff_strategy = swarm_retry_backoff_strategy::fixed;
    std::chrono::milliseconds backoff{0};
    /* Zero means that exponential backoff is uncapped. */
    std::chrono::milliseconds max_backoff{0};
};

struct swarm_parent_metadata {
    std::optional<swarm_task_id> parent_task_id;
    std::optional<swarm_wave_id> parent_wave_id;
    std::uint32_t child_index = 0u;
    std::string relation;
};

struct swarm_scheduler_config {
    std::size_t max_concurrent = 1u;
    std::optional<std::size_t> max_queued_tasks;
    std::size_t max_event_history = 1024u;
};

struct swarm_agent_spec {
    std::string name;
    std::size_t max_concurrent = 1u;
};

struct swarm_wave_spec {
    std::string name;
    std::optional<std::size_t> max_concurrent;
    swarm_parent_metadata parent;
};

struct swarm_task_spec {
    std::string name;
    int priority = 0;
    std::optional<swarm_wave_id> wave_id;
    swarm_parent_metadata parent;
    swarm_budget budget;
    swarm_retry_policy retry_policy;
};

struct swarm_agent_result {
    bool accepted = false;
    swarm_agent_id agent_id = 0u;
    std::string error;
};

struct swarm_wave_result {
    bool accepted = false;
    swarm_wave_id wave_id = 0u;
    std::string error;
};

struct swarm_task_result {
    bool accepted = false;
    swarm_task_id task_id = 0u;
    std::string error;
};

struct swarm_task_lease {
    swarm_task_id task_id = 0u;
    swarm_wave_id wave_id = 0u;
    swarm_agent_id agent_id = 0u;
    std::uint32_t attempt = 0u;
    int priority = 0;
    swarm_time_point admitted_at{};
    std::optional<swarm_time_point> deadline;
    swarm_budget budget;
    swarm_parent_metadata parent;
};

struct swarm_task_completion {
    swarm_task_outcome outcome = swarm_task_outcome::failed;
    /* Tokens consumed by this attempt, not the aggregate across retries. */
    std::uint64_t tokens_used = 0u;
    std::string message;
};

struct swarm_cancel_result {
    swarm_cancel_disposition disposition = swarm_cancel_disposition::not_found;
    bool changed = false;
};

struct swarm_progress_result {
    swarm_progress_disposition disposition = swarm_progress_disposition::rejected_unknown_task;
    swarm_task_state state = swarm_task_state::queued;
    std::uint64_t tokens_used = 0u;
    bool accepted = false;
};

struct swarm_completion_result {
    swarm_completion_disposition disposition = swarm_completion_disposition::rejected_unknown_task;
    swarm_task_state state = swarm_task_state::queued;
    std::uint32_t attempt = 0u;
    bool accepted = false;
    bool retry_scheduled = false;
    std::string message;
};

struct swarm_event {
    std::uint64_t sequence = 0u;
    swarm_event_type type = swarm_event_type::task_submitted;
    swarm_time_point timestamp{};
    swarm_task_id task_id = 0u;
    swarm_wave_id wave_id = 0u;
    swarm_agent_id agent_id = 0u;
    std::uint32_t attempt = 0u;
    std::optional<swarm_task_state> task_state;
    std::optional<swarm_wave_state> wave_state;
    std::optional<swarm_agent_state> agent_state;
    std::uint64_t tokens_used = 0u;
    std::chrono::milliseconds retry_delay{0};
    std::string message;
};

struct swarm_telemetry_snapshot {
    std::uint64_t events_emitted = 0u;
    std::uint64_t tasks_submitted = 0u;
    std::uint64_t tasks_rejected = 0u;
    std::uint64_t attempts_admitted = 0u;
    std::uint64_t attempts_started = 0u;
    std::uint64_t retries_scheduled = 0u;
    std::uint64_t progress_reports = 0u;
    std::uint64_t final_succeeded = 0u;
    std::uint64_t final_failed = 0u;
    std::uint64_t final_cancelled = 0u;
    std::uint64_t final_timed_out = 0u;
    std::uint64_t final_budget_exhausted = 0u;
    std::uint64_t active_tasks = 0u;
    std::uint64_t queued_tasks = 0u;
    std::uint64_t registered_agents = 0u;
    std::uint64_t available_agents = 0u;
};

struct swarm_task_snapshot {
    swarm_task_id task_id = 0u;
    swarm_wave_id wave_id = 0u;
    std::string name;
    int priority = 0;
    swarm_task_state state = swarm_task_state::queued;
    std::uint32_t attempts_started = 0u;
    std::uint32_t current_attempt = 0u;
    std::uint32_t max_attempts = 1u;
    std::optional<swarm_agent_id> agent_id;
    swarm_parent_metadata parent;
    swarm_budget budget;
    swarm_retry_policy retry_policy;
    bool cancellation_requested = false;
    std::uint64_t tokens_used = 0u;
    std::uint64_t tokens_used_total = 0u;
    std::optional<swarm_time_point> admitted_at;
    std::optional<swarm_time_point> deadline;
    std::optional<swarm_time_point> retry_not_before;
    std::string terminal_reason;
};

struct swarm_wave_snapshot {
    swarm_wave_id wave_id = 0u;
    std::string name;
    swarm_wave_state state = swarm_wave_state::pending;
    std::optional<std::size_t> max_concurrent;
    swarm_parent_metadata parent;
    bool cancellation_requested = false;
    std::uint64_t task_count = 0u;
    std::uint64_t active_tasks = 0u;
    std::uint64_t terminal_tasks = 0u;
};

struct swarm_agent_snapshot {
    swarm_agent_id agent_id = 0u;
    std::string name;
    swarm_agent_state state = swarm_agent_state::idle;
    std::size_t max_concurrent = 1u;
    std::uint64_t active_tasks = 0u;
};

struct swarm_scheduler_snapshot {
    std::uint64_t revision = 0u;
    swarm_telemetry_snapshot telemetry;
    std::vector<swarm_task_snapshot> tasks;
    std::vector<swarm_wave_snapshot> waves;
    std::vector<swarm_agent_snapshot> agents;
    std::vector<swarm_event> events;
};

class qwen38_swarm_scheduler {
public:
    explicit qwen38_swarm_scheduler(swarm_scheduler_config config = {});
    ~qwen38_swarm_scheduler();

    qwen38_swarm_scheduler(const qwen38_swarm_scheduler &) = delete;
    qwen38_swarm_scheduler &operator=(const qwen38_swarm_scheduler &) = delete;

    swarm_agent_result register_agent(
            const swarm_agent_spec &spec,
            swarm_time_point now);

    swarm_wave_result create_wave(
            const swarm_wave_spec &spec,
            swarm_time_point now);

    swarm_task_result submit(
            const swarm_task_spec &spec,
            swarm_time_point now);

    /* Selects the highest-priority eligible task; ties are FIFO. */
    std::optional<swarm_task_lease> admit_next(swarm_time_point now);

    swarm_progress_result report_progress(
            swarm_task_id task_id,
            std::uint64_t tokens_used,
            swarm_time_point now);

    swarm_completion_result complete(
            swarm_task_id task_id,
            const swarm_task_completion &completion,
            swarm_time_point now);

    swarm_cancel_result cancel_task(
            swarm_task_id task_id,
            const std::string &reason,
            swarm_time_point now);

    /* A running task receives a cooperative cancellation request and keeps
     * its lease until completion. Queued/retry-wait tasks are terminalized
     * immediately. A worker can acknowledge a request with complete(...,
     * swarm_task_outcome::cancelled, ...). */
    std::size_t cancel_wave(
            swarm_wave_id wave_id,
            const std::string &reason,
            swarm_time_point now,
            bool include_descendants = true);

    /* Expires running deadlines. Returned IDs are in ascending task ID order. */
    std::vector<swarm_task_id> tick(swarm_time_point now);

    bool set_agent_state(
            swarm_agent_id agent_id,
            swarm_agent_state state,
            swarm_time_point now,
            const std::string &reason = {});

    bool is_cancellation_requested(swarm_task_id task_id) const;

    std::optional<swarm_task_snapshot> task_snapshot(swarm_task_id task_id) const;
    swarm_scheduler_snapshot snapshot() const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace qwen38
}  // namespace axiom

#endif  // AXIOM_QWEN38_SWARM_SCHEDULER_HPP
