#include "axiom/qwen38_swarm_scheduler.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

namespace {

using namespace axiom::qwen38;

swarm_time_point at_ms(std::int64_t milliseconds) {
    return swarm_time_point{} + std::chrono::milliseconds(milliseconds);
}

bool expect(bool condition, const char *message) {
    if (condition) {
        return true;
    }
    std::fprintf(stderr, "qwen38-swarm-scheduler-test: %s\n", message);
    return false;
}

swarm_task_completion success() {
    swarm_task_completion completion;
    completion.outcome = swarm_task_outcome::succeeded;
    return completion;
}

swarm_task_completion failure(const char *message = "synthetic failure") {
    swarm_task_completion completion;
    completion.outcome = swarm_task_outcome::failed;
    completion.message = message;
    return completion;
}

bool ordering_and_admission_test() {
    swarm_scheduler_config config;
    config.max_concurrent = 2u;
    qwen38_swarm_scheduler scheduler(config);
    const swarm_time_point now = at_ms(0);
    const swarm_agent_result agent = scheduler.register_agent({"agent-a", 2u}, now);
    if (!expect(agent.accepted, "agent registration failed")) return false;

    swarm_task_spec low;
    low.name = "low";
    low.priority = 1;
    swarm_task_spec high_first;
    high_first.name = "high-first";
    high_first.priority = 10;
    swarm_task_spec high_second = high_first;
    high_second.name = "high-second";
    const swarm_task_result low_id = scheduler.submit(low, now);
    const swarm_task_result first_id = scheduler.submit(high_first, now);
    const swarm_task_result second_id = scheduler.submit(high_second, now);
    if (!expect(low_id.accepted && first_id.accepted && second_id.accepted,
                "task submission failed")) return false;

    const auto first_lease = scheduler.admit_next(now);
    const auto second_lease = scheduler.admit_next(now);
    if (!expect(first_lease.has_value() && second_lease.has_value(),
                "two eligible tasks were not admitted")) return false;
    if (!expect(first_lease->task_id == first_id.task_id &&
                        second_lease->task_id == second_id.task_id,
                "priority/FIFO ordering was not deterministic")) return false;
    if (!expect(!scheduler.admit_next(now).has_value(),
                "max concurrent admission limit was not enforced")) return false;

    if (!expect(scheduler.complete(first_id.task_id, success(), now).accepted,
                "first completion failed")) return false;
    if (!expect(scheduler.complete(second_id.task_id, success(), now).accepted,
                "second completion failed")) return false;
    const auto low_lease = scheduler.admit_next(now);
    if (!expect(low_lease.has_value() && low_lease->task_id == low_id.task_id,
                "FIFO fallback after priority work was not admitted")) return false;
    return expect(scheduler.complete(low_id.task_id, success(), now).accepted,
                  "low priority completion failed");
}

bool per_agent_admission_test() {
    swarm_scheduler_config config;
    config.max_concurrent = 3u;
    qwen38_swarm_scheduler scheduler(config);
    const swarm_time_point now = at_ms(10);
    if (!expect(scheduler.register_agent({"single-slot", 1u}, now).accepted,
                "single-slot agent registration failed")) return false;
    swarm_task_spec first_spec;
    first_spec.name = "first";
    swarm_task_spec second_spec;
    second_spec.name = "second";
    const swarm_task_result first = scheduler.submit(first_spec, now);
    const swarm_task_result second = scheduler.submit(second_spec, now);
    if (!expect(first.accepted && second.accepted, "per-agent tasks were not submitted")) {
        return false;
    }
    if (!expect(scheduler.admit_next(now).has_value(), "first task was not admitted")) return false;
    if (!expect(!scheduler.admit_next(now).has_value(),
                "agent capacity did not block the second task")) return false;
    const swarm_scheduler_snapshot blocked = scheduler.snapshot();
    if (!expect(blocked.telemetry.active_tasks == 1u && blocked.telemetry.queued_tasks == 1u,
                "admission snapshot counters are incorrect")) return false;
    if (!expect(scheduler.complete(first.task_id, success(), now).accepted,
                "per-agent first completion failed")) return false;
    if (!expect(scheduler.admit_next(now).has_value(),
                "queued task was not admitted after capacity was released")) return false;
    return true;
}

bool cancellation_test() {
    qwen38_swarm_scheduler scheduler;
    const swarm_time_point now = at_ms(20);
    if (!expect(scheduler.register_agent({"cancel-agent", 1u}, now).accepted,
                "cancel agent registration failed")) return false;
    swarm_task_spec running_spec;
    running_spec.name = "running";
    swarm_task_spec queued_spec;
    queued_spec.name = "queued";
    const swarm_task_result running = scheduler.submit(running_spec, now);
    const swarm_task_result queued = scheduler.submit(queued_spec, now);
    if (!expect(running.accepted && queued.accepted, "cancel tasks were not submitted")) return false;
    const auto running_lease = scheduler.admit_next(now);
    if (!expect(running_lease.has_value() && running_lease->task_id == running.task_id,
                "running cancellation task was not admitted")) return false;

    const swarm_cancel_result queued_cancel = scheduler.cancel_task(
            queued.task_id, "client cancelled before start", now);
    if (!expect(queued_cancel.disposition == swarm_cancel_disposition::cancelled_before_run,
                "queued cancellation did not terminalize")) return false;
    const swarm_cancel_result running_cancel = scheduler.cancel_task(
            running.task_id, "client cancelled in flight", now);
    if (!expect(running_cancel.disposition == swarm_cancel_disposition::cancellation_requested &&
                        scheduler.is_cancellation_requested(running.task_id),
                "running cancellation request was not recorded")) return false;

    swarm_task_completion late_success = success();
    const swarm_completion_result cancelled = scheduler.complete(
            running.task_id, late_success, now);
    if (!expect(cancelled.accepted &&
                        cancelled.disposition == swarm_completion_disposition::cancelled,
                "cancellation did not win over a late success")) return false;
    const auto queued_snapshot = scheduler.task_snapshot(queued.task_id);
    const auto running_snapshot = scheduler.task_snapshot(running.task_id);
    if (!expect(queued_snapshot.has_value() && running_snapshot.has_value() &&
                        queued_snapshot->state == swarm_task_state::cancelled &&
                        running_snapshot->state == swarm_task_state::cancelled,
                "cancelled task states are not visible in snapshot")) return false;
    return expect(scheduler.cancel_task(running.task_id, "duplicate", now).disposition ==
                          swarm_cancel_disposition::already_terminal,
                  "terminal cancellation was not idempotent");
}

bool retry_test() {
    qwen38_swarm_scheduler scheduler;
    const swarm_time_point start = at_ms(30);
    if (!expect(scheduler.register_agent({"retry-agent", 1u}, start).accepted,
                "retry agent registration failed")) return false;
    swarm_task_spec spec;
    spec.name = "retryable";
    spec.retry_policy.max_attempts = 2u;
    spec.retry_policy.backoff = std::chrono::milliseconds(5);
    const swarm_task_result task = scheduler.submit(spec, start);
    const auto first_lease = scheduler.admit_next(start);
    if (!expect(task.accepted && first_lease.has_value() && first_lease->attempt == 1u,
                "first retry attempt was not admitted")) return false;
    const swarm_completion_result first = scheduler.complete(task.task_id, failure(), start);
    if (!expect(first.accepted && first.retry_scheduled &&
                        first.disposition == swarm_completion_disposition::retry_scheduled,
                "failed attempt did not schedule a retry")) return false;
    if (!expect(!scheduler.admit_next(at_ms(34)).has_value(),
                "retry admitted before its backoff elapsed")) return false;
    const auto retry_lease = scheduler.admit_next(at_ms(35));
    if (!expect(retry_lease.has_value() && retry_lease->attempt == 2u,
                "second retry attempt was not admitted at the deterministic deadline")) return false;
    if (!expect(scheduler.complete(task.task_id, success(), at_ms(35)).accepted,
                "successful retry completion failed")) return false;
    const auto snapshot = scheduler.task_snapshot(task.task_id);
    if (!expect(snapshot.has_value() && snapshot->state == swarm_task_state::succeeded &&
                        snapshot->attempts_started == 2u,
                "retry task did not reach final success")) return false;
    return expect(scheduler.snapshot().telemetry.retries_scheduled == 1u,
                  "retry telemetry count is incorrect");
}

bool timeout_and_budget_test() {
    qwen38_swarm_scheduler timeout_scheduler;
    const swarm_time_point start = at_ms(40);
    if (!expect(timeout_scheduler.register_agent({"timeout-agent", 1u}, start).accepted,
                "timeout agent registration failed")) return false;
    swarm_task_spec timeout_spec;
    timeout_spec.name = "deadline";
    timeout_spec.budget.max_duration = std::chrono::milliseconds(10);
    const swarm_task_result timeout_task = timeout_scheduler.submit(timeout_spec, start);
    if (!expect(timeout_task.accepted && timeout_scheduler.admit_next(start).has_value(),
                "deadline task was not admitted")) return false;
    const std::vector<swarm_task_id> expired = timeout_scheduler.tick(at_ms(50));
    if (!expect(expired.size() == 1u && expired.front() == timeout_task.task_id,
                "deadline expiry was not reported deterministically")) return false;
    const auto timeout_snapshot = timeout_scheduler.task_snapshot(timeout_task.task_id);
    if (!expect(timeout_snapshot.has_value() &&
                        timeout_snapshot->state == swarm_task_state::timed_out,
                "deadline task did not enter timed_out state")) return false;

    qwen38_swarm_scheduler budget_scheduler;
    if (!expect(budget_scheduler.register_agent({"budget-agent", 1u}, start).accepted,
                "budget agent registration failed")) return false;
    swarm_task_spec budget_spec;
    budget_spec.name = "token-cap";
    budget_spec.budget.max_tokens = 3u;
    const swarm_task_result budget_task = budget_scheduler.submit(budget_spec, start);
    if (!expect(budget_task.accepted && budget_scheduler.admit_next(start).has_value(),
                "budget task was not admitted")) return false;
    if (!expect(budget_scheduler.report_progress(budget_task.task_id, 2u, start).disposition ==
                        swarm_progress_disposition::accepted,
                "under-budget progress was rejected")) return false;
    const swarm_progress_result exhausted = budget_scheduler.report_progress(
            budget_task.task_id, 3u, start);
    if (!expect(exhausted.disposition == swarm_progress_disposition::budget_exhausted,
                "token budget did not stop the attempt")) return false;
    const auto budget_snapshot = budget_scheduler.task_snapshot(budget_task.task_id);
    return expect(budget_snapshot.has_value() &&
                          budget_snapshot->state == swarm_task_state::budget_exhausted &&
                          budget_scheduler.snapshot().telemetry.final_budget_exhausted == 1u,
                  "budget state or telemetry is incorrect");
}

bool parent_snapshot_and_thread_safety_test() {
    swarm_scheduler_config config;
    config.max_concurrent = 2u;
    config.max_event_history = 256u;
    qwen38_swarm_scheduler scheduler(config);
    const swarm_time_point start = at_ms(60);
    if (!expect(scheduler.register_agent({"snapshot-agent", 2u}, start).accepted,
                "snapshot agent registration failed")) return false;
    swarm_wave_spec root_spec;
    root_spec.name = "root";
    const swarm_wave_result root = scheduler.create_wave(root_spec, start);
    if (!expect(root.accepted, "root wave creation failed")) return false;
    swarm_wave_spec child_wave_spec;
    child_wave_spec.name = "child-wave";
    child_wave_spec.parent.parent_wave_id = root.wave_id;
    const swarm_wave_result child = scheduler.create_wave(child_wave_spec, start);
    if (!expect(child.accepted, "child wave creation failed")) return false;

    swarm_task_spec parent_spec;
    parent_spec.name = "parent";
    parent_spec.wave_id = root.wave_id;
    const swarm_task_result parent = scheduler.submit(parent_spec, start);
    if (!expect(parent.accepted, "parent task submission failed")) return false;
    swarm_task_spec child_spec;
    child_spec.name = "child";
    child_spec.wave_id = child.wave_id;
    child_spec.parent.parent_task_id = parent.task_id;
    child_spec.parent.parent_wave_id = root.wave_id;
    child_spec.parent.child_index = 1u;
    child_spec.parent.relation = "fanout";
    const swarm_task_result child_task = scheduler.submit(child_spec, start);
    if (!expect(child_task.accepted, "child task submission failed")) return false;

    const auto initial = scheduler.snapshot();
    if (!expect(initial.tasks.size() == 2u && initial.waves.size() == 2u &&
                        initial.tasks[1].parent.parent_task_id == parent.task_id &&
                        initial.tasks[1].parent.parent_wave_id == root.wave_id,
                "parent/child metadata was not snapshotted")) return false;
    for (std::size_t index = 1u; index < initial.events.size(); ++index) {
        if (!expect(initial.events[index - 1u].sequence < initial.events[index].sequence,
                    "event snapshot ordering is not monotonic")) return false;
    }

    std::atomic<bool> reader_ok{true};
    std::thread reader([&scheduler, &reader_ok]() {
        for (int index = 0; index < 2000; ++index) {
            const swarm_scheduler_snapshot snapshot = scheduler.snapshot();
            for (std::size_t event_index = 1u; event_index < snapshot.events.size(); ++event_index) {
                if (snapshot.events[event_index - 1u].sequence >=
                    snapshot.events[event_index].sequence) {
                    reader_ok.store(false);
                    return;
                }
            }
        }
    });

    for (int index = 0; index < 32; ++index) {
        swarm_task_spec spec;
        spec.name = "snapshot-load";
        spec.priority = index % 3;
        const swarm_task_result task = scheduler.submit(spec, at_ms(61 + index));
        if (task.accepted) {
            const auto lease = scheduler.admit_next(at_ms(61 + index));
            if (lease.has_value()) {
                scheduler.complete(task.task_id, success(), at_ms(61 + index));
            }
        }
    }
    reader.join();
    if (!expect(reader_ok.load(), "concurrent snapshot read was inconsistent")) return false;
    const swarm_scheduler_snapshot final_snapshot = scheduler.snapshot();
    if (!expect(final_snapshot.telemetry.events_emitted >= initial.events.size(),
                "telemetry event count was not captured")) return false;
    return expect(final_snapshot.tasks.size() >= 2u,
                  "final snapshot lost task records");
}

}  // namespace

int main() {
    bool ok = true;
    ok = ordering_and_admission_test() && ok;
    ok = per_agent_admission_test() && ok;
    ok = cancellation_test() && ok;
    ok = retry_test() && ok;
    ok = timeout_and_budget_test() && ok;
    ok = parent_snapshot_and_thread_safety_test() && ok;
    if (ok) {
        std::puts("qwen38-swarm-scheduler-test: PASS");
    }
    return ok ? 0 : 1;
}
