#include "axiom/qwen38_swarm_scheduler.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace axiom {
namespace qwen38 {

namespace {

bool valid_optional_id(const std::optional<std::uint64_t> &id) {
    return !id.has_value() || *id != 0u;
}

const char *outcome_name(swarm_task_outcome outcome) {
    switch (outcome) {
        case swarm_task_outcome::succeeded: return "succeeded";
        case swarm_task_outcome::failed: return "failed";
        case swarm_task_outcome::cancelled: return "cancelled";
        case swarm_task_outcome::timed_out: return "timed out";
        case swarm_task_outcome::budget_exhausted: return "budget exhausted";
    }
    return "unknown outcome";
}

swarm_task_state task_state_for_outcome(swarm_task_outcome outcome) {
    switch (outcome) {
        case swarm_task_outcome::succeeded: return swarm_task_state::succeeded;
        case swarm_task_outcome::failed: return swarm_task_state::failed;
        case swarm_task_outcome::cancelled: return swarm_task_state::cancelled;
        case swarm_task_outcome::timed_out: return swarm_task_state::timed_out;
        case swarm_task_outcome::budget_exhausted:
            return swarm_task_state::budget_exhausted;
    }
    return swarm_task_state::failed;
}

swarm_event_type event_type_for_outcome(swarm_task_outcome outcome) {
    switch (outcome) {
        case swarm_task_outcome::succeeded: return swarm_event_type::task_succeeded;
        case swarm_task_outcome::failed: return swarm_event_type::task_failed;
        case swarm_task_outcome::cancelled: return swarm_event_type::task_cancelled;
        case swarm_task_outcome::timed_out: return swarm_event_type::task_timed_out;
        case swarm_task_outcome::budget_exhausted:
            return swarm_event_type::task_budget_exhausted;
    }
    return swarm_event_type::task_failed;
}

bool is_terminal(swarm_task_state state) {
    return state == swarm_task_state::succeeded ||
           state == swarm_task_state::failed ||
           state == swarm_task_state::cancelled ||
           state == swarm_task_state::timed_out ||
           state == swarm_task_state::budget_exhausted;
}

bool is_terminal_wave(swarm_wave_state state) {
    return state == swarm_wave_state::succeeded ||
           state == swarm_wave_state::failed ||
           state == swarm_wave_state::cancelled ||
           state == swarm_wave_state::timed_out ||
           state == swarm_wave_state::budget_exhausted;
}

}  // namespace

struct qwen38_swarm_scheduler::impl {
    struct agent_record {
        swarm_agent_id id = 0u;
        swarm_agent_spec spec;
        swarm_agent_state administrative_state = swarm_agent_state::idle;
        std::uint64_t active_tasks = 0u;
    };

    struct wave_record {
        swarm_wave_id id = 0u;
        swarm_wave_spec spec;
        swarm_wave_state state = swarm_wave_state::pending;
        bool cancellation_requested = false;
        bool has_started = false;
        std::uint64_t active_tasks = 0u;
    };

    struct task_record {
        swarm_task_id id = 0u;
        swarm_task_spec spec;
        swarm_task_state state = swarm_task_state::queued;
        std::uint64_t queue_sequence = 0u;
        std::uint32_t attempts_started = 0u;
        std::uint64_t tokens_used = 0u;
        std::uint64_t tokens_used_total = 0u;
        std::optional<swarm_agent_id> agent_id;
        bool cancellation_requested = false;
        std::optional<swarm_time_point> admitted_at;
        std::optional<swarm_time_point> deadline;
        std::optional<swarm_time_point> retry_not_before;
        std::string terminal_reason;
    };

    explicit impl(swarm_scheduler_config config_in)
        : config(std::move(config_in)) {
        if (config.max_concurrent == 0u) {
            throw std::invalid_argument("swarm scheduler max_concurrent must be positive");
        }
    }

    swarm_scheduler_config config;
    mutable std::mutex mutex;
    std::map<swarm_agent_id, agent_record> agents;
    std::map<swarm_wave_id, wave_record> waves;
    std::map<swarm_task_id, task_record> tasks;
    std::deque<swarm_event> events;
    swarm_telemetry_snapshot telemetry;
    std::uint64_t active_tasks = 0u;
    std::uint64_t next_agent_id = 1u;
    std::uint64_t next_wave_id = 1u;
    std::uint64_t next_task_id = 1u;
    std::uint64_t next_queue_sequence = 1u;
    std::uint64_t next_event_sequence = 1u;
    std::uint64_t revision = 0u;

    swarm_agent_state visible_agent_state(const agent_record &agent) const {
        if (agent.administrative_state != swarm_agent_state::idle) {
            return agent.administrative_state;
        }
        return agent.active_tasks == 0u ? swarm_agent_state::idle : swarm_agent_state::busy;
    }

    std::uint64_t queued_task_count() const {
        std::uint64_t count = 0u;
        for (const auto &entry : tasks) {
            if (entry.second.state == swarm_task_state::queued ||
                entry.second.state == swarm_task_state::retry_wait) {
                ++count;
            }
        }
        return count;
    }

    void emit(
            swarm_event_type type,
            swarm_time_point now,
            swarm_task_id task_id = 0u,
            swarm_wave_id wave_id = 0u,
            swarm_agent_id agent_id = 0u,
            std::uint32_t attempt = 0u,
            std::optional<swarm_task_state> task_state = std::nullopt,
            std::optional<swarm_wave_state> wave_state = std::nullopt,
            std::optional<swarm_agent_state> agent_state = std::nullopt,
            std::uint64_t tokens_used = 0u,
            std::chrono::milliseconds retry_delay = std::chrono::milliseconds(0),
            const std::string &message = {}) {
        swarm_event event;
        event.sequence = next_event_sequence++;
        event.type = type;
        event.timestamp = now;
        event.task_id = task_id;
        event.wave_id = wave_id;
        event.agent_id = agent_id;
        event.attempt = attempt;
        event.task_state = task_state;
        event.wave_state = wave_state;
        event.agent_state = agent_state;
        event.tokens_used = tokens_used;
        event.retry_delay = retry_delay;
        event.message = message;
        ++telemetry.events_emitted;
        ++revision;

        if (config.max_event_history == 0u) {
            return;
        }
        events.push_back(std::move(event));
        while (events.size() > config.max_event_history) {
            events.pop_front();
        }
    }

    void set_wave_state(
            wave_record &wave,
            swarm_wave_state state,
            swarm_time_point now,
            const std::string &reason) {
        if (wave.state == state) {
            return;
        }
        wave.state = state;
        emit(
                swarm_event_type::wave_state_changed,
                now,
                0u,
                wave.id,
                0u,
                0u,
                std::nullopt,
                state,
                std::nullopt,
                0u,
                std::chrono::milliseconds(0),
                reason);
    }

    bool wave_is_descendant(
            swarm_wave_id candidate,
            swarm_wave_id ancestor,
            bool include_candidate = true) const {
        if (include_candidate && candidate == ancestor) {
            return true;
        }
        std::set<swarm_wave_id> visited;
        swarm_wave_id current = candidate;
        while (visited.insert(current).second) {
            const auto wave_it = waves.find(current);
            if (wave_it == waves.end()) {
                return false;
            }
            const swarm_parent_metadata &parent = wave_it->second.spec.parent;
            if (parent.parent_wave_id.has_value()) {
                current = *parent.parent_wave_id;
            } else if (parent.parent_task_id.has_value()) {
                const auto task_it = tasks.find(*parent.parent_task_id);
                if (task_it == tasks.end() || !task_it->second.spec.wave_id.has_value()) {
                    return false;
                }
                current = *task_it->second.spec.wave_id;
            } else {
                return false;
            }
            if (current == ancestor) {
                return true;
            }
        }
        return false;
    }

    bool task_belongs_to_wave_tree(
            const task_record &task,
            swarm_wave_id ancestor,
            bool include_descendants) const {
        if (task.spec.wave_id.has_value() &&
            wave_is_descendant(*task.spec.wave_id, ancestor, include_descendants)) {
            return true;
        }

        if (!include_descendants) {
            return task.spec.parent.parent_wave_id.has_value() &&
                   *task.spec.parent.parent_wave_id == ancestor;
        }

        std::set<swarm_task_id> visited;
        const task_record *current = &task;
        while (visited.insert(current->id).second) {
            const swarm_parent_metadata &parent = current->spec.parent;
            if (parent.parent_wave_id.has_value() &&
                wave_is_descendant(*parent.parent_wave_id, ancestor, true)) {
                return true;
            }
            if (!parent.parent_task_id.has_value()) {
                break;
            }
            const auto parent_it = tasks.find(*parent.parent_task_id);
            if (parent_it == tasks.end()) {
                break;
            }
            current = &parent_it->second;
            if (current->spec.wave_id.has_value() &&
                wave_is_descendant(*current->spec.wave_id, ancestor, true)) {
                return true;
            }
        }
        return false;
    }

    void recompute_wave(swarm_wave_id wave_id, swarm_time_point now) {
        const auto wave_it = waves.find(wave_id);
        if (wave_it == waves.end()) {
            return;
        }
        wave_record &wave = wave_it->second;
        bool has_tasks = false;
        bool has_nonterminal = false;
        bool has_failed = false;
        bool has_timed_out = false;
        bool has_budget_exhausted = false;
        bool has_cancelled = false;
        for (const auto &entry : tasks) {
            const task_record &task = entry.second;
            if (!task.spec.wave_id.has_value() || *task.spec.wave_id != wave_id) {
                continue;
            }
            has_tasks = true;
            if (!is_terminal(task.state)) {
                has_nonterminal = true;
            }
            has_failed = has_failed || task.state == swarm_task_state::failed;
            has_timed_out = has_timed_out || task.state == swarm_task_state::timed_out;
            has_budget_exhausted =
                    has_budget_exhausted || task.state == swarm_task_state::budget_exhausted;
            has_cancelled = has_cancelled || task.state == swarm_task_state::cancelled;
        }

        if (!has_tasks) {
            if (wave.cancellation_requested) {
                set_wave_state(wave, swarm_wave_state::cancelled, now, "wave cancelled");
            }
            return;
        }
        if (wave.cancellation_requested) {
            set_wave_state(
                    wave,
                    has_nonterminal ? swarm_wave_state::cancelling : swarm_wave_state::cancelled,
                    now,
                    has_nonterminal ? "wave cancellation pending" : "wave cancelled");
            return;
        }
        if (has_nonterminal) {
            if (wave.has_started) {
                set_wave_state(wave, swarm_wave_state::running, now, "wave running");
            }
            return;
        }

        swarm_wave_state final_state = swarm_wave_state::succeeded;
        if (has_failed) {
            final_state = swarm_wave_state::failed;
        } else if (has_timed_out) {
            final_state = swarm_wave_state::timed_out;
        } else if (has_budget_exhausted) {
            final_state = swarm_wave_state::budget_exhausted;
        } else if (has_cancelled) {
            final_state = swarm_wave_state::cancelled;
        }
        set_wave_state(wave, final_state, now, "wave terminal");
    }

    void release_resources(task_record &task) {
        if (task.agent_id.has_value()) {
            const auto agent_it = agents.find(*task.agent_id);
            if (agent_it != agents.end() && agent_it->second.active_tasks > 0u) {
                --agent_it->second.active_tasks;
            }
            task.agent_id.reset();
        }
        if (active_tasks > 0u) {
            --active_tasks;
        }
        if (task.spec.wave_id.has_value()) {
            const auto wave_it = waves.find(*task.spec.wave_id);
            if (wave_it != waves.end() && wave_it->second.active_tasks > 0u) {
                --wave_it->second.active_tasks;
            }
        }
        task.admitted_at.reset();
        task.deadline.reset();
    }

    std::chrono::milliseconds retry_delay(
            const swarm_retry_policy &policy,
            std::uint32_t completed_attempt) const {
        const std::int64_t base = policy.backoff.count();
        if (base <= 0) {
            return std::chrono::milliseconds(0);
        }
        std::int64_t value = base;
        if (policy.backoff_strategy == swarm_retry_backoff_strategy::exponential &&
            completed_attempt > 1u) {
            const std::uint32_t shift = completed_attempt - 1u;
            const std::int64_t max_value = std::numeric_limits<std::int64_t>::max();
            if (shift >= 63u || base > (max_value >> shift)) {
                value = max_value;
            } else {
                value = base << shift;
            }
        }
        if (policy.max_backoff.count() > 0 && value > policy.max_backoff.count()) {
            value = policy.max_backoff.count();
        }
        return std::chrono::milliseconds(value);
    }

    bool should_retry(
            const task_record &task,
            swarm_task_outcome outcome) const {
        if (task.attempts_started >= task.spec.retry_policy.max_attempts) {
            return false;
        }
        switch (outcome) {
            case swarm_task_outcome::succeeded:
            case swarm_task_outcome::cancelled:
                return false;
            case swarm_task_outcome::failed:
                return task.spec.retry_policy.retry_on_failure;
            case swarm_task_outcome::timed_out:
                return task.spec.retry_policy.retry_on_timeout;
            case swarm_task_outcome::budget_exhausted:
                return task.spec.retry_policy.retry_on_budget_exhausted;
        }
        return false;
    }

    bool finish_attempt(
            task_record &task,
            swarm_task_outcome outcome,
            swarm_time_point now,
            const std::string &reason) {
        const std::uint32_t attempt = task.attempts_started;
        const std::uint64_t attempt_tokens = task.tokens_used;
        task.tokens_used_total += attempt_tokens;
        release_resources(task);

        const bool retry = should_retry(task, outcome);
        if (retry) {
            const std::chrono::milliseconds delay = retry_delay(
                    task.spec.retry_policy, attempt);
            task.state = swarm_task_state::retry_wait;
            task.retry_not_before = now + delay;
            task.queue_sequence = next_queue_sequence++;
            task.tokens_used = 0u;
            task.cancellation_requested = false;
            task.terminal_reason = reason;
            ++telemetry.retries_scheduled;
            emit(
                    event_type_for_outcome(outcome),
                    now,
                    task.id,
                    task.spec.wave_id.value_or(0u),
                    0u,
                    attempt,
                    task.state,
                    std::nullopt,
                    std::nullopt,
                    attempt_tokens,
                    std::chrono::milliseconds(0),
                    reason);
            emit(
                    swarm_event_type::task_retry_scheduled,
                    now,
                    task.id,
                    task.spec.wave_id.value_or(0u),
                    0u,
                    attempt,
                    task.state,
                    std::nullopt,
                    std::nullopt,
                    0u,
                    delay,
                    reason);
        } else {
            task.state = task_state_for_outcome(outcome);
            task.retry_not_before.reset();
            task.terminal_reason = reason;
            switch (task.state) {
                case swarm_task_state::succeeded: ++telemetry.final_succeeded; break;
                case swarm_task_state::failed: ++telemetry.final_failed; break;
                case swarm_task_state::cancelled: ++telemetry.final_cancelled; break;
                case swarm_task_state::timed_out: ++telemetry.final_timed_out; break;
                case swarm_task_state::budget_exhausted:
                    ++telemetry.final_budget_exhausted;
                    break;
                case swarm_task_state::queued:
                case swarm_task_state::running:
                case swarm_task_state::retry_wait:
                    break;
            }
            emit(
                    event_type_for_outcome(outcome),
                    now,
                    task.id,
                    task.spec.wave_id.value_or(0u),
                    0u,
                    attempt,
                    task.state,
                    std::nullopt,
                    std::nullopt,
                    attempt_tokens,
                    std::chrono::milliseconds(0),
                    reason);
        }
        if (task.spec.wave_id.has_value()) {
            recompute_wave(*task.spec.wave_id, now);
        }
        return retry;
    }

    swarm_cancel_disposition cancel_task_locked(
            task_record &task,
            const std::string &reason,
            swarm_time_point now) {
        if (is_terminal(task.state)) {
            return swarm_cancel_disposition::already_terminal;
        }
        if (task.state == swarm_task_state::queued ||
            task.state == swarm_task_state::retry_wait) {
            task.cancellation_requested = true;
            task.state = swarm_task_state::cancelled;
            task.retry_not_before.reset();
            task.terminal_reason = reason;
            ++telemetry.final_cancelled;
            emit(
                    swarm_event_type::task_cancelled,
                    now,
                    task.id,
                    task.spec.wave_id.value_or(0u),
                    0u,
                    task.attempts_started,
                    task.state,
                    std::nullopt,
                    std::nullopt,
                    task.tokens_used,
                    std::chrono::milliseconds(0),
                    reason);
            if (task.spec.wave_id.has_value()) {
                recompute_wave(*task.spec.wave_id, now);
            }
            return swarm_cancel_disposition::cancelled_before_run;
        }
        if (!task.cancellation_requested) {
            task.cancellation_requested = true;
            emit(
                    swarm_event_type::task_cancel_requested,
                    now,
                    task.id,
                    task.spec.wave_id.value_or(0u),
                    task.agent_id.value_or(0u),
                    task.attempts_started,
                    task.state,
                    std::nullopt,
                    std::nullopt,
                    task.tokens_used,
                    std::chrono::milliseconds(0),
                    reason);
            return swarm_cancel_disposition::cancellation_requested;
        }
        return swarm_cancel_disposition::cancellation_requested;
    }

    void expire_running(
            swarm_time_point now,
            std::vector<swarm_task_id> *expired_ids) {
        std::vector<swarm_task_id> candidates;
        for (const auto &entry : tasks) {
            const task_record &task = entry.second;
            if (task.state == swarm_task_state::running && task.deadline.has_value() &&
                now >= *task.deadline) {
                candidates.push_back(task.id);
            }
        }
        for (const swarm_task_id task_id : candidates) {
            const auto task_it = tasks.find(task_id);
            if (task_it == tasks.end() || task_it->second.state != swarm_task_state::running) {
                continue;
            }
            task_record &task = task_it->second;
            const swarm_task_outcome outcome = task.cancellation_requested
                    ? swarm_task_outcome::cancelled
                    : swarm_task_outcome::timed_out;
            finish_attempt(task, outcome, now, outcome == swarm_task_outcome::cancelled
                    ? "cancelled at deadline"
                    : "task deadline exceeded");
            if (expired_ids != nullptr) {
                expired_ids->push_back(task_id);
            }
        }
    }

    swarm_task_snapshot make_task_snapshot(const task_record &task) const {
        swarm_task_snapshot result;
        result.task_id = task.id;
        result.wave_id = task.spec.wave_id.value_or(0u);
        result.name = task.spec.name;
        result.priority = task.spec.priority;
        result.state = task.state;
        result.attempts_started = task.attempts_started;
        result.current_attempt = task.attempts_started;
        result.max_attempts = task.spec.retry_policy.max_attempts;
        result.agent_id = task.agent_id;
        result.parent = task.spec.parent;
        result.budget = task.spec.budget;
        result.retry_policy = task.spec.retry_policy;
        result.cancellation_requested = task.cancellation_requested;
        result.tokens_used = task.tokens_used;
        result.tokens_used_total = task.tokens_used_total;
        result.admitted_at = task.admitted_at;
        result.deadline = task.deadline;
        result.retry_not_before = task.retry_not_before;
        result.terminal_reason = task.terminal_reason;
        return result;
    }

    swarm_wave_snapshot make_wave_snapshot(const wave_record &wave) const {
        swarm_wave_snapshot result;
        result.wave_id = wave.id;
        result.name = wave.spec.name;
        result.state = wave.state;
        result.max_concurrent = wave.spec.max_concurrent;
        result.parent = wave.spec.parent;
        result.cancellation_requested = wave.cancellation_requested;
        result.active_tasks = wave.active_tasks;
        for (const auto &entry : tasks) {
            const task_record &task = entry.second;
            if (!task.spec.wave_id.has_value() || *task.spec.wave_id != wave.id) {
                continue;
            }
            ++result.task_count;
            if (is_terminal(task.state)) {
                ++result.terminal_tasks;
            }
        }
        return result;
    }

    swarm_agent_snapshot make_agent_snapshot(const agent_record &agent) const {
        swarm_agent_snapshot result;
        result.agent_id = agent.id;
        result.name = agent.spec.name;
        result.state = visible_agent_state(agent);
        result.max_concurrent = agent.spec.max_concurrent;
        result.active_tasks = agent.active_tasks;
        return result;
    }
};

qwen38_swarm_scheduler::qwen38_swarm_scheduler(swarm_scheduler_config config)
    : impl_(std::make_unique<impl>(std::move(config))) {}

qwen38_swarm_scheduler::~qwen38_swarm_scheduler() = default;

swarm_agent_result qwen38_swarm_scheduler::register_agent(
        const swarm_agent_spec &spec,
        swarm_time_point now) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    swarm_agent_result result;
    if (spec.max_concurrent == 0u) {
        result.error = "agent max_concurrent must be positive";
        return result;
    }
    impl::agent_record agent;
    agent.id = impl_->next_agent_id++;
    agent.spec = spec;
    impl_->agents.emplace(agent.id, agent);
    result.accepted = true;
    result.agent_id = agent.id;
    impl_->emit(
            swarm_event_type::agent_registered,
            now,
            0u,
            0u,
            agent.id,
            0u,
            std::nullopt,
            std::nullopt,
            swarm_agent_state::idle,
            0u,
            std::chrono::milliseconds(0),
            spec.name);
    return result;
}

swarm_wave_result qwen38_swarm_scheduler::create_wave(
        const swarm_wave_spec &spec,
        swarm_time_point now) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    swarm_wave_result result;
    if (spec.max_concurrent.has_value() && *spec.max_concurrent == 0u) {
        result.error = "wave max_concurrent must be positive";
        return result;
    }
    if (!valid_optional_id(spec.parent.parent_wave_id) ||
        !valid_optional_id(spec.parent.parent_task_id)) {
        result.error = "parent IDs must be nonzero";
        return result;
    }
    if (spec.parent.parent_wave_id.has_value() &&
        impl_->waves.find(*spec.parent.parent_wave_id) == impl_->waves.end()) {
        result.error = "parent wave does not exist";
        return result;
    }
    if (spec.parent.parent_task_id.has_value() &&
        impl_->tasks.find(*spec.parent.parent_task_id) == impl_->tasks.end()) {
        result.error = "parent task does not exist";
        return result;
    }
    impl::wave_record wave;
    wave.id = impl_->next_wave_id++;
    wave.spec = spec;
    impl_->waves.emplace(wave.id, wave);
    result.accepted = true;
    result.wave_id = wave.id;
    impl_->emit(
            swarm_event_type::wave_created,
            now,
            0u,
            wave.id,
            0u,
            0u,
            std::nullopt,
            swarm_wave_state::pending,
            std::nullopt,
            0u,
            std::chrono::milliseconds(0),
            spec.name);
    return result;
}

swarm_task_result qwen38_swarm_scheduler::submit(
        const swarm_task_spec &spec,
        swarm_time_point now) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    swarm_task_result result;
    auto reject = [&](const std::string &error) {
        result.error = error;
        ++impl_->telemetry.tasks_rejected;
        impl_->emit(
                swarm_event_type::task_rejected,
                now,
                0u,
                spec.wave_id.value_or(0u),
                0u,
                0u,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                0u,
                std::chrono::milliseconds(0),
                error);
        return result;
    };

    if (!valid_optional_id(spec.wave_id) ||
        !valid_optional_id(spec.parent.parent_wave_id) ||
        !valid_optional_id(spec.parent.parent_task_id)) {
        return reject("task IDs must be nonzero");
    }
    if (spec.retry_policy.max_attempts == 0u) {
        return reject("retry policy max_attempts must be positive");
    }
    if (spec.retry_policy.backoff.count() < 0 ||
        (spec.retry_policy.max_backoff.count() < 0 &&
         spec.retry_policy.max_backoff.count() != 0)) {
        return reject("retry backoff must not be negative");
    }
    if (spec.budget.max_duration.has_value() && spec.budget.max_duration->count() < 0) {
        return reject("task max_duration must not be negative");
    }
    if (spec.wave_id.has_value()) {
        const auto wave_it = impl_->waves.find(*spec.wave_id);
        if (wave_it == impl_->waves.end()) {
            return reject("task wave does not exist");
        }
        if (is_terminal_wave(wave_it->second.state) || wave_it->second.cancellation_requested) {
            return reject("task wave is terminal or cancelling");
        }
    }
    if (spec.parent.parent_wave_id.has_value() &&
        impl_->waves.find(*spec.parent.parent_wave_id) == impl_->waves.end()) {
        return reject("parent wave does not exist");
    }
    if (spec.parent.parent_task_id.has_value() &&
        impl_->tasks.find(*spec.parent.parent_task_id) == impl_->tasks.end()) {
        return reject("parent task does not exist");
    }
    if (impl_->config.max_queued_tasks.has_value() &&
        impl_->queued_task_count() >= *impl_->config.max_queued_tasks) {
        return reject("scheduler queue admission limit reached");
    }

    impl::task_record task;
    task.id = impl_->next_task_id++;
    task.spec = spec;
    task.queue_sequence = impl_->next_queue_sequence++;
    impl_->tasks.emplace(task.id, task);
    ++impl_->telemetry.tasks_submitted;
    result.accepted = true;
    result.task_id = task.id;
    impl_->emit(
            swarm_event_type::task_submitted,
            now,
            task.id,
            spec.wave_id.value_or(0u),
            0u,
            0u,
            swarm_task_state::queued,
            std::nullopt,
            std::nullopt,
            0u,
            std::chrono::milliseconds(0),
            spec.name);
    return result;
}

std::optional<swarm_task_lease> qwen38_swarm_scheduler::admit_next(
        swarm_time_point now) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->expire_running(now, nullptr);
    if (impl_->active_tasks >= impl_->config.max_concurrent) {
        return std::nullopt;
    }

    std::optional<swarm_agent_id> selected_agent;
    for (const auto &entry : impl_->agents) {
        const impl::agent_record &agent = entry.second;
        if (agent.administrative_state != swarm_agent_state::idle ||
            agent.active_tasks >= agent.spec.max_concurrent) {
            continue;
        }
        selected_agent = agent.id;
        break;
    }
    if (!selected_agent.has_value()) {
        return std::nullopt;
    }

    impl::task_record *selected_task = nullptr;
    for (auto &entry : impl_->tasks) {
        impl::task_record &task = entry.second;
        if (task.cancellation_requested ||
            (task.state != swarm_task_state::queued &&
             task.state != swarm_task_state::retry_wait)) {
            continue;
        }
        if (task.state == swarm_task_state::retry_wait &&
            task.retry_not_before.has_value() && now < *task.retry_not_before) {
            continue;
        }
        if (task.spec.wave_id.has_value()) {
            const auto wave_it = impl_->waves.find(*task.spec.wave_id);
            if (wave_it == impl_->waves.end() || wave_it->second.cancellation_requested ||
                (wave_it->second.spec.max_concurrent.has_value() &&
                 wave_it->second.active_tasks >= *wave_it->second.spec.max_concurrent)) {
                continue;
            }
        }
        if (selected_task == nullptr || task.spec.priority > selected_task->spec.priority ||
            (task.spec.priority == selected_task->spec.priority &&
             (task.queue_sequence < selected_task->queue_sequence ||
              (task.queue_sequence == selected_task->queue_sequence &&
               task.id < selected_task->id)))) {
            selected_task = &task;
        }
    }
    if (selected_task == nullptr) {
        return std::nullopt;
    }

    impl::agent_record &agent = impl_->agents.find(*selected_agent)->second;
    ++agent.active_tasks;
    ++impl_->active_tasks;
    selected_task->state = swarm_task_state::running;
    ++selected_task->attempts_started;
    selected_task->tokens_used = 0u;
    selected_task->agent_id = agent.id;
    selected_task->admitted_at = now;
    selected_task->retry_not_before.reset();
    if (selected_task->spec.budget.max_duration.has_value()) {
        selected_task->deadline = now + *selected_task->spec.budget.max_duration;
    } else {
        selected_task->deadline.reset();
    }
    if (selected_task->spec.wave_id.has_value()) {
        auto wave_it = impl_->waves.find(*selected_task->spec.wave_id);
        if (wave_it != impl_->waves.end()) {
            ++wave_it->second.active_tasks;
            wave_it->second.has_started = true;
            impl_->set_wave_state(wave_it->second, swarm_wave_state::running, now, "wave started");
        }
    }
    ++impl_->telemetry.attempts_admitted;
    ++impl_->telemetry.attempts_started;
    impl_->emit(
            swarm_event_type::task_admitted,
            now,
            selected_task->id,
            selected_task->spec.wave_id.value_or(0u),
            agent.id,
            selected_task->attempts_started,
            selected_task->state);
    impl_->emit(
            swarm_event_type::task_started,
            now,
            selected_task->id,
            selected_task->spec.wave_id.value_or(0u),
            agent.id,
            selected_task->attempts_started,
            selected_task->state);

    swarm_task_lease lease;
    lease.task_id = selected_task->id;
    lease.wave_id = selected_task->spec.wave_id.value_or(0u);
    lease.agent_id = agent.id;
    lease.attempt = selected_task->attempts_started;
    lease.priority = selected_task->spec.priority;
    lease.admitted_at = now;
    lease.deadline = selected_task->deadline;
    lease.budget = selected_task->spec.budget;
    lease.parent = selected_task->spec.parent;
    return lease;
}

swarm_progress_result qwen38_swarm_scheduler::report_progress(
        swarm_task_id task_id,
        std::uint64_t tokens_used,
        swarm_time_point now) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->expire_running(now, nullptr);
    swarm_progress_result result;
    const auto task_it = impl_->tasks.find(task_id);
    if (task_it == impl_->tasks.end()) {
        result.disposition = swarm_progress_disposition::rejected_unknown_task;
        return result;
    }
    impl::task_record &task = task_it->second;
    result.state = task.state;
    result.tokens_used = task.tokens_used;
    if (task.state != swarm_task_state::running) {
        result.disposition = swarm_progress_disposition::rejected_not_running;
        return result;
    }
    if (tokens_used < task.tokens_used) {
        result.disposition = swarm_progress_disposition::rejected_non_monotonic_tokens;
        return result;
    }
    task.tokens_used = tokens_used;
    result.tokens_used = tokens_used;
    result.state = task.state;
    result.accepted = true;
    ++impl_->telemetry.progress_reports;
    impl_->emit(
            swarm_event_type::task_progress,
            now,
            task.id,
            task.spec.wave_id.value_or(0u),
            task.agent_id.value_or(0u),
            task.attempts_started,
            task.state,
            std::nullopt,
            std::nullopt,
            tokens_used,
            std::chrono::milliseconds(0),
            "progress");
    if (task.cancellation_requested) {
        result.disposition = swarm_progress_disposition::cancellation_requested;
        return result;
    }
    if (task.spec.budget.max_tokens.has_value() &&
        tokens_used >= *task.spec.budget.max_tokens) {
        impl_->finish_attempt(task, swarm_task_outcome::budget_exhausted, now, "token budget exhausted");
        result.disposition = swarm_progress_disposition::budget_exhausted;
        result.state = task.state;
        return result;
    }
    result.disposition = swarm_progress_disposition::accepted;
    return result;
}

swarm_completion_result qwen38_swarm_scheduler::complete(
        swarm_task_id task_id,
        const swarm_task_completion &completion,
        swarm_time_point now) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->expire_running(now, nullptr);
    swarm_completion_result result;
    const auto task_it = impl_->tasks.find(task_id);
    if (task_it == impl_->tasks.end()) {
        result.disposition = swarm_completion_disposition::rejected_unknown_task;
        result.message = "task does not exist";
        return result;
    }
    impl::task_record &task = task_it->second;
    result.attempt = task.attempts_started;
    result.state = task.state;
    if (task.state != swarm_task_state::running) {
        result.disposition = swarm_completion_disposition::rejected_not_running;
        result.message = "task is not running";
        return result;
    }
    if (completion.tokens_used < task.tokens_used) {
        result.disposition = swarm_completion_disposition::rejected_non_monotonic_tokens;
        result.message = "completion token count moved backwards";
        return result;
    }
    task.tokens_used = completion.tokens_used;

    swarm_task_outcome outcome = completion.outcome;
    if (task.cancellation_requested) {
        outcome = swarm_task_outcome::cancelled;
    } else if (task.spec.budget.max_tokens.has_value() &&
               completion.tokens_used > *task.spec.budget.max_tokens) {
        outcome = swarm_task_outcome::budget_exhausted;
    }
    const std::string reason = completion.message.empty()
            ? outcome_name(outcome)
            : completion.message;
    const bool retry = impl_->finish_attempt(task, outcome, now, reason);
    result.accepted = true;
    result.retry_scheduled = retry;
    result.state = task.state;
    result.message = reason;
    if (retry) {
        result.disposition = swarm_completion_disposition::retry_scheduled;
    } else {
        switch (outcome) {
            case swarm_task_outcome::cancelled:
                result.disposition = swarm_completion_disposition::cancelled;
                break;
            case swarm_task_outcome::timed_out:
                result.disposition = swarm_completion_disposition::timed_out;
                break;
            case swarm_task_outcome::budget_exhausted:
                result.disposition = swarm_completion_disposition::budget_exhausted;
                break;
            case swarm_task_outcome::succeeded:
            case swarm_task_outcome::failed:
                result.disposition = swarm_completion_disposition::accepted;
                break;
        }
    }
    return result;
}

swarm_cancel_result qwen38_swarm_scheduler::cancel_task(
        swarm_task_id task_id,
        const std::string &reason,
        swarm_time_point now) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->expire_running(now, nullptr);
    swarm_cancel_result result;
    const auto task_it = impl_->tasks.find(task_id);
    if (task_it == impl_->tasks.end()) {
        result.disposition = swarm_cancel_disposition::not_found;
        return result;
    }
    result.disposition = impl_->cancel_task_locked(task_it->second, reason, now);
    result.changed = result.disposition == swarm_cancel_disposition::cancelled_before_run ||
                     result.disposition == swarm_cancel_disposition::cancellation_requested;
    return result;
}

std::size_t qwen38_swarm_scheduler::cancel_wave(
        swarm_wave_id wave_id,
        const std::string &reason,
        swarm_time_point now,
        bool include_descendants) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->expire_running(now, nullptr);
    if (impl_->waves.find(wave_id) == impl_->waves.end()) {
        return 0u;
    }

    std::set<swarm_wave_id> selected_waves;
    for (const auto &entry : impl_->waves) {
        if (entry.first == wave_id ||
            (include_descendants && impl_->wave_is_descendant(entry.first, wave_id, false))) {
            selected_waves.insert(entry.first);
        }
    }
    for (const swarm_wave_id selected_wave_id : selected_waves) {
        impl::wave_record &wave = impl_->waves.find(selected_wave_id)->second;
        wave.cancellation_requested = true;
        if (wave.state == swarm_wave_state::pending ||
            wave.state == swarm_wave_state::running) {
            impl_->set_wave_state(wave, swarm_wave_state::cancelling, now, reason);
        }
    }

    std::size_t changed = 0u;
    for (auto &entry : impl_->tasks) {
        impl::task_record &task = entry.second;
        bool selected = false;
        if (task.spec.wave_id.has_value() &&
            selected_waves.find(*task.spec.wave_id) != selected_waves.end()) {
            selected = true;
        } else if (include_descendants &&
                   impl_->task_belongs_to_wave_tree(task, wave_id, true)) {
            selected = true;
        }
        if (!selected) {
            continue;
        }
        const swarm_cancel_disposition disposition =
                impl_->cancel_task_locked(task, reason, now);
        if (disposition == swarm_cancel_disposition::cancelled_before_run ||
            disposition == swarm_cancel_disposition::cancellation_requested) {
            ++changed;
        }
    }
    for (const swarm_wave_id selected_wave_id : selected_waves) {
        impl_->recompute_wave(selected_wave_id, now);
    }
    return changed;
}

std::vector<swarm_task_id> qwen38_swarm_scheduler::tick(swarm_time_point now) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<swarm_task_id> expired;
    impl_->expire_running(now, &expired);
    return expired;
}

bool qwen38_swarm_scheduler::set_agent_state(
        swarm_agent_id agent_id,
        swarm_agent_state state,
        swarm_time_point now,
        const std::string &reason) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto agent_it = impl_->agents.find(agent_id);
    if (agent_it == impl_->agents.end() || state == swarm_agent_state::busy) {
        return false;
    }
    impl::agent_record &agent = agent_it->second;
    agent.administrative_state = state;
    impl_->emit(
            swarm_event_type::agent_state_changed,
            now,
            0u,
            0u,
            agent.id,
            0u,
            std::nullopt,
            std::nullopt,
            impl_->visible_agent_state(agent),
            0u,
            std::chrono::milliseconds(0),
            reason);
    return true;
}

bool qwen38_swarm_scheduler::is_cancellation_requested(swarm_task_id task_id) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto task_it = impl_->tasks.find(task_id);
    return task_it != impl_->tasks.end() && task_it->second.cancellation_requested;
}

std::optional<swarm_task_snapshot> qwen38_swarm_scheduler::task_snapshot(
        swarm_task_id task_id) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto task_it = impl_->tasks.find(task_id);
    if (task_it == impl_->tasks.end()) {
        return std::nullopt;
    }
    return impl_->make_task_snapshot(task_it->second);
}

swarm_scheduler_snapshot qwen38_swarm_scheduler::snapshot() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    swarm_scheduler_snapshot result;
    result.revision = impl_->revision;
    result.telemetry = impl_->telemetry;
    result.telemetry.active_tasks = impl_->active_tasks;
    result.telemetry.queued_tasks = impl_->queued_task_count();
    result.telemetry.registered_agents = impl_->agents.size();
    for (const auto &entry : impl_->agents) {
        const impl::agent_record &agent = entry.second;
        if (agent.administrative_state == swarm_agent_state::idle &&
            agent.active_tasks < agent.spec.max_concurrent) {
            ++result.telemetry.available_agents;
        }
        result.agents.push_back(impl_->make_agent_snapshot(agent));
    }
    for (const auto &entry : impl_->waves) {
        result.waves.push_back(impl_->make_wave_snapshot(entry.second));
    }
    for (const auto &entry : impl_->tasks) {
        result.tasks.push_back(impl_->make_task_snapshot(entry.second));
    }
    result.events.assign(impl_->events.begin(), impl_->events.end());
    return result;
}

}  // namespace qwen38
}  // namespace axiom
