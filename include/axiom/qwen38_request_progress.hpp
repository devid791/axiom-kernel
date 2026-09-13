#ifndef AXIOM_QWEN38_REQUEST_PROGRESS_HPP
#define AXIOM_QWEN38_REQUEST_PROGRESS_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace axiom::qwen38 {

// Request-owned observations, never a timer-driven estimate or a cache commit.
// Readers do not acquire the model/generation lock. Heartbeats only read this.
class request_progress {
public:
    using clock = std::chrono::steady_clock;
    enum class phase { session_setup, prefill, decode };
    struct snapshot {
        phase stage = phase::session_setup;
        uint32_t cached_tokens = 0;
        uint32_t prompt_tokens = 0;
        uint32_t processed_tokens = 0; // New prompt tokens only, not cached tokens.
        uint32_t generated_tokens = 0; // Includes non-visible/tool tokens.
        uint32_t pass = 1; // Thinking and forced visible continuation are separate passes.
        uint64_t revision = 0;
        double elapsed_seconds = 0;
        double seconds_since_advance = 0;
    };

    explicit request_progress(clock::time_point now = clock::now())
        : started_(now), advanced_(now) {}

    bool prefill(uint32_t cached, uint32_t position, uint32_t total,
                 clock::time_point now = clock::now()) {
        if (cached > position || position > total) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (value_.stage == phase::decode) return false;
        const uint32_t done = position - cached;
        if (value_.stage == phase::prefill &&
            (cached != value_.cached_tokens || total != value_.prompt_tokens ||
             done < value_.processed_tokens)) return false;
        if (value_.stage != phase::prefill || done != value_.processed_tokens) {
            value_.stage = phase::prefill;
            value_.cached_tokens = cached;
            value_.prompt_tokens = total;
            value_.processed_tokens = done;
            advance(now);
        }
        return true;
    }

    bool decode(uint32_t generated, clock::time_point now = clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (value_.stage == phase::session_setup ||
            value_.processed_tokens != value_.prompt_tokens - value_.cached_tokens ||
            generated < pass_generated_) return false;
        if (value_.stage != phase::decode || generated != pass_generated_) {
            value_.stage = phase::decode;
            value_.generated_tokens += generated - pass_generated_;
            pass_generated_ = generated;
            advance(now);
        }
        return true;
    }

    bool next_pass(clock::time_point now = clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (value_.stage != phase::decode) return false;
        value_.stage = phase::session_setup;
        value_.cached_tokens = value_.prompt_tokens = value_.processed_tokens = 0;
        pass_generated_ = 0;
        ++value_.pass;
        advance(now);
        return true;
    }

    snapshot read(clock::time_point now = clock::now()) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto result = value_;
        // A reader can capture `now` before waiting for a writer's lock.
        result.elapsed_seconds = std::max(0.0, std::chrono::duration<double>(now - started_).count());
        result.seconds_since_advance = std::max(0.0, std::chrono::duration<double>(now - advanced_).count());
        return result;
    }

private:
    void advance(clock::time_point now) { ++value_.revision; advanced_ = now; }
    mutable std::mutex mutex_;
    snapshot value_;
    uint32_t pass_generated_ = 0;
    clock::time_point started_, advanced_;
};
} // namespace axiom::qwen38
#endif
