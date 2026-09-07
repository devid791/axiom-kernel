// CPU only: tests the production worker implementation, not a parallel mock.
#define AXIOM_QWEN4EXP_PLE_WORKER_TEST_ONLY
#include "../../src/qwen4exp/ple_adapter.cu"

#include <array>
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>

namespace q4 = axiom::qwen4exp;
using q4::detail::ple_io_result;
using q4::detail::ple_io_worker;
using namespace std::chrono_literals;

namespace {
int checks = 0;
void require(bool value, const char *message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}

struct synthetic_rows {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, release = false, finished = false;
    int mode = 0;
    std::thread::id thread;
    std::array<float, 2560> staging{};
    q4::ple_head_ids observed{};

    static ple_io_result read(void *opaque, const q4::ple_head_ids &ids) {
        auto &self = *static_cast<synthetic_rows *>(opaque);
        {
            std::unique_lock<std::mutex> lock(self.mutex);
            self.entered = true;
            self.thread = std::this_thread::get_id();
            self.changed.notify_all();
            if (!self.changed.wait_for(lock, 2s, [&] { return self.release; }))
                return {q4::ple_adapter_status::checkpoint_io_error, "synthetic deadline"};
        }
        if (self.mode == 1) return {q4::ple_adapter_status::checkpoint_io_error, "synthetic row 7 short read"};
        if (self.mode == 2) throw std::bad_alloc();
        if (self.mode == 3) throw std::runtime_error("synthetic failure");
        for (std::size_t head = 0; head < ids.size(); ++head)
            for (std::size_t column = 0; column < 160; ++column)
                self.staging[head * 160 + column] = static_cast<float>(ids[head] + column);
        self.observed = ids;
        self.finished = true;
        return {};
    }

    bool await_entry() {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, 2s, [&] { return entered; });
    }
    void unblock() {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
        changed.notify_all();
    }
};

void worker_lifecycle() {
    synthetic_rows rows;
    ple_io_worker worker;
    q4::ple_head_ids ids{};
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = 10 + i;
    const auto original = ids;
    require(!worker.submit(&synthetic_rows::read, &rows, ids), "submit before start");
    require(worker.start(), "start worker");
    require(!worker.start(), "double start accepted");
    require(worker.submit(&synthetic_rows::read, &rows, ids), "first submit");
    ids.fill(-1);
    require(rows.await_entry(), "worker did not enter");
    require(!worker.submit(&synthetic_rows::read, &rows, ids), "second pending job accepted");
    const auto persistent_thread = rows.thread;
    require(persistent_thread != std::this_thread::get_id(), "read ran inline");
    // Model-side work can run while the synthetic I/O is deliberately blocked.
    std::uint64_t independent = 0;
    for (std::uint64_t i = 0; i < 4096; ++i) independent += i;
    require(independent == 8386560, "independent CPU work");
    rows.unblock();
    require(worker.collect().code == q4::ple_adapter_status::ok, "collect read");
    require(rows.observed == original, "IDs not copied at submission");
    require(rows.finished && rows.staging[2559] == original[15] + 159, "incomplete staging");
    require(worker.collect().code == q4::ple_adapter_status::invalid_state, "double collect");

    for (int mode = 0; mode != 4; ++mode) {
        rows.mode = mode; // Previous job has been collected: exclusive ownership.
        require(worker.submit(&synthetic_rows::read, &rows, original), "resubmit");
        const auto result = worker.collect();
        const std::array<q4::ple_adapter_status, 4> expected{{
                q4::ple_adapter_status::ok, q4::ple_adapter_status::checkpoint_io_error,
                q4::ple_adapter_status::resource_exhausted, q4::ple_adapter_status::embedding_error}};
        require(result.code == expected[mode], "worker lost error status");
        if (mode == 1) require(result.message == "synthetic row 7 short read", "worker lost error detail");
        require(rows.thread == persistent_thread, "worker recreated per job");
    }
    worker.shutdown();
    worker.shutdown();
    require(!worker.submit(&synthetic_rows::read, &rows, original), "submit after shutdown");
    require(!worker.start(), "restart after shutdown");
}

void destruction_drains() {
    // Declared first: source/staging outlive worker, exactly as production does.
    synthetic_rows rows;
    auto worker = std::make_unique<ple_io_worker>();
    q4::ple_head_ids ids{};
    require(worker->start(), "destruction start");
    require(worker->submit(&synthetic_rows::read, &rows, ids), "destruction submit");
    require(rows.await_entry(), "destruction entry");
    std::promise<void> destroying;
    auto begun = destroying.get_future();
    auto destroyed = std::async(std::launch::async, [&] {
        destroying.set_value();
        worker.reset();
    });
    require(begun.wait_for(2s) == std::future_status::ready, "destructor task not scheduled");
    const bool retained = destroyed.wait_for(20ms) == std::future_status::timeout;
    rows.unblock(); // Always release before assertions/stack unwinding.
    require(destroyed.wait_for(2s) == std::future_status::ready, "destructor did not join");
    destroyed.get();
    require(retained && rows.finished, "destruction returned before accepted job completed");
}

void cancelled_history_replay() {
    q4::ple_metadata metadata;
    metadata.unigram_vocab_size = 248320;
    metadata.eos_token_id = 248044;
    require(q4::ple_derive_layer_multipliers(metadata.unigram_vocab_size, 1, 1234,
            &metadata.layer_multipliers).ok(), "derive multipliers");
    require(q4::ple_derive_head_layout(20000000, 1, &metadata.ngram_heads_offsets,
            &metadata.ngram_heads_vocab_sizes).ok(), "derive layout");
    q4::ple_session_state history;
    require(history.configure(metadata, 1, 4).ok(), "configure history");
    synthetic_rows rows;
    ple_io_worker worker;
    require(worker.start(), "history worker start");
    q4::ple_head_ids ids{}, replay{};
    float marker = 0;
    std::array<float, q4::kPleConvKernelSize> taps{};
    require(history.begin_transaction().ok(), "begin cancelled history");
    require(history.stage_token(101, &ids).ok(), "stage cancelled token");
    require(history.stage_conv_frame(&marker, 1, taps.data(), taps.size()).ok(), "stage marker");
    require(worker.submit(&synthetic_rows::read, &rows, ids), "history submit");
    require(rows.await_entry(), "history entry");
    rows.unblock();
    require(worker.collect().code == q4::ple_adapter_status::ok, "drain before rollback");
    require(history.rollback().ok() && history.committed_tokens() == 0, "cancel published history");
    require(history.begin_transaction().ok() && history.stage_token(101, &replay).ok(), "replay history");
    require(ids == replay, "cancel changed EOS-aware hash IDs");
    require(history.rollback().ok(), "replay rollback");
}
} // namespace

int main() {
    try {
        worker_lifecycle();
        destruction_drains();
        cancelled_history_replay();
        std::printf("ple-io-worker-test: PASS checks=%d cpu_only=1 synthetic=1\n", checks);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "ple-io-worker-test: FAIL %s\n", error.what());
        return 1;
    }
}
