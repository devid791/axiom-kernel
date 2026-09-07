#include "axiom/qwen4exp/expert_bridge.hpp"
#include "axiom/qwen4exp/moe_adapter.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q4 = axiom::qwen4exp;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

q4::expert_bridge::prepared_generation prepare(
        q4::expert_bridge& bridge,
        const std::array<std::uint32_t, q4::kMoeTopK>& experts) {
    q4::expert_bridge::prepared_generation result;
    std::string error;
    check(bridge.prepare(experts.data(), experts.size(), nullptr, &result, &error), error);
    check(result.valid() && result.slot_count() == q4::kMoeTopK,
          "prepared generation did not acquire all ten shared slots");
    return result;
}

using expert_set = std::array<std::uint32_t, q4::kMoeTopK>;

expert_set expert_ids(std::uint32_t first) {
    expert_set result{};
    for (std::uint32_t i = 0u; i < result.size(); ++i) result[i] = first + i;
    return result;
}

struct test_stream {
    cudaStream_t value = nullptr;
    test_stream() {
        cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "test stream");
    }
    ~test_stream() { (void)cudaStreamDestroy(value); }
};

struct device_floats {
    float* value = nullptr;
    explicit device_floats(std::size_t count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&value), count * sizeof(float)),
                   "test allocation");
    }
    ~device_floats() { (void)cudaFree(value); }
};

struct numerical_fixture {
    device_floats input{q4::kMoeHidden};
    device_floats weights{q4::kMoeTopK};
    device_floats mid{q4::kMoeTopK * q4::kMoeIntermediate};
    device_floats output{q4::kMoeHidden};

    numerical_fixture() {
        std::array<float, q4::kMoeHidden> h{};
        std::array<float, q4::kMoeTopK> w{};
        for (std::size_t i = 0u; i < h.size(); ++i) {
            h[i] = static_cast<float>(static_cast<int>(i % 29u) - 14) / 64.0f;
        }
        for (std::size_t i = 0u; i < w.size(); ++i) w[i] = float(i + 1u) / 55.0f;
        cuda_check(cudaMemcpy(input.value, h.data(), sizeof(h), cudaMemcpyHostToDevice),
                   "test input upload");
        cuda_check(cudaMemcpy(weights.value, w.data(), sizeof(w), cudaMemcpyHostToDevice),
                   "test router weights upload");
    }

    void enqueue(const q4::expert_bridge::prepared_generation& generation,
                 cudaStream_t stream) {
        check(q4::moe_forward_slots_f32_cuda(
                      0, q4::moe_config{}, generation.physical_slot_capacity(),
                      generation.planes(), generation.slot_indices_device(),
                      weights.value, input.value, mid.value, output.value, stream) ==
                      q4::moe_status::ok,
              "physical-slot indexed kernel failed");
    }

    std::array<float, q4::kMoeHidden> collect(cudaStream_t stream) {
        std::array<float, q4::kMoeHidden> result{};
        cuda_check(cudaMemcpyAsync(result.data(), output.value, sizeof(result),
                                    cudaMemcpyDeviceToHost, stream), "test output copy");
        cuda_check(cudaStreamSynchronize(stream), "test output completion");
        for (const auto value : result) check(std::isfinite(value), "nonfinite output");
        return result;
    }
};

std::unique_ptr<q4::expert_bridge> open_bridge(
        const std::shared_ptr<const q4::expert_checkpoint_catalog>& catalog,
        const std::shared_ptr<q4::expert_slot_arena>& arena, std::uint32_t layer) {
    q4::expert_bridge_options options;
    options.device = 0;
    options.prefetch_workers = 1u;
    options.shared_arena = arena;
    std::unique_ptr<q4::expert_bridge> result;
    std::string error;
    check(q4::expert_bridge::open(catalog, layer, options, &result, &error), error);
    return result;
}

void persistent_cache_test(
        const std::shared_ptr<const q4::expert_checkpoint_catalog>& catalog,
        std::uint32_t capacity) {
    std::string error;
    std::shared_ptr<q4::expert_slot_arena> arena, transient;
    check(q4::expert_slot_arena::create(0, q4::kMoeTopK, capacity, &arena, &error), error);
    check(q4::expert_slot_arena::create(0, q4::kMoeTopK, &transient, &error), error);
    check(arena->metrics().persistent_cache && arena->metrics().physical_slots == capacity &&
                  arena->metrics().bounded_slots == q4::kMoeTopK &&
                  arena->metrics().capacity_bytes == capacity * 2764812ull + 40u &&
                  !transient->metrics().persistent_cache,
          "persistent physical/active slot capacity contract");
    auto cached = open_bridge(catalog, arena, 0u);
    auto alias = open_bridge(catalog, arena, 0u);
    auto other_layer = open_bridge(catalog, arena, 1u);
    auto reference = open_bridge(catalog, transient, 0u);
    auto other_reference = open_bridge(catalog, transient, 1u);
    numerical_fixture math;
    test_stream stream;

    auto compare = [&](q4::expert_bridge& bridge, q4::expert_bridge& baseline,
                       const expert_set& experts, std::uint64_t hits,
                       std::uint64_t misses, std::uint64_t evictions) {
        const auto before = arena->metrics();
        const auto pager_before = bridge.pager_metrics();
        q4::expert_bridge::prepared_generation generation;
        check(bridge.prepare(experts.data(), experts.size(), stream.value,
                              &generation, &error), error);
        check(generation.physical_slot_capacity() == capacity, "wrong kernel slot bound");
        expert_set mapped{};
        std::copy_n(generation.slot_indices_host(), mapped.size(), mapped.data());
        std::array<std::array<float, q4::kMoeTopK>, 3> scales{};
        for (std::uint32_t p = 0u; p < 3u; ++p) {
            for (std::uint32_t i = 0u; i < q4::kMoeTopK; ++i) {
                check(mapped[i] < capacity, "slot exceeds allocation");
                scales[p][i] = generation.input_scale(static_cast<q4::ExpertProjection>(p), i);
            }
        }
        math.enqueue(generation, stream.value);
        check(generation.handoff_after_consumers(stream.value, &error), error);
        const auto actual = math.collect(stream.value);
        check(generation.commit(&error), error);
        const auto after = arena->metrics();
        check(after.cache_hits - before.cache_hits == hits &&
                      after.cache_misses - before.cache_misses == misses &&
                      after.evictions - before.evictions == evictions &&
                      after.upload_bytes - before.upload_bytes == misses * 2764812ull,
              "cache hits/misses/uploads/evictions mismatch");
        if (misses == 0u) {
            const auto pager_after = bridge.pager_metrics();
            check(pager_after.requests == pager_before.requests &&
                          pager_after.nvme_reads == pager_before.nvme_reads &&
                          pager_after.nvme_bytes_read == pager_before.nvme_bytes_read,
                  "GPU cache hit accessed host pager/payload");
        }
        q4::expert_bridge::prepared_generation original;
        check(baseline.prepare(experts.data(), experts.size(), stream.value,
                                &original, &error), error);
        for (std::uint32_t p = 0u; p < 3u; ++p) {
            for (std::uint32_t i = 0u; i < q4::kMoeTopK; ++i) {
                check(scales[p][i] == original.input_scale(
                              static_cast<q4::ExpertProjection>(p), i),
                      "cached input scale differs from transient source");
            }
        }
        math.enqueue(original, stream.value);
        const auto expected = math.collect(stream.value);
        check(original.commit(&error), error);
        check(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
              "persistent output is not bitwise equal to transient output");
        return mapped;
    };

    const auto a = expert_ids(0u);
    const auto first_mapping = compare(*cached, *reference, a, 0u, 10u, 0u);
    compare(*cached, *reference, a, 10u, 0u, 0u);
    auto permutation = a;
    std::reverse(permutation.begin(), permutation.end());
    const auto permuted_mapping = compare(*alias, *reference, permutation, 10u, 0u, 0u);
    for (std::size_t i = 0u; i < a.size(); ++i) {
        check(permuted_mapping[i] == first_mapping[a.size() - 1u - i],
              "permutation did not preserve expert identity slots");
    }
    {
        // A changed checkpoint fingerprint must miss, even with identical
        // layer/expert numbers. Its key is intentionally absent in this pager.
        auto* identity = const_cast<q4::expert_tensor_descriptor*>(catalog->find(
                0u, a[0], q4::ExpertProjection::gate, q4::ExpertPlane::weight));
        check(identity != nullptr, "missing identity fault descriptor");
        std::string saved = identity->key.fingerprint;
        identity->key.fingerprint = "test-only-different-checkpoint";
        const auto before = arena->metrics();
        q4::expert_bridge::prepared_generation wrong_checkpoint;
        const bool accepted = cached->prepare(a.data(), a.size(), stream.value,
                                               &wrong_checkpoint, &error);
        identity->key.fingerprint.swap(saved);
        const auto after = arena->metrics();
        check(!accepted && !wrong_checkpoint.valid() &&
                      after.cache_hits - before.cache_hits == 9u &&
                      after.cache_misses - before.cache_misses == 1u &&
                      after.active_leases == 0u && !after.poisoned,
              "cache identity omitted checkpoint fingerprint");
    }
    for (std::uint32_t first = 10u; first < capacity; first += 10u) {
        const auto mapping = compare(*cached, *reference, expert_ids(first), 0u, 10u, 0u);
        check(*std::min_element(mapping.begin(), mapping.end()) >= 10u,
              "test did not exercise kernel physical indices beyond topK");
    }
    // A is oldest and the cache is full. Position zero misses, but positions
    // 1..9 must all remain hits even if their slots are the oldest victims.
    expert_set mixed{};
    mixed[0] = 100u;
    for (std::uint32_t i = 1u; i < 10u; ++i) mixed[i] = permutation[i - 1u];
    const auto protected_mapping = compare(*cached, *reference, mixed, 9u, 1u, 1u);
    for (std::uint32_t i = 1u; i < 10u; ++i) {
        check(protected_mapping[i] == first_mapping[mixed[i]], "later batch hit evicted");
    }
    compare(*alias, *reference, mixed, 10u, 0u, 0u);
    compare(*other_layer, *other_reference, a, 0u, 10u, 10u);
    compare(*other_layer, *other_reference, a, 10u, 0u, 0u);

    // Cross-stream handoff + late rollback of an older generation on the SAME
    // bridge must not release a newer active reservation or invalidate cache.
    const auto b = expert_ids(200u);
    compare(*other_layer, *other_reference, b, 0u, 10u, 10u);
    test_stream second_stream;
    numerical_fixture second_math;
    q4::expert_bridge::prepared_generation older, newer, rejected;
    const auto before_handoff = arena->metrics();
    const auto before_handoff_pager = other_layer->pager_metrics();
    check(other_layer->prepare(a.data(), a.size(), stream.value, &older, &error), error);
    // Keep consumers queued while the hit-only next batch remaps shared indices.
    for (unsigned repeat = 0u; repeat < 4u; ++repeat) math.enqueue(older, stream.value);
    check(older.handoff_after_consumers(stream.value, &error), error);
    check(other_layer->prepare(b.data(), b.size(), second_stream.value, &newer, &error), error);
    check(arena->metrics().cache_hits - before_handoff.cache_hits == 20u &&
                  arena->metrics().upload_bytes == before_handoff.upload_bytes &&
                  other_layer->pager_metrics().requests == before_handoff_pager.requests,
          "cross-stream handoff was not hit-only");
    check(older.rollback(&error), error);
    check(arena->metrics().active_leases == 1u, "late rollback released newer reservation");
    check(!cached->prepare(a.data(), a.size(), stream.value, &rejected, &error),
          "late rollback permitted unsafe arena overwrite");
    second_math.enqueue(newer, second_stream.value);
    // No explicit handoff: commit must record consumers, not merely uploads.
    check(newer.commit(&error), error);
    const auto older_output = math.collect(stream.value);
    const auto newer_output = second_math.collect(second_stream.value);
    for (const auto* ids : {&a, &b}) {
        q4::expert_bridge::prepared_generation expected_generation;
        check(other_reference->prepare(ids->data(), ids->size(), stream.value,
                                        &expected_generation, &error), error);
        math.enqueue(expected_generation, stream.value);
        const auto expected = math.collect(stream.value);
        check(expected_generation.commit(&error), error);
        const auto& actual = ids == &a ? older_output : newer_output;
        check(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
              "cross-stream handoff/rollback changed numerical output");
    }
    compare(*other_layer, *other_reference, b, 10u, 0u, 0u);

    // Test-only descriptor fault, never a checkpoint-file mutation: mismatch a
    // late plane after earlier planes have already been enqueued. Restore the
    // originally non-const catalog object immediately after prepare returns.
    const auto bad_ids = expert_ids(300u);
    auto* descriptor = const_cast<q4::expert_tensor_descriptor*>(catalog->find(
            0u, bad_ids[0], q4::ExpertProjection::down, q4::ExpertPlane::weight));
    check(descriptor != nullptr, "missing fault-injection descriptor");
    const auto saved_bytes = descriptor->bytes;
    ++descriptor->bytes;
    q4::expert_bridge::prepared_generation failed;
    const bool unexpectedly_prepared = cached->prepare(
            bad_ids.data(), bad_ids.size(), stream.value, &failed, &error);
    descriptor->bytes = saved_bytes;
    check(!unexpectedly_prepared && !failed.valid() &&
                  arena->metrics().active_leases == 0u && !arena->metrics().poisoned,
          "partial-write failure did not cleanly release arena");
    // Every reserved miss must be invalid; no partially populated entry hits.
    compare(*cached, *reference, bad_ids, 0u, 10u, 0u);
    {
        q4::expert_bridge::prepared_generation abandoned;
        check(cached->prepare(bad_ids.data(), bad_ids.size(), stream.value,
                               &abandoned, &error), error);
        math.enqueue(abandoned, stream.value);
        // Destructor supplies the missing consumer handoff and pager rollback.
    }
    compare(*alias, *reference, bad_ids, 10u, 0u, 0u);
    check(!arena->metrics().poisoned && arena->metrics().active_leases == 0u,
          "persistent cache leaked/poisoned its lease");
    std::printf("persistent-cache: PASS slots=%u repeat permutation layer-key LRU "
                "fingerprint batch-protection cross-stream late-rollback partial-write "
                "bitwise-parity\n",
                capacity);
}

// Snapshot all twelve planes in catalog order, including CPU input scales.
// Numerical equality alone could miss a scalar or a zero-weight plane error.
std::vector<std::uint8_t> snapshot(q4::expert_bridge::prepared_generation& generation,
                                   cudaStream_t stream) {
    std::string error;
    check(generation.wait(&error), error);
    const auto& planes = generation.planes();
    const std::array<const std::uint8_t*, 3> weights{{planes.gate_weight, planes.up_weight,
                                                    planes.down_weight}};
    const std::array<const std::uint8_t*, 3> scales{{planes.gate_block_scale, planes.up_block_scale,
                                                   planes.down_block_scale}};
    const std::array<const float*, 3> globals{{planes.gate_global_scale, planes.up_global_scale,
                                              planes.down_global_scale}};
    std::vector<std::uint8_t> result(generation.slot_count() * q4::kExpertBridgeBytesPerExpert);
    std::size_t offset = 0u;
    for (std::uint32_t i = 0u; i < generation.slot_count(); ++i) {
        const auto slot = generation.slot_indices_host()[i];
        for (std::uint32_t p = 0u; p < 3u; ++p) {
            auto copy = [&](const void* source, std::size_t bytes) {
                cuda_check(cudaMemcpyAsync(result.data() + offset, source, bytes,
                                            cudaMemcpyDeviceToHost, stream), "plane snapshot");
                offset += bytes;
            };
            copy(weights[p] + slot * 819200ull, 819200u);
            const float input_scale = generation.input_scale(static_cast<q4::ExpertProjection>(p), i);
            std::memcpy(result.data() + offset, &input_scale, sizeof(input_scale));
            offset += sizeof(input_scale);
            copy(scales[p] + slot * 102400ull, 102400u);
            copy(globals[p] + slot, sizeof(float));
        }
    }
    cuda_check(cudaStreamSynchronize(stream), "plane snapshot completion");
    return result;
}

void exclusive_cache_test(
        const std::shared_ptr<const q4::expert_checkpoint_catalog>& catalog,
        std::uint32_t capacity) {
    constexpr auto bytes = q4::kExpertBridgeBytesPerExpert;
    constexpr auto transfer_bytes = bytes - 12u;
    std::string error;
    std::shared_ptr<q4::expert_slot_arena> arena, transient;
    check(!q4::expert_slot_arena::create(0, q4::kMoeTopK, capacity, bytes - 1u,
                                        &arena, &error) && !arena,
          "sub-expert cold budget was accepted");
    check(q4::expert_slot_arena::create(0, q4::kMoeTopK, capacity, 12u * bytes,
                                       &arena, &error), error);
    check(q4::expert_slot_arena::create(0, q4::kMoeTopK, &transient, &error), error);
    check(transient->metrics().host_cold_capacity_bytes == 0u,
          "legacy overload enabled host caching");
    auto cached = open_bridge(catalog, arena, 0u);
    auto alias = open_bridge(catalog, arena, 0u);
    auto other = open_bridge(catalog, arena, 1u);
    auto reference = open_bridge(catalog, transient, 0u);
    auto other_reference = open_bridge(catalog, transient, 1u);
    numerical_fixture math;
    test_stream stream;
    auto bounded = [&] {
        const auto m = arena->metrics();
        check(m.host_cold_capacity_bytes == 12u * bytes &&
                      m.host_cold_current_bytes <= m.host_cold_capacity_bytes &&
                      m.host_cold_peak_bytes <= m.host_cold_capacity_bytes && !m.poisoned,
              "exclusive host capacity (including leased buffers) exceeded");
    };
    auto compare = [&](q4::expert_bridge& bridge, q4::expert_bridge& baseline,
                       const expert_set& ids, std::uint64_t hot, std::uint64_t cold,
                       std::uint64_t nvme, bool rollback = false) {
        const auto before = arena->metrics();
        const auto pager = bridge.pager_metrics();
        q4::expert_bridge::prepared_generation generation;
        check(bridge.prepare(ids.data(), ids.size(), stream.value, &generation, &error), error);
        bounded();  // Check BEFORE ready_event releases the last promotion lease.
        const auto actual_planes = snapshot(generation, stream.value);
        math.enqueue(generation, stream.value);
        const auto actual = math.collect(stream.value);
        check(rollback ? generation.rollback(&error) : generation.commit(&error), error);
        const auto after = arena->metrics();
        const auto pager_after = bridge.pager_metrics();
        check(after.cache_hits - before.cache_hits == hot &&
                      after.host_cold_hits - before.host_cold_hits == cold &&
                      after.host_cold_misses - before.host_cold_misses == nvme &&
                      after.upload_bytes - before.upload_bytes == (cold + nvme) * transfer_bytes,
              "exclusive GPU/RAM/NVMe hit accounting mismatch");
        check(pager_after.requests - pager.requests == nvme * 12u &&
                      pager_after.nvme_bytes_read - pager.nvme_bytes_read == nvme * bytes &&
                      pager_after.current_ram_bytes == 0u && pager_after.active_leases == 0u,
              "exclusive path accessed NVMe on a hit or retained a persistent pager duplicate");
        q4::expert_bridge::prepared_generation original;
        check(baseline.prepare(ids.data(), ids.size(), stream.value, &original, &error), error);
        check(actual_planes == snapshot(original, stream.value), "exclusive twelve-plane roundtrip changed bits");
        math.enqueue(original, stream.value);
        const auto expected = math.collect(stream.value);
        check(original.commit(&error), error);
        check(std::memcmp(actual.data(), expected.data(), sizeof(actual)) == 0,
              "exclusive output differs bitwise from transient bridge");
        bounded();
    };
    const auto a = expert_ids(0u);
    for (std::uint32_t first = 0u; first < capacity; first += 10u) {
        compare(*cached, *reference, expert_ids(first), 0u, 0u, 10u);
    }
    compare(*cached, *reference, expert_ids(capacity), 0u, 0u, 10u);
    check(arena->metrics().host_demote_bytes == 10u * transfer_bytes &&
                  arena->metrics().host_cold_current_bytes == 10u * bytes,
          "GPU eviction did not demote exact payload bytes");
    compare(*alias, *reference, a, 0u, 10u, 0u, true);
    compare(*cached, *reference, a, 10u, 0u, 0u);
    compare(*other, *other_reference, a, 0u, 0u, 10u);
    compare(*other, *other_reference, a, 10u, 0u, 0u);

    // Fault a late source plane after earlier H2D writes. The old GPU victim
    // must already be safely demoted; partially uploaded new identities miss.
    const auto bad = expert_ids(400u);
    auto* descriptor = const_cast<q4::expert_tensor_descriptor*>(catalog->find(
            0u, bad[0], q4::ExpertProjection::down, q4::ExpertPlane::weight));
    const auto saved_bytes = descriptor->bytes;
    ++descriptor->bytes;
    q4::expert_bridge::prepared_generation failed;
    const bool accepted = cached->prepare(bad.data(), bad.size(), stream.value, &failed, &error);
    descriptor->bytes = saved_bytes;
    check(!accepted && !failed.valid() && arena->metrics().active_leases == 0u &&
                  cached->pager_metrics().active_leases == 0u &&
                  cached->pager_metrics().current_ram_bytes == 0u,
          "exclusive partial-copy failure leaked leases/source payloads");
    compare(*cached, *reference, bad, 0u, 0u, 10u);

    // A handed-off cold upload retains its host source until ready/commit.
    // A late rollback must neither release nor invalidate the new reservation.
    test_stream second;
    q4::expert_bridge::prepared_generation older, newer;
    check(cached->prepare(a.data(), a.size(), stream.value, &older, &error), error);
    math.enqueue(older, stream.value);
    check(older.handoff_after_consumers(stream.value, &error), error);
    check(other->prepare(bad.data(), bad.size(), second.value, &newer, &error), error);
    bounded();
    check(older.rollback(&error), error);
    check(arena->metrics().active_leases == 1u, "exclusive late rollback released newer arena owner");
    check(newer.commit(&error), error);
    bounded();

    std::printf("exclusive-cache: PASS slots=%u exact-planes bitwise-output cold-roundtrip "
                "repeat cross-layer rollback partial-copy bounded-inflight\n", capacity);
}

void exclusive_tiny_cache_test(
        const std::shared_ptr<const q4::expert_checkpoint_catalog>& catalog,
        std::uint32_t capacity) {
    constexpr auto bytes = q4::kExpertBridgeBytesPerExpert;
    std::string error;
    std::shared_ptr<q4::expert_slot_arena> arena;
    check(q4::expert_slot_arena::create(0, q4::kMoeTopK, capacity, 2u * bytes,
                                       &arena, &error), error);
    auto bridge = open_bridge(catalog, arena, 0u);
    test_stream stream;
    for (std::uint32_t first = 0u; first <= capacity; first += 10u) {
        auto ids = expert_ids(first);
        q4::expert_bridge::prepared_generation generation;
        check(bridge->prepare(ids.data(), ids.size(), stream.value, &generation, &error), error);
        check(generation.commit(&error), error);
    }
    // Only experts 8 and 9 survive the ten demotions into a two-expert budget.
    auto one = [&](std::uint32_t id, bool cold_hit, bool rollback = false) {
        const auto before = arena->metrics();
        const auto pager = bridge->pager_metrics();
        q4::expert_bridge::prepared_generation generation;
        check(bridge->prepare(&id, 1u, stream.value, &generation, &error), error);
        const auto in_flight = arena->metrics();
        check(in_flight.host_cold_current_bytes <= 2u * bytes &&
                      in_flight.host_cold_peak_bytes <= 2u * bytes,
              "tiny budget ignored retained promotion source");
        const auto payload = snapshot(generation, stream.value);
        check(rollback ? generation.rollback(&error) : generation.commit(&error), error);
        check(arena->metrics().host_cold_hits - before.host_cold_hits == (cold_hit ? 1u : 0u) &&
                      bridge->pager_metrics().nvme_bytes_read - pager.nvme_bytes_read ==
                              (cold_hit ? 0u : bytes) && bridge->pager_metrics().current_ram_bytes == 0u,
              "tiny RAM LRU did not distinguish retained versus NVMe-evicted expert");
        return payload;
    };
    const auto recovered = one(9u, true, true);
    one(8u, false);
    std::shared_ptr<q4::expert_slot_arena> transient;
    check(q4::expert_slot_arena::create(0, q4::kMoeTopK, &transient, &error), error);
    auto reference = open_bridge(catalog, transient, 0u);
    const std::uint32_t nine = 9u;
    q4::expert_bridge::prepared_generation baseline;
    check(reference->prepare(&nine, 1u, stream.value, &baseline, &error), error);
    check(recovered == snapshot(baseline, stream.value), "tiny-cache D2H/H2D changed exact planes");
    check(baseline.commit(&error), error);

    // Most recently demoted expert is 11. A different fingerprint cannot hit
    // its cold bytes. The pager rejects this unregistered test-only identity.
    const std::uint32_t eleven = 11u;
    auto* identity = const_cast<q4::expert_tensor_descriptor*>(catalog->find(
            0u, eleven, q4::ExpertProjection::gate, q4::ExpertPlane::weight));
    std::string saved = identity->key.fingerprint;
    identity->key.fingerprint = "exclusive-test-different-checkpoint";
    const auto before = arena->metrics();
    q4::expert_bridge::prepared_generation wrong;
    const bool accepted = bridge->prepare(&eleven, 1u, stream.value, &wrong, &error);
    identity->key.fingerprint.swap(saved);
    check(!accepted && !wrong.valid() &&
                  arena->metrics().host_cold_hits == before.host_cold_hits &&
                  arena->metrics().active_leases == 0u && !arena->metrics().poisoned,
          "exclusive cold identity omitted fingerprint or failure leaked arena");
    one(eleven, true);
    std::printf("exclusive-tiny-cache: PASS slots=%u cold-experts=2 LRU NVMe-fallback fingerprint\n",
                capacity);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    try {
        std::string error;
        std::shared_ptr<const q4::expert_checkpoint_catalog> catalog;
        check(q4::expert_checkpoint_catalog::open(
                      argv[1], q4::pinned_expert_checkpoint_identity(), &catalog, &error),
              error);

        std::size_t free_before = 0u;
        std::size_t total = 0u;
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        cuda_check(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo(before)");

        std::shared_ptr<q4::expert_slot_arena> arena;
        check(q4::expert_slot_arena::create(0, q4::kMoeTopK, &arena, &error), error);
        const q4::expert_slot_arena_metrics created = arena->metrics();
        check(created.device == 0 && created.bounded_slots == q4::kMoeTopK &&
                      created.capacity_bytes == 27648160u &&
                      created.attached_bridges == 0u && created.active_leases == 0u,
              "new SM120 arena capacity/lease contract mismatch");

        std::vector<std::unique_ptr<q4::expert_bridge>> bridges;
        bridges.reserve(q4::kExpertBridgeLayers);
        std::uint64_t owned_sum = 0u;
        for (std::uint32_t layer = 0u; layer < q4::kExpertBridgeLayers; ++layer) {
            q4::expert_bridge_options options;
            options.device = 0;
            options.max_slots = q4::kMoeTopK;
            options.ram_capacity_bytes =
                    q4::kExpertBridgeBytesPerExpert * q4::kMoeTopK + 4096u;
            options.prefetch_workers = 1u;
            options.shared_arena = arena;
            std::unique_ptr<q4::expert_bridge> bridge;
            check(q4::expert_bridge::open(catalog, layer, options, &bridge, &error), error);
            const q4::expert_bridge_metrics metrics = bridge->metrics();
            check(metrics.layer == layer && metrics.external_arena &&
                          metrics.gpu_owned_bytes == 0u &&
                          metrics.gpu_shared_bytes == created.capacity_bytes &&
                          metrics.gpu_capacity_bytes == created.capacity_bytes,
                  "bridge did not report shared-only GPU expert capacity");
            owned_sum += metrics.gpu_owned_bytes;
            bridges.push_back(std::move(bridge));
        }
        const q4::expert_slot_arena_metrics attached = arena->metrics();
        check(attached.attached_bridges == q4::kExpertBridgeLayers &&
                      attached.capacity_bytes == created.capacity_bytes && owned_sum == 0u,
              "48 bridges multiplied or misreported the shared GPU capacity");

        std::size_t free_after_bridges = 0u;
        cuda_check(cudaMemGetInfo(&free_after_bridges, &total),
                   "cudaMemGetInfo(after bridges)");
        const std::size_t device_delta = free_before - free_after_bridges;
        check(device_delta >= created.capacity_bytes &&
                      device_delta < created.capacity_bytes + 8u * 1024u * 1024u,
              "48 shared bridges allocated more than one bounded GPU arena");

        std::array<std::uint32_t, q4::kMoeTopK> experts{};
        for (std::uint32_t index = 0u; index < q4::kMoeTopK; ++index) {
            experts[index] = index;
        }

        {
            auto first = prepare(*bridges[0], experts);
            check(arena->metrics().active_leases == 1u,
                  "first prepared generation did not hold the global lease");
            q4::expert_bridge::prepared_generation rejected;
            error.clear();
            check(!bridges[1]->prepare(experts.data(), experts.size(), nullptr,
                                       &rejected, &error) &&
                          error.find("arena is busy") != std::string::npos,
                  "concurrent bridge did not fail busy on the shared arena");
            check(first.rollback(&error), error);
            check(arena->metrics().active_leases == 0u,
                  "rollback did not release the global arena lease");
        }

        {
            auto committed = prepare(*bridges[1], experts);
            check(committed.commit(&error) && committed.committed(), error);
            check(arena->metrics().active_leases == 0u,
                  "commit did not release the global arena lease");
        }

        {
            auto first = prepare(*bridges[1], experts);
            check(first.handoff_after_consumers(nullptr, &error), error);
            check(first.valid() && arena->metrics().active_leases == 0u,
                  "ordered handoff did not retain transaction and release slots");
            auto second = prepare(*bridges[2], experts);
            check(second.rollback(&error), error);
            check(first.commit(&error), error);
        }

        {
            auto disconnected = prepare(*bridges[2], experts);
            check(arena->metrics().active_leases == 1u,
                  "disconnect setup did not hold a lease");
        }
        check(arena->metrics().active_leases == 0u,
              "prepared-generation destruction did not release the lease");
        {
            auto after_disconnect = prepare(*bridges[3], experts);
            check(after_disconnect.rollback(&error), error);
        }

        bridges.clear();
        check(arena->metrics().attached_bridges == 0u,
              "destroyed bridge set remained attached to the shared arena");

        q4::moe_adapter_options first_options;
        first_options.device = 0;
        first_options.layer = 4u;
        first_options.shared_expert_arena = arena;
        first_options.shared_expert_catalog = catalog;
        q4::moe_adapter_options second_options = first_options;
        second_options.layer = 5u;
        std::unique_ptr<q4::routed_moe_layer_adapter> first_layer;
        std::unique_ptr<q4::routed_moe_layer_adapter> second_layer;
        check(q4::routed_moe_layer_adapter::load(
                      argv[1], first_options, nullptr, &first_layer, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        check(q4::routed_moe_layer_adapter::load(
                      argv[1], second_options, nullptr, &second_layer, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        const q4::moe_adapter_metrics first_metrics = first_layer->metrics();
        const q4::moe_adapter_metrics second_metrics = second_layer->metrics();
        check(first_metrics.expert_gpu_owned_bytes == 0u &&
                      first_metrics.expert_gpu_shared_bytes == created.capacity_bytes &&
                      second_metrics.expert_gpu_owned_bytes == 0u &&
                      second_metrics.expert_gpu_shared_bytes == created.capacity_bytes,
              "MoE layer metrics did not separate owned/shared expert bytes");

        float* device_input = nullptr;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_input), 2560u * sizeof(float)),
                   "cudaMalloc(MoE route input)");
        std::array<float, 2560u> host_input{};
        for (std::size_t index = 0u; index < host_input.size(); ++index) {
            host_input[index] = static_cast<float>((index % 29u) + 1u) / 64.0f;
        }
        cuda_check(cudaMemcpy(device_input, host_input.data(),
                              host_input.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy(MoE route input)");
        check(first_layer->enqueue_route(device_input, nullptr, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        check(second_layer->enqueue_route(device_input, nullptr, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        q4::routed_moe_layer_adapter::prepared_generation first_prepared;
        q4::routed_moe_layer_adapter::prepared_generation second_rejected;
        check(first_layer->prepare_route(nullptr, &first_prepared, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        error.clear();
        check(second_layer->prepare_route(nullptr, &second_rejected, &error) ==
                      q4::moe_adapter_status::busy,
              "concurrent MoE layer did not map the shared-arena conflict to busy");
        check(first_prepared.rollback(&error), error);
        check(second_layer->enqueue_route(device_input, nullptr, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        q4::routed_moe_layer_adapter::prepared_generation second_prepared;
        check(second_layer->prepare_route(nullptr, &second_prepared, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        check(second_prepared.rollback(&error), error);

        float* first_output = nullptr;
        float* second_output = nullptr;
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&first_output),
                              2560u * sizeof(float)),
                   "cudaMalloc(first MoE output)");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&second_output),
                              2560u * sizeof(float)),
                   "cudaMalloc(second MoE output)");
        check(first_layer->enqueue_route(device_input, nullptr, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        check(second_layer->enqueue_route(device_input, nullptr, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        q4::routed_moe_layer_adapter::prepared_generation first_forward;
        q4::routed_moe_layer_adapter::prepared_generation second_forward;
        check(first_layer->prepare_route(nullptr, &first_forward, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        check(first_forward.forward(device_input, first_output, nullptr,
                                    &error) == q4::moe_adapter_status::ok,
              error);
        check(first_forward.arena_handed_off() &&
                      arena->metrics().active_leases == 0u,
              "first MoE forward did not hand off the shared slots");
        check(second_layer->prepare_route(nullptr, &second_forward, &error) ==
                      q4::moe_adapter_status::ok,
              error);
        check(second_forward.forward(device_input, second_output, nullptr,
                                     &error) == q4::moe_adapter_status::ok,
              error);
        check(second_forward.arena_handed_off(),
              "second MoE forward did not hand off the shared slots");
        check(first_forward.commit(&error), error);
        check(second_forward.commit(&error), error);
        cuda_check(cudaFree(second_output), "cudaFree(second MoE output)");
        cuda_check(cudaFree(first_output), "cudaFree(first MoE output)");
        cuda_check(cudaFree(device_input), "cudaFree(MoE route input)");

        const q4::expert_slot_arena_metrics final_arena = arena->metrics();
        check(final_arena.ordered_handoffs >= 3u &&
                      final_arena.stream_waits >= 2u &&
                      final_arena.active_leases == 0u,
              "stream-ordered arena reuse metrics are incomplete");

        persistent_cache_test(catalog, 20u);
        persistent_cache_test(catalog, 30u);
        exclusive_cache_test(catalog, 20u);
        exclusive_cache_test(catalog, 30u);
        exclusive_tiny_cache_test(catalog, 20u);
        exclusive_tiny_cache_test(catalog, 30u);

        std::printf(
                "qwen4exp-expert-shared-arena-test: PASS bridges=48 slots=10 "
                "capacity=%llu owned_sum=%llu device_delta=%zu "
                "bridge_busy=pass layer_busy=pass commit=release "
                "rollback=release disconnect=release handoffs=%llu waits=%llu\n",
                static_cast<unsigned long long>(created.capacity_bytes),
                static_cast<unsigned long long>(owned_sum), device_delta,
                static_cast<unsigned long long>(final_arena.ordered_handoffs),
                static_cast<unsigned long long>(final_arena.stream_waits));
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "qwen4exp-expert-shared-arena-test: %s\n",
                     exception.what());
        return 1;
    }
}
