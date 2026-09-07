#include "axiom/qwen4exp/expert_bridge.hpp"

#include "axiom_aliced_json.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace q4 = axiom::qwen4exp;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void check(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void pread_exact(int descriptor,
                 void* destination,
                 std::size_t bytes,
                 std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        fail("independent pread offset exceeds off_t");
    }
    std::size_t complete = 0u;
    while (complete < bytes) {
        const ssize_t count = ::pread(
                descriptor, static_cast<std::uint8_t*>(destination) + complete,
                bytes - complete, static_cast<off_t>(offset + complete));
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            fail(std::string("independent pread failed: ") + std::strerror(errno));
        }
        complete += static_cast<std::size_t>(count);
    }
}

std::uint64_t load_u64_le(const std::uint8_t bytes[8]) noexcept {
    std::uint64_t value = 0u;
    for (unsigned index = 0; index < 8u; ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8u);
    }
    return value;
}

struct independent_shard {
    explicit independent_shard(const std::string& path) : path_(path) {
        struct stat status {};
        check(::lstat(path.c_str(), &status) == 0 && !S_ISLNK(status.st_mode) &&
                      S_ISREG(status.st_mode),
              "independent shard is not a regular non-symlink file");
        descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        check(descriptor_ >= 0, "independent shard open failed");
        std::uint8_t prefix[8]{};
        pread_exact(descriptor_, prefix, sizeof(prefix), 0u);
        header_bytes_ = load_u64_le(prefix);
        check(header_bytes_ > 0u && header_bytes_ <= 64ull * 1024ull * 1024ull,
              "independent Safetensors header length is invalid");
        std::string header(static_cast<std::size_t>(header_bytes_), '\0');
        pread_exact(descriptor_, header.data(), header.size(), 8u);
        std::string parse_error;
        check(ajson_parse(header, parsed_, parse_error) && parsed_.is_object(),
              "independent Safetensors JSON parse failed: " + parse_error);
    }

    ~independent_shard() {
        if (descriptor_ >= 0) {
            ::close(descriptor_);
        }
    }

    independent_shard(const independent_shard&) = delete;
    independent_shard& operator=(const independent_shard&) = delete;

    struct tensor {
        std::string dtype;
        std::vector<std::uint64_t> shape;
        std::uint64_t file_offset = 0u;
        std::uint64_t bytes = 0u;
    };

    tensor describe(const std::string& name) const {
        const ajson* value = parsed_.get(name.c_str());
        check(value != nullptr && value->is_object(),
              "independent header is missing tensor " + name);
        const ajson* dtype = value->get("dtype");
        const ajson* shape = value->get("shape");
        const ajson* offsets = value->get("data_offsets");
        check(dtype != nullptr && dtype->is_string() && shape != nullptr &&
                      shape->is_array() && offsets != nullptr && offsets->is_array() &&
                      offsets->arr.size() == 2u && offsets->arr[0].is_int() &&
                      offsets->arr[1].is_int() && offsets->arr[0].i >= 0 &&
                      offsets->arr[1].i >= offsets->arr[0].i,
              "independent tensor descriptor is malformed: " + name);
        tensor result;
        result.dtype = dtype->s;
        for (const ajson& dimension : shape->arr) {
            check(dimension.is_int() && dimension.i >= 0,
                  "independent tensor shape is malformed: " + name);
            result.shape.push_back(static_cast<std::uint64_t>(dimension.i));
        }
        const std::uint64_t begin = static_cast<std::uint64_t>(offsets->arr[0].i);
        const std::uint64_t end = static_cast<std::uint64_t>(offsets->arr[1].i);
        check(header_bytes_ <= std::numeric_limits<std::uint64_t>::max() - 8u - begin,
              "independent tensor offset overflow");
        result.file_offset = 8u + header_bytes_ + begin;
        result.bytes = end - begin;
        return result;
    }

    std::vector<std::uint8_t> read(const tensor& value) const {
        check(value.bytes <= std::numeric_limits<std::size_t>::max(),
              "independent tensor is too large for size_t");
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(value.bytes));
        pread_exact(descriptor_, bytes.data(), bytes.size(), value.file_offset);
        return bytes;
    }

private:
    std::string path_;
    int descriptor_ = -1;
    std::uint64_t header_bytes_ = 0u;
    ajson parsed_;
};

std::string dtype_name(q4::tensor_dtype dtype) {
    switch (dtype) {
        case q4::tensor_dtype::f32: return "F32";
        case q4::tensor_dtype::f8_e4m3: return "F8_E4M3";
        case q4::tensor_dtype::u8: return "U8";
        default: return "unsupported";
    }
}

void verify_descriptor(const q4::expert_tensor_descriptor& descriptor,
                       const independent_shard::tensor& independent) {
    check(dtype_name(descriptor.dtype) == independent.dtype,
          "catalog dtype differs from independent Safetensors header");
    check(descriptor.rank == independent.shape.size(),
          "catalog rank differs from independent Safetensors header");
    for (std::size_t index = 0; index < independent.shape.size(); ++index) {
        check(descriptor.shape[index] == independent.shape[index],
              "catalog shape differs from independent Safetensors header");
    }
    check(descriptor.file_offset == independent.file_offset &&
                  descriptor.bytes == independent.bytes,
          "catalog byte range differs from independent Safetensors header");
}

std::vector<std::uint8_t> copy_device_bytes(const void* source, std::size_t bytes) {
    std::vector<std::uint8_t> result(bytes);
    cuda_check(cudaMemcpy(result.data(), source, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy(device to independent host bytes)");
    return result;
}

void compare_real_slot_plane(const q4::expert_checkpoint_catalog& catalog,
                             std::uint32_t layer,
                             std::uint32_t expert,
                             q4::ExpertProjection projection,
                             q4::ExpertPlane plane,
                             const void* device_source,
                             std::size_t expected_bytes) {
    const q4::expert_tensor_descriptor* descriptor =
            catalog.find(layer, expert, projection, plane);
    check(descriptor != nullptr, "real expert descriptor is missing");
    independent_shard shard(descriptor->file);
    const independent_shard::tensor independent = shard.describe(descriptor->name);
    verify_descriptor(*descriptor, independent);
    check(independent.bytes == expected_bytes, "real expert plane byte count is unexpected");
    const std::vector<std::uint8_t> disk = shard.read(independent);
    const std::vector<std::uint8_t> gpu = copy_device_bytes(device_source, expected_bytes);
    check(disk == gpu, "GPU slot bytes differ from independent real-checkpoint pread");
}

float independent_scalar(const q4::expert_checkpoint_catalog& catalog,
                         std::uint32_t layer,
                         std::uint32_t expert,
                         q4::ExpertProjection projection,
                         q4::ExpertPlane plane) {
    const q4::expert_tensor_descriptor* descriptor =
            catalog.find(layer, expert, projection, plane);
    check(descriptor != nullptr, "real scalar descriptor is missing");
    independent_shard shard(descriptor->file);
    const independent_shard::tensor independent = shard.describe(descriptor->name);
    verify_descriptor(*descriptor, independent);
    const std::vector<std::uint8_t> bytes = shard.read(independent);
    check(bytes.size() == sizeof(float), "real scalar is not F32");
    float value = 0.0f;
    std::memcpy(&value, bytes.data(), sizeof(value));
    check(std::isfinite(value) && value > 0.0f, "real scalar is non-positive or non-finite");
    return value;
}

void test_symlink_root_rejected(const std::string& model_root) {
    std::array<char, 64> pattern{};
    const char prefix[] = "/tmp/axiom-qwen4exp-bridge-XXXXXX";
    std::memcpy(pattern.data(), prefix, sizeof(prefix));
    char* directory = ::mkdtemp(pattern.data());
    check(directory != nullptr, "mkdtemp failed");
    const std::string link = std::string(directory) + "/model";
    check(::symlink(model_root.c_str(), link.c_str()) == 0, "test symlink creation failed");
    std::shared_ptr<const q4::expert_checkpoint_catalog> rejected;
    std::string error;
    check(!q4::expert_checkpoint_catalog::open(
                  link, q4::pinned_expert_checkpoint_identity(), &rejected, &error),
          "symlink model root was not rejected");
    check(::unlink(link.c_str()) == 0, "test symlink cleanup failed");
    check(::rmdir(directory) == 0, "test directory cleanup failed");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    try {
        const std::string model_root = argv[1];
        std::string error;

        q4::expert_checkpoint_identity bad = q4::pinned_expert_checkpoint_identity();
        bad.revision = "not-the-pinned-revision";
        std::shared_ptr<const q4::expert_checkpoint_catalog> rejected;
        check(!q4::expert_checkpoint_catalog::open(model_root, bad, &rejected, &error),
              "mismatched revision was not rejected");
        bad = q4::pinned_expert_checkpoint_identity();
        bad.weight_manifest_sha256.assign(64u, '0');
        check(!q4::expert_checkpoint_catalog::open(model_root, bad, &rejected, &error),
              "mismatched payload fingerprint was not rejected");
        test_symlink_root_rejected(model_root);

        std::shared_ptr<const q4::expert_checkpoint_catalog> catalog;
        check(q4::expert_checkpoint_catalog::open(
                      model_root, q4::pinned_expert_checkpoint_identity(), &catalog, &error),
              error);
        check(catalog != nullptr && catalog->descriptor_count() == 294912u,
              "real routed-expert catalog cardinality mismatch");
        check(catalog->layout_sha256() == q4::kExpertLayoutSha256,
              "real routed-expert layout hash mismatch");

        constexpr std::uint32_t layer = 7u;
        constexpr std::uint32_t first_expert = 333u;
        constexpr std::uint32_t second_expert = 17u;
        const q4::expert_tensor_descriptor* real = catalog->find(
                layer, first_expert, q4::ExpertProjection::gate,
                q4::ExpertPlane::weight);
        check(real != nullptr &&
                      real->shard == "layer-00007-experts-0256-0383.safetensors" &&
                      real->file_offset > 0u && real->bytes == 819200u,
              "real descriptor does not preserve expected shard/offset/bytes");

        q4::expert_bridge_options options;
        options.device = 0;
        options.max_slots = 2u;
        options.ram_capacity_bytes = q4::kExpertBridgeBytesPerExpert * 2u + 4096u;
        options.prefetch_workers = 2u;
        std::unique_ptr<q4::expert_bridge> bridge;
        check(q4::expert_bridge::open(catalog, layer, options, &bridge, &error), error);
        check(bridge != nullptr && bridge->layer() == layer && bridge->max_slots() == 2u &&
                      bridge->gpu_capacity_bytes() == 5529632u,
              "bounded bridge capacity mismatch");

        const std::uint32_t duplicate[2]{first_expert, first_expert};
        q4::expert_bridge::prepared_generation invalid;
        check(!bridge->prepare(duplicate, 2u, nullptr, &invalid, &error),
              "duplicate expert selection was not rejected");

        {
            const std::uint32_t selected[2]{first_expert, second_expert};
            q4::expert_bridge::prepared_generation prepared;
            check(bridge->prepare(selected, 2u, nullptr, &prepared, &error), error);
            check(prepared.valid() && prepared.generation_id() != 0u &&
                          prepared.slot_count() == 2u &&
                          prepared.selected_experts()[0] == first_expert &&
                          prepared.selected_experts()[1] == second_expert &&
                          prepared.slot_indices_host()[0] == 0u &&
                          prepared.slot_indices_host()[1] == 1u,
                  "selected expert to resident slot mapping is incorrect");
            check(prepared.wait(&error), error);
            std::uint32_t device_slots[2]{};
            cuda_check(cudaMemcpy(device_slots, prepared.slot_indices_device(),
                                  sizeof(device_slots), cudaMemcpyDeviceToHost),
                       "cudaMemcpy(slot indices)");
            check(device_slots[0] == 0u && device_slots[1] == 1u,
                  "device slot index mapping is incorrect");

            const q4::moe_slot_planes& planes = prepared.planes();
            compare_real_slot_plane(*catalog, layer, first_expert,
                                    q4::ExpertProjection::gate,
                                    q4::ExpertPlane::weight,
                                    planes.gate_weight, 819200u);
            compare_real_slot_plane(*catalog, layer, first_expert,
                                    q4::ExpertProjection::gate,
                                    q4::ExpertPlane::weight_scale,
                                    planes.gate_block_scale, 102400u);
            compare_real_slot_plane(*catalog, layer, first_expert,
                                    q4::ExpertProjection::gate,
                                    q4::ExpertPlane::weight_scale_2,
                                    planes.gate_global_scale, sizeof(float));
            compare_real_slot_plane(*catalog, layer, first_expert,
                                    q4::ExpertProjection::up,
                                    q4::ExpertPlane::weight,
                                    planes.up_weight, 819200u);
            compare_real_slot_plane(*catalog, layer, first_expert,
                                    q4::ExpertProjection::up,
                                    q4::ExpertPlane::weight_scale,
                                    planes.up_block_scale, 102400u);
            compare_real_slot_plane(*catalog, layer, first_expert,
                                    q4::ExpertProjection::up,
                                    q4::ExpertPlane::weight_scale_2,
                                    planes.up_global_scale, sizeof(float));
            compare_real_slot_plane(*catalog, layer, first_expert,
                                    q4::ExpertProjection::down,
                                    q4::ExpertPlane::weight,
                                    planes.down_weight, 819200u);
            compare_real_slot_plane(*catalog, layer, first_expert,
                                    q4::ExpertProjection::down,
                                    q4::ExpertPlane::weight_scale,
                                    planes.down_block_scale, 102400u);
            compare_real_slot_plane(*catalog, layer, first_expert,
                                    q4::ExpertProjection::down,
                                    q4::ExpertPlane::weight_scale_2,
                                    planes.down_global_scale, sizeof(float));

            for (std::uint32_t projection = 0u; projection < 3u; ++projection) {
                const auto value = static_cast<q4::ExpertProjection>(projection);
                const float independent = independent_scalar(
                        *catalog, layer, first_expert, value,
                        q4::ExpertPlane::input_scale);
                const float bridged = prepared.input_scale(value, 0u);
                check(std::memcmp(&independent, &bridged, sizeof(float)) == 0,
                      "host input scale differs from independent real-checkpoint pread");
            }
            check(prepared.commit(&error) && prepared.committed(), error);
        }

        {
            const std::uint32_t selected[1]{first_expert};
            q4::expert_bridge::prepared_generation rolled_back;
            check(bridge->prepare(selected, 1u, nullptr, &rolled_back, &error), error);
            check(rolled_back.rollback(&error) && !rolled_back.valid(), error);
        }

        const q4::ExpertPagerMetrics metrics = bridge->pager_metrics();
        check(metrics.active_generations == 0u && metrics.active_leases == 0u &&
                      metrics.requests == 36u && metrics.nvme_reads >= 24u &&
                      metrics.nvme_bytes_read >= q4::kExpertBridgeBytesPerExpert * 2u,
              "pager transaction or bounded real-checkpoint I/O metrics mismatch");
        std::printf(
                "expert_bridge_test: OK descriptors=%zu layout=%s layer=%u "
                "experts=2 gpu_bytes=%llu requests=%llu nvme_reads=%llu nvme_bytes=%llu\n",
                catalog->descriptor_count(), catalog->layout_sha256().c_str(), layer,
                static_cast<unsigned long long>(bridge->gpu_capacity_bytes()),
                static_cast<unsigned long long>(metrics.requests),
                static_cast<unsigned long long>(metrics.nvme_reads),
                static_cast<unsigned long long>(metrics.nvme_bytes_read));
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "expert_bridge_test: %s\n", exception.what());
        return 1;
    }
}
