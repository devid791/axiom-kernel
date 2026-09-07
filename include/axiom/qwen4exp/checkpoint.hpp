#ifndef AXIOM_QWEN4EXP_CHECKPOINT_HPP
#define AXIOM_QWEN4EXP_CHECKPOINT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

enum class tensor_dtype : std::uint8_t {
    unknown = 0,
    bf16,
    f32,
    f8_e4m3,
    u8,
    i64,
};

struct tensor_span {
    std::string name;
    std::string shard;
    tensor_dtype dtype = tensor_dtype::unknown;
    std::uint32_t rank = 0;
    std::array<std::uint64_t, 5> shape{};
    std::uint64_t file_offset = 0;
    std::uint64_t bytes = 0;
};

/* Immutable, range-readable catalog for the pinned qwen4_exp Safetensors
 * checkpoint.  read_range() always issues a bounded pread directly from the
 * source shard; it never materializes or caches an entire tensor. */
class checkpoint_catalog final {
public:
    checkpoint_catalog();
    ~checkpoint_catalog();
    checkpoint_catalog(checkpoint_catalog &&) noexcept;
    checkpoint_catalog &operator=(checkpoint_catalog &&) noexcept;
    checkpoint_catalog(const checkpoint_catalog &) = delete;
    checkpoint_catalog &operator=(const checkpoint_catalog &) = delete;

    static bool open(const std::string &model_root,
                     std::unique_ptr<checkpoint_catalog> *out,
                     std::string *error) noexcept;

    const tensor_span *find(const std::string &name) const noexcept;
    const tensor_span *at(std::size_t index) const noexcept;
    std::size_t tensor_count() const noexcept;
    const std::string &model_root() const noexcept;

    bool read_range(const std::string &name,
                    std::uint64_t tensor_byte_offset,
                    void *destination,
                    std::size_t bytes,
                    std::string *error) const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
