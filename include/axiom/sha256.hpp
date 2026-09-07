#ifndef AXIOM_SHA256_HPP
#define AXIOM_SHA256_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace axiom::crypto {

using sha256_digest = std::array<std::uint8_t, 32>;

class sha256 final {
public:
    sha256() noexcept;

    void update(const void *data, std::size_t bytes) noexcept;
    sha256_digest finish() noexcept;

private:
    void transform(const std::uint8_t block[64]) noexcept;

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> buffer_{};
    std::uint64_t total_bytes_ = 0;
    std::size_t buffered_ = 0;
    bool finished_ = false;
    sha256_digest digest_{};
};

std::string sha256_hex(const sha256_digest &digest);
std::string sha256_bytes_hex(const void *data, std::size_t bytes);
std::string sha256_string_hex(const std::string &value);

/* Hashes one regular file without following a symbolic link.  `error` is
 * optional and receives a stable diagnostic suitable for a fail-closed
 * startup or release gate. */
bool sha256_file_hex(
        const std::string &path, std::string *hex,
        std::uint64_t *bytes = nullptr, std::string *error = nullptr);

}  // namespace axiom::crypto

#endif
