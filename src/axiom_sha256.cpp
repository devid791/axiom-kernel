#include "axiom/sha256.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace axiom::crypto {
namespace {

constexpr std::uint32_t kRoundConstants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

constexpr std::uint32_t rotate_right(std::uint32_t value, unsigned bits) noexcept {
    return (value >> bits) | (value << (32u - bits));
}

std::uint32_t load_be32(const std::uint8_t *source) noexcept {
    return (static_cast<std::uint32_t>(source[0]) << 24u) |
           (static_cast<std::uint32_t>(source[1]) << 16u) |
           (static_cast<std::uint32_t>(source[2]) << 8u) |
           static_cast<std::uint32_t>(source[3]);
}

void store_be32(std::uint8_t *destination, std::uint32_t value) noexcept {
    destination[0] = static_cast<std::uint8_t>(value >> 24u);
    destination[1] = static_cast<std::uint8_t>(value >> 16u);
    destination[2] = static_cast<std::uint8_t>(value >> 8u);
    destination[3] = static_cast<std::uint8_t>(value);
}

void set_error(std::string *error, const std::string &message) {
    if (error) *error = message;
}

}  // namespace

sha256::sha256() noexcept
    : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
             0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

void sha256::transform(const std::uint8_t block[64]) noexcept {
    std::uint32_t words[64]{};
    for (unsigned index = 0; index < 16; ++index) {
        words[index] = load_be32(block + index * 4u);
    }
    for (unsigned index = 16; index < 64; ++index) {
        const std::uint32_t s0 = rotate_right(words[index - 15], 7u) ^
                rotate_right(words[index - 15], 18u) ^ (words[index - 15] >> 3u);
        const std::uint32_t s1 = rotate_right(words[index - 2], 17u) ^
                rotate_right(words[index - 2], 19u) ^ (words[index - 2] >> 10u);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (unsigned index = 0; index < 64; ++index) {
        const std::uint32_t big_s1 = rotate_right(e, 6u) ^ rotate_right(e, 11u) ^
                rotate_right(e, 25u);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t temp1 = h + big_s1 + choose + kRoundConstants[index] + words[index];
        const std::uint32_t big_s0 = rotate_right(a, 2u) ^ rotate_right(a, 13u) ^
                rotate_right(a, 22u);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = big_s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

void sha256::update(const void *data, std::size_t bytes) noexcept {
    if (finished_ || bytes == 0u) return;
    if (!data || total_bytes_ > std::numeric_limits<std::uint64_t>::max() - bytes) {
        finished_ = true;
        digest_.fill(0u);
        return;
    }
    const auto *source = static_cast<const std::uint8_t *>(data);
    total_bytes_ += static_cast<std::uint64_t>(bytes);
    if (buffered_ != 0u) {
        const std::size_t count = std::min(bytes, buffer_.size() - buffered_);
        std::memcpy(buffer_.data() + buffered_, source, count);
        buffered_ += count;
        source += count;
        bytes -= count;
        if (buffered_ == buffer_.size()) {
            transform(buffer_.data());
            buffered_ = 0u;
        }
    }
    while (bytes >= buffer_.size()) {
        transform(source);
        source += buffer_.size();
        bytes -= buffer_.size();
    }
    if (bytes != 0u) {
        std::memcpy(buffer_.data(), source, bytes);
        buffered_ = bytes;
    }
}

sha256_digest sha256::finish() noexcept {
    if (finished_) return digest_;
    const std::uint64_t bit_count = total_bytes_ * 8u;
    buffer_[buffered_++] = 0x80u;
    if (buffered_ > 56u) {
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_), buffer_.end(), 0u);
        transform(buffer_.data());
        buffered_ = 0u;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_), buffer_.begin() + 56, 0u);
    for (unsigned index = 0; index < 8; ++index) {
        buffer_[63u - index] = static_cast<std::uint8_t>(bit_count >> (index * 8u));
    }
    transform(buffer_.data());
    for (unsigned index = 0; index < state_.size(); ++index) {
        store_be32(digest_.data() + index * 4u, state_[index]);
    }
    finished_ = true;
    return digest_;
}

std::string sha256_hex(const sha256_digest &digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string output(digest.size() * 2u, '0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        output[index * 2u] = kHex[digest[index] >> 4u];
        output[index * 2u + 1u] = kHex[digest[index] & 0x0fu];
    }
    return output;
}

std::string sha256_bytes_hex(const void *data, std::size_t bytes) {
    sha256 context;
    context.update(data, bytes);
    return sha256_hex(context.finish());
}

std::string sha256_string_hex(const std::string &value) {
    return sha256_bytes_hex(value.data(), value.size());
}

bool sha256_file_hex(
        const std::string &path, std::string *hex, std::uint64_t *bytes,
        std::string *error) {
    if (!hex) {
        set_error(error, "missing output digest");
        return false;
    }
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        set_error(error, "open failed: " + std::string(std::strerror(errno)));
        return false;
    }
    struct stat status {};
    bool ok = fstat(fd, &status) == 0 && S_ISREG(status.st_mode) && status.st_size >= 0;
    if (!ok) set_error(error, "path is not a regular file");
    sha256 context;
    std::array<std::uint8_t, 4u * 1024u * 1024u> buffer{};
    std::uint64_t total = 0u;
    while (ok) {
        const ssize_t count = read(fd, buffer.data(), buffer.size());
        if (count < 0) {
            if (errno == EINTR) continue;
            set_error(error, "read failed: " + std::string(std::strerror(errno)));
            ok = false;
            break;
        }
        if (count == 0) break;
        context.update(buffer.data(), static_cast<std::size_t>(count));
        total += static_cast<std::uint64_t>(count);
    }
    struct stat after {};
    if (ok && (fstat(fd, &after) != 0 || after.st_dev != status.st_dev ||
               after.st_ino != status.st_ino || after.st_size != status.st_size ||
               after.st_mtim.tv_sec != status.st_mtim.tv_sec ||
               after.st_mtim.tv_nsec != status.st_mtim.tv_nsec ||
               after.st_ctim.tv_sec != status.st_ctim.tv_sec ||
               after.st_ctim.tv_nsec != status.st_ctim.tv_nsec)) {
        set_error(error, "file changed while hashing");
        ok = false;
    }
    if (close(fd) != 0 && ok) {
        set_error(error, "close failed: " + std::string(std::strerror(errno)));
        ok = false;
    }
    if (!ok || total != static_cast<std::uint64_t>(status.st_size)) {
        if (ok) set_error(error, "file changed while hashing");
        return false;
    }
    *hex = sha256_hex(context.finish());
    if (bytes) *bytes = total;
    return true;
}

}  // namespace axiom::crypto
