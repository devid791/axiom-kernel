#ifndef AXIOM_REASONING_PROFILES_H
#define AXIOM_REASONING_PROFILES_H

/*
 * Model-independent reasoning policy shared by the Axiom capability plane
 * and every native model adapter.  An adapter may clamp the budget to its own
 * supported maximum, but it must keep these public profile names and the
 * meaning of Ultra-fast/Off stable.
 *
 * The budget is an exact upper bound for hidden thinking tokens.  It is not a
 * promise that the model will spend the whole budget: the measured
 * `thinking_tokens` value in the request telemetry is the actual consumption.
 */

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace axiom {
namespace reasoning {

struct profile {
    const char *id;
    const char *label;
    uint32_t thinking_budget_tokens;
    bool thinking;
    float temperature;
    float top_p;
    uint32_t top_k;
};

inline constexpr uint32_t kMaxThinkingBudgetTokens = 32768u;

/* Canonical public order: fastest first, then progressively deeper reasoning. */
inline constexpr profile kProfiles[] = {
    {"ultra-fast", "Ultra-fast / Off", 0u, false, 0.0f, 1.0f, 1u},
    {"minimal", "Minimal", 1024u, true, 0.6f, 0.95f, 20u},
    {"low", "Low", 2048u, true, 0.6f, 0.95f, 20u},
    {"medium", "Medium", 4096u, true, 0.6f, 0.95f, 20u},
    {"high", "High", 8192u, true, 0.6f, 0.95f, 20u},
    {"xhigh", "Extra-high", 16384u, true, 0.6f, 0.95f, 20u},
    {"max", "Maximum", 32768u, true, 0.6f, 0.95f, 20u},
};

inline constexpr std::size_t kProfileCount = sizeof(kProfiles) / sizeof(kProfiles[0]);
static_assert(kProfileCount == 7u,
              "Axiom exposes exactly seven public reasoning profiles");

inline std::string lower(std::string value) {
    for (char &character : value) {
        character = static_cast<char>(
                std::tolower(static_cast<unsigned char>(character)));
    }
    return value;
}

inline std::string canonical_name(std::string value) {
    return lower(std::move(value));
}

inline const profile *find(const std::string &requested) {
    const std::string id = canonical_name(requested);
    for (std::size_t index = 0u; index < kProfileCount; ++index) {
        if (id == kProfiles[index].id) return &kProfiles[index];
    }
    return nullptr;
}

}  // namespace reasoning
}  // namespace axiom

#endif  // AXIOM_REASONING_PROFILES_H
