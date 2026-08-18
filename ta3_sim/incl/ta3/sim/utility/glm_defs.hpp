#pragma once
#include "ta3/sim/ivec2.hpp"

#include <glm/glm.hpp>

#include <cstddef>

namespace ta3::sim {
// vec2 is now a device-clean POD (see ivec2.hpp); glm stays for the host-only float/size types
using fvec3 = glm::vec<3, float>;
using szvec2 = glm::vec<2, size_t>;

template<int S, class T>
constexpr T norm1(glm::vec<S, T> const& vec) {
    T result{};
    for(size_t i = 0; i < S; i++) result += vec[i];

    return result;
}
}