#pragma once
#include <glm/glm.hpp>

#include <cstddef>

namespace ta3::sim {
using vec2 = glm::vec<2, int>;
using fvec3 = glm::vec<3, float>;
using szvec2 = glm::vec<2, size_t>;

template<int S, class T>
constexpr T norm1(glm::vec<S, T> const& vec) {
    T result{};
    for(size_t i = 0; i < S; i++) result += vec[i];

    return result;
}
}