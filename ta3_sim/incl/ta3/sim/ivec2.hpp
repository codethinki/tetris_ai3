#pragma once
#include <compare>
#include <cstddef>

namespace ta3::sim {

/**
 * @brief plain 2D integer vector -- device-clean replacement for glm::vec<2,int>
 * @note trivial POD (no glm, no SIMD intrinsics) so it compiles on SYCL/CUDA device code;
 *  aggregate-initializable as @c ivec2{x, y}
 */
struct ivec2 {
    int x{};
    int y{};

    friend constexpr ivec2 operator+(ivec2 a, ivec2 b) { return {a.x + b.x, a.y + b.y}; }
    friend constexpr ivec2 operator-(ivec2 a, ivec2 b) { return {a.x - b.x, a.y - b.y}; }

    constexpr ivec2& operator+=(ivec2 o) { x += o.x; y += o.y; return *this; }
    constexpr ivec2& operator-=(ivec2 o) { x -= o.x; y -= o.y; return *this; }

    friend constexpr bool operator==(ivec2, ivec2) = default;
    friend constexpr auto operator<=>(ivec2, ivec2) = default;
};

using vec2 = ivec2;

/**
 * @brief plain 2D double vector -- device-clean replacement for glm::dvec2
 * @note used for model weight bounds; supports @c operator[] for the pagmo bound builders
 */
struct dvec2 {
    double x{};
    double y{};

    constexpr double operator[](size_t i) const { return i == 0 ? x : y; }
    constexpr double& operator[](size_t i) { return i == 0 ? x : y; }

    friend constexpr bool operator==(dvec2, dvec2) = default;
};

}
