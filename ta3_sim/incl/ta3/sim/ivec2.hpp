#pragma once
#include <compare>

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

}
