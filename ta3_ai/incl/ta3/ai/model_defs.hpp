#pragma once

#include <ta3/sim/utility/cuda_constant.hpp>
#include <ta3/sim/utility/tetris_defs.hpp>

#include <array>
#include <cstdint>

namespace ta3::ai {

using data_t = float;
using board_t = std::array<data_t, sim::BOARD_SIZE>;


/** single, double, triple, tetris */
inline constexpr std::size_t CLEAR_KINDS = sim::BLOCKS;
inline constexpr std::size_t CLEAR_LANE_BITS = 32 / CLEAR_KINDS;
inline constexpr std::uint32_t CLEAR_LANE_MASK = (std::uint32_t{1} << CLEAR_LANE_BITS) - 1;

/**
 * @brief the lane increment for a placement that cleared @c n lines, indexed @c [0, sim::BLOCKS]
 * @attention TA3_CUDA_CONSTANT: device code indexes this with a runtime value, and a host constexpr array
 *  there compiles to a device trap on any path where the index does not constant-fold.
 */
TA3_CUDA_CONSTANT std::array<std::uint32_t, sim::BLOCKS + 1> CLEAR_BUMP{
    std::uint32_t{0},
    std::uint32_t{1} << (0 * CLEAR_LANE_BITS),
    std::uint32_t{1} << (1 * CLEAR_LANE_BITS),
    std::uint32_t{1} << (2 * CLEAR_LANE_BITS),
    std::uint32_t{1} << (3 * CLEAR_LANE_BITS),
};

/**
 * @brief the clear histogram along one search path: lane @c k counts the @c (k+1)-line clears
 * @note one 32-bit word, so the frontier carries it in a register exactly like the old scalar clear score
 */
struct clear_hist_t {
    std::uint32_t bits = 0;

    /** @brief the number of @c (k+1)-line clears, @c k in @c [0, CLEAR_KINDS) */
    [[nodiscard]] constexpr std::uint32_t count(std::size_t k) const {
        return (bits >> (k * CLEAR_LANE_BITS)) & CLEAR_LANE_MASK;
    }

    /** @brief total lines cleared along the path */
    [[nodiscard]] constexpr std::uint32_t lines() const {
        std::uint32_t n = 0;
        for(std::size_t k = 0; k < CLEAR_KINDS; ++k)
            n += (static_cast<std::uint32_t>(k) + 1u) * count(k);
        return n;
    }

    /** @brief this histogram with a placement that cleared @p cleared lines folded in */
    [[nodiscard]] constexpr clear_hist_t added(std::uint32_t cleared) const {
        return {bits + CLEAR_BUMP[cleared]};
    }

    [[nodiscard]] constexpr bool operator==(clear_hist_t const&) const = default;
};

}
