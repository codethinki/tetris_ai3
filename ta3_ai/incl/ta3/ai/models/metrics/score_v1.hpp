#pragma once
#include "tetris_stats.hpp"

#include <ta3/sim/utility/cuda_constant.hpp>


namespace ta3::ai {
inline constexpr double SURVIVAL_WEIGHT = 0.02;
inline constexpr double AVG_HEIGHT_WEIGHT = 1.0;
inline constexpr double AVG_HEIGHT_BUDGET = 12.0;
inline constexpr double AVG_HOLE_DEPTHS_WEIGHT = 1.5;
// TA3_CUDA_CONSTANT: indexed with a runtime value from device code (the kernel scores games with
// score_v4); a plain host constexpr array there silently compiles to a device trap.
TA3_CUDA_CONSTANT double PAYOUTS[] = {0.0, 0.2, 0.8, 2.0, 10.0};

/**
 * final game score
 * @param stats the aggregate stat block; a score metric reads it via stats.get(...)
 */
constexpr auto score_v1 = [][[nodiscard]](auto const& stats) {
    double clearBonus = 0;
    auto const histogram = stats.get(metric::total_clears);
    for(size_t k = 1; k < histogram.size(); ++k)
        clearBonus += static_cast<double>(PAYOUTS[k] * histogram[k]); // tetris (16) >> four singles (4)
    auto const survivalBonus = SURVIVAL_WEIGHT * static_cast<double>(stats.piecesPlaced());


    auto const avgMaxHeight = stats.get(metric::avg_max_height);

    auto const avgMaxHeightPenalty = std::max(avgMaxHeight - AVG_HEIGHT_BUDGET, 0.) * AVG_HEIGHT_WEIGHT;
    auto const avgHolesPenalty = stats.get(metric::avg_hole_depths) * AVG_HOLE_DEPTHS_WEIGHT;

    return clearBonus + survivalBonus - avgMaxHeightPenalty - avgHolesPenalty;
};

}
