#include "ta3/ai/fitness.hpp"

#include <ta3/sim/pieces/piece_defs.hpp>

#include <algorithm>
#include <cmath>

namespace ta3::ai {

namespace {
    constexpr auto adjustedHeight(double height) {
        static constexpr double AVG_HEIGHT_THRESHOLD = 4.0;

        auto const tmp = std::max(0.0, height - AVG_HEIGHT_THRESHOLD);
        return (1 / 9.) * tmp * tmp;
    }


    auto adjustedHoles(double holes) { return std::pow(holes, 1. / 3.); }

    constexpr auto adjustedRoughness(double roughness) {
        static constexpr double AVG_ROUGHNESS_THRESHOLD = 1.0;
        return std::max(0.0, roughness - AVG_ROUGHNESS_THRESHOLD);
    }
}

double calculateFitness(sim::Stats const& stats, bool dead) {
    auto const numPieces = static_cast<double>(stats.pieces());

    if(stats.pieces() == 0) { return -10000.0; }

    static constexpr double W_PIECE_SURVIVAL = 0.5;

    static constexpr std::array CLEAR_REWARDS_ADJUSTED = {
        0.0,
        2.0,
        5.0,
        12.0,
        60.0
    };
    static constexpr double DEATH_PENALTY = 1000;

    static constexpr double W_HOLES = 15;
    static constexpr double W_HEIGHT = 0.5;
    static constexpr double W_ROUGHNESS = .5;


    double fitness = 0.0;

    fitness += numPieces * W_PIECE_SURVIVAL;

    auto const& clearedLinesCounts = stats.linesCleared();
    for(size_t i = 1; i <= sim::BLOCKS; ++i)
        fitness += static_cast<double>(clearedLinesCounts[i]) * CLEAR_REWARDS_ADJUSTED[i];

    auto const avgHoles = stats.holesSum() / numPieces;
    auto const avgHeight = stats.heightSum() / numPieces;
    auto const avgRoughness = stats.roughnessSum() / numPieces;


    fitness -= adjustedHoles(avgHoles) * W_HOLES;
    fitness -= adjustedHeight(avgHeight) * W_HEIGHT;
    fitness -= adjustedRoughness(avgRoughness) * W_ROUGHNESS;

    if(dead) fitness -= DEATH_PENALTY;

    return fitness;
}

}
