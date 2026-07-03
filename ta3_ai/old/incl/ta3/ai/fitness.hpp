#pragma once
#include <ta3/sim/tetris_stats.hpp>

namespace ta3::ai {
[[nodiscard]] double calculateFitness(sim::Stats const& stats, bool dead);
}
