#pragma once
#include "metrics.hpp"
#include "tetris_stats.hpp"

namespace ta3::ai {
/** the v4 stat block: the stateful metrics the models + score read */
using stats_v5 = metric::tetris_stats<

    //score
    metric::total_clears,
    metric::avg_max_height
>;


}
