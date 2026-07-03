#pragma once
#include "aggregate.hpp"
#include "metrics.hpp"
#include "score_v1.hpp"

namespace ta3::ai {
/** the v4 stat block: the stateful metrics the models + score read */
using stats_v4 = metric::stats<
    score_v4,
    metric::total_lines_cleared,
    metric::new_holes,
    //score
    metric::total_clears,
    metric::avg_max_height,
    metric::avg_hole_depths
>;


}
