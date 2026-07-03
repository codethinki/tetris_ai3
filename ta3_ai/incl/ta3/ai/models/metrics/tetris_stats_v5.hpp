#pragma once
#include "aggregate.hpp"
#include "metrics.hpp"

namespace ta3::ai {
/** the v4 stat block: the stateful metrics the models + score read */
using stats_v5 = metric::aggregate<

    //score
    metric::total_clears,
    metric::avg_max_height
>;


}
