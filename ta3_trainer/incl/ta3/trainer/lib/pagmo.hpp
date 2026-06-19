#pragma once
#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/topologies/ring.hpp>

namespace pagmo {
using bounds_vec2 = std::pair<vector_double, vector_double>;
}

namespace ta3::trainer {
using pvecd = pagmo::vector_double;
}
