#include "ta3/trainer/tetris_problem2.hpp"

#include "ta3/gpu/model_eval.hpp"
#include "ta3/sim/utility/xoshiro256ss.hpp"

#include <algorithm>
#include <ranges>
#include <span>
#include <vector>
#include <print>

namespace ta3::trn {

bounds_vec2 TetrisProblem2::get_bounds() const {
    static auto const MODEL_PARAMS = ai::model_t::params();

    return {
        pvecd(MODEL_PARAMS, ai::m::BOUNDS[0]),
        pvecd(MODEL_PARAMS, ai::m::BOUNDS[1])
    };
}

// single-model fitness is just a one-model batch -- both paths funnel through evalBatch / one launch.
pvecd TetrisProblem2::fitness(pvecd const& dv) const {
    cth::log::msg<cth::except::WARNING>("Not using batch fitness");
    return batch_fitness(dv);
}

pvecd TetrisProblem2::batch_fitness(pvecd const& dvs) const {
    auto const params = ai::model_t::params();
    auto const count = dvs.size() / params;

    // pagmo hands us the whole population's decision vectors concatenated, which is already the
    // model-major layout the batched evaluator wants -- just narrow double -> weights_t (float).
    std::vector<gpu::weights_t> weights;
    weights.reserve(dvs.size());
    for(double const x : dvs)
        weights.push_back(static_cast<gpu::weights_t>(x));

    // one submission -> one kernel launch of count*simulationsPerEval blocks.
    return cth::co::sync(spawnBatch(std::move(weights), count));
}

auto TetrisProblem2::evalBatch(
    std::vector<gpu::weights_t> weights,
    std::size_t num_models
) const
    -> cth::co::executor_task<std::vector<double>> {
    auto const games = _config->simulationsPerEval;
    std::vector<double> means(num_models, 0.0);

    try {
        auto& evaluator = _config->evalPool->acquire();

        // model-major: model m on game g lives at results[m*games + g].
        auto const results = gpu::simulate(evaluator, weights, num_models, games, *_config->seed, _config->maxMoves);


        for(std::size_t m = 0; m < num_models; ++m) {
            auto const modelGames = std::span{results}.subspan(m * games, games);
            auto const sum = std::ranges::fold_left(modelGames, 0.0, [](double s, float v) { return s + v; });
            // score_v4 is higher-is-better; pagmo minimises, so hand it the negation.
            means[m] = -(sum / static_cast<double>(games));
        }
    }
    catch(std::exception const& e) {
        cth::log::msg<cth::except::ERR>("{}", e.what());
        abort();
    }

    co_return means;
}

auto TetrisProblem2::spawnBatch(
    std::vector<gpu::weights_t> weights,
    std::size_t num_models
) const
    -> cth::co::sync_task<std::vector<double>> {
    co_return co_await _config->modelScheduler.spawn(evalBatch(std::move(weights), num_models));
}


}
