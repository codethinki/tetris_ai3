#include "ta3/trainer/tetris_problem2.hpp"

#include "ta3/ai/ai_multi_tetris.hpp"

#include <algorithm>
#include <span>
#include <vector>

namespace ta3::trn {

bounds_vec2 TetrisProblem2::get_bounds() const {
    static auto const MODEL_PARAMS = ai::model_t::params();

    return {
        pvecd(MODEL_PARAMS, ai::m::BOUNDS[0]),
        pvecd(MODEL_PARAMS, ai::m::BOUNDS[1])
    };
}

pvecd TetrisProblem2::fitness(pvecd const& dv) const {
    return {cth::co::sync(_gameScheduler->spawn(evalGames(dv)))};
}

pvecd TetrisProblem2::batch_fitness(pvecd const& dvs) const {
    auto const params = ai::model_t::params();
    auto const count = dvs.size() / params;

    std::vector<cth::co::sync_task<double>> evals{};
    evals.reserve(count);
    for(auto i = 0uz; i < count; ++i)
        evals.push_back(spawnEval(std::span{dvs}.subspan(i * params, params)));

    // start them all, then join -- awaiting an already finished task is a noop
    for(auto& eval : evals)
        eval.start();

    pvecd fitnesses(count);
    for(auto i = 0uz; i < count; ++i)
        fitnesses[i] = evals[i].await();

    return fitnesses;
}

auto TetrisProblem2::evalGames(std::span<double const> dv) const -> cth::co::executor_task<double> {
    ai::AiMultiTetris games{SIMULATIONS_PER_EVAL, _seed};

    for(auto pieces = 0uz; !games.empty() && pieces < MAX_PIECES; ++pieces) {
        co_await _modelScheduler->schedule();

        auto& model = _modelPool->acquire();
        model.loadWeights(dv);
        auto const scores = model.batchForward(games.inputs());

        co_await _gameScheduler->schedule();
        games.next(scores);
    }

    co_return -meanScore(games);
}

auto TetrisProblem2::spawnEval(std::span<double const> dv) const -> cth::co::sync_task<double> {
    co_return co_await _gameScheduler->spawn(evalGames(dv));
}

double TetrisProblem2::meanScore(ai::AiMultiTetris const& games) {
    auto const dead = games.deadStats();
    auto const alive = games.gamesStats();
    auto const count = static_cast<double>(dead.size() + alive.size());

    auto const score = [count](double sum, ai::tetris_stats_t const& stats) { return sum + stats.score() / count; };

    return std::ranges::fold_left(alive, std::ranges::fold_left(dead, 0., score), score);
}
}
