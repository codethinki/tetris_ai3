#pragma once
#include "ta3/ai/model.hpp"

#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include <cth/coro/executor.hpp>
#include <cth/coro/sync.hpp>
#include <cth/coro/tasks/executor_task.hpp>
#include <cth/coro/tasks/sync_task.hpp>
#include <cth/data/thread_pool.hpp>

#include <memory>
#include <span>


namespace ta3::ai {
class AiMultiTetris;
}

namespace ta3::trn {

using bounds_vec2 = std::pair<pagmo::vector_double, pagmo::vector_double>;
using pvecd = pagmo::vector_double;

/** one model per model scheduler thread, leased on first use */
using model_pool_t = cth::dt::thread_pool<ai::model_t>;

struct TetrisProblem2Config {
    cth::co::executor gameScheduler;
    cth::co::executor modelScheduler;
    std::shared_ptr<model_pool_t> modelPool;
    size_t simulationsPerEval;
    size_t maxMoves;
    uint64_t const* seed;
};

class TetrisProblem2 {
public:
    using Config = TetrisProblem2Config;
    // pagmo's is_udp requires default constructibility; the null state is never evaluated
    constexpr TetrisProblem2() = default;

    constexpr TetrisProblem2(Config);

    constexpr ~TetrisProblem2() = default;


    [[nodiscard]] bounds_vec2 get_bounds() const;

    [[nodiscard]] pvecd fitness(pvecd const& dv) const;

    [[nodiscard]] pvecd batch_fitness(pvecd const& dvs) const;

private:
    using clock_t = std::chrono::steady_clock;
    using time_p = std::chrono::time_point<clock_t>;

    /** evaluates @ref dv over @ref SIMULATIONS_PER_EVAL games, scoring on the model scheduler */
    [[nodiscard]] auto evalGames(std::span<double const> dv) const -> cth::co::executor_task<double>;

    /** wraps @ref evalGames into a blockable task spawned on the game scheduler */
    [[nodiscard]] auto spawnEval(std::span<double const> dv) const -> cth::co::sync_task<double>;

    /** mean score across the finished and the still alive games */
    [[nodiscard]] static double meanScore(ai::AiMultiTetris const& games);

    std::optional<Config> _config;

    [[nodiscard]] cth::co::executor gameScheduler() const { return _config->gameScheduler; }
    [[nodiscard]] cth::co::executor modelScheduler() const { return _config->modelScheduler; }

public:
    [[nodiscard]] pagmo::thread_safety get_thread_safety() const { return pagmo::thread_safety::constant; };
    [[nodiscard]] bool has_fitness() const { return true; }
    [[nodiscard]] bool has_batch_fitness() const { return true; }

    TetrisProblem2(TetrisProblem2 const& other) = default;
    TetrisProblem2(TetrisProblem2&& other) noexcept = default;
    TetrisProblem2& operator=(TetrisProblem2 const& other) = default;
    TetrisProblem2& operator=(TetrisProblem2&& other) noexcept = default;

};
}

namespace ta3::trn {
constexpr TetrisProblem2::TetrisProblem2(Config config) : _config{std::move(config)} {}
}
