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



class TetrisProblem2 {
    static constexpr size_t GAME_BATCH_SIZE = 1'000'000;
    static constexpr size_t SIMULATIONS_PER_EVAL = 200;
    static constexpr size_t MAX_PIECES = 1000;

public:
    // pagmo's is_udp requires default constructibility; the null state is never evaluated
    constexpr TetrisProblem2() = default;

    constexpr TetrisProblem2(
        cth::co::executor game_scheduler,
        cth::co::executor model_scheduler,
        std::shared_ptr<model_pool_t> model_pool,
        uint64_t seed
    );

    ~TetrisProblem2() = default;

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

    std::optional<cth::co::executor> _gameScheduler;
    std::optional<cth::co::executor> _modelScheduler;
    std::shared_ptr<model_pool_t> _modelPool;
    uint64_t _seed{};

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
constexpr TetrisProblem2::TetrisProblem2(
    cth::co::executor game_scheduler,
    cth::co::executor model_scheduler,
    std::shared_ptr<model_pool_t> model_pool,
    uint64_t seed
) : _gameScheduler{game_scheduler},
    _modelScheduler{model_scheduler},
    _modelPool{std::move(model_pool)},
    _seed{seed} {}
}
