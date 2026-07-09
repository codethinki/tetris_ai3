#pragma once
#include <ta3/ai/model.hpp>
#include <ta3/gpu/model_eval.hpp>



#include <pagmo/threading.hpp>
#include <pagmo/types.hpp>

#include <cth/coro/executor.hpp>
#include <cth/coro/sync.hpp>
#include <cth/coro/tasks/executor_task.hpp>
#include <cth/coro/tasks/sync_task.hpp>
#include <cth/data/thread_pool.hpp>

#include <memory>
#include <span>
#include <vector>


namespace ta3::trn {

using bounds_vec2 = std::pair<pagmo::vector_double, pagmo::vector_double>;
using pvecd = pagmo::vector_double;


using eval_pool_t = cth::dt::thread_pool<gpu::ModelEvaluator>;

struct TetrisProblem2Config {
    cth::co::executor modelScheduler;
    std::shared_ptr<eval_pool_t> evalPool;
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
    /** evaluates @ref numModels models (weights, model-major NUM_PARAMS each) over @c simulationsPerEval
     *  shared games in a single kernel launch, returning one mean score per model.
     *  takes @ref weights BY VALUE: as a coroutine it suspends before use, so the data must live in the
     *  frame -- a span/reference to a caller temporary would dangle after the first suspension. */
    [[nodiscard]] auto evalBatch(std::vector<gpu::weights_t> weights, std::size_t num_models) const
        -> cth::co::executor_task<std::vector<double>>;

    /** wraps @ref evalBatch into a blockable task spawned on the model scheduler */
    [[nodiscard]] auto spawnBatch(std::vector<gpu::weights_t> weights, std::size_t num_models) const
        -> cth::co::sync_task<std::vector<double>>;

    std::optional<Config> _config;

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
