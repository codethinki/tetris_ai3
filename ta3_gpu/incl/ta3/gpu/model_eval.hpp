#pragma once
#include "ta3/sim/utility/xoshiro256ss.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

/**
 * @file model_eval.hpp
 * @brief host-clean entry point to the GPU beam-search evaluator (backend picked at build time via
 *  TA3_GPU_BACKEND: CUDA or SYCL).
 * @details pimpl boundary: no GPU-runtime types leak here, so MSVC TUs call this while the backend's
 *  implementation is compiled by nvcc or AdaptiveCpp's clang++. both @ref ModelEvaluator::eval and
 *  @ref ModelEvaluator::eval_host run the same beamed search (see @c ta3_ai/.../search/beam.hpp) --
 *  eval_host is the CPU parity reference and must match eval bit-for-bit.
 */
namespace ta3::gpu {
using weights_t = float;

/**
 * owns one evaluation context: a GPU stream/queue + grow-only persistent device buffers (no malloc/free
 * churn across the thousands of eval calls of a training run). NOT thread safe -- intended to be pooled,
 * one instance per model-scheduler thread (@c cth::dt::thread_pool, leased on first use), so concurrent
 * threads submit on distinct streams/queues. destruction frees the device resources.
 */
class ModelEvaluator {
public:
    ModelEvaluator();
    ~ModelEvaluator();

    /**
     * batched search evaluator: @ref num_models models, each played on the SAME @ref seeds games, in a
     * single launch of @c num_models*seeds.size() blocks/work-groups (one per (model, game)) on this
     * evaluator's stream/queue.
     *
     * @param weights   all models' parameters, model-major and contiguous: model @c m owns
     *                  @c weights[m*P .. (m+1)*P) where @c P = @c model_t::NUM_PARAMS
     *                  (so @c weights.size() == num_models * P).
     * @param num_models number of models packed into @ref weights.
     * @param seeds     one seed per game, shared across every model (common random numbers).
     * @param max_moves move cap per game.
     * @return per-(model, game) fitness (@c ai::stats_v4 @c score(), higher is better), model-major:
     *         index @c m*seeds.size() + g holds model @c m on game @c g. Size = num_models*seeds.size().
     */
    [[nodiscard]] std::vector<float> eval(
        std::span<weights_t const> weights,
        std::uint32_t num_models,
        std::span<std::uint64_t const> seeds,
        std::uint32_t max_moves
    );

    /** single-model convenience: the @c num_models==1 special case of the batched @ref eval. */
    [[nodiscard]] std::vector<float> eval(
        std::span<weights_t const> weights,
        std::span<std::uint64_t const> seeds,
        std::uint32_t max_moves
    ) {
        return eval(weights, 1u, seeds, max_moves);
    }

    /** identical computation on the CPU (shared search core) -- the parity reference for @ref eval. */
    [[nodiscard]] static std::vector<float> eval_host(
        std::span<weights_t const> weights,
        std::span<std::uint64_t const> seeds,
        std::uint32_t max_moves
    );

    ModelEvaluator(ModelEvaluator&&) noexcept;
    ModelEvaluator& operator=(ModelEvaluator&&) noexcept;
    ModelEvaluator(ModelEvaluator const&) = delete;
    ModelEvaluator& operator=(ModelEvaluator const&) = delete;

private:
    struct Impl; // backend-specific stream/queue + device buffers; defined in the linked backend's src
    std::unique_ptr<Impl> _impl;
};

/**
 * batched simulate: evaluate @ref num_models models over @ref games shared games (seeded from @ref seed)
 * on @ref evaluator. returns per-(model, game) fitness, model-major (model @c m on game @c g at
 * @c m*games + g).
 */
inline std::vector<float> simulate(
    ModelEvaluator& evaluator,
    std::span<weights_t const> model_weights,
    size_t num_models,
    size_t games,
    uint64_t seed,
    size_t max_moves
) {
    ta3::sim::xoshiro256ss seedGen{seed};
    std::vector<uint64_t> seeds(games);
    for(auto& s : seeds)
        s = seedGen();

    return evaluator.eval(
        model_weights,
        static_cast<std::uint32_t>(num_models),
        seeds,
        static_cast<std::uint32_t>(max_moves)
    );
}

/** single-model simulate: the @c num_models==1 special case. */
inline std::vector<float> simulate(
    ModelEvaluator& evaluator,
    std::span<weights_t const> model_weights,
    size_t games,
    uint64_t seed,
    size_t max_moves
) {
    return simulate(evaluator, model_weights, 1, games, seed, max_moves);
}

} // namespace ta3::gpu
