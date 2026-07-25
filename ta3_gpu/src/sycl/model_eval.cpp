#include "ta3/gpu/model_eval.hpp"

#include <ta3/gpu/detail/value_model.hpp>

#include "detail/eval_kernel.hpp"

#include <ta3/sim/tetris_engine.hpp>

#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

/**
 * @file model_eval.cpp
 * @brief host driver for the SYCL beam-search evaluator: pooled USM device buffers + the single kernel
 *  submission (kernel mechanics live in detail/eval_kernel.hpp).
 */
namespace ta3::gpu {

namespace sim = ta3::sim;

/**
 * one evaluation context: a SYCL in-order queue + grow-only USM device buffers, reused across every eval
 * of this instance. instances are pooled by the trainer -- one per model-scheduler thread -- so concurrent
 * batches submit on distinct queues. the destructor releases everything back to the runtime.
 */
struct ModelEvaluator::Impl {
    sycl::queue queue;
    sim::TetrisEngine* games = nullptr;
    float* weights = nullptr;
    float* fitness = nullptr;
    std::size_t gamesCap = 0, weightsCap = 0, fitnessCap = 0;

    /** prefer a GPU (@c gpu_selector_v throws if none matches); fall back to @c default_selector_v otherwise. */
    static sycl::queue make_queue() {
        sycl::property_list const props{sycl::property::queue::in_order()};
        try {
            return sycl::queue{sycl::gpu_selector_v, props};
        } catch(sycl::exception const&) {
            return sycl::queue{sycl::default_selector_v, props};
        }
    }

    Impl() : queue{make_queue()} {}

    void ensure(std::size_t blocks, std::size_t weight_count) {
        auto const grow = [this](auto*& ptr, std::size_t& cap, std::size_t need, char const* what) {
            if(need <= cap)
                return;
            if(ptr != nullptr)
                sycl::free(ptr, queue);
            using T = std::remove_pointer_t<std::remove_reference_t<decltype(ptr)>>;
            ptr = sycl::malloc_device<T>(need, queue);
            if(ptr == nullptr)
                throw std::runtime_error(std::string{"sycl: "} + what);
            cap = need;
        };
        grow(games, gamesCap, blocks, "malloc games");
        grow(weights, weightsCap, weight_count, "malloc weights");
        grow(fitness, fitnessCap, blocks, "malloc fitness");
    }

    ~Impl() {
        // best effort: a dying evaluator must not throw; sycl::free is safe/no-op on a null pointer.
        if(games != nullptr)
            sycl::free(games, queue);
        if(weights != nullptr)
            sycl::free(weights, queue);
        if(fitness != nullptr)
            sycl::free(fitness, queue);
    }

    Impl(Impl const&) = delete;
    Impl& operator=(Impl const&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;
};

ModelEvaluator::ModelEvaluator() : _impl{std::make_unique<Impl>()} {}
ModelEvaluator::~ModelEvaluator() = default;
ModelEvaluator::ModelEvaluator(ModelEvaluator&&) noexcept = default;
ModelEvaluator& ModelEvaluator::operator=(ModelEvaluator&&) noexcept = default;

std::vector<float> ModelEvaluator::eval(
    std::span<float const> weights,
    std::uint32_t num_models,
    std::span<std::uint64_t const> seeds,
    std::uint32_t max_moves
) {
    std::uint32_t const games = static_cast<std::uint32_t>(seeds.size()); // games per model
    std::uint32_t const blocks = num_models * games; // one work-group per (model, game)
    std::vector<float> out(blocks, 0.0f);
    if(blocks == 0)
        return out;

    // weights must be model-major with a full NUM_PARAMS slice per model.
    if(weights.size() != static_cast<std::size_t>(num_models) * ai::model_t::NUM_PARAMS)
        throw std::runtime_error("sycl: eval: weights.size() must equal num_models * NUM_PARAMS");

    // one engine per work-group; work-group g plays seeds[g % games], so every model sees the same set of
    // games.
    std::vector<sim::TetrisEngine> hostGames;
    hostGames.reserve(blocks);
    for(std::uint32_t g = 0; g < blocks; ++g)
        hostGames.emplace_back(seeds[g % games]);

    Impl& ctx = *_impl;
    ctx.ensure(blocks, weights.size());

    try {
        // in-order queue: this memcpy, the kernel submission, and the closing memcpy execute in submission
        // order without explicit depends_on (SYCL 2020 in-order queue guarantee).
        ctx.queue.memcpy(ctx.games, hostGames.data(), blocks * sizeof(sim::TetrisEngine));
        ctx.queue.memcpy(ctx.weights, weights.data(), weights.size() * sizeof(float));

        ctx.queue.submit([&](sycl::handler& cgh) {
            // one work-group-scope WorkGroupState allocation (local_accessor<WorkGroupState, 0>).
            sycl::local_accessor<WorkGroupState, 0> shared{cgh};

            sim::TetrisEngine* const gamesPtr = ctx.games;
            float const* const weightsPtr = ctx.weights;
            float* const fitnessPtr = ctx.fitness;
            std::uint32_t const numBlocks = blocks;
            std::uint32_t const numGames = games;
            std::uint32_t const maxMoves = max_moves;

            cgh.parallel_for(
                sycl::nd_range<1>{
                    sycl::range<1>{static_cast<std::size_t>(blocks) * BLOCK},
                    sycl::range<1>{BLOCK}
                },
                [=](sycl::nd_item<1> it) {
                    WorkGroupState& wgroup = shared;
                    eval_kernel(it, wgroup, gamesPtr, numBlocks, numGames, weightsPtr, maxMoves, fitnessPtr);
                }
            );
        });

        ctx.queue.memcpy(out.data(), ctx.fitness, blocks * sizeof(float));
        // the single host synchronisation of the whole evaluation.
        ctx.queue.wait();
    } catch(sycl::exception const& e) {
        throw std::runtime_error(std::string{"sycl: kernel run: "} + e.what());
    }
    return out;
}

} // namespace ta3::gpu
