#include "ta3/gpu/model_eval.hpp"

#include <ta3/gpu/detail/value_model.hpp>

#include "detail/eval_kernel.cuh"

#include <ta3/sim/tetris_engine.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file model_eval.cu
 * @brief host driver for the GPU beam-search evaluator: pooled device buffers + the single kernel launch.
 *
 * the kernel itself (@c eval_kernel) and its per-move stages live in detail/eval_kernel.cuh -- see that
 * file for the beam-search mechanics (root/middle/leaf staging, block-wide selection, the per-level
 * fallbacks). this file only owns what's specific to hosting it: growing the persistent device buffers
 * (@ref ModelEvaluator::Impl), staging one batch's games onto the device, launching the kernel on a pooled
 * stream, and copying the fitness vector back -- a single host synchronisation per @ref
 * ModelEvaluator::eval call.
 */
namespace ta3::gpu {

namespace sim = ta3::sim;

namespace {

    void cuda_check(cudaError_t e, char const* what) {
        if(e != cudaSuccess)
            throw std::runtime_error(std::string{"cuda: "} + what + ": " + cudaGetErrorString(e));
    }

} // namespace

/**
 * one evaluation context: a CUDA stream + grow-only device buffers, reused across every eval of this
 * instance (no per-call cudaMalloc/cudaFree). instances are pooled by the trainer -- one per
 * model-scheduler thread -- so concurrent batches submit on distinct streams. the destructor releases
 * everything back to the driver.
 */
struct ModelEvaluator::Impl {
    sim::TetrisEngine* games = nullptr;
    float* weights = nullptr;
    float* fitness = nullptr;
    std::size_t gamesCap = 0, weightsCap = 0, fitnessCap = 0;
    cudaStream_t stream{};
    bool ready = false;

    void ensure(std::size_t blocks, std::size_t weight_count) {
        if(!ready) {
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
            ready = true;
        }
        auto const grow = [this](auto*& ptr, std::size_t& cap, std::size_t need, char const* what) {
            if(need <= cap)
                return;
            if(ptr != nullptr)
                cuda_check(cudaFree(ptr), "free (grow)");
            cuda_check(cudaMalloc(&ptr, need * sizeof(*ptr)), what);
            cap = need;
        };
        grow(games, gamesCap, blocks, "malloc games");
        grow(weights, weightsCap, weight_count, "malloc weights");
        grow(fitness, fitnessCap, blocks, "malloc fitness");
    }

    ~Impl() {
        // best effort: a dying evaluator must not throw, and at process teardown the context may
        // already be gone -- the driver reclaims whatever these miss.
        if(games != nullptr)
            static_cast<void>(cudaFree(games));
        if(weights != nullptr)
            static_cast<void>(cudaFree(weights));
        if(fitness != nullptr)
            static_cast<void>(cudaFree(fitness));
        if(ready)
            static_cast<void>(cudaStreamDestroy(stream));
    }

    Impl() = default;
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
    std::uint32_t const blocks = num_models * games; // grid = one block per (model, game)
    std::vector<float> out(blocks, 0.0f);
    if(blocks == 0)
        return out;

    // weights must be model-major with a full NUM_PARAMS slice per model.
    if(weights.size() != static_cast<std::size_t>(num_models) * ai::model_t::NUM_PARAMS)
        throw std::runtime_error("cuda: eval: weights.size() must equal num_models * NUM_PARAMS");

    // one engine per block; block g plays seeds[g % games], so every model sees the same set of games.
    std::vector<sim::TetrisEngine> hostGames;
    hostGames.reserve(blocks);
    for(std::uint32_t g = 0; g < blocks; ++g)
        hostGames.emplace_back(seeds[g % games]);

    Impl& ctx = *_impl;
    ctx.ensure(blocks, weights.size());

    cuda_check(
        cudaMemcpyAsync(
            ctx.games,
            hostGames.data(),
            blocks * sizeof(sim::TetrisEngine),
            cudaMemcpyHostToDevice,
            ctx.stream
        ),
        "copy games"
    );
    cuda_check(
        cudaMemcpyAsync(
            ctx.weights,
            weights.data(),
            weights.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            ctx.stream
        ),
        "copy weights"
    );

    eval_kernel<<<blocks, BLOCK, 0, ctx.stream>>>(ctx.games, blocks, games, ctx.weights, max_moves, ctx.fitness);
    cuda_check(cudaGetLastError(), "kernel launch");

    cuda_check(
        cudaMemcpyAsync(out.data(), ctx.fitness, blocks * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream),
        "copy fitness"
    );
    // the single host synchronisation of the whole evaluation.
    cuda_check(cudaStreamSynchronize(ctx.stream), "kernel run");
    return out;
}

} // namespace ta3::gpu
