#include "ta3/gpu/device_canary.hpp"
#include "ta3/gpu/model_eval.hpp"

#include <ta3/ai/model.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

/**
 * @file parity_test.cpp
 * @brief GPU evaluator tests: a device bit-intrinsics canary, and device-vs-host bit-exact parity checks.
 */
namespace ta3::gpu {
namespace {

    /** deterministic weight vector within the model's parameter bounds, seeded by @p seed. */
    [[nodiscard]] std::vector<float> random_weights(std::uint32_t seed) {
        std::mt19937 rng{seed};
        auto const bounds = ai::model_t::bounds(); // {x = lower, y = upper}, see ModelV7::BOUNDS
        std::uniform_real_distribution<float> dist{
            static_cast<float>(bounds.x),
            static_cast<float>(bounds.y)
        };

        std::vector<float> w(ai::model_t::NUM_PARAMS);
        for(auto& v : w)
            v = dist(rng);
        return w;
    }

    constexpr std::uint32_t MAX_MOVES = 200;

    [[nodiscard]] std::vector<std::uint64_t> const& parity_seeds() {
        static std::vector<std::uint64_t> const seeds{1, 2, 3, 4};
        return seeds;
    }

} // namespace

TEST(GpuModelEval, DeviceBitsCanary) {
    EXPECT_TRUE(device_bits_canary())
        << "ta3::sim::ctz/popcnt/bit_width32 mismatched on the selected GPU device -- see "
           "device_canary.cpp and ta3_sim/incl/ta3/sim/utility/bits.hpp";
}

// device eval() must match the CPU eval_host() parity reference EXACTLY (bit-for-bit), across several
// deterministic weight vectors, for the single-model overload.
TEST(GpuModelEval, SingleModelMatchesHostExactly) {
    ModelEvaluator evaluator;
    auto const& seeds = parity_seeds();

    for(std::uint32_t modelSeed : {11u, 22u, 33u}) {
        auto const weights = random_weights(modelSeed);

        auto const device = evaluator.eval(weights, seeds, MAX_MOVES);
        auto const host = ModelEvaluator::eval_host(weights, seeds, MAX_MOVES);

        ASSERT_EQ(device.size(), host.size());
        ASSERT_EQ(device.size(), seeds.size());
        for(std::size_t g = 0; g < device.size(); ++g)
            EXPECT_EQ(device[g], host[g]) << "modelSeed=" << modelSeed << " game=" << g;
    }
}

// the batched, multi-model eval() must be model-major-equivalent to calling the single-model eval() once
// per model: model m's slice of the batched result must match evaluating model m alone on the same games.
TEST(GpuModelEval, MultiModelBatchMatchesPerModelSingleEval) {
    ModelEvaluator evaluator;
    auto const& seeds = parity_seeds();
    constexpr std::uint32_t NUM_MODELS = 3;

    std::vector<std::vector<float>> perModel;
    std::vector<float> packed;
    packed.reserve(static_cast<std::size_t>(NUM_MODELS) * ai::model_t::NUM_PARAMS);
    for(std::uint32_t m = 0; m < NUM_MODELS; ++m) {
        auto w = random_weights(100u + m);
        packed.insert(packed.end(), w.begin(), w.end());
        perModel.push_back(std::move(w));
    }

    auto const batched = evaluator.eval(packed, NUM_MODELS, seeds, MAX_MOVES);
    ASSERT_EQ(batched.size(), static_cast<std::size_t>(NUM_MODELS) * seeds.size());

    for(std::uint32_t m = 0; m < NUM_MODELS; ++m) {
        auto const single = evaluator.eval(perModel[m], seeds, MAX_MOVES);
        ASSERT_EQ(single.size(), seeds.size());
        for(std::size_t g = 0; g < seeds.size(); ++g)
            EXPECT_EQ(batched[m * seeds.size() + g], single[g]) << "model=" << m << " game=" << g;
    }

    // also cross-check the batched result against the host parity reference, per model.
    for(std::uint32_t m = 0; m < NUM_MODELS; ++m) {
        auto const host = ModelEvaluator::eval_host(perModel[m], seeds, MAX_MOVES);
        for(std::size_t g = 0; g < seeds.size(); ++g)
            EXPECT_EQ(batched[m * seeds.size() + g], host[g]) << "model=" << m << " game=" << g;
    }
}

} // namespace ta3::gpu
