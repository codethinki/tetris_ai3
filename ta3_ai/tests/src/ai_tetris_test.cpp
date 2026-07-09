#include "ta3/ai/ai_tetris.hpp"

#include "ta3/ai/model.hpp"

#include "test.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#define AI_TETRIS_TEST(suite, name) CTH_EX_TEST(_ai_tetris, suite, name)

namespace ta3::ai {

namespace {

    // AiTetris steps a game through the same constexpr search core as the GPU evaluator, so a single
    // move must be evaluable at compile time just like the engine underneath it
    constexpr bool plays_one_move_at_compile_time() {
        AiTetris game{99};
        if(game.gameOver())
            return false;

        model_t const model{}; // zero weights: a compile-time smoke test of the search + commit path
        game.step(model);
        return true;
    }
    //static_assert(plays_one_move_at_compile_time());

    std::vector<double> random_weights(uint64_t seed) {
        std::mt19937_64 rng{seed};
        std::uniform_real_distribution<double> dist{m::BOUNDS[0], m::BOUNDS[1]};

        std::vector<double> weights(model_t::params());
        for(auto& w : weights)
            w = dist(rng);
        return weights;
    }

} // namespace



} // namespace ta3::ai
