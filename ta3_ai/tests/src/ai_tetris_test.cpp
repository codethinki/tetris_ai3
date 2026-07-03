#include "ta3/ai/ai_tetris.hpp"

#include "ta3/ai/ai_multi_tetris.hpp"
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

    // AiTetris and AiMultiTetris expand a game through the same generate_variations kernel, so a
    // single AiTetris must be constexpr-evaluable just like the engine underneath it
    constexpr bool plays_one_move_at_compile_time() {
        AiTetris game{99};
        if(game.gameOver() || game.variations() == 0)
            return false;

        std::vector<data_t> const scores(game.variations(), 0);
        game.next(scores);
        return !game.gameOver();
    }
    static_assert(plays_one_move_at_compile_time());

    std::vector<double> random_weights(uint64_t seed) {
        std::mt19937_64 rng{seed};
        std::uniform_real_distribution<double> dist{m::BOUNDS[0], m::BOUNDS[1]};

        std::vector<double> weights(model_t::params());
        for(auto& w : weights)
            w = dist(rng);
        return weights;
    }

} // namespace

// the single-game driver must reproduce the one-game batch driver move for move under one model,
// which is the guarantee that both share the same variation kernel and selection
AI_TETRIS_TEST(equivalence, matches_single_game_multi) {
    constexpr uint64_t SEED = 20260703;

    model_t model{};
    model.loadWeights(random_weights(1));

    AiTetris single{SEED};
    AiMultiTetris multi{1, SEED};

    size_t guard = 0;
    while(!multi.empty() && guard++ < 100000) {
        ASSERT_FALSE(single.gameOver());

        // identical candidate encodings for the one live game
        ASSERT_EQ(single.inputs().size(), multi.inputs().size());
        EXPECT_TRUE(std::ranges::equal(single.inputs(), multi.inputs()));

        single.next(model.batchForward(single.inputs()));
        multi.next(model.batchForward(multi.inputs()));
    }

    EXPECT_LT(guard, 100000u);
    EXPECT_TRUE(single.gameOver());
}

} // namespace ta3::ai
