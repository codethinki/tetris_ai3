#include "ta3/ai/multi_tetris2.hpp"

#include "test.hpp"

#include <algorithm>
#include <array>
#include <span>

#define MULTI_TEST(suite, name) CTH_EX_TEST(_ai_multi_tetris2, suite, name)

namespace ta3::ai {

namespace {

    // one game keeps the constexpr step count modest; this only proves genInputs is
    // constexpr-evaluable -- tiling/determinism over many games are checked at runtime below
    constexpr bool expands_at_compile_time() {
        MultiTetris2 multi{1, 99};
        size_t const total = multi.genInputs();
        return total > 0
            && multi.inputs().size() == total * model_t::INPUTS
            && multi.placements(0).size() == total
            && multi.inputs(0).size() == total * model_t::INPUTS;
    }
    static_assert(expands_at_compile_time());

} // namespace

MULTI_TEST(genInputs, fills_buffer_and_tiles_per_game) {
    MultiTetris2 multi{4, 123};
    size_t const total = multi.genInputs();

    EXPECT_GT(total, 0u);
    EXPECT_EQ(multi.inputs().size(), total * model_t::INPUTS);

    size_t sum = 0;
    for(size_t g = 0; g < multi.games().size(); ++g)
        sum += multi.placements(g).size();
    EXPECT_EQ(sum, total);
}

MULTI_TEST(genInputs, every_live_game_has_a_variation) {
    MultiTetris2 multi{8, 7};
    multi.genInputs();

    for(size_t g = 0; g < multi.games().size(); ++g)
        EXPECT_FALSE(multi.placements(g).empty());
}

MULTI_TEST(genInputs, variations_within_a_game_are_distinct) {
    MultiTetris2 multi{1, 31};
    multi.genInputs();

    auto const slices = multi.inputs(0);
    size_t const count = multi.placements(0).size();

    // dedup guarantees no two candidates of one game share an encoding
    for(size_t i = 0; i < count; ++i)
        for(size_t j = i + 1; j < count; ++j) {
            auto const a = slices.subspan(i * model_t::INPUTS, model_t::INPUTS);
            auto const b = slices.subspan(j * model_t::INPUTS, model_t::INPUTS);
            EXPECT_FALSE(std::ranges::equal(a, b));
        }
}

MULTI_TEST(step, advances_each_game_to_its_next_piece) {
    MultiTetris2 multi{3, 50};
    multi.genInputs();

    std::array<sim::PieceType, 3> nextUp{};
    for(size_t g = 0; g < nextUp.size(); ++g)
        nextUp[g] = multi.games()[g].lookahead().front();

    std::array<size_t, 3> const choices{0, 0, 0};
    multi.step(choices);

    for(size_t g = 0; g < nextUp.size(); ++g)
        EXPECT_EQ(multi.games()[g].currentPiece(), nextUp[g]);
}

MULTI_TEST(determinism, same_seed_same_inputs) {
    MultiTetris2 a{4, 2024};
    MultiTetris2 b{4, 2024};

    EXPECT_EQ(a.genInputs(), b.genInputs());
    EXPECT_TRUE(std::ranges::equal(a.inputs(), b.inputs()));
}

MULTI_TEST(gameOver, batch_finishes_when_all_top_out) {
    MultiTetris2 multi{2, 11};
    EXPECT_FALSE(multi.gameOver());

    int guard = 0;
    std::array<size_t, 2> const choices{0, 0};
    while(!multi.gameOver() && guard++ < 10000) {
        multi.genInputs();
        multi.step(choices);
    }

    EXPECT_TRUE(multi.gameOver());
    EXPECT_LT(guard, 10000);
}

}
