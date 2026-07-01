//#include "ta3/ai/ai_multi_tetris.hpp"
//
//#include "test.hpp"
//
//#include <algorithm>
//#include <array>
//#include <span>
//
//#define MULTI_TEST(suite, name) CTH_EX_TEST(_ai_multi_tetris2, suite, name)
//
//namespace ta3::ai {
//
//namespace {
//
//    // one game keeps the constexpr step count modest; this only proves genInputs is
//    // constexpr-evaluable -- tiling/determinism over many games are checked at runtime below
//    constexpr bool expands_at_compile_time() {
//        AiMultiTetris multi{1, 99};
//        auto const total = multi.genInputs();
//        return total > 0
//            && multi.inputs().size() == total * m::INPUTS
//            && multi.placements(0).size() == total
//            && multi.inputs(0).size() == total * m::INPUTS;
//    }
//    static_assert(expands_at_compile_time());
//
//} // namespace
//
//MULTI_TEST(genInputs, fills_buffer_and_tiles_per_game) {
//    AiMultiTetris multi{4, 123};
//    auto const total = multi.genInputs();
//
//    EXPECT_GT(total, 0u);
//    EXPECT_EQ(multi.inputs().size(), total * m::INPUTS);
//
//    auto sum = 0uz;
//    for(auto g = 0uz; g < multi.games().size(); ++g)
//        sum += multi.placements(g).size();
//    EXPECT_EQ(sum, total);
//}
//
//MULTI_TEST(genInputs, every_live_game_has_a_variation) {
//    AiMultiTetris multi{8, 7};
//    multi.genInputs();
//
//    for(auto g = 0uz; g < multi.games().size(); ++g)
//        EXPECT_FALSE(multi.placements(g).empty());
//}
//
//MULTI_TEST(genInputs, variations_within_a_game_are_distinct) {
//    AiMultiTetris multi{1, 31};
//    multi.genInputs();
//
//    auto const slices = multi.inputs(0);
//    auto const count = multi.placements(0).size();
//
//    // dedup guarantees no two candidates of one game share an encoding
//    for(auto i = 0uz; i < count; ++i)
//        for(auto j = i + 1; j < count; ++j) {
//            auto const a = slices.subspan(i * m::INPUTS, m::INPUTS);
//            auto const b = slices.subspan(j * m::INPUTS, m::INPUTS);
//            EXPECT_FALSE(std::ranges::equal(a, b));
//        }
//}
//
//MULTI_TEST(step, advances_each_game_to_its_next_piece) {
//    AiMultiTetris multi{3, 50};
//    multi.genInputs();
//
//    std::array<sim::PieceType, 3> nextUp{};
//    for(auto g = 0uz; g < nextUp.size(); ++g)
//        nextUp[g] = multi.games()[g].lookahead().front();
//
//    std::array<size_t, 3> const choices{0, 0, 0};
//    multi.next(choices);
//
//    for(auto g = 0uz; g < nextUp.size(); ++g)
//        EXPECT_EQ(multi.games()[g].currentPiece(), nextUp[g]);
//}
//
//MULTI_TEST(determinism, same_seed_same_inputs) {
//    AiMultiTetris a{4, 2024};
//    AiMultiTetris b{4, 2024};
//
//    EXPECT_EQ(a.genInputs(), b.genInputs());
//    EXPECT_TRUE(std::ranges::equal(a.inputs(), b.inputs()));
//}
//
//MULTI_TEST(gameOver, batch_finishes_when_all_top_out) {
//    AiMultiTetris multi{2, 11};
//    EXPECT_FALSE(multi.empty());
//
//    auto guard = 0;
//    std::array<size_t, 2> const choices{0, 0};
//    while(!multi.empty() && guard++ < 10000) {
//        multi.genInputs();
//        multi.next(choices);
//    }
//
//    EXPECT_TRUE(multi.empty());
//    EXPECT_LT(guard, 10000);
//}
//
//}
