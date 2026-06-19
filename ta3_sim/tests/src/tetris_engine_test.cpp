#include "ta3/sim/tetris_engine.hpp"

#include "test.hpp"

#include <array>

#define ENGINE_TEST(suite, name) CTH_EX_TEST(_sim_tetris_engine, suite, name)

namespace ta3::sim {

namespace {

    // places the current piece in TOP orientation at the first legal column, returning its type
    constexpr PieceType drainOne(TetrisEngine& engine) {
        PieceType const type = engine.currentPiece();
        for(int x = -2; x < static_cast<int>(WIDTH) + 2; ++x)
            if(engine.board().available(type, Orientation::TOP, vec2{x, 0})) {
                engine.place(Orientation::TOP, x);
                break;
            }
        return type;
    }

    constexpr bool plays_at_compile_time() {
        TetrisEngine engine{123};
        if(engine.gameOver()) return false;

        size_t const cleared = engine.place(Orientation::TOP, 3);
        engine.hold();
        return cleared == 0 && !engine.gameOver();
    }
    static_assert(plays_at_compile_time());

    constexpr bool bag_is_a_permutation_at_compile_time() {
        TetrisEngine engine{7};

        std::array<int, *PieceType::COUNT> counts{};
        for(size_t i = 0; i < *PieceType::COUNT; ++i)
            ++counts[*drainOne(engine)];

        for(auto const c : counts)
            if(c != 1) return false;
        return true;
    }
    static_assert(bag_is_a_permutation_at_compile_time());

} // namespace

ENGINE_TEST(reset, starts_active_with_a_full_queue) {
    TetrisEngine const engine{1};

    EXPECT_FALSE(engine.gameOver());
    EXPECT_EQ(engine.heldPiece(), PieceType::COUNT);
    EXPECT_EQ(engine.lookahead().size(), PIECE_QUEUE_SIZE - 1);
}

ENGINE_TEST(reset, same_seed_yields_same_sequence) {
    TetrisEngine a{42};
    TetrisEngine b{42};

    for(int i = 0; i < 20; ++i)
        EXPECT_EQ(drainOne(a), drainOne(b));
}

ENGINE_TEST(reset, different_seeds_diverge) {
    TetrisEngine a{1};
    TetrisEngine b{2};

    bool anyDifferent = false;
    for(int i = 0; i < 2 * static_cast<int>(*PieceType::COUNT); ++i)
        if(drainOne(a) != drainOne(b)) anyDifferent = true;

    EXPECT_TRUE(anyDifferent);
}

ENGINE_TEST(place, advances_to_the_next_piece) {
    TetrisEngine engine{3};
    PieceType const next = engine.lookahead().front();

    EXPECT_EQ(engine.place(Orientation::TOP, 3), 0u); // nothing clears on an empty board
    EXPECT_EQ(engine.currentPiece(), next);
}

ENGINE_TEST(hold, fills_then_swaps_the_slot) {
    TetrisEngine engine{8};
    PieceType const first = engine.currentPiece();
    ASSERT_EQ(engine.heldPiece(), PieceType::COUNT);

    engine.hold();
    EXPECT_EQ(engine.heldPiece(), first);

    PieceType const swappedIn = engine.currentPiece();
    engine.hold();
    EXPECT_EQ(engine.heldPiece(), swappedIn);
    EXPECT_EQ(engine.currentPiece(), first);
}

ENGINE_TEST(gameOver, tops_out_when_the_spawn_column_fills) {
    TetrisEngine engine{5};

    // stack everything in the center, never completing a row, until the spawn is blocked
    int guard = 0;
    while(!engine.gameOver() && guard++ < 1000)
        engine.place(Orientation::TOP, 3);

    EXPECT_TRUE(engine.gameOver());
    EXPECT_LT(guard, 1000);
}

}
