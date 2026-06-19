#include "ta3/sim/tetris_game.hpp"

#include "test.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#define GAME_TEST(suite, name) CTH_EX_TEST(_sim_tetris_game, suite, name)

namespace ta3::sim {

namespace {

    // smallest seed whose first in-flight piece is @ref type
    constexpr uint64_t seed_with_current(PieceType type) {
        for(uint64_t s = 0; s < 1000; ++s)
            if(TetrisGame{s}.currentPiece() == type) return s;
        return 0;
    }

    constexpr bool plays_at_compile_time() {
        TetrisGame game{123};
        if(game.gameOver()) return false;
        if(game.orientation() != Orientation::TOP) return false;
        if(game.offset().x != 3 || game.offset().y != 0) return false;

        game.move(MoveType::RIGHT);
        game.rotate(RotationType::RIGHT);
        size_t const cleared = game.place();

        // the piece respawns at the top and nothing clears on a near-empty board
        return cleared == 0 && !game.gameOver()
            && game.orientation() == Orientation::TOP && game.offset().x == 3;
    }
    static_assert(plays_at_compile_time());

    // the lock is a hard drop: where the piece sat in flight must not change where it lands
    constexpr bool hard_drop_ignores_height() {
        TetrisGame a{55};
        TetrisGame b{55};

        b.move(MoveType::DOWN);
        b.move(MoveType::DOWN);
        b.move(MoveType::DOWN);

        return a.place() == b.place() && a.board() == b.board();
    }
    static_assert(hard_drop_ignores_height());

} // namespace

GAME_TEST(reset, starts_at_spawn_with_a_full_queue) {
    TetrisGame const game{1};

    EXPECT_FALSE(game.gameOver());
    EXPECT_EQ(game.heldPiece(), PieceType::COUNT);
    EXPECT_EQ(game.orientation(), Orientation::TOP);
    EXPECT_EQ(game.offset().x, 3);
    EXPECT_EQ(game.offset().y, 0);
    EXPECT_EQ(game.lookahead().size(), PIECE_QUEUE_SIZE - 1);
}

GAME_TEST(reset, restarts_back_to_the_start_state) {
    TetrisGame game{42};
    PieceType const first = game.currentPiece();

    game.move(MoveType::LEFT);
    game.place();
    game.place();

    game.reset(42);
    EXPECT_FALSE(game.gameOver());
    EXPECT_EQ(game.currentPiece(), first);
    EXPECT_EQ(game.orientation(), Orientation::TOP);
    EXPECT_EQ(game.offset().x, 3);
    EXPECT_EQ(game.offset().y, 0);
}

GAME_TEST(reset, same_seed_under_same_input_yields_same_board) {
    TetrisGame a{42};
    TetrisGame b{42};

    constexpr std::array script{
        Instruction::LEFT, Instruction::RRIGHT, Instruction::DOWN, Instruction::PLACE,
        Instruction::RIGHT, Instruction::HOLD, Instruction::PLACE, Instruction::RLEFT,
    };
    for(auto const instr : script) {
        a.step(instr);
        b.step(instr);
    }

    EXPECT_TRUE(a.board() == b.board());
    EXPECT_EQ(a.currentPiece(), b.currentPiece());
    EXPECT_EQ(a.heldPiece(), b.heldPiece());
}

GAME_TEST(engine, mirrors_the_engine_queue) {
    TetrisGame const game{9};

    EXPECT_EQ(game.currentPiece(), game.engine().currentPiece());
    EXPECT_EQ(game.heldPiece(), game.engine().heldPiece());
    EXPECT_EQ(game.lookahead().size(), game.engine().lookahead().size());
    EXPECT_EQ(game.gameOver(), game.engine().gameOver());
}

GAME_TEST(move, shifts_one_cell) {
    TetrisGame game{6};
    int const x = game.offset().x;

    ASSERT_TRUE(game.move(MoveType::RIGHT));
    EXPECT_EQ(game.offset().x, x + 1);

    ASSERT_TRUE(game.move(MoveType::LEFT));
    EXPECT_EQ(game.offset().x, x);
}

GAME_TEST(move, rejects_into_the_wall_and_leaves_state) {
    TetrisGame game{6};

    while(game.move(MoveType::LEFT)) {}
    int const wall = game.offset().x;

    EXPECT_FALSE(game.move(MoveType::LEFT));
    EXPECT_EQ(game.offset().x, wall);
}

GAME_TEST(move, soft_drop_reaches_the_ghost_then_stops) {
    TetrisGame game{6};
    vec2 const ghost = game.landingOffset();

    while(game.move(MoveType::DOWN)) {}

    EXPECT_FALSE(game.move(MoveType::DOWN));
    EXPECT_EQ(game.offset().x, ghost.x);
    EXPECT_EQ(game.offset().y, ghost.y);
}

GAME_TEST(rotate, advances_orientation_on_open_board) {
    TetrisGame game{6};
    ASSERT_EQ(game.orientation(), Orientation::TOP);

    ASSERT_TRUE(game.rotate(RotationType::RIGHT));
    EXPECT_EQ(game.orientation(), Orientation::RIGHT);

    ASSERT_TRUE(game.rotate(RotationType::LEFT));
    EXPECT_EQ(game.orientation(), Orientation::TOP);
}

GAME_TEST(rotate, full_cycle_returns_to_top_in_place) {
    TetrisGame game{6};

    for(int i = 0; i < 4; ++i)
        ASSERT_TRUE(game.rotate(RotationType::RIGHT));

    EXPECT_EQ(game.orientation(), Orientation::TOP);
    EXPECT_EQ(game.offset().x, 3);
    EXPECT_EQ(game.offset().y, 0);
}

GAME_TEST(rotate, kicks_when_the_identity_offset_is_blocked) {
    TetrisGame game{seed_with_current(PieceType::I)};
    ASSERT_EQ(game.currentPiece(), PieceType::I);

    // a flat I on the floor cannot stand up in place -- the bottom would clip the floor
    while(game.move(MoveType::DOWN)) {}
    vec2 const before = game.offset();

    ASSERT_TRUE(game.rotate(RotationType::RIGHT));
    EXPECT_NE(game.offset().y, before.y); // a non-identity kick was applied
    EXPECT_TRUE(game.board().available(game.currentPiece(), game.orientation(), game.offset()));
}

GAME_TEST(place, locks_at_the_hard_drop_landing_regardless_of_height) {
    TetrisGame shallow{77};
    TetrisGame sunk{77};

    sunk.move(MoveType::DOWN);
    sunk.move(MoveType::DOWN);
    sunk.move(MoveType::DOWN);

    EXPECT_EQ(shallow.place(), sunk.place());
    EXPECT_TRUE(shallow.board() == sunk.board());
}

GAME_TEST(place, advances_and_respawns_at_the_top) {
    TetrisGame game{3};
    PieceType const next = game.lookahead().front();

    game.move(MoveType::LEFT);
    EXPECT_EQ(game.place(), 0u); // nothing clears on an empty board

    EXPECT_EQ(game.currentPiece(), next);
    EXPECT_EQ(game.orientation(), Orientation::TOP);
    EXPECT_EQ(game.offset().x, 3);
    EXPECT_EQ(game.offset().y, 0);
}

GAME_TEST(place, fanned_play_clears_rows_within_bounds) {
    TetrisGame game{17};

    int guard = 0;
    bool anyCleared = false;
    while(!game.gameOver() && guard < 2000) {
        // spread placements across all columns so bottom rows actually complete
        int const target = guard % static_cast<int>(WIDTH);
        while(game.offset().x > target && game.move(MoveType::LEFT)) {}
        while(game.offset().x < target && game.move(MoveType::RIGHT)) {}

        size_t const cleared = game.place();
        EXPECT_LE(cleared, 4u);
        anyCleared = anyCleared || cleared > 0;
        ++guard;
    }

    EXPECT_TRUE(anyCleared);
}

GAME_TEST(hold, fills_then_swaps_and_respawns_at_top) {
    TetrisGame game{8};
    PieceType const first = game.currentPiece();
    PieceType const next = game.lookahead().front();
    ASSERT_EQ(game.heldPiece(), PieceType::COUNT);

    game.move(MoveType::RIGHT);
    game.rotate(RotationType::RIGHT);
    game.hold();

    EXPECT_EQ(game.heldPiece(), first);
    EXPECT_EQ(game.currentPiece(), next);
    EXPECT_EQ(game.orientation(), Orientation::TOP);
    EXPECT_EQ(game.offset().x, 3);
    EXPECT_EQ(game.offset().y, 0);

    PieceType const swappedIn = game.currentPiece();
    game.hold();
    EXPECT_EQ(game.heldPiece(), swappedIn);
    EXPECT_EQ(game.currentPiece(), first);
}

GAME_TEST(step, none_is_a_noop) {
    TetrisGame game{2};
    Orientation const orientation = game.orientation();
    vec2 const offset = game.offset();
    PieceType const piece = game.currentPiece();

    EXPECT_EQ(game.step(Instruction::NONE), 0u);

    EXPECT_EQ(game.orientation(), orientation);
    EXPECT_EQ(game.offset().x, offset.x);
    EXPECT_EQ(game.offset().y, offset.y);
    EXPECT_EQ(game.currentPiece(), piece);
}

GAME_TEST(step, routes_moves_and_rotations) {
    TetrisGame game{2};

    EXPECT_EQ(game.step(Instruction::LEFT), 0u);
    EXPECT_EQ(game.offset().x, 2);
    EXPECT_EQ(game.step(Instruction::RIGHT), 0u);
    EXPECT_EQ(game.offset().x, 3);

    game.step(Instruction::RRIGHT);
    EXPECT_EQ(game.orientation(), Orientation::RIGHT);
    game.step(Instruction::RLEFT);
    EXPECT_EQ(game.orientation(), Orientation::TOP);

    game.step(Instruction::DOWN);
    EXPECT_GT(game.offset().y, 0);
}

GAME_TEST(step, routes_place_and_hold) {
    TetrisGame game{4};
    PieceType const next = game.lookahead().front();

    EXPECT_EQ(game.step(Instruction::PLACE), 0u);
    EXPECT_EQ(game.currentPiece(), next);

    PieceType const current = game.currentPiece();
    game.step(Instruction::HOLD);
    EXPECT_EQ(game.heldPiece(), current);
}

GAME_TEST(gameOver, tops_out_when_the_spawn_column_fills) {
    TetrisGame game{5};

    int guard = 0;
    while(!game.gameOver() && guard++ < 1000)
        game.place(); // always at the centered spawn column, never completing a row

    EXPECT_TRUE(game.gameOver());
    EXPECT_LT(guard, 1000);
}

GAME_TEST(landingOffset, equals_the_board_drop_landing) {
    TetrisGame game{11};

    vec2 const ghost = game.landingOffset();
    vec2 const drop = game.board().dropPlace(game.currentPiece(), game.orientation(), game.offset());

    EXPECT_EQ(ghost.x, drop.x);
    EXPECT_EQ(ghost.y, drop.y);
}

}
