#include "ta3/sim/board2.hpp"

#include "test.hpp"

#define BOARD2_TEST(suite, name) CTH_EX_TEST(_sim_board2, suite, name)

namespace ta3::sim {

namespace {

    // fills @ref row completely (cols 0..9) and leaves a 2-tall O overhang at row-1 on cols 8,9
    constexpr void fill_row(Board2& board, int row) {
        board.place(PieceType::I, Orientation::TOP, vec2{0, row - 1});
        board.place(PieceType::I, Orientation::TOP, vec2{4, row - 1});
        board.place(PieceType::O, Orientation::TOP, vec2{7, row - 1});
    }

    // fills the bottom @ref n rows completely, no overhang; @ref n must be even
    constexpr void fill_rows(Board2& board, int n) {
        int const bottom = static_cast<int>(ROWS) - 1;
        for(int row = bottom; row > bottom - n; --row) {
            board.place(PieceType::I, Orientation::TOP, vec2{0, row - 1}); // cols 0..3
            board.place(PieceType::I, Orientation::TOP, vec2{4, row - 1}); // cols 4..7
        }
        for(int row = bottom; row > bottom - n; row -= 2)
            board.place(PieceType::O, Orientation::TOP, vec2{7, row - 1}); // cols 8,9, two rows
    }

    // hard-drops a piece from @ref spawn and locks it where it lands
    constexpr void hard_drop(Board2& board, PieceType type, Orientation orientation, vec2 spawn) {
        board.place(type, orientation, board.dropPlace(type, orientation, spawn));
    }

    // builds a stack in cols 3,4 with an I bridged over it, leaving a ceiling at row 16 on cols 5,6
    // and empty space (a well) beneath it
    constexpr void build_overhang(Board2& board) {
        hard_drop(board, PieceType::O, Orientation::TOP, vec2{2, 0});
        hard_drop(board, PieceType::O, Orientation::TOP, vec2{2, 0});
        hard_drop(board, PieceType::O, Orientation::TOP, vec2{2, 0});
        hard_drop(board, PieceType::I, Orientation::TOP, vec2{3, 0});
    }

    constexpr bool drops_and_clears_at_compile_time() {
        Board2 board{};
        fill_row(board, 22);
        if(board.fullLines() != 1) return false;
        if(board.clearLines() != 1) return false;
        if(board.fullLines() != 0) return false;
        return board.dropPlace(PieceType::O, Orientation::TOP, vec2{7, 0}).y == 20;
    }
    static_assert(drops_and_clears_at_compile_time());

    constexpr bool availability_folds_at_compile_time() {
        Board2 board{};
        board.place(PieceType::O, Orientation::TOP, vec2{3, 21});
        return !board.available(PieceType::O, Orientation::TOP, vec2{3, 21})
            && board.available(PieceType::O, Orientation::TOP, vec2{3, 19});
    }
    static_assert(availability_folds_at_compile_time());

} // namespace

BOARD2_TEST(empty, starts_cleared) {
    Board2 const board{};

    EXPECT_EQ(board.holes(), 0);
    EXPECT_EQ(board.fullLines(), 0u);
    EXPECT_DOUBLE_EQ(board.roughness(), 0.0);
}

BOARD2_TEST(available, accepts_free_placement_on_empty_board) {
    Board2 const board{};

    EXPECT_TRUE(board.available(PieceType::O, Orientation::TOP, vec2{3, 0}));
    EXPECT_TRUE(board.available(PieceType::I, Orientation::TOP, vec2{0, 0}));
    EXPECT_TRUE(board.available(PieceType::I, Orientation::RIGHT, vec2{0, 0}));
}

BOARD2_TEST(available, rejects_overlapping_placement) {
    Board2 board{};
    board.place(PieceType::O, Orientation::TOP, vec2{3, 21});

    EXPECT_FALSE(board.available(PieceType::O, Orientation::TOP, vec2{3, 21}));
    EXPECT_TRUE(board.available(PieceType::O, Orientation::TOP, vec2{3, 19}));
}

BOARD2_TEST(available, rejects_out_of_bounds_sides) {
    Board2 const board{};

    EXPECT_TRUE(board.available(PieceType::O, Orientation::TOP, vec2{-1, 0})); // cols 0,1
    EXPECT_FALSE(board.available(PieceType::O, Orientation::TOP, vec2{-2, 0})); // col -1
    EXPECT_TRUE(board.available(PieceType::O, Orientation::TOP, vec2{7, 0})); // cols 8,9
    EXPECT_FALSE(board.available(PieceType::O, Orientation::TOP, vec2{8, 0})); // col 10
}

BOARD2_TEST(available, rejects_below_the_floor) {
    Board2 const board{};

    EXPECT_TRUE(board.available(PieceType::O, Orientation::TOP, vec2{3, 21})); // rows 21,22
    EXPECT_FALSE(board.available(PieceType::O, Orientation::TOP, vec2{3, 22})); // row 23 is off-board
}

BOARD2_TEST(dropPlace, drops_to_floor_on_empty_board) {
    Board2 const board{};

    auto const landing = board.dropPlace(PieceType::O, Orientation::TOP, vec2{3, 0});

    EXPECT_EQ(landing.x, 3);
    EXPECT_EQ(landing.y, 21);
}

BOARD2_TEST(dropPlace, vertical_i_drops_to_floor) {
    Board2 const board{};

    auto const landing = board.dropPlace(PieceType::I, Orientation::RIGHT, vec2{0, 0});

    EXPECT_EQ(landing.y, 19); // four tall, bottom block on row 22
}

BOARD2_TEST(dropPlace, lands_on_existing_stack) {
    Board2 board{};
    board.place(PieceType::O, Orientation::TOP, vec2{3, 21});

    auto const landing = board.dropPlace(PieceType::O, Orientation::TOP, vec2{3, 0});

    EXPECT_EQ(landing.y, 19);
}

BOARD2_TEST(dropPlace, falls_past_overhang_into_well) {
    Board2 board{};
    build_overhang(board); // ceiling at row 16 over cols 5,6, empty beneath
    EXPECT_EQ(board.holes(), 12); // cols 5,6 each trap rows 17..22

    vec2 const tucked{4, 17}; // O on cols 5,6, started under the ceiling
    ASSERT_TRUE(board.available(PieceType::O, Orientation::TOP, tucked));

    auto const landing = board.dropPlace(PieceType::O, Orientation::TOP, tucked);
    EXPECT_EQ(landing.y, 21); // ignores the ceiling, falls to the floor
}

BOARD2_TEST(dropPlace, every_piece_lands_and_stacks) {
    for(uint32_t t = 0; t < *PieceType::COUNT; ++t) {
        for(uint32_t o = 0; o < *Orientation::SIZE; ++o) {
            auto const type = static_cast<PieceType>(t);
            auto const orientation = static_cast<Orientation>(o);
            vec2 const spawn{3, 0};

            Board2 board{};
            ASSERT_TRUE(board.available(type, orientation, spawn));

            auto const first = board.dropPlace(type, orientation, spawn);
            EXPECT_TRUE(board.available(type, orientation, first));
            board.place(type, orientation, first);
            EXPECT_FALSE(board.available(type, orientation, first));

            auto const second = board.dropPlace(type, orientation, spawn);
            EXPECT_LT(second.y, first.y); // identical piece stacks strictly higher
        }
    }
}

BOARD2_TEST(optDropPlace, matches_dropPlace_for_legal_offset) {
    Board2 const board{};

    auto const landing = board.optDropPlace(PieceType::O, Orientation::TOP, vec2{3, 0});

    ASSERT_TRUE(landing.has_value());
    EXPECT_EQ(landing->x, board.dropPlace(PieceType::O, Orientation::TOP, vec2{3, 0}).x);
    EXPECT_EQ(landing->y, board.dropPlace(PieceType::O, Orientation::TOP, vec2{3, 0}).y);
}

BOARD2_TEST(optDropPlace, returns_nullopt_for_out_of_bounds) {
    Board2 const board{};

    EXPECT_FALSE(board.optDropPlace(PieceType::O, Orientation::TOP, vec2{3, 22}).has_value());
    EXPECT_FALSE(board.optDropPlace(PieceType::O, Orientation::TOP, vec2{8, 0}).has_value());
}

BOARD2_TEST(optDropPlace, returns_nullopt_when_start_overlaps) {
    Board2 board{};
    board.place(PieceType::O, Orientation::TOP, vec2{3, 21});

    EXPECT_FALSE(board.optDropPlace(PieceType::O, Orientation::TOP, vec2{3, 21}).has_value());
}

BOARD2_TEST(place, occupies_the_landed_cells) {
    Board2 board{};
    auto const landing = board.dropPlace(PieceType::O, Orientation::TOP, vec2{3, 0});
    board.place(PieceType::O, Orientation::TOP, landing);

    EXPECT_FALSE(board.available(PieceType::O, Orientation::TOP, landing));
    EXPECT_EQ(board.holes(), 0);
}

BOARD2_TEST(fullLines, detects_complete_row) {
    Board2 board{};
    fill_row(board, 22);

    EXPECT_EQ(board.fullLines(), 1u);
}

BOARD2_TEST(fullLines, ignores_incomplete_row) {
    Board2 board{};
    board.place(PieceType::O, Orientation::TOP, vec2{3, 21});

    EXPECT_EQ(board.fullLines(), 0u);
}

BOARD2_TEST(clearLines, clears_nothing_when_no_full_row) {
    Board2 board{};
    board.place(PieceType::O, Orientation::TOP, vec2{3, 21});

    EXPECT_EQ(board.clearLines(), 0u);
    EXPECT_EQ(board.fullLines(), 0u);
}

BOARD2_TEST(clearLines, removes_and_counts_single_row) {
    Board2 board{};
    fill_row(board, 22);

    EXPECT_EQ(board.clearLines(), 1u);
    EXPECT_EQ(board.fullLines(), 0u);
}

BOARD2_TEST(clearLines, clears_two_rows) {
    Board2 board{};
    fill_rows(board, 2);

    EXPECT_EQ(board.fullLines(), 2u);
    EXPECT_EQ(board.clearLines(), 2u);
    EXPECT_EQ(board.fullLines(), 0u);
}

BOARD2_TEST(clearLines, clears_four_rows) {
    Board2 board{};
    fill_rows(board, 4);

    EXPECT_EQ(board.fullLines(), 4u);
    EXPECT_EQ(board.clearLines(), 4u);
    EXPECT_EQ(board.fullLines(), 0u);
    EXPECT_EQ(board.holes(), 0);
}

BOARD2_TEST(clearLines, drops_survivors_after_multi_clear) {
    Board2 board{};
    fill_rows(board, 2); // rows 21,22 full
    hard_drop(board, PieceType::O, Orientation::TOP, vec2{0, 0}); // O on cols 1,2 at rows 19,20

    ASSERT_EQ(board.clearLines(), 2u);
    EXPECT_EQ(board.fullLines(), 0u);
    EXPECT_EQ(board.holes(), 0); // survivor fell onto the floor, no gap below
    EXPECT_GT(board.roughness(), 0.0);
}

BOARD2_TEST(clearLines, drops_overhang_onto_cleared_row) {
    Board2 board{};
    fill_row(board, 22);
    ASSERT_EQ(board.clearLines(), 1u);

    // the row-21 overhang on cols 8,9 falls onto row 22, so an O dropped there lands one row higher
    auto const landing = board.dropPlace(PieceType::O, Orientation::TOP, vec2{7, 0});
    EXPECT_EQ(landing.y, 20);
}

BOARD2_TEST(holes, counts_gap_under_surface) {
    Board2 board{};
    board.place(PieceType::O, Orientation::TOP, vec2{0, 20}); // cols 1,2 at rows 20,21; row 22 trapped

    EXPECT_EQ(board.holes(), 2);
}

BOARD2_TEST(holes, empty_board_has_none) {
    Board2 const board{};

    EXPECT_EQ(board.holes(), 0);
}

BOARD2_TEST(roughness, empty_board_is_zero) {
    Board2 const board{};

    EXPECT_DOUBLE_EQ(board.roughness(), 0.0);
}

BOARD2_TEST(roughness, uneven_surface_is_positive) {
    Board2 board{};
    board.place(PieceType::O, Orientation::TOP, vec2{0, 21});
    board.place(PieceType::O, Orientation::TOP, vec2{4, 19});

    EXPECT_GT(board.roughness(), 0.0);
}

BOARD2_TEST(equality, distinguishes_boards_by_occupancy) {
    Board2 a{};
    Board2 b{};
    EXPECT_TRUE(a == b);

    a.place(PieceType::O, Orientation::TOP, vec2{3, 21});
    EXPECT_FALSE(a == b);

    b.place(PieceType::O, Orientation::TOP, vec2{3, 21});
    EXPECT_TRUE(a == b);
}

BOARD2_TEST(equality, cleared_overhang_leaves_residue) {
    Board2 a{};
    Board2 const b{};

    fill_row(a, 22);
    a.clearLines();

    // fill_row leaves a row-21 overhang on cols 8,9 that drops to row 22 after the clear, so the
    // boards are not equal -- the cleared board still has those two cells
    EXPECT_FALSE(a == b);
}

}
