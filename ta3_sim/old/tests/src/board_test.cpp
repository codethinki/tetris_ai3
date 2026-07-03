#include "ta3/sim/board.hpp"
#include "ta3/sim/pieces/piece_offsets.hpp"

#include "test.hpp"

namespace ta3::sim {

namespace {

    Board empty_board() { return Board{}; }

    void fill_row(Board& board, int row) {
        board.place(PieceType::I, Orientation::TOP, vec2{0, row - 1});
        board.place(PieceType::I, Orientation::TOP, vec2{4, row - 1});
        board.place(PieceType::O, Orientation::TOP, vec2{7, row - 1});
    }

    void fill_bottom_two_rows(Board& board) {
        fill_row(board, 22);
        board.place(PieceType::I, Orientation::TOP, vec2{0, 20});
        board.place(PieceType::I, Orientation::TOP, vec2{4, 20});
    }

} // namespace

BOARD_TEST(empty, starts_cleared) {
    Board const board = empty_board();

    EXPECT_EQ(board.highest(), HEIGHT);
    EXPECT_TRUE(std::ranges::all_of(board.flat(), [](BlockType block) { return block == Board::EMPTY; }));
    EXPECT_TRUE(std::ranges::all_of(board.heightMapView(), [](int height) { return height == HEIGHT; }));
}

BOARD_TEST(place, updates_highest_and_height_map) {
    Board board = empty_board();

    board.place(PieceType::O, Orientation::TOP, vec2{3, 21});

    EXPECT_EQ(board.highest(), 21);
    EXPECT_EQ(board.heightMapView()[4], 21);
    EXPECT_EQ(board.heightMapView()[5], 21);

    vec2 const low_block{4, 21};
    vec2 const high_block{5, 22};
    EXPECT_EQ(board[low_block], BlockType::O);
    EXPECT_EQ(board[high_block], BlockType::O);
}

BOARD_TEST(place, rejects_overlapping_placement) {
    Board board = empty_board();
    board.place(PieceType::O, Orientation::TOP, vec2{3, 21});

    EXPECT_FALSE(board.available(PieceType::O, Orientation::TOP, vec2{3, 21}));
}

BOARD_TEST(placeOffset, drops_to_surface_on_empty_board) {
    Board const board = empty_board();

    auto const landing = board.placeOffset(PieceType::O, Orientation::TOP, vec2{3, 0});

    ASSERT_TRUE(landing.has_value());
    EXPECT_EQ(landing->x, 3);
    EXPECT_EQ(landing->y, 21);
}

BOARD_TEST(placeOffset, lands_on_existing_stack) {
    Board board = empty_board();
    board.place(PieceType::O, Orientation::TOP, vec2{3, 21});

    auto const landing = board.placeOffset(PieceType::O, Orientation::TOP, vec2{3, 0});

    ASSERT_TRUE(landing.has_value());
    EXPECT_EQ(landing->x, 3);
    EXPECT_EQ(landing->y, 19);
}

BOARD_TEST(fullLines, detects_complete_row) {
    Board board = empty_board();
    fill_row(board, 22);

    auto const lines = board.fullLines();

    ASSERT_EQ(lines.size(), 1);
    EXPECT_EQ(lines[0], 22);
}

BOARD_TEST(fullLines, ignores_incomplete_row) {
    Board board = empty_board();
    board.place(PieceType::O, Orientation::TOP, vec2{3, 21});

    EXPECT_TRUE(board.fullLines().empty());
}

BOARD_TEST(clearLines, removes_and_counts_full_rows) {
    Board board = empty_board();
    fill_row(board, 22);

    EXPECT_EQ(board.clearLines(), 1);
    EXPECT_TRUE(board.fullLines().empty());
}

BOARD_TEST(clearLines, clears_multiple_rows) {
    Board board = empty_board();
    fill_bottom_two_rows(board);

    EXPECT_EQ(board.clearLines(), 2);
    EXPECT_TRUE(board.fullLines().empty());
}

BOARD_TEST(clearLines, raises_highest_after_clear) {
    Board board = empty_board();
    fill_row(board, 22);

    auto const before_clear = board.highest();
    EXPECT_EQ(board.clearLines(), 1);
    EXPECT_GT(board.highest(), before_clear);
}

BOARD_TEST(holes, counts_gap_under_surface) {
    Board board = empty_board();
    board.place(PieceType::O, Orientation::TOP, vec2{0, 20});

    EXPECT_EQ(board.holes(1), 1);
    EXPECT_EQ(board.holes(), 2);
}

BOARD_TEST(holes, empty_board_has_none) {
    Board const board = empty_board();

    EXPECT_EQ(board.holes(), 0);
}

BOARD_TEST(roughness, empty_board_is_zero) {
    Board const board = empty_board();

    EXPECT_DOUBLE_EQ(board.roughness(), 0.0);
}

BOARD_TEST(roughness, uneven_surface_is_positive) {
    Board board = empty_board();
    board.place(PieceType::O, Orientation::TOP, vec2{0, 21});
    board.place(PieceType::O, Orientation::TOP, vec2{4, 19});

    EXPECT_GT(board.roughness(), 0.0);
}

BOARD_TEST(normalize, collapses_block_types) {
    Board board = empty_board();
    board.place(PieceType::I, Orientation::TOP, vec2{0, 21});
    board.place(PieceType::T, Orientation::TOP, vec2{4, 21});

    board.normalize();

    for(auto const block : board.flat()) {
        if(Board::occupied(block))
            EXPECT_EQ(block, Board::BLOCKED);
        else
            EXPECT_EQ(block, Board::EMPTY);
    }
}

}
