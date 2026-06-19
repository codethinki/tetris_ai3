#include "ta3/sim/board.hpp"
#include "ta3/sim/pieces/piece.hpp"
#include "ta3/sim/pieces/piece_offsets.hpp"

#include "test.hpp"

namespace ta3::sim {

namespace {

Board empty_board() { return Board{}; }

Piece make_piece(Board const& board, PieceType type) {
    auto piece = Piece::New(board, type);
    EXPECT_TRUE(piece.has_value());
    return std::move(*piece);
}

} // namespace

PIECE_TEST(piece, start_offset_constant) {
    EXPECT_EQ(Piece::START_OFFSET, vec2(3, 0));
}

PIECE_TEST(New, succeeds_for_all_piece_types) {
    Board const board = empty_board();

    for(PieceType type = PieceType::FIRST; *type < *PieceType::COUNT; type = static_cast<PieceType>(*type + 1)) {
        auto const piece = Piece::New(board, type);
        ASSERT_TRUE(piece.has_value()) << to_string(type);
        EXPECT_EQ(piece->type(), type);
        EXPECT_EQ(piece->orientation(), Orientation::TOP);
        EXPECT_EQ(piece->offset(), Piece::START_OFFSET);
        EXPECT_RANGE_EQ(piece->blocks(), piece_blocks(type, Orientation::TOP, Piece::START_OFFSET));
    }
}

PIECE_TEST(New, fails_when_spawn_overlaps_stack) {
    Board board = empty_board();
    board.place(PieceType::I, Orientation::TOP, Piece::START_OFFSET);

    EXPECT_FALSE(Piece::New(board, PieceType::I).has_value());
}

PIECE_TEST(move, left_until_wall) {
    Board const board = empty_board();
    Piece piece = make_piece(board, PieceType::O);

    int successful_moves = 0;
    while(piece.move(MoveType::LEFT)) {
        ++successful_moves;
    }

    EXPECT_GT(successful_moves, 0);
    EXPECT_FALSE(piece.move(MoveType::LEFT));

    for(auto const& block : piece.blocks()) {
        EXPECT_GE(block.x, 0);
    }
}

PIECE_TEST(move, right_until_wall) {
    Board const board = empty_board();
    Piece piece = make_piece(board, PieceType::O);

    int successful_moves = 0;
    while(piece.move(MoveType::RIGHT)) {
        ++successful_moves;
    }

    EXPECT_GT(successful_moves, 0);
    EXPECT_FALSE(piece.move(MoveType::RIGHT));

    for(auto const& block : piece.blocks()) {
        EXPECT_LT(block.x, WIDTH);
    }
}

PIECE_TEST(move, rejected_move_preserves_state) {
    Board const board = empty_board();
    Piece piece = make_piece(board, PieceType::O);

    while(piece.move(MoveType::LEFT)) {}

    auto const before = piece.offset();
    auto const blocks_before = piece.blocks();

    EXPECT_FALSE(piece.move(MoveType::LEFT));
    EXPECT_EQ(piece.offset(), before);
    EXPECT_RANGE_EQ(piece.blocks(), blocks_before);
}

PIECE_TEST(rotate, advances_orientation) {
    Board const board = empty_board();
    Piece piece = make_piece(board, PieceType::T);

    ASSERT_TRUE(piece.rotate(1));
    EXPECT_EQ(piece.orientation(), Orientation::RIGHT);
    EXPECT_RANGE_EQ(piece.blocks(), piece_blocks(PieceType::T, Orientation::RIGHT, piece.offset()));
}

PIECE_TEST(rotate, full_cycle_returns_to_origin) {
    Board const board = empty_board();
    Piece piece = make_piece(board, PieceType::T);
    auto const start = piece.offset();

    ASSERT_TRUE(piece.rotate(4));
    EXPECT_EQ(piece.orientation(), Orientation::TOP);
    EXPECT_EQ(piece.offset(), start);
}

PIECE_TEST(reorientate, fails_when_blocked) {
    Board board = empty_board();
    Piece piece = make_piece(board, PieceType::T);

    for(int x = 0; x <= WIDTH - 3; x += 2)
        board.place(PieceType::O, Orientation::TOP, vec2{x, 0});

    auto const orientation_before = piece.orientation();
    auto const offset_before = piece.offset();

    EXPECT_FALSE(piece.reorientate(Orientation::RIGHT));
    EXPECT_EQ(piece.orientation(), orientation_before);
    EXPECT_EQ(piece.offset(), offset_before);
}

PIECE_TEST(reorientate, wall_kick_when_needed) {
    Board board = empty_board();
    Piece piece = make_piece(board, PieceType::T);

    while(piece.move(MoveType::LEFT)) {}

    auto const orientation_before = piece.orientation();
    ASSERT_TRUE(piece.reorientate(Orientation::RIGHT));
    EXPECT_NE(piece.orientation(), orientation_before);
    EXPECT_TRUE(board.available(piece.type(), piece.orientation(), piece.offset()));
}

PIECE_TEST(placeTo, drops_to_surface) {
    Board board = empty_board();
    Piece piece = make_piece(board, PieceType::O);

    auto const landing = piece.placeTo(board);
    EXPECT_EQ(landing.y, HEIGHT - 2);
    EXPECT_EQ(piece.offset(), Piece::START_OFFSET);

    for(auto const& block : piece_blocks(PieceType::O, Orientation::TOP, landing)) {
        EXPECT_TRUE(Board::occupied(board[block]));
    }
}

PIECE_TEST(dummyOffset, matches_drop_position) {
    Board const board = empty_board();
    Piece piece = make_piece(board, PieceType::T);

    Board scratch = board;
    auto const landing = piece.placeTo(scratch);

    EXPECT_EQ(piece.dummyOffset(), landing);
    EXPECT_RANGE_EQ(piece.dummyBlocks(), piece_blocks(PieceType::T, Orientation::TOP, landing));
}

PIECE_TEST(blocks, tracks_movement) {
    Board const board = empty_board();
    Piece piece = make_piece(board, PieceType::S);

    ASSERT_TRUE(piece.move(MoveType::RIGHT));
    EXPECT_RANGE_EQ(piece.blocks(), piece_blocks(PieceType::S, Orientation::TOP, piece.offset()));
}

}
