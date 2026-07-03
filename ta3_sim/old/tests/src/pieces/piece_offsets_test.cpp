#include "ta3/sim/board.hpp"
#include "ta3/sim/pieces/piece.hpp"
#include "ta3/sim/pieces/piece_offsets.hpp"

#include "test.hpp"

#include <algorithm>

namespace ta3::sim {

namespace {

blocks_t const& base_blocks(PieceType type, Orientation orientation) {
    return PIECE_ORIENTATION_OFFSETS[*type][*orientation];
}

} // namespace

PIECE_TEST(piece_blocks, applies_offset) {
    vec2 const offset{4, 7};
    auto const shifted = piece_blocks(PieceType::T, Orientation::RIGHT, offset);
    auto const expected = base_blocks(PieceType::T, Orientation::RIGHT);

    for(size_t i = 0; i < BLOCKS; ++i) {
        EXPECT_EQ(shifted[i], expected[i] + offset);
    }
}

PIECE_TEST(piece_blocks, zero_offset_matches_table) {
    for(PieceType type = PieceType::FIRST; *type < *PieceType::COUNT; type = static_cast<PieceType>(*type + 1)) {
        for(Orientation orientation = Orientation::BEGIN; *orientation < *Orientation::SIZE;
            orientation = static_cast<Orientation>(*orientation + 1)) {
            EXPECT_RANGE_EQ(piece_blocks(type, orientation, vec2{0, 0}), base_blocks(type, orientation));
        }
    }
}

PIECE_TEST(PIECE_ORIENTATION_OFFSETS, o_piece_is_orientation_invariant) {
    auto const& reference = base_blocks(PieceType::O, Orientation::TOP);

    for(Orientation orientation = static_cast<Orientation>(*Orientation::BEGIN + 1); *orientation < *Orientation::SIZE;
        orientation = static_cast<Orientation>(*orientation + 1)) {
        EXPECT_RANGE_EQ(base_blocks(PieceType::O, orientation), reference);
    }
}

PIECE_TEST(PIECE_ORIENTATION_OFFSETS, each_shape_has_four_blocks) {
    for(PieceType type = PieceType::FIRST; *type < *PieceType::COUNT; type = static_cast<PieceType>(*type + 1)) {
        for(Orientation orientation = Orientation::BEGIN; *orientation < *Orientation::SIZE;
            orientation = static_cast<Orientation>(*orientation + 1)) {
            auto const& blocks = base_blocks(type, orientation);
            EXPECT_EQ(blocks.size(), BLOCKS);
        }
    }
}

PIECE_TEST(PIECE_ORIENTATION_OFFSETS, i_piece_changes_span) {
    auto const horizontal = base_blocks(PieceType::I, Orientation::TOP);
    auto const vertical = base_blocks(PieceType::I, Orientation::LEFT);

    int horizontal_min_x = horizontal[0].x;
    int horizontal_max_x = horizontal[0].x;
    int vertical_min_y = vertical[0].y;
    int vertical_max_y = vertical[0].y;

    for(auto const& block : horizontal) {
        horizontal_min_x = std::min(horizontal_min_x, block.x);
        horizontal_max_x = std::max(horizontal_max_x, block.x);
    }

    for(auto const& block : vertical) {
        vertical_min_y = std::min(vertical_min_y, block.y);
        vertical_max_y = std::max(vertical_max_y, block.y);
    }

    EXPECT_EQ(horizontal_max_x - horizontal_min_x, 3);
    EXPECT_EQ(vertical_max_y - vertical_min_y, 3);
}

PIECE_TEST(PIECE_ORIENTATION_OFFSETS, default_spawn_fits_board) {
    for(PieceType type = PieceType::FIRST; *type < *PieceType::COUNT; type = static_cast<PieceType>(*type + 1)) {
        auto const blocks = piece_blocks(type, Orientation::TOP, Piece::START_OFFSET);
        for(auto const& block : blocks) {
            EXPECT_TRUE(Board::inBounds(block));
        }
    }
}

}
