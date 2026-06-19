#include "ta3/sim/pieces/piece_defs.hpp"

#include "test.hpp"

namespace ta3::sim {

PIECE_TEST(piece_defs, blocks_constant) {
    EXPECT_EQ(BLOCKS, 4);
}

PIECE_TEST(to_block_type, maps_known_pieces) {
    EXPECT_EQ(to_block_type(PieceType::I), BlockType::I);
    EXPECT_EQ(to_block_type(PieceType::O), BlockType::O);
    EXPECT_EQ(to_block_type(PieceType::T), BlockType::T);
    EXPECT_EQ(to_block_type(PieceType::L), BlockType::L);
}

PIECE_TEST(to_block_type, sentinel_piece_type) {
    EXPECT_EQ(to_block_type(PieceType::COUNT), BlockType::SIZE);
}

PIECE_TEST(to_move_type, translation_instructions) {
    ASSERT_TRUE(to_move_type(Instruction::LEFT).has_value());
    EXPECT_EQ(*to_move_type(Instruction::LEFT), MoveType::LEFT);
    ASSERT_TRUE(to_move_type(Instruction::RIGHT).has_value());
    EXPECT_EQ(*to_move_type(Instruction::RIGHT), MoveType::RIGHT);
    ASSERT_TRUE(to_move_type(Instruction::DOWN).has_value());
    EXPECT_EQ(*to_move_type(Instruction::DOWN), MoveType::DOWN);
}

PIECE_TEST(to_move_type, non_translation_instructions) {
    EXPECT_FALSE(to_move_type(Instruction::PLACE).has_value());
    EXPECT_FALSE(to_move_type(Instruction::HOLD).has_value());
    EXPECT_FALSE(to_move_type(Instruction::NONE).has_value());
    EXPECT_FALSE(to_move_type(Instruction::RLEFT).has_value());
    EXPECT_FALSE(to_move_type(Instruction::RRIGHT).has_value());
}

PIECE_TEST(to_rotation_type, rotation_instructions) {
    ASSERT_TRUE(to_rotation_type(Instruction::RLEFT).has_value());
    EXPECT_EQ(*to_rotation_type(Instruction::RLEFT), RotationType::LEFT);
    ASSERT_TRUE(to_rotation_type(Instruction::RRIGHT).has_value());
    EXPECT_EQ(*to_rotation_type(Instruction::RRIGHT), RotationType::RIGHT);
}

PIECE_TEST(to_rotation_type, non_rotation_instructions) {
    EXPECT_FALSE(to_rotation_type(Instruction::LEFT).has_value());
    EXPECT_FALSE(to_rotation_type(Instruction::DOWN).has_value());
    EXPECT_FALSE(to_rotation_type(Instruction::PLACE).has_value());
}

PIECE_TEST(rotate, identity_and_full_cycle) {
    EXPECT_EQ(rotate(0, Orientation::TOP), Orientation::TOP);
    EXPECT_EQ(rotate(4, Orientation::BOTTOM), Orientation::BOTTOM);
    EXPECT_EQ(rotate(-4, Orientation::LEFT), Orientation::LEFT);
}

PIECE_TEST(rotate, step_directions) {
    EXPECT_EQ(rotate(1, Orientation::TOP), Orientation::RIGHT);
    EXPECT_EQ(rotate(-1, Orientation::TOP), Orientation::LEFT);
    EXPECT_EQ(rotate(3, Orientation::TOP), Orientation::LEFT);
}

PIECE_TEST(to_offset, move_vectors) {
    EXPECT_EQ(to_offset(MoveType::LEFT), vec2(-1, 0));
    EXPECT_EQ(to_offset(MoveType::RIGHT), vec2(1, 0));
    EXPECT_EQ(to_offset(MoveType::DOWN), vec2(0, 1));
}

PIECE_TEST(span_helpers, preserve_elements) {
    std::array<int, 3> const data{7, 8, 9};

    auto const span = to_c_span(data);
    auto const fixed_span = to_cs_span(data);

    EXPECT_EQ(span.size(), 3);
    EXPECT_EQ(fixed_span.size(), 3);
    EXPECT_RANGE_EQ(span, data);
    EXPECT_RANGE_EQ(fixed_span, data);
}

}
