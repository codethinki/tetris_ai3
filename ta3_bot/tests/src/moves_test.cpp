#include "test.hpp"

#include "ta3/bot/moves.hpp"

#include <vector>

namespace ta3::bot {

BOT_TEST(moves, converts_placement) {
    EXPECT_EQ(
        jstris::moves_for(sim::PieceType::I, {sim::Orientation::LEFT, 1, true}),
        (std::vector{
            sim::Instruction::HOLD,
            sim::Instruction::RLEFT,
            sim::Instruction::LEFT,
            sim::Instruction::LEFT,
            sim::Instruction::PLACE
        })
    );
}

BOT_TEST(moves, uses_shortest_bottom_rotation) {
    EXPECT_EQ(
        jstris::moves_for(sim::PieceType::I, {sim::Orientation::BOTTOM, 5, false}),
        (std::vector{
            sim::Instruction::RRIGHT,
            sim::Instruction::RRIGHT,
            sim::Instruction::RIGHT,
            sim::Instruction::RIGHT,
            sim::Instruction::PLACE
        })
    );
}

BOT_TEST(moves, adjusts_vertical_s_origin_for_jstris) {
    EXPECT_EQ(
        jstris::moves_for(sim::PieceType::S, {sim::Orientation::RIGHT, 2, false}),
        (std::vector{
            sim::Instruction::RRIGHT,
            sim::Instruction::LEFT,
            sim::Instruction::LEFT,
            sim::Instruction::PLACE
        })
    );
}

}
