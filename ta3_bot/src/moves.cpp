#include "ta3/bot/moves.hpp"

#include <cth/io/log.hpp>
#include <ta3/sim/pieces/piece_offsets.hpp>
#include <ta3/sim/utility/tetris_defs.hpp>

#include <cstdlib>

namespace ta3::bot::jstris {

std::vector<sim::Instruction> moves_for(sim::PieceType piece, sim::drop_place_t placement) {
    std::vector<sim::Instruction> result;
    result.reserve(8);

    if(placement.hold)
        result.push_back(sim::Instruction::HOLD);

    switch(placement.orientation) {
        case sim::Orientation::TOP: break;
        case sim::Orientation::RIGHT:
            result.push_back(sim::Instruction::RRIGHT);
            break;
        case sim::Orientation::BOTTOM:
            result.push_back(sim::Instruction::RRIGHT);
            result.push_back(sim::Instruction::RRIGHT);
            break;
        case sim::Orientation::LEFT:
            result.push_back(sim::Instruction::RLEFT);
            break;
        case sim::Orientation::SIZE:
            CTH_CRITICAL(true, "invalid searched orientation") {}
            break;
    }

    constexpr auto spawnX = static_cast<int>((sim::WIDTH - sim::PIECE_WIDTH) / 2);
    auto targetX = placement.x;
    if(piece == sim::PieceType::S) {
        if(placement.orientation == sim::Orientation::RIGHT)
            --targetX;
        else if(placement.orientation == sim::Orientation::LEFT)
            ++targetX;
    }

    auto const offset = targetX - spawnX;
    auto const translation = offset < 0 ? sim::Instruction::LEFT : sim::Instruction::RIGHT;
    for(auto i = 0; i < std::abs(offset); ++i)
        result.push_back(translation);

    result.push_back(sim::Instruction::PLACE);
    return result;
}

}
