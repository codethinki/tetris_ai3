#pragma once

#include <ta3/sim/pieces/piece_defs.hpp>
#include <ta3/sim/utility/placement.hpp>

#include <vector>

namespace ta3::bot::jstris {

/** converts a searched straight-drop placement to Jstris actions */
[[nodiscard]] std::vector<sim::Instruction> moves_for(
    sim::PieceType piece,
    sim::drop_place_t placement
);

}
