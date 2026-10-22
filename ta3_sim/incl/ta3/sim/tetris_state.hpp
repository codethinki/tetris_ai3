#pragma once
#include "ta3/sim/board2.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"
#include "ta3/sim/utility/tetris_defs.hpp"

#include <array>

namespace ta3::sim {

/**
 * board and piece information required to search one move.
 */
struct TetrisState {
    Board2 board{};
    PieceType current = NO_PIECE;
    PieceType held = NO_PIECE;
    std::array<PieceType, PIECE_QUEUE_SIZE - 1> lookahead{};
    bool gameOver = false;
};

}
