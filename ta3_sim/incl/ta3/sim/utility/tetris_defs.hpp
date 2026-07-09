#pragma once
#include "ta3/sim/ivec2.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"

#include <array>
#include <cstddef>

namespace ta3::sim {
inline constexpr size_t ROWS = 23, COLS = 10, BOARD_SIZE = ROWS * COLS;
inline constexpr size_t WIDTH = COLS, HEIGHT = ROWS;
// current piece + 5 lookahead (matches typical online preview counts); the search needs DEPTH + 1
inline constexpr size_t PIECE_QUEUE_SIZE = 6;

using piece_coords_t = std::array<vec2, BLOCKS>;


using grid_t = std::array<BlockType, BOARD_SIZE>;

}
