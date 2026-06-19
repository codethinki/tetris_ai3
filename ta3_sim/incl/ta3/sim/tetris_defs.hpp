#pragma once
#include "ta3/sim/glm_defs.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"

#include <cth/macro.hpp>
#include <cth/meta/md.hpp>

#include <array>
#include <cstddef>

namespace ta3::sim {
inline constexpr size_t ROWS = 23, COLS = 10, BOARD_SIZE = ROWS * COLS;
inline constexpr size_t WIDTH = COLS, HEIGHT = ROWS;
inline constexpr size_t PIECE_QUEUE_SIZE = 5;

using board_view_t = cth::mta::mdspan_t<BlockType, HEIGHT, WIDTH>;
using cboard_view_t = cth::mta::mdspan_t<BlockType const, HEIGHT, WIDTH>;
using piece_coords_t = std::array<vec2, BLOCKS>;


using grid_t = std::array<BlockType, BOARD_SIZE>;

}
