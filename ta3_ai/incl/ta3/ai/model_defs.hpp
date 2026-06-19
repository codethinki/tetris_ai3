#pragma once

#include <ta3/sim/tetris_defs.hpp>

#include <array>

namespace ta3::ai {

using data_t = float;
using board_t = std::array<data_t, sim::BOARD_SIZE>;
using board_view_t = cth::mta::mdspan_t<data_t, sim::HEIGHT, sim::WIDTH>;
using cboard_view_t = cth::mta::mdspan_t<data_t const, sim::HEIGHT, sim::WIDTH>;

}
