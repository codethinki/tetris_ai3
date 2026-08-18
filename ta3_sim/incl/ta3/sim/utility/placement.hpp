#pragma once
#include "ta3/sim/pieces/piece_defs.hpp"

namespace ta3::sim {

/**
 * holds drop place move data
 */
struct drop_place_t {
    Orientation orientation{};
    int x{};

    /** whether the move swapped the hold slot first (@ref TetrisEngine::hold) */
    bool hold = false;

    friend constexpr bool operator==(drop_place_t const&, drop_place_t const&) = default;
};

}
