#pragma once
#include "ta3/sim/tetris_defs.hpp"
#include "ta3/sim/glm_defs.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"
#include "ta3/sim/pieces/piece_offsets.hpp"
#include "ta3/sim/board.hpp"

#include <array>
#include <optional>
#include <span>

namespace ta3::sim {

class Piece {
public:
    static constexpr vec2 START_OFFSET{3, 0};


    static constexpr std::array<fvec3, 7> PIECE_COLORS{{
        {0.5, 0, 1},
        {1, 0.5, 0},
        {0, 0, 1},
        {0, 1, 1},
        {0, 1, 0},
        {1, 0, 0},
        {1, 1, 0}
    }};

    using o_off_view = std::span<vec2 const, BLOCKS>;

    Piece(Board const& tetris, PieceType type);
    static std::optional<Piece> New(Board const& tetris, PieceType type);

    ~Piece() = default;



    bool move(MoveType type);
    bool reorientate(Orientation orientation);
    bool rotate(int steps) { return reorientate(ta3::sim::rotate(steps, _orientation)); }

    vec2 placeTo(Board& grid) const;

private:
    [[nodiscard]] piece_coords_t offsetCoords(vec2 offset) const { return piece_blocks(_type, _orientation, offset); }
    [[nodiscard]] std::optional<vec2> fitRotation(Orientation orientation) const;



    Board const* _tetris;

    PieceType _type;

    vec2 _center{START_OFFSET};
    Orientation _orientation = Orientation::TOP;

public:
    [[nodiscard]] vec2 dummyOffset() const;
    [[nodiscard]] declauto blocks() const { return offsetCoords(_center); }
    [[nodiscard]] declauto dummyBlocks() const { return offsetCoords(dummyOffset()); }
    [[nodiscard]] auto type() const { return _type; }
    [[nodiscard]] Orientation orientation() const { return _orientation; }
    [[nodiscard]] auto offset() const { return _center; }
};


}
