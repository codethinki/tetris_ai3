#include "ta3/sim/pieces/piece.hpp"
#include "ta3/sim/glm_defs.hpp"

#include "ta3/sim/pieces/piece_offsets.hpp"


namespace ta3::sim {


Piece::Piece(Board const& tetris, PieceType type) : _tetris{&tetris}, _type{type} {
    CTH_CRITICAL(!_tetris->available(_type, _orientation, _center), "piece is constructed with illegal position") {}
}


std::optional<Piece> Piece::New(Board const& tetris, PieceType type) {
    if(!tetris.available(type, Orientation::BEGIN, START_OFFSET))
        return std::nullopt;
    return Piece{tetris, type};
}


std::optional<vec2> Piece::fitRotation(Orientation orientation) const {
    constexpr std::array<vec2, 7> fitOffsets{{{0, 0}, {0, -1}, {1, 0}, {-1, 0}, {0, -2}, {2, 0}, {-2, 0}}};

    for(auto const& offset : fitOffsets)
        if(_tetris->available(_type, orientation, offset + _center))
            return offset;

    return std::nullopt;
}


bool Piece::reorientate(Orientation orientation) {
    auto const fit = fitRotation(orientation);

    if(!fit) return false;

    _orientation = orientation;
    _center += *fit;

    return true;
}


vec2 Piece::placeTo(Board& grid) const {
    auto const offset = grid.placeOffset(_type, _orientation, _center);

    CTH_CRITICAL(!offset, "place must work") {}

    grid.place(_type, _orientation, *offset);

    return *offset;
}



vec2 Piece::dummyOffset() const {
    auto const offset = _tetris->placeOffset(_type, _orientation, _center);
    CTH_CRITICAL(!offset.has_value(), "offset must have value otherwise the piece is not valid") {}

    return *offset;
}



bool Piece::move(MoveType type) {
    auto const off = to_offset(type);


    auto const available = _tetris->available(_type, _orientation, _center + off);

    if(!available) return false;

    _center += off;
    return true;
}


}
