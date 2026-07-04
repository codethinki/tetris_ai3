#pragma once
#include "ta3/sim/ivec2.hpp"

#include <cth/enums.hpp>
#include <cth/macro.hpp>
#include <cth/numeric.hpp>
#include <cth/string/format.hpp>
#include <cth/io/log.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

namespace ta3::sim {
inline constexpr size_t BLOCKS = 4;


enum class PieceType : uint32_t {
    FIRST = 0,
    I = FIRST,
    O,
    T,
    S,
    Z,
    J,
    L,
    COUNT,
};

}

CTH_GEN_ENUM_DEREF_OVERLOAD(ta3::sim::PieceType)

namespace ta3::sim {

enum class BlockType : uint32_t {
    NONE,
    EMPTY = NONE,

    I,
    OCCUPIED = I,

    O,
    T,
    S,
    Z,
    J,
    L,

    SIZE,
    SIZE2,
    SIZE3
};

constexpr BlockType to_block_type(PieceType type) {
    if(type == PieceType::COUNT) return BlockType::SIZE;
    return static_cast<BlockType>(*type + 1);
}

}

CTH_GEN_ENUM_DEREF_OVERLOAD(ta3::sim::BlockType)

namespace ta3::sim {

enum class Instruction : uint32_t {
    LEFT,
    RIGHT,
    DOWN,
    RLEFT,
    RRIGHT,
    PLACE,
    HOLD,
    NONE,
    SIZE,
    MODEL_SIZE = NONE
};

}

CTH_GEN_ENUM_DEREF_OVERLOAD(ta3::sim::Instruction)

namespace ta3::sim {

enum class MoveType : uint32_t {
    LEFT,
    RIGHT,
    DOWN,
    SIZE
};

constexpr std::optional<MoveType> to_move_type(Instruction instruction) {
    switch(instruction) {
        case Instruction::LEFT: return MoveType::LEFT;
        case Instruction::RIGHT: return MoveType::RIGHT;
        case Instruction::DOWN: return MoveType::DOWN;
        default: return std::nullopt;
    }
}

}

CTH_GEN_ENUM_DEREF_OVERLOAD(ta3::sim::MoveType)

namespace ta3::sim {

enum class RotationType : int {
    LEFT = -1,
    RIGHT = 1,
};

constexpr std::optional<RotationType> to_rotation_type(Instruction instruction) {
    switch(instruction) {
        case Instruction::RLEFT: return RotationType::LEFT;
        case Instruction::RRIGHT: return RotationType::RIGHT;
        default: return std::nullopt;
    }
}

}

CTH_GEN_ENUM_DEREF_OVERLOAD(ta3::sim::RotationType)

namespace ta3::sim {

enum class Orientation : uint32_t {
    BEGIN,
    TOP = BEGIN,
    RIGHT,
    BOTTOM,
    LEFT,
    SIZE
};

}

CTH_GEN_ENUM_DEREF_OVERLOAD(ta3::sim::Orientation)

namespace ta3::sim {

constexpr Orientation rotate(int steps, Orientation offset) {
    return static_cast<Orientation>(cth::num::cycle<int>(*offset + steps, 0, *Orientation::SIZE));
}



constexpr vec2 to_offset(MoveType type) {
    CTH_CRITICAL(type == MoveType::SIZE, "must be a move"){}

    switch(type) {
        case MoveType::LEFT: return {-1, 0};
        case MoveType::RIGHT: return {1, 0};
        case MoveType::DOWN: return {0, 1};
        case MoveType::SIZE:
        default: std::unreachable();
    }


}

template<class T, size_t S>
std::span<T const> to_c_span(std::array<T, S> const& arr) { return std::span<T const>{arr.data(), S}; }
template<class T, size_t S>
std::span<T const, S> to_cs_span(std::array<T, S> const& arr) { return std::span<T const, S>{arr.data(), S}; }

}

namespace ta3::sim {
constexpr std::string_view to_string(Orientation e) {
    switch(e) {
        case Orientation::TOP: return "TOP";
        case Orientation::LEFT: return "LEFT";
        case Orientation::BOTTOM: return "BOTTOM";
        case Orientation::RIGHT: return "RIGHT";
        case Orientation::SIZE: return "SIZE";
        default: return "unknown";
    }
}

constexpr std::string_view to_string(Instruction e) {
    switch(e) {
        case Instruction::LEFT: return "LEFT";
        case Instruction::RIGHT: return "RIGHT";
        case Instruction::DOWN: return "DOWN";
        case Instruction::RLEFT: return "RLEFT";
        case Instruction::RRIGHT: return "RRIGHT";
        case Instruction::PLACE: return "PLACE";
        case Instruction::NONE: return "NONE";
        case Instruction::SIZE: return "SIZE";
        default: return "unknown";
    }
}

constexpr std::string_view to_string(MoveType e) {
    switch(e) {
        case MoveType::LEFT: return "LEFT";
        case MoveType::RIGHT: return "RIGHT";
        case MoveType::DOWN: return "DOWN";
        case MoveType::SIZE: return "SIZE";
        default: return "unknown";
    }
}

constexpr std::string_view to_string(BlockType e) {
    switch(e) {
        case BlockType::I: return "I";
        case BlockType::O: return "O";
        case BlockType::T: return "T";
        case BlockType::S: return "S";
        case BlockType::Z: return "Z";
        case BlockType::J: return "J";
        case BlockType::L: return "L";
        case BlockType::NONE: return "NONE";
        case BlockType::SIZE: return "SIZE";
        default: return "unknown";
    }
}

constexpr std::string_view to_string(PieceType e) {
    switch(e) {
        case PieceType::I: return "I";
        case PieceType::O: return "O";
        case PieceType::T: return "T";
        case PieceType::S: return "S";
        case PieceType::Z: return "Z";
        case PieceType::J: return "J";
        case PieceType::L: return "L";
        case PieceType::COUNT: return "COUNT";
        default: return "unknown";
    }
}

}

CTH_FORMAT_CLASS(
    ta3::sim::Orientation,
    "{}",
    ([](auto const& v) { return ta3::sim::to_string(v); })
);
CTH_FORMAT_CLASS(
    ta3::sim::MoveType,
    "{}",
    ([](auto const& v) { return ta3::sim::to_string(v); })
);
CTH_FORMAT_CLASS(
    ta3::sim::BlockType,
    "{}",
    ([](auto const& v) { return ta3::sim::to_string(v); })
);
CTH_FORMAT_CLASS(
    ta3::sim::PieceType,
    "{}",
    ([](auto const& v) { return ta3::sim::to_string(v); })
);
CTH_FORMAT_CLASS(
    ta3::sim::Instruction,
    "{}",
    ([](auto const& v) { return ta3::sim::to_string(v); })
);
