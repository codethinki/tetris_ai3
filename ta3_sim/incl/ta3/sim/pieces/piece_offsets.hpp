#pragma once
#include "ta3/sim/pieces/piece_defs.hpp"

#include <cth/macro.hpp>

#include <array>
#include <span>

namespace ta3::sim {
using blocks_t = std::array<vec2, BLOCKS>;
using cblocks_view_t = std::span<vec2 const, BLOCKS>;

using o_off_t = std::array<blocks_t, *Orientation::SIZE>;



constexpr std::array<o_off_t, *PieceType::COUNT> PIECE_ORIENTATION_OFFSETS = [] {
    std::array<o_off_t, *PieceType::COUNT> out{};
    auto set = [&out](
        PieceType type,
        blocks_t const& bottom,
        blocks_t const& top,
        blocks_t const& left,
        blocks_t const& right
    ) {
        out[*type][*Orientation::TOP] = top;
        out[*type][*Orientation::RIGHT] = right;
        out[*type][*Orientation::BOTTOM] = bottom;
        out[*type][*Orientation::LEFT] = left;
    };

    set(
        PieceType::J,
        {{{0, 1}, {1, 1}, {2, 1}, {2, 2}}},
        {{{0, 0}, {0, 1}, {1, 1}, {2, 1}}},
        {{{1, 0}, {1, 1}, {0, 2}, {1, 2}}},
        {{{1, 0}, {2, 0}, {1, 1}, {1, 2}}}
    );
    set(
        PieceType::L,
        {{{0, 1}, {1, 1}, {2, 1}, {0, 2}}},
        {{{2, 0}, {0, 1}, {1, 1}, {2, 1}}},
        {{{0, 0}, {1, 0}, {1, 1}, {1, 2}}},
        {{{1, 0}, {1, 1}, {1, 2}, {2, 2}}}
    );
    set(
        PieceType::I,
        {{{0, 2}, {1, 2}, {2, 2}, {3, 2}}},
        {{{0, 1}, {1, 1}, {2, 1}, {3, 1}}},
        {{{1, 0}, {1, 1}, {1, 2}, {1, 3}}},
        {{{2, 0}, {2, 1}, {2, 2}, {2, 3}}}
    );

    set(
        PieceType::S,
        {{{1, 1}, {2, 1}, {1, 2}, {0, 2}}},
        {{{1, 0}, {2, 0}, {0, 1}, {1, 1}}},
        {{{1, 0}, {1, 1}, {2, 1}, {2, 2}}},
        {{{0, 0}, {0, 1}, {1, 1}, {1, 2}}}
    );

    set(
        PieceType::T,
        {{{0, 1}, {1, 1}, {2, 1}, {1, 2}}},
        {{{1, 0}, {0, 1}, {1, 1}, {2, 1}}},
        {{{1, 0}, {0, 1}, {1, 1}, {1, 2}}},
        {{{1, 0}, {1, 1}, {2, 1}, {1, 2}}}
    );

    set(
        PieceType::Z,
        {{{1, 1}, {0, 1}, {1, 2}, {2, 2}}},
        {{{0, 0}, {1, 0}, {1, 1}, {2, 1}}},
        {{{1, 0}, {0, 1}, {1, 1}, {0, 2}}},
        {{{2, 0}, {1, 1}, {2, 1}, {1, 2}}}
    );

    constexpr blocks_t oCoords{{{1, 0}, {2, 0}, {1, 1}, {2, 1}}};

    set(PieceType::O, oCoords, oCoords, oCoords, oCoords);

    return out;
}();

constexpr blocks_t piece_blocks(PieceType type, Orientation orientation, vec2 offset) {
    auto blocks = PIECE_ORIENTATION_OFFSETS[*type][*orientation];
    for(auto& block : blocks)
        block += offset;
    return blocks;
}

// --- v2: bit-column piece representation (matches Board2's layout) -------------------------------

/**
 * @brief one bit-column. bit @c y marks local cell occupied, bit 0 is the top row -- identical
 *  convention to Board2::column_t, so masks OR/AND straight into board columns
 */
using piece_column_t = uint32_t;
inline constexpr size_t PIECE_WIDTH = 4; // a piece sits in a 4x4 box

namespace dev {
    using box_columns_t = std::array<piece_column_t, PIECE_WIDTH>;
    using o_columns_t = std::array<box_columns_t, *Orientation::SIZE>;

    /**
     * @brief full 4-wide backing store, derived from PIECE_ORIENTATION_OFFSETS so the block
     *  coordinates stay the single source of truth
     * @note empty box columns are 0; piece_columns2 trims them off
     */
    constexpr std::array<o_columns_t, *PieceType::COUNT> PIECE_ORIENTATION_COLUMNS2 = [] {
        std::array<o_columns_t, *PieceType::COUNT> out{};
        for(size_t t = 0; t < *PieceType::COUNT; ++t)
            for(size_t o = 0; o < *Orientation::SIZE; ++o)
                for(auto const& block : PIECE_ORIENTATION_OFFSETS[t][o])
                    out[t][o][block.x] |= static_cast<piece_column_t>(piece_column_t{1} << block.y);
        return out;
    }();
} // namespace dev

/**
 * @brief a piece's occupied bit-columns and where they start
 * @note tetromino columns are always contiguous, so this is the occupied sub-range with no gaps:
 *  mask @c i lands in board column @c offset.x + left + i, shifted down by @c offset.y
 */
struct piece_columns2_t {
    int left; /**< local x of the first occupied column */
    std::span<piece_column_t const> cols; /**< the occupied columns, left to right */
};

constexpr piece_columns2_t piece_columns2(PieceType type, Orientation orientation) {
    auto const& box = dev::PIECE_ORIENTATION_COLUMNS2[*type][*orientation];

    int left = 0;
    while(left < PIECE_WIDTH && box[left] == 0)
        ++left;
    size_t right = PIECE_WIDTH;
    while(right > left && box[right - 1] == 0)
        --right;

    return {left, std::span<piece_column_t const>{box.data() + left, right - left}};
}
}
