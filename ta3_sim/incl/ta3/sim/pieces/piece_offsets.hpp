#pragma once
#include "ta3/sim/utility/bits.hpp"
#include "ta3/sim/utility/cuda_constant.hpp"
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
    TA3_CUDA_CONSTANT std::array<o_columns_t, *PieceType::COUNT> PIECE_ORIENTATION_COLUMNS2 = [] {
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

// --- nibble-packed shapes: the hot-path piece representation -------------------------------------

/**
 * @brief a piece's occupied bit-columns, nibble-packed and positioned on the board
 * @details nibble @c i of @ref cols = the occupied bits of board column @c col0 + i (same bit
 *  convention as @ref piece_column_t; a tetromino is at most 4 tall, so one column fits a nibble).
 *  nibble 0 is non-zero and trailing nibbles are 0, so the packed word alone bounds the width.
 *  replaces (PieceType, Orientation, x) triples on the hot search path: all geometry resolves to
 *  shifts of @ref cols instead of runtime-indexed walks over column arrays -- which is what keeps a
 *  register-resident @c Board2 from being demoted to local memory in device code (see the
 *  @c piece_shape overloads of @c Board2::available / @c dropDistance / @c place).
 */
struct piece_shape {
    std::uint16_t cols = 0; ///< 4 nibbles, one per occupied board column, left to right
    int col0 = 0; ///< first occupied board column (window x-origin + @c left)
};

/** occupied columns of @p cols -- valid because nibble 0 is non-zero and there are no gaps */
[[nodiscard]] constexpr int shape_width(std::uint16_t cols) { return (bit_width32(cols) + 3) / 4; }

namespace dev {

    static_assert(
        [] {
            for(auto const& piece : PIECE_ORIENTATION_COLUMNS2)
                for(auto const& box : piece)
                    for(auto const c : box)
                        if(c > 0xFu)
                            return false;
            return true;
        }(),
        "piece columns must fit one nibble (a tetromino sits in a 4-tall box)"
    );

    /** @ref piece_shape sans position: the packed columns plus the box-local x they start at */
    struct oriented_shape_t {
        std::uint16_t cols = 0;
        std::int8_t left = 0;
    };

    /** per (piece, orientation): the nibble-packed occupied columns, left-trimmed */
    TA3_CUDA_CONSTANT std::array<std::array<oriented_shape_t, *Orientation::SIZE>, *PieceType::COUNT>
        PIECE_SHAPES = [] {
            std::array<std::array<oriented_shape_t, *Orientation::SIZE>, *PieceType::COUNT> out{};
            for(size_t t = 0; t < *PieceType::COUNT; ++t)
                for(size_t o = 0; o < *Orientation::SIZE; ++o) {
                    auto const [left, cols] = piece_columns2(
                        static_cast<PieceType>(t),
                        static_cast<Orientation>(o)
                    );
                    auto& s = out[t][o];
                    s.left = static_cast<std::int8_t>(left);
                    for(size_t c = 0; c < cols.size(); ++c)
                        s.cols = static_cast<std::uint16_t>(s.cols | (cols[c] << (4 * c)));
                }
            return out;
        }();

} // namespace dev

/** the @ref piece_shape of @p type at @p orientation, positioned at window x-origin @p x */
[[nodiscard]] constexpr piece_shape piece_shape_at(PieceType type, Orientation orientation, int x) {
    auto const& s = dev::PIECE_SHAPES[*type][*orientation];
    return {s.cols, x + s.left};
}
}
