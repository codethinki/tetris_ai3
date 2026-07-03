#pragma once
#include "ta3/sim/glm_defs.hpp"
#include "ta3/sim/tetris_defs.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"
#include "ta3/sim/pieces/piece_offsets.hpp"

#include <cth/numeric.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace ta3::sim {

/**
 * occupancy-only tetris board backed by one bit-column per board column
 * @details bit @c y of a column word marks cell (column, @c y) as occupied. bit 0 is the top row,
 *  so the top-most occupied cell of a column is its @c std::countr_zero
 * @details only whole pieces can be placed. there is no per-cell mutation on purpose, the board owns
 *  all geometry
 */
class Board2 {
public:
    constexpr Board2() = default;

    /**
     * checks if @ref PieceType can be placed with @ref Orientation on @ref offset
     * @param offset place offset
     */
    [[nodiscard]] constexpr bool available(PieceType, Orientation, vec2 offset) const;

    [[nodiscard]] constexpr bool available(vec2 coord) const;
    /**
     * resolves the drop location
     * @pre @ref offset is legal (passes @ref available), so the piece always lands
     * @post offset is legal placement location
     */
    [[nodiscard]] constexpr vec2 dropLocation(PieceType, Orientation, vec2 offset) const;

    /**
     * drop places @ref PieceType , guarding the precondition of @ref dropPlace
     * @return the landing offset, or std::nullopt if @ref offset is not legal
     */
    [[nodiscard]] constexpr std::optional<vec2> optDropLocation(
        PieceType,
        Orientation,
        vec2 offset
    ) const;

    constexpr void dropPlace(PieceType, Orientation, vec2 offset);
    [[nodiscard]] constexpr bool optDropPlace(PieceType, Orientation, vec2 offset);

    /**
     * locks @ref type at @ref orientation into @ref offset
     * @pre @ref offset is @ref available
     */
    constexpr void place(PieceType, Orientation, vec2 offset);

    /**
     * clears lines
     * @return the number of cleared rows
     */
    constexpr size_t clearLines();

    /** counts the empty cells trapped below the surface across all columns */
    [[nodiscard]] constexpr size_t holes() const;


    /** number of fully occupied rows */
    [[nodiscard]] constexpr size_t fullLines() const { return static_cast<size_t>(std::popcount(fullRowsMask())); }

    /**
     * the raw bit-column of board column.
     * @details @ref x; bit @c y marks cell (@ref x, @c y) occupied 
     */
    [[nodiscard]] constexpr uint32_t raw_column(size_t x) const { return _cols[x]; }

    /**
     * stack height of column
     * @details @ref x: rows from its top-most occupied cell down, 0 when empty 
     */
    [[nodiscard]] constexpr size_t height(size_t x) const {
        return HEIGHT - static_cast<size_t>(top(static_cast<int>(x)));
    }

    /** mirror across the vertical axis (reverses the column order) */
    [[nodiscard]] constexpr Board2 vFlipped() const {
        Board2 flipped;
        for(auto x = 0uz; x < WIDTH; ++x)
            flipped._cols[x] = _cols[WIDTH - 1 - x];
        return flipped;
    }

    /** canonical mirror form: the smaller of the board and its @ref vFlipped */
    [[nodiscard]] constexpr Board2 canonical() const {
        auto const flipped = vFlipped();
        return flipped < *this ? flipped : *this;
    }

    friend constexpr bool operator==(Board2 const&, Board2 const&) = default;
    friend constexpr auto operator<=>(Board2 const&, Board2 const&) = default;

private:
    using column_t = uint32_t;
    static_assert(HEIGHT <= std::numeric_limits<column_t>::digits, "a column must fit into column_t");

    /** mask of the valid rows inside a column word */
    static constexpr column_t FIELD = (column_t{1} << HEIGHT) - 1;

    /** sentinel bit just past the field so an empty column reports a surface of @ref HEIGHT */
    static constexpr column_t FLOOR = column_t{1} << HEIGHT;

    /**
     * checks [first, first + with) in [0, WIDTH) and y in [0, HEIGHT)
     */
    [[nodiscard]] static constexpr bool offsetInBounds(int first, size_t width, int y);

    /**
     * @pre @ref offset is legal
     * @post offset.y + return is legal
     */
    [[nodiscard]] constexpr int dropDistance(piece_columns2_t const& piece, vec2 offset) const;

    /**
     * top-most occupied row of a column word
     * @details @ref column, or @ref HEIGHT when empty, bounded by [0, HEIGHT]
     */
    [[nodiscard]] static constexpr int columnTop(column_t column) { return std::countr_zero(column | FLOOR); }

    /**
     * y of the top-most occupied cell in column
     * @details @ref x, or @ref HEIGHT for an empty column
     */
    [[nodiscard]] constexpr int top(int x) const { return columnTop(_cols[x]); }

    /** empty cells trapped below the surface of column @ref x */
    [[nodiscard]] constexpr size_t holes(size_t x) const;

    /** bit @c y is set for every fully occupied row */
    [[nodiscard]] constexpr column_t fullRowsMask() const;

    /**
     * removes a single full row from @ref col and drops the survivors above it down by one
     * @param col column word to compact
     * @param row single-bit mask of the row being cleared
     * @return the compacted column
     */
    [[nodiscard]] static constexpr column_t clearLine(column_t col, column_t row);

    /**
     * removes the rows flagged in @ref full_lines from @ref col and drops the survivors down
     * @param col column word to compact
     * @param full_lines mask of the rows being cleared
     * @return the compacted column with the freed rows left empty at the top
     */
    [[nodiscard]] static constexpr column_t clearLines(column_t col, column_t full_lines);

    std::array<column_t, WIDTH> _cols{};
};

}

namespace ta3::sim {

constexpr Board2::column_t Board2::fullRowsMask() const {
    column_t mask = FIELD;
    for(auto const c : _cols)
        mask &= c;
    return mask;
}

constexpr Board2::column_t Board2::clearLine(column_t col, column_t row) {
    column_t const above = col & (row - 1); // rows above
    column_t const below = col & ~((row << 1) - 1); // rows below (not moved)
    return above << 1 | below;
}

constexpr Board2::column_t Board2::clearLines(column_t col, column_t full_lines) {
    auto const fullLineC = std::popcount(full_lines);
    [[assume(fullLineC <= static_cast<int>(BLOCKS))]];

    for(int i = 0; i < fullLineC; i++) {
        column_t const row = full_lines & (~full_lines + 1); // top-most cleared row (lowest set bit)
        col = clearLine(col, row);
        full_lines ^= row; // remaining rows are below it, positions unchanged, so just drop the bit
    }
    return col;
}

constexpr bool Board2::offsetInBounds(int first, size_t width, int y) {
    return 0 <= first && first + static_cast<int>(width) <= static_cast<int>(WIDTH)
        && 0 <= y && y < static_cast<int>(HEIGHT);
}

constexpr bool Board2::available(PieceType type, Orientation orientation, vec2 offset) const {
    auto const [left, cols] = piece_columns2(type, orientation);
    int col = offset.x + left;

    if(!offsetInBounds(col, cols.size(), offset.y))
        return false;

    for(auto const mask : cols) {
        column_t const shifted = mask << offset.y;

        // check y out of bounds for piece blocks
        if((shifted & ~FIELD) != 0)
            return false;
        // check collision
        if((_cols[col] & shifted) != 0)
            return false;
        ++col;
    }
    return true;
}
constexpr bool Board2::available(vec2 coord) const {
    column_t const shifted = column_t{1} << coord.y;
    return (_cols[coord.x] & shifted) != 0;
}

constexpr int Board2::dropDistance(piece_columns2_t const& piece, vec2 offset) const {
    int col = offset.x + piece.left;
    int minDrop = HEIGHT;
    for(auto const column : piece.cols) {
        int const low = offset.y + std::bit_width(column) - 1; // piece's lowest block in this column

        auto const mask = ~((column_t{1} << (low + 1)) - 1);
        // ignore the stack at/above that block, then count down to the first cell below it
        auto const below = _cols[col++] & mask;
        auto const drop = columnTop(below) - low - 1;
        minDrop = std::min(minDrop, drop);
    }
    return minDrop;
}

constexpr vec2 Board2::dropLocation(PieceType type, Orientation orientation, vec2 offset) const {
    auto const piece = piece_columns2(type, orientation);
    return vec2{offset.x, offset.y + dropDistance(piece, offset)};
}

constexpr std::optional<vec2> Board2::optDropLocation(PieceType type, Orientation orientation, vec2 offset) const {
    if(!available(type, orientation, offset))
        return std::nullopt;
    return dropLocation(type, orientation, offset);
}

constexpr void Board2::dropPlace(PieceType type, Orientation o, vec2 offset) {
    place(type, o, dropLocation(type, o, offset));
}
constexpr bool Board2::optDropPlace(PieceType type, Orientation o, vec2 offset) {
    auto const loc = optDropLocation(type, o, offset);

    if(!loc)
        return false;

    place(type, o, *loc);
    return true;
}


constexpr void Board2::place(PieceType type, Orientation orientation, vec2 offset) {
    CTH_CRITICAL(!available(type, orientation, offset), "place location must be available") {}

    auto const [left, cols] = piece_columns2(type, orientation);
    int col = offset.x + left;
    for(auto const mask : cols)
        _cols[col++] |= mask << offset.y;
}

constexpr size_t Board2::clearLines() {
    column_t const full = fullRowsMask();
    if(full == 0)
        return 0;

    for(auto& col : _cols)
        col = clearLines(col, full);

    return static_cast<size_t>(std::popcount(full));
}

constexpr size_t Board2::holes(size_t x) const {
    return HEIGHT - static_cast<size_t>(top(x)) - static_cast<size_t>(std::popcount(_cols[x]));
}

constexpr size_t Board2::holes() const {
    size_t sum = 0;
    for(size_t x = 0; x < WIDTH; ++x)
        sum += holes(x);
    return sum;
}

}
