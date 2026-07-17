#pragma once
#include "ta3/ai/model_defs.hpp" // clear_hist_t

#include <ta3/sim/utility/cuda_constant.hpp>

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>
#include <ta3/sim/pieces/piece_offsets.hpp>
#include <ta3/sim/utility/placement.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

/**
 * @file placements.hpp
 * @brief board-independent placement enumeration for the GPU search.
 *
 * a piece's legal drop columns split into two kinds of constraint:
 *  - **theoretical bounds** -- for a given @c (piece, orientation) the window x-origin ranges over
 *    @c [xMin, xMax], derived purely from @ref sim::piece_columns2 (the piece's occupied column span).
 *    board-independent, so it is a pure @c constexpr index space (@ref theo_count / @ref nth_placement).
 *  - **collision** -- whether that in-range placement actually fits the current board. board-dependent,
 *    checked at replay time via @ref sim::Board2::available.
 *
 * these bounds are precomputed once, at compile time, into a flat @b sparse-encoded table
 * (@ref dev::PLACEMENTS): every legal @c (orientation, x) of every piece packed to one byte, laid out
 * back-to-back and sliced per piece by @ref dev::PLACEMENT_OFFSET. so the search just loops
 * @c [0, theo_count(piece)) and decodes -- no per-call orientation arithmetic.
 *
 * geometrically identical orientations are @b deduped (O's four squares collapse to one, the two
 * horizontals / two verticals of I / S / Z collapse to one each), keyed by the vertically-normalised
 * occupied-column shape. a single representative orientation is kept per shape; the dropped ones produce
 * identical resting boards, so this only removes redundant leaves (an @c atomicMax over equal values is a
 * no-op). @note the sole behavioural edge: a dropped vertical-translate could differ from its
 * representative in @c available() at spawn on a near-topped-out board; irrelevant unless the stack
 * reaches the top rows.
 *
 * everything here is @c constexpr and device-clean (no allocation, no std::vector).
 */
namespace ta3::ai::search {

using clear_t = ai::clear_hist_t;
using value_t = float;

/**
 * a concrete drop: an orientation and the window x-origin (@c Board2 offset.x, y implied 0), plus the
 * resolved @ref sim::piece_shape -- the hot path (@c apply) reads only @c shape; the commit path reads
 * @c orientation / @c x. dead fields are DCE'd per call site.
 */
struct placement {
    sim::Orientation orientation{};
    int x{};
    sim::piece_shape shape{};
};

/**
 * the inclusive x-origin range of @p piece in @p orientation, board-independent.
 * @details mirrors the loop bounds in the legacy @c for_each_placement / @c generate_variations_ctx:
 *  @c xMin = -left, @c xMax = WIDTH - left - cols.size(). @c count = xMax - xMin + 1.
 */
struct ori_range {
    int xMin{};
    int xMax{};
    [[nodiscard]] constexpr std::uint32_t count() const {
        return static_cast<std::uint32_t>(xMax - xMin + 1);
    }
};

[[nodiscard]] constexpr ori_range orientation_range(sim::PieceType piece, sim::Orientation orientation) {
    auto const [left, cols] = sim::piece_columns2(piece, orientation);
    auto const xMin = -left;
    auto const xMax = static_cast<int>(sim::WIDTH) - left - static_cast<int>(cols.size());
    return {xMin, xMax};
}

namespace dev {

    /** x-origin bias so a packed entry stays non-negative (observed x in @c [-2, 8]). */
    inline constexpr int X_BIAS = 2;

    /** pack @c (orientation, x) into one byte: bits @c [0,1] orientation, bits @c [2,5] biased x. */
    [[nodiscard]] constexpr std::uint8_t encode(sim::Orientation orientation, int x) {
        return static_cast<std::uint8_t>((static_cast<std::uint32_t>(*orientation) & 0x3u)
            | ((static_cast<std::uint32_t>(x + X_BIAS) & 0xFu) << 2));
    }

    [[nodiscard]] constexpr placement decode(sim::PieceType piece, std::uint8_t byte) {
        auto const orientation = static_cast<sim::Orientation>(byte & 0x3u);
        auto const x = static_cast<int>((byte >> 2) & 0xFu) - X_BIAS;
        return {orientation, x, sim::piece_shape_at(piece, orientation, x)};
    }

    /** vertically-normalised occupied-column shape -- the drop-equivalence key for orientation dedup. */
    struct shape_sig {
        std::array<sim::piece_column_t, sim::PIECE_WIDTH> cols{};
        std::uint32_t n = 0;
    };

    [[nodiscard]] constexpr std::uint32_t ctz(sim::piece_column_t v) {
        std::uint32_t b = 0;
        while(((v >> b) & 1u) == 0)
            ++b;
        return b;
    }

    [[nodiscard]] constexpr shape_sig signature(sim::PieceType piece, sim::Orientation orientation) {
        auto const pc = sim::piece_columns2(piece, orientation);

        std::uint32_t minbit = 32;
        for(auto const c : pc.cols)
            if(c != 0) {
                auto const b = ctz(c);
                if(b < minbit)
                    minbit = b;
            }
        if(minbit == 32)
            minbit = 0;

        shape_sig s{};
        s.n = static_cast<std::uint32_t>(pc.cols.size());
        for(std::uint32_t i = 0; i < s.n; ++i)
            s.cols[i] = static_cast<sim::piece_column_t>(pc.cols[i] >> minbit);
        return s;
    }

    [[nodiscard]] constexpr bool same_sig(shape_sig const& a, shape_sig const& b) {
        if(a.n != b.n)
            return false;
        for(std::uint32_t i = 0; i < a.n; ++i)
            if(a.cols[i] != b.cols[i])
                return false;
        return true;
    }

    /** the distinct (deduped) orientations of @p piece, lowest enum index kept as representative. */
    struct rep_list {
        std::array<sim::Orientation, *sim::Orientation::SIZE> ori{};
        std::uint32_t n = 0;
    };

    [[nodiscard]] constexpr rep_list representatives(sim::PieceType piece) {
        rep_list reps{};
        std::array<shape_sig, *sim::Orientation::SIZE> seen{};
        std::uint32_t seenN = 0;

        for(std::uint32_t o = 0; o < *sim::Orientation::SIZE; ++o) {
            auto const orientation = static_cast<sim::Orientation>(o);
            auto const sg = signature(piece, orientation);

            auto dup = false;
            for(std::uint32_t j = 0; j < seenN; ++j)
                if(same_sig(seen[j], sg)) {
                    dup = true;
                    break;
                }
            if(dup)
                continue;

            seen[seenN++] = sg;
            reps.ori[reps.n++] = orientation;
        }
        return reps;
    }

    /** deduped placement count of @p piece (compile-time; runtime uses @ref PLACEMENT_OFFSET). */
    [[nodiscard]] constexpr std::uint32_t piece_count(sim::PieceType piece) {
        auto const reps = representatives(piece);
        std::uint32_t n = 0;
        for(std::uint32_t i = 0; i < reps.n; ++i)
            n += orientation_range(piece, reps.ori[i]).count();
        return n;
    }

    /** total packed entries across all pieces -- the exact size of @ref PLACEMENTS. */
    inline constexpr std::uint32_t TOTAL = [] {
        std::uint32_t n = 0;
        for(std::uint32_t p = 0; p < *sim::PieceType::COUNT; ++p)
            n += piece_count(static_cast<sim::PieceType>(p));
        return n;
    }();

    struct lut {
        std::array<std::uint8_t, TOTAL> data{};
        std::array<std::uint32_t, *sim::PieceType::COUNT + 1> offset{};
    };

    /** build the flat sparse table + per-piece offsets, deduping orientations by shape. */
    TA3_CUDA_CONSTANT lut PLACEMENT_LUT = [] {
        lut out{};
        std::uint32_t w = 0;
        for(std::uint32_t p = 0; p < *sim::PieceType::COUNT; ++p) {
            out.offset[p] = w;
            auto const piece = static_cast<sim::PieceType>(p);
            auto const reps = representatives(piece);
            for(std::uint32_t i = 0; i < reps.n; ++i) {
                auto const orientation = reps.ori[i];
                auto const range = orientation_range(piece, orientation);
                for(auto x = range.xMin; x <= range.xMax; ++x)
                    out.data[w++] = encode(orientation, x);
            }
        }
        out.offset[*sim::PieceType::COUNT] = w;
        return out;
    }();

} // namespace dev

/** number of (deduped) theoretical placements of @p piece -- the loop bound for the search. */
[[nodiscard]] constexpr std::uint32_t n_theoretical_placements(sim::PieceType piece) {
    return dev::PLACEMENT_LUT.offset[*piece + 1] - dev::PLACEMENT_LUT.offset[*piece];
}

/**
 * decode the @p i-th theoretical placement of @p piece into @c (orientation, x).
 * @pre @c i < @ref theo_count(piece)
 */
[[nodiscard]] constexpr placement nth_placement(sim::PieceType piece, std::uint32_t i) {
    return dev::decode(piece, dev::PLACEMENT_LUT.data[dev::PLACEMENT_LUT.offset[*piece] + i]);
}

/** upper bound on placements of any single piece -- the tightest static cap over all piece types. */
inline constexpr std::uint32_t MAX_PLACEMENTS = [] {
    std::uint32_t m = 0;
    for(std::uint32_t typeIdx = 0; typeIdx < *sim::PieceType::COUNT; ++typeIdx) {
        auto const c = n_theoretical_placements(static_cast<sim::PieceType>(typeIdx));
        if(c > m)
            m = c;
    }
    return m;
}();

} // namespace ta3::ai::search
