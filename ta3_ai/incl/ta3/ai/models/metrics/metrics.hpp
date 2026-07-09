#pragma once
#include <ta3/sim/utility/bits.hpp>
#include <ta3/sim/utility/cuda_constant.hpp>
#include "ta3/ai/model_defs.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <type_traits>





namespace ta3::ai::metric {

/** @attention TA3_CUDA_CONSTANT: indexed from device code with a runtime value (see @ref ai::CLEAR_BUMP) */
TA3_CUDA_CONSTANT std::array<int, sim::BLOCKS + 1> CLEAR_SCORE_TABLE{{0, -1, -2, -3, 10}};


constexpr auto board = [][[nodiscard]](sim::Board2 const& board) { return board; };

constexpr auto holes = [][[nodiscard]](sim::Board2 const& board) { return board.holes(); };
constexpr auto norm_holes = [][[nodiscard]](sim::Board2 const& board) { return static_cast<data_t>(holes(board)) / data_t{20}; };

namespace dev {
    /** bit-plane masks: @c PLANE[b] has bit @c y set iff bit @c b of @c y is set, over the field rows */
    TA3_CUDA_CONSTANT auto HOLE_DEPTH_PLANES = [] {
        constexpr size_t B = std::bit_width(static_cast<unsigned>(sim::HEIGHT - 1));
        std::array<uint32_t, B> planes{};
        for(size_t b = 0; b < B; ++b)
            for(uint32_t y = 0; y < sim::HEIGHT; ++y)
                if((y >> b) & 1u)
                    planes[b] |= uint32_t{1} << y;
        return planes;
    }();

    /** sum of the indices of set bits board a column, via one @c popcount per bit-plane (no per-row scan) */
    [[nodiscard]] constexpr size_t sum_set_bit_indices(uint32_t c) {
        size_t s = 0;
        for(size_t b = 0; b < HOLE_DEPTH_PLANES.size(); ++b)
            s += (size_t{1} << b) * static_cast<size_t>(sim::popcount(c & HOLE_DEPTH_PLANES[b]));
        return s;
    }
} // namespace dev


constexpr auto hole_depths = [][[nodiscard]](sim::Board2 const& board) {
    constexpr size_t H = sim::HEIGHT;
    size_t sum = 0;
    for(size_t x = 0; x < sim::WIDTH; ++x) {
        auto const c = board.raw_column(x);
        size_t const n = static_cast<size_t>(sim::popcount(c));
        size_t const s = dev::sum_set_bit_indices(c);

        size_t const pos = (H - 1) * n + n * (n + 1) / 2; // grouped so the subtraction stays >= 0
        size_t const neg = s + n * n;
        sum += pos - neg;
    }
    return sum;
};

constexpr auto norm_hole_depths = [][[nodiscard]](sim::Board2 const& board) {
    return static_cast<data_t>(hole_depths(board)) / data_t{200};
};

/**
 * device-clean constexpr sqrt. at runtime (host and device) this is a single correctly-rounded hardware
 * sqrtf; heron / newton fallback for constexpr
 */
[[nodiscard]] constexpr float fsqrt(float x) {
    if(x <= 0)
        return 0;
    if(std::is_constant_evaluated()) {
        float c = x, p = 0;
        for(int i = 0; i < 32 && c != p; ++i) {
            p = c;
            c = 0.5f * (c + x / c);
        }
        return c;
    }
    return std::sqrt(x); // nvcc maps this to sqrtf in __device__ code; libm on the host
}

constexpr auto surface_variance = [][[nodiscard]](sim::Board2 const& board) {
    int sum = 0;
    auto prevHeight = board.height(0);
    for(int x = 1; x < static_cast<int>(sim::WIDTH); ++x) {
        auto const h = board.height(x);
        int const diff = h - std::exchange(prevHeight, h);
        sum += diff * diff;
    }
    return fsqrt(static_cast<float>(sum));
};
constexpr auto norm_surface_var = [][[nodiscard]](sim::Board2 const& board) {
    return static_cast<data_t>(surface_variance(board)) / data_t{100};
};

constexpr auto max_height = [][[nodiscard]](sim::Board2 const& board) {
    size_t max = board.height(0);
    for(size_t i = 1; i < sim::WIDTH; ++i)
        max = std::max(board.height(i), max);
    return max;
};

constexpr auto agg_height = [][[nodiscard]](sim::Board2 const& board) {
    size_t sum = 0;
    for(size_t i = 0; i < sim::WIDTH; ++i)
        sum += board.height(i);
    return sum;
};
constexpr auto norm_agg_height = [][[nodiscard]](sim::Board2 const& board) {
    return static_cast<data_t>(agg_height(board)) / data_t{200};
};


/** vertical filled<->empty transitions per column, counting the floor below the field as filled */
constexpr auto y_transitions = [][[nodiscard]](sim::Board2 const& board) {
    static constexpr uint32_t FIELD = (uint32_t{1} << sim::HEIGHT) - 1; // field bits [0, HEIGHT)
    size_t sum = 0;
    for(size_t x = 0; x < sim::WIDTH; ++x) {
        // set the floor bit at HEIGHT so an empty bottom cell counts as a transition
        uint32_t const c = board.raw_column(x) | (uint32_t{1} << sim::HEIGHT);
        // bit y of (c ^ c>>1) marks a boundary between rows y and y+1; mask to the field
        sum += static_cast<size_t>(sim::popcount((c ^ (c >> 1)) & FIELD));
    }
    return sum;
};
constexpr auto norm_y_transitions = [][[nodiscard]](sim::Board2 const& board) {
    return static_cast<data_t>(y_transitions(board)) / data_t{40};
};


constexpr auto bumpiness = [][[nodiscard]](sim::Board2 const& board) {
    size_t sum = 0;
    auto prevHeight = board.height(0);
    for(size_t i = 1; i < sim::WIDTH; i++) {
        auto const h = board.height(i);
        sum += h > prevHeight ? h - prevHeight : prevHeight - h;
        prevHeight = h;
    }
    return sum;
};

}

namespace ta3::ai::metric {

}

namespace ta3::ai::metric::dev {

/**
 * @attention do not use, inherit from @ref stateful_metric
 */
struct stateful_metric_base {};

/**
     * general metric template
     * @tparam Update f(agg_t&, input_t const&) -> void
     * @tparam Initial value; agg_t = decltype(Initial)
     * @tparam Combinator f(agg_t const&, input_t const&) -> auto
     */
template<auto Update, auto Initial, auto Combinator>
struct stateful_metric : stateful_metric_base {
    // remove_const: decltype of a class-type NTTP object is const-qualified per the standard (strict on
    // gcc/clang; MSVC/nvcc are lenient) -- the aggregate must stay mutable for advance().
    using agg_t = std::remove_const_t<decltype(Initial)>;

    constexpr void advance(sim::Board2 const& board, uint32_t lines_cleared) { Update(agg, board, lines_cleared); }
    constexpr auto eval(uint32_t pieces_placed) const { return Combinator(agg, pieces_placed); }

    agg_t agg = Initial;
};

/**
     * per-move summing metric
     * @tparam MoveMetric f(Board const& board, uint32_t lines_cleared) -> agg_t
     * @tparam Combinator f(game_input_t const&, agg_t) -> auto
     * @note a struct, not a @ref stateful_metric alias: nvcc's front end loses an alias-template's
     *  parameter inside lambdas written in the alias ("identifier MoveMetric is undefined"), so the
     *  lambda body lives in a member function instead. semantics identical; required since the stat
     *  block is committed on-device by the eval kernel.
     */
template<auto MoveMetric, auto Combinator>
struct sum_board_metric : stateful_metric_base {
    using agg_t = std::invoke_result_t<decltype(MoveMetric), sim::Board2 const&, uint32_t>;

    constexpr void advance(sim::Board2 const& board, uint32_t lines_cleared) {
        agg += MoveMetric(board, lines_cleared);
    }
    constexpr auto eval(uint32_t pieces_placed) const { return Combinator(agg, pieces_placed); }

    agg_t agg{}; // zero of the move-metric's result type
};

/**
     * sums up the MoveMetric
     */
template<auto MoveMetric>
using agg_metric = dev::sum_board_metric<MoveMetric, [](auto const& sum, uint32_t) { return sum; }>;

/**
     * returns avg of move metric (double)
     */
template<auto MoveMetric>
using avg_metric = dev::sum_board_metric<MoveMetric, [](auto const& sum, uint32_t pieces_placed) {
    return static_cast<double>(sum) / static_cast<double>(pieces_placed);
}>;
}

namespace ta3::ai::metric::dev {

// the board metrics above are f(board); sum_board_metric feeds MoveMetric(board, lines_cleared),
// so lift each into a move metric that ignores lines_cleared.
using agg_max_height_t  = agg_metric<[](sim::Board2 const& b, uint32_t) { return max_height(b); }>;
using agg_bumpiness_t   = agg_metric<[](sim::Board2 const& b, uint32_t) { return bumpiness(b); }>;
using avg_max_height_t  = avg_metric<[](sim::Board2 const& b, uint32_t) { return max_height(b); }>;
using avg_holes_t       = avg_metric<[](sim::Board2 const& b, uint32_t) { return holes(b); }>;
using avg_hole_depths_t = avg_metric<[](sim::Board2 const& b, uint32_t) { return hole_depths(b); }>;

// total lines cleared over the game -- the move contribution is lines_cleared itself.
using total_lines_cleared_t = agg_metric<[](sim::Board2 const&, uint32_t lines_cleared) {
    return static_cast<size_t>(lines_cleared);
}>;


using clears_agg_t = std::array<size_t, sim::BLOCKS + 1>;

using total_clears_t = stateful_metric<
    [](clears_agg_t& agg, sim::Board2 const&, uint32_t lines_cleared) { ++agg[lines_cleared]; },
    clears_agg_t{},
    [](clears_agg_t const& agg, uint32_t) { return std::span{agg}; }
>;

/** holes gained over the last placement; negative when a clear freed trapped cells */
struct holes_delta_t {
    size_t prev = 0;
    size_t cur = 0;
};

using new_holes_t = stateful_metric<
    [](holes_delta_t& agg, sim::Board2 const& board, uint32_t) {
        agg.prev = agg.cur;
        agg.cur = holes(board);
    },
    holes_delta_t{},
    [](holes_delta_t const& d, uint32_t) { return static_cast<double>(d.cur) - static_cast<double>(d.prev); }
>;
}

namespace ta3::ai::metric {
constexpr dev::agg_max_height_t agg_max_height{};
constexpr dev::agg_bumpiness_t agg_bumpiness{};
constexpr dev::avg_max_height_t avg_max_height{};
constexpr dev::total_lines_cleared_t total_lines_cleared{};
constexpr dev::total_clears_t total_clears{};
constexpr dev::new_holes_t new_holes{};
constexpr dev::avg_holes_t avg_holes{};
constexpr dev::avg_hole_depths_t avg_hole_depths{};
}
