#pragma once
#include "ta3/ai/model_defs.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <algorithm>
#include <array>
#include <span>
#include <type_traits>


namespace ta3::ai::metric {
struct input_t {
    size_t clearedLines;
    size_t clearedPieceCells;
    sim::Board2 const* board;
    sim::PieceType placedPiece;
    sim::PieceType heldPiece;
    size_t piecesPlaced;
    std::span<sim::PieceType const> pieceQueue;
};

}



namespace ta3::ai::metric {

constexpr auto placed_piece = [][[nodiscard]](input_t const& in) { return in.placedPiece; };

constexpr auto pieces_placed = [][[nodiscard]](input_t const& in) { return in.piecesPlaced; };

constexpr auto held_piece = [][[nodiscard]](input_t const& in) { return in.heldPiece; };
constexpr auto piece_queue = [][[nodiscard]](input_t const& in) { return in.pieceQueue; };
constexpr auto next_piece = [][[nodiscard]](input_t const& in) {
    return !in.pieceQueue.empty() ? in.pieceQueue.front() : sim::PieceType::COUNT;
};

constexpr auto lines_cleared = [][[nodiscard]](input_t const& in) { return in.clearedLines; };

constexpr auto board = [][[nodiscard]](input_t const& in) { return in.board; };

constexpr auto holes = [][[nodiscard]](input_t const& in) { return in.board->holes(); };
constexpr auto norm_holes = [][[nodiscard]](input_t const& in) { return static_cast<data_t>(holes(in)) / data_t{20}; };

constexpr auto surface_variance = [][[nodiscard]](input_t const& in) {
    int sum = 0;
    auto prevHeight = in.board->height(0);
    for(int x = 1; x < static_cast<int>(sim::WIDTH); ++x) {
        auto const h = in.board->height(x);
        int const diff = h - std::exchange(prevHeight, h);
        sum += diff * diff;
    }
    return cth::num::heron_sqrt<double>(sum);
};
constexpr auto norm_surface_var = [][[nodiscard]](input_t const& in) {
    return static_cast<data_t>(surface_variance(in)) / data_t{100};
};

constexpr auto max_height = [][[nodiscard]](input_t const& in) {
    size_t max = in.board->height(0);
    for(size_t i = 1; i < sim::WIDTH; ++i)
        max = std::max(in.board->height(i), max);
    return max;
};

constexpr auto agg_height = [][[nodiscard]](input_t const& in) {
    size_t sum = 0;
    for(size_t i = 0; i < sim::WIDTH; ++i)
        sum += in.board->height(i);
    return sum;
};
constexpr auto norm_agg_height = [][[nodiscard]](input_t const& in) {
    return static_cast<data_t>(agg_height(in)) / data_t{200};
};


/** vertical filled<->empty transitions per column, counting the floor below the field as filled */
constexpr auto y_transitions = [][[nodiscard]](input_t const& in) {
    static constexpr uint32_t FIELD = (uint32_t{1} << sim::HEIGHT) - 1; // field bits [0, HEIGHT)
    size_t sum = 0;
    for(size_t x = 0; x < sim::WIDTH; ++x) {
        // set the floor bit at HEIGHT so an empty bottom cell counts as a transition
        uint32_t const c = in.board->raw_column(x) | (uint32_t{1} << sim::HEIGHT);
        // bit y of (c ^ c>>1) marks a boundary between rows y and y+1; mask to the field
        sum += static_cast<size_t>(std::popcount((c ^ (c >> 1)) & FIELD));
    }
    return sum;
};
constexpr auto norm_y_transitions = [][[nodiscard]](input_t const& in) {
    return static_cast<data_t>(y_transitions(in)) / data_t{40};
};

constexpr auto eroded_score = [][[nodiscard]](input_t const& in) { return in.clearedLines * in.clearedPieceCells; };
constexpr auto norm_eroded_score = [][[nodiscard]](input_t const& in) {
    return static_cast<data_t>(eroded_score(in)) / data_t{16};
};


constexpr auto bumpiness = [][[nodiscard]](input_t const& in) {
    size_t sum = 0;
    auto prevHeight = in.board->height(0);
    for(size_t i = 1; i < sim::WIDTH; i++) {
        auto const h = in.board->height(i);
        sum += h > prevHeight ? h - prevHeight : prevHeight - h;
        prevHeight = h;
    }
    return sum;
};

}

namespace ta3::ai::metric::dev {

/**
     * general metric template
     * @tparam Update f(agg_t&, input_t const&) -> void
     * @tparam Initial value; agg_t = decltype(Initial)
     * @tparam Combinator f(agg_t const&, input_t const&) -> auto
     */
template<auto Update, auto Initial, auto Combinator>
struct game_metric {
    using agg_t = decltype(Initial);

    constexpr void advance(input_t const& in) { Update(agg, in); }
    constexpr auto operator()(input_t const& in) const { return Combinator(agg, in); }

    agg_t agg = Initial;
};

/**
     *
     * @tparam MoveMetric f(input_t const&) -> agg_t
     * @tparam Combinator f(game_input_t const&, agg_t) -> auto
     */
template<auto MoveMetric, auto Combinator>
using sum_metric = game_metric<
    [](auto& agg, input_t const& in) { agg += MoveMetric(in); },
    std::invoke_result_t<decltype(MoveMetric), input_t>{}, // zero of the move-metric's result type
    Combinator
>;

/** empty aggregate for game metrics that only surface a group-owned counter */
struct stateless_t {};

/**
     * sums up the MoveMetric
     */
template<auto MoveMetric>
using agg_metric = dev::sum_metric<MoveMetric, [](auto const& sum, auto const&) { return sum; }>;

/**
     * returns avg of move metric (double)
     */
template<auto MoveMetric>
using avg_metric = dev::sum_metric<MoveMetric, [](auto const& sum, auto const& in) {
    return static_cast<double>(sum) / static_cast<double>(in.piecesPlaced);
}>;
}

namespace ta3::ai::metric::dev {

using agg_max_height_t = agg_metric<max_height>;
using agg_bumpiness_t = agg_metric<bumpiness>;
using avg_max_height_t = avg_metric<max_height>;
using avg_holes_t = avg_metric<holes>;

/** total lines cleared this game */
using total_lines_cleared_t = agg_metric<lines_cleared>;



using clears_agg_t = std::array<size_t, sim::BLOCKS + 1>;

using total_clears_t = game_metric<
    [](clears_agg_t& agg, input_t const& in) { ++agg[lines_cleared(in)]; },
    clears_agg_t{},
    [](clears_agg_t const& agg, input_t const&) { return std::span{agg}; }
>;

/** holes gained over the last placement; negative when a clear freed trapped cells */
struct holes_delta_t {
    size_t prev = 0;
    size_t cur = 0;
};

using new_holes_t = game_metric<
    [](holes_delta_t& agg, input_t const& in) {
        agg.prev = agg.cur;
        agg.cur = holes(in);
    },
    holes_delta_t{},
    [](holes_delta_t const& d, input_t const&) { return static_cast<double>(d.cur) - static_cast<double>(d.prev); }
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
}
