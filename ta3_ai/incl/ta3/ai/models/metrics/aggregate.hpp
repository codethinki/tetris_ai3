#pragma once
#include "metrics.hpp"

namespace ta3::ai::metric {

/**
 * @tparam Score constexpr score metric (invoked with the whole aggregate)
 * @tparam GameMetrics constexpr metric values to track (those with an advance)
 */
template<auto Score, auto... GameMetrics>
class aggregate {
public:
    /** fold one placement into every game metric and cache it for later reads */
    constexpr void advance(
        size_t cleared,
        sim::Board2 const& board,
        sim::PieceType placed,
        sim::PieceType held,
        std::span<sim::PieceType const> piece_queue = {},
        size_t cleared_piece_cells = 0
    ) {
        _in = input_t{cleared, cleared_piece_cells, &board, placed, held, ++_piecesPlaced, piece_queue};
        std::apply([this](auto&... m) { (m.advance(_in), ...); }, _metrics);
    }

    /**
     * @param metric a constexpr metric value
     */
    template<class Metric>
    [[nodiscard]] constexpr auto get(Metric const& metric) const {
        if constexpr(requires(Metric m, input_t const& in) { m.advance(in); })
            return std::get<Metric>(_metrics)(_in); // stateful game metric
        else if constexpr(requires(Metric m) { m(*this); })
            return metric(*this); // score metric: reads the whole block
        else
            return metric(_in); // stateless move metric
    }

private:
    input_t _in{};
    size_t _piecesPlaced = 0;
    std::tuple<decltype(GameMetrics)...> _metrics{};
};

/**
 * an aggregate plus a score, exposed through score()
 * @tparam Score constexpr score metric; reads the whole block via get
 * @tparam GameStats the stateful metrics to track
 */
template<auto Score, auto... GameStats>
class stats : public aggregate<GameStats...> {
public:
    [[nodiscard]] constexpr auto score() const { return this->get(Score); }
};
}
