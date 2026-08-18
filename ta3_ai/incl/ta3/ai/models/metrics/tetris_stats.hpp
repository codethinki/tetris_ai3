#pragma once
#include "metrics.hpp"

#include <tuple>
#include <type_traits>

namespace ta3::ai::metric {

/**
 * @tparam Score constexpr score metric (invoked with the whole aggregate)
 * @tparam StatefulMetrics constexpr metric values to track (those with an advance)
 */
template<auto Score, auto... StatefulMetrics> requires(
    std::is_base_of_v<dev::stateful_metric_base, decltype(StatefulMetrics)> && ...
)
class tetris_stats {
public:
    /** fold one committed placement into every tracked metric and cache the board for later reads */
    constexpr void advance(sim::Board2 const& board, uint32_t lines_cleared) {
        ++_piecesPlaced;
        _board = board;
        std::apply([&](auto&... m) { (m.advance(board, lines_cleared), ...); }, _metrics);
    }

    /**
     * @param metric a constexpr metric value
     */
    template<class Metric>
    [[nodiscard]] constexpr auto get(Metric const& metric) const {
        if constexpr(std::is_base_of_v<dev::stateful_metric_base, Metric>)
            return std::get<Metric>(_metrics).eval(_piecesPlaced); // tracked game metric
        else if constexpr(requires { metric(*this); })
            return metric(*this); // score metric: reads the whole block
        else
            return metric(_board); // stateless board metric on the last committed board
    }
    [[nodiscard]] constexpr auto score() const { return this->get(Score); }

    [[nodiscard]] constexpr uint32_t piecesPlaced() const { return _piecesPlaced; }

private:
    sim::Board2 _board{};
    uint32_t _piecesPlaced = 0;
    std::tuple<std::remove_const_t<decltype(StatefulMetrics)>...> _metrics{};
};

}
