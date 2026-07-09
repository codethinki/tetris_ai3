#pragma once
#include "ta3/sim/utility/placement.hpp"
#include "ta3/sim/tetris_engine.hpp"
#include "ta3/sim/utility/xoshiro256ss.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"

#include <cth/io/log.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace ta3::sim {

/**
 * runs multiple @ref TetrisEngine 's at once
 */
class MultiTetris {
public:
    /** @see sim::drop_place_t (lifted to its own header for device use; aliased here for the existing API) */
    using drop_place_t = sim::drop_place_t;

    /**
      * @param count games
      * @param seed to gen seeds with @ref sim::Xoshiro256ss
      */
    constexpr MultiTetris(size_t count, uint64_t seed);

    /**
     * commits one move per game, drops dead games after
     * @param moves one per game
     * @return lines cleared per game, @ref TetrisEngine::DIED if it died
     * @pre placement is legal
     */
    constexpr std::vector<size_t> next(std::span<drop_place_t const> moves);

    /**
     * @return true if all games ended
     */
    [[nodiscard]] constexpr bool empty() const;

    [[nodiscard]] constexpr std::span<TetrisEngine const> games() const { return _games; }
    [[nodiscard]] constexpr size_t size() const { return _games.size(); }

    [[nodiscard]] constexpr TetrisEngine const& operator[](size_t i) const {
        return _games[i];
    }

private:
    std::vector<TetrisEngine> _games;
};

}

namespace ta3::sim {

constexpr MultiTetris::MultiTetris(size_t count, uint64_t seed) {
    Xoshiro256ss seedGen{seed};

    _games.reserve(count);
    for(auto i = 0uz; i < count; ++i)
        _games.emplace_back(seedGen());
}

constexpr std::vector<size_t> MultiTetris::next(std::span<drop_place_t const> moves) {
    CTH_CRITICAL(
        moves.size() != _games.size(),
        "expected one move per game ({} moves, {} games)",
        moves.size(),
        _games.size()
    ) {}

    std::vector<size_t> cleared(_games.size());
    for(auto i = 0uz; i < _games.size(); ++i) {
        auto const [orientation, x, hold] = moves[i];
        if(hold)
            _games[i].hold();
        cleared[i] = _games[i].place(orientation, x);
    }

    auto const dead = std::ranges::remove_if(_games, [](TetrisEngine const& game) { return game.gameOver(); });
    _games.erase(dead.begin(), dead.end());

    return cleared;
}

constexpr bool MultiTetris::empty() const { return _games.empty(); }

}
