#pragma once
#include "ta3/ai/model.hpp"
#include "ta3/ai/search/beam.hpp"
#include "ta3/ai/search/search.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/utility/placement.hpp>
#include <ta3/sim/tetris_engine.hpp>

#include <cstdint>

namespace ta3::ai {

/**
 * simulates a tetris game with @ref sim::TetrisEngine on the CPU
 */
class AiTetris {
public:
    /** new game, its whole piece sequence determined by @p seed */
    constexpr explicit AiTetris(uint64_t seed) : _game{seed} {}

    /** restarts the game with a fresh piece sequence from @p seed */
    constexpr void reset(uint64_t seed) {
        _game.reset(seed);
        _lastMove = {};
    }

    /**
     * search the committed state with @p model and commit the best root move.
     * @return rows cleared, or @ref sim::TetrisEngine::DIED on a game-ending move or when no move is legal.
     */
    constexpr size_t step(model_t const& model) {
        auto const searchResult = search::search_move_beam(_game, model);
        if(searchResult.none())
            return sim::TetrisEngine::DIED;

        _lastMove = searchResult.move;
        if(_lastMove.hold)
            _game.hold();
        return _game.place(_lastMove.orientation, _lastMove.x);
    }

    /** @return true once the game has topped out */
    [[nodiscard]] constexpr bool gameOver() const { return _game.gameOver(); }

    /** the backing engine, for board metrics and the upcoming-piece queue */
    [[nodiscard]] constexpr sim::TetrisEngine const& engine() const { return _game; }

    /** committed board of the game */
    [[nodiscard]] constexpr sim::Board2 const& board() const { return _game.board(); }

    /** the move committed by the last @ref step */
    [[nodiscard]] constexpr sim::drop_place_t lastMove() const { return _lastMove; }

private:
    sim::TetrisEngine _game;
    sim::drop_place_t _lastMove{};
};

}
