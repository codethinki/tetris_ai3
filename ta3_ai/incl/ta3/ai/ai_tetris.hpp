#pragma once
#include "ta3/ai/generate_variations.hpp"
#include "ta3/ai/model.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/tetris_engine.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace ta3::ai {

/**
 * runs a single persistent @ref sim::TetrisEngine, providing the inputs for @ref model_t and
 *  committing the highest rated variation each move
 * @details the single-game counterpart of @ref AiMultiTetris -- both expand a game through the same
 *  @ref generate_variations kernel; this one keeps one board alive so it can be observed, rendered
 *  (snapshot @ref board into a renderer) or driven move by move
 */
class AiTetris {
public:
    /** new game, its whole piece sequence determined by @p seed */
    constexpr explicit AiTetris(uint64_t seed);

    /** restarts the game with a fresh piece sequence from @p seed */
    constexpr void reset(uint64_t seed);

    /**
     * commits the highest rated variation and regenerates the next move's variations
     * @param variation_scores one score per entry of @ref inputs (== @ref variations)
     * @return rows cleared, or @ref sim::TetrisEngine::DIED on a game-ending move
     */
    constexpr size_t next(std::span<data_t const> variation_scores);

    /** convenience: scores @ref inputs with @p model and commits the best (see @ref next) */
    size_t step(model_t const& model) { return next(model.batchForward(inputs())); }

    /** @return true once the game has topped out */
    [[nodiscard]] constexpr bool gameOver() const { return _game.gameOver(); }

    /**
     * input buffer for the pending move
     * @details @c m::INPUTS-sized slice per variation
     */
    [[nodiscard]] constexpr std::span<data_t const> inputs() const { return _inputs; }

    /** number of variations offered for the pending move */
    [[nodiscard]] constexpr size_t variations() const { return _placements.size(); }

    /** the backing engine, for board metrics and the upcoming-piece queue */
    [[nodiscard]] constexpr sim::TetrisEngine const& engine() const { return _game; }

    /** committed board of the game */
    [[nodiscard]] constexpr sim::Board2 const& board() const { return _game.board(); }

    /** committed stats of the game */
    [[nodiscard]] constexpr tetris_stats_t const& stats() const { return _stats; }

    /** the variation committed by the last @ref next */
    [[nodiscard]] constexpr placement_t lastMove() const { return _lastMove; }

private:
    /** regenerates the pending move's variations for the current committed state */
    constexpr void regen();

    /** local index of the highest rated variation */
    [[nodiscard]] constexpr size_t bestChoice(std::span<data_t const> scores) const;

    sim::TetrisEngine _game;

    /** committed state of the game */
    tetris_stats_t _stats{};

    /** pending move's variations */
    std::vector<placement_t> _placements;

    /** flattened inputs, @c m::INPUTS values per placement */
    std::vector<data_t> _inputs;

    /** projected state per variation, parallel to @ref _placements */
    std::vector<tetris_stats_t> _variationStats;

    /** dedup scratch reused across @ref regen calls */
    std::vector<sim::Board2> _boardBuffer;

    /** variation committed by the last @ref next */
    placement_t _lastMove{};
};

}

namespace ta3::ai {

constexpr AiTetris::AiTetris(uint64_t seed) : _game{seed} { regen(); }

constexpr void AiTetris::reset(uint64_t seed) {
    _game.reset(seed);
    _stats = {};
    _lastMove = {};
    regen();
}

constexpr size_t AiTetris::next(std::span<data_t const> variation_scores) {
    auto const choice = bestChoice(variation_scores);

    // adopt the chosen variation's projected state as the new committed state
    _lastMove = _placements[choice];
    _stats = _variationStats[choice];

    if(_lastMove.hold)
        _game.hold();
    auto const cleared = _game.place(_lastMove.orientation, _lastMove.x);

    if(_game.gameOver()) {
        _placements.clear();
        _inputs.clear();
        _variationStats.clear();
    }
    else
        regen();

    return cleared;
}

constexpr void AiTetris::regen() {
    _placements.clear();
    _inputs.clear();
    _variationStats.clear();

    generate_variations(_game, _stats, _placements, _inputs, _variationStats, _boardBuffer);
}

constexpr size_t AiTetris::bestChoice(std::span<data_t const> scores) const {
    return static_cast<size_t>(std::ranges::distance(scores.begin(), std::ranges::max_element(scores)));
}

}
