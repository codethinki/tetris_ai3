#pragma once
#include "ta3/ai/model.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/tetris_engine.hpp>
#include <ta3/sim/tetris_defs.hpp>
#include <ta3/sim/utility.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>
#include <ta3/sim/pieces/piece_offsets.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace ta3::ai {

/**
 * @brief a batch of independent TetrisEngine games expanded one placement-ply at a time for the AI
 * @note on each ply every live game enumerates its legal placements ("variations"); each variation's
 *  resulting board is encoded into one model-input slice of the shared @ref inputs buffer. the caller
 *  scores the buffer, picks one variation per game, and commits it with @ref step
 * @note model-agnostic through the @c model_t alias: the input layout and its stride come solely from
 *  @c model_t::INPUTS / @c model_t::extractInputs
 */
class MultiTetris2 {
public:
    /** a candidate hard-drop placement: the current piece's orientation and 4x4 window x-origin */
    struct placement_t {
        sim::Orientation orientation;
        int x;
    };

    /** @brief starts @ref count games, each seeded off @ref seed so the whole batch reproduces */
    constexpr MultiTetris2(size_t count, uint64_t seed);

    /**
     * @brief enumerates every legal placement of each live game's current piece and encodes the
     *  resulting board into that variation's slice of @ref inputs
     * @return the total number of variations across all games (== input slices written)
     * @post @ref inputs and the per-game @ref placements describe this ply until the next @ref genInputs
     */
    constexpr size_t genInputs();

    /**
     * @brief commits one chosen variation per game, advancing each live game by that placement
     * @param choices per game, the index into its @ref placements range to play; the entry for a
     *  finished game is ignored
     * @pre @ref genInputs has run since the last @ref step
     */
    constexpr void step(std::span<size_t const> choices);

    /** @brief whether every game has topped out */
    [[nodiscard]] constexpr bool gameOver() const;

    [[nodiscard]] constexpr std::span<sim::TetrisEngine const> games() const { return _games; }

    /** @brief the whole input buffer of the current ply, one @c model_t::INPUTS-sized slice per variation */
    [[nodiscard]] constexpr std::span<data_t const> inputs() const { return _inputs; }

    /** @brief game @ref game's placements for the current ply, parallel to its @ref inputs slices */
    [[nodiscard]] constexpr std::span<placement_t const> placements(size_t game) const;

    /** @brief game @ref game's packed input slices for the current ply, parallel to its @ref placements */
    [[nodiscard]] constexpr std::span<data_t const> inputs(size_t game) const;

private:
    /**
     * @brief appends @ref game's legal placements to @ref _placements and their encoded inputs to
     *  @ref _inputs, dropping any placement whose resulting board duplicates one already added
     * @param base the game's committed stats, projected onto each candidate before encoding
     * @param seen the resulting boards accepted for this game so far, used to reject symmetric
     *  duplicates; cleared on entry and reused across games to avoid re-allocating each call
     */
    constexpr void genInputs(sim::TetrisEngine const& game, stats_t const& base, std::vector<sim::Board2>& seen);

    std::vector<sim::TetrisEngine> _games;

    /** committed game state parallel to @ref _games; advanced by @ref step, projected per candidate */
    std::vector<stats_t> _stats;

    /** flattened placements of the current ply; game g's range is [_begins[g], _begins[g + 1]) */
    std::vector<placement_t> _placements;

    /** prefix sums into @ref _placements (and @ref _inputs by @c model_t::INPUTS), length @c _games.size() + 1 */
    std::vector<size_t> _begins;

    /** packed model inputs of the current ply, @c model_t::INPUTS values per placement */
    std::vector<data_t> _inputs;

    /** draws the per-game seeds so one master seed fixes the whole batch */
    sim::Xoshiro256ss _seedGen;
};

}

namespace ta3::ai {

constexpr MultiTetris2::MultiTetris2(size_t count, uint64_t seed) : _seedGen{seed} {
    _games.reserve(count);
    for(size_t i = 0; i < count; ++i)
        _games.emplace_back(_seedGen());

    _stats.resize(_games.size());
    _begins.resize(_games.size() + 1);
}

constexpr size_t MultiTetris2::genInputs() {
    _placements.clear();
    _inputs.clear();

    std::vector<sim::Board2> seen{};
    for(size_t g = 0; g < _games.size(); ++g) {
        _begins[g] = _placements.size();
        genInputs(_games[g], _stats[g], seen);
    }
    _begins.back() = _placements.size();

    return _placements.size();
}

constexpr void MultiTetris2::genInputs(sim::TetrisEngine const& game, stats_t const& base, std::vector<sim::Board2>& seen) {
    seen.clear();
    if(game.gameOver())
        return;

    sim::PieceType const type = game.currentPiece();
    sim::PieceType const held = game.heldPiece();
    sim::Board2 const& board = game.board();

    for(uint32_t o = 0; o < *sim::Orientation::SIZE; ++o) {
        auto const orientation = static_cast<sim::Orientation>(o);

        // x ranges over every window origin where a block can still touch the field
        for(int x = -static_cast<int>(sim::PIECE_WIDTH); x <= static_cast<int>(sim::WIDTH); ++x) {
            if(!board.available(type, orientation, sim::vec2{x, 0}))
                continue;

            // the placed, pre-clear board is the model's view: its full rows feed the cleared-lines metric
            sim::Board2 next = board;
            next.place(type, orientation, board.dropPlace(type, orientation, sim::vec2{x, 0}));

            // a rotationally symmetric piece reaches the same board through several orientations
            if(std::ranges::contains(seen, next))
                continue;

            seen.push_back(next);
            _placements.push_back({orientation, x});
        }
    }

    size_t const offset = _inputs.size();
    _inputs.resize(offset + seen.size() * model_t::INPUTS);
    for(size_t i = 0; i < seen.size(); ++i) {
        // project the committed stats onto this candidate so its metadata is the post-placement state
        stats_t candidate = base;
        candidate.advance(seen[i].fullLines(), seen[i], type, held);

        parse_inputs_t const parsed{candidate, seen[i], game.lookahead()};
        model_t::extractInputs(parsed, std::span{_inputs}.subspan(offset + i * model_t::INPUTS, model_t::INPUTS));
    }
}

constexpr void MultiTetris2::step(std::span<size_t const> choices) {
    for(size_t g = 0; g < _games.size(); ++g) {
        auto& game = _games[g];
        if(game.gameOver())
            continue;

        auto const placement = placements(g)[choices[g]];
        sim::PieceType const placed = game.currentPiece();
        size_t const cleared = game.place(placement.orientation, placement.x);
        _stats[g].advance(cleared, game.board(), placed, game.heldPiece());
    }
}

constexpr bool MultiTetris2::gameOver() const {
    return std::ranges::all_of(_games, [](sim::TetrisEngine const& game) { return game.gameOver(); });
}

constexpr std::span<MultiTetris2::placement_t const> MultiTetris2::placements(size_t game) const {
    return std::span{_placements}.subspan(_begins[game], _begins[game + 1] - _begins[game]);
}

constexpr std::span<data_t const> MultiTetris2::inputs(size_t game) const {
    return std::span{_inputs}.subspan(_begins[game] * model_t::INPUTS, (_begins[game + 1] - _begins[game]) * model_t::INPUTS);
}

}
