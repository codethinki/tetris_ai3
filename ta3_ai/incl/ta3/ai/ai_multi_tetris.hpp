#pragma once
#include "ta3/ai/model.hpp"

#include <ta3/sim/multi_tetris.hpp>

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace ta3::ai {

/**
 * runs multiple @ref sim::TetrisEngine at once providing the inputs for @ref model_t
 */
class AiMultiTetris {
public:
    using placement_t = sim::MultiTetris::placement_t;

    /**
      * @param count games
      * @param seed to gen seeds with @ref sim::Xoshiro256ss
      */
    constexpr AiMultiTetris(size_t count, uint64_t seed);

    /**
     * commits the highest rated variation per game, stashes dead games' stats in @ref deadStats, regens
     * @param games_variation_scores one per variation
     */
    constexpr void next(std::span<data_t const> games_variation_scores);

    /** @return true if no games left */
    [[nodiscard]] constexpr bool empty() const { return _games.empty(); }

    /**
     * gets input buffer
     * @details @c m::INPUTS-sized slice per variation
     */
    [[nodiscard]] constexpr std::span<data_t const> inputs() const { return _inputs; }

    /** final stats of every dead game, in death order */
    [[nodiscard]] constexpr std::span<tetris_stats_t const> deadStats() const { return _deadStats; }

    /** committed stats of every still alive game */
    [[nodiscard]] constexpr std::span<tetris_stats_t const> gamesStats() const { return _gamesStats; }

private:
    /** gens + encodes the variations for every game */
    constexpr size_t genInputs();

    /**
     * gens + encodes every variation of a game across both piece contexts
     * @details the current piece dropped directly, plus the held piece reached through a swap
     * @param game_stats committed state of the game
     * @param[out] placements appends entry for each variation
     * @param[out] inputs appends @c m::INPUTS for each variation
     * @param[out] variation_stats appends the projected state of each variation
     * @param[out] board_buffer for dedup alloc reuse
     */
    constexpr void genInputs(
        sim::TetrisEngine const& game,
        tetris_stats_t const& game_stats,
        std::vector<placement_t>& placements,
        std::vector<data_t>& inputs,
        std::vector<tetris_stats_t>& variation_stats,
        std::vector<sim::Board2>& board_buffer
    );

    /**
     * gens + encodes the variations for one piece context
     * @details drop-places @ref place in every legal column and orientation and drops symmetry duplicates
     * @param piece the piece dropped this move
     * @param held_piece the hold slot after the move
     * @param[out] placements appends entry for each variation, hold defaulted (the caller flags swaps)
     * @param[out] inputs appends @c m::INPUTS for each variation
     * @param[out] variation_stats appends the projected state of each variation
     * @param[in,out] board_buffer dedup scratch, cleared on entry
     */
    constexpr void genVariations(
        sim::TetrisEngine const& game,
        tetris_stats_t const& game_stats,
        sim::PieceType piece,
        sim::PieceType held_piece,
        std::vector<placement_t>& placements,
        std::vector<data_t>& inputs,
        std::vector<tetris_stats_t>& variation_stats,
        std::vector<sim::Board2>& board_buffer
    );

    /** per game, the local index of its highest rated variation */
    [[nodiscard]] constexpr std::vector<size_t> bestChoices(std::span<data_t const> scores) const;

    [[nodiscard]] static constexpr std::span<data_t, m::INPUTS> reserveInputs(std::vector<data_t>& buffer);

    sim::MultiTetris _games;

    /** committed state for each game in @ref _games */
    std::vector<tetris_stats_t> _gamesStats;

    /**
     * prefix sums into each games variations
     * @details last is sentinel
     */
    std::vector<size_t> _begins;

    /**
     * flattened placements
     * @details game i's range is [_begins[i], _begins[i + 1])
     */
    std::vector<placement_t> _placements;

    /**
     * flattened inputs
     * @details @c m::INPUTS values per placement
     */
    std::vector<data_t> _inputs;

    /** projected state per variation, parallel to @ref _placements */
    std::vector<tetris_stats_t> _variationStats;

    /** final stats of every dead game, in death order */
    std::vector<tetris_stats_t> _deadStats;
};

}

namespace ta3::ai {

constexpr AiMultiTetris::AiMultiTetris(size_t count, uint64_t seed) : _games{count, seed} {
    _gamesStats.resize(count);
    genInputs();
}

constexpr void AiMultiTetris::next(std::span<data_t const> games_variation_scores) {
    auto const choices = bestChoices(games_variation_scores);

    // pick each game's move and adopt its projected state as the new committed state
    std::vector<placement_t> moves(choices.size());
    for(auto i = 0uz; i < moves.size(); ++i) {
        auto const variationIdx = _begins[i] + choices[i];
        moves[i] = _placements[variationIdx];
        _gamesStats[i] = _variationStats[variationIdx];
    }

    auto const cleared = _games.next(moves);

    // stash the dead games' final stats, compact the survivors' state to stay parallel
    auto alive = 0uz;
    for(auto i = 0uz; i < cleared.size(); ++i)
        if(cleared[i] == sim::TetrisEngine::DIED)
            _deadStats.push_back(_gamesStats[i]);
        else
            _gamesStats[alive++] = _gamesStats[i];
    _gamesStats.resize(alive);

    genInputs();
}

constexpr size_t AiMultiTetris::genInputs() {
    _placements.clear();
    _inputs.clear();
    _variationStats.clear();

    auto const games = _games.games();
    _begins.resize(games.size() + 1);

    std::vector<sim::Board2> boardBuffer{};
    for(auto i = 0uz; i < games.size(); ++i) {
        _begins[i] = _placements.size();
        genInputs(games[i], _gamesStats[i], _placements, _inputs, _variationStats, boardBuffer);
    }
    _begins.back() = _placements.size();

    return _placements.size();
}

constexpr void AiMultiTetris::genInputs(
    sim::TetrisEngine const& game,
    tetris_stats_t const& game_stats,
    std::vector<placement_t>& placements,
    std::vector<data_t>& inputs,
    std::vector<tetris_stats_t>& variation_stats,
    std::vector<sim::Board2>& board_buffer
) {
    auto const current = game.currentPiece();
    auto const held = game.heldPiece();

    // current piece variations
    genVariations(game, game_stats, current, held, placements, inputs, variation_stats, board_buffer);

    // hold variations
    if(game.holdsPiece() && held != current) {
        auto const swapBegin = placements.size();

        genVariations(game, game_stats, held, current, placements, inputs, variation_stats, board_buffer);

        for(auto& p : placements | std::views::drop(swapBegin))
            p.hold = true;
    }
}

constexpr void AiMultiTetris::genVariations(
    sim::TetrisEngine const& game,
    tetris_stats_t const& game_stats,
    sim::PieceType piece,
    sim::PieceType held_piece,
    std::vector<placement_t>& placements,
    std::vector<data_t>& inputs,
    std::vector<tetris_stats_t>& variation_stats,
    std::vector<sim::Board2>& board_buffer
) {
    board_buffer.clear();

    auto const& board = game.board();
    auto const lookahead = game.lookahead();


    for(auto o = 0u; o < *sim::Orientation::SIZE; ++o) {
        auto const orientation = static_cast<sim::Orientation>(o);

        auto const [left, cols] = sim::piece_columns2(piece, orientation);
        auto const xMin = -left;
        auto const xMax = static_cast<int>(sim::WIDTH) - left - static_cast<int>(cols.size());

        for(auto x = xMin; x <= xMax; ++x) {
            sim::vec2 const placeLoc{x, 0};
            // skip occupied
            if(!board.available(piece, orientation, placeLoc))
                continue;

            // duplicate board, place & clear
            auto next = board;
            next.dropPlace(piece, orientation, placeLoc);
            auto const cleared = next.clearLines();

            // ai only runs on canonicalized board
            auto const canonical = next.canonical();

            if(std::ranges::contains(board_buffer, canonical))
                continue;
            board_buffer.push_back(canonical);
            placements.emplace_back(orientation, x);

            auto candidateStats = game_stats;
            candidateStats.advance(cleared, canonical, piece, held_piece);
            variation_stats.push_back(candidateStats);

            auto const parsed = parse_inputs_t{candidateStats, canonical, lookahead};
            model_t::extractInputs(parsed, reserveInputs(inputs));
        }
    }
}

constexpr std::vector<size_t> AiMultiTetris::bestChoices(std::span<data_t const> scores) const {
    auto const games = _games.games().size();

    std::vector<size_t> choices(games);
    for(auto i = 0uz; i < games; ++i) {
        auto const range = scores.subspan(_begins[i], _begins[i + 1] - _begins[i]);
        choices[i] = static_cast<size_t>(std::ranges::distance(range.begin(), std::ranges::max_element(range)));
    }
    return choices;
}

constexpr std::span<data_t, m::INPUTS> AiMultiTetris::reserveInputs(std::vector<data_t>& buffer) {
    auto const offset = buffer.size();
    buffer.resize(offset + m::INPUTS);

    return std::span<data_t, m::INPUTS>{&buffer[offset], m::INPUTS};
}

}
