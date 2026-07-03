#pragma once
#include "ta3/ai/generate_variations.hpp"
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
    /** gens + encodes the variations for every game via @ref generate_variations */
    constexpr size_t genInputs();

    /** per game, the local index of its highest rated variation */
    [[nodiscard]] constexpr std::vector<size_t> bestChoices(std::span<data_t const> scores) const;

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
        generate_variations(games[i], _gamesStats[i], _placements, _inputs, _variationStats, boardBuffer);
    }
    _begins.back() = _placements.size();

    return _placements.size();
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

}
