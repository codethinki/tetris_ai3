#pragma once
#include "ta3/ai/model.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/multi_tetris.hpp>
#include <ta3/sim/tetris_engine.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>
#include <ta3/sim/pieces/piece_offsets.hpp>

#include <algorithm>
#include <bit>
#include <ranges>
#include <span>
#include <vector>

namespace ta3::ai {

/** a single committed move: an orientation, a window x-origin and whether it followed a hold-swap */
using placement_t = sim::MultiTetris::placement_t;

namespace dev {

/** reserves an @c m::INPUTS-wide slice at the end of @p buffer and returns it */
[[nodiscard]] constexpr std::span<data_t, m::INPUTS> reserve_inputs(std::vector<data_t>& buffer) {
    auto const offset = buffer.size();
    buffer.resize(offset + m::INPUTS);

    return std::span<data_t, m::INPUTS>{&buffer[offset], m::INPUTS};
}

/**
 * eroded piece cells (Dellacherie): how many of the just-placed piece's own cells sit in rows
 * that are about to clear.
 * @param placed board after the piece is dropped but before @c clearLines
 * @param piece the piece that was dropped
 * @param orientation the piece's orientation
 * @param landing where the piece came to rest
 */
[[nodiscard]] constexpr size_t cleared_piece_cells(
    sim::Board2 const& placed,
    sim::PieceType piece,
    sim::Orientation orientation,
    sim::vec2 landing
) {
    // a row clears iff every column is filled there, i.e. the AND of all columns
    uint32_t fullMask = ~uint32_t{0};
    for(auto c = 0uz; c < sim::WIDTH; ++c)
        fullMask &= placed.raw_column(c);

    // sum this piece's blocks that fall in those rows
    auto const cols = sim::piece_columns2(piece, orientation).cols;
    size_t sum = 0;
    for(auto const mask : cols)
        sum += static_cast<size_t>(std::popcount((static_cast<uint32_t>(mask) << landing.y) & fullMask));
    return sum;
}

/**
 * gens + encodes the variations for one piece context
 * @details drop-places @p piece in every legal column and orientation and drops symmetry duplicates
 * @param piece the piece dropped this move
 * @param held_piece the hold slot after the move
 * @param[out] placements appends entry for each variation, hold defaulted (the caller flags swaps)
 * @param[out] inputs appends @c m::INPUTS for each variation
 * @param[out] variation_stats appends the projected state of each variation
 * @param[in,out] board_buffer dedup scratch, cleared on entry
 */
constexpr void generate_variations_ctx(
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

            auto const landing = board.dropLocation(piece, orientation, placeLoc);
            auto const clearedPieceCells = cleared_piece_cells(next, piece, orientation, landing);

            auto const cleared = next.clearLines();

            // ai only runs on canonicalized board
            auto const canonical = next.canonical();

            if(std::ranges::contains(board_buffer, canonical))
                continue;
            board_buffer.push_back(canonical);
            placements.emplace_back(orientation, x);

            auto candidateStats = game_stats;
            candidateStats.advance(cleared, canonical, piece, held_piece, lookahead, clearedPieceCells);
            variation_stats.push_back(candidateStats);

            model_t::extractInputs(candidateStats, reserve_inputs(inputs));
        }
    }
}

} // namespace dev

/**
 * appends one engine's legal, canonical-dedup'd placement variations across both piece contexts
 * @details the current piece dropped directly, plus the held piece reached through a swap
 * @param game the engine to expand; its committed state is left untouched
 * @param game_stats committed state of the game
 * @param[out] placements appends an entry for each variation (hold flagged on swap variations)
 * @param[out] inputs appends @c m::INPUTS values for each variation
 * @param[out] variation_stats appends the projected state of each variation, parallel to @p placements
 * @param[in,out] board_buffer dedup scratch, reused across calls
 */
constexpr void generate_variations(
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
    dev::generate_variations_ctx(game, game_stats, current, held, placements, inputs, variation_stats, board_buffer);

    // hold variations
    if(game.holdsPiece() && held != current) {
        auto const swapBegin = placements.size();

        dev::generate_variations_ctx(game, game_stats, held, current, placements, inputs, variation_stats, board_buffer);

        for(auto& p : placements | std::views::drop(swapBegin))
            p.hold = true;
    }
}

} // namespace ta3::ai
