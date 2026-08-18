#pragma once
#include "ta3/ai/search/placements.hpp"
#include "ta3/ai/search/variation_sequences.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/utility/cuda_constant.hpp>
#include <ta3/sim/utility/placement.hpp>
#include <ta3/sim/tetris_engine.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <array>
#include <cstdint>
#include <limits>

/**
 * @file search.hpp
 * @brief depth-@ref DEPTH brute-force lookahead for one game -- the backend-agnostic core.
 *
 * no dedup, no frontier: every leaf is an independent depth-@ref DEPTH line replayed from the committed
 * board. the decision rule is @c argmax over root moves of the max leaf value, and since
 * @c max_root(max_leaf) == the root owning the global-max leaf, the whole search reduces to "find the best
 * leaf, return its root move." all @c constexpr, so this runs identically host-side (unit tests,
 * @c static_assert) and, under CUDA/SYCL, from a kernel via @c --expt-relaxed-constexpr.
 *
 * the GPU port reuses @ref apply + the model call unchanged; it only changes @b work @b assignment
 * (one block per game, threads over @c (sequence, p0, p1) prefixes, each walking the last piece), which
 * cannot change the leaf set or its values.
 */
namespace ta3::ai::search {

static_assert(DEPTH <= ai::CLEAR_LANE_MASK, "a clear-histogram lane must hold DEPTH clears of one kind");

/** value sentinel for "no move / dead"; kept for kernel argmax init and callers. */
inline constexpr value_t DEATH_VALUE = -std::numeric_limits<value_t>::infinity();

/** the chosen root move plus its looked-ahead value; @c none when the game is already over. */
struct search_result {
    sim::drop_place_t move{};
    value_t value = DEATH_VALUE;

    [[nodiscard]] constexpr bool none() const { return value == DEATH_VALUE; }
};

/** result of dropping one piece: the resulting board, the path's clear histogram, and legality. */
struct step {
    sim::Board2 nextBoard{};
    clear_t nextClearHist{};
    bool legal = false;
};

/**
 * drop @p pl onto a copy of @p board, folding the lines it cleared into the histogram @p accum.
 * @return @c {next, accum.added(cleared), true} on a legal drop, or @c {board, accum, false} if @p pl does
 *  not fit at spawn (the leaf simply stops -- "if nothing goes wrong, else do nothing").
 * @note the single shared primitive: the GPU leaf calls exactly this, per level. two fused board walks
 *  (@ref sim::Board2::fit = spawn-collision + drop distance, @ref sim::Board2::placed = copy + lock-in)
 *  instead of the four separate available/copy/dropDistance/place passes: fewer instructions and a
 *  smaller live window in the register-capped kernel. horizontal bounds hold by LUT construction.
 *  @return @ref step object
 */
[[nodiscard]] constexpr step apply(sim::Board2 const& board, clear_t clear_hist, placement pl) {
    auto const [dist, legal] = board.fit(pl.shape);
    if(!legal)
        return {board, clear_hist, false};

    auto next = board.placed(pl.shape, dist);
    auto const cleared = static_cast<std::uint32_t>(next.clearLines());
    return {next, clear_hist.added(cleared), true};
}

namespace dev {

    /**
     * expand levels @c [level, DEPTH) of one hold-resolved sequence under @p board.
     * @return whether ANY placement of @c seq.pieces[level] fit -- when it did not, the CALLER scores its
     *  own board as the leaf ("a prefix that cannot be extended is scored where it stopped").
     * @details plain recursion over the level index (bounded by @ref DEPTH), which keeps the exhaustive
     *  reference depth-agnostic like the beamed search.
     */
    template<class Model, class Consider>
    constexpr bool expand_suffix(
        move_seq const& seq,
        std::uint32_t level,
        sim::Board2 const& board,
        clear_t clear_hist,
        Model const& model,
        Consider const& consider
    ) {
        auto const p = seq.pieces[level];

        auto any = false;
        for(std::uint32_t i = 0; i < n_theoretical_placements(p); ++i) {
            auto const [nextBoard, nextClearHist, legal] = apply(board, clear_hist, nth_placement(p, i));
            if(!legal)
                continue;
            any = true;

            if(level + 1 == DEPTH || !expand_suffix(seq, level + 1, nextBoard, nextClearHist, model, consider))
                consider(model.evaluate(nextClearHist, nextBoard.canonical(), seq.heldIsI[level]));
        }
        return any;
    }

} // namespace dev

/**
 * depth-@ref DEPTH brute lookahead for one game.
 * @param game  the committed game state (left untouched)
 * @param model device value model: @c model(clear_t clearAccum, sim::Board2 const& canonicalLeaf) -> value_t
 * @return the best root move + its looked-ahead value, or a result with @c none() true if the game is over / has no move
 * @details expands every hold-resolved sequence (@ref generate_sequences) and, under it, the full
 *  cartesian product of legal placements (@ref dev::expand_suffix). a prefix that cannot be extended (no
 *  legal next placement) is scored where it stopped, so any legal root move always yields at least a
 *  depth-1 leaf.
 */
template<class Model>
[[nodiscard]] constexpr search_result search_move(sim::TetrisEngine const& game, Model const& model) {
    if(game.gameOver())
        return {};

    auto const& board = game.board();
    auto const sequences = generate_sequences(game.currentPiece(), game.heldPiece(), game.lookahead());

    search_result best{};

    auto const consider = [&](value_t v, sim::drop_place_t root) {
        if(best.none() || v > best.value)
            best = {root, v};
    };

    for(auto const& seq : sequences) {
        auto const nextPiece = seq.pieces[0];

        for(std::uint32_t i = 0; i < n_theoretical_placements(nextPiece); ++i) {
            auto const placement = nth_placement(nextPiece, i);
            auto const [nextBoard, clearHist, legal] = apply(board, {}, placement);
            if(!legal)
                continue;

            sim::drop_place_t const root{placement.orientation, placement.x, seq.rootHold};
            auto const consider_leaf = [&](value_t v) { consider(v, root); };

            // p1 never fit: depth-1 leaf
            if(!dev::expand_suffix(seq, 1, nextBoard, clearHist, model, consider_leaf))
                consider(model.evaluate(clearHist, nextBoard.canonical(), seq.heldIsI[0]), root);
        }
    }

    return best;
}

} // namespace ta3::ai::search
