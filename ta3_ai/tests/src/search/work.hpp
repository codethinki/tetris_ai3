#pragma once
#include "ta3/ai/search/placements.hpp"
#include "ta3/ai/search/search.hpp"
#include "ta3/ai/search/variation_sequences.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/utility/placement.hpp>
#include <ta3/sim/tetris_engine.hpp>

#include <array>
#include <cstdint>

/**
 * @file work.hpp
 * @brief thread <-> work-id mapping for the tiled GPU search -- the last backend-agnostic piece.
 *
 * the kernel maps one block to one game and its threads to @b prefixes. a prefix fixes the first
 * @c DEPTH-1 pieces' placements @c (i0, i1) of one hold-resolved @ref move_seq; the owning thread replays
 * that prefix @b once and walks the last piece's placements (the plan's "fix DEPTH-1, walk the last"
 * tiling -- ~3x fewer @c dropPlace calls than one-leaf-per-thread).
 *
 * a move's prefixes are numbered @c [0, prefix_count) across all sequences. @ref build_layout precomputes
 * each sequence's prefix count @c (theo_count(p0) * theo_count(p1)) and their exclusive prefix-sum, so a
 * linear thread id decodes to @c (sequence, i0, i1) with one small scan + a divmod (@ref decode_prefix).
 * @ref eval_prefix then reproduces @ref search_move's exact leaf set for that prefix, emitting each leaf's
 * value + root move through a @c consider callback -- host-side a lambda, device-side the @c atomicMax.
 * @ref search_move_tiled loops every prefix through it and, by construction, equals @ref search_move.
 *
 * @note LEGACY, inherently depth-3 (a prefix is exactly (i0, i1)): the exhaustive tiled kernel it served
 *  was replaced by the depth-agnostic beam (@ref beam.hpp). gated on @c TA3_SEARCH_DEPTH == 3 rather than
 *  generalized -- at other depths the tiled decomposition and its tests simply drop out of the build.
 */
#if TA3_SEARCH_DEPTH == 3
namespace ta3::ai::search {

/** a thread's prefix: which sequence, and the fixed placement indices of the first @c DEPTH-1 pieces. */
struct work_prefix {
    std::uint32_t seq = 0;
    std::uint32_t i0 = 0;
    std::uint32_t i1 = 0;
};

/**
 * per-move work decomposition: the exclusive prefix-sum of each sequence's prefix count.
 * @details @c offset[s] is the first prefix id of sequence @c s; @c offset[count] is the total. built once
 *  per move (host-side, or cooperatively into shared memory) and read by every thread.
 */
struct work_layout {
    std::array<std::uint32_t, MAX_SEQUENCES + 1> offset{};
    std::uint32_t count = 0;
};

/** total prefixes (threads' worth of work) for this move. */
[[nodiscard]] constexpr std::uint32_t prefix_count(work_layout const& w) { return w.offset[w.count]; }

/** number of prefixes contributed by one sequence: @c theo_count(p0) * theo_count(p1). */
[[nodiscard]] constexpr std::uint32_t seq_prefixes(move_seq const& seq) {
    return theo_count(seq.pieces[0]) * theo_count(seq.pieces[1]);
}

/** exclusive prefix-sum the sequences' prefix counts, in place. */
constexpr void build_layout_into(work_layout& w, seq_set const& seqs) {
    w.count = seqs.count;
    std::uint32_t acc = 0;
    for(std::uint32_t s = 0; s < seqs.count; ++s) {
        w.offset[s] = acc;
        acc += seq_prefixes(seqs[s]);
    }
    w.offset[seqs.count] = acc;
}

/**
 * return-by-value convenience wrapper. on the GPU prefer @ref build_layout_into to write straight into the
 * block's shared @c work_layout rather than returning the aggregate onto the device stack.
 */
[[nodiscard]] constexpr work_layout build_layout(seq_set const& seqs) {
    work_layout w{};
    build_layout_into(w, seqs);
    return w;
}

/** decode a linear prefix @p id into @c (sequence, i0, i1). @pre @c id < prefix_count(w). */
[[nodiscard]] constexpr work_prefix decode_prefix(seq_set const& seqs, work_layout const& w, std::uint32_t id) {
    std::uint32_t s = 0;
    while(s + 1 < w.count && w.offset[s + 1] <= id)
        ++s;

    std::uint32_t const local = id - w.offset[s];
    std::uint32_t const b1 = theo_count(seqs[s].pieces[1]);
    return {s, local / b1, local % b1};
}

/** the inverse of @ref decode_prefix (for tests / host bookkeeping). */
[[nodiscard]] constexpr std::uint32_t encode_prefix(seq_set const& seqs, work_layout const& w, work_prefix p) {
    return w.offset[p.seq] + p.i0 * theo_count(seqs[p.seq].pieces[1]) + p.i1;
}

namespace dev {

/**
 * expand the last piece over the depth-2 board @p s1: emit a depth-3 leaf for every legal placement of @p p2
 * (ranked @c base + i2, matching @ref search_move's emission order).
 * @return whether any placement fit -- lets the caller fall back to scoring the depth-2 board.
 */
template<class Model, class Consider>
[[nodiscard]] constexpr bool expand_last(
    step const& s1,
    sim::PieceType p2,
    sim::drop_place_t root,
    std::uint32_t base,
    Model const& model,
    Consider const& consider
) {
    bool expanded = false;
    for(std::uint32_t i2 = 0; i2 < theo_count(p2); ++i2) {
        step const s2 = apply(s1.board, s1.accum, nth_placement(p2, i2));
        if(!s2.ok)
            continue;
        expanded = true;
        consider(model.evaluate(s2.accum, s2.board.canonical()), root, base + i2);
    }
    return expanded;
}

/** whether the continuation piece @p p1 has @b any legal placement on the depth-1 board @p s0. */
[[nodiscard]] constexpr bool has_continuation(step const& s0, sim::PieceType p1) {
    for(std::uint32_t i1 = 0; i1 < theo_count(p1); ++i1)
        if(apply(s0.board, s0.accum, nth_placement(p1, i1)).ok)
            return true;
    return false;
}

} // namespace dev

/**
 * evaluate one prefix, emitting the same leaves @ref search_move would for it.
 * @param consider @c consider(value_t, sim::drop_place_t root, std::uint32_t order) -- host lambda, or the
 *  device @c atomicMax. @p order is a global leaf rank @c (id*MAX_PLACEMENTS + i2) that increases in exact
 *  @ref search_move emission order, so a device reduction can break value ties like the host (first wins)
 *  by preferring the smaller order. host callers that break ties by call sequence ignore it.
 * @details depth-3 leaves and the depth-2 fallback (no last placement fits) are prefix-local. the depth-1
 *  fallback (a legal root with @b no legal continuation) is a per-root property, so it is attributed to the
 *  @c i1==0 prefix, which confirms no @c i1 fits before scoring the root alone -- exactly @ref search_move's
 *  @c "if(!expanded1)" branch. an illegal root placement contributes nothing.
 */
template<class Model, class Consider>
constexpr void eval_prefix(
    sim::Board2 const& board,
    seq_set const& seqs,
    work_layout const& w,
    std::uint32_t id,
    Model const& model,
    Consider&& consider
) {
    work_prefix const wp = decode_prefix(seqs, w, id);
    move_seq const& seq = seqs[wp.seq];
    auto const [p0, p1, p2] = seq.pieces;

    std::uint32_t const base = id * MAX_PLACEMENTS; // this prefix's leaf ranks: [base, base + theo_count(p2))

    // root: the first piece. an illegal root placement owns no leaves.
    placement const pl0 = nth_placement(p0, wp.i0);
    step const s0 = apply(board, {}, pl0);
    if(!s0.ok)
        return;

    sim::drop_place_t const root{pl0.orientation, pl0.x, seq.rootHold};

    // continuation: the second piece at this prefix's i1.
    step const s1 = apply(s0.board, s0.accum, nth_placement(p1, wp.i1));

    if(s1.ok) {
        // depth-3 leaves; if the last piece never fits, the depth-2 board itself is the leaf.
        if(!dev::expand_last(s1, p2, root, base, model, consider))
            consider(model.evaluate(s1.accum, s1.board.canonical()), root, base);
        return;
    }

    // no continuation at this prefix. the depth-1 fallback (legal root, but nothing extends it) is a per-root
    // property, so attribute it once -- from the i1 == 0 prefix -- exactly like search_move's !expanded1 branch.
    if(wp.i1 == 0 && !dev::has_continuation(s0, p1))
        consider(model.evaluate(s0.accum, s0.board.canonical()), root, base);
}

/**
 * host reference for the tiled search: loops every prefix through @ref eval_prefix.
 * @details identical result to @ref search_move by construction; the device kernel replaces this loop with
 *  a grid-stride over prefixes and a float @c atomicMax for @c consider.
 */
template<class Model>
[[nodiscard]] constexpr search_result search_move_tiled(sim::TetrisEngine const& game, Model const& model) {
    if(game.gameOver())
        return {};

    sim::Board2 const& board = game.board();
    seq_set const seqs = generate_sequences(game.currentPiece(), game.heldPiece(), game.lookahead());
    work_layout const w = build_layout(seqs);

    search_result best;
    auto const consider = [&](value_t 