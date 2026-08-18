#pragma once

#include "block_reduce.cuh"

#include <ta3/gpu/detail/value_model.hpp>

#include <ta3/ai/search/beam.hpp>

#include <ta3/ai/models/metrics/tetris_stats_v4.hpp>

#include <ta3/sim/tetris_engine.hpp>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>

/**
 * @file eval_kernel.cuh
 * @brief the beamed search kernel: one CUDA block per (model, game), depth-agnostic cooperative stages
 *  per move.
 * @details structural twin of the SYCL kernel in ../../sycl/detail/eval_kernel.hpp -- same stages, same
 *  function boundaries, same names -- so the two stay diffable; only the mechanics differ (@c __shared__
 *  + @c __syncthreads() + warp shuffles here, one @c local_accessor + @c group_barrier + sub-group
 *  primitives there, where the width is never assumed to be 32). one @c __shared__ @ref BlockState
 *  replaces a pile of individual @c __shared__ variables. parameters that alias this block-local storage
 *  carry a @c block_ prefix (the SYCL file's @c wgroup_), to set them apart from genuinely per-thread
 *  values (@c t, @c g, @c lane, ...) in the same signature. all function parameters use all_lower
 *  snake_case; locals (including struct members) use camelCase. loop variables follow @c i / @c j / @c k
 *  by nesting depth, or an @c <expressiveName>Idx when a bare letter would lose meaning.
 */
namespace ta3::gpu {
namespace {

    namespace search = ta3::ai::search;
    namespace sim = ta3::sim;

    using search::plan_tree;

    constexpr std::uint32_t BLOCK = 128;
    constexpr std::uint32_t WARPS = BLOCK / 32;

    constexpr std::uint32_t DEPTH = search::DEPTH;
    // compile-time folds of the per-level widths only. the runtime-indexed access in the middle-stage
    // loop deliberately reads search::BEAM_WIDTHS itself, which is TA3_CUDA_CONSTANT for exactly that --
    // indexing this constexpr copy with a runtime level would materialise it into local memory.
    constexpr search::beam_widths_t KS = search::BEAM_WIDTHS; // per-level widths; KS[0] = roots
    constexpr std::uint32_t K1 = KS[0];
    constexpr std::uint32_t KMAX = search::max_width(KS);
    constexpr std::uint32_t ROOT_SLOTS = search::ROOT_SLOTS;
    constexpr std::uint32_t MID_SLOTS = search::mid_slots(KS);
    constexpr std::uint32_t LEAF_ORDERS = search::leaf_orders(search::last_width(KS));
    constexpr std::uint32_t SPAN = search::order_span(KS);
    constexpr std::uint32_t MP = search::MAX_PLACEMENTS;

    /** middle-stage segment capacity: one segment per (kept frontier node, child). */
    constexpr std::uint32_t SEGMENT_CAP = KMAX * search::MAX_CHILD_MID;

    static_assert(KMAX <= 32, "the fallback mask and the frontier caches assume ranks fit one 32-bit mask");
    static_assert(KMAX <= BLOCK, "frontier caching runs one thread per kept node");
    static_assert(search::MAX_LAST <= 32, "the leaf-stage legality mask assumes the last-piece lists fit one u32");
    static_assert(SEGMENT_CAP * MP <= 0xFFFFu, "segment offsets are u16");
    static_assert(std::is_trivially_copyable_v<sim::TetrisEngine>, "engine must be blittable to the device");

    /** every per-block scratch variable, gathered into one struct for a single @c __shared__ allocation.
     *  @c warpBest is sized to the warp count, not the block -- see block_reduce.cuh (the SYCL twin sizes
     *  it to the full work-group because sub-group width is a runtime property there). */
    struct BlockState {
        std::array<ai::data_t, ai::model_t::NUM_PARAMS> weights;
        sim::Board2 board; // the committed board
        std::array<sim::PieceType, search::NUM_SLOTS> slot; // pieces per abstract window slot
        std::uint32_t treeIdx; // which PLAN_TREE (held empty?)

        std::array<std::uint32_t, ROOT_SLOTS> rootScores;
        std::array<std::uint32_t, MID_SLOTS> midScores;
        std::array<std::uint32_t, K1> selectedRoot, selectedRootScore;
        std::array<std::uint32_t, KMAX> selectedMid;
        std::array<std::uint32_t, KMAX> selectedMidScore;

        // the kept frontier, ping-ponged level to level: board, clear history, score, tree node, root rank.
        std::array<std::array<sim::Board2, KMAX>, 2> beamBoards;
        std::array<std::array<search::clear_t, KMAX>, 2> beamClearHists;
        std::array<std::array<std::uint32_t, KMAX>, 2> beamScores;
        std::array<std::array<std::uint8_t, KMAX>, 2> beamNodes;
        std::array<std::array<std::uint8_t, KMAX>, 2> beamRoots;
        std::uint32_t hasChild; // bit i: beam slot i has a legal continuation

        // middle-stage compaction: segment table over the frontier's real child x placement spans
        // -- see build_mid_segments / cache_kept_mid.
        std::array<std::uint16_t, SEGMENT_CAP + 1> segmentOffset;
        std::array<std::uint8_t, SEGMENT_CAP> segmentBeamIdx;
        std::array<std::uint8_t, SEGMENT_CAP> segmentNode;
        std::array<sim::PieceType, SEGMENT_CAP> segmentPiece;
        std::array<std::uint8_t, SEGMENT_CAP> segmentHeld; // per-segment: child's hold slot resolves to an I piece
        std::uint32_t segmentCount;

        std::array<unsigned long long, WARPS> warpBest;
        int over;
    };

    /**
     * per-move staging: thread 0 refreshes the committed board / held piece / plan tree and gathers the
     * concrete pieces per abstract window slot; every thread zeroes its stride of the root-score array so
     * stale scores from the previous move (or previous game, for slots the current tree doesn't use) never
     * leak into this move's selection. (the middle-stage score array is zeroed per level instead.)
     */
    __device__ void reset_move_state(
        sim::TetrisEngine* __restrict__ games,
        std::uint32_t g,
        sim::Board2& block_board,
        std::uint32_t& block_tree_idx,
        std::span<sim::PieceType, search::NUM_SLOTS> block_slot,
        std::span<std::uint32_t, ROOT_SLOTS> block_root_scores,
        std::uint32_t t
    ) {
        if(t == 0) {
            block_board = games[g].board();
            auto const held = games[g].heldPiece();
            block_tree_idx = held == sim::TetrisEngine::NO_PIECE ? 1u : 0u;
            auto const gathered = search::gather_slots(games[g].currentPiece(), held, games[g].lookahead());
            for(std::uint32_t i = 0; i < search::NUM_SLOTS; ++i)
                block_slot[i] = gathered[i];
        }
        for(std::uint32_t i = t; i < ROOT_SLOTS; i += BLOCK)
            block_root_scores[i] = 0;
    }

    /** stage 0: threads stride the root-slot space and score every legal depth-1 board. */
    __device__ void score_roots(
        sim::Board2 const& block_board,
        std::span<sim::PieceType const, search::NUM_SLOTS> block_slot,
        plan_tree const& tree,
        net_ref const& model,
        std::span<std::uint32_t, ROOT_SLOTS> block_root_scores,
        std::uint32_t t
    ) {
        for(std::uint32_t slotIdx = t; slotIdx < ROOT_SLOTS; slotIdx += BLOCK) {
            auto const g = slotIdx / MP;
            auto const rootPlacementIdx = slotIdx % MP;
            if(g >= tree.count[0])
                continue;
            auto const& node = tree.nodes[0][g];
            auto const rootPiece = block_slot[node.slot];
            if(rootPlacementIdx >= search::n_theoretical_placements(rootPiece))
                continue;
            auto const [nextBoard, nextClearHist, legal] = search::apply(
                block_board,
                {},
                search::nth_placement(rootPiece, rootPlacementIdx)
            );
            if(legal) {
                auto const heldIsI = node.heldSlot == search::HELD_NONE_SLOT
                                     ? false
                                     : block_slot[node.heldSlot] == sim::PieceType::I;
                block_root_scores[slotIdx] = search::ordered_bits(
                    model.evaluate(nextClearHist, nextBoard, heldIsI)
                );
            }
        }
    }

    /**
     * seed the kept frontier from the selected roots (one cheap re-apply beats storing all of stage 0):
     * board, clear history, score, tree node index and root rank per kept root, into frontier row 0.
     */
    __device__ void cache_kept_roots(
        sim::Board2 const& block_board,
        std::span<sim::PieceType const, search::NUM_SLOTS> block_slot,
        plan_tree const& tree,
        std::span<std::uint32_t const, K1> block_selected_root,
        std::span<std::uint32_t const, K1> block_selected_root_score,
        std::uint32_t root_count,
        std::span<sim::Board2, KMAX> block_beam_boards,
        std::span<search::clear_t, KMAX> block_beam_clear_hists,
        std::span<std::uint32_t, KMAX> block_beam_scores,
        std::span<std::uint8_t, KMAX> block_beam_nodes,
        std::span<std::uint8_t, KMAX> block_beam_roots,
        std::uint32_t t
    ) {
        if(t < root_count) {
            auto const g = block_selected_root[t] / MP;
            auto const rootPiece = block_slot[tree.nodes[0][g].slot];
            auto const rootStep = search::apply(
                block_board,
                {},
                search::nth_placement(rootPiece, block_selected_root[t] % MP)
            );
            block_beam_boards[t] = rootStep.nextBoard;
            block_beam_clear_hists[t] = rootStep.nextClearHist;
            block_beam_scores[t] = block_selected_root_score[t];
            block_beam_nodes[t] = static_cast<std::uint8_t>(g);
            block_beam_roots[t] = static_cast<std::uint8_t>(t);
        }
    }

    /**
     * build one middle level's segment table: one segment per (kept frontier node, child), spanning
     * that child piece's REAL theoretical placement count. compacted work ids enumerate (beam index,
     * child index, placement index) lexicographically -- order-isomorphic to the host's padded slot
     * ids, so the later top-K selection (max score, lowest id on ties) picks the identical sequence.
     * thread 0 only (<= SEGMENT_CAP trivial iterations, once per level); the caller owes the barrier.
     */
    __device__ void build_mid_segments(
        std::span<sim::PieceType const, search::NUM_SLOTS> block_slot,
        plan_tree const& tree,
        std::uint32_t level,
        std::span<std::uint8_t const, KMAX> block_beam_nodes,
        std::uint32_t beam_width,
        std::span<std::uint16_t, SEGMENT_CAP + 1> block_segment_offset,
        std::span<std::uint8_t, SEGMENT_CAP> block_segment_beam_idx,
        std::span<std::uint8_t, SEGMENT_CAP> block_segment_node,
        std::span<sim::PieceType, SEGMENT_CAP> block_segment_piece,
        std::span<std::uint8_t, SEGMENT_CAP> block_segment_held,
        std::uint32_t& block_segment_count
    ) {
        std::uint32_t ns = 0;
        std::uint16_t off = 0;
        for(std::uint32_t i = 0; i < beam_width; ++i) {
            auto const& pnode = tree.nodes[level - 1][block_beam_nodes[i]];
            for(std::uint32_t j = pnode.childBegin; j < pnode.childEnd; ++j) {
                auto const& cnode = tree.nodes[level][j];
                auto const p = block_slot[cnode.slot];
                block_segment_offset[ns] = off;
                block_segment_beam_idx[ns] = static_cast<std::uint8_t>(i);
                block_segment_node[ns] = static_cast<std::uint8_t>(j);
                block_segment_piece[ns] = p;
                block_segment_held[ns] = cnode.heldSlot != search::HELD_NONE_SLOT
                    && block_slot[cnode.heldSlot] == sim::PieceType::I;
                off = static_cast<std::uint16_t>(off + search::n_theoretical_placements(p));
                ++ns;
            }
        }
        block_segment_offset[ns] = off;
        block_segment_count = ns;
    }

    /**
     * middle stage scoring: threads stride the COMPACTED continuation space (see @ref build_mid_segments)
     * and score every legal board -- every work id is a real (kept node, child, placement) triple, no
     * padding lanes. the segment cursor only moves forward because work ids increase with the stride, so
     * the decode costs O(segments) over the whole loop, not per item.
     */
    __device__ void score_mid(
        std::span<std::uint16_t const, SEGMENT_CAP + 1> block_segment_offset,
        std::span<std::uint8_t const, SEGMENT_CAP> block_segment_beam_idx,
        std::span<sim::PieceType const, SEGMENT_CAP> block_segment_piece,
        std::span<std::uint8_t const, SEGMENT_CAP> block_segment_held,
        std::uint32_t segment_count,
        std::span<sim::Board2 const, KMAX> block_beam_boards,
        std::span<search::clear_t const, KMAX> block_beam_clear_hists,
        net_ref const& model,
        std::span<std::uint32_t, MID_SLOTS> block_mid_scores,
        std::uint32_t& block_has_child,
        std::uint32_t t
    ) {
        // segmentOffset is u16 (packed, see SEGMENT_CAP's static_assert); widened to u32 on purpose so the
        // stride comparison against BLOCK-stepped `i` below never narrows.
        std::uint32_t const total = block_segment_offset[segment_count];
        std::uint32_t j = 0;
        for(std::uint32_t i = t; i < total; i += BLOCK) {
            while(block_segment_offset[j + 1] <= i)
                ++j;
            // segmentBeamIdx is u8; widened to u32 on purpose to index the u32-addressed beam arrays below.
            std::uint32_t const beamIdx = block_segment_beam_idx[j];
            auto const s = search::apply(
                block_beam_boards[beamIdx],
                block_beam_clear_hists[beamIdx],
                search::nth_placement(block_segment_piece[j], i - block_segment_offset[j])
            );
            if(s.legal) {
                block_mid_scores[i] = search::ordered_bits(
                    model.evaluate(s.nextClearHist, s.nextBoard, block_segment_held[j] != 0)
                );
                if(((block_has_child >> beamIdx) & 1u) == 0) // racy pre-check: only ever skips an already-set bit
                    atomicOr(&block_has_child, 1u << beamIdx);
            }
        }
    }

    /**
     * advance the kept frontier one level: decode each kept continuation's (rank, tree node) from its
     * compacted id exactly once, re-apply its placement, and write the next frontier row -- carrying the
     * root rank forward so the final decode never needs per-level history.
     */
    __device__ void cache_kept_mid(
        std::span<std::uint16_t const, SEGMENT_CAP + 1> block_segment_offset,
        std::span<std::uint8_t const, SEGMENT_CAP> block_segment_beam_idx,
        std::span<std::uint8_t const, SEGMENT_CAP> block_segment_node,
        std::span<sim::PieceType const, SEGMENT_CAP> block_segment_piece,
        std::span<std::uint32_t const, KMAX> block_selected_mid,
        std::span<std::uint32_t const, KMAX> block_selected_mid_score,
        std::uint32_t n_next,
        std::span<sim::Board2 const, KMAX> block_beam_boards,
        std::span<search::clear_t const, KMAX> block_beam_clear_hists,
        std::span<std::uint8_t const, KMAX> block_beam_roots,
        std::span<sim::Board2, KMAX> block_next_beam_boards,
        std::span<search::clear_t, KMAX> block_next_beam_clear_hists,
        std::span<std::uint32_t, KMAX> block_next_beam_scores,
        std::span<std::uint8_t, KMAX> block_next_beam_nodes,
        std::span<std::uint8_t, KMAX> block_next_beam_roots,
        std::uint32_t t
    ) {
        if(t >= n_next)
            return;

        auto const idx = block_selected_mid[t];
        std::uint32_t i = 0;
        while(block_segment_offset[i + 1] <= idx)
            ++i;
        // segmentBeamIdx is u8; widened to u32 on purpose to index the u32-addressed beam arrays below.
        std::uint32_t const beamIdx = block_segment_beam_idx[i];
        auto const s = search::apply(
            block_beam_boards[beamIdx],
            block_beam_clear_hists[beamIdx],
            search::nth_placement(block_segment_piece[i], idx - block_segment_offset[i])
        );
        block_next_beam_boards[t] = s.nextBoard;
        block_next_beam_clear_hists[t] = s.nextClearHist;
        block_next_beam_scores[t] = block_selected_mid_score[t];
        block_next_beam_nodes[t] = block_segment_node[i];
        block_next_beam_roots[t] = block_beam_roots[beamIdx];
    }

    /**
     * leaf stage: one warp per kept frontier node expands the last piece. the (list, placement) pairs are
     * FLATTENED: lanes stride the node's total placement count across all last-piece lists, so a
     * 17-placement list no longer leaves half the warp idle before the next list starts. leaves (plus the
     * per-list fallbacks) fold into the per-thread packed (score | order) maxima -- seeded with @p best,
     * which already carries any mid-level fallbacks this thread emitted -- combined by ONE block
     * reduction. the per-list fallback ("this list's piece never fit") comes from a warp-wide legality
     * bitmask (bit per list, shuffle-OR across lanes), identical semantics to the host's per-list
     * @c !expanded branch. returns the winning packed key (never 0 whenever rootCount > 0).
     */
    [[nodiscard]] __device__ unsigned long long expand_leaves_and_reduce(
        std::span<sim::PieceType const, search::NUM_SLOTS> block_slot,
        plan_tree const& tree,
        std::span<sim::Board2 const, KMAX> block_beam_boards,
        std::span<search::clear_t const, KMAX> block_beam_clear_hists,
        std::span<std::uint32_t const, KMAX> block_beam_scores,
        std::span<std::uint8_t const, KMAX> block_beam_nodes,
        std::uint32_t beam_width,
        net_ref const& model,
        unsigned long long best,
        std::span<unsigned long long, WARPS> block_warp_best,
        std::uint32_t t,
        std::uint32_t lane,
        std::uint32_t wid
    ) {
        auto const consider = [&](std::uint32_t v_ordered, std::uint32_t order) {
            unsigned long long const key = search::pack_key(v_ordered, order, SPAN);
            best = key > best ? key : best;
        };

        for(std::uint32_t i = wid; i < beam_width; i += WARPS) {
            // warp per kept node, lanes over flat placements
            auto const& lnode = tree.nodes[DEPTH - 2][block_beam_nodes[i]];

            auto const lists = static_cast<std::uint32_t>(lnode.childEnd - lnode.childBegin);

            std::uint32_t total = 0;
            for(std::uint32_t listIdx = 0; listIdx < lists; ++listIdx)
                total += search::n_theoretical_placements(
                    block_slot[tree.nodes[DEPTH - 1][lnode.childBegin + listIdx].slot]
                );

            std::uint32_t legal = 0; // bit listIdx: some placement of list listIdx fit
            for(std::uint32_t j = lane; j < total; j += 32) {
                // decode (listIdx, placementIdx) by walking the (<= MAX_LAST) list counts -- no runtime-indexed array
                auto placementIdx = j;
                std::uint32_t listIdx = 0;

                auto p = block_slot[tree.nodes[DEPTH - 1][lnode.childBegin].slot];
                for(std::uint32_t listCount = search::n_theoretical_placements(p); placementIdx >= listCount;
                    listCount = search::n_theoretical_placements(p)) {
                    placementIdx -= listCount;
                    p = block_slot[tree.nodes[DEPTH - 1][lnode.childBegin + ++listIdx].slot];
                }

                auto const [nextBoard, nextClearHist, stepLegal] = search::apply(
                    block_beam_boards[i],
                    block_beam_clear_hists[i],
                    search::nth_placement(p, placementIdx)
                );
                if(!stepLegal)
                    continue;
                legal |= 1u << listIdx;
                auto const& cnode = tree.nodes[DEPTH - 1][lnode.childBegin + listIdx];
                auto const heldIsI = cnode.heldSlot == search::HELD_NONE_SLOT
                                     ? false
                                     : block_slot[cnode.heldSlot] == sim::PieceType::I;
                consider(
                    search::ordered_bits(model.evaluate(nextClearHist, nextBoard, heldIsI)),
                    (i * search::MAX_LAST + listIdx) * MP + placementIdx
                );
            }
            // per-list fallback: the piece never fit anywhere -> the frontier board is the leaf. combine
            // the legality mask across the warp.
#pragma unroll
            for(int offset = 16; offset > 0; offset >>= 1)
                legal |= __shfl_xor_sync(0xFFFFFFFFu, legal, offset);
            if(lane == 0)
                for(std::uint32_t listIdx = 0; listIdx < lists; ++listIdx)
                    if(((legal >> listIdx) & 1u) == 0)
                        consider(block_beam_scores[i], (i * search::MAX_LAST + listIdx) * MP);
        }

        return block_max_bcast<BLOCK>(best, block_warp_best, t);
    }

    /** decode the winning (score | order) key back to the root move and advance the engine. thread 0 only. */
    __device__ void commit_move(
        sim::TetrisEngine* __restrict__ games,
        std::uint32_t g,
        std::span<sim::PieceType const, search::NUM_SLOTS> block_slot,
        plan_tree const& tree,
        std::span<std::uint32_t const, K1> block_selected_root,
        std::span<std::uint8_t const, KMAX> block_beam_roots,
        unsigned long long winner,
        ai::stats_t& stats,
        int& block_over
    ) {
        auto const order = search::key_pos(winner, SPAN);
        auto const rootRank = order >= LEAF_ORDERS
                              ? order - LEAF_ORDERS
                              : block_beam_roots[order / (search::MAX_LAST * MP)];

        auto const& rootNode = tree.nodes[0][block_selected_root[rootRank] / MP];
        auto const rootPlacement =
            search::nth_placement(block_slot[rootNode.slot], block_selected_root[rootRank] % MP);

        if(rootNode.rootHold)
            games[g].hold();
        auto const cleared = games[g].place(rootPlacement.orientation, rootPlacement.x);

        if(cleared == sim::TetrisEngine::DIED) { block_over = 1; }
        else {
            stats.advance(games[g].board(), static_cast<std::uint32_t>(cleared));
            if(games[g].gameOver())
                block_over = 1;
        }
    }

    /**
     * play one move: stage 0 scores every legal root and keeps the best K1; each middle level scores
     * the frontier's continuations and keeps the best KS[level]; the leaf stage expands the last piece
     * and reduces to a single winning (score | order) key; commit decodes that key back to the root
     * move and advances the engine. thread 0 is the only one that touches @p games / @p stats directly;
     * every other thread cooperates purely through @p block.
     */
    __device__ void play_move(
        BlockState& block,
        sim::TetrisEngine* __restrict__ games,
        std::uint32_t g,
        std::uint32_t t,
        std::uint32_t lane,
        std::uint32_t wid,
        net_ref const& model,
        ai::stats_t& stats
    ) {
        reset_move_state(games, g, block.board, block.treeIdx, block.slot, block.rootScores, t);
        __syncthreads();

        auto const& tree = search::dev::PLAN_TREES[block.treeIdx];

        score_roots(block.board, block.slot, tree, model, block.rootScores, t);
        __syncthreads();

        auto const rootCount = select_top_block<BLOCK, ROOT_SLOTS, K1>(
            block.rootScores,
            block.selectedRoot,
            block.selectedRootScore,
            block.warpBest,
            t,
            K1
        );
        if(rootCount == 0) { // no legal root placement: the game is over (matches the host's `r.none`)
            if(t == 0)
                block.over = 1;
            __syncthreads();
            return;
        }

        cache_kept_roots(
            block.board,
            block.slot,
            tree,
            block.selectedRoot,
            block.selectedRootScore,
            rootCount,
            block.beamBoards[0],
            block.beamClearHists[0],
            block.beamScores[0],
            block.beamNodes[0],
            block.beamRoots[0],
            t
        );
        // publish the seeded frontier: the next reader (build_mid_segments' beamNodes scan on thread 0,
        // or the leaf stage directly when DEPTH == 2) crosses threads.
        __syncthreads();

        // per-thread packed (score | order) maximum: mid-level fallbacks fold in here as they are
        // discovered; the leaf stage adds the leaves and runs the single block reduction.
        unsigned long long best = 0;
        auto n = rootCount;

        std::uint32_t cur = 0;

        for(std::uint32_t levelIdx = 1; levelIdx + 1 < DEPTH; ++levelIdx) {
            for(std::uint32_t k = t; k < MID_SLOTS; k += BLOCK)
                block.midScores[k] = 0;
            if(t == 0) {
                block.hasChild = 0;
                build_mid_segments(
                    block.slot,
                    tree,
                    levelIdx,
                    block.beamNodes[cur],
                    n,
                    block.segmentOffset,
                    block.segmentBeamIdx,
                    block.segmentNode,
                    block.segmentPiece,
                    block.segmentHeld,
                    block.segmentCount
                );
            }
            __syncthreads();

            score_mid(
                block.segmentOffset,
                block.segmentBeamIdx,
                block.segmentPiece,
                block.segmentHeld,
                block.segmentCount,
                block.beamBoards[cur],
                block.beamClearHists[cur],
                model,
                block.midScores,
                block.hasChild,
                t
            );
            __syncthreads();

            auto const nNext = select_top_block<BLOCK, MID_SLOTS, KMAX>(
                block.midScores,
                block.selectedMid,
                block.selectedMidScore,
                block.warpBest,
                t,
                search::BEAM_WIDTHS[levelIdx] // runtime level index: reads the __device__ LUT, not KS
            );

            // kept nodes with no legal continuation: leaves at this depth, keyed by root rank.
            if(t < n && ((block.hasChild >> t) & 1u) == 0) {
                unsigned long long const key =
                    search::pack_key(block.beamScores[cur][t], LEAF_ORDERS + block.beamRoots[cur][t], SPAN);
                best = key > best ? key : best;
            }

            cache_kept_mid(
                block.segmentOffset,
                block.segmentBeamIdx,
                block.segmentNode,
                block.segmentPiece,
                block.selectedMid,
                block.selectedMidScore,
                nNext,
                block.beamBoards[cur],
                block.beamClearHists[cur],
                block.beamRoots[cur],
                block.beamBoards[cur ^ 1],
                block.beamClearHists[cur ^ 1],
                block.beamScores[cur ^ 1],
                block.beamNodes[cur ^ 1],
                block.beamRoots[cur ^ 1],
                t
            );
            __syncthreads();

            cur ^= 1;
            n = nNext;
        }

        unsigned long long const winner = expand_leaves_and_reduce(
            block.slot,
            tree,
            block.beamBoards[cur],
            block.beamClearHists[cur],
            block.beamScores[cur],
            block.beamNodes[cur],
            n,
            model,
            best,
            block.warpBest,
            t,
            lane,
            wid
        );

        if(t == 0)
            commit_move(
                games,
                g,
                block.slot,
                tree,
                block.selectedRoot,
                block.beamRoots[cur],
                winner,
                stats,
                block.over
            );
        __syncthreads();
    }

    /** the beamed search kernel body: one CUDA block per (model, game). */
    __global__ void __launch_bounds__(BLOCK, 5) eval_kernel(
        sim::TetrisEngine* __restrict__ games,
        std::uint32_t num_blocks,
        std::uint32_t num_games,
        float const* __restrict__ weights,
        std::uint32_t max_moves,
        float* __restrict__ fitness
    ) {
        // one block per (model, game): block g plays game (g % numGames) with model (g / numGames).
        std::uint32_t const g = blockIdx.x;
        if(g >= num_blocks)
            return;
        std::uint32_t const t = threadIdx.x;
        std::uint32_t const lane = t & 31u;
        std::uint32_t const wid = t >> 5;

        __shared__ BlockState block;

        // the game's stat block (ai::stats_v4): committed by thread 0 on every move, scored at the end.
        // thread 0's local copy is the only live one (t==0 paths); not part of BlockState because its
        // default member initialisers make it non-trivially-default-constructible.
        ai::stats_t stats{};

        for(std::uint32_t i = t; i < ai::model_t::NUM_PARAMS; i += BLOCK)
            block.weights[i] = weights[static_cast<std::size_t>(g / num_games) * ai::model_t::NUM_PARAMS + i];

        if(t == 0)
            block.over = games[g].gameOver() ? 1 : 0;
        __syncthreads();

        net_ref const model{block.weights};

        for(std::uint32_t moveIdx = 0; moveIdx < max_moves; ++moveIdx) {
            if(block.over)
                break;
            play_move(block, games, g, t, lane, wid, model, stats);
        }

        // avg metrics divide by piecesPlaced; 0 placed pieces cannot happen on a fresh board, but guard
        // the NaN anyway with a score below anything reachable.
        if(t == 0)
            fitness[g] = stats.piecesPlaced() == 0 ? -1.0e6f : static_cast<float>(stats.score());
    }

} // namespace
} // namespace ta3::gpu
