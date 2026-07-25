#pragma once

#include "block_reduce.hpp"


#include <ta3/gpu/detail/value_model.hpp>

#include <ta3/ai/search/beam.hpp>

#include <ta3/sim/tetris_engine.hpp>

#include <sycl/sycl.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>
//TEMP continue refactoring this file until its understanable and usable code, current is very much trash
/**
 * @file eval_kernel.hpp
 * @brief the beamed search kernel: one SYCL work-group per (model, game), depth-agnostic cooperative
 *  stages per move.
 * @details one work-group-scope @c sycl::local_accessor<WorkGroupState, 0> replaces CUDA's individual
 *  @c __shared__ variables; sub-group primitives (never assumed 32-wide) replace warp-level constructs.
 *  parameters that alias this work-group-local storage carry a @c wgroup_ prefix, to set them apart
 *  from genuinely per-thread values (@c t, @c g, @c n, ...) in the same signature. all function
 *  parameters use all_lower snake_case; locals (including struct members) use camelCase. loop
 *  variables follow @c i / @c j / @c k by nesting depth, or an @c <expressiveName>Idx when a bare
 *  letter would lose meaning.
 */
namespace ta3::gpu {
namespace {

    namespace search = ta3::ai::search;
    namespace sim = ta3::sim;

    using search::plan_tree;

    constexpr uint32_t BLOCK = 128;

    constexpr uint32_t DEPTH = search::DEPTH;
    constexpr search::beam_widths_t KS = search::BEAM_WIDTHS; // per-level widths; KS[0] = roots
    constexpr uint32_t K1 = KS[0];
    constexpr uint32_t KMAX = search::max_width(KS);
    constexpr uint32_t ROOT_SLOTS = search::ROOT_SLOTS;
    constexpr uint32_t MID_SLOTS = search::mid_slots(KS);
    constexpr uint32_t LEAF_ORDERS = search::leaf_orders(search::last_width(KS));
    constexpr uint32_t SPAN = search::order_span(KS);
    constexpr uint32_t MP = search::MAX_PLACEMENTS;

    /** middle-stage segment capacity: one segment per (kept frontier node, child). */
    constexpr uint32_t SEGMENT_CAP = KMAX * search::MAX_CHILD_MID;

    static_assert(KMAX <= 32, "the fallback mask and the frontier caches assume ranks fit one 32-bit mask");
    static_assert(KMAX <= BLOCK, "frontier caching runs one thread per kept node");
    static_assert(search::MAX_LAST <= 32, "the leaf-stage legality mask assumes the last-piece lists fit one u32");
    static_assert(SEGMENT_CAP * MP <= 0xFFFFu, "segment offsets are u16");
    static_assert(std::is_trivially_copyable_v<sim::TetrisEngine>, "engine must be blittable to the device");

    /** every per-work-group scratch variable, gathered into one struct for a single local_accessor
     *  allocation. @c warpBest is sized to the full work-group, not BLOCK/32 -- see block_reduce.hpp. */
    struct WorkGroupState {
        std::array<ai::data_t, ai::model_t::NUM_PARAMS> weights;
        sim::Board2 board; // the committed board
        std::array<sim::PieceType, search::NUM_SLOTS> slot; // pieces per abstract window slot
        uint32_t treeIdx; // which PLAN_TREE (held empty?)

        std::array<uint32_t, ROOT_SLOTS> rootScores;
        std::array<uint32_t, MID_SLOTS> midScores;
        std::array<uint32_t, K1> selectedRoot, selectedRootScore;
        std::array<uint32_t, KMAX> selectedMid;
        std::array<uint32_t, KMAX> selectedMidScore;

        // the kept frontier, ping-ponged level to level: board, clear history, score, tree node, root rank.
        std::array<std::array<sim::Board2, KMAX>, 2> beamBoards;
        std::array<std::array<search::clear_t, KMAX>, 2> beamClearHists;
        std::array<std::array<uint32_t, KMAX>, 2> beamScores;
        std::array<std::array<uint8_t, KMAX>, 2> beamNodes;
        std::array<std::array<uint8_t, KMAX>, 2> beamRoots;
        uint32_t hasChild; // bit i: beam slot i has a legal continuation

        // middle-stage compaction: segment table over the frontier's real child x placement spans
        // -- see build_mid_segments / cache_kept_mid.
        std::array<uint16_t, SEGMENT_CAP + 1> segmentOffset;
        std::array<uint8_t, SEGMENT_CAP> segmentBeamIdx;
        std::array<uint8_t, SEGMENT_CAP> segmentNode;
        std::array<sim::PieceType, SEGMENT_CAP> segmentPiece;
        std::array<uint8_t, SEGMENT_CAP> segmentHeld; // per-segment: child's hold slot resolves to an I piece
        uint32_t segmentCount;

        std::array<unsigned long long, BLOCK> warpBest; // see block_reduce.hpp's file-level note on sizing
        int over;
    };

    /**
     * per-move staging: thread 0 refreshes the committed board / held piece / plan tree and gathers the
     * concrete pieces per abstract window slot; every thread zeroes its stride of the root-score array so
     * stale scores from the previous move (or previous game, for slots the current tree doesn't use) never
     * leak into this move's selection. (the middle-stage score array is zeroed per level instead.)
     */
    inline void reset_move_state(
        sim::TetrisEngine* __restrict__ games,
        uint32_t g,
        sim::Board2& wgroup_board,
        uint32_t& wgroup_tree_idx,
        std::span<sim::PieceType, search::NUM_SLOTS> wgroup_slot,
        std::span<uint32_t, ROOT_SLOTS> wgroup_root_scores,
        uint32_t t
    ) {
        if(t == 0) {
            wgroup_board = games[g].board();
            auto const held = games[g].heldPiece();
            wgroup_tree_idx = held == sim::TetrisEngine::NO_PIECE ? 1u : 0u;
            auto const gathered = search::gather_slots(games[g].currentPiece(), held, games[g].lookahead());
            for(uint32_t i = 0; i < search::NUM_SLOTS; ++i)
                wgroup_slot[i] = gathered[i];
        }
        for(uint32_t i = t; i < ROOT_SLOTS; i += BLOCK)
            wgroup_root_scores[i] = 0;
    }

    /** stage 0: threads stride the root-slot space and score every legal depth-1 board. */
    inline void score_roots(
        sim::Board2 const& wgroup_board,
        std::span<sim::PieceType const, search::NUM_SLOTS> wgroup_slot,
        plan_tree const& tree,
        net_ref const& model,
        std::span<uint32_t, ROOT_SLOTS> wgroup_root_scores,
        uint32_t t
    ) {
        for(uint32_t slotIdx = t; slotIdx < ROOT_SLOTS; slotIdx += BLOCK) {
            auto const g = slotIdx / MP;
            auto const rootPlacementIdx = slotIdx % MP;
            if(g >= tree.count[0])
                continue;
            auto const& node = tree.nodes[0][g];
            auto const rootPiece = wgroup_slot[node.slot];
            if(rootPlacementIdx >= search::n_theoretical_placements(rootPiece))
                continue;
            auto const [nextBoard, nextClearHist, legal] = search::apply(
                wgroup_board,
                {},
                search::nth_placement(
                    rootPiece,
                    rootPlacementIdx
                )
            );
            if(legal) {
                auto const heldIsI = node.heldSlot == search::HELD_NONE_SLOT
                                     ? false
                                     : wgroup_slot[node.heldSlot] == sim::PieceType::I;
                wgroup_root_scores[slotIdx] = search::ordered_bits(
                    model.evaluate(nextClearHist, nextBoard, heldIsI)
                );
            }
        }
    }

    /**
     * seed the kept frontier from the selected roots (one cheap re-apply beats storing all of stage 0):
     * board, clear history, score, tree node index and root rank per kept root, into frontier row 0.
     */
    inline void cache_kept_roots(
        sim::Board2 const& wgroup_board,
        std::span<sim::PieceType const, search::NUM_SLOTS> wgroup_slot,
        plan_tree const& tree,
        std::span<uint32_t const, K1> wgroup_selected_root,
        std::span<uint32_t const, K1> wgroup_selected_root_score,
        uint32_t root_count,
        std::span<sim::Board2, KMAX> wgroup_beam_boards,
        std::span<search::clear_t, KMAX> wgroup_beam_clear_hists,
        std::span<uint32_t, KMAX> wgroup_beam_scores,
        std::span<uint8_t, KMAX> wgroup_beam_nodes,
        std::span<uint8_t, KMAX> wgroup_beam_roots,
        uint32_t t
    ) {
        if(t < root_count) {
            auto const g = wgroup_selected_root[t] / MP;
            auto const rootPiece = wgroup_slot[tree.nodes[0][g].slot];
            auto const rootStep = search::apply(
                wgroup_board,
                {},
                search::nth_placement(rootPiece, wgroup_selected_root[t] % MP)
            );
            wgroup_beam_boards[t] = rootStep.nextBoard;
            wgroup_beam_clear_hists[t] = rootStep.nextClearHist;
            wgroup_beam_scores[t] = wgroup_selected_root_score[t];
            wgroup_beam_nodes[t] = static_cast<uint8_t>(g);
            wgroup_beam_roots[t] = static_cast<uint8_t>(t);
        }
    }

    /**
     * build one middle level's segment table: one segment per (kept frontier node, child), spanning
     * that child piece's REAL theoretical placement count. compacted work ids enumerate (beam index,
     * child index, placement index) lexicographically -- order-isomorphic to the host's padded slot
     * ids, so the later top-K selection (max score, lowest id on ties) picks the identical sequence.
     * thread 0 only (<= SEGMENT_CAP trivial iterations, once per level); the caller owes the barrier.
     */
    inline void build_mid_segments(
        std::span<sim::PieceType const, search::NUM_SLOTS> wgroup_slot,
        plan_tree const& tree,
        uint32_t level,
        std::span<uint8_t const, KMAX> wgroup_beam_nodes,
        uint32_t beam_width,
        std::span<uint16_t, SEGMENT_CAP + 1> wgroup_segment_offset,
        std::span<uint8_t, SEGMENT_CAP> wgroup_segment_beam_idx,
        std::span<uint8_t, SEGMENT_CAP> wgroup_segment_node,
        std::span<sim::PieceType, SEGMENT_CAP> wgroup_segment_piece,
        std::span<uint8_t, SEGMENT_CAP> wgroup_segment_held,
        uint32_t& wgroup_segment_count
    ) {
        uint32_t ns = 0;
        uint16_t off = 0;
        for(uint32_t i = 0; i < beam_width; ++i) {
            auto const& pnode = tree.nodes[level - 1][wgroup_beam_nodes[i]];
            for(uint32_t j = pnode.childBegin; j < pnode.childEnd; ++j) {
                auto const& cnode = tree.nodes[level][j];
                auto const p = wgroup_slot[cnode.slot];
                wgroup_segment_offset[ns] = off;
                wgroup_segment_beam_idx[ns] = static_cast<uint8_t>(i);
                wgroup_segment_node[ns] = static_cast<uint8_t>(j);
                wgroup_segment_piece[ns] = p;
                wgroup_segment_held[ns] = cnode.heldSlot != search::HELD_NONE_SLOT
                    && wgroup_slot[cnode.heldSlot] == sim::PieceType::I;
                off = static_cast<uint16_t>(off + search::n_theoretical_placements(p));
                ++ns;
            }
        }
        wgroup_segment_offset[ns] = off;
        wgroup_segment_count = ns;
    }

    /**
     * middle stage scoring: threads stride the COMPACTED continuation space (see @ref build_mid_segments)
     * and score every legal board -- every work id is a real (kept node, child, placement) triple, no
     * padding lanes. the segment cursor only moves forward because work ids increase with the stride, so
     * the decode costs O(segments) over the whole loop, not per item.
     */
    inline void score_mid(
        std::span<uint16_t const, SEGMENT_CAP + 1> wgroup_segment_offset,
        std::span<uint8_t const, SEGMENT_CAP> wgroup_segment_beam_idx,
        std::span<sim::PieceType const, SEGMENT_CAP> wgroup_segment_piece,
        std::span<uint8_t const, SEGMENT_CAP> wgroup_segment_held,
        uint32_t segment_count,
        std::span<sim::Board2 const, KMAX> wgroup_beam_boards,
        std::span<search::clear_t const, KMAX> wgroup_beam_clear_hists,
        net_ref const& model,
        std::span<uint32_t, MID_SLOTS> wgroup_mid_scores,
        uint32_t& wgroup_has_child,
        uint32_t t
    ) {
        // segmentOffset is u16 (packed, see SEGMENT_CAP's static_assert); widened to u32 on purpose so the
        // stride comparison against BLOCK-stepped `i` below never narrows.
        uint32_t const total = wgroup_segment_offset[segment_count];
        uint32_t j = 0;
        for(uint32_t i = t; i < total; i += BLOCK) {
            while(wgroup_segment_offset[j + 1] <= i)
                ++j;
            // segmentBeamIdx is u8; widened to u32 on purpose to index the u32-addressed beam arrays below.
            uint32_t const beamIdx = wgroup_segment_beam_idx[j];
            auto const s = search::apply(
                wgroup_beam_boards[beamIdx],
                wgroup_beam_clear_hists[beamIdx],
                search::nth_placement(wgroup_segment_piece[j], i - wgroup_segment_offset[j])
            );
            if(s.legal) {
                wgroup_mid_scores[i] = search::ordered_bits(
                    model.evaluate(s.nextClearHist, s.nextBoard, wgroup_segment_held[j] != 0)
                );
                if(((wgroup_has_child >> beamIdx) & 1u) == 0) { // racy pre-check: only ever skips an already-set bit
                    sycl::atomic_ref<
                        uint32_t,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::work_group,
                        sycl::access::address_space::local_space
                    > bits{wgroup_has_child};
                    bits.fetch_or(1u << beamIdx);
                }
            }
        }
    }

    /**
     * advance the kept frontier one level: decode each kept continuation's (rank, tree node) from its
     * compacted id exactly once, re-apply its placement, and write the next frontier row -- carrying the
     * root rank forward so the final decode never needs per-level history.
     */
    inline void cache_kept_mid(
        std::span<uint16_t const, SEGMENT_CAP + 1> wgroup_segment_offset,
        std::span<uint8_t const, SEGMENT_CAP> wgroup_segment_beam_idx,
        std::span<uint8_t const, SEGMENT_CAP> wgroup_segment_node,
        std::span<sim::PieceType const, SEGMENT_CAP> wgroup_segment_piece,
        std::span<uint32_t const, KMAX> wgroup_selected_mid,
        std::span<uint32_t const, KMAX> wgroup_selected_mid_score,
        uint32_t n_next,
        std::span<sim::Board2 const, KMAX> wgroup_beam_boards,
        std::span<search::clear_t const, KMAX> wgroup_beam_clear_hists,
        std::span<uint8_t const, KMAX> wgroup_beam_roots,
        std::span<sim::Board2, KMAX> wgroup_next_beam_boards,
        std::span<search::clear_t, KMAX> wgroup_next_beam_clear_hists,
        std::span<uint32_t, KMAX> wgroup_next_beam_scores,
        std::span<uint8_t, KMAX> wgroup_next_beam_nodes,
        std::span<uint8_t, KMAX> wgroup_next_beam_roots,
        uint32_t t
    ) {
        if(t >= n_next)
            return;

        auto const idx = wgroup_selected_mid[t];
        uint32_t i = 0;
        while(wgroup_segment_offset[i + 1] <= idx)
            ++i;
        // segmentBeamIdx is u8; widened to u32 on purpose to index the u32-addressed beam arrays below.
        uint32_t const beamIdx = wgroup_segment_beam_idx[i];
        auto const s = search::apply(
            wgroup_beam_boards[beamIdx],
            wgroup_beam_clear_hists[beamIdx],
            search::nth_placement(wgroup_segment_piece[i], idx - wgroup_segment_offset[i])
        );
        wgroup_next_beam_boards[t] = s.nextBoard;
        wgroup_next_beam_clear_hists[t] = s.nextClearHist;
        wgroup_next_beam_scores[t] = wgroup_selected_mid_score[t];
        wgroup_next_beam_nodes[t] = wgroup_segment_node[i];
        wgroup_next_beam_roots[t] = wgroup_beam_roots[beamIdx];
    }

    /**
     * leaf stage: one sub-group per kept frontier node expands the last piece. the (list, placement)
     * pairs are FLATTENED: lanes stride the node's total placement count across all last-piece lists, so a
     * 17-placement list no longer leaves half the sub-group idle before the next list starts. leaves (plus
     * the per-list fallbacks) fold into the per-thread packed (score | order) maxima -- seeded with
     * @p best, which already carries any mid-level fallbacks this thread emitted -- combined by ONE
     * work-group reduction. the per-list fallback ("this list's piece never fit") comes from a sub-group-
     * wide legality bitmask (bit per list, OR-reduced across lanes via @c sycl::reduce_over_group),
     * identical semantics to the host's per-list @c !expanded branch. returns the winning packed key
     * (never 0 whenever rootCount > 0). sub-group width and count are read at runtime, never assumed
     * 32-wide.
     */
    [[nodiscard]] inline unsigned long long expand_leaves_and_reduce(
        std::span<sim::PieceType const, search::NUM_SLOTS> wgroup_slot,
        plan_tree const& tree,
        std::span<sim::Board2 const, KMAX> wgroup_beam_boards,
        std::span<search::clear_t const, KMAX> wgroup_beam_clear_hists,
        std::span<uint32_t const, KMAX> wgroup_beam_scores,
        std::span<uint8_t const, KMAX> wgroup_beam_nodes,
        uint32_t beam_width,
        net_ref const& model,
        unsigned long long best,
        std::span<unsigned long long, BLOCK> wgroup_warp_best,
        sycl::nd_item<1> it
    ) {
        auto const subgroup = it.get_sub_group();
        auto const lane = static_cast<uint32_t>(subgroup.get_local_linear_id());
        auto const laneRange = static_cast<uint32_t>(subgroup.get_local_linear_range());
        auto const wid = static_cast<uint32_t>(subgroup.get_group_linear_id());
        auto const numSubgroups = static_cast<uint32_t>(subgroup.get_group_linear_range());

        auto const consider = [&](uint32_t v_ordered, uint32_t order) {
            unsigned long long const key = search::pack_key(v_ordered, order, SPAN);
            best = key > best ? key : best;
        };

        for(uint32_t i = wid; i < beam_width; i += numSubgroups) {
            // sub-group per kept node, lanes over flat placements
            auto const& lnode = tree.nodes[DEPTH - 2][wgroup_beam_nodes[i]];

            auto const lists = static_cast<uint32_t>(lnode.childEnd - lnode.childBegin);

            uint32_t total = 0;
            for(uint32_t listIdx = 0; listIdx < lists; ++listIdx)
                total += search::n_theoretical_placements(
                    wgroup_slot[tree.nodes[DEPTH - 1][lnode.childBegin + listIdx].slot]
                );

            uint32_t legal = 0; // bit listIdx: some placement of list listIdx fit
            for(uint32_t j = lane; j < total; j += laneRange) {
                // decode (listIdx, placementIdx) by walking the (<= MAX_LAST) list counts -- no runtime-indexed array
                auto placementIdx = j;
                uint32_t listIdx = 0;

                auto p = wgroup_slot[tree.nodes[DEPTH - 1][lnode.childBegin].slot];
                for(uint32_t listCount = search::n_theoretical_placements(p); placementIdx >= listCount; listCount =
                    search::n_theoretical_placements(p)) {
                    placementIdx -= listCount;
                    p = wgroup_slot[tree.nodes[DEPTH - 1][lnode.childBegin + ++listIdx].slot];
                }

                auto const [nextBoard, nextClearHist, stepLegal] = search::apply(
                    wgroup_beam_boards[i],
                    wgroup_beam_clear_hists[i],
                    search::nth_placement(p, placementIdx)
                );
                if(!stepLegal)
                    continue;
                legal |= 1u << listIdx;
                auto const& cnode = tree.nodes[DEPTH - 1][lnode.childBegin + listIdx];
                auto const heldIsI = cnode.heldSlot == search::HELD_NONE_SLOT
                                     ? false
                                     : wgroup_slot[cnode.heldSlot] == sim::PieceType::I;
                consider(
                    search::ordered_bits(model.evaluate(nextClearHist, nextBoard, heldIsI)),
                    (i * search::MAX_LAST + listIdx) * MP + placementIdx
                );
            }
            // per-list fallback: the piece never fit anywhere -> the frontier board is the leaf. combine
            // the legality mask across the sub-group.
            legal = sycl::reduce_over_group(subgroup, legal, sycl::bit_or<uint32_t>{});
            if(lane == 0)
                for(uint32_t listIdx = 0; listIdx < lists; ++listIdx)
                    if(((legal >> listIdx) & 1u) == 0)
                        consider(wgroup_beam_scores[i], (i * search::MAX_LAST + listIdx) * MP);
        }

        return block_max_bcast<BLOCK>(best, it, wgroup_warp_best);
    }

    /** decode the winning (score | order) key back to the root move and advance the engine. thread 0 only. */
    inline void commit_move(
        sim::TetrisEngine * __restrict__ games,
        uint32_t g,
        std::span<sim::PieceType const, search::NUM_SLOTS> wgroup_slot,
        plan_tree const&tree,
        std::span<uint32_t const, K1> wgroup_selected_root,
        std::span<uint8_t const, KMAX> wgroup_beam_roots,
        
    unsigned long long winner,
        ai::stats_v4&stats,
        
    int& wgroup_over
    )
    {
        auto const order = search::key_pos(winner, SPAN);
        auto const rootRank = order >= LEAF_ORDERS
                              ? order - LEAF_ORDERS
                              : wgroup_beam_roots[order / (search::MAX_LAST * MP)];

        auto const& rootNode = tree.nodes[0][wgroup_selected_root[rootRank] / MP];
        auto const rootPlacement =
            search::nth_placement(wgroup_slot[rootNode.slot], wgroup_selected_root[rootRank] % MP);

        if(rootNode.rootHold)
            games[g].hold();
        auto const cleared = games[g].place(rootPlacement.orientation, rootPlacement.x);

        if(cleared == sim::TetrisEngine::DIED) { wgroup_over = 1; }
        else {
            stats.advance(games[g].board(), static_cast<uint32_t>(cleared));
            if(games[g].gameOver())
                wgroup_over = 1;
        }
    }

    /**
     * play one move: stage 0 scores every legal root and keeps the best K1; each middle level scores
     * the frontier's continuations and keeps the best KS[level]; the leaf stage expands the last piece
     * and reduces to a single winning (score | order) key; commit decodes that key back to the root
     * move and advances the engine. thread 0 is the only one that touches @p games / @p stats directly;
     * every other thread cooperates purely through @p wgroup.
     */
    inline void play_move(
        sycl::nd_item<1> it,
        WorkGroupState& wgroup,
        sim::TetrisEngine* __restrict__ games,
        uint32_t g,
        uint32_t t,
        net_ref const& model,
        ai::stats_v4& stats
    ) {
        reset_move_state(games, g, wgroup.board, wgroup.treeIdx, wgroup.slot, wgroup.rootScores, t);
        sycl::group_barrier(it.get_group());

        auto const& tree = search::dev::PLAN_TREES[wgroup.treeIdx];

        score_roots(wgroup.board, wgroup.slot, tree, model, wgroup.rootScores, t);
        sycl::group_barrier(it.get_group());

        auto const rootCount = select_top_block<BLOCK, ROOT_SLOTS, K1>(
            wgroup.rootScores,
            wgroup.selectedRoot,
            wgroup.selectedRootScore,
            wgroup.warpBest,
            it,
            t,
            K1
        );
        if(rootCount == 0) { // no legal root placement: the game is over (matches the host's `r.none`)
            if(t == 0)
                wgroup.over = 1;
            sycl::group_barrier(it.get_group());
            return;
        }

        cache_kept_roots(
            wgroup.board,
            wgroup.slot,
            tree,
            wgroup.selectedRoot,
            wgroup.selectedRootScore,
            rootCount,
            wgroup.beamBoards[0],
            wgroup.beamClearHists[0],
            wgroup.beamScores[0],
            wgroup.beamNodes[0],
            wgroup.beamRoots[0],
            t
        );
        // publish the seeded frontier: the next reader (build_mid_segments' beamNodes scan on thread 0,
        // or the leaf stage directly when DEPTH == 2) crosses threads.
        sycl::group_barrier(it.get_group());

        // per-thread packed (score | order) maximum: mid-level fallbacks fold in here as they are
        // discovered; the leaf stage adds the leaves and runs the single work-group reduction.
        unsigned long long best = 0;
        auto n = rootCount;

        uint32_t cur = 0;

        for(uint32_t levelIdx = 1; levelIdx + 1 < DEPTH; ++levelIdx) {
            for(uint32_t k = t; k < MID_SLOTS; k += BLOCK)
                wgroup.midScores[k] = 0;
            if(t == 0) {
                wgroup.hasChild = 0;
                build_mid_segments(
                    wgroup.slot,
                    tree,
                    levelIdx,
                    wgroup.beamNodes[cur],
                    n,
                    wgroup.segmentOffset,
                    wgroup.segmentBeamIdx,
                    wgroup.segmentNode,
                    wgroup.segmentPiece,
                    wgroup.segmentHeld,
                    wgroup.segmentCount
                );
            }
            sycl::group_barrier(it.get_group());

            score_mid(
                wgroup.segmentOffset,
                wgroup.segmentBeamIdx,
                wgroup.segmentPiece,
                wgroup.segmentHeld,
                wgroup.segmentCount,
                wgroup.beamBoards[cur],
                wgroup.beamClearHists[cur],
                model,
                wgroup.midScores,
                wgroup.hasChild,
                t
            );
            sycl::group_barrier(it.get_group());

            auto const nNext = select_top_block<BLOCK, MID_SLOTS, KMAX>(
                wgroup.midScores,
                wgroup.selectedMid,
                wgroup.selectedMidScore,
                wgroup.warpBest,
                it,
                t,
                KS[levelIdx]
            );

            // kept nodes with no legal continuation: leaves at this depth, keyed by root rank.
            if(t < n && ((wgroup.hasChild >> t) & 1u) == 0) {
                unsigned long long const key =
                    search::pack_key(wgroup.beamScores[cur][t], LEAF_ORDERS + wgroup.beamRoots[cur][t], SPAN);
                best = key > best ? key : best;
            }

            cache_kept_mid(
                wgroup.segmentOffset,
                wgroup.segmentBeamIdx,
                wgroup.segmentNode,
                wgroup.segmentPiece,
                wgroup.selectedMid,
                wgroup.selectedMidScore,
                nNext,
                wgroup.beamBoards[cur],
                wgroup.beamClearHists[cur],
                wgroup.beamRoots[cur],
                wgroup.beamBoards[cur ^ 1],
                wgroup.beamClearHists[cur ^ 1],
                wgroup.beamScores[cur ^ 1],
                wgroup.beamNodes[cur ^ 1],
                wgroup.beamRoots[cur ^ 1],
                t
            );
            sycl::group_barrier(it.get_group());

            cur ^= 1;
            n = nNext;
        }

        unsigned long long const winner = expand_leaves_and_reduce(
            wgroup.slot,
            tree,
            wgroup.beamBoards[cur],
            wgroup.beamClearHists[cur],
            wgroup.beamScores[cur],
            wgroup.beamNodes[cur],
            n,
            model,
            best,
            wgroup.warpBest,
            it
        );

        if(t == 0)
            commit_move(
                games,
                g,
                wgroup.slot,
                tree,
                wgroup.selectedRoot,
                wgroup.beamRoots[cur],
                winner,
                stats,
                wgroup.over
            );
        sycl::group_barrier(it.get_group());
    }

    /** the beamed search kernel body: one SYCL work-group per (model, game). */
    inline void eval_kernel(
        sycl::nd_item<1> it,
        WorkGroupState& wgroup,
        sim::TetrisEngine* __restrict__ games,
        uint32_t num_blocks,
        uint32_t num_games,
        float const* __restrict__ weights,
        uint32_t max_moves,
        float* __restrict__ fitness
    ) {
        // one work-group per (model, game): work-group g plays game (g % numGames) with model (g / numGames).
        auto const g = static_cast<uint32_t>(it.get_group_linear_id());
        if(g >= num_blocks)
            return;
        auto const t = static_cast<uint32_t>(it.get_local_linear_id());

        // the game's stat block (ai::stats_v4): committed by thread 0 on every move, scored at the end.
        // thread 0's local copy is the only live one (t==0 paths); not part of WorkGroupState because its
        // default member initialisers make it non-trivially-default-constructible.
        ai::stats_t stats{};

        for(uint32_t i = t; i < ai::model_t::NUM_PARAMS; i += BLOCK)
            wgroup.weights[i] = weights[static_cast<size_t>(g / num_games) * ai::model_t::NUM_PARAMS + i];

        if(t == 0)
            wgroup.over = games[g].gameOver() ? 1 : 0;
        sycl::group_barrier(it.get_group());

        net_ref const model{wgroup.weights};

        for(uint32_t moveIdx = 0; moveIdx < max_moves; ++moveIdx) {
            if(wgroup.over)
                break;
            play_move(it, wgroup, games, g, t, model, stats);
        }

        // avg metrics divide by piecesPlaced; 0 placed pieces cannot happen on a fresh board, but guard
        // the NaN anyway with a score below anything reachable.
        if(t == 0)
            fitness[g] = stats.piecesPlaced() == 0 ? -1.0e6f : static_cast<float>(stats.score());
    }

} // namespace
} // namespace ta3::gpu
