#pragma once

#include "block_reduce.cuh"

#include <ta3/gpu/detail/value_model.hpp>

#include <ta3/ai/search/beam.hpp>

#include <ta3/ai/models/metrics/tetris_stats_v4.hpp>

#include <ta3/sim/tetris_engine.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <type_traits>

/**
 * @file eval_kernel.cuh
 * @brief the beamed search kernel: one block per (model, game), depth-agnostic cooperative stages per move.
*/
namespace ta3::gpu {
namespace {

    namespace search = ta3::ai::search;
    namespace sim = ta3::sim;

    using search::plan_tree;

    constexpr std::uint32_t BLOCK = 128;
    constexpr std::uint32_t WARPS = BLOCK / 32;

    constexpr std::uint32_t DEPTH = search::DEPTH;
    // scalar folds of the per-level widths (search::BEAM_WIDTHS); the runtime-indexed access in the
    // middle-stage loop reads search::BEAM_WIDTHS itself, which is TA3_CUDA_CONSTANT for exactly that.
    constexpr std::uint32_t K1 = search::BEAM_WIDTHS[0];
    constexpr std::uint32_t KMAX = search::max_width(search::BEAM_WIDTHS);
    constexpr std::uint32_t ROOT_SLOTS = search::ROOT_SLOTS;
    constexpr std::uint32_t MID_SLOTS = search::mid_slots(search::BEAM_WIDTHS);
    constexpr std::uint32_t LEAF_ORDERS = search::leaf_orders(search::last_width(search::BEAM_WIDTHS));
    constexpr std::uint32_t SPAN = search::order_span(search::BEAM_WIDTHS);
    constexpr std::uint32_t MP = search::MAX_PLACEMENTS;

    /** middle-stage segment capacity: one segment per (kept frontier node, child). */
    constexpr std::uint32_t SEG = KMAX * search::MAX_CHILD_MID;

    static_assert(KMAX <= 32, "the fallback mask and the frontier caches assume ranks fit one 32-bit mask");
    static_assert(KMAX <= BLOCK, "frontier caching runs one thread per kept node");
    static_assert(search::MAX_LAST <= 32, "the leaf-stage legality mask assumes the last-piece lists fit one u32");
    static_assert(SEG * MP <= 0xFFFFu, "segment offsets are u16");
    static_assert(std::is_trivially_copyable_v<sim::TetrisEngine>, "engine must be blittable to the device");

    /**
     * per-move staging: thread 0 refreshes the committed board / held piece / plan tree and gathers the
     * concrete pieces per abstract window slot; every thread zeroes its stride of the root-value array so
     * stale scores from the previous move (or previous game, for slots the current tree doesn't use) never
     * leak into this move's selection. (the middle-stage value array is zeroed per level instead.)
     */
    __device__ void reset_move_state(
        sim::TetrisEngine* __restrict__ games,
        std::uint32_t g,
        sim::Board2& boardS,
        std::uint32_t& treeIdxS,
        std::span<sim::PieceType, search::NUM_SLOTS> slotS,
        std::span<std::uint32_t, ROOT_SLOTS> rootValsS,
        std::uint32_t t
    ) {
        if(t == 0) {
            boardS = games[g].board();
            auto const held = games[g].heldPiece();
            treeIdxS = held == sim::TetrisEngine::NO_PIECE ? 1u : 0u;
            auto const gathered = search::gather_slots(games[g].currentPiece(), held, games[g].lookahead());
            for(std::uint32_t i = 0; i < search::NUM_SLOTS; ++i)
                slotS[i] = gathered[i];
        }
        for(std::uint32_t i = t; i < ROOT_SLOTS; i += BLOCK)
            rootValsS[i] = 0;
    }

    /** stage 0: threads stride the root-slot space and score every legal depth-1 board. */
    __device__ void score_roots(
        sim::Board2 const& board,
        std::span<sim::PieceType const, search::NUM_SLOTS> slot_sequence,
        plan_tree const& tree,
        net_ref const& model,
        std::span<std::uint32_t, ROOT_SLOTS> root_value_sequence,
        std::uint32_t t
    ) {
        for(std::uint32_t i = t; i < ROOT_SLOTS; i += BLOCK) {
            auto const g = i / MP, i0 = i % MP;
            if(g >= tree.count[0])
                continue;
            auto const& node = tree.nodes[0][g];
            auto const p0 = slot_sequence[node.slot];
            if(i0 >= search::theoretical_placement_count(p0))
                continue;
            auto const [nextBoard, nextClearHist, legal] = search::apply(board, {}, search::nth_placement(p0, i0));
            if(legal) {
                bool const heldIsI = node.heldSlot == search::HELD_NONE_SLOT
                                     ? false
                                     : slot_sequence[node.heldSlot] == sim::PieceType::I;
                root_value_sequence[i] = search::ordered_bits(model.evaluate(nextClearHist, nextBoard, heldIsI));
            }
        }
    }

    /**
     * seed the kept frontier from the selected roots (one cheap re-apply beats storing all of stage 0):
     * board, accum, value, tree node index and root rank per kept root, into frontier row 0.
     */
    __device__ void cache_kept_roots(
        sim::Board2 const& boardS,
        std::span<sim::PieceType const, search::NUM_SLOTS> slotS,
        plan_tree const& tree,
        std::span<std::uint32_t const, K1> sel0S,
        std::span<std::uint32_t const, K1> sel0ValS,
        std::uint32_t n0,
        std::span<sim::Board2, KMAX> kBoard,
        std::span<search::clear_t, KMAX> kAccum,
        std::span<std::uint32_t, KMAX> kVal,
        std::span<std::uint8_t, KMAX> kNode,
        std::span<std::uint8_t, KMAX> kRoot,
        std::uint32_t t
    ) {
        if(t < n0) {
            std::uint32_t const g = sel0S[t] / MP;
            auto const firstPiece = slotS[tree.nodes[0][g].slot];
            auto const step = search::apply(boardS, {}, search::nth_placement(firstPiece, sel0S[t] % MP));
            kBoard[t] = step.nextBoard;
            kAccum[t] = step.nextClearHist;
            kVal[t] = sel0ValS[t];
            kNode[t] = static_cast<std::uint8_t>(g);
            kRoot[t] = static_cast<std::uint8_t>(t);
        }
    }

    /**
     * build one middle level's segment table: one segment per (kept frontier node k, child c), spanning
     * that child piece's REAL theoretical placement count. compacted work ids enumerate (k, c, i)
     * lexicographically -- order-isomorphic to the host's padded slot ids, so the later top-K selection
     * (max value, lowest id on ties) picks the identical sequence. thread 0 only (<= SEG trivial
     * iterations, once per level); the caller owes the barrier.
     */
    __device__ void build_mid_segments(
        std::span<sim::PieceType const, search::NUM_SLOTS> slotS,
        plan_tree const& tree,
        std::uint32_t level,
        std::span<std::uint8_t const, KMAX> kNode,
        std::uint32_t n,
        std::span<std::uint16_t, SEG + 1> seg_off,
        std::span<std::uint8_t, SEG> seg_k,
        std::span<std::uint8_t, SEG> seg_node,
        std::span<sim::PieceType, SEG> seg_piece,
        std::span<std::uint8_t, SEG> seg_held,
        std::uint32_t& segCountS
    ) {
        std::uint32_t ns = 0;
        std::uint16_t off = 0;
        for(std::uint32_t k = 0; k < n; ++k) {
            auto const& pnode = tree.nodes[level - 1][kNode[k]];
            for(std::uint32_t c = pnode.childBegin; c < pnode.childEnd; ++c) {
                auto const& cnode = tree.nodes[level][c];
                auto const p = slotS[cnode.slot];
                seg_off[ns] = off;
                seg_k[ns] = static_cast<std::uint8_t>(k);
                seg_node[ns] = static_cast<std::uint8_t>(c);
                seg_piece[ns] = p;
                seg_held[ns] = cnode.heldSlot != search::HELD_NONE_SLOT
                    && slotS[cnode.heldSlot] == sim::PieceType::I;
                off = static_cast<std::uint16_t>(off + search::theoretical_placement_count(p));
                ++ns;
            }
        }
        seg_off[ns] = off;
        segCountS = ns;
    }

    /**
     * middle stage scoring: threads stride the COMPACTED continuation space (see @ref build_mid_segments)
     * and score every legal board -- every work id is a real (kept node, child, placement) triple, no
     * padding lanes. the segment cursor only moves forward because work ids increase with the stride, so
     * the decode costs O(segments) over the whole loop, not per item.
     */
    __device__ void score_mid(
        std::span<std::uint16_t const, SEG + 1> seg_off,
        std::span<std::uint8_t const, SEG> seg_k,
        std::span<sim::PieceType const, SEG> seg_piece,
        std::span<std::uint8_t const, SEG> seg_held,
        std::uint32_t seg_count,
        std::span<sim::Board2 const, KMAX> kBoard,
        std::span<search::clear_t const, KMAX> kAccum,
        net_ref const& model,
        std::span<std::uint32_t, MID_SLOTS> midValsS,
        std::uint32_t& hasChildS,
        std::uint32_t t
    ) {
        std::uint32_t const total = seg_off[seg_count];
        std::uint32_t j = 0;
        for(std::uint32_t i = t; i < total; i += BLOCK) {
            while(seg_off[j + 1] <= i)
                ++j;
            auto const k = seg_k[j];
            auto const [nextBoard, nextClearHist, legal] = search::apply(
                kBoard[k],
                kAccum[k],
                search::nth_placement(seg_piece[j], i - seg_off[j])
            );
            if(legal) {
                midValsS[i] = search::ordered_bits(model.evaluate(nextClearHist, nextBoard, seg_held[j] != 0));
                if(((hasChildS >> k) & 1u) == 0) // racy pre-check: only ever skips an already-set bit
                    atomicOr(&hasChildS, 1u << k);
            }
        }
    }

    /**
     * advance the kept frontier one level: decode each kept continuation's (rank, tree node) from its
     * compacted id exactly once, re-apply its placement, and write the next frontier row -- carrying the
     * root rank forward so the final decode never needs per-level history.
     */
    __device__ void cache_kept_mid(
        std::span<std::uint16_t const, SEG + 1> seg_off,
        std::span<std::uint8_t const, SEG> seg_k,
        std::span<std::uint8_t const, SEG> seg_node,
        std::span<sim::PieceType const, SEG> seg_piece,
        std::span<std::uint32_t const, KMAX> selMS,
        std::span<std::uint32_t const, KMAX> selMValS,
        std::uint32_t nNext,
        std::span<sim::Board2 const, KMAX> kBoard,
        std::span<search::clear_t const, KMAX> kAccum,
        std::span<std::uint8_t const, KMAX> kRoot,
        std::span<sim::Board2, KMAX> nBoard,
        std::span<search::clear_t, KMAX> nAccum,
        std::span<std::uint32_t, KMAX> nVal,
        std::span<std::uint8_t, KMAX> nNode,
        std::span<std::uint8_t, KMAX> nRoot,
        std::uint32_t t
    ) {
        if(t < nNext) {
            std::uint32_t const idx = selMS[t];
            std::uint32_t j = 0;
            while(seg_off[j + 1] <= idx)
                ++j;
            std::uint32_t const k = seg_k[j];
            search::step const step = search::apply(
                kBoard[k],
                kAccum[k],
                search::nth_placement(seg_piece[j], idx - seg_off[j])
            );
            nBoard[t] = step.nextBoard;
            nAccum[t] = step.nextClearHist;
            nVal[t] = selMValS[t];
            nNode[t] = seg_node[j];
            nRoot[t] = kRoot[k];
        }
    }

    /**
     * leaf stage: one warp per kept frontier node expands the last piece. the (list, placement) pairs are
     * FLATTENED: lanes stride the node's total placement count across all last-piece lists, so a
     * 17-placement list no longer leaves half the warp idle before the next list starts. leaves (plus the
     * per-list fallbacks) fold into the per-thread packed (value | order) maxima -- seeded with @p best,
     * which already carries any mid-level fallbacks this thread emitted -- combined by ONE block
     * reduction. the per-list fallback ("this list's piece never fit") comes from a warp-wide legality
     * bitmask (bit per list, shuffle-OR across lanes), identical semantics to the host's per-list
     * @c !expanded branch. returns the winning packed key (never 0 whenever n0 > 0).
     */
    __device__ unsigned long long expand_leaves_and_reduce(
        std::span<sim::PieceType const, search::NUM_SLOTS> slotS,
        plan_tree const& tree,
        std::span<sim::Board2 const, KMAX> kBoard,
        std::span<search::clear_t const, KMAX> kAccum,
        std::span<std::uint32_t const, KMAX> kVal,
        std::span<std::uint8_t const, KMAX> kNode,
        std::uint32_t n,
        net_ref const& model,
        unsigned long long best,
        std::span<unsigned long long, WARPS> warpBestS,
        std::uint32_t t,
        std::uint32_t lane,
        std::uint32_t wid
    ) {
        auto const consider = [&](std::uint32_t v_ordered, std::uint32_t order) {
            unsigned long long const key = search::pack_key(v_ordered, order, SPAN);
            best = key > best ? key : best;
        };

        for(std::uint32_t j = wid; j < n; j += WARPS) { // warp per kept node, lanes over flat placements
            auto const& lnode = tree.nodes[DEPTH - 2][kNode[j]];
            std::uint32_t const lists = lnode.childEnd - lnode.childBegin;

            std::uint32_t total = 0;
            for(std::uint32_t li = 0; li < lists; ++li)
                total += search::theoretical_placement_count(slotS[tree.nodes[DEPTH - 1][lnode.childBegin + li].slot]);

            std::uint32_t anyLegal = 0; // bit li: some placement of list li fit

            for(std::uint32_t idx = lane; idx < total; idx += 32) {
                // decode (li, i) by walking the (<= MAX_LAST) list counts -- no runtime-indexed array
                std::uint32_t i = idx, li = 0;
                sim::PieceType p = slotS[tree.nodes[DEPTH - 1][lnode.childBegin].slot];
                for(std::uint32_t c = search::theoretical_placement_count(p); i >= c; c = search::theoretical_placement_count(p)) {
                    i -= c;
                    p = slotS[tree.nodes[DEPTH - 1][lnode.childBegin + ++li].slot];
                }

                auto const [nextBoard, nextClearHist, legal] = search::apply(kBoard[j], kAccum[j], search::nth_placement(p, i));
                if(!legal)
                    continue;
                anyLegal |= 1u << li;

                auto const& cnode = tree.nodes[DEPTH - 1][lnode.childBegin + li];
                bool const heldIsI = cnode.heldSlot == search::HELD_NONE_SLOT
                                     ? false
                                     : slotS[cnode.heldSlot] == sim::PieceType::I;
                consider(
                    search::ordered_bits(model.evaluate(nextClearHist, nextBoard, heldIsI)),
                    (j * search::MAX_LAST + li) * MP + i
                );
            }
            // per-list fallback: the piece never fit anywhere -> the frontier board is the leaf.
#pragma unroll
            for(int o = 16; o > 0; o >>= 1)
                anyLegal |= __shfl_xor_sync(0xFFFFFFFFu, anyLegal, o);
            if(lane == 0)
                for(std::uint32_t li = 0; li < lists; ++li)
                    if(((anyLegal >> li) & 1u) == 0)
                        consider(kVal[j], (j * search::MAX_LAST + li) * MP);
        }

        return block_max_bcast<BLOCK>(best, warpBestS, t);
    }

    /** decode the winning (value | order) key back to the root move and advance the engine. thread 0 only. */
    __device__ void commit_move(
        sim::TetrisEngine* __restrict__ games,
        std::uint32_t g,
        std::span<sim::PieceType const, search::NUM_SLOTS> slotS,
        plan_tree const& tree,
        std::span<std::uint32_t const, K1> sel0S,
        std::span<std::uint8_t const, KMAX> kRoot,
        unsigned long long winner,
        ai::stats_v4& stats,
        int& overS
    ) {
        std::uint32_t const order = search::key_pos(winner, SPAN);
        std::uint32_t const rootRank = order >= LEAF_ORDERS
                                       ? order - LEAF_ORDERS
                                       : kRoot[order / (search::MAX_LAST * MP)];

        auto const& rootNode = tree.nodes[0][sel0S[rootRank] / MP];
        search::placement const pl0 = search::nth_placement(slotS[rootNode.slot], sel0S[rootRank] % MP);

        if(rootNode.rootHold)
            games[g].hold();
        std::size_t const cleared = games[g].place(pl0.orientation, pl0.x);

        if(cleared == sim::TetrisEngine::DIED) { overS = 1; }
        else {
            stats.advance(games[g].board(), static_cast<std::uint32_t>(cleared));
            if(games[g].gameOver())
                overS = 1;
        }
    }

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

        // ---- shared state: the value net + one move's beam ---------------------------------------
        __shared__ ai::data_t weightsS[ai::model_t::NUM_PARAMS];
        __shared__ sim::Board2 boardS; // the committed board
        __shared__ sim::PieceType slotS[search::NUM_SLOTS]; // pieces per abstract window slot
        __shared__ std::uint32_t treeIdxS; // which PLAN_TREE (held empty?)

        __shared__ std::uint32_t rootValsS[ROOT_SLOTS];
        __shared__ std::uint32_t midValsS[MID_SLOTS];
        __shared__ std::uint32_t sel0S[K1], sel0ValS[K1];
        __shared__ std::uint32_t selMS[KMAX], selMValS[KMAX];

        // the kept frontier, ping-ponged level to level: board, accum, value, tree node, root rank.
        __shared__ sim::Board2 kBoardS[2][KMAX];
        __shared__ search::clear_t kAccumS[2][KMAX];
        __shared__ std::uint32_t kValS[2][KMAX];
        __shared__ std::uint8_t kNodeS[2][KMAX];
        __shared__ std::uint8_t kRootS[2][KMAX];
        __shared__ std::uint32_t hasChildS; // bit k: kept frontier node k has a legal continuation

        // middle-stage compaction: segment table over the frontier's real child x placement spans
        // -- see build_mid_segments / cache_kept_mid.
        __shared__ std::uint16_t segOffS[SEG + 1];
        __shared__ std::uint8_t segKS[SEG];
        __shared__ std::uint8_t segNodeS[SEG];
        __shared__ sim::PieceType segPieceS[SEG];
        __shared__ std::uint8_t segHeldS[SEG]; // per-segment: child's hold slot resolves to an I piece
        __shared__ std::uint32_t segCountS;

        __shared__ unsigned long long warpBestS[WARPS];
        __shared__ int overS;

        // the game's stat block (ai::stats_v4): committed by thread 0 on every move, scored at the end.
        // thread 0's local copy is the only live one (t==0 paths); not __shared__ because its default
        // member initialisers make it non-trivially-default-constructible.
        ai::stats_v4 stats{};

        for(std::uint32_t i = t; i < ai::model_t::NUM_PARAMS; i += BLOCK)
            weightsS[i] = weights[static_cast<std::size_t>(g / num_games) * ai::model_t::NUM_PARAMS + i];
        if(t == 0) { overS = games[g].gameOver() ? 1 : 0; }
        __syncthreads();

        net_ref const model{weightsS};

        for(std::uint32_t mv = 0; mv < max_moves; ++mv) {
            if(overS)
                break;

            // ---- per-move staging ----------------------------------------------------------------
            reset_move_state(games, g, boardS, treeIdxS, slotS, rootValsS, t);
            __syncthreads();

            plan_tree const& tree = search::dev::PLAN_TREES[treeIdxS];

            // ---- stage 0: score all roots, keep the best K1 ---------------------------------------
            score_roots(boardS, slotS, tree, model, rootValsS, t);
            __syncthreads();

            std::uint32_t const n0 = select_top_block<BLOCK, ROOT_SLOTS, K1>(
                rootValsS,
                sel0S,
                sel0ValS,
                warpBestS,
                t,
                K1
            );
            if(n0 == 0) { // no legal root placement: the game is over (matches the host's `r.none`)
                if(t == 0)
                    overS = 1;
                __syncthreads();
                continue;
            }

            cache_kept_roots(
                boardS,
                slotS,
                tree,
                sel0S,
                sel0ValS,
                n0,
                kBoardS[0],
                kAccumS[0],
                kValS[0],
                kNodeS[0],
                kRootS[0],
                t
            );
            // publish the seeded frontier: the next reader (build_mid_segments' kNode scan on thread 0,
            // or the leaf stage directly when DEPTH == 2) crosses threads.
            __syncthreads();

            // per-thread packed (value | order) maximum: mid-level fallbacks fold in here as they are
            // discovered; the leaf stage adds the leaves and runs the single block reduction.
            unsigned long long best = 0;
            std::uint32_t cur = 0, n = n0;

            // ---- middle stages: score the frontier's continuations, keep the best KS[l] -----------
            for(std::uint32_t l = 1; l + 1 < DEPTH; ++l) {
                for(std::uint32_t i = t; i < MID_SLOTS; i += BLOCK)
                    midValsS[i] = 0;
                if(t == 0) {
                    hasChildS = 0;
                    build_mid_segments(
                        slotS,
                        tree,
                        l,
                        kNodeS[cur],
                        n,
                        segOffS,
                        segKS,
                        segNodeS,
                        segPieceS,
                        segHeldS,
                        segCountS
                    );
                }
                __syncthreads();

                score_mid(
                    segOffS,
                    segKS,
                    segPieceS,
                    segHeldS,
                    segCountS,
                    kBoardS[cur],
                    kAccumS[cur],
                    model,
                    midValsS,
                    hasChildS,
                    t
                );
                __syncthreads();

                std::uint32_t const nNext = select_top_block<BLOCK, MID_SLOTS, KMAX>(
                    midValsS,
                    selMS,
                    selMValS,
                    warpBestS,
                    t,
                    search::BEAM_WIDTHS[l] // runtime level index: reads the __device__ LUT
                );

                // kept nodes with no legal continuation: leaves at this depth, keyed by root rank.
                if(t < n && ((hasChildS >> t) & 1u) == 0) {
                    unsigned long long const key =
                        search::pack_key(kValS[cur][t], LEAF_ORDERS + kRootS[cur][t], SPAN);
                    best = key > best ? key : best;
                }

                cache_kept_mid(
                    segOffS,
                    segKS,
                    segNodeS,
                    segPieceS,
                    selMS,
                    selMValS,
                    nNext,
                    kBoardS[cur],
                    kAccumS[cur],
                    kRootS[cur],
                    kBoardS[cur ^ 1],
                    kAccumS[cur ^ 1],
                    kValS[cur ^ 1],
                    kNodeS[cur ^ 1],
                    kRootS[cur ^ 1],
                    t
                );
                __syncthreads();

                cur ^= 1;
                n = nNext;
            }

            // ---- leaf stage: expand the kept frontier; fold into per-thread packed maxima ----------
            unsigned long long const winner = expand_leaves_and_reduce(
                slotS,
                tree,
                kBoardS[cur],
                kAccumS[cur],
                kValS[cur],
                kNodeS[cur],
                n,
                model,
                best,
                warpBestS,
                t,
                lane,
                wid
            );

            // ---- commit: decode (value | order) back to the root move, advance the engine ---------
            if(t == 0)
                commit_move(games, g, slotS, tree, sel0S, kRootS[cur], winner, stats, overS);
            __syncthreads();
        }

        // avg metrics divide by piecesPlaced; 0 placed pieces cannot happen on a fresh board, but guard
        // the NaN anyway with a score below anything reachable.
        if(t == 0)
            fitness[g] = stats.piecesPlaced() == 0 ? -1.0e6f : static_cast<float>(stats.score());
    }

} // namespace
} // namespace ta3::gpu
