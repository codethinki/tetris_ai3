#pragma once
#include "ta3/ai/search/placements.hpp"
#include "ta3/ai/search/search.hpp"
#include "ta3/ai/search/variation_sequences.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/tetris_engine.hpp>
#include <ta3/sim/utility/cuda_constant.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <span>

/**
 * @file beam.hpp
 * @brief beamed depth-DEPTH search for one move; shared core of the GPU kernel and its CPU parity
 *  reference (@ref search_move_beam).
 * @details root stage -> DEPTH-2 beamed middle stages -> leaf stage; @ref BEAM_WIDTHS sets DEPTH. hold
 *  decisions collapse at compile time into a prefix tree over abstract window slots (@ref plan_tree,
 *  from @ref dev::PLANS): beaming over tree nodes instead of (sequence, placement) pairs keeps
 *  hold-duplicated prefixes from crowding the beam. selection is "max score, lowest slot wins ties";
 *  leaves carry a total order so host and device pick identical moves bit-exactly. a kept frontier node
 *  whose next piece never fits is scored as a leaf at its own depth, keyed by root rank (all levels
 *  share one order band of width KS[0]) -- a legal root therefore always yields at least one leaf.
 */
namespace ta3::ai::search {

/** @brief small index into a plan_tree layer/node array. */
using layer_idx_t = uint8_t;

/** @brief rank of a selected root candidate: index into the top-K0 root selection. */
using root_rank_t = uint8_t;

/** @brief flat index into a placement-scoring space (a root or mid-level slot space). */
using slot_t = std::uint32_t;

/** @brief ordered_bits-encoded ranking score; see @ref ordered_bits. */
using score_t = std::uint32_t;

/** @brief position within the final decode order-key space; see @ref pack_key / @ref order_span. */
using order_t = std::uint32_t;

/**
 * @brief compile-time hold-decision prefix tree over abstract window slots; one node layer per level.
 * @details children of a node are contiguous in the next layer (stored as a range). leaves (layer
 *  DEPTH-1) correspond 1:1 to the distinct plans of @ref dev::PLANS. built by @ref dev::build_tree.
 */
struct plan_tree {
    struct node_t {
        layer_idx_t slot = 0; ///< window slot placed at this level
        layer_idx_t parentIdx = 0; ///< node index at the previous layer (0 at layer 0)
        layer_idx_t childBegin = 0; ///< child range [childBegin, childEnd) at the next layer
        layer_idx_t childEnd = 0; ///< (empty at the last layer)
        bool rootHold = false; ///< whether reaching this node commits a hold (layer 0 only)
        layer_idx_t heldSlot = HELD_NONE_SLOT;
        ///< window slot in hold after this level's decision (HELD_NONE_SLOT if empty)
    };

    std::array<std::array<node_t, MAX_SEQUENCES>, DEPTH> nodes{};
    std::array<layer_idx_t, DEPTH> count{}; ///< nodes per layer

    static_assert(MAX_SEQUENCES <= std::numeric_limits<layer_idx_t>::max(), "node indices and layer counts overflow");
};

namespace dev {

    /** @brief regroup the flat slot plans into the per-level prefix tree. */
    [[nodiscard]] constexpr plan_tree build_tree(bool held_empty) {
        plan_tree tree{};
        auto const& plans = PLANS[held_empty ? 1 : 0];

        // planNode[planIdx]: the node owning that plan's prefix at the layer being built
        std::array<layer_idx_t, MAX_SEQUENCES> planNode{};

        // layer 0: dedup (slot0, rootHold) in first-appearance order
        for(std::uint32_t planIdx = 0; planIdx < plans.count; ++planIdx) {
            auto const& plan = plans.data[planIdx];

            std::uint32_t nodeIdx = 0;
            while(nodeIdx < tree.count[0]
                && !(tree.nodes[0][nodeIdx].slot == plan.slot(0) && tree.nodes[0][nodeIdx].rootHold == plan.rootHold()))
                ++nodeIdx;
            if(nodeIdx == tree.count[0]) {
                tree.nodes[0][nodeIdx].slot = plan.slot(0);
                tree.nodes[0][nodeIdx].rootHold = plan.rootHold();
                tree.nodes[0][nodeIdx].heldSlot = plan.held_slot(0);
                ++tree.count[0];
            }
            else if(tree.nodes[0][nodeIdx].heldSlot != plan.held_slot(0))
                throw "plan_tree::build_tree: held_slot mismatch at layer 0";
            planNode[planIdx] = static_cast<layer_idx_t>(nodeIdx);
        }

        // layers 1..DEPTH-1: children grouped contiguously under their parent, parents in layer order,
        // children in first-appearance (plan) order -- the same emission order the search stages walk.
        for(std::uint32_t level = 1; level < DEPTH; ++level) {
            std::array<layer_idx_t, MAX_SEQUENCES> nextNode{};

            for(std::uint32_t parentIdx = 0; parentIdx < tree.count[level - 1]; ++parentIdx) {
                tree.nodes[level - 1][parentIdx].childBegin = tree.count[level];

                for(std::uint32_t planIdx = 0; planIdx < plans.count; ++planIdx) {
                    if(planNode[planIdx] != parentIdx)
                        continue;
                    auto const& plan = plans.data[planIdx];

                    std::uint32_t nodeIdx = tree.nodes[level - 1][parentIdx].childBegin;
                    while(nodeIdx < tree.count[level] && tree.nodes[level][nodeIdx].slot != plan.slot(level))
                        ++nodeIdx;
                    if(nodeIdx == tree.count[level]) {
                        tree.nodes[level][nodeIdx].slot = plan.slot(level);
                        tree.nodes[level][nodeIdx].parentIdx = static_cast<layer_idx_t>(parentIdx);
                        tree.nodes[level][nodeIdx].heldSlot = plan.held_slot(level);
                        ++tree.count[level];
                    }
                    else if(tree.nodes[level][nodeIdx].heldSlot != plan.held_slot(level))
                        throw "plan_tree::build_tree: held_slot mismatch";
                    nextNode[planIdx] = static_cast<layer_idx_t>(nodeIdx);
                }
                tree.nodes[level - 1][parentIdx].childEnd = tree.count[level];
            }
            planNode = nextNode;
        }
        return tree;
    }

    TA3_CUDA_CONSTANT std::array<plan_tree, 2> PLAN_TREES{build_tree(false), build_tree(true)};

    /** @brief widest child range of any node at layers [lo, hi], over both hold states (>= 1). */
    [[nodiscard]] constexpr std::uint32_t max_child_span(std::uint32_t lo, std::uint32_t hi) {
        std::uint32_t m = 1;
        for(auto const& tree : PLAN_TREES)
            for(auto level = lo; level <= hi && level < DEPTH; ++level)
                for(std::uint32_t nodeIdx = 0; nodeIdx < tree.count[level]; ++nodeIdx) {
                    auto const width = static_cast<std::uint32_t>(tree.nodes[level][nodeIdx].childEnd)
                        - tree.nodes[level][nodeIdx].childBegin;
                    m = width > m ? width : m;
                }
        return m;
    }

} // namespace dev

// tight static caps over both hold states -- these size the kernel's shared-memory slot arrays.
/** max root nodes (layer 0). */
inline constexpr std::uint32_t MAX_ROOT_GROUPS = [] {
    std::uint32_t m = 0;
    for(auto const& tree : dev::PLAN_TREES)
        m = tree.count[0] > m ? tree.count[0] : m;
    return m;
}();
/** max children of any frontier node feeding a beamed MIDDLE stage (parents at layers 0..DEPTH-3). */
inline constexpr std::uint32_t MAX_CHILD_MID = DEPTH > 2 ? dev::max_child_span(0, DEPTH - 3) : 1;
/** max last-piece lists under one kept frontier node (parents at layer DEPTH-2). */
inline constexpr std::uint32_t MAX_LAST = dev::max_child_span(DEPTH - 2, DEPTH - 2);

/** stage-0 slot space: root slot = groupIdx * MAX_PLACEMENTS + i0. */
inline constexpr std::uint32_t ROOT_SLOTS = MAX_ROOT_GROUPS * MAX_PLACEMENTS;

/** @brief widest beam width across all kept levels -- sizes the frontier buffers. */
[[nodiscard]] constexpr std::uint32_t max_width(beam_widths_t const& ks) {
    std::uint32_t m = 0;
    for(auto const width : ks)
        m = width > m ? width : m;
    return m;
}

/**
 * @brief widest middle-stage slot space for widths @p ks.
 * @details level slot = (rank * MAX_CHILD_MID + child) * MAX_PLACEMENTS + i; each middle level hangs off
 *  ks[level-1] kept nodes. 1 (unused) when DEPTH == 2.
 */
[[nodiscard]] constexpr std::uint32_t mid_slots(beam_widths_t const& ks) {
    std::uint32_t m = 0;
    for(std::uint32_t level = 1; level + 1 < DEPTH; ++level)
        m = ks[level - 1] > m ? ks[level - 1] : m;
    return m == 0 ? 1 : m * MAX_CHILD_MID * MAX_PLACEMENTS;
}

/** @brief width of the LAST kept frontier (the one the leaf stage expands). */
[[nodiscard]] constexpr std::uint32_t last_width(beam_widths_t const& ks) { return ks[DEPTH - 2]; }

/**
 * @brief leaf order-key count for last-frontier width @p k_last: order = (rank * MAX_LAST + t) * MAX_PLACEMENTS + i.
 * @note name is load-bearing: called by name from the GPU kernel (@c search::leaf_orders) -- do not rename.
 */
[[nodiscard]] constexpr std::uint32_t leaf_orders(std::uint32_t k_last) { return k_last * MAX_LAST * MAX_PLACEMENTS; }

/**
 * @brief total order-key span for widths @p ks: leaf orders plus one root-rank band for mid-level fallbacks.
 * @details fallback order = leaf_orders(...) + rootRank. two fallbacks may share an order (distinct
 *  dead-end nodes under one root), but then also share the decoded root move, so argmax stays deterministic.
 * @note name is load-bearing: called by name from the GPU kernel (@c search::order_span) -- do not rename.
 */
[[nodiscard]] constexpr std::uint32_t order_span(beam_widths_t const& ks) {
    return leaf_orders(last_width(ks)) + ks[0];
}

/**
 * @brief monotonic float -> uint32 key (larger float -> larger unsigned).
 * @details 0 never occurs for model outputs, so it doubles as the "illegal / empty slot" sentinel.
 */
[[nodiscard]] constexpr score_t ordered_bits(float value) {
    auto const u = std::bit_cast<std::uint32_t>(value);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
/** @brief inverse of @ref ordered_bits. */
[[nodiscard]] constexpr float ordered_bits_inv(score_t score) {
    return std::bit_cast<float>((score & 0x80000000u) ? (score ^ 0x80000000u) : ~score);
}

/**
 * @brief pack (value, position) so an unsigned max yields "largest value, lowest position on ties".
 * @pre order_count exceeds every pos; a zero key means "nothing".
 */
[[nodiscard]] constexpr std::uint64_t pack_key(score_t ordered_val, order_t pos, order_t order_count) {
    return (static_cast<std::uint64_t>(ordered_val) << 32) | static_cast<std::uint64_t>(order_count - pos);
}
/** @brief inverse of @ref pack_key: recover the position. */
[[nodiscard]] constexpr order_t key_pos(std::uint64_t key, order_t order_count) {
    return order_count - static_cast<order_t>(key & 0xFFFFFFFFull);
}

/** @brief gather the concrete piece per abstract window slot (identical to @ref generate_sequences_into). */
[[nodiscard]] constexpr std::array<sim::PieceType, NUM_SLOTS> gather_slots(
    sim::PieceType current,
    sim::PieceType held,
    std::span<sim::PieceType const> lookahead
) {
    std::array<sim::PieceType, NUM_SLOTS> slot{};
    slot[SEQ_HELD_SLOT] = held;
    slot[win_slot(0)] = current;
    for(std::uint32_t i = 0; i < DEPTH; ++i)
        slot[win_slot(1 + i)] = lookahead[i];
    return slot;
}

/**
 * @brief "best K" selection: repeatedly take the max non-zero score (lowest slot on ties), record, zero it.
 * @pre out_slots.size() == out_scores.size() -- this many are selected, at most.
 * @return number selected (< out_slots.size() when fewer legal candidates exist).
 */
constexpr std::uint32_t select_top(
    std::span<score_t> scores,
    std::span<slot_t> out_slots,
    std::span<score_t> out_scores
) {
    std::uint32_t count = 0;
    for(; count < out_slots.size(); ++count) {
        score_t bestScore = 0;
        slot_t bestSlot = 0;
        for(std::uint32_t i = 0; i < scores.size(); ++i)
            if(scores[i] > bestScore) { // strict: the first (lowest) slot wins ties
                bestScore = scores[i];
                bestSlot = i;
            }
        if(bestScore == 0)
            break;
        out_slots[count] = bestSlot;
        out_scores[count] = bestScore;
        scores[bestSlot] = 0;
    }
    return count;
}

namespace dev {

    /** @brief one ping-pong slice of the kept beam frontier: parallel spans over kept nodes, indexed by rank. */
    struct frontier_t {
        std::span<sim::Board2> board;
        std::span<clear_t> clears; ///< clear histogram accumulated along the path to this node
        std::span<score_t> score; ///< ordered_bits score of the kept board
        std::span<layer_idx_t> node; ///< tree node index at the frontier's layer
        std::span<root_rank_t> rootRank; ///< which selected root (by rank) this kept node descends from
    };

    /**
     * @brief score every legal root placement.
     * @return per-slot score from @ref ordered_bits; 0 where illegal.
     */
    template<class Model>
    [[nodiscard]] constexpr std::array<score_t, ROOT_SLOTS> score_roots(
        plan_tree const& tree,
        std::array<sim::PieceType, NUM_SLOTS> const& slot,
        sim::Board2 const& board,
        Model const& model
    ) {
        std::array<score_t, ROOT_SLOTS> scores{};
        for(std::uint32_t groupIdx = 0; groupIdx < tree.count[0]; ++groupIdx) {
            auto const& node = tree.nodes[0][groupIdx];
            auto const piece0 = slot[node.slot];
            auto const heldIsI = node.heldSlot == HELD_NONE_SLOT ? false : slot[node.heldSlot] == sim::PieceType::I;
            for(std::uint32_t i0 = 0; i0 < n_theoretical_placements(piece0); ++i0) {
                auto const [newBoard, clears, ok] = apply(board, {}, nth_placement(piece0, i0));
                scores[groupIdx * MAX_PLACEMENTS + i0] =
                    ok ? ordered_bits(static_cast<float>(model.evaluate(clears, newBoard, heldIsI))) : 0u;
            }
        }
        return scores;
    }

    /**
     * @brief build the initial kept frontier (layer 0) from the selected roots @p root_slots / @p root_scores.
     * @param kept0 [out] frontier slice to populate.
     */
    constexpr void seed_frontier(
        plan_tree const& tree,
        std::array<sim::PieceType, NUM_SLOTS> const& slot,
        sim::Board2 const& board,
        std::span<slot_t const> root_slots,
        std::span<score_t const> root_scores,
        std::uint32_t root_count,
        frontier_t& kept0
    ) {
        for(std::uint32_t k = 0; k < root_count; ++k) {
            auto const groupIdx = root_slots[k] / MAX_PLACEMENTS;
            auto const s0 = apply(
                board,
                {},
                nth_placement(slot[tree.nodes[0][groupIdx].slot], root_slots[k] % MAX_PLACEMENTS)
            );
            kept0.board[k] = s0.nextBoard;
            kept0.clears[k] = s0.nextClearHist;
            kept0.score[k] = root_scores[k];
            kept0.node[k] = static_cast<layer_idx_t>(groupIdx);
            kept0.rootRank[k] = static_cast<root_rank_t>(k);
        }
    }

    /**
     * @brief score every continuation of the kept frontier at level @p level.
     * @param mid_scores [out] ordered_bits score per (child, placement) slot; 0 where illegal.
     * @param has_child [out] which kept nodes have >= 1 legal continuation.
     */
    template<class Model>
    constexpr void score_mid_level(
        plan_tree const& tree,
        std::array<sim::PieceType, NUM_SLOTS> const& slot,
        std::uint32_t level,
        frontier_t const& frontier,
        std::uint32_t kept_count,
        Model const& model,
        std::span<score_t> mid_scores,
        std::span<bool> has_child
    ) {
        for(auto& v : mid_scores)
            v = 0;
        for(auto& h : has_child)
            h = false;

        for(std::uint32_t k = 0; k < kept_count; ++k) {
            auto const& pnode = tree.nodes[level - 1][frontier.node[k]];
            for(std::uint32_t childIdx = pnode.childBegin; childIdx < pnode.childEnd; ++childIdx) {
                auto const& cnode = tree.nodes[level][childIdx];
                auto const piece = slot[cnode.slot];
                auto const heldIsI = cnode.heldSlot == HELD_NONE_SLOT
                                     ? false
                                     : slot[cnode.heldSlot] == sim::PieceType::I;
                auto const base = (k * MAX_CHILD_MID + (childIdx - pnode.childBegin)) * MAX_PLACEMENTS;
                for(std::uint32_t i = 0; i < n_theoretical_placements(piece); ++i) {
                    auto const s = apply(frontier.board[k], frontier.clears[k], nth_placement(piece, i));
                    if(!s.legal)
                        continue;
                    mid_scores[base + i] = ordered_bits(
                        static_cast<float>(model.evaluate(s.nextClearHist, s.nextBoard, heldIsI))
                    );
                    has_child[k] = true;
                }
            }
        }
    }

    /**
     * @brief build the next kept frontier from the level-@p level candidates selected by @ref select_top.
     * @pre selected_slots.size() == selected_scores.size().
     * @param next [out] frontier slice to populate; only the first selected_slots.size() entries are written.
     */
    constexpr void advance_frontier(
        plan_tree const& tree,
        std::array<sim::PieceType, NUM_SLOTS> const& slot,
        std::uint32_t level,
        frontier_t const& frontier,
        std::span<slot_t const> selected_slots,
        std::span<score_t const> selected_scores,
        frontier_t& next
    ) {
        for(std::uint32_t j = 0; j < selected_slots.size(); ++j) {
            auto const k = selected_slots[j] / (MAX_CHILD_MID * MAX_PLACEMENTS);
            auto const& pnode = tree.nodes[level - 1][frontier.node[k]];
            auto const childIdx = pnode.childBegin + (selected_slots[j] / MAX_PLACEMENTS) % MAX_CHILD_MID;
            auto const step = apply(
                frontier.board[k],
                frontier.clears[k],
                nth_placement(slot[tree.nodes[level][childIdx].slot], selected_slots[j] % MAX_PLACEMENTS)
            );
            next.board[j] = step.nextBoard;
            next.clears[j] = step.nextClearHist;
            next.score[j] = selected_scores[j];
            next.node[j] = static_cast<layer_idx_t>(childIdx);
            next.rootRank[j] = frontier.rootRank[k];
        }
    }

    /** @brief expand the kept frontier to leaves, folding each (score, order) into @p consider. */
    template<class Model, class Consider>
    constexpr void expand_leaves(
        plan_tree const& tree,
        std::array<sim::PieceType, NUM_SLOTS> const& slot,
        frontier_t const& frontier,
        std::uint32_t kept_count,
        Model const& model,
        Consider const& consider
    ) {
        for(std::uint32_t j = 0; j < kept_count; ++j) {
            auto const& lnode = tree.nodes[DEPTH - 2][frontier.node[j]];
            for(std::uint32_t childIdx = lnode.childBegin; childIdx < lnode.childEnd; ++childIdx) {
                auto const& cnode = tree.nodes[DEPTH - 1][childIdx];
                auto const piece = slot[cnode.slot];
                auto const heldIsI = cnode.heldSlot == HELD_NONE_SLOT
                                     ? false
                                     : slot[cnode.heldSlot] == sim::PieceType::I;
                auto const base = (j * MAX_LAST + (childIdx - lnode.childBegin)) * MAX_PLACEMENTS;
                auto expanded = false;
                for(std::uint32_t i = 0; i < n_theoretical_placements(piece); ++i) {
                    auto const [nextBoard, nextClearHist, legal] = apply(
                        frontier.board[j],
                        frontier.clears[j],
                        nth_placement(piece, i)
                    );
                    if(!legal)
                        continue;
                    expanded = true;
                    consider(
                        ordered_bits(static_cast<float>(model.evaluate(nextClearHist, nextBoard, heldIsI))),
                        base + i
                    );
                }
                if(!expanded) // the last piece never fit: the frontier board is the leaf (score already known)
                    consider(frontier.score[j], base);
            }
        }
    }

    /**
     * @brief decode the winning (score, order) key @p best_key into a move.
     * @pre best_key encodes >= 1 considered candidate (guaranteed once root_count > 0).
     * @return best root move + its looked-ahead score.
     */
    [[nodiscard]] constexpr search_result decode_move(
        plan_tree const& tree,
        std::array<sim::PieceType, NUM_SLOTS> const& slot,
        std::span<slot_t const> root_slots,
        frontier_t const& frontier,
        std::uint64_t best_key,
        order_t order_count,
        order_t leaf_order_count
    ) {
        auto const order = key_pos(best_key, order_count);
        auto const rootRank = order >= leaf_order_count
                              ? order - leaf_order_count
                              : frontier.rootRank[order / (MAX_LAST * MAX_PLACEMENTS)];

        auto const& rootNode = tree.nodes[0][root_slots[rootRank] / MAX_PLACEMENTS];
        auto const pl0 = nth_placement(slot[rootNode.slot], root_slots[rootRank] % MAX_PLACEMENTS);
        return {
            sim::drop_place_t{pl0.orientation, pl0.x, rootNode.rootHold},
            ordered_bits_inv(static_cast<score_t>(best_key >> 32))
        };
    }

} // namespace dev

/**
 * @brief beamed depth-DEPTH search for one move; CPU parity reference of the GPU kernel.
 * @details frontier ping-pongs through DEPTH-2 middle stages, carrying (board, clears, score, tree node,
 *  root rank) per kept node. leaves score the raw board -- no canonical() mirror fold, since every
 *  model_t feature is mirror-invariant.
 * @tparam Ks per-level beam widths (default @ref BEAM_WIDTHS).
 * @param model model(clear_t, sim::Board2 const&) -> value_t.
 * @return best root move + looked-ahead score, or a result with @c none() true if game over / no legal move.
 */
template<beam_widths_t Ks = BEAM_WIDTHS, class Model>
[[nodiscard]] constexpr search_result search_move_beam(sim::TetrisEngine const& game, Model const& model) {
    static_assert(
        [] {
            for(auto const width : Ks)
                if(width == 0)
                    return false;
            return true;
        }(),
        "every beam width must be >= 1"
    );
    constexpr auto k0 = Ks[0];
    constexpr auto kMax = max_width(Ks);
    constexpr auto mid = mid_slots(Ks);
    constexpr auto leafOrderCount = leaf_orders(last_width(Ks));
    constexpr auto orderCount = order_span(Ks);

    if(game.gameOver())
        return {};

    auto const slot = gather_slots(game.currentPiece(), game.heldPiece(), game.lookahead());
    auto const& tree = dev::PLAN_TREES[game.heldPiece() == sim::TetrisEngine::NO_PIECE ? 1 : 0];
    auto board = game.board();

    // stage 0: score and select the roots
    auto rootScores = dev::score_roots(tree, slot, board, model);
    std::array<slot_t, k0> topRootSlots{};
    std::array<score_t, k0> topRootScores{};
    auto const rootCount = select_top(rootScores, topRootSlots, topRootScores);
    if(rootCount == 0)
        return {}; // no legal root placement

    std::uint64_t bestKey = 0;
    auto const consider = [&](score_t ordered_score, order_t order) {
        auto const key = pack_key(ordered_score, order, orderCount);
        bestKey = key > bestKey ? key : bestKey;
    };

    // backing storage for the ping-ponged kept frontier (see dev::frontier_t)
    std::array<std::array<sim::Board2, kMax>, 2> boardBuf{};
    std::array<std::array<clear_t, kMax>, 2> clearsBuf{};
    std::array<std::array<score_t, kMax>, 2> scoreBuf{};
    std::array<std::array<layer_idx_t, kMax>, 2> nodeBuf{};
    std::array<std::array<root_rank_t, kMax>, 2> rootRankBuf{};
    std::array<dev::frontier_t, 2> kept{
        dev::frontier_t{boardBuf[0], clearsBuf[0], scoreBuf[0], nodeBuf[0], rootRankBuf[0]},
        dev::frontier_t{boardBuf[1], clearsBuf[1], scoreBuf[1], nodeBuf[1], rootRankBuf[1]}
    };
    std::uint32_t cur = 0, keptCount = rootCount;

    dev::seed_frontier(tree, slot, board, topRootSlots, topRootScores, rootCount, kept[0]);

    // middle stages: score the continuations under the frontier, keep the best KS[level]
    std::array<score_t, mid> midScores{};
    std::array<bool, kMax> hasChild{};
    std::array<slot_t, kMax> topMidSlots{};
    std::array<score_t, kMax> topMidScores{};
    for(std::uint32_t level = 1; level + 1 < DEPTH; ++level) {
        dev::score_mid_level(tree, slot, level, kept[cur], keptCount, model, midScores, hasChild);

        auto const nextCount = select_top(
            midScores,
            std::span{topMidSlots}.first(Ks[level]),
            std::span{topMidScores}.first(Ks[level])
        );

        for(std::uint32_t k = 0; k < keptCount; ++k) // kept nodes with no legal continuation: depth-level leaves
            if(!hasChild[k])
                consider(kept[cur].score[k], leafOrderCount + kept[cur].rootRank[k]);

        dev::advance_frontier(
            tree,
            slot,
            level,
            kept[cur],
            std::span{topMidSlots}.first(nextCount),
            std::span{topMidScores}.first(nextCount),
            kept[cur ^ 1]
        );
        cur ^= 1;
        keptCount = nextCount;
    }

    // leaf stage: expand the kept frontier; argmax (score, order)
    dev::expand_leaves(tree, slot, kept[cur], keptCount, model, consider);

    return dev::decode_move(tree, slot, topRootSlots, kept[cur], bestKey, orderCount, leafOrderCount);
}

} // namespace ta3::ai::search
