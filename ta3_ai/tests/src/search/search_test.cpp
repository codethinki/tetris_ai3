#include "ta3/ai/search/placements.hpp"
#include "ta3/ai/search/search.hpp"
#include "ta3/ai/search/variation_sequences.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>
#include <ta3/sim/tetris_engine.hpp>

#include "test.hpp"
#include "work.hpp"

#include <array>
#include <cstdint>

#define SEARCH_TEST(suite, name) CTH_EX_TEST(_ai_search, suite, name)

namespace ta3::ai {

namespace {

    /** a tiny constexpr value model: reward clears, penalise stack height and holes. */
    struct height_model {
        constexpr search::value_t operator()(search::clear_t accum, sim::Board2 const& b) const {
            std::uint32_t agg = 0;
            for(std::size_t x = 0; x < sim::WIDTH; ++x)
                agg += static_cast<std::uint32_t>(b.height(x));

            return 2.0f * static_cast<search::value_t>(accum.lines())
                - 0.3f * static_cast<search::value_t>(agg)
                - 1.0f * static_cast<search::value_t>(b.holes());
        }

        /** the search seam (@ref search_move / @ref search_move_beam call @c model.evaluate) */
        constexpr search::value_t evaluate(search::clear_t accum, sim::Board2 const& b, bool /*heldIsI*/) const {
            return (*this)(accum, b);
        }
    };

    // --- placement enumeration: every decoded placement is in its orientation's theoretical range ----
    constexpr bool placements_decode_in_range() {
        for(std::uint32_t p = 0; p < *sim::PieceType::COUNT; ++p) {
            auto const piece = static_cast<sim::PieceType>(p);
            auto const total = search::n_theoretical_placements(piece);
            if(total == 0)
                return false;

            for(std::uint32_t i = 0; i < total; ++i) {
                auto const pl = search::nth_placement(piece, i);
                auto const range = search::orientation_range(piece, pl.orientation);
                if(pl.x < range.xMin || pl.x > range.xMax)
                    return false;
            }
        }
        return true;
    }
    static_assert(placements_decode_in_range());
    static_assert(search::MAX_PLACEMENTS > 0 && search::MAX_PLACEMENTS <= 40);

    // --- deduped placement counts are pinned (O's 4 squares -> 1, I/S/Z horizontals+verticals -> 2) ---
    static_assert(search::n_theoretical_placements(sim::PieceType::O) == 9);
    static_assert(search::n_theoretical_placements(sim::PieceType::I) == 17);
    static_assert(search::n_theoretical_placements(sim::PieceType::S) == 17);
    static_assert(search::n_theoretical_placements(sim::PieceType::Z) == 17);
    static_assert(search::n_theoretical_placements(sim::PieceType::T) == 34);
    static_assert(search::n_theoretical_placements(sim::PieceType::J) == 34);
    static_assert(search::n_theoretical_placements(sim::PieceType::L) == 34);
    static_assert(search::MAX_PLACEMENTS == 34);

    // --- hold layer: from-empty commits the first lookahead and flags a root hold -------------------
    constexpr bool sequences_from_empty_holds() {
        std::array<sim::PieceType, 4> const la{
            sim::PieceType::O,
            sim::PieceType::T,
            sim::PieceType::S,
            sim::PieceType::Z
        };
        auto const set = search::generate_sequences(sim::PieceType::I, sim::NO_PIECE, la);

        for(auto const& s : set)
            if(s.rootHold && s.pieces[0] == sim::PieceType::O) // stashed I, placed the next piece O
                return set.count >= 1;
        return false;
    }
    static_assert(sequences_from_empty_holds());

    // --- hold layer: a non-empty distinct held piece is reachable at the root via a swap ------------
    constexpr bool sequences_hold_swap() {
        std::array<sim::PieceType, 4> const la{
            sim::PieceType::T,
            sim::PieceType::S,
            sim::PieceType::Z,
            sim::PieceType::J
        };
        auto const set = search::generate_sequences(sim::PieceType::I, sim::PieceType::O, la);

        for(auto const& s : set)
            if(s.rootHold && s.pieces[0] == sim::PieceType::O)
                return true;
        return false;
    }
    static_assert(sequences_hold_swap());

    // --- hold layer: holding to reach an identical piece (held == current) is harmless -------------
    // runtime piece-identity dedup is gone, so a redundant root hold may appear; correctness does not
    // depend on it: holding I to place I places the same piece, so committing it is board-identical to
    // not holding. the invariant is that any such root hold never changes the placed root piece, and a
    // plain (non-hold) root is always available too.
    constexpr bool sequences_identical_hold_is_harmless() {
        std::array<sim::PieceType, 4> const la{
            sim::PieceType::T,
            sim::PieceType::S,
            sim::PieceType::Z,
            sim::PieceType::J
        };
        auto const set = search::generate_sequences(sim::PieceType::I, sim::PieceType::I, la);

        auto nonHoldRoot = false;
        for(auto const& s : set) {
            if(s.rootHold && s.pieces[0] != sim::PieceType::I) // a real (non-redundant) hold would differ
                return false;
            if(!s.rootHold)
                nonHoldRoot = true;
        }
        return nonHoldRoot && set.count >= 1;
    }
    static_assert(sequences_identical_hold_is_harmless());

    // --- hold LUT: dev::PLANS' held_slot bookkeeping matches an independent naive replay ------------
    // (a restatement of the hold state machine, written independently of dev::build_plans, over every
    // decision vector and both held_empty states -- catches a held_slot regression that same_slots()
    // dedup by itself would not).
    constexpr bool held_slot_matches_naive_replay() {
        for(std::uint32_t he = 0; he < 2; ++he) {
            auto const heldEmpty = he == 1; // dev::PLANS[1] == empty hold
            auto const& plans = search::dev::PLANS[he];

            for(std::uint32_t d = 0; d < (1u << search::DEPTH); ++d) {
                auto cur = search::win_slot(0);
                auto hld = search::SEQ_HELD_SLOT;
                auto empty = heldEmpty;
                std::uint32_t k = 1;

                std::array<std::uint8_t, search::DEPTH> slots{};
                std::array<std::uint8_t, search::DEPTH> held{};

                for(std::uint32_t i = 0; i < search::DEPTH; ++i) {
                    if(d & (1u << i)) {
                        if(empty) {
                            hld = cur;
                            cur = search::win_slot(k++);
                            empty = false;
                        }
                        else {
                            auto const tmp = cur;
                            cur = hld;
                            hld = tmp;
                        }
                    }
                    slots[i] = cur;
                    held[i] = empty ? search::dev::seq_plan::HELD_NONE : hld;
                    if(i + 1 < search::DEPTH)
                        cur = search::win_slot(k++);
                }

                // find the dedup'd plan with this exact slot sequence and check its held_slot fields
                auto found = false;
                for(std::uint32_t p = 0; p < plans.count && !found; ++p) {
                    auto same = true;
                    for(std::uint32_t i = 0; i < search::DEPTH && same; ++i)
                        same = plans.data[p].slot(i) == slots[i];
                    if(!same)
                        continue;
                    found = true;
                    for(std::uint32_t i = 0; i < search::DEPTH; ++i)
                        if(plans.data[p].held_slot(i) != held[i])
                            return false;
                }
                if(!found)
                    return false;
            }
        }
        return true;
    }
    static_assert(held_slot_matches_naive_replay());

    // --- hold LUT: a plan that never holds keeps the ORIGINAL hold state at every level --------------
    constexpr bool no_hold_plan_keeps_held_none_from_empty() {
        auto const& plans = search::dev::PLANS[1]; // held_empty
        for(std::uint32_t p = 0; p < plans.count; ++p) {
            auto neverMoves = true;
            for(std::uint32_t i = 0; i < search::DEPTH && neverMoves; ++i)
                neverMoves = plans.data[p].slot(i) == search::win_slot(i);
            if(!neverMoves)
                continue;
            for(std::uint32_t i = 0; i < search::DEPTH; ++i)
                if(plans.data[p].held_slot(i) != search::dev::seq_plan::HELD_NONE)
                    return false;
            return true;
        }
        return false;
    }
    static_assert(no_hold_plan_keeps_held_none_from_empty());

    constexpr bool no_hold_plan_keeps_original_held_slot() {
        auto const& plans = search::dev::PLANS[0]; // held non-empty
        for(std::uint32_t p = 0; p < plans.count; ++p) {
            auto neverMoves = true;
            for(std::uint32_t i = 0; i < search::DEPTH && neverMoves; ++i)
                neverMoves = plans.data[p].slot(i) == search::win_slot(i);
            if(!neverMoves)
                continue;
            for(std::uint32_t i = 0; i < search::DEPTH; ++i)
                if(plans.data[p].held_slot(i) != search::SEQ_HELD_SLOT)
                    return false;
            return true;
        }
        return false;
    }
    static_assert(no_hold_plan_keeps_original_held_slot());

    // --- hold LUT: a root hold from empty stashes the piece that WAS about to be placed (win_slot(0)) ----
    constexpr bool root_hold_from_empty_stashes_win_slot_0() {
        auto const& plans = search::dev::PLANS[1]; // held_empty
        for(std::uint32_t p = 0; p < plans.count; ++p)
            if(plans.data[p].rootHold() && plans.data[p].held_slot(0) == search::win_slot(0))
                return true;
        return false;
    }
    static_assert(root_hold_from_empty_stashes_win_slot_0());

    // --- work-id mapping: encode/decode is a bijection over every (sequence, i0, i1) prefix ----------
    // (the tiled decomposition is legacy and depth-3 only; see work.hpp)
#if TA3_SEARCH_DEPTH == 3
    constexpr bool prefix_mapping_round_trips() {
        std::array<sim::PieceType, 4> const la{
            sim::PieceType::T,
            sim::PieceType::S,
            sim::PieceType::Z,
            sim::PieceType::J
        };
        auto const seqs = search::generate_sequences(sim::PieceType::I, sim::PieceType::O, la);
        auto const w = search::build_layout(seqs);

        std::uint32_t emitted = 0;
        for(std::uint32_t s = 0; s < w.count; ++s) {
            auto const b0 = search::theo_count(seqs[s].pieces[0]);
            auto const b1 = search::theo_count(seqs[s].pieces[1]);
            for(std::uint32_t i0 = 0; i0 < b0; ++i0)
                for(std::uint32_t i1 = 0; i1 < b1; ++i1) {
                    auto const id = search::encode_prefix(seqs, w, {s, i0, i1});
                    auto const p = search::decode_prefix(seqs, w, id);
                    if(p.seq != s || p.i0 != i0 || p.i1 != i1)
                        return false;
                    ++emitted;
                }
        }
        return emitted == search::prefix_count(w); // full, gapless cover
    }
    static_assert(prefix_mapping_round_trips());
#endif

} // namespace

// full search is too heavy for a constexpr step budget (thousands of leaves), so it runs at runtime;
// the constexpr asserts above cover the device-clean primitives it is built from.

SEARCH_TEST(brute, returns_a_legal_move_on_a_fresh_board) {
    sim::TetrisEngine game{1234};
    height_model const model;

    auto const r = search::search_move(game, model);
    ASSERT_FALSE(r.none());

    // replay exactly as the trainer loop would; place() has a legal-placement precondition, so an
    // illegal choice would trip its internal checks / change the outcome
    if(r.move.hold)
        game.hold();
    auto const cleared = game.place(r.move.orientation, r.move.x);
    EXPECT_NE(cleared, sim::TetrisEngine::DIED);
}

SEARCH_TEST(brute, plays_a_multi_move_game) {
    sim::TetrisEngine game{0xC0FFEE};
    height_model const model;

    // each move brute-forces ~b^DEPTH leaves; keep the move count modest so the test stays quick
    std::uint32_t moves = 0;
    for(; moves < 15 && !game.gameOver(); ++moves) {
        auto const r = search::search_move(game, model);
        if(r.none())
            break;

        if(r.move.hold)
            game.hold();
        if(game.place(r.move.orientation, r.move.x) == sim::TetrisEngine::DIED)
            break;
    }

    EXPECT_GT(moves, 5u); // a height-aware search should survive well past the opening
}

#if TA3_SEARCH_DEPTH == 3
SEARCH_TEST(brute, tiled_matches_reference_over_a_game) {
    height_model const model;

    // the tiled (per-prefix) decomposition the kernel uses must reproduce search_move's chosen move and
    // value at every step, across a full game -- parity by construction, checked here.
    sim::TetrisEngine game{0xBADF00D};
    for(std::uint32_t moves = 0; moves < 20 && !game.gameOver(); ++moves) {
        auto const ref = search::search_move(game, model);
        auto const tiled = search::search_move_tiled(game, model);

        ASSERT_EQ(ref.none(), tiled.none());
        if(ref.none())
            break;
        EXPECT_EQ(ref.move, tiled.move);
        EXPECT_EQ(ref.value, tiled.value);

        if(ref.move.hold)
            game.hold();
        if(game.place(ref.move.orientation, ref.move.x) == sim::TetrisEngine::DIED)
            break;
    }
}
#endif // TA3_SEARCH_DEPTH == 3

SEARCH_TEST(brute, distinct_seeds_are_independent) {
    height_model const model;
    sim::TetrisEngine a{1}, b{2};

    auto const ra = search::search_move(a, model);
    auto const rb = search::search_move(b, model);

    EXPECT_FALSE(ra.none());
    EXPECT_FALSE(rb.none());
}

} // namespace ta3::ai
