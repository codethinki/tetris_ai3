#pragma once
#include <ta3/sim/utility/cuda_constant.hpp>

#include <ta3/sim/pieces/piece_defs.hpp>
#include <ta3/sim/utility/tetris_defs.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ta3::ai::search {

/**
 * the beam width kept after scoring each level: @c [0] = root placements (stage 0), @c [l] = level-@c l
 * continuations. the widths ARE the search shape -- @ref DEPTH derives from the count, so adding a width
 * deepens the search by one level. override via @c TA3_BEAM_WIDTHS.
 * @attention TA3_CUDA_CONSTANT, not plain @c inline @c constexpr: the kernel indexes this with a RUNTIME
 *  level, and under @c --expt-relaxed-constexpr a host constexpr array is usable in device code only where
 *  the access constant-folds (see @c CLEAR_SCORE / @c PLACEMENT_LUT for the same rule).
 */
TA3_CUDA_CONSTANT auto BEAM_WIDTHS = std::to_array<std::uint32_t>({16, 32, 32});

/** searched lookahead depth */
inline constexpr std::uint32_t DEPTH = static_cast<std::uint32_t>(BEAM_WIDTHS.size()) + 1;

/** per-move beam widths, one entry per kept level (levels @c [0, DEPTH-1)). */
using beam_widths_t = std::array<std::uint32_t, DEPTH - 1>;

static_assert(DEPTH >= 2, "the beamed search needs at least a root stage and a leaf stage");
// hold-from-empty can consume up to window slot DEPTH == lookahead[DEPTH-1], and the engine exposes
// PIECE_QUEUE_SIZE-1 lookahead pieces -- the binding depth cap.
static_assert(DEPTH + 1 <= sim::PIECE_QUEUE_SIZE, "DEPTH exceeds the engine's lookahead");
static_assert(
    [] {
        for(auto const k : BEAM_WIDTHS)
            if(k == 0)
                return false;
        return true;
    }(),
    "every beam width must be >= 1"
);

/** upper bound on distinct sequences per move (@c 2^DEPTH decision vectors before dedup). */
inline constexpr std::uint32_t MAX_SEQUENCES = 1u << DEPTH;

/**
 * gather slots the placed pieces are drawn from: @c 0 = held, @c 1 = current (window slot 0),
 * @c 1+i = @c lookahead[i-1] (window slot @c i). @ref dev::build_plans references at most slot @c DEPTH+1.
 */
inline constexpr std::uint32_t NUM_SLOTS = DEPTH + 2;
inline constexpr std::uint8_t SEQ_HELD_SLOT = 0;

} // namespace ta3::ai::search


namespace ta3::ai::search {

struct move_seq {
    /** placed piece sequence */
    std::array<sim::PieceType, DEPTH> pieces{};
    /** true if root held before place */
    bool rootHold = false;
    /** hold-contains-I flag in effect when the board at each level is evaluated */
    std::array<bool, DEPTH> heldIsI{};
};

/** the small fixed-capacity set of distinct sequences for one move. */
struct seq_set {
    std::array<move_seq, MAX_SEQUENCES> data{};
    std::uint32_t count = 0;

    [[nodiscard]] constexpr move_seq const& operator[](std::uint32_t i) const { return data[i]; }
    [[nodiscard]] constexpr move_seq const* begin() const { return data.data(); }
    [[nodiscard]] constexpr move_seq const* end() const { return data.data() + count; }
};

/** the window slot for lookahead index @p i (@c 0 = current, @c 1+i = @c lookahead[i]). */
[[nodiscard]] constexpr std::uint8_t win_slot(std::uint32_t i);

/**
 * @brief enumerate the distinct hold-resolved piece sequences for one move.
 * @param current the piece to place now (window slot 0)
 * @param held    the held piece, or @ref sim::NO_PIECE if the slot is empty
 * @param lookahead the upcoming pieces after @p current, soonest first (@c TetrisEngine::lookahead)
 * @details pure dispatch: select the compile-time slot plan for this hold state, then gather the concrete
 *  pieces into its slots. no per-move hold simulation and no branches; @c count is fixed per hold state.
 *  reads @c lookahead[0..DEPTH-1] (a @ref sim::PIECE_QUEUE_SIZE of 5 has margin).
 */
constexpr void generate_sequences_into(
    seq_set& set,
    sim::PieceType current,
    sim::PieceType held,
    std::span<sim::PieceType const> lookahead
);

/**
 * return-by-value convenience wrapper. on the GPU prefer @ref generate_sequences_into to write straight into
 * the block's shared @c seq_set -- returning the ~132-byte aggregate here lands it on the (tiny) device stack.
 */
[[nodiscard]] constexpr seq_set generate_sequences(
    sim::PieceType current,
    sim::PieceType held,
    std::span<sim::PieceType const> lookahead
);

} // namespace ta3::ai::search


namespace ta3::ai::search {

[[nodiscard]] constexpr std::uint8_t win_slot(std::uint32_t i) { return static_cast<std::uint8_t>(1 + i); }

} // namespace ta3::ai::search



namespace ta3::ai::search {

namespace dev {

    /**
     * @brief encodes which @b slot is placed at each search stage, the level-0 hold flag, and the slot
     *  sitting in the hold slot after each level's hold decision.
     * @details packed into two @c uint32_t words. @c _bits layout: @c DEPTH slot fields of @ref SLOT_BITS
     *  bits (slots are @c 0..DEPTH+1, so 3 bits each), then the root-hold flag at @ref ROOT_HOLD_BIT.
     *  @c _held layout: @c DEPTH held-slot fields of @ref SLOT_BITS bits, one per level, using @ref
     *  HELD_NONE for "hold empty".
     */
    class seq_plan {
    public:
        static constexpr std::uint32_t SLOT_BITS = 3;
        static constexpr std::uint32_t SLOT_MASK = (1u << SLOT_BITS) - 1;
        static constexpr std::uint32_t ROOT_HOLD_BIT = DEPTH * SLOT_BITS;
        static constexpr std::uint32_t SLOTS_MASK = (1u << ROOT_HOLD_BIT) - 1;
        /** sentinel @ref held_slot value: the hold slot is empty. */
        static constexpr std::uint8_t HELD_NONE = static_cast<std::uint8_t>(SLOT_MASK);

        static_assert(NUM_SLOTS < (1u << SLOT_BITS), "slot ids must leave SLOT_MASK free as the HELD_NONE sentinel");
        static_assert(ROOT_HOLD_BIT < 32, "the packed plan must fit a uint32_t");
        static_assert(DEPTH * SLOT_BITS <= 32, "the packed held-slot fields must fit a uint32_t");

        /** the slot placed at level @p i (@c i < DEPTH). */
        [[nodiscard]] constexpr std::uint8_t slot(std::uint32_t i) const {
            return static_cast<std::uint8_t>((_bits >> (i * SLOT_BITS)) & SLOT_MASK);
        }
        /** whether level 0 committed a hold. */
        [[nodiscard]] constexpr bool rootHold() const { return (_bits >> ROOT_HOLD_BIT) & 1u; }
        /** the raw packed representation. */
        [[nodiscard]] constexpr std::uint32_t bits() const { return _bits; }
        /** the slot sitting in the hold slot after level @p i's hold decision (@ref HELD_NONE if empty). */
        [[nodiscard]] constexpr std::uint8_t held_slot(std::uint32_t i) const {
            return static_cast<std::uint8_t>((_held >> (i * SLOT_BITS)) & SLOT_MASK);
        }

        constexpr void set_slot(std::uint32_t i, std::uint8_t s) {
            _bits &= ~(SLOT_MASK << (i * SLOT_BITS));
            _bits |= (static_cast<std::uint32_t>(s) & SLOT_MASK) << (i * SLOT_BITS);
        }
        constexpr void set_root_hold(bool v) {
            _bits = (_bits & ~(1u << ROOT_HOLD_BIT)) | (static_cast<std::uint32_t>(v) << ROOT_HOLD_BIT);
        }
        constexpr void set_held_slot(std::uint32_t i, std::uint8_t s) {
            _held &= ~(SLOT_MASK << (i * SLOT_BITS));
            _held |= (static_cast<std::uint32_t>(s) & SLOT_MASK) << (i * SLOT_BITS);
        }

    private:
        std::uint32_t _bits = 0;
        std::uint32_t _held = 0;
    };

} // namespace dev

/** sentinel window-slot id meaning "the hold slot is empty" at some level (mirrors @c dev::seq_plan::HELD_NONE). */
inline constexpr std::uint8_t HELD_NONE_SLOT = dev::seq_plan::HELD_NONE;

namespace dev {

    /** the fixed-capacity set of distinct slot plans for one hold state. */
    struct plan_set {
        std::array<seq_plan, MAX_SEQUENCES> data{};
        std::uint32_t count = 0;
    };

    [[nodiscard]] constexpr bool same_slots(seq_plan const& a, seq_plan const& b) {
        return (a.bits() & seq_plan::SLOTS_MASK) == (b.bits() & seq_plan::SLOTS_MASK);
    }

    /**
     * @brief insert @p plan, collapsing identical slot plans.
     * @param[in, out] set to insert to
     * @param plan to insert
     * @details the placed-slot path uniquely determines every hold decision, two
     *  plans agreeing on slot bits MUST agree on @c held_slot too
     */
    constexpr void insert(plan_set& set, seq_plan const& plan) {
        /* for(std::uint32_t i = 0; i < set.count; ++i)
             if(same_slots(set.data[i], plan)) {
                 for(std::uint32_t lvl = 0; lvl < DEPTH; ++lvl)
                     if(set.data[i].held_slot(lvl) != plan.held_slot(lvl))
                         throw "seq_plan::insert: held_slot mismatch on identical slot plans";
                 return;
             }*/
        set.data[set.count++] = plan;
    }


    /**
     * simulate every @c 2^DEPTH hold/no-hold decision vector over abstract window slots.
     * @param held_empty true if hold cell is empty
     * @details mirrors @ref sim::TetrisEngine: a plain hold swaps current<->held; a hold from an empty
     *  slot stashes current and pulls the next window slot (consuming one extra). references at most window
     *  slot @c DEPTH, so @ref NUM_SLOTS has margin.
     */
    [[nodiscard]] constexpr plan_set build_plans(bool held_empty) {
        plan_set set{};

        for(std::uint32_t d = 0; d < (1u << DEPTH); ++d) {
            auto cur = win_slot(0);
            auto hld = SEQ_HELD_SLOT;
            auto empty = held_empty;
            std::uint32_t k = 1; // next unread window slot (slot 0 is `cur`)

            seq_plan plan{};
            for(std::uint32_t i = 0; i < DEPTH; ++i) {
                auto didHold = false;
                if(d & (1u << i)) {
                    if(empty) {
                        hld = cur; // stash current
                        cur = win_slot(k++); // hold-from-empty pulls the next slot
                        empty = false;
                    }
                    else {
                        auto const tmp = cur; // plain swap, no consumption
                        cur = hld;
                        hld = tmp;
                    }
                    didHold = true;
                }

                plan.set_slot(i, cur);
                if(i == 0)
                    plan.set_root_hold(didHold);
                plan.set_held_slot(i, empty ? seq_plan::HELD_NONE : hld);

                if(i + 1 < DEPTH)
                    cur = win_slot(k++); // place consumes cur; pull the next for the following level
            }

            insert(set, plan);
        }

        return set;
    }


    /** the entire hold LUT: two slot-plan sets, indexed by @c heldEmpty (@c PLANS[1] == empty hold). */
    TA3_CUDA_CONSTANT std::array<plan_set, 2> PLANS = {dev::build_plans(false), dev::build_plans(true)};

} // namespace dev

constexpr void generate_sequences_into(
    seq_set& set,
    sim::PieceType current,
    sim::PieceType held,
    std::span<sim::PieceType const> lookahead
) {
    auto const& plans = dev::PLANS[held == sim::NO_PIECE ? 1 : 0];

    std::array<sim::PieceType, NUM_SLOTS> sequence{};
    sequence[SEQ_HELD_SLOT] = held;
    sequence[win_slot(0)] = current;
    for(std::uint32_t i = 0; i < DEPTH; ++i)
        sequence[win_slot(1 + i)] = lookahead[i];

    set.count = plans.count;
    for(std::uint32_t j = 0; j < plans.count; ++j) {
        for(std::uint32_t i = 0; i < DEPTH; ++i) {
            set.data[j].pieces[i] = sequence[plans.data[j].slot(i)];
            set.data[j].heldIsI[i] = plans.data[j].held_slot(i) != dev::seq_plan::HELD_NONE
                && sequence[plans.data[j].held_slot(i)] == sim::PieceType::I;
        }
        set.data[j].rootHold = plans.data[j].rootHold();
    }
}

[[nodiscard]] constexpr seq_set generate_sequences(
    sim::PieceType current,
    sim::PieceType held,
    std::span<sim::PieceType const> lookahead
) {
    seq_set set{};
    generate_sequences_into(set, current, held, lookahead);
    return set;
}

} // namespace ta3::ai::search
