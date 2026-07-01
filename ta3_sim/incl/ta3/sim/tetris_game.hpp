#pragma once
#include "ta3/sim/board2.hpp"
#include "ta3/sim/tetris_defs.hpp"
#include "ta3/sim/tetris_engine.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ta3::sim {

/**
 * @brief the playable, move-level layer over @ref TetrisEngine: one piece falls under an
 *  orientation and a 4x4-window offset, driven one @ref Instruction at a time
 * @note the engine owns the board, the 7-bag queue and hold; this adds only the in-flight piece
 *  and resolves every placement to a hard drop, so a piece always locks at its straight-down
 *  landing for the current column -- there are no tucks or soft-drop spins
 */
class TetrisGame {
public:
    /** @brief starts a game whose entire piece sequence is determined by @ref seed */
    constexpr explicit TetrisGame(uint64_t seed);

    /** @brief restarts the game with a fresh piece sequence from @ref seed */
    constexpr void reset(uint64_t seed);

    /**
     * @brief applies one instruction to the in-flight piece
     * @note @c Instruction::NONE, and any move or rotation the board rejects, leave the state
     *  unchanged
     * @attention @c Instruction::PLACE and @c Instruction::HOLD can top the game out; check
     *  @ref gameOver afterwards
     * @return the rows cleared this step, non-zero only for @c Instruction::PLACE
     */
    constexpr size_t step(Instruction instruction);

    /**
     * @brief shifts the in-flight piece one cell in @ref move's direction
     * @return whether the piece moved (false when the target cell is blocked or off-board)
     */
    constexpr bool move(MoveType move);

    /**
     * @brief rotates the in-flight piece, trying the wall-kick offsets in order until one fits
     * @return whether the piece rotated (false when no kick offset is free)
     */
    constexpr bool rotate(RotationType rotation);

    /**
     * @brief hard-drops and locks the in-flight piece, clears any full rows, then advances to the
     *  next piece
     * @attention spawning the next piece can top the game out; check @ref gameOver afterwards
     * @return the number of rows cleared
     */
    constexpr size_t place();

    /**
     * @brief swaps the in-flight piece into the hold slot, taking the next piece when it is empty,
     *  then spawns the swapped-in piece at the top
     * @attention spawning can top the game out; check @ref gameOver afterwards
     */
    constexpr void hold();

    /** @brief whether the current piece can no longer spawn (top-out) */
    [[nodiscard]] constexpr bool gameOver() const { return _engine.gameOver(); }

    [[nodiscard]] constexpr Board2 const& board() const { return _engine.board(); }

    /** @brief the engine backing this game, for board metrics and the upcoming-piece queue */
    [[nodiscard]] constexpr TetrisEngine const& engine() const { return _engine; }

    /** @brief the type of the in-flight piece */
    [[nodiscard]] constexpr PieceType currentPiece() const { return _engine.currentPiece(); }

    /** @brief the held piece, or @c PieceType::COUNT when the hold slot is empty */
    [[nodiscard]] constexpr PieceType heldPiece() const { return _engine.heldPiece(); }

    /** @brief the upcoming pieces after the current one, soonest first */
    [[nodiscard]] constexpr std::span<PieceType const> lookahead() const { return _engine.lookahead(); }

    /** @brief the in-flight piece's orientation */
    [[nodiscard]] constexpr Orientation orientation() const { return _orientation; }

    /** @brief the in-flight piece's 4x4-window origin (== Board2 offset) */
    [[nodiscard]] constexpr vec2 offset() const { return _offset; }

    /**
     * @brief the offset the in-flight piece would lock at on a hard drop -- its ghost position
     * @pre not @ref gameOver
     */
    [[nodiscard]] constexpr vec2 landingOffset() const;

private:
    /** spawn position: top orientation, 4x4 window centered horizontally (matches the engine) */
    static constexpr vec2 SPAWN{static_cast<int>((WIDTH - PIECE_WIDTH) / 2), 0};

    /**
     * @brief wall-kick offsets tried in order when rotating, the identity first then nudges up and
     *  to the sides; the first that fits is taken
     */
    static constexpr std::array<vec2, 7> ROTATION_KICKS{
        {{0, 0}, {0, -1}, {1, 0}, {-1, 0}, {0, -2}, {2, 0}, {-2, 0}}
    };

    /** @brief places the in-flight piece at the spawn; the engine reports top-out when it cannot fit */
    constexpr void spawn();

    /** @brief whether the current piece fits at @ref orientation and @ref offset */
    [[nodiscard]] constexpr bool available(Orientation orientation, vec2 offset) const;

    TetrisEngine _engine;

    Orientation _orientation = Orientation::TOP;
    vec2 _offset{SPAWN};
};

}

namespace ta3::sim {

constexpr TetrisGame::TetrisGame(uint64_t seed) : _engine{seed} { spawn(); }

constexpr void TetrisGame::reset(uint64_t seed) {
    _engine.reset(seed);
    spawn();
}

constexpr void TetrisGame::spawn() {
    _orientation = Orientation::TOP;
    _offset = SPAWN;
}

constexpr bool TetrisGame::available(Orientation orientation, vec2 offset) const {
    return _engine.board().available(_engine.currentPiece(), orientation, offset);
}

constexpr size_t TetrisGame::step(Instruction instruction) {
    if(auto const move = to_move_type(instruction)) {
        this->move(*move);
        return 0;
    }
    if(auto const rotation = to_rotation_type(instruction)) {
        rotate(*rotation);
        return 0;
    }
    if(instruction == Instruction::PLACE)
        return place();
    if(instruction == Instruction::HOLD)
        hold();
    return 0;
}

constexpr bool TetrisGame::move(MoveType move) {
    vec2 const target = _offset + to_offset(move);
    if(!available(_orientation, target))
        return false;

    _offset = target;
    return true;
}

constexpr bool TetrisGame::rotate(RotationType rotation) {
    Orientation const target = ta3::sim::rotate(*rotation, _orientation);

    for(auto const& kick : ROTATION_KICKS)
        if(available(target, _offset + kick)) {
            _orientation = target;
            _offset += kick;
            return true;
        }
    return false;
}

constexpr size_t TetrisGame::place() {
    size_t const cleared = _engine.place(_orientation, _offset.x);
    spawn();
    return cleared;
}

constexpr void TetrisGame::hold() {
    _engine.hold();
    spawn();
}

constexpr vec2 TetrisGame::landingOffset() const {
    return _engine.board().dropLocation(_engine.currentPiece(), _orientation, _offset);
}

}
