#pragma once
#include "ta3/sim/board2.hpp"
#include "ta3/sim/tetris_defs.hpp"
#include "ta3/sim/utility.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"
#include "ta3/sim/pieces/piece_offsets.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

namespace ta3::sim {

/**
 * tetris, one hard-drop placement at a time -- logical core, no move or gravity layer
 * @details
 * - current piece committed straight to its landing; playable @c Tetris wraps this
 * - seeded 7-bag supply, one seed reproduces the whole game
 */
class TetrisEngine {
public:
    /** new game, sequence from @ref seed */
    constexpr explicit TetrisEngine(uint64_t seed);

    /** restart, fresh sequence from @ref seed */
    constexpr void reset(uint64_t seed);

    /** clear-count sentinel for a game-ending move */
    static constexpr size_t DIED = std::numeric_limits<size_t>::max();

    /** absent piece / empty hold sentinel */
    static constexpr auto NO_PIECE = PieceType::COUNT;

    /**
     * hard-drop current piece, clear rows, advance
     * @param orientation drop orientation
     * @param x window x-origin (== Board2 offset.x), ~[-2, WIDTH + 2]
     * @pre legal placement (fits at spawn, drops without collision)
     * @return rows cleared, or @ref DIED on a game-ending move
     */
    constexpr size_t place(Orientation orientation, int x);

    /** swap current piece with hold slot, taking the next piece when empty */
    constexpr void hold();

    /** current piece can no longer spawn (top-out) */
    [[nodiscard]] constexpr bool gameOver() const { return _gameOver; }

    [[nodiscard]] constexpr Board2 const& board() const { return _board; }

    /** piece to place next */
    [[nodiscard]] constexpr PieceType currentPiece() const { return _queue.front(); }

    /** held piece, or @ref NO_PIECE if empty */
    [[nodiscard]] constexpr PieceType heldPiece() const { return _held; }

    /** whether a piece is held */
    [[nodiscard]] constexpr bool holdsPiece() const { return _held != NO_PIECE; }

    /** upcoming pieces after the current, soonest first */
    [[nodiscard]] constexpr std::span<PieceType const> lookahead() const {
        return std::span<PieceType const>{_queue}.subspan(1);
    }

private:
    /** spawn: top orientation, 4x4 window centered horizontally */
    static constexpr vec2 SPAWN{static_cast<int>((WIDTH - PIECE_WIDTH) / 2), 0};

    /** next piece from the 7-bag, reshuffling when exhausted */
    [[nodiscard]] constexpr PieceType drawPiece();

    /** drop current piece, refill lookahead from the bag */
    constexpr void advanceQueue();

    /** set @ref _gameOver if current piece cannot spawn */
    constexpr void updateGameOver();

    Board2 _board{};

    /** visible queue: @c [0] current, rest lookahead */
    std::array<PieceType, PIECE_QUEUE_SIZE> _queue{};

    /** held piece, or @ref NO_PIECE if empty */
    PieceType _held = NO_PIECE;
    bool _gameOver = false;

    /** 7-bag draw order */
    std::array<PieceType, *NO_PIECE> _bag{};
    size_t _bagPos = 0;

    Xoshiro256ss _rng;
};

}

namespace ta3::sim {

constexpr TetrisEngine::TetrisEngine(uint64_t seed) { reset(seed); }

constexpr void TetrisEngine::reset(uint64_t seed) {
    _board = Board2{};
    _held = NO_PIECE;
    _gameOver = false;
    _bagPos = 0;
    _rng.seed(seed);

    for(auto& piece : _queue)
        piece = drawPiece();

    updateGameOver();
}

constexpr PieceType TetrisEngine::drawPiece() {
    if(_bagPos > 0)
        return _bag[--_bagPos];

    for(size_t i = 0; i < _bag.size(); ++i)
        _bag[i] = static_cast<PieceType>(i);

    // fisher-yates shuffle
    for(size_t i = _bag.size() - 1; i > 0; --i)
        std::swap(_bag[i], _bag[_rng.bounded(i + 1)]);

    _bagPos = _bag.size() - 1;
    return _bag.back();
}

constexpr void TetrisEngine::advanceQueue() {
    for(size_t i = 1; i < _queue.size(); ++i)
        _queue[i - 1] = _queue[i];
    _queue.back() = drawPiece();
}

constexpr size_t TetrisEngine::place(Orientation orientation, int x) {
    PieceType const type = currentPiece();
    vec2 const landing = _board.dropLocation(type, orientation, vec2{x, 0});
    _board.place(type, orientation, landing);

    size_t const cleared = _board.clearLines();
    advanceQueue();
    updateGameOver();
    return _gameOver ? DIED : cleared;
}

constexpr void TetrisEngine::hold() {
    if(_held == NO_PIECE) {
        _held = _queue.front();
        advanceQueue();
    }
    else
        std::swap(_held, _queue.front());

    updateGameOver();
}

constexpr void TetrisEngine::updateGameOver() {
    _gameOver = !_board.available(currentPiece(), Orientation::TOP, SPAWN);
}

}
