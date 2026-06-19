#pragma once
#include "ta3/sim/board2.hpp"
#include "ta3/sim/tetris_defs.hpp"
#include "ta3/sim/utility.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"
#include "ta3/sim/pieces/piece_offsets.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace ta3::sim {

/**
 * @brief tetris advanced one hard-drop placement at a time -- the logical core, no move or gravity
 *  layer. the current piece is committed straight to its landing; a playable @c Tetris wraps this.
 * @note the piece supply is a seeded 7-bag, so one seed reproduces the whole game
 */
class TetrisEngine {
public:
    /** @brief starts a game whose entire piece sequence is determined by @ref seed */
    constexpr explicit TetrisEngine(uint64_t seed);

    /** @brief restarts the game with a fresh piece sequence from @ref seed */
    constexpr void reset(uint64_t seed);

    /**
     * @brief hard-drops the current piece in @ref orientation at window x-origin @ref x, clears full
     *  rows, and advances to the next piece
     * @param x x-origin of the piece's 4x4 window (== Board2 offset.x), naturally in [-2, WIDTH+2]
     * @pre the placement is legal: the piece fits at the spawn and drops without collision
     * @post @ref gameOver may become true if the next piece cannot spawn
     * @return the number of rows cleared by this placement
     */
    constexpr size_t place(Orientation orientation, int x);

    /** @brief swaps the current piece with the hold slot, taking the next piece when it is empty */
    constexpr void hold();

    /** @brief whether the current piece can no longer spawn (top-out) */
    [[nodiscard]] constexpr bool gameOver() const { return _gameOver; }

    [[nodiscard]] constexpr Board2 const& board() const { return _board; }

    /** @brief the piece to be placed next */
    [[nodiscard]] constexpr PieceType currentPiece() const { return _queue.front(); }

    /** @brief the held piece, or @c PieceType::COUNT when the hold slot is empty */
    [[nodiscard]] constexpr PieceType heldPiece() const { return _held; }

    /** @brief the upcoming pieces after the current one, soonest first */
    [[nodiscard]] constexpr std::span<PieceType const> lookahead() const {
        return std::span<PieceType const>{_queue}.subspan(1);
    }

private:
    /** spawn position: top orientation, 4x4 window centered horizontally */
    static constexpr vec2 SPAWN{static_cast<int>((WIDTH - PIECE_WIDTH) / 2), 0};

    /** @brief next piece from the 7-bag, reshuffling a fresh bag when the current one is exhausted */
    [[nodiscard]] constexpr PieceType drawPiece();

    /** @brief drops the current piece out of the queue and refills the lookahead from the bag */
    constexpr void advanceQueue();

    /** @brief sets @ref _gameOver when the current piece cannot spawn */
    constexpr void updateGameOver();

    Board2 _board{};

    /** visible queue: @c [0] is the current piece, the rest are lookahead */
    std::array<PieceType, PIECE_QUEUE_SIZE> _queue{};

    /** held piece, or @c PieceType::COUNT for an empty hold slot */
    PieceType _held = PieceType::COUNT;
    bool _gameOver = false;

    /** 7-bag draw order and the read position into it */
    std::array<PieceType, *PieceType::COUNT> _bag{};
    size_t _bagPos = 0;

    Xoshiro256ss _rng;
};

}

namespace ta3::sim {

constexpr TetrisEngine::TetrisEngine(uint64_t seed) { reset(seed); }

constexpr void TetrisEngine::reset(uint64_t seed) {
    _board = Board2{};
    _held = PieceType::COUNT;
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
    vec2 const landing = _board.dropPlace(type, orientation, vec2{x, 0});
    _board.place(type, orientation, landing);

    size_t const cleared = _board.clearLines();
    advanceQueue();
    updateGameOver();
    return cleared;
}

constexpr void TetrisEngine::hold() {
    if(_held == PieceType::COUNT) {
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
