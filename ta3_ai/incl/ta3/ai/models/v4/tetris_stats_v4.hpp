#pragma once
#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

namespace ta3::ai {

/**
 * @brief the per-game state the v4 model reads, carried across placements so deltas like new holes
 *  need no previous board; this is also where custom game state rides along
 * @note the encoder reads its metadata from here instead of recomputing it off the board, so a game
 *  can evolve state that the board alone does not capture
 */
class TetrisStatsV4 {
public:
    /**
     * @brief records the placement of @ref placed (clearing @ref cleared rows, hold now @ref held)
     *  and recomputes the board-derived metrics against @ref board
     * @param board the state to measure -- the placed, pre-clear board for a candidate, or the
     *  committed post-clear board once a move is chosen
     * @post @ref newHoles reflects the change in holes from before this call
     */
    constexpr void advance(size_t cleared, sim::Board2 const& board, sim::PieceType placed, sim::PieceType held) {
        _prevHoles = _holes;
        _holes = board.holes();
        _roughness = board.roughness();
        _linesCleared = cleared;
        _lines += cleared;
        ++_clears[cleared];
        _held = held;
        _last = placed;
        ++_pieces;

        auto prev = board.height(0);
        _aggregateHeight = prev;
        _maxHeight = prev;
        _bumpiness = 0;
        for(auto x = 1uz; x < sim::WIDTH; ++x) {
            auto const h = board.height(x);
            _aggregateHeight += h;
            _maxHeight = std::max(_maxHeight, h);
            _bumpiness += h > prev ? h - prev : prev - h;
            prev = h;
        }
    }

    /** @brief rows cleared by the last recorded placement */
    [[nodiscard]] constexpr size_t linesCleared() const { return _linesCleared; }

    [[nodiscard]] constexpr size_t holes() const { return _holes; }

    /** @brief holes gained over the last placement; negative when a clear freed trapped cells */
    [[nodiscard]] constexpr double newHoles() const { return static_cast<double>(_holes) - static_cast<double>(_prevHoles); }

    [[nodiscard]] constexpr double roughness() const { return _roughness; }

    /** @brief summed stack height across all columns */
    [[nodiscard]] constexpr size_t aggregateHeight() const { return _aggregateHeight; }

    /** @brief tallest column's stack height */
    [[nodiscard]] constexpr size_t maxHeight() const { return _maxHeight; }

    /** @brief summed absolute height difference between adjacent columns */
    [[nodiscard]] constexpr size_t bumpiness() const { return _bumpiness; }

    /** @brief the held piece, or @c PieceType::COUNT when the hold slot is empty */
    [[nodiscard]] constexpr sim::PieceType heldPiece() const { return _held; }

    /** @brief the last placed piece, or @c PieceType::COUNT before the first placement */
    [[nodiscard]] constexpr sim::PieceType lastPiece() const { return _last; }

    [[nodiscard]] constexpr size_t pieces() const { return _pieces; }

    /** total lines cleared this game */
    [[nodiscard]] constexpr size_t lines() const { return _lines; }

    /** placements that cleared k lines, indexed [0, BLOCKS] */
    [[nodiscard]] constexpr std::span<size_t const> clears() const { return _clears; }

    /** final game score: quadratic clear reward (strong tetris bias) plus a small survival term */
    [[nodiscard]] constexpr double score() const {
        double s = 0;
        for(auto k = 1uz; k < _clears.size(); ++k)
            s += static_cast<double>(k * k * _clears[k]); // tetris (16) >> four singles (4)
        return s + SURVIVAL_W * static_cast<double>(_pieces);
    }

private:
    /** survival weight, small so an extra tetris always beats a longer timid game */
    static constexpr double SURVIVAL_W = 0.02;

    size_t _holes = 0;
    size_t _prevHoles = 0;
    double _roughness = 0;
    size_t _linesCleared = 0;
    size_t _aggregateHeight = 0;
    size_t _maxHeight = 0;
    size_t _bumpiness = 0;
    size_t _pieces = 0;
    size_t _lines = 0;
    std::array<size_t, sim::BLOCKS + 1> _clears{};
    sim::PieceType _held = sim::PieceType::COUNT;
    sim::PieceType _last = sim::PieceType::COUNT;
};

/** @brief everything a model's @c extractInputs needs to encode one candidate into its buffer slice */
struct parse_inputs_t {
    TetrisStatsV4 const& stats;
    sim::Board2 const& board;
    std::span<sim::PieceType const> lookahead;
};

}
