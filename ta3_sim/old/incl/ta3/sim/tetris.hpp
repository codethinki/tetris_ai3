#pragma once
#include "ta3/sim/board.hpp"
#include "ta3/sim/tetris_defs.hpp"
#include "ta3/sim/tetris_stats.hpp"
#include "ta3/sim/pieces/piece.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"

#include <cth/macro.hpp>
#include <cth/io/log.hpp>
#include <cth/meta/md.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <optional>
#include <random>
#include <string>

namespace ta3::sim {
using std::chrono::high_resolution_clock;



class Tetris {
public:
    static constexpr int MAX_MOVE_DOWNS = 20;
    static constexpr int FORCE_DOWN_WEIGHT = 5;
    static constexpr size_t NETWORK_MOVE_DOWN_DELAY = 9; //TEMP


    template<class T>
    using board_view_t = cth::mta::mdspan_t<T, HEIGHT, WIDTH>;

    template<class T>
    using board_t = std::array<T, BOARD_SIZE>;

    explicit Tetris();
    ~Tetris();



    void update(Instruction instruction = Instruction::NONE);


    [[nodiscard]] std::string string() const;
    [[nodiscard]] static bool inBounds(vec2 const& coord, bool no_ceil = false);

private:
    void hold();
    void movePiece(MoveType move);
    void moveDown(bool force = false);

    void rotatePiece(RotationType rotation);

    void newPiece();


    void place();

    void fillPieceQueue();

    void end();


    bool _active = true;

    std::mt19937 _rnd{std::random_device{}()};
    std::uniform_int_distribution<uint32_t> _distribution{0, *PieceType::COUNT - 1};

    std::deque<PieceType> _pieceQueue;


    Board _board{};


    uint32_t _forceDownMoveCounter = 0;
    int _moveDowns = 0;
    Instruction _lastInstruction = Instruction::NONE;


    Stats _stats;


    std::optional<Piece> _currentPiece = std::nullopt;
    std::optional<PieceType> _held = std::nullopt;
    vec2 _lastPlace{};

public:
    [[nodiscard]] bool gameOver() const { return !_active; }
    [[nodiscard]] PieceType heldPiece() const {
        CTH_CRITICAL(
            _pieceQueue.size() < 2 && !_held,
            "piece queue must be at least size 2 or a piece must be held"
        ) {}

        return _held.has_value() ? *_held : _pieceQueue[1];
    }
    [[nodiscard]] size_t pieceQueueSize() const { return _pieceQueue.size(); }
    [[nodiscard]] PieceType currentPiece() const {
        CTH_CRITICAL(!_currentPiece, "no current piece") {}
        return _currentPiece->type();
    }


    [[nodiscard]] Instruction lastInstruction() const { return _lastInstruction; }

    [[nodiscard]] auto const& stats() const { return _stats; }
    [[nodiscard]] declauto linesCleared() const { return _stats.linesCleared(); }
    [[nodiscard]] auto const& board() const { return _board; }
    [[nodiscard]] std::array<PieceType, PIECE_QUEUE_SIZE> pieceQueue() const;
    [[nodiscard]] declauto pieceCoords() const { return _currentPiece->blocks(); }
    [[nodiscard]] declauto pieceOffset() const { return _currentPiece->offset(); }
    [[nodiscard]] auto lastPlaceLocation() const { return _lastPlace; }
    [[nodiscard]] declauto pieceOrientation() const { return _currentPiece->orientation(); }
    [[nodiscard]] declauto highestBlock() const { return _board.highest(); }
};


}
