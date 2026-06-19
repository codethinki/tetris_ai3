#pragma once
#include "ta3/sim/board.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"

#include <array>
#include <cstddef>

namespace ta3::sim {
struct Stats {

    void instruction(Instruction instruction);
    void illegalMove(MoveType move);
    void move(MoveType move);
    void rotation();

    void piecePlaced(size_t cleared, Board const& grid);

private:
    void evalGrid(Board const& board);

    static double calcDensity(Board const& grid);
    static double calcHeight(Board const& grid);


    double _heightSum = 0;
    double _roughnessSum = 0;
    double _holesSum = 0;

    size_t _rotations = 0;
    size_t _totalMoves = 0;
    size_t _pieces = 0;

    std::array<size_t, BLOCKS + 1> _clearedLines{};

    std::array<size_t, *Instruction::SIZE> _instructions{};
    std::array<size_t, *MoveType::SIZE> _moves{};
    std::array<size_t, *MoveType::SIZE> _illegalMoves{};

public:
    [[nodiscard]] constexpr auto pieces() const { return _pieces; }
    [[nodiscard]] constexpr auto totalMoves() const { return _totalMoves; }
    [[nodiscard]] constexpr auto rotations() const { return _rotations; }

    [[nodiscard]] constexpr auto heightSum() const { return _heightSum; }
    [[nodiscard]] constexpr auto holesSum() const { return _holesSum; }
    [[nodiscard]] constexpr auto roughnessSum() const { return _roughnessSum; }


    [[nodiscard]] constexpr auto const& linesCleared() const { return _clearedLines; }
};
}
