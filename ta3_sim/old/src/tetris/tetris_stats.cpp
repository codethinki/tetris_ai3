#include "ta3/sim/tetris_stats.hpp"

#include <utility>

#include "ta3/sim/tetris_defs.hpp"

namespace ta3::sim {

void Stats::instruction(Instruction instruction) {
    ++_instructions[*instruction];
    // CTH_LOG(true, "received instruction: {}", instruction) {}
}

void Stats::illegalMove(MoveType move) {
    //CTH_LOG(true, "received illegal move: {}", move) {}

    ++_illegalMoves[*move];
    ++_totalMoves;
}

void Stats::move(MoveType move) {
    //CTH_LOG(true, "received move: {}", move) {}

    ++_moves[*move];
    ++_totalMoves;
}

void Stats::rotation() {
    //CTH_LOG(true, "received rotation") {}
    ++_rotations;
}

void Stats::piecePlaced(size_t cleared, Board const& grid) {
    ++_pieces;
    ++_clearedLines[cleared];

    evalGrid(grid);
    //CTH_LOG(true, "received piece place of {}, with {} cleared", type, cleared) {}
}

void Stats::evalGrid(Board const& board) {


    _roughnessSum += board.roughness();
    _holesSum += board.holes();
    _heightSum += calcHeight(board);
}

double Stats::calcDensity(Board const& grid) {
    size_t blocks = 0;

    auto const highest = grid.highest();
    if(std::cmp_greater_equal(highest, HEIGHT)) return 1;

    for(int y = highest; std::cmp_less(y, HEIGHT); y++)
        blocks += std::ranges::count_if(grid.line(y), [](auto const& block) { return Board::occupied(block); });


    return static_cast<double>(blocks) / ((static_cast<int>(HEIGHT) - highest) * static_cast<int>(WIDTH));
}
double Stats::calcHeight(Board const& grid) { return static_cast<double>(HEIGHT) - grid.highest(); }
}
