#pragma once
#include "ta3/ai/models/v4/input_data_v4.hpp"

#include <ta3/sim/board.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <array>
#include <span>
#include <vector>

namespace ta3::ai {
namespace test {

inline sim::Board empty_board() { return sim::Board{}; }

inline void fill_row(sim::Board& board, int row) {
    board.place(sim::PieceType::I, sim::Orientation::TOP, sim::vec2{0, row - 1});
    board.place(sim::PieceType::I, sim::Orientation::TOP, sim::vec2{4, row - 1});
    board.place(sim::PieceType::O, sim::Orientation::TOP, sim::vec2{7, row - 1});
}

inline std::array<sim::PieceType, sim::PIECE_QUEUE_SIZE> default_piece_queue() {
    return {sim::PieceType::I, sim::PieceType::O, sim::PieceType::T, sim::PieceType::S, sim::PieceType::Z};
}

inline input_data_v4 make_input(sim::Board const& prev, sim::Board const& current, std::span<sim::PieceType const> pieces) {
    return input_data_v4{prev, current, pieces};
}

inline input_data_v4 make_input(sim::Board const& prev, sim::Board const& current) {
    auto const queue = default_piece_queue();
    return make_input(prev, current, queue);
}

inline std::vector<double> zero_weights(size_t count) { return std::vector<double>(count, 0.0); }

}
}
