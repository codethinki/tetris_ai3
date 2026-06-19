#include "ta3/ai/models/v4/input_data_v4.hpp"

#include <ta3/sim/board.hpp>

namespace ta3::ai {

input_data_v4::input_data_v4(ta3::sim::Board const& prev, ta3::sim::Board const& current, std::span<ta3::sim::PieceType const> pieces) {
    setBoard(prev, current);
    setLinesCleared(current);
    setHoles(current);
    setNewHoles(prev, current);
    setRoughness(current);
    setPieces(pieces);
}

void input_data_v4::setBoard(ta3::sim::Board const& prev, ta3::sim::Board const& current) {
    std::span board{&_data[BOARD_SECT[0]], BOARD_SECT[1]};

    auto const prevView = prev.view();
    auto const currentView = current.view();

    // crop the hidden buffer rows above the visible field off the top
    constexpr size_t TOP_CROP = ta3::sim::HEIGHT - HEIGHT;
    for(size_t y = 0; y < HEIGHT; y++)
        for(size_t x = 0; x < WIDTH; x++) {
            auto const currentBlock = currentView[y + TOP_CROP, x];
            board[y * WIDTH + x] = static_cast<data_t>(*currentBlock + (*currentBlock - *prevView[y + TOP_CROP, x]));
        }
}

void input_data_v4::setLinesCleared(ta3::sim::Board const& board) { _data[CLEAR_SECT[0]] = static_cast<data_t>(board.fullLines().size()); }

void input_data_v4::setHoles(ta3::sim::Board const& current) { _data[HOLES_SECT[0]] = static_cast<data_t>(current.holes()); }

void input_data_v4::setNewHoles(ta3::sim::Board const& prev, ta3::sim::Board const& current) {
    _data[NEW_HOLES_SECT[0]] = static_cast<data_t>(current.holes() - prev.holes());
}
void input_data_v4::setRoughness(ta3::sim::Board const& board) {
    auto const view = board.heightMapView();

    data_t sum = 0;
    for(int i = 1; i < view.size(); i++) sum += static_cast<data_t>(std::abs(view[i] - view[i - 1]));

    _data[ROUGH_SECT[0]] = sum / static_cast<data_t>(view.size());
}
void input_data_v4::setPieces(std::span<ta3::sim::PieceType const> pieces) {
    std::ranges::transform(pieces, _data.begin() + PIECES_SECT[0], [](ta3::sim::PieceType piece) { return static_cast<data_t>(*piece); });
}

}
