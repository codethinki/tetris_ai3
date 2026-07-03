#pragma once
#include "ta3/ai/model_defs.hpp"

#include "ta3/ai/models/v4/tetris_stats_v4.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>
#include <ta3/sim/tetris_defs.hpp>

#include <dlib/matrix/matrix.h>
#include <dlib/matrix/matrix_mat.h>

#include <span>

namespace ta3::sim {
class Board;
}

namespace ta3::ai {

class input_data_v4 {

public:
    static constexpr size_t HEIGHT = std::min(ta3::sim::HEIGHT, 20uz), WIDTH = ta3::sim::WIDTH;
    static constexpr size_t BOARD_SIZE = WIDTH * HEIGHT;

private:
    static constexpr ta3::sim::szvec2 BOARD_SECT{0, BOARD_SIZE};


    static constexpr ta3::sim::szvec2 CLEAR_SECT{ta3::sim::norm1(BOARD_SECT), 1};
    static constexpr ta3::sim::szvec2 HOLES_SECT{ta3::sim::norm1(CLEAR_SECT), 1};
    static constexpr ta3::sim::szvec2 NEW_HOLES_SECT{ta3::sim::norm1(HOLES_SECT), 1};
    static constexpr ta3::sim::szvec2 ROUGH_SECT{ta3::sim::norm1(NEW_HOLES_SECT), 1};
    static constexpr ta3::sim::szvec2 PIECES_SECT{ta3::sim::norm1(ROUGH_SECT), ta3::sim::PIECE_QUEUE_SIZE};

public:
    static constexpr size_t SIZE = ta3::sim::norm1(PIECES_SECT);
    static constexpr size_t METADATA_SIZE = SIZE - BOARD_SIZE;

    using board_matrix_t = dlib::matrix<ai::data_t, HEIGHT, WIDTH>;
    using metadata_t = std::span<ai::data_t const, METADATA_SIZE>;

    input_data_v4(ta3::sim::Board const& prev, ta3::sim::Board const& current, std::span<ta3::sim::PieceType const> pieces);

    [[nodiscard]] board_matrix_t board_matrix() const { return dlib::mat(_data.data(), HEIGHT, WIDTH); }
    [[nodiscard]] metadata_t metadata_vector() const { return metadata_t{&_data[BOARD_SIZE], METADATA_SIZE}; }

    [[nodiscard]] auto board_data() const { return std::span<ai::data_t const, BOARD_SIZE>{_data.data(), BOARD_SIZE}; }
    [[nodiscard]] auto metadata() const { return std::span<ai::data_t const, METADATA_SIZE>{&_data[BOARD_SIZE], METADATA_SIZE}; }

    /**
     * @brief packs a single candidate state directly into @ref out -- the board's occupancy plus the
     *  scalar metadata read off the stats -- with no intermediate object
     * @param in the candidate's stats (every metadata scalar + held piece), board (occupancy grid)
     *  and lookahead
     * @param out exactly @ref SIZE values to overwrite
     * @note the piece section is the lookahead followed by the held piece in its last slot
     */
    static constexpr void extractInputs(parse_inputs_t const& in, std::span<ai::data_t> out) {
        // crop the hidden buffer rows above the visible field off the top
        constexpr size_t TOP_CROP = ta3::sim::HEIGHT - HEIGHT;
        for(size_t x = 0; x < WIDTH; ++x) {
            uint32_t const column = in.board.raw_column(x);
            for(size_t y = 0; y < HEIGHT; ++y)
                out[BOARD_SECT[0] + y * WIDTH + x] = static_cast<ai::data_t>((column >> (y + TOP_CROP)) & 1u);
        }

        out[CLEAR_SECT[0]] = static_cast<ai::data_t>(in.stats.linesCleared());
        out[HOLES_SECT[0]] = static_cast<ai::data_t>(in.stats.holes());
        out[NEW_HOLES_SECT[0]] = static_cast<ai::data_t>(in.stats.newHoles());
        out[ROUGH_SECT[0]] = static_cast<ai::data_t>(in.stats.roughness());

        for(size_t i = 0; i < in.lookahead.size(); ++i)
            out[PIECES_SECT[0] + i] = static_cast<ai::data_t>(*in.lookahead[i]);
        out[PIECES_SECT[0] + in.lookahead.size()] = static_cast<ai::data_t>(*in.stats.heldPiece());
    }

private:
    void setBoard(ta3::sim::Board const& prev, ta3::sim::Board const& current);
    void setLinesCleared(ta3::sim::Board const& board);
    void setHoles(ta3::sim::Board const& current);
    void setNewHoles(ta3::sim::Board const& prev, ta3::sim::Board const& current);
    void setRoughness(ta3::sim::Board const& board);
    void setPieces(std::span<ta3::sim::PieceType const> pieces);

    std::array<ai::data_t, SIZE> _data{};
};
}
