#pragma once
#include "ta3/ai/model.hpp"

#include <ta3/sim/board.hpp>
#include <ta3/sim/pieces/piece.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>
#include <ta3/sim/tetris.hpp>
#include <ta3/sim/tetris_defs.hpp>

#include <span>
#include <string>
#include <unordered_set>
#include <vector>

namespace ta3::ai {

class MultiTetris {
public:
    struct variation_t {
        sim::vec2 offset;
        sim::Orientation orientation;
        sim::PieceType type;
    };

    static constexpr size_t DIMENSIONS = *sim::Orientation::SIZE * sim::WIDTH;

    explicit MultiTetris(sim::Tetris& tetris);
    ~MultiTetris() = default;


    bool genVariations();
    void update(size_t index);

    [[nodiscard]] bool gameOver() const;

    [[nodiscard]] std::string string() const;

private:
    void updateBoard();
    void resetVariations();
    void genVariation(sim::Piece const& piece);

    void genDirVariations(sim::PieceType type, sim::MoveType direction, bool gen_origin);



    sim::Board _board{};
    std::vector<sim::Board> _boards{};
    std::vector<variation_t> _variations{};
    std::unordered_set<uint32_t> _variationHashes{};

    variation_t _lastVariation{};

    sim::Tetris* _tetris;


    [[nodiscard]] sim::cboard_view_t view() const;

public:
    [[nodiscard]] std::span<sim::Board const> boards() const { return _boards; }

    [[nodiscard]] std::vector<input_t> inputs() const;
    [[nodiscard]] size_t variants() const { return _boards.size(); }

    [[nodiscard]] sim::Tetris const& game() const;
    [[nodiscard]] double adjustedScore() const;


    MultiTetris(MultiTetris const& other) = delete;
    MultiTetris& operator=(MultiTetris const& other) = delete;
    MultiTetris(MultiTetris&& other) noexcept = default;
    MultiTetris& operator=(MultiTetris&& other) noexcept = default;
};

struct MultiTetrisGame {
    sim::Tetris tetris{};
    MultiTetris multi{tetris};
};

}
