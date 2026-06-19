#include "ta3/ai/multi_tetris.hpp"

#include "ta3/ai/fitness.hpp"

namespace ta3::ai {
namespace {
    constexpr uint32_t hash_top_piece(sim::piece_coords_t const& blocks) {
        uint32_t result{};
        for(size_t i = 0; i < blocks.size(); i++) {
            auto const& block = blocks[i];

            static constexpr size_t BLOCK_BITS = 8;
            static constexpr size_t X_BITS = 4;
            result |= (block.x & 0b1111) << (BLOCK_BITS * i);
            result |= (block.y & 0b1111) << (X_BITS + BLOCK_BITS * i);
        }
        return result;
    }
}


MultiTetris::MultiTetris(sim::Tetris& tetris) : _tetris{&tetris} {
    _boards.reserve(DIMENSIONS);
    _variations.reserve(DIMENSIONS);
    _variationHashes.reserve(DIMENSIONS);
}



void MultiTetris::updateBoard() {
    _board = _tetris->board();
    _board.normalize();
}
void MultiTetris::resetVariations() {
    _boards.clear();
    _variations.clear();
    _variationHashes.clear();
}

bool MultiTetris::genVariations() {
    if(gameOver())
        return false;
    resetVariations();
    updateBoard();

    genDirVariations(_tetris->currentPiece(), sim::MoveType::LEFT, true);
    genDirVariations(_tetris->currentPiece(), sim::MoveType::RIGHT, false);

    auto const held = _tetris->heldPiece();
    genDirVariations(held, sim::MoveType::LEFT, true);
    genDirVariations(held, sim::MoveType::RIGHT, false);

    return !_boards.empty();
}

void MultiTetris::update(size_t index) {
    auto const [offset, orientation, type] = _variations[index];

    if(type != _tetris->currentPiece())
        _tetris->update(sim::Instruction::HOLD);

    CTH_STABLE_THROW(_tetris->currentPiece() != type, "failed to switch to correct piece")
        details->add("expected: {}, actual: {}", sim::to_string(type), sim::to_string(_tetris->currentPiece()));


    auto const pieceOffset = _tetris->pieceOffset();

    while(orientation != _tetris->pieceOrientation())
        _tetris->update(sim::Instruction::RRIGHT);

    auto const diff = offset.x - pieceOffset.x;
    for(int i = 0; i < std::abs(diff); i++)
        _tetris->update(diff < 0 ? sim::Instruction::LEFT : sim::Instruction::RIGHT);

    _tetris->update(sim::Instruction::PLACE);

    auto const last = _tetris->lastPlaceLocation();
    CTH_STABLE_THROW(last != offset, "piece did not reach destination")
        details->add("{}", string());

    _lastVariation = _variations[index];
}
bool MultiTetris::gameOver() const {
    return _tetris->gameOver() || _tetris->highestBlock() < static_cast<int>(sim::BLOCKS) + 1;
}
std::string MultiTetris::string() const {
    return std::format(
        "{0}\n score: {1}\n lines cleared: {2}\n last location: [{3}, {4}], {5}\n",
        _tetris->string(),
        calculateFitness(_tetris->stats(), gameOver()),
        _tetris->linesCleared(),
        _lastVariation.offset.x,
        _lastVariation.offset.y,
        _lastVariation.orientation
    );
}
void MultiTetris::genVariation(sim::Piece const& piece) {
    auto const hash = hash_top_piece(piece.blocks());

    if(_variationHashes.contains(hash))
        return;

    _variationHashes.insert(hash);

    _boards.emplace_back(_board);

    auto const offset = piece.placeTo(_boards.back());
    _variations.emplace_back(offset, piece.orientation(), piece.type());
}
void MultiTetris::genDirVariations(sim::PieceType type, sim::MoveType direction, bool gen_origin) {
    constexpr std::array<int, 4> rotations{-1, 1, 2};

    sim::Piece const refPiece{_board, type};

    for(auto const rotation : rotations) {
        auto piece = refPiece;
        auto const rotated = piece.rotate(rotation);
        CTH_STABLE_THROW(!rotated, "failed to rotate piece") {}

        if(gen_origin)
            genVariation(piece);

        while(piece.move(direction))
            genVariation(piece);
    }
}

sim::cboard_view_t MultiTetris::view() const { return _tetris->board().view(); }

std::vector<input_t> MultiTetris::inputs() const {
    auto const boards = this->boards();

    std::vector<input_t> inputs{};
    inputs.reserve(boards.size());

    for(size_t i = 0; i < boards.size(); i++) {
        auto pieceQueue = _tetris->pieceQueue();
        auto const held = _tetris->heldPiece();
        pieceQueue.back() = held;

        if(held == _variations[i].type)
            std::swap(pieceQueue.front(), pieceQueue.back());

        inputs.emplace_back(this->_board, boards[i], std::span<sim::PieceType const>{pieceQueue});
    }

    return inputs;
}
sim::Tetris const& MultiTetris::game() const { return *_tetris; }
double MultiTetris::adjustedScore() const { return calculateFitness(_tetris->stats(), gameOver()); }
}
