#include "ta3/sim/tetris.hpp"

#include "ta3/sim/ui/renderer.hpp"

#include <algorithm>



namespace ta3::sim {
Tetris::Tetris() {
    newPiece();

    if(renderer) renderer->hook(*this);
}
Tetris::~Tetris() { if(renderer) renderer->unhook(*this); }


void Tetris::update(Instruction const instruction) {
    _stats.instruction(instruction);

    _lastInstruction = instruction;

    if(instruction == Instruction::NONE) return;

    if(auto const move = to_move_type(instruction); move) movePiece(*move);
    else if(auto const rotation = to_rotation_type(instruction); rotation) rotatePiece(*rotation);
    else
        switch(instruction) {
            case Instruction::PLACE: place();
                break;
            case Instruction::HOLD: hold();
                break;
            default: std::unreachable();
        }


    ++_forceDownMoveCounter %= NETWORK_MOVE_DOWN_DELAY;

    if(_forceDownMoveCounter == 0) moveDown(true);
}

std::string Tetris::string() const {
    constexpr size_t length = COLS * 3 + 2;

    Board board{_board};

    for(auto& coord : _currentPiece->blocks())
        board[coord] = BlockType::SIZE;
    for(auto& coord : _currentPiece->dummyBlocks())
        board[coord] = BlockType::SIZE2;

    for(int x = 0; x < WIDTH; x++) {
        auto const y = board.heightMapView()[x];
        if(std::cmp_less(y, HEIGHT)) board[y, x] = BlockType::SIZE3;
    }

    std::string out{};
    std::string border(length, '-');
    border.push_back('\n');

    out.append(border);
    for(size_t y = 0; y < ROWS; y++) {
        std::string line{};
        line.reserve(length);

        line.push_back('|');

        for(size_t x = 0; x < COLS; x++) {
            if(x != 0) line.push_back(' ');

            auto const block = board[y, x];
            if(block == BlockType::SIZE3) line.append("##");
            else if(block == BlockType::SIZE) line.append("{}");
            else if(block == BlockType::SIZE2) line.append("()");
            else if(block != BlockType::NONE) line.append("[]");
            else if(std::cmp_equal(y, board.highest())) line.append("--");
            else line.append("  ");
        }
        line.push_back('|');
        line.push_back('\n');
        out.append(line);
    }
    out.append(border);

    return out;

}
bool Tetris::inBounds(vec2 const& coord, bool no_ceil) {
    return cth::num::in(
        coord.x, 0, static_cast<int>(WIDTH),
        coord.y, no_ceil ? -static_cast<int>(BLOCKS) : 0, static_cast<int>(HEIGHT)
    );
}
void Tetris::hold() {
    if(!_held) {
        _held = _pieceQueue.front();
        _pieceQueue.pop_front();
    } else {
        auto const tmp = *_held;
        _held = _currentPiece->type();
        _currentPiece = std::nullopt;
        _pieceQueue.pop_front();
        _pieceQueue.push_front(tmp);
    }
    _pieceQueue.push_front(PieceType::COUNT);

    newPiece();
}

void Tetris::movePiece(MoveType move) {
    if(move == MoveType::DOWN) moveDown();
    else if(!_currentPiece->move(move)) {
        _stats.illegalMove(move);
        return;
    }

    _stats.move(move);

}
void Tetris::moveDown(bool force) {
    _forceDownMoveCounter = 0;
    auto const amount = std::max(1, _moveDowns - MAX_MOVE_DOWNS);

    for(int i = 0; i < amount; i++)
        if(!_currentPiece->move(MoveType::DOWN)) {
            place();
            break;
        }
    _moveDowns += (force && _moveDowns < MAX_MOVE_DOWNS) ? FORCE_DOWN_WEIGHT : 1;
}


void Tetris::rotatePiece(RotationType rotation) {
    _currentPiece->rotate(static_cast<int>(*rotation));
    _stats.rotation();
}


void Tetris::newPiece() {
    fillPieceQueue();

    _pieceQueue.pop_front();
    _currentPiece = Piece::New(_board, _pieceQueue.front());

    if(!_currentPiece) end();
}



void Tetris::place() {
    _moveDowns = 0;
    _forceDownMoveCounter = 0;


    _lastPlace = _currentPiece->placeTo(_board);

    _stats.piecePlaced(_board.clearLines(), _board);

    newPiece();
}



void Tetris::fillPieceQueue() {
    while(_pieceQueue.size() <= *PieceType::COUNT) {
        std::vector<PieceType> pieces(*PieceType::COUNT);
        for(size_t i = 0; i < pieces.size(); i++)
            pieces[i] = static_cast<PieceType>(*PieceType::FIRST + i);

        std::ranges::shuffle(pieces, _rnd);

        _pieceQueue.insert_range(_pieceQueue.end(), pieces);
    }
}


void Tetris::end() { _active = false; }

std::array<PieceType, PIECE_QUEUE_SIZE> Tetris::pieceQueue() const {
    std::array<PieceType, PIECE_QUEUE_SIZE> queue{};
    std::ranges::copy(std::ranges::subrange{_pieceQueue.begin(), _pieceQueue.begin() + PIECE_QUEUE_SIZE}, queue.begin());
    return queue;
}

}
