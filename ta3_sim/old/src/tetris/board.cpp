#include "ta3/sim/board.hpp"

#include "ta3/sim/pieces/piece_offsets.hpp"

#include <span>
#include <utility>

namespace ta3::sim {


bool Board::available(PieceType piece, Orientation orientation, vec2 offset) const {
    auto const blocks = piece_blocks(piece, orientation, offset);

    if(!std::ranges::all_of(blocks, [](vec2 const& block) { return inBounds(block); }))
        return false;

    if(offset.y + BLOCKS < _highest)
        return true;


    return std::ranges::all_of(blocks, [this](vec2 const& block) { return available(block); });
}

std::optional<vec2> Board::placeOffset(PieceType type, Orientation orientation, vec2 offset) const {
    auto const blocks = piece_blocks(type, orientation, offset);

    int diff = HEIGHT;
    for(auto const& block : blocks)
        diff = std::min(diff, _heightMap[block.x] - block.y);

    if(diff >= 1)
        return vec2{offset.x, offset.y + diff - 1};

    if(!available(blocks))
        return std::nullopt;

    do { ++offset.y; }
    while(available(type, orientation, offset));
    return vec2{offset.x, offset.y - 1};
}

void Board::place(this Board& self, PieceType type, Orientation orientation, vec2 offset) {
    CTH_CRITICAL(!self.available(type, orientation, offset), "place location must be available") {}

    auto const blocks = piece_blocks(type, orientation, offset);

    for(auto const block : blocks)
        self[block] = to_block_type(type);

    self.updateMetadata(blocks);

    self.debugCheck();
}

size_t Board::clearLines() {
    auto const lines = fullLines();

    for(auto const y : lines) {
        auto const l = line(y);
        std::ranges::fill(l, EMPTY);
    }
    for(auto const lineY : lines) {
        for(int y = lineY - 1; y >= _highest; --y) {
            auto const src = line(y);
            auto const dst = line(y + 1);

            std::ranges::copy(src, dst.begin());
            std::ranges::fill(src, EMPTY);
        }
        ++_highest;
    }
    calcHeightMap();
    calcHoles();

    return lines.size();
}
std::vector<int> Board::fullLines() const {
    std::vector<int> lines{};
    lines.reserve(BLOCKS);

    auto const lowestHighest = std::ranges::max(_heightMap);

    for(int y = lowestHighest; std::cmp_less(y, HEIGHT) && lines.size() < BLOCKS; y++) {
        auto const l = line(y);
        bool const remove = std::ranges::all_of(l, [](BlockType const& block) { return Board::occupied(block); });

        if(remove)
            lines.push_back(y);
    }
    return lines;
}
void Board::normalize() {
    for(auto& block : _grid)
        block = occupied(block) ? BLOCKED : EMPTY;
}


double Board::roughness() const {
    int sum = 0;
    for(size_t x = 1; x < WIDTH; x++) {
        auto const diff = std::abs(_heightMap[x] - _heightMap[x - 1]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
void Board::calcHoles(int x) {
    int holes = 0;
    for(int i = _heightMap[x] + 1; std::cmp_less(i, HEIGHT); ++i)
        if(available({x, i}))
            ++holes;

    _holeMap[x] = holes;
}
void Board::calcHoles() {
    for(int i = 0; std::cmp_less(i, WIDTH); ++i)
        calcHoles(i);
}

void Board::updateMetadata(std::span<vec2 const, BLOCKS> blocks) {
    for(auto const& block : blocks) {
        auto const x = block.x;

        auto& rowHighest = _heightMap[x];
        rowHighest = std::min(block.y, rowHighest);
        _highest = std::min(rowHighest, _highest);

        calcHoles(x);
    }
}
void Board::calcHeightMap() {
    for(size_t x = 0; x < WIDTH; x++) {
        int y = _highest;
        while(std::cmp_less(y, HEIGHT) && available({x, y}))
            y++;
        _heightMap[x] = y;
    }
}
void Board::debugCheck() {
    CTH_STABLE_THROW(_highest != std::ranges::min(_heightMap), "highest must be equal to highest col") { abort(); }

    for(int x = 0; x < WIDTH; x++) {
        vec2 const block{x, _heightMap[x]};
        CTH_STABLE_THROW(inBounds(block) && available(block), "highest in a column must be occupied") { abort(); }
    }
}



}
