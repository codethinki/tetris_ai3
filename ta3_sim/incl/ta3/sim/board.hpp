#pragma once
#include "ta3/sim/tetris_defs.hpp"
#include "ta3/sim/glm_defs.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace ta3::sim {

class Board {
public:
    static constexpr BlockType EMPTY = BlockType::EMPTY;
    static constexpr BlockType BLOCKED = BlockType::OCCUPIED;
    static_assert(EMPTY < BLOCKED);

    static constexpr bool occupied(BlockType block) { return block > EMPTY; }
    static constexpr bool available(BlockType block) { return block == EMPTY; }

    [[nodiscard]] static constexpr bool inBounds(vec2 block) {
        return cth::num::in(block.x, 0, WIDTH) && cth::num::in(block.y, 0, HEIGHT);
    }

    constexpr Board() {
        _grid.fill(EMPTY);
        _heightMap.fill(HEIGHT);
    }

    [[nodiscard]] constexpr auto view() const { return cboard_view_t{_grid.data()}; }
    [[nodiscard]] constexpr auto view() { return board_view_t{_grid.data()}; }

    [[nodiscard]] constexpr auto operator[](vec2 const& coord) const { return view()[coord.y, coord.x]; }
    [[nodiscard]] constexpr auto& operator[](vec2 const& coord) { return view()[coord.y, coord.x]; }
    [[nodiscard]] constexpr auto operator[](int y, int x) const { return view()[y, x]; }
    [[nodiscard]] constexpr auto& operator[](int y, int x) { return view()[y, x]; }

    [[nodiscard]] constexpr auto line(int y) const { return std::span{&view()[y, 0], WIDTH}; }
    [[nodiscard]] constexpr auto line(int y) { return std::span{&view()[y, 0], WIDTH}; }

    [[nodiscard]] constexpr auto data() const { return _grid.data(); }

    [[nodiscard]] constexpr auto flat() const { return std::span{_grid.data(), _grid.size()}; }
    [[nodiscard]] constexpr auto flat() { return std::span{_grid.data(), _grid.size()}; }

    [[nodiscard]] constexpr auto const& grid() const { return _grid; }

    [[nodiscard]] constexpr auto highest() const { return _highest; }
    [[nodiscard]] constexpr auto heightMapView() const { return std::span<int const>{_heightMap}; }

    [[nodiscard]] constexpr bool occupied(this auto const& self, int y, int x) { return Board::occupied(self[y, x]); }

    void place(this Board& self, PieceType type, Orientation orientation, vec2 offset);

    [[nodiscard]] bool available(PieceType piece, Orientation orientation, vec2 offset) const;
    [[nodiscard]] bool available(this Board const& self, vec2 block) { return available(self[block]); }
    [[nodiscard]] bool available(this Board const& self, std::span<vec2 const, BLOCKS> blocks) {
        return std::ranges::all_of(blocks, [&self](auto const& block) { return self.available(block); });
    }
    [[nodiscard]] std::optional<vec2> placeOffset(PieceType type, Orientation orientation, vec2 offset) const;

    [[nodiscard]] size_t clearLines();
    [[nodiscard]] std::vector<int> fullLines() const;

    void normalize();

    [[nodiscard]] double roughness() const;

    [[nodiscard]] int holes(int x) const { return _holeMap[x]; }
    [[nodiscard]] int holes() const { return std::ranges::fold_left(_holeMap, 0, std::plus{}); }

private:
    void calcHoles(int x);
    void calcHoles();

    void updateMetadata(std::span<vec2 const, BLOCKS> blocks);
    void calcHeightMap();

    void debugCheck();

    int _highest = HEIGHT;
    std::array<int, WIDTH> _heightMap{};
    std::array<int, WIDTH> _holeMap{};
    grid_t _grid{};
};

}
