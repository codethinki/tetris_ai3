#pragma once

#include "ta3/sim/tetris_engine.hpp"
#include "ta3/sim/tetris_defs.hpp"
#include "ta3/sim/pieces/piece_defs.hpp"

#include <cth/macro.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <set>
#include <stop_token>
#include <thread>
#include <vector>

namespace ta3::sim {
class Tetris;

class Renderer {
public:
    static constexpr int FPS = 60;
    static constexpr vec2 GAME_GRID_EXTENT{6, 2};
    static constexpr int DISPLAYS = GAME_GRID_EXTENT.x * GAME_GRID_EXTENT.y;
    static constexpr size_t CACHE_SIZE = DISPLAYS * BOARD_SIZE;

    static constexpr int BORDER_DIV = 20;
    static constexpr int SEPARATOR_DIV = 30;
    static constexpr int LINE_WIDTH = 1;
    static constexpr vec2 INIT_SCREEN_EXTENT{645, 450};

    using BoardGrid = std::vector<std::vector<BlockType>>;

    explicit Renderer();


    ~Renderer() {
        _renderThread.request_stop();
        _renderThread.join();
    }

    void hook(TetrisEngine const& tetris);

    void unhook(TetrisEngine const& tetris);

    void resetStats();

private:
    void toggleFullscreen();

    using boards_view_t = cth::mta::mdspan_t<BlockType, DISPLAYS, HEIGHT, WIDTH>;

    bool _fullscreen = false;
    vec2 _screenExtents{INIT_SCREEN_EXTENT};

    int _cellSizePx{};
    vec2 _border{};
    vec2 _separator{};

    vec2 _boardExtents;



    std::array<TetrisEngine const*, DISPLAYS> _displays;
    std::vector<BlockType> _boardCache;
    boards_view_t _boards;
    std::atomic<size_t> _registrations = 0;
    std::atomic<size_t> _finishedGames = 0;

    std::set<TetrisEngine const*> _waitingQueue;

    std::mutex _displaySlotLock;
    std::jthread _renderThread;

    void setSizes();

    void draw(size_t index, vec2 origin) const;

    void refreshBoards();
    void copyBoards();
    void drawBoards() const;
    void drawStats(std::chrono::system_clock::time_point start) const;
    void renderLoopInternal(std::stop_token const& stop);
};

}

namespace ta3::sim {
inline std::unique_ptr<Renderer> renderer{};
}
