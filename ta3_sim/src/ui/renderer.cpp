#include "ta3/sim/ui/renderer.hpp"

#include "raylib.h"

#include "ta3/sim/tetris.hpp"


#include <mutex>
#include <thread>

Color to_color(ta3::sim::BlockType type) {
    switch(type) {
        case ta3::sim::BlockType::EMPTY: return BLANK;
        case ta3::sim::BlockType::I: return SKYBLUE;
        case ta3::sim::BlockType::O: return YELLOW;
        case ta3::sim::BlockType::T: return PURPLE;
        case ta3::sim::BlockType::S: return GREEN;
        case ta3::sim::BlockType::Z: return RED;
        case ta3::sim::BlockType::J: return BLUE;
        case ta3::sim::BlockType::L: return ORANGE;
        default: return BLACK;
    }
}

namespace ta3::sim {

Renderer::Renderer() : _boardCache{BOARD_SIZE * DISPLAYS}, _boards{_boardCache.data()} {
    setSizes();

    _displays.fill(nullptr);


    _renderThread = std::jthread{[this](std::stop_token const& stop) { renderLoopInternal(stop); }};
}
void Renderer::hook(Tetris const& tetris) {
    std::lock_guard _{_displaySlotLock};

    auto const ptr = &tetris;

    if(_waitingQueue.contains(ptr) || std::ranges::contains(_displays, ptr)) return;

    _waitingQueue.insert(ptr);

    ++_registrations;
}
void Renderer::unhook(Tetris const& tetris) {
    std::lock_guard _{_displaySlotLock};

    auto const ptr = &tetris;
    if(_waitingQueue.contains(ptr)) { _waitingQueue.erase(ptr); } else if(std::ranges::contains(_displays, ptr))
        std::ranges::replace(_displays, ptr,
            nullptr);

    ++_finishedGames;
}
void Renderer::resetStats() {
    _registrations = 0;
    _finishedGames = 0;
}
void Renderer::toggleFullscreen() {
    auto const monitor = GetCurrentMonitor();

    glm::ivec2 const screen{GetMonitorWidth(monitor), GetMonitorHeight(monitor)};

    auto const size = _fullscreen ? screen / 2 : screen;
    auto const pos = _fullscreen ? screen / 4 : glm::ivec2{0, 0};

    auto const funcPtr = _fullscreen ? &ClearWindowState : &SetWindowState;
    funcPtr(FLAG_WINDOW_UNDECORATED);

    SetWindowPosition(pos.x, pos.y);
    SetWindowSize(size.x, size.y);

    _fullscreen = !_fullscreen;

    _screenExtents = size;
    setSizes();
}
void Renderer::setSizes() {
    _border = _screenExtents / BORDER_DIV;

    _separator = _screenExtents / SEPARATOR_DIV;

    auto const height = static_cast<int>((_screenExtents.y - _border.y * 2.f - (GAME_GRID_EXTENT.y - 1.f) * _separator.y) /
        (GAME_GRID_EXTENT.y * static_cast<float>(HEIGHT)));
    auto const width = static_cast<int>((_screenExtents.x - _border.x * 2.f - (GAME_GRID_EXTENT.x - 1.f) * _separator.x)
        / (GAME_GRID_EXTENT.x * static_cast<float>(WIDTH)));


    _cellSizePx = std::min(height, width);
    _boardExtents = vec2{WIDTH, HEIGHT} * (_cellSizePx + LINE_WIDTH) + LINE_WIDTH;

}


void Renderer::draw(size_t index, vec2 origin) const {

    DrawRectangle(origin.x, origin.y, _boardExtents.x, _boardExtents.y, ColorAlpha(WHITE, 0.1f));


    for(int y = 0; y < HEIGHT; ++y) {
        for(int x = 0; x < WIDTH; ++x) {
            auto const block = _boards[index, y, x];
            auto const color = ColorAlpha(to_color(block), 0.6f);

            auto const coords = vec2{x, y} * (_cellSizePx + LINE_WIDTH) + origin + LINE_WIDTH;


            if(block != BlockType::EMPTY)
                DrawRectangle(
                    coords.x,
                    coords.y,
                    _cellSizePx,
                    _cellSizePx,
                    color
                );
        }
    }

    for(int y = 0; std::cmp_less_equal(y, HEIGHT); y++) {
        int const yPos = origin.y + y * (_cellSizePx + LINE_WIDTH);
        DrawLine(origin.x, yPos, origin.x + _boardExtents.x, yPos, DARKGRAY);
    }
    for(int x = 0; std::cmp_less_equal(x, WIDTH); x++) {
        int const xPos = origin.x + x * (_cellSizePx + LINE_WIDTH);
        DrawLine(xPos + LINE_WIDTH, origin.y, xPos + LINE_WIDTH, origin.y + _boardExtents.y, DARKGRAY);
    }

}

void Renderer::refreshBoards() {
    static std::mt19937 rnd{std::random_device{}()};
    std::lock_guard _{_displaySlotLock};


    auto unassigned = std::views::filter(_displays, [](auto& ptr) { return ptr == nullptr; });

    for(auto& ptr : unassigned) {
        if(_waitingQueue.empty()) return;

        std::uniform_int_distribution<size_t> dist{0, _waitingQueue.size() - 1};

        if(ptr != nullptr) return;

        auto it = _waitingQueue.begin();

        std::advance(it, dist(rnd));

        ptr = *it;
        _waitingQueue.erase(it);
    }
}

void Renderer::copyBoards() {
    std::lock_guard lock{_displaySlotLock};
    for(size_t i = 0; i < DISPLAYS; ++i) {
        auto const tetris = _displays[i];
        if(tetris == nullptr) continue;

        auto boardView = tetris->board().flat();
        std::ranges::copy(boardView, _boardCache.begin() + static_cast<ptrdiff_t>(i * BOARD_SIZE));
    }
}

void Renderer::drawBoards() const {


    ClearBackground(BLACK);

    for(int i = 0; i < DISPLAYS; ++i) {
        vec2 const pos{i % GAME_GRID_EXTENT.x, i / GAME_GRID_EXTENT.x};

        vec2 const origin{pos.x * (_boardExtents.x + _separator.x) + _border.x, pos.y * (_boardExtents.y + _separator.y) + _border.y};

        draw(i, origin);
    }


}
void Renderer::drawStats(std::chrono::system_clock::time_point start) const {
    static constexpr int FONT_SIZE = 20;
    vec2 const offset{10, _screenExtents.y - 20};

    auto now = std::chrono::system_clock::now();

    auto const elapsedTime = std::chrono::duration<float>(now - start).count();

    auto const registrationsStr = std::format("registrations: {}, {}/s",
        _registrations.load(), _registrations == 0 ? 0.f : static_cast<float>(_registrations) / elapsedTime);
    auto const finishedGamesStr = std::format("finished Games {}, {}/s",
        _finishedGames.load(), _finishedGames == 0 ? 0.f : static_cast<float>(_finishedGames) / elapsedTime);

    DrawText(registrationsStr.c_str(), offset.x, offset.y, FONT_SIZE, LIME);
    DrawText(finishedGamesStr.c_str(), offset.x, offset.y - 20, FONT_SIZE, LIME);
}
void Renderer::renderLoopInternal(std::stop_token const& stop) {
    auto start = std::chrono::system_clock::now();

    SetTraceLogLevel(TraceLogLevel::LOG_ERROR);
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    InitWindow(_screenExtents.x, _screenExtents.y, "Tetris AI Visualizer");
    SetTargetFPS(FPS);

    SetExitKey(KEY_NULL);
    while(!stop.stop_requested() && !WindowShouldClose()) {
        if(IsWindowMinimized()) {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
            continue;
        }
        if(IsWindowResized()) {
            _screenExtents.x = GetScreenWidth();
            _screenExtents.y = GetScreenHeight();
            setSizes();
        } else if(!_fullscreen && IsKeyPressed(KEY_F11) || _fullscreen && IsKeyPressed(KEY_ESCAPE))
            toggleFullscreen();

        refreshBoards();
        copyBoards();
        BeginDrawing();

        drawBoards();
        drawStats(start);
        DrawFPS(10, 10);

        EndDrawing();
    }
    CloseWindow();
}
}
