#include "ta3/sim/ui/ai_renderer2.hpp"

#include "raylib.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>
#include <print>

namespace ta3::sim {

AiRenderer2::AiRenderer2(std::string title) : _title{std::move(title)} {
    _renderThread = std::jthread{[this](std::stop_token stop) { renderLoop(std::move(stop)); }};
}

AiRenderer2::~AiRenderer2() = default; // jthread stops + joins the render loop

void AiRenderer2::update(std::span<Board2 const> boards) {
    std::scoped_lock lock{_mutex};
    _snapshot.assign(boards.begin(), boards.end());
}

void AiRenderer2::stop() { _renderThread.request_stop(); }

AiRenderer2::Layout AiRenderer2::computeLayout(vec2 screen, size_t n) {
    Layout layout{};
    if(n == 0)
        return layout;

    int const availX = screen.x - 2 * PADDING;
    int const availY = screen.y - 2 * PADDING;

    // a board of cell size c occupies WIDTH*c + (WIDTH+1)*LINE px (cells plus enclosing grid lines)
    constexpr int LINES_X = (static_cast<int>(WIDTH) + 1) * LINE_WIDTH;
    constexpr int LINES_Y = (static_cast<int>(VISIBLE_HEIGHT) + 1) * LINE_WIDTH;

    int bestCols = 1, bestCell = 0;
    for(int cols = 1; cols <= static_cast<int>(n); ++cols) {
        int const rows = (static_cast<int>(n) + cols - 1) / cols; // ceil(n / cols)

        int const tileX = (availX - (cols - 1) * GAP) / cols;
        int const tileY = (availY - (rows - 1) * GAP) / rows;
        if(tileX <= LINES_X || tileY <= LINES_Y)
            continue;

        int const cell = std::min(
            (tileX - LINES_X) / static_cast<int>(WIDTH),
            (tileY - LINES_Y) / static_cast<int>(VISIBLE_HEIGHT)
        );
        if(cell > bestCell) {
            bestCell = cell;
            bestCols = cols;
        }
    }

    layout.cols = bestCols;
    layout.rows = (static_cast<int>(n) + bestCols - 1) / bestCols;
    layout.cellPx = bestCell;
    layout.boardPx = vec2{
        static_cast<int>(WIDTH) * bestCell + LINES_X,
        static_cast<int>(VISIBLE_HEIGHT) * bestCell + LINES_Y
    };

    // center the whole grid block in the window (padding is a floor, never exceeded)
    vec2 const gridPx{
        layout.cols * layout.boardPx.x + (layout.cols - 1) * GAP,
        layout.rows * layout.boardPx.y + (layout.rows - 1) * GAP
    };
    layout.origin0 = vec2{(screen.x - gridPx.x) / 2, (screen.y - gridPx.y) / 2};

    return layout;
}

void AiRenderer2::drawBoard(Board2 const& board, vec2 origin, int cell_px) const {
    int const step = cell_px + LINE_WIDTH;
    vec2 const boardPx{
        static_cast<int>(WIDTH) * step + LINE_WIDTH,
        static_cast<int>(VISIBLE_HEIGHT) * step + LINE_WIDTH
    };

    DrawRectangle(origin.x, origin.y, boardPx.x, boardPx.y, ColorAlpha(WHITE, 0.1f));

    // occupancy-only board -> one colour; bit 0 is the top row, hidden buffer rows cropped off the top
    for(size_t y = 0; y < VISIBLE_HEIGHT; ++y)
        for(size_t x = 0; x < WIDTH; ++x) {
            if(board.available(vec2{static_cast<int>(x), static_cast<int>(y + TOP_CROP)}))
                continue; // empty cell

            int const px = origin.x + static_cast<int>(x) * step + LINE_WIDTH;
            int const py = origin.y + static_cast<int>(y) * step + LINE_WIDTH;
            DrawRectangle(px, py, cell_px, cell_px, ColorAlpha(SKYBLUE, 0.8f));
        }

    for(size_t y = 0; y <= VISIBLE_HEIGHT; ++y) {
        int const yPos = origin.y + static_cast<int>(y) * step;
        DrawLine(origin.x, yPos, origin.x + boardPx.x, yPos, DARKGRAY);
    }
    for(size_t x = 0; x <= WIDTH; ++x) {
        int const xPos = origin.x + static_cast<int>(x) * step;
        DrawLine(xPos, origin.y, xPos, origin.y + boardPx.y, DARKGRAY);
    }
}

void AiRenderer2::renderLoop(std::stop_token stop) {
    SetTraceLogLevel(LOG_ERROR);
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    SetExitKey(KEY_NULL);
    InitWindow(_screenExtents.x, _screenExtents.y, _title.c_str());
    SetTargetFPS(FPS);

    std::vector<Board2> frame;

    while(!stop.stop_requested() && !WindowShouldClose()) {
        PollInputEvents();
        if(IsWindowResized())
            continue;

        if(!IsWindowMinimized()) {
            {
                std::scoped_lock lock{_mutex};
                frame = _snapshot;
            }

            BeginDrawing();
            ClearBackground(BLACK);

            if(!frame.empty()) {
                auto const layout = computeLayout({GetScreenWidth(), GetScreenHeight()}, frame.size());
                if(layout.cellPx > 0)
                    for(size_t i = 0; i < frame.size(); ++i) {
                        vec2 const origin{
                            layout.origin0.x + static_cast<int>(i % layout.cols) * (layout.boardPx.x + GAP),
                            layout.origin0.y + static_cast<int>(i / layout.cols) * (layout.boardPx.y + GAP)
                        };
                        drawBoard(frame[i], origin, layout.cellPx);
                    }
            }
            EndDrawing();
            SwapScreenBuffer();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{16});
    }

    CloseWindow();
}

}
