#pragma once
#include "ta3/sim/board2.hpp"
#include "ta3/sim/utility/tetris_defs.hpp"

#include <cstddef>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace ta3::sim {

/**
 * watches a set of boards and renders them on its own window + thread
 * @details push model: the owner calls @ref update with a fresh snapshot whenever the games advance.
 *  there is no subscription and no per-game identity -- the renderer only ever knows @ref Board2, so
 *  it copies what it is handed and draws the latest. the grid is laid out dynamically to fill the
 *  window (largest shared cell size that fits every board, with padding)
 * @note raylib owns a single OS window, so at most one instance should exist at a time
 */
class AiRenderer2 {
public:
    static constexpr int FPS = 60;

    /** rows of the @c HEIGHT-tall board that are visible; the rest is hidden spawn buffer at the top */
    static constexpr size_t VISIBLE_HEIGHT = 20;
    static constexpr size_t TOP_CROP = HEIGHT - VISIBLE_HEIGHT;

    static constexpr int PADDING = 16; // outer window padding, px
    static constexpr int GAP = 8; // gap between boards, px
    static constexpr int LINE_WIDTH = 1; // grid line thickness, px

    static constexpr vec2 INIT_SCREEN_EXTENT{645, 450};

    explicit AiRenderer2(std::string title = "Tetris AI");
    ~AiRenderer2();

    AiRenderer2(AiRenderer2 const&) = delete;
    AiRenderer2& operator=(AiRenderer2 const&) = delete;

    /**
     * publishes a fresh snapshot for the render thread to display
     * @details copies @p boards (cheap: @c WIDTH bit-columns per board); call once per AI step. the
     *  number of boards may change between calls -- the grid re-lays out to match
     */
    void update(std::span<Board2 const> boards);

    /** requests the render thread to stop; also done by the destructor */
    void stop();

private:
    /** the dynamic grid: the largest shared cell size fitting @p n boards into @p screen with padding */
    struct Layout {
        int cols = 1;
        int rows = 1;
        int cellPx = 0;
        vec2 boardPx{}; // full board extent incl. grid lines
        vec2 origin0{}; // top-left of the (centered) grid block
    };
    [[nodiscard]] static Layout computeLayout(vec2 screen, size_t n);

    void drawBoard(Board2 const& board, vec2 origin, int cell_px) const;
    void renderLoop(std::stop_token stop);

    std::string _title;

    // guarded snapshot: update() writes it, the render thread copies it out under the same lock
    std::mutex _mutex;
    std::vector<Board2> _snapshot;

    // touched only by the render thread
    vec2 _screenExtents{INIT_SCREEN_EXTENT};

    // declared last: its destructor stops + joins the worker before the members above die under it
    std::jthread _renderThread;
};

}
