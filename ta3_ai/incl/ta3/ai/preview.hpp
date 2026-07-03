#pragma once
#include "ai_tetris.hpp"
#include "model.hpp"

#include "ta3/sim/ui/ai_renderer2.hpp"

#include <chrono>
#include <optional>
#include <thread>
#include <vector>

namespace ta3::ai {

/**
 * live visualization of the model playing: one persistent @ref AiTetris per weight set, stepped on a
 *  worker thread and pushed to an @ref sim::AiRenderer2 as board snapshots. dead games restart in
 *  place, so the watched set stays fixed
 */
class AiPreview {
    static constexpr auto MOVE_DELAY = std::chrono::milliseconds{50};

public:
    AiPreview(size_t games) : _params(games * model_t::params(), 0.0) {}
    AiPreview(std::vector<double> params);

    void run();
    void refresh(std::vector<double> params);
    void stop();

private:
    void startRenderer();
    void stopRenderer();

    std::vector<double> _params;
    std::optional<sim::AiRenderer2> _renderer{};
    // declared last so its destructor stops + joins the worker before _renderer dies under it
    std::optional<std::jthread> _runThread{};

public:
    bool running() const { return _runThread.has_value(); }
    size_t simulations() const { return _params.size() / model_t::params(); }
};
}
