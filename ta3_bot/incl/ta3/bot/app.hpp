#pragma once

#include "ta3/bot/jstris.hpp"

#include <cth/coro/executor.hpp>
#include <cth/coro/scheduler.hpp>
#include <cth/coro/tasks/executor_task.hpp>
#include <cth/io/keybd/keys.hpp>
#include <cth/os/fence.hpp>
#include <ta3/ai/model.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

/**
 * @file app.hpp
 * @brief the bot application shell -- owns the execution context every other stage runs on.
 */
namespace ta3::bot {

/**
 * static configuration of @ref BotApp
 */
struct app_config_t {
    /** scheduler worker count; one worker may search while another handles input */
    std::size_t workers = 2;

    cth::io::Key startKey = cth::io::Key::F8;
    cth::io::Key stopKey = cth::io::Key::F9;
    cth::io::Key exitKey = cth::io::Key::ESCAPE;
    cth::io::Key fasterKey = cth::io::Key::COMMA;
    cth::io::Key slowerKey = cth::io::Key::PERIOD;

    /** delay between two actions; the only timing changed at runtime, see @ref jstris::move_timing */
    std::chrono::milliseconds inputDelay{80};
    /** delay from the last hard drop to the next piece's first action, in percent of @ref inputDelay */
    std::uint32_t initialMoveDelayPercent = 200;
    std::chrono::milliseconds speedStep{5};
    std::chrono::milliseconds minInputDelay{20};
    std::chrono::milliseconds maxInputDelay{500};

    /** every value must be usable as is; @ref inputDelay is clamped to the delay range instead */
    void validate() const;
};

/**
 * owns the bot's execution context and drives its pipeline.
 */
class BotApp {
public:
    using config_t = app_config_t;

    explicit BotApp(ai::model_t model, config_t = {});
    ~BotApp();

    /**
     * runs the bot until its root task completes, blocks the calling thread
     * @return process exit code
     */
    int run();

private:
    /**
     * root coroutine -- every stage is spawned from here
     */
    [[nodiscard]] auto mainTask() -> cth::co::executor_task<int>;

    [[nodiscard]] auto inputTask() -> cth::co::executor_task<void>;
    [[nodiscard]] auto playTask() -> cth::co::executor_task<void>;

    /** suspends while paused; false once shutdown was requested */
    [[nodiscard]] auto waitUntilPlaying() -> cth::co::executor_task<bool>;

    /** searches the next placement; nullopt and stops playing when the game is over or no move is legal */
    [[nodiscard]] std::optional<std::vector<sim::Instruction>> planMoves(sim::TetrisState const& state);

    void stop();
    void startPlaying();
    void stopPlaying();
    void changeInputDelay(std::chrono::milliseconds delta);
    void handleInput(cth::io::Key key);

    config_t _config;
    std::stop_source _shutdown;
    std::atomic_bool _playing = false;
    cth::os::fence _playingFence;
    std::atomic<std::chrono::milliseconds::rep> _inputDelayMs;
    ai::model_t _model;
    Jstris _jstris;
    cth::co::scheduler _scheduler;

public:
    [[nodiscard]] cth::co::executor executor() const noexcept { return cth::co::executor{_scheduler}; }

    [[nodiscard]] config_t const& config() const noexcept { return _config; }

    BotApp(BotApp const&) = delete;
    BotApp& operator=(BotApp const&) = delete;
    BotApp(BotApp&&) = delete;
    BotApp& operator=(BotApp&&) = delete;
};

}
