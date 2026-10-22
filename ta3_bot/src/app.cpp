#include "ta3/bot/app.hpp"
#include "ta3/bot/jstris.hpp"
#include "ta3/bot/moves.hpp"

#include <cth/coro/sync.hpp>
#include <cth/io/log.hpp>
#include <cth/win/coro/keyboard.hpp>
#include <ta3/ai/search/beam.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <utility>

namespace ta3::bot {
namespace {
    [[nodiscard]] sim::PieceType placed_piece(
        sim::TetrisState const& state,
        sim::drop_place_t placement
    ) {
        if(!placement.hold)
            return state.current;
        return state.held == sim::NO_PIECE ? state.lookahead.front() : state.held;
    }

    [[nodiscard]] app_config_t validated(app_config_t config) {
        config.validate();
        return config;
    }

}

void app_config_t::validate() const {
    CTH_CRITICAL(workers == 0, "worker count must be positive") {}
    CTH_CRITICAL(speedStep <= std::chrono::milliseconds{0}, "speed step must be positive") {}
    CTH_CRITICAL(minInputDelay < std::chrono::milliseconds{0}, "minimum input delay cannot be negative") {}
    CTH_CRITICAL(maxInputDelay < minInputDelay, "invalid input delay range") {}
    // the runtime delay stays within [minInputDelay, maxInputDelay], so the slowest timing covers all
    jstris::move_timing(maxInputDelay, initialMoveDelayPercent).validate();
}

// _config is declared before _scheduler, so config.workers is live by the time the scheduler reads it.
BotApp::BotApp(ai::model_t model, config_t config) :
    _config{validated(config)},
    _inputDelayMs{std::clamp(_config.inputDelay, _config.minInputDelay, _config.maxInputDelay).count()},
    _model{std::move(model)},
    _jstris{},
    _scheduler{cth::co::autostart, _config.workers} {}

BotApp::~BotApp() {
    stop();
    _scheduler.await_stop();
}

int BotApp::run() { return cth::co::sync(executor().spawn(mainTask())); }

auto BotApp::mainTask() -> cth::co::executor_task<int> {
    auto play = executor().spawn(playTask());
    play.handle().resume();

    co_await inputTask();

    stop();
    co_await std::move(play);
    co_return 0;
}

auto BotApp::inputTask() -> cth::co::executor_task<void> {
    cth::win::co::keybd_event_queue keyboard;
    std::array<bool, cth::io::keys_size()> pressed{};

    cth::log::msg<cth::except::INFO>(
        "ta3_bot: {} start, {} stop, {} exit, {}/{} speed ({} ms)",
        _config.startKey,
        _config.stopKey,
        _config.exitKey,
        _config.fasterKey,
        _config.slowerKey,
        _inputDelayMs.load()
    );

    while(!_shutdown.stop_requested()) {
        auto const keyState = (co_await keyboard.next()).data;
        auto& wasDown = pressed[static_cast<std::size_t>(keyState.exKey.key)];
        auto const newPress = keyState.down && !wasDown;
        wasDown = keyState.down;

        if(newPress)
            handleInput(keyState.exKey.key);
    }
}

auto BotApp::playTask() -> cth::co::executor_task<void> {
    try {
        if(!(co_await _jstris.connect()))
            co_return;
    } catch(std::exception const& e) {
        cth::log::msg<cth::except::ERR>("ta3_bot: Jstris bridge failed: {}", e.what());
        co_return;
    }

    while(co_await waitUntilPlaying()) {
        try {
            auto const state = co_await _jstris.next_board();
            if(!state)
                break;

            auto moves = planMoves(*state);
            if(!moves || !_playing.load())
                continue;

            // the script paces the moves; its reply to the next board request queues behind them
            auto const timing = jstris::move_timing(
                std::chrono::milliseconds{_inputDelayMs.load()},
                _config.initialMoveDelayPercent
            );
            if(!(co_await _jstris.move(std::move(*moves), timing)))
                cth::log::msg<cth::except::WARNING>("ta3_bot: Jstris bridge disconnected before move");
        } catch(std::exception const& e) {
            cth::log::msg<cth::except::ERR>("ta3_bot: Jstris bridge failed: {}", e.what());
        }
    }
}

auto BotApp::waitUntilPlaying() -> cth::co::executor_task<bool> {
    while(!_shutdown.stop_requested() && !_playing.load())
        co_await _playingFence;
    co_return !_shutdown.stop_requested();
}

auto BotApp::planMoves(sim::TetrisState const& state) -> std::optional<std::vector<sim::Instruction>> {
    if(state.gameOver) {
        cth::log::msg<cth::except::INFO>("ta3_bot: Jstris game finished");
        stopPlaying();
        return std::nullopt;
    }

    auto const result = ai::search::search_move_beam(state, _model);
    if(result.none()) {
        cth::log::msg<cth::except::WARNING>("ta3_bot: no legal Jstris move");
        stopPlaying();
        return std::nullopt;
    }

    return jstris::moves_for(placed_piece(state, result.move), result.move);
}

void BotApp::stop() {
    _shutdown.request_stop();
    _playing.store(false);
    _playingFence.signal();
    _jstris.stop();
}

void BotApp::startPlaying() {
    if(_shutdown.stop_requested())
        return;
    _playing.store(true);
    _playingFence.signal();
    cth::log::msg<cth::except::INFO>("ta3_bot: started");
}

void BotApp::stopPlaying() {
    _playing.store(false);
    _playingFence.reset();
    cth::log::msg<cth::except::INFO>("ta3_bot: stopped");
}

void BotApp::changeInputDelay(std::chrono::milliseconds delta) {
    auto const delay = std::clamp(
        std::chrono::milliseconds{_inputDelayMs.load()} + delta,
        _config.minInputDelay,
        _config.maxInputDelay
    );

    _inputDelayMs.store(delay.count());
    cth::log::msg<cth::except::INFO>("ta3_bot: input delay {} ms", delay.count());
}

void BotApp::handleInput(cth::io::Key key) {
    if(key == _config.startKey)
        startPlaying();
    else if(key == _config.stopKey)
        stopPlaying();
    else if(key == _config.exitKey)
        stop();
    else if(key == _config.fasterKey)
        changeInputDelay(-_config.speedStep);
    else if(key == _config.slowerKey)
        changeInputDelay(_config.speedStep);
}

}
