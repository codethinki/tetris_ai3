#include "ta3/ai/preview.hpp"

#include "boost/property_map/property_map.hpp"

#include "ta3/sim/ui/ai_renderer2.hpp"

#include <execution>
#include <print>

#include <ta3/sim/board2.hpp>

#include <random>
#include <ranges>
#include <span>
#include <vector>

#include <cth/chrono.hpp>

namespace ta3::ai {
namespace stdchr = std::chrono;

AiPreview::AiPreview(std::vector<double> params) : _params{std::move(params)} {}

void AiPreview::run() {
    _renderer.emplace("Tetris AI");
    startRenderer();
}

void AiPreview::refresh(std::vector<double> params) {
    stopRenderer();
    _params = std::move(params);
    startRenderer();
}

void AiPreview::stop() {
    stopRenderer();
    _renderer = std::nullopt;
}
void AiPreview::startRenderer() {
    _runThread = std::jthread{
        [this](std::stop_token t) {
            auto const sims = simulations();
            if(sims == 0)
                return;

            constexpr auto paramCount = model_t::params();


            std::random_device rd{};
            std::mutex rdMtx{};

            // one model + one persistent game per simulation
            std::vector<model_t> models(sims);
            std::vector<AiTetris> games{};
            games.reserve(sims);
            for(size_t i = 0; i < sims; ++i) {
                models[i].loadWeights(std::span{_params}.subspan(i * paramCount, paramCount));
                games.emplace_back(rd());
            }

            std::vector<sim::Board2> boards(sims);

            while(!t.stop_requested()) {
                auto const frameStart = stdchr::high_resolution_clock::now();

                auto zip = std::views::zip(games, boards, models);

                std::for_each(
                    std::execution::par_unseq,
                    zip.begin(),
                    zip.end(),
                    [&rd, &rdMtx](auto const& tuple) {
                        auto& [game, board, model] = tuple;
                        if(game.gameOver()) {
                            std::scoped_lock _{rdMtx};
                            game.reset(rd());
                        }
                        game.step(model);
                        board = game.board();
                    }
                );


                _renderer->update(boards);
                auto const frameEnd = stdchr::high_resolution_clock::now();
                auto const frameTime = stdchr::duration_cast<stdchr::milliseconds>(frameEnd - frameStart);

                if(auto const sleep = MOVE_DELAY - frameTime; sleep > std::chrono::milliseconds{0})
                    std::this_thread::sleep_for(sleep);
            }
        }
    };
}
void AiPreview::stopRenderer() {
    if(_runThread)
        _runThread->request_stop();
}
}
