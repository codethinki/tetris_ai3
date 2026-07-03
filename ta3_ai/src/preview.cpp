#include "ta3/ai/preview.hpp"
#include "ta3/sim/ui/ai_renderer2.hpp"

#include <ta3/sim/board2.hpp>

#include <random>
#include <span>
#include <vector>

namespace ta3::ai {

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
            size_t const sims = simulations();
            if(sims == 0)
                return;

            constexpr auto paramCount = model_t::params();


            std::random_device rd{};

            // one model + one persistent game per simulation
            std::vector<model_t> models(sims);
            std::vector<AiTetris> games;
            games.reserve(sims);
            for(size_t i = 0; i < sims; ++i) {
                models[i].loadWeights(std::span{_params}.subspan(i * paramCount, paramCount));
                games.emplace_back(rd());
            }

            std::vector<sim::Board2> boards(sims);

            while(!t.stop_requested()) {
                for(size_t i = 0; i < sims; ++i) {
                    if(games[i].gameOver())
                        games[i].reset(rd()); // restart dead games in place
                    games[i].step(models[i]);
                    boards[i] = games[i].board();
                }

                _renderer->update(boards);
                std::this_thread::sleep_for(MOVE_DELAY);
            }
        }
    };
}
void AiPreview::stopRenderer() {
    if(_runThread)
        _runThread->request_stop();
}
}
