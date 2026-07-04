#pragma once

#include "ta3/ai/preview.hpp"
#include "ta3/trainer/tetris_problem2.hpp"

#include <ta3/sim/utility.hpp>
#include <cth/coro/scheduler.hpp>
#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>

#include <filesystem>
#include <memory>
#include <stop_token>

namespace ta3::trn {
namespace stdfs = std::filesystem;

struct TrainerConfig2 {
    size_t islands;
    size_t populationSize;
    size_t gensPerIteration;
    size_t parallelModelInstances;
    size_t parallelGameInstances;
    size_t simulationsPerEval;
    size_t maxMoves;

    size_t iterationsPerCycle;
    size_t maxBackups;

    uint64_t trainingSeed = 0x3e28df7b1811145b;
};

class Trainer2 {
public:
    using Config = TrainerConfig2;
    Trainer2(stdfs::path save_file, Config const& config);
    ~Trainer2();

    void run(std::stop_token const& stop = {});

private:
    void initState();
    void init();

    void newState();
    bool loadState();
    void saveState();

    /** builds @ref _arch with fresh problems around @ref algo, one copy per island */
    void buildArchipelago(pagmo::algorithm const& algo);
    void buildPreview();

    void reloadPreviewGames();

    void logRunBegin();
    static void logImprovement(double current, double previous);
    void logTimeEstimate(std::chrono::steady_clock::time_point start, size_t iteration) const;
    void logRunEnd(std::chrono::steady_clock::time_point start, size_t iterations) const;

    [[nodiscard]] double extractBestFitness() const;

    void startSchedulers();
    void stopSchedulers();

    void nextGameSeed() const;
    void runIteration(size_t i) const;

    /** backs up the state, then runs @ref TrainerConfig2::iterationsPerCycle iterations (which only save) */
    void runCycle(
        std::stop_token const& stop,
        size_t& iteration
    );


    stdfs::path _saveFile;
    Config _config;
    mutable sim::Xoshiro256ss _generator;

    cth::co::scheduler _modelScheduler;
    cth::co::scheduler _gameScheduler;

    std::shared_ptr<model_pool_t> _modelPool;

    std::unique_ptr<pagmo::archipelago> _arch = nullptr;
    std::optional<ai::AiPreview> _preview{};
    mutable uint64_t _iterationSeed = 0;

    [[nodiscard]] bool running() const;
};
}
