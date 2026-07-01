#pragma once
#include "ta3/trainer/tetris_problem2.hpp"

#include <cth/coro/scheduler.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>

#include <filesystem>
#include <memory>

namespace ta3::trn {
namespace stdfs = std::filesystem;

struct TrainerConfig2 {
    size_t islands;
    size_t populationSize;
    size_t cycles;
    size_t gensPerCycle;
    size_t parallelModelInstances;
    size_t parallelGameInstances;
    uint64_t gameSeed = 0x1010101010101010;
    unsigned populationSeed = std::random_device{}();
};

class Trainer2 {
public:
    using Config = TrainerConfig2;
    Trainer2(stdfs::path save_file, Config const& config);

    void run();

private:
    static void initDlib();
    void initState();
    void init();

    void newState();
    bool loadState();
    void saveState();

    /** builds @ref _arch with fresh problems around @ref algo, one copy per island */
    void buildArchipelago(pagmo::algorithm const& algo);

    void logRunBegin();
    static void logImprovement(double current, double best);
    void logTimeEstimate(std::chrono::steady_clock::time_point start, size_t iteration) const;
    void logRunEnd(std::chrono::steady_clock::time_point start) const;

    [[nodiscard]] double extractBestFitness() const;

    void startSchedulers();
    void stopSchedulers();

    void runCycle(size_t i) const;


    stdfs::path _saveFile;
    Config _config;

    cth::co::scheduler _modelScheduler;
    cth::co::scheduler _gameScheduler;

    std::shared_ptr<model_pool_t> _modelPool;

    std::unique_ptr<pagmo::archipelago> _arch = nullptr;
};
}
