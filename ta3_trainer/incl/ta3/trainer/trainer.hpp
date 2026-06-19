#pragma once
#include "ta3/trainer/lib/pagmo.hpp"

#include "pagmo/archipelago.hpp"
#include "pagmo/batch_evaluators/member_bfe.hpp"
#include "pagmo/topologies/ring.hpp"

#include <filesystem>
#include <memory>

#include <cth/macro.hpp>

namespace ta3::trainer {

struct TrainerConfig {
    size_t islands;
    size_t populationSize;
    size_t cycles;
    size_t gensPerCycle;
};

class Trainer {
    using topology_t = pagmo::ring;
    using bfe_t = pagmo::member_bfe;
    static constexpr size_t MIN_TOPOLOGY_ISLANDS = 3;
    static constexpr size_t LOGS_PER_CYCLE = 3;

public:
    explicit Trainer(std::filesystem::path save_file, TrainerConfig const& config);
    void init();

    void logRunBegin();
    static void logImprovement(double current, double best);
    void logTimeEstimate(std::chrono::steady_clock::time_point start, size_t iteration) const;
    void logRunEnd(std::chrono::steady_clock::time_point start);
    void run();

private:
    [[nodiscard]] bool enableTopology() const { return _arch->size() >= MIN_TOPOLOGY_ISLANDS; }
    void setDefaultTopology() const;
    void setRingTopology() const;

    void fresh();
    bool load();
    void save() const;

    [[nodiscard]] double extractBestFitness() const;

    void runCycle(size_t i) const;

    std::filesystem::path _saveFile;
    TrainerConfig _config;
    std::unique_ptr<pagmo::archipelago> _arch = nullptr;

};
}
