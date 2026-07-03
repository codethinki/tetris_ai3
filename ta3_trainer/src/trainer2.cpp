#include "ta3/trainer/trainer2.hpp"


#include "ta3/trainer/backup.hpp"
#include "ta3/trainer/tetris_problem2.hpp"

#include <zstd.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <cth/macro.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/topologies/ring.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace ta3::trn {
namespace {
    // high level -- the cmaes covariance is large but only saved once per cycle
    constexpr int COMPRESSION_LEVEL = 19;
}

Trainer2::Trainer2(stdfs::path save_file, Config const& config) : _saveFile{std::move(save_file)},
    _config{config},
    _modelScheduler{_config.parallelModelInstances},
    _gameScheduler{_config.parallelGameInstances},
    _generator{_config.gameSeed} { init(); }
Trainer2::~Trainer2() = default;


void Trainer2::initState() {
    if(!loadState())
        newState();
    buildPreview();

}
void Trainer2::init() {
    cth::log::msg<cth::except::INFO>("initializing...");
    startSchedulers();

    _modelPool = std::make_shared<model_pool_t>();
    for(auto i = 0uz; i < _config.parallelModelInstances; ++i)
        _modelPool->emplace();

    initState();

    stopSchedulers();
    cth::log::msg<cth::except::INFO>("initialized!\n\n");
}
void Trainer2::newState() {
    cth::log::msg<cth::except::INFO>("creating new training state");

    auto algo = pagmo::cmaes{
        static_cast<unsigned>(_config.gensPerIteration),
        -1,
        -1,
        -1,
        -1,
        1.3,
        1e-12,
        1e-12,
        true,
        false
    };
    algo.set_verbosity(0);
    algo.set_bfe(pagmo::bfe{pagmo::member_bfe{}});

    buildArchipelago(pagmo::algorithm{std::move(algo)});
}
void Trainer2::buildArchipelago(pagmo::algorithm const& algo) {
    _arch = std::make_unique<pagmo::archipelago>(
        pagmo::topology{pagmo::ring{1}},
        _config.islands,
        // island args
        algo,
        TetrisProblem2{
            {
                _gameScheduler,
                _modelScheduler,
                _modelPool,
                _config.simulationsPerEval,
                _config.maxMoves,
                &_iterationSeed
            }
        },
        pagmo::member_bfe{},
        _config.populationSize
    );


    cth::log::msg("created archipelago with: {} islands with {} instances!", _config.islands, _config.populationSize);
}
void Trainer2::buildPreview() { _preview.emplace(_arch->size()); }
void Trainer2::reloadPreviewGames() {
    _preview->refresh({std::from_range, _arch->get_champions_x() | std::views::join});
}


bool Trainer2::loadState() {
    if(!std::filesystem::exists(_saveFile))
        return false;

    cth::log::msg<cth::except::INFO>("loading state from {}", _saveFile.string());

    std::string compressed{};
    {
        std::ifstream file{_saveFile, std::ios::binary};
        compressed.assign(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
    }

    auto const rawSize = ZSTD_getFrameContentSize(compressed.data(), compressed.size());
    CTH_CRITICAL(
        rawSize == ZSTD_CONTENTSIZE_ERROR || rawSize == ZSTD_CONTENTSIZE_UNKNOWN,
        "save file is not a valid zstd frame"
    ) {}

    std::string raw(static_cast<size_t>(rawSize), '\0');
    auto const result = ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size());
    CTH_CRITICAL(ZSTD_isError(result), "zstd decompression failed: {}", ZSTD_getErrorName(result)) {}

    // the saved cmaes carries the full adaptation state; memory=true resumes the search from it
    pagmo::algorithm algo{};
    {
        std::istringstream stream{raw, std::ios::binary};
        boost::archive::binary_iarchive archive{stream};
        archive >> algo;
    }

    buildArchipelago(algo);

    cth::log::msg("loaded state, seeded {} islands from the saved cmaes\n", _config.islands);
    return true;
}
void Trainer2::saveState() {
    auto const championsF = _arch->get_champions_f();
    auto const championsX = _arch->get_champions_x();

    // the best island: lowest fitness, since pagmo minimizes
    auto bestIsland = 0uz;
    auto bestFitness = std::numeric_limits<double>::max();
    for(auto i = 0uz; i < championsF.size(); ++i)
        if(!championsF[i].empty() && championsF[i][0] < bestFitness) {
            bestFitness = championsF[i][0];
            bestIsland = i;
        }

    // serialize just the best island's algorithm (its cmaes adaptation state), then compress
    std::ostringstream raw{std::ios::binary};
    {
        auto const algo = (*_arch)[bestIsland].get_algorithm();
        boost::archive::binary_oarchive archive{raw};
        archive << algo;
    }
    auto const rawStr = raw.str();

    std::vector<char> compressed(ZSTD_compressBound(rawStr.size()));
    auto const compressedSize = ZSTD_compress(
        compressed.data(),
        compressed.size(),
        rawStr.data(),
        rawStr.size(),
        COMPRESSION_LEVEL
    );
    CTH_CRITICAL(ZSTD_isError(compressedSize), "zstd compression failed: {}", ZSTD_getErrorName(compressedSize)) {}

    {
        std::ofstream file{_saveFile, std::ios::trunc | std::ios::binary};
        CTH_STABLE_THROW(!file.is_open(), "failed to open file") {}
        file.write(compressed.data(), static_cast<std::streamsize>(compressedSize));
    }

    // champion weights, stored separately as raw doubles for loading a model later
    auto championFile = _saveFile;
    championFile.replace_extension("champ");
    {
        auto const& champion = championsX[bestIsland];
        auto const count = champion.size();

        std::ofstream file{championFile, std::ios::trunc | std::ios::binary};
        CTH_STABLE_THROW(!file.is_open(), "failed to open file") {}
        file.write(reinterpret_cast<char const*>(&count), sizeof(count));
        file.write(
            reinterpret_cast<char const*>(champion.data()),
            static_cast<std::streamsize>(count * sizeof(double))
        );
    }

    cth::log::msg("saved state (island {}, fitness {})", bestIsland, bestFitness);
}
void Trainer2::logRunBegin() { cth::log::msg<cth::except::INFO>("starting training, stop to end...\n"); }

void Trainer2::logImprovement(double current, double best) {
    double increase = current == 0 ? 0 : static_cast<float>((current - best) / current * 100.);
    std::string changeStr{};
    if(increase > 1e-6)
        changeStr = std::format(", change: {:.2f}%", increase);

    cth::log::msg<cth::except::INFO>("iteration champion fitness: {}{}", current, changeStr);
}

void Trainer2::logTimeEstimate(std::chrono::steady_clock::time_point start, size_t iteration) const {
    auto const elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
    cth::log::msg("elapsed: {}s, avg/iteration: {}s", elapsed, elapsed / (static_cast<float>(iteration) + 1.f));
    std::println("\n\n");
}
void Trainer2::logRunEnd(std::chrono::steady_clock::time_point start, size_t iterations) const {
    auto const total = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
    cth::log::msg<cth::except::INFO>(
        "training stopped after {} iterations, took {}s, average: {}s",
        iterations,
        total,
        iterations == 0 ? 0.f : total / static_cast<float>(iterations)
    );
}

double Trainer2::extractBestFitness() const {
    double bestOverallFitness = std::numeric_limits<double>::max();
    auto const championsF = _arch->get_champions_f();

    for(auto const& i : championsF)
        if(!i.empty() && i[0] < bestOverallFitness)
            bestOverallFitness = i[0];
    return bestOverallFitness;
}

void Trainer2::startSchedulers() {
    _modelScheduler.start();
    _gameScheduler.start();
}
void Trainer2::stopSchedulers() {
    _modelScheduler.request_stop();
    _gameScheduler.request_stop();
    _modelScheduler.await_stop();
    _gameScheduler.await_stop();

    _modelPool->clear();
}
void Trainer2::nextGameSeed() const { _iterationSeed = _generator(); }
void Trainer2::runIteration(size_t i) const {
    cth::log::msg<cth::except::INFO>("starting iteration {}", i);

    auto const start = std::chrono::steady_clock::now();

    nextGameSeed();
    _arch->evolve(1);

    try { _arch->wait_check(); }
    catch(std::exception const& e) { cth::log::msg<cth::except::ERR>("{}", e.what()); }


    auto const end = std::chrono::steady_clock::now();

    cth::log::msg("completed iteration {}, took {}s", i, std::chrono::duration<float>{end - start}.count());
}
void Trainer2::runCycle(
    std::stop_token const& stop,
    size_t& iteration,
    double& best_fitness
) {
    cth::log::msg<cth::except::INFO>("running cycle with {} iterations", _config.iterationsPerCycle);

    auto const cycleStart = cth::chrono::clock_t::now();

    for(size_t it = 0; it < _config.iterationsPerCycle && !stop.stop_requested(); ++it, ++iteration) {
        reloadPreviewGames();

        runIteration(iteration);

        auto const currentBestFitness = -extractBestFitness();
        logImprovement(currentBestFitness, best_fitness);
        best_fitness = std::max(best_fitness, currentBestFitness);

        saveState();

        logTimeEstimate(cycleStart, iteration);
    }

    CTH_STABLE_THROW(!stdfs::exists(_saveFile), "save file must exist for backup") {}

    trainer::backup(_saveFile, _config.maxBackups);

    auto championFile = _saveFile;
    championFile.replace_extension("champ");
    trainer::backup(championFile, _config.maxBackups);

    auto const cycleEnd = cth::chrono::clock_t::now();
    cth::log::msg<cth::except::INFO>("ended cycle in {}s", std::chrono::duration<float>{cycleEnd - cycleStart}.count());
    std::println("\n\n");
}
bool Trainer2::running() const {
    return _modelScheduler.active() || _gameScheduler.active() || _preview && _preview->running();
}
void Trainer2::run(std::stop_token const& stop) {
    logRunBegin();

    auto const start = std::chrono::steady_clock::now();

    startSchedulers();
    _preview->run();

    size_t iteration = 0;
    double bestFitness = 0;
    while(!stop.stop_requested())
        runCycle(stop, iteration, bestFitness);

    _preview->stop();
    stopSchedulers();

    logRunEnd(start, iteration);
}
}
