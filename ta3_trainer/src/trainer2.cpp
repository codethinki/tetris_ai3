#include "ta3/trainer/trainer2.hpp"


#include "ta3/trainer/backup.hpp"
#include "ta3/trainer/tetris_problem2.hpp"

#include <pagmo/bfe.hpp>
#include <pagmo/population.hpp>
#include <pagmo/algorithms/cmaes.hpp>
#include <pagmo/batch_evaluators/member_bfe.hpp>
#include <pagmo/topologies/ring.hpp>

#include <dlib/dnn.h>

#include <cth/macro.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <zstd.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
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
    _gameScheduler{_config.parallelGameInstances} { init(); }

void Trainer2::initDlib() { dlib::set_dnn_prefer_fastest_algorithms(); }

void Trainer2::initState() {
    if(!loadState())
        newState();
}
void Trainer2::init() {
    cth::log::msg<cth::except::INFO>("initializing...");
    startSchedulers();
    initDlib();

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
        static_cast<unsigned>(_config.gensPerCycle),
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
    algo.set_verbosity(1);
    algo.set_bfe(pagmo::bfe{pagmo::member_bfe{}});

    buildArchipelago(pagmo::algorithm{std::move(algo)});
}
void Trainer2::buildArchipelago(pagmo::algorithm const& algo) {
    auto const pop = std::make_unique<pagmo::population>(
        TetrisProblem2{
            cth::co::executor{_gameScheduler},
            cth::co::executor{_modelScheduler},
            _modelPool,
            _config.gameSeed
        },
        pagmo::member_bfe{},
        _config.populationSize,
        _config.populationSeed
    );

    _arch = std::make_unique<pagmo::archipelago>(pagmo::topology{pagmo::ring{1}}, _config.islands, algo, *pop);

    cth::log::msg("created archipelago with: {} islands with {} instances!", _config.islands, _config.populationSize);
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
        file.write(compressed.data(), static_cast<std::streamsize>(compressedSize));
    }

    // champion weights, stored separately as raw doubles for loading a model later
    auto championFile = _saveFile;
    championFile.replace_extension("champion");
    {
        auto const& champion = championsX[bestIsland];
        auto const count = champion.size();

        std::ofstream file{championFile, std::ios::trunc | std::ios::binary};
        file.write(reinterpret_cast<char const*>(&count), sizeof(count));
        file.write(reinterpret_cast<char const*>(champion.data()), static_cast<std::streamsize>(count * sizeof(double)));
    }

    trainer::backup(_saveFile);

    cth::log::msg("saved state (island {}, fitness {})\n", bestIsland, bestFitness);
}
void Trainer2::logRunBegin() {
    cth::log::msg<cth::except::INFO>("starting training iteration...");
    cth::log::msg("calculating {} cycles!", _config.cycles);
}

void Trainer2::logImprovement(double current, double best) {
    size_t increase = best == 0 ? 0 : static_cast<float>((current - best) / current * 100);
    cth::log::msg<cth::except::INFO>("champion fitness: {}, change: {}%\n", current, increase);
}

void Trainer2::logTimeEstimate(std::chrono::steady_clock::time_point start, size_t iteration) const {
    auto const elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
    cth::log::msg(
        "est. time left: {}s\n\n",
        elapsed / (static_cast<float>(iteration) + 1.f) * static_cast<float>(_config.cycles) - elapsed
    );
}
void Trainer2::logRunEnd(std::chrono::steady_clock::time_point start) const {
    auto const total = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
    cth::log::msg<cth::except::INFO>("training cycles finished, took {}s, average: {}s", total, total / _config.cycles);
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
}
void Trainer2::runCycle(size_t i) const {
    cth::log::msg<cth::except::INFO>("starting cycle {}, left: {}\n", i, _config.cycles - 1 - i);

    auto const start = std::chrono::steady_clock::now();

    _arch->evolve(1);

    try { _arch->wait_check(); }
    catch(std::exception const& e) { cth::log::msg<cth::except::ERR>("{}", e.what()); }


    auto const end = std::chrono::steady_clock::now();

    cth::log::msg("completed cycle {}, took {}s\n", i, std::chrono::duration<float>{end - start}.count());
}
void Trainer2::run() {
    logRunBegin();

    auto const start = std::chrono::steady_clock::now();
    double bestFitness = 0;

    startSchedulers();
    for(size_t i = 0; i < _config.cycles; i++) {
        runCycle(i);

        auto currentBestFitness = -extractBestFitness();
        logImprovement(currentBestFitness, bestFitness);

        bestFitness = std::max(bestFitness, currentBestFitness);


        saveState();

        logTimeEstimate(start, i);
    }
    stopSchedulers();

    logRunEnd(start);
}
}
