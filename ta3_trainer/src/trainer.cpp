#include "ta3/trainer/trainer.hpp"

#include "ta3/trainer/tetris_problem.hpp"

#include "ta3/ai/lib/dlib.hpp"

#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"
#include "boost/serialization/export.hpp"

#include <pagmo/archipelago.hpp>
#include <pagmo/population.hpp>
#include <pagmo/topology.hpp>
#include <pagmo/algorithms/cmaes.hpp>

#include <cth/io/file.hpp>

#include <fstream>

namespace ta3::trainer {

Trainer::Trainer(std::filesystem::path save_file, TrainerConfig const& config) : _saveFile{std::move(save_file)},
    _config{config} { dlib::set_dnn_prefer_fastest_algorithms(); }
void Trainer::init() {
    cth::log::msg<cth::except::INFO>("initializing...");

    if(!std::filesystem::exists(_saveFile) || !load())
        fresh();

    cth::log::msg<cth::except::INFO>("initialized!\n\n");
}


void Trainer::logRunBegin() {
    cth::log::msg<cth::except::INFO>("starting training iteration...");
    cth::log::msg("calculating {} cycles!", _config.cycles);
}
void Trainer::logImprovement(double current, double best) {
    size_t increase = best == 0 ? 0 : static_cast<float>((current - best) / current * 100);
    cth::log::msg<cth::except::INFO>("champion fitness: {}, change: {}%\n", current, increase);
}

void Trainer::logTimeEstimate(std::chrono::steady_clock::time_point start, size_t iteration) const {
    auto const elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
    cth::log::msg(
        "est. time left: {}s\n\n",
        elapsed / (static_cast<float>(iteration) + 1.f) * static_cast<float>(_config.cycles) - elapsed
    );
}
void Trainer::logRunEnd(std::chrono::steady_clock::time_point start) {
    auto const total = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
    cth::log::msg<cth::except::INFO>("training cycles finished, took {}s, average: {}s", total, total / _config.cycles);
}


void Trainer::run() {
    logRunBegin();

    auto const start = std::chrono::steady_clock::now();
    double bestFitness = 0;

    for(size_t i = 0; i < _config.cycles; i++) {
        runCycle(i);

        auto currentBestFitness = -extractBestFitness();
        logImprovement(currentBestFitness, bestFitness);

        bestFitness = std::max(bestFitness, currentBestFitness);


        save();

        logTimeEstimate(start, i);
    }
    logRunEnd(start);
}


void Trainer::setDefaultTopology() const {
    if(!enableTopology())
        return;

    auto defaultTopology = pagmo::topology{};
    defaultTopology.push_back(_arch->size());

    _arch->set_topology(defaultTopology);
}


void Trainer::setRingTopology() const {
    if(!enableTopology())
        return;

    _arch->set_topology(pagmo::topology{topology_t{_config.islands, 1}});
}


void Trainer::fresh() {
    cth::log::msg<cth::except::INFO>("loading fresh training state");


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
    algo.set_bfe(pagmo::bfe{bfe_t{}});

    auto const pop = std::make_unique<pagmo::population>(TetrisProblem{}, bfe_t{}, _config.populationSize);

    _arch = std::make_unique<pagmo::archipelago>(pagmo::topology{topology_t{1}}, _config.islands, algo, *pop);

    cth::log::msg("created archipelago with: {} islands with {} instances!", _config.islands, _config.populationSize);
}


bool Trainer::load() {
    cth::log::msg("loading training state from: {}", _saveFile.string());

    _arch = std::make_unique<pagmo::archipelago>();

    try {
        std::ifstream file{_saveFile, std::ios::binary};
        boost::archive::binary_iarchive archive{file};


        archive >> *_arch;

        file.close();
        setRingTopology();

        cth::log::msg("loaded training state ({}mb)", cth::io::file::size<cth::io::file::MB>(_saveFile));
    }
    catch(std::exception const& e) {
        cth::log::msg<cth::except::WARNING>("failed to load file, error: {}", e.what());
        _arch = nullptr;
    }
    catch(...) {
        cth::log::msg<cth::except::WARNING>("failed to load file, error: UNKNOWN");
        _arch = nullptr;
    }

    if(_arch == nullptr)
        return false;

    bool compatible = true;
    CTH_STABLE_WARN(_arch->size() != _config.islands, "failed to load archive, island size mismatch") {
        details->add("loaded: {}", _arch->size());
        details->add("expected: {}", _config.islands);
        compatible = false;
    }
    auto const populationSize = _arch->begin()->get_population().size();

    CTH_STABLE_WARN(populationSize != _config.populationSize, "failed to load archive, population size mismatch") {
        details->add("loaded: {}", populationSize);
        details->add("expected: {}", _config.populationSize);
        compatible = false;
    }

    if(!compatible)
        _arch = nullptr;
    return compatible;
}


void Trainer::save() const {
    cth::log::msg<cth::except::INFO>("saving to {}...", _saveFile.string());

    setDefaultTopology();

    try {
        if(!std::filesystem::exists(_saveFile))
            std::filesystem::create_directories(std::filesystem::path{_saveFile}.remove_filename());

        std::ofstream archF{_saveFile, std::ios::trunc | std::ios::binary};
        CTH_STABLE_ERR(!archF.is_open(), "failed to open save file: {}", _saveFile.string()) { return; }

        boost::archive::binary_oarchive archive{archF};
        archive << *_arch;
        archF.close();


        cth::log::msg("saved state! ({}mb)\n", cth::io::file::size<cth::io::file::MB>(_saveFile));

        auto champFile = _saveFile;
        champFile.replace_extension("champ");

        std::ofstream champF{champFile, std::ios::trunc | std::ios::binary};
        CTH_STABLE_ERR(!champF.is_open(), "failed to open champion file: {}", champFile.string()) { return; }
        auto championW = _arch->get_champions_x();
        champF.write(
            reinterpret_cast<char const*>(championW.data()),
            sizeof(decltype(championW)::value_type) * championW.size()
        );
        champF.close();
    }
    catch(std::exception const& e) {
        cth::log::msg<cth::except::ERR>("failed to save to file :(, err: ", e.what());
        if(std::filesystem::exists(_saveFile))
            std::filesystem::remove(_saveFile);
    }

    setRingTopology();
}


double Trainer::extractBestFitness() const {
    double bestOverallFitness = std::numeric_limits<double>::max();
    auto const championsF = _arch->get_champions_f();

    for(auto const& i : championsF)
        if(!i.empty() && i[0] < bestOverallFitness)
            bestOverallFitness = i[0];
    return bestOverallFitness;
}


void Trainer::runCycle(size_t i) const {
    cth::log::msg<cth::except::INFO>("starting cycle {}, left: {}\n", i, _config.cycles - 1 - i);

    auto const start = std::chrono::steady_clock::now();

    _arch->evolve(1);

    try { _arch->wait_check(); }
    catch(std::exception const& e) { cth::log::msg<cth::except::ERR>("{}", e.what()); }


    auto const end = std::chrono::steady_clock::now();

    cth::log::msg("completed cycle {}, took {}s\n", i, std::chrono::duration<float>{end - start}.count());
}

}

BOOST_CLASS_EXPORT(ta3::trainer::TetrisProblem)

BOOST_CLASS_EXPORT(pagmo::detail::prob_inner<ta3::trainer::TetrisProblem>)
