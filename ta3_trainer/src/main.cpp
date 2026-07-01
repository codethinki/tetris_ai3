#include "ta3/trainer/backup.hpp"
#include "ta3/trainer/tetris_problem2.hpp"
#include "ta3/trainer/trainer2.hpp"


#include <cth/numeric.hpp>
#include <cth/io/log.hpp>

#include <omp.h>
#include <Eigen/Core>
#include <pagmo/types.hpp>

#include <iostream>
#include <print>
#include <random>



constexpr size_t ISLANDS = 4;
constexpr pagmo::pop_size_t POPULATION_SIZE = 50;
constexpr size_t CYCLES = 3;
constexpr size_t GENS_PER_CYCLE = 10;

constexpr size_t ITERATIONS = 10000;
constexpr size_t GAME_THREADS = 10;
constexpr size_t MODEL_THREADS = 2;



std::filesystem::path query_model_path() {
    std::println("input a model path (empty -> res/model.bin):");
    std::string path{};
    std::cin >> path;
    if(path.empty() || path == "\n") return {"res/model.bin"};
    while(!std::ranges::all_of(path, [](char c) {
        return cth::num::in_inc(c, 'a', 'z')
            || cth::num::in_inc(c, 'A', 'Z')
            || cth::num::in_inc(c, '0', '9')
            || c == '_'
            || c == '-'
            || c == '.'
            || c == '\\'
            || c == '/';
    })) {
        std::println("invalid filename");
        std::cin >> path;
    }

    return {path};
}



void train() {

#ifndef NDEBUG
    auto const path = std::filesystem::path{std::format("debug_{}.bin", std::random_device{}())};
#else
    auto const path = query_model_path();
#endif
    cth::log::msg<cth::except::INFO>("starting training for {}\n\n", path.string());

    ta3::trn::Trainer2 trainer{
        path,
        {
            ISLANDS,
            POPULATION_SIZE,
            CYCLES,
            GENS_PER_CYCLE,
            MODEL_THREADS,
            GAME_THREADS
        }
    };

    for(size_t i = 0; i < ITERATIONS; i++) {
        trainer.run();
        ta3::trainer::backup(path);
        ta3::trainer::backup(std::filesystem::path{path}.replace_extension("champ"));
    }
}

int main() {
    Eigen::setNbThreads(std::max(1, omp_get_max_threads()));

    try { train(); }
    catch(std::exception const& e) { CTH_STABLE_THROW(true, "exited with exception: {}", e.what()) {} }
    catch(...) { CTH_STABLE_THROW(true, "exited with unknown error") {} }

    return 0;
}
