#include "ta3/trainer/tetris_problem2.hpp"
#include "ta3/trainer/trainer2.hpp"


#include <cth/numeric.hpp>
#include <cth/io/log.hpp>

#include <pagmo/types.hpp>

#include <iostream>
#include <limits>
#include <print>
#include <random>
#include <stop_token>
#include <string>
#include <thread>



constexpr size_t ISLANDS = 4;
constexpr pagmo::pop_size_t POPULATION_SIZE = 50;
constexpr size_t GENS_PER_ITERATION = 10;
constexpr size_t ITERATIONS_PER_CYCLE = 10;


// double buffering
constexpr size_t MODEL_THREADS = 2;

constexpr size_t GAMES_PER_EVAL = 30;
constexpr size_t MAX_MOVES = 300;

constexpr size_t MAX_BACKUPS = 5;

constexpr auto ALGO = ta3::trn::TrainerAlgo2::CMAES;



std::filesystem::path query_model_path() {
    auto randomDefault = std::format("out/model_{}.bin", std::random_device{}());

    std::println("input a model path (empty -> {}):", randomDefault);
    std::string path{};
    std::cin >> path;
    if(path.empty() || path == "\n")
        return randomDefault;
    while(!std::ranges::all_of(
        path,
        [](char c) {
            return cth::num::in_inc(c, 'a', 'z')
                || cth::num::in_inc(c, 'A', 'Z')
                || cth::num::in_inc(c, '0', '9')
                || c == '_'
                || c == '-'
                || c == '.'
                || c == '\\'
                || c == '/';
        }
    )) {
        std::println("invalid filename");
        std::cin >> path;
    }

    return {path};
}



void train() {
#ifndef NDEBUG
    auto const path = std::filesystem::path{std::format("out/debug_{}.bin", std::random_device{}())};
#else
    auto const path = query_model_path();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // drop the rest of the path line
#endif
    if(auto const parent = path.parent_path(); !parent.empty() && !std::filesystem::is_directory(parent))
        std::filesystem::create_directories(parent);

    cth::log::msg<cth::except::INFO>("starting training for {}\n\n", path.string());

    ta3::trn::Trainer2 trainer{
        path,
        {
            ISLANDS,
            POPULATION_SIZE,
            GENS_PER_ITERATION,
            MODEL_THREADS,
            GAMES_PER_EVAL,
            MAX_MOVES,
            ITERATIONS_PER_CYCLE,
            MAX_BACKUPS,
            ALGO
        }
    };

    // train on a worker thread; run() checkpoints every iteration and stops cleanly on the token
    std::jthread worker{
        [&trainer](std::stop_token stop) {
            try { trainer.run(stop); }
            catch(std::exception const& e) { cth::log::msg<cth::except::ERR>("training stopped: {}", e.what()); }
        }
    };

    std::println("training -- press enter to stop after the current iteration");
    std::string line;
    std::getline(std::cin, line);

    worker.request_stop();
    // scope exit joins the worker: waits for the current iteration to finish + checkpoint
}

int main() {
#ifndef NDEBUG
    train();
#else
    try { train(); }
    catch(std::exception const& e) {cth::log::msg<cth::except::Severity::ERR>("{}", e.what()); }
    catch(...) { CTH_STABLE_THROW(true, "exited with unknown error") {} }
#endif
    return 0;
}
