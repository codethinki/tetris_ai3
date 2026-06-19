#pragma once
#include "ta3/trainer/lib/pagmo.hpp"

#include <ta3/ai/model.hpp>
#include <ta3/sim/glm_defs.hpp>

#include <chrono>
#include <span>

namespace ta3::trainer {

class TetrisProblem {
    static constexpr size_t SIMULATION_THREADS = 5;
    static constexpr size_t SIMULATION_BATCHES = 5;
    static constexpr size_t GAME_BATCH_SIZE = 1'000'000;

    static constexpr size_t SIMULATIONS_PER_EVAL = 200;
    static constexpr size_t MAX_PIECES = 1000;


    static constexpr std::chrono::milliseconds PRINT_INTERVAL{16};
    static constexpr glm::dvec2 BOUNDS{-4, 4};

    static size_t _dimension;

public:
    explicit TetrisProblem() = default;
    ~TetrisProblem() = default;

    [[nodiscard]] pagmo::bounds_vec2 get_bounds() const;

    [[nodiscard]] pvecd fitness(pvecd const& dv) const;

    [[nodiscard]] pvecd batch_fitness(pvecd const& dvs) const;

private:
    using clock_t = std::chrono::steady_clock;
    using time_p = std::chrono::time_point<clock_t>;


    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, unsigned int) {}

    [[nodiscard]] static double evalFitness(std::span<double const> dv);
    [[nodiscard]] static double evalFitness(ai::model_t const& model);

    static double simulateGame(ai::model_t const& model);

    static double simulateGames(ai::model_t const& model);

public:
    [[nodiscard]] pagmo::thread_safety get_thread_safety() const { return pagmo::thread_safety::constant; };
    [[nodiscard]] bool has_fitness() const { return true; }
    [[nodiscard]] bool has_batch_fitness() const { return true; }

    TetrisProblem(TetrisProblem const& other) = default;
    TetrisProblem(TetrisProblem&& other) noexcept = default;
    TetrisProblem& operator=(TetrisProblem const& other) = default;
    TetrisProblem& operator=(TetrisProblem&& other) noexcept = default;

};
inline size_t TetrisProblem::_dimension = ai::model_t{}.size();
}
