#include "ta3/trainer/tetris_problem.hpp"

#include "ta3/trainer/lib/pagmo.hpp"

#include <ta3/ai/multi_tetris.hpp>
#include <ta3/sim/tetris.hpp>

#include <execution>
#include <span>
#include <utility>

namespace ta3::trainer {


namespace {
    template<class Rng>
    auto max_index(Rng const& rng) { return std::ranges::distance(rng.begin(), std::ranges::max_element(rng)); }
}

pagmo::bounds_vec2 TetrisProblem::get_bounds() const {
    return std::pair{
        pvecd(_dimension, BOUNDS[0]),
        pvecd(_dimension, BOUNDS[1])
    };
}
pvecd TetrisProblem::fitness(pvecd const& dv) const { return {evalFitness(std::span{dv.data(), dv.size()})}; }

pvecd TetrisProblem::batch_fitness(pvecd const& dvs) const {
    std::vector instanceDvs{
        std::from_range,
        dvs | std::views::chunk(_dimension) | std::views::transform(
            [](auto const& rng) { return std::span<double const>{rng.data(), std::ranges::size(rng)}; }
        )
    };
    auto const instances = instanceDvs.size();
    pvecd out(instances);


    if constexpr(cth::debug_mode())
        for(size_t i = 0; i < instances; i++)
            out[i] = evalFitness(instanceDvs[i]);
    else {
        size_t const batches = instances / SIMULATION_BATCHES + (instances % SIMULATION_BATCHES != 0);

        for(int i = 0; i < batches; i++) {
            ptrdiff_t const offset = i * SIMULATION_BATCHES;
            auto const batchSize = std::min(instances, offset + SIMULATION_BATCHES) - offset;
            std::span batchDvs{&instanceDvs[offset], batchSize};


            std::transform(
                std::execution::par_unseq,
                batchDvs.begin(),
                batchDvs.end(),
                out.begin() + offset,
                [](auto& dv) { return evalFitness(dv); }
            );
        }
    }


    return out;
}



double TetrisProblem::evalFitness(std::span<double const> dv) {
    auto const model = std::make_unique<ai::model_t>(dv);
    return evalFitness(*model);
}
double TetrisProblem::evalFitness(ai::model_t const& model) { return -simulateGames(model); }


double TetrisProblem::simulateGame(ai::model_t const& model) {
    auto const game = std::make_unique<sim::Tetris>();

    ai::MultiTetris tetris{*game};

    while(tetris.genVariations()) {
        auto out = model.batchForward(tetris.inputs());

        tetris.update(max_index(out));
    }

    return tetris.adjustedScore();
}
//TEMP optimize with few big allocs instead of multiple
double TetrisProblem::simulateGames(ai::model_t const& model) {
    static std::atomic<bool> reachedMax = false;
    double score = 0;

    std::vector<ai::MultiTetrisGame> tetrisGames{SIMULATIONS_PER_EVAL};
    for(auto& [_, multi] : tetrisGames)
        multi.genVariations();

    std::vector<ai::input_t> inputs{};

    size_t move = 0;

    std::vector<ai::data_t> inputsBuffer{};
    for(; move < MAX_PIECES && !tetrisGames.empty(); move++) {
        inputs.clear();

        for(auto& [_, multi] : tetrisGames)
            inputs.insert_range(inputs.end(), multi.inputs());

        auto const chunkView = inputs | std::views::chunk(GAME_BATCH_SIZE);

        inputsBuffer.reserve(inputs.size());
        for(auto const chunk : chunkView) {
            auto const result = model.batchForward(std::span{chunk.data(), chunk.size()});
            inputsBuffer.insert_range(inputsBuffer.end(), result);
        }
        size_t index = 0;
        for(auto& [_, multi] : tetrisGames) {
            auto const size = multi.variants() * ai::model_t::OUTPUTS;
            multi.update(max_index(std::span{&inputsBuffer[index], size}));
            index += size;
        }

        inputsBuffer.clear();

        for(ptrdiff_t i = 0; std::cmp_less(i, tetrisGames.size()); i++) {
            auto& [_, multi] = tetrisGames[i];
            if(multi.genVariations())
                continue;

            score += multi.adjustedScore();
            tetrisGames.erase(tetrisGames.begin() + i);
            --i;
        }
    }

    for(auto& [_, multi] : tetrisGames)
        score += multi.adjustedScore();

    if(!reachedMax && move >= MAX_PIECES) {
        cth::log::msg<cth::except::INFO>("reached max pieces, limiting moves :) ({})", MAX_PIECES);
        reachedMax = true;
    }

    return score / static_cast<double>(SIMULATIONS_PER_EVAL);
}

}
