#pragma once
#include "models/models/model_v7.hpp"

#include <array>
#include <concepts>
#include <span>
#include <vector>


namespace ta3::ai {

/** @brief the interface MultiTetris2 and the trainer require of a value model */
template<class T>
concept model =
    std::default_initializable<T>
    && std::constructible_from<T, std::span<double const>>
    && requires {
        typename T::tetris_stats_t;
        { T::inputs() } -> std::convertible_to<size_t>;
        { T::outputs() } -> std::convertible_to<size_t>;
        { T::bounds() } -> std::convertible_to<glm::dvec2>;
        { T::params() } -> std::convertible_to<size_t>;
    }
    && requires(
    T const m,
    T mut,
    typename T::tetris_stats_t const& stats,
    std::span<data_t> out,
    std::span<data_t const, T::INPUTS> input,
    std::span<data_t const> buffer,
    std::span<double const> weights
) {
        // encode one candidate into a buffer slice
        { T::extractInputs(stats, out) };

        // reuse one instance across evaluations
        { mut.loadWeights(weights) };

        // score a single input / a packed buffer of inputs
        { m.forward(input) } -> std::same_as<std::array<data_t, T::OUTPUTS>>;
        { m.batchForward(buffer) } -> std::same_as<std::vector<data_t>>;
    };


using model_t = ModelV7;
static_assert(model<model_t>, "the selected model version must satisfy the model concept");

namespace m {
    constexpr auto INPUTS = model_t::inputs();
    constexpr auto OUTPUTS = model_t::outputs();
    constexpr auto BOUNDS = model_t::bounds();
}

using tetris_stats_t = model_t::tetris_stats_t;
}
