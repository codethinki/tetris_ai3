#pragma once
#include "models/models/model_v11.hpp"

#include <ta3/sim/board2.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <span>


namespace ta3::ai {

/** @brief the interface the trainer and CPU preview require of a value model */
template<class T>
concept model =
    std::default_initializable<T>
    && std::constructible_from<T, std::span<double const>>
    && requires {
        { T::inputs() } -> std::convertible_to<size_t>;
        { T::outputs() } -> std::convertible_to<size_t>;
        { T::bounds() } -> std::convertible_to<ta3::sim::dvec2>;
        { T::params() } -> std::convertible_to<size_t>;
    }
    && requires(
    T const m,
    T mut,
    std::array<data_t, T::INPUTS> const& input,
    std::span<double const> weights,
    sim::Board2 const& board
) {
        // reuse one instance across evaluations
        { mut.loadWeights(weights) };

        // score a single input vector
        { m.forward(input) } -> std::same_as<std::array<data_t, T::OUTPUTS>>;

        // the search seam: value = model(path clear histogram, leaf board, hold-is-I flag) -- CPU search
        // + the GPU net_ref
        { m.evaluate(clear_hist_t{}, board, bool{}) } -> std::convertible_to<data_t>;
    };


using model_t = ModelV11;
using stats_t = stats_v4;

static_assert(model<model_t>, "the selected model version must satisfy the model concept");

namespace m {
    constexpr auto INPUTS = model_t::inputs();
    constexpr auto OUTPUTS = model_t::outputs();
    constexpr auto BOUNDS = model_t::bounds();
}
}
