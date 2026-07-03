#pragma once

#include "ta3/ai/model_defs.hpp"
#include "../metrics/tetris_stats_v4.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <array>
#include <algorithm>
#include <memory>
#include <span>
#include <vector>
#include <stdexcept>

namespace ta3::ai {

constexpr size_t V6_INPUTS = 16;

// a tiny fully-connected stack that just weights the stat vector: 16 -> 12 -> 8 -> 4 -> 1
constexpr size_t V6_FC1_SIZE = 12, V6_FC2_SIZE = 8, V6_FC3_SIZE = 4, V6_OUT_SIZE = 1;

/** @brief a minimal stats-only value model -- three fc layers over the v5 feature vector */
class ModelV6 {
    enum feature : size_t {
        NEXT_PIECE_0,
        NEXT_PIECE_1,
        NEXT_PIECE_2,
        NEXT_PIECE_3,
        NEXT_PIECE_4,
        NEXT_PIECE_5,
        NEXT_PIECE_6,
        HELD,
        HOLES,
        NEW_HOLES,
        SURFACE_VAR,
        LINES_CLEARED,
        AGG_HEIGHT,
        MAX_HEIGHT,
        BUMPINESS,
        PIECES,
        FEATURE_COUNT,
    };
    static_assert(FEATURE_COUNT == V6_INPUTS, "feature layout must match the net input width");

public:
    static constexpr size_t INPUTS = V6_INPUTS;
    static constexpr size_t OUTPUTS = V6_OUT_SIZE;
    static constexpr glm::dvec2 BOUNDS{-2, 2};

    // Calculate parameter count at compile time for the flat vector array
    static constexpr size_t NUM_PARAMS = (V6_INPUTS * V6_FC1_SIZE + V6_FC1_SIZE) +
        (V6_FC1_SIZE * V6_FC2_SIZE + V6_FC2_SIZE) +
        (V6_FC2_SIZE * V6_FC3_SIZE + V6_FC3_SIZE) +
        (V6_FC3_SIZE * V6_OUT_SIZE + V6_OUT_SIZE);

    static constexpr size_t inputs() { return INPUTS; }
    static constexpr size_t outputs() { return OUTPUTS; }
    static constexpr glm::dvec2 bounds() { return BOUNDS; }

    using tetris_stats_t = stats_v4;

    /**
     * @brief encodes one candidate: the next piece, the held piece, then the board stats
     * @param[out] out exactly @ref INPUTS values to overwrite
     */
    static constexpr void extractInputs(tetris_stats_t const& stats, std::span<data_t> out) {
        auto const queue = stats.get(metric::piece_queue);
        if(!queue.empty())
            out[NEXT_PIECE_0 + *queue.front()] = 1;

        out[HELD] = static_cast<data_t>(*stats.get(metric::held_piece));
        out[HOLES] = static_cast<data_t>(stats.get(metric::holes));
        out[NEW_HOLES] = static_cast<data_t>(stats.get(metric::new_holes));
        out[SURFACE_VAR] = static_cast<data_t>(stats.get(metric::surface_variance));
        out[LINES_CLEARED] = static_cast<data_t>(stats.get(metric::lines_cleared));
        out[AGG_HEIGHT] = static_cast<data_t>(stats.get(metric::agg_height));
        out[MAX_HEIGHT] = static_cast<data_t>(stats.get(metric::max_height));
        out[BUMPINESS] = static_cast<data_t>(stats.get(metric::bumpiness));
        out[PIECES] = static_cast<data_t>(stats.get(metric::pieces_placed));
    }

    constexpr ModelV6() { init(); }

    constexpr ModelV6(std::span<double const> weights) { loadWeights(weights); }


    /** @brief overwrites the net weights in place (no re-allocation), for reuse across evaluations */
    constexpr void loadWeights(std::span<double const> weights);

    [[nodiscard]] constexpr std::array<ai::data_t, OUTPUTS> forward(std::span<ai::data_t const, INPUTS> input) const;

    /** @brief scores a packed buffer of @ref INPUTS-wide parsed inputs; the count is its size / @ref INPUTS */
    [[nodiscard]] constexpr std::vector<ai::data_t> batchForward(std::span<ai::data_t const> inputs) const;

    [[nodiscard]] constexpr size_t size() const { return NUM_PARAMS; }

    [[nodiscard]] static constexpr size_t params() { return NUM_PARAMS; }

private:
    constexpr void init();

    // Flat, L1-cache friendly array replacing the dlib unique_ptr
    std::array<ai::data_t, NUM_PARAMS> _weights{};
};

} // namespace ta3::ai



namespace ta3::ai {



constexpr void ModelV6::init() { _weights.fill(0); }

constexpr void ModelV6::loadWeights(std::span<double const> weights) {
    CTH_CRITICAL(weights.size() != NUM_PARAMS, "weights do not match size") {}

    for(size_t i = 0; i < NUM_PARAMS; ++i)
        _weights[i] = static_cast<ai::data_t>(weights[i]);
}

constexpr std::array<ai::data_t, ModelV6::OUTPUTS> ModelV6::forward(
    std::span<ai::data_t const, INPUTS> input
) const {
    // A small inline helper lambda to do dense layer math cleanly
    auto dense_layer = [&](
        std::span<ai::data_t const> in,
        std::span<ai::data_t> out,
        size_t& offset,
        bool relu
    ) constexpr {
        size_t inDim = in.size();
        size_t outDim = out.size();

        for(size_t o = 0; o < outDim; ++o) {
            // Bias is stored right after the weights for this layer
            ai::data_t val = _weights[offset + (inDim * outDim) + o];

            for(size_t i = 0; i < inDim; ++i) { val += in[i] * _weights[offset + (o * inDim) + i]; }
            out[o] = relu ? std::max<ai::data_t>(0, val) : val;
        }
        offset += (inDim * outDim) + outDim; // Advance the pointer for the next layer
    };

    std::array<ai::data_t, V6_FC1_SIZE> out1{};
    std::array<ai::data_t, V6_FC2_SIZE> out2{};
    std::array<ai::data_t, V6_FC3_SIZE> out3{};
    std::array<ai::data_t, OUTPUTS> out4{};

    size_t wOffset = 0;

    // Evaluate Network: 16 -> 12 (ReLU) -> 8 (ReLU) -> 4 (ReLU) -> 1 (Linear)
    dense_layer(input, out1, wOffset, true);
    dense_layer(out1, out2, wOffset, true);
    dense_layer(out2, out3, wOffset, true);
    dense_layer(out3, out4, wOffset, false);

    return out4;
}

constexpr std::vector<ai::data_t> ModelV6::batchForward(
    std::span<ai::data_t const> inputs
) const {
    CTH_CRITICAL(inputs.size() % INPUTS != 0, "batch forward inputs not a multiple of INPUTS") {}

    size_t const count = inputs.size() / INPUTS;
    if(count == 0)
        return {};

    // Note: C++20 supports constexpr std::vector
    std::vector<ai::data_t> result(count * OUTPUTS);

    for(size_t i = 0; i < count; ++i) {
        std::span<ai::data_t const, INPUTS> singleIn(&inputs[i * INPUTS], INPUTS);
        auto singleOut = forward(singleIn);
        std::copy_n(singleOut.begin(), OUTPUTS, result.begin() + (i * OUTPUTS));
    }

    return result;
}

} // namespace ta3::ai
