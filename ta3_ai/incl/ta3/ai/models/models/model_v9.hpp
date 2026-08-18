#pragma once

#include "ta3/ai/model_defs.hpp"
#include "ta3/ai/models/metrics/tetris_stats_v4.hpp"
#include "ta3/ai/models/models/utility.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <algorithm>
#include <array>
#include <span>

namespace ta3::ai {

constexpr size_t V9_INPUTS = 10;
constexpr size_t V9_FC1_SIZE = 16, V9_FC2_SIZE = 8, V9_OUT_SIZE = 1;

/** @brief the v8 feature vector plus a well-count metric -- three fc layers over the v9 feature vector */
class ModelV9 {
    enum feature : size_t {
        SURFACE_VARIANCE,
        HOLES,
        CLEAR_0,
        CLEAR_1,
        CLEAR_2,
        CLEAR_3,
        CLEAR_SCORE,
        AGG_HEIGHT,
        HELD_I,
        WELLS,
        FEATURE_COUNT,
    };
    static_assert(FEATURE_COUNT == V9_INPUTS, "feature layout must match the net input width");

public:
    static constexpr size_t INPUTS = V9_INPUTS;
    static constexpr size_t OUTPUTS = V9_OUT_SIZE;
    static constexpr sim::dvec2 BOUNDS{-2, 2};

    // Calculate parameter count at compile time for the flat vector array
    static constexpr size_t NUM_PARAMS = fc_layer_params(V9_INPUTS, V9_FC1_SIZE) +
        fc_layer_params(V9_FC1_SIZE, V9_FC2_SIZE) +
        fc_layer_params(V9_FC2_SIZE, V9_OUT_SIZE);

    /** @brief a full flat weight buffer (host array or device shared memory), fixed at NUM_PARAMS */
    using weights_t = std::span<ai::data_t const, NUM_PARAMS>;

    static constexpr size_t inputs() { return INPUTS; }
    static constexpr size_t outputs() { return OUTPUTS; }
    static constexpr sim::dvec2 bounds() { return BOUNDS; }

    static constexpr data_t CLEAR_COUNT_NORM = data_t{1} / data_t{4};
    static constexpr data_t CLEAR_SCORE_NORM = data_t{1} / data_t{10};

    /**
     * @brief encodes the inputs
     * @param clears clears accumulated from the committed board down to @p board
     * @param held_is_i whether the piece sitting in the hold slot is an I piece
     * @param[out] out exactly @ref INPUTS values to overwrite
     */
    static constexpr void extractInputs(
        ai::clear_hist_t clears,
        sim::Board2 const& board,
        bool held_is_i,
        std::array<data_t, INPUTS>& out
    ) {
        static_assert(CLEAR_1 == CLEAR_0 + 1 && CLEAR_2 == CLEAR_0 + 2 && CLEAR_3 == CLEAR_0 + 3);
        static_assert(ai::CLEAR_KINDS == 4);

        auto rawScore = 0;
        for(size_t k = 0; k < ai::CLEAR_KINDS; ++k) {
            auto const n = clears.count(k);
            out[CLEAR_0 + k] = static_cast<data_t>(n) * CLEAR_COUNT_NORM;
            rawScore += metric::CLEAR_SCORE_TABLE[k + 1] * static_cast<int>(n); // table indexed by LINES
        }

        out[SURFACE_VARIANCE] = metric::norm_surface_var(board);
        out[HOLES] = metric::norm_holes(board);
        out[CLEAR_SCORE] = static_cast<data_t>(rawScore) * CLEAR_SCORE_NORM;
        out[AGG_HEIGHT] = metric::norm_agg_height(board);
        out[HELD_I] = held_is_i ? data_t{1} : data_t{0};
        out[WELLS] = metric::norm_wells(board);
    }

    constexpr ModelV9() { init(); }

    constexpr ModelV9(std::span<double const> weights) { loadWeights(weights); }


    /** @brief overwrites the net weights in place (no re-allocation), for reuse across evaluations */
    constexpr void loadWeights(std::span<double const> weights);

    [[nodiscard]] constexpr std::array<ai::data_t, OUTPUTS> forward(
        std::array<ai::data_t, INPUTS> const& input
    ) const;

    /** @brief forward pass reading weights from an external buffer (host array or device shared memory) */
    [[nodiscard]] static constexpr std::array<ai::data_t, OUTPUTS> forward(
        std::array<ai::data_t, INPUTS> const& input,
        weights_t w
    );


    /**
     * evals the board with the weights
     */
    [[nodiscard]] static constexpr ai::data_t evaluate(
        ai::clear_hist_t clears,
        sim::Board2 const& board,
        bool held_is_i,
        weights_t w
    ) {
        std::array<ai::data_t, INPUTS> in{};
        extractInputs(clears, board, held_is_i, in);
        return forward(in, w)[0];
    }

    /**
     * evals the board with the weights
     */
    [[nodiscard]] constexpr ai::data_t evaluate(
        ai::clear_hist_t clears,
        sim::Board2 const& board,
        bool held_is_i
    ) const {
        std::array<ai::data_t, INPUTS> in{};
        extractInputs(clears, board, held_is_i, in);
        return forward(in, weights_t{_weights})[0];
    }

    [[nodiscard]] constexpr size_t size() const { return NUM_PARAMS; }

    [[nodiscard]] static constexpr size_t params() { return NUM_PARAMS; }

private:
    constexpr void init();

    std::array<ai::data_t, NUM_PARAMS> _weights{};
};

} // namespace ta3::ai



namespace ta3::ai {



constexpr void ModelV9::init() { _weights.fill(0); }

constexpr void ModelV9::loadWeights(std::span<double const> weights) {
    for(size_t i = 0; i < NUM_PARAMS; ++i)
        _weights[i] = static_cast<ai::data_t>(weights[i]);
}



constexpr std::array<ai::data_t, ModelV9::OUTPUTS> ModelV9::forward(
    std::array<ai::data_t, INPUTS> const& input,
    weights_t w
) {
    /** @brief per-layer param counts, for slicing the flat weight buffer */
    constexpr size_t l1Params = fc_layer_params(V9_INPUTS, V9_FC1_SIZE);
    constexpr size_t l2Params = fc_layer_params(V9_FC1_SIZE, V9_FC2_SIZE);
    constexpr size_t l3Params = fc_layer_params(V9_FC2_SIZE, V9_OUT_SIZE);
    static_assert(
        l1Params + l2Params + l3Params == NUM_PARAMS,
        "layer slices must cover the flat weight buffer exactly"
    );

    auto const h1 = dense<INPUTS, V9_FC1_SIZE, true>(input, w.subspan<0, l1Params>());
    auto const h2 = dense<V9_FC1_SIZE, V9_FC2_SIZE, true>(h1, w.subspan<l1Params, l2Params>());
    return dense<V9_FC2_SIZE, OUTPUTS, false>(h2, w.subspan<l1Params + l2Params, l3Params>());
}

constexpr std::array<ai::data_t, ModelV9::OUTPUTS> ModelV9::forward(
    std::array<ai::data_t, INPUTS> const& input
) const { return forward(input, weights_t{_weights}); }

} // namespace ta3::ai
