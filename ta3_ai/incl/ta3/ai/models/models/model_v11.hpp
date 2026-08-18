#pragma once

#include "ta3/ai/model_defs.hpp"
#include "ta3/ai/models/metrics/tetris_stats_v4.hpp"
#include "ta3/ai/models/metrics/v_board_metrics.hpp"
#include "ta3/ai/models/models/utility.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <algorithm>
#include <array>
#include <span>

namespace ta3::ai {

constexpr size_t V11_INPUTS = 10;
constexpr size_t V11_FC1_SIZE = 8;
constexpr size_t V11_OUT_SIZE = 1;

/** @brief the v8 feature vector plus a well-count metric -- three fc layers over the v9 feature vector */
class ModelV11 {
    enum feature : size_t {
        CLEAR_0,
        CLEAR_1,
        CLEAR_2,
        CLEAR_3,
        SURFACE_VARIANCE,
        HOLES,
        HOLE_DEPTHS,
        MAX_HEIGHT,
        HELD_I,
        WELLS,
        FEATURE_COUNT,
    };
    static_assert(FEATURE_COUNT == V11_INPUTS, "feature layout must match the net input width");

public:
    static constexpr size_t INPUTS = V11_INPUTS;
    static constexpr size_t OUTPUTS = V11_OUT_SIZE;
    static constexpr sim::dvec2 BOUNDS{-2, 2};

    // Calculate parameter count at compile time for the flat vector array
    static constexpr size_t NUM_PARAMS = fc_layer_params(V11_INPUTS, V11_FC1_SIZE)
        + fc_layer_params(V11_FC1_SIZE, V11_OUT_SIZE);

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

        for(size_t k = 0; k < ai::CLEAR_KINDS; ++k) {
            auto const n = clears.count(k);
            out[CLEAR_0 + k] = static_cast<data_t>(n) * CLEAR_COUNT_NORM;
        }

        // single fused column pass over the board, reading the normalised metrics directly (max_height raw)
        metric::fuse_v_board_metrics<
            metric::norm_surface_var_t,
            metric::norm_holes_t,
            metric::norm_hole_depths_t,
            metric::max_height_t,
            metric::norm_wells_t
        > const m{board};

        out[SURFACE_VARIANCE] = m.get<metric::norm_surface_var_t>();
        out[HOLES] = m.get<metric::norm_holes_t>();
        out[HOLE_DEPTHS] = m.get<metric::norm_hole_depths_t>();
        out[MAX_HEIGHT] = static_cast<data_t>(m.get<metric::max_height_t>());
        out[HELD_I] = held_is_i ? data_t{1} : data_t{0};
        out[WELLS] = m.get<metric::norm_wells_t>();
    }

    constexpr ModelV11() { init(); }

    constexpr ModelV11(std::span<double const> weights) { loadWeights(weights); }


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



constexpr void ModelV11::init() { _weights.fill(0); }

constexpr void ModelV11::loadWeights(std::span<double const> weights) {
    for(size_t i = 0; i < NUM_PARAMS; ++i)
        _weights[i] = static_cast<ai::data_t>(weights[i]);
}



constexpr std::array<ai::data_t, ModelV11::OUTPUTS> ModelV11::forward(
    std::array<ai::data_t, INPUTS> const& input,
    weights_t w
) {
    /** @brief per-layer param counts, for slicing the flat weight buffer */
    constexpr size_t l1Params = fc_layer_params(V11_INPUTS, V11_FC1_SIZE);
    constexpr size_t l2Params = fc_layer_params(V11_FC1_SIZE, V11_OUT_SIZE);
    static_assert(
        l1Params + l2Params == NUM_PARAMS,
        "layer slices must cover the flat weight buffer exactly"
    );

    auto const h1 = dense<INPUTS, V11_FC1_SIZE, true>(input, w.subspan<0, l1Params>());
    return dense<V11_FC1_SIZE, OUTPUTS, false>(h1, w.subspan<l1Params, l2Params>());
}

constexpr std::array<ai::data_t, ModelV11::OUTPUTS> ModelV11::forward(
    std::array<ai::data_t, INPUTS> const& input
) const { return forward(input, weights_t{_weights}); }

} // namespace ta3::ai
