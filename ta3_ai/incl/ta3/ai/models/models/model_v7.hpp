#pragma once

#include "ta3/ai/model_defs.hpp"
#include "ta3/ai/models/metrics/tetris_stats_v4.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <algorithm>
#include <array>
#include <span>

namespace ta3::ai {

constexpr size_t V7_INPUTS = 8;
constexpr size_t V7_FC1_SIZE = 16, V7_FC2_SIZE = 8, V7_OUT_SIZE = 1;

/** @brief a minimal stats-only value model -- three fc layers over the v7 feature vector */
class ModelV7 {
    enum feature : size_t {
        SURFACE_VARIANCE,
        HOLES,
        CLEAR_0,
        CLEAR_1,
        CLEAR_2,
        CLEAR_3,
        CLEAR_SCORE,
        AGG_HEIGHT,
        FEATURE_COUNT,
    };
    static_assert(FEATURE_COUNT == V7_INPUTS, "feature layout must match the net input width");

public:
    static constexpr size_t INPUTS = V7_INPUTS;
    static constexpr size_t OUTPUTS = V7_OUT_SIZE;
    static constexpr sim::dvec2 BOUNDS{-2, 2};

    // Calculate parameter count at compile time for the flat vector array
    static constexpr size_t NUM_PARAMS = (V7_INPUTS * V7_FC1_SIZE + V7_FC1_SIZE) +
        (V7_FC1_SIZE * V7_FC2_SIZE + V7_FC2_SIZE) +
        (V7_FC2_SIZE * V7_OUT_SIZE + V7_OUT_SIZE);

    /** @brief a full flat weight buffer (host array or device shared memory), fixed at NUM_PARAMS */
    using weights_t = std::span<ai::data_t const, NUM_PARAMS>;

    static constexpr size_t inputs() { return INPUTS; }
    static constexpr size_t outputs() { return OUTPUTS; }
    static constexpr sim::dvec2 bounds() { return BOUNDS; }

    static constexpr data_t CLEAR_COUNT_NORM = data_t{1} / data_t{4};
    static constexpr data_t CLEAR_SCORE_NORM = data_t{1} / data_t{10};

    /**
     * @brief encodes one candidate: the path's clear histogram, then the board stats
     * @param clears clears accumulated from the committed board down to @p board
     * @param[out] out exactly @ref INPUTS values to overwrite
     */
    static constexpr void extractInputs(
        ai::clear_hist_t clears,
        sim::Board2 const& board,
        std::array<data_t, INPUTS>& out
    ) {
        static_assert(CLEAR_1 == CLEAR_0 + 1 && CLEAR_2 == CLEAR_0 + 2 && CLEAR_3 == CLEAR_0 + 3);
        static_assert(ai::CLEAR_KINDS == 4);

        int rawScore = 0;
        for(size_t k = 0; k < ai::CLEAR_KINDS; ++k) {
            auto const n = clears.count(k);
            out[CLEAR_0 + k] = static_cast<data_t>(n) * CLEAR_COUNT_NORM;
            rawScore += metric::CLEAR_SCORE_TABLE[k + 1] * static_cast<int>(n); // table indexed by LINES
        }

        out[SURFACE_VARIANCE] = metric::norm_surface_var(board);
        out[HOLES] = metric::norm_holes(board);
        out[CLEAR_SCORE] = static_cast<data_t>(rawScore) * CLEAR_SCORE_NORM;
        out[AGG_HEIGHT] = metric::norm_agg_height(board);
    }

    constexpr ModelV7() { init(); }

    constexpr ModelV7(std::span<double const> weights) { loadWeights(weights); }


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
        weights_t w
    ) {
        std::array<ai::data_t, INPUTS> in{};
        extractInputs(clears, board, in);
        return forward(in, w)[0];
    }

    /**
     * evals the board with the weights
     */
    [[nodiscard]] constexpr ai::data_t evaluate(
        ai::clear_hist_t clears,
        sim::Board2 const& board
    ) const {
        std::array<ai::data_t, INPUTS> in{};
        extractInputs(clears, board, in);
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



constexpr void ModelV7::init() { _weights.fill(0); }

constexpr void ModelV7::loadWeights(std::span<double const> weights) {
    for(size_t i = 0; i < NUM_PARAMS; ++i)
        _weights[i] = static_cast<ai::data_t>(weights[i]);
}

/**
 * @brief one fully-connected layer, fully sized at compile time
 * @tparam In   input width
 * @tparam Out  output width
 * @tparam Relu apply relu (else linear output)
 * @param in    input activations, by direct array reference so layers chain cleanly
 * @param w     this layer's slice, fixed at Out*In weight matrix + Out biases
 * @return the Out output activations
 * @note weight layout (weights then biases) matches the stored champion files
 * @note activations are std::array, NEVER std::span: spans are pointer objects, and routing the local
 *  activation arrays through one blocked SROA -- on the device that spilled every layer's in/out to
 *  local memory inside the leaf loop. the weights stay a span on purpose: they alias a real external
 *  buffer (device: __shared__), so a pointer is exactly right there.
 */
template<size_t In, size_t Out, bool Relu>
[[nodiscard]] constexpr std::array<data_t, Out> dense(
    std::array<data_t, In> const& in,
    std::span<data_t const, In * Out + Out> w
) {
    std::array<data_t, Out> out{};
    // cap the unroll: fully unrolling interleaves ALL Out accumulator chains for ILP, which multiplies
    // the concurrent live registers (Out=16 chains + address streams) and is a top spill source in the
    // register-capped kernel. 4 concurrent chains still cover FFMA latency.
    // the __CUDA_ARCH__ guard is dead under acpp generic SSCP (single-pass clang, macro never defined);
    // the __clang__ branch re-arms the same cap via llvm.loop.unroll.count metadata, which survives
    // SSCP stage 2 because the full LLVM opt pipeline runs at JIT time. "unroll_count" (rather than the
    // bare "#pragma unroll 4" spelling clang also accepts) is used for clarity, since the two are
    // documented as equivalent (clang lowers "#pragma unroll N" to "#pragma clang loop unroll_count(N)").
#if defined(__CUDA_ARCH__)
#pragma unroll 4
#elif defined(__clang__)
#pragma clang loop unroll_count(4)
#endif
    for(size_t o = 0; o < Out; ++o) {
        data_t v = w[In * Out + o]; // bias, stored after the matrix
        for(size_t i = 0; i < In; ++i)
            v += in[i] * w[o * In + i];
        out[o] = Relu ? std::max<data_t>(0, v) : v;
    }
    return out;
}

/** @brief per-layer param counts, for slicing the flat weight buffer */
constexpr size_t V7_L1 = V7_INPUTS * V7_FC1_SIZE + V7_FC1_SIZE;
constexpr size_t V7_L2 = V7_FC1_SIZE * V7_FC2_SIZE + V7_FC2_SIZE;
constexpr size_t V7_L3 = V7_FC2_SIZE * V7_OUT_SIZE + V7_OUT_SIZE;
static_assert(V7_L1 + V7_L2 + V7_L3 == ModelV7::NUM_PARAMS, "layer slices must cover the flat weight buffer exactly");

constexpr std::array<ai::data_t, ModelV7::OUTPUTS> ModelV7::forward(
    std::array<ai::data_t, INPUTS> const& input,
    weights_t w
) {
    auto const h1 = dense<INPUTS, V7_FC1_SIZE, true>(input, w.subspan<0, V7_L1>());
    auto const h2 = dense<V7_FC1_SIZE, V7_FC2_SIZE, true>(h1, w.subspan<V7_L1, V7_L2>());
    return dense<V7_FC2_SIZE, OUTPUTS, false>(h2, w.subspan<V7_L1 + V7_L2, V7_L3>());
}

constexpr std::array<ai::data_t, ModelV7::OUTPUTS> ModelV7::forward(
    std::array<ai::data_t, INPUTS> const& input
) const { return forward(input, weights_t{_weights}); }

} // namespace ta3::ai
