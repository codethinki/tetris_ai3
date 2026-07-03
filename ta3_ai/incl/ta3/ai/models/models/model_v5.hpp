#pragma once
#include "../metrics/tetris_stats_v4.hpp"

#include "ta3/ai/model_defs.hpp"

#include "ta3/ai/lib/dlib.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace ta3::ai {

constexpr size_t V5_INPUTS = 10;

// a tiny fully-connected stack that just weights the stat vector: 10 -> 8 -> 4 -> 1
constexpr size_t V5_FC1_SIZE = 8, V5_FC2_SIZE = 4, V5_OUT_SIZE = 1;

using v5_input_matrix = dlib::matrix<ai::data_t, V5_INPUTS, 1>;
using v5_in = dlib::input<v5_input_matrix>;
using v5_1 = dlib::relu<dlib::fc<V5_FC1_SIZE, v5_in>>;
using v5_2 = dlib::relu<dlib::fc<V5_FC2_SIZE, v5_1>>;
using v5_out = dlib::fc<V5_OUT_SIZE, v5_2>;

/** @brief a minimal stats-only value model -- three fc layers over the v5 feature vector */
class ModelV5 {
    enum feature : size_t {
        NEXT_PIECE,
        HELD,
        HOLES,
        NEW_HOLES,
        ROUGHNESS,
        LINES_CLEARED,
        AGG_HEIGHT,
        MAX_HEIGHT,
        BUMPINESS,
        PIECES,
        FEATURE_COUNT,
    };
    static_assert(FEATURE_COUNT == V5_INPUTS, "feature layout must match the net input width");

public:
    static constexpr size_t INPUTS = V5_INPUTS;
    static constexpr size_t OUTPUTS = V5_OUT_SIZE;
    static constexpr glm::dvec2 BOUNDS{-2, 2};
    static constexpr size_t inputs() { return INPUTS; }
    static constexpr size_t outputs() { return OUTPUTS; }
    static constexpr glm::dvec2 bounds() { return BOUNDS; }

    using input_matrix_t = v5_input_matrix;
    using tetris_stats_t = stats_v4;


    /**
     * @brief encodes one candidate: the next piece, the held piece, then the board stats
     * @param[out] out exactly @ref INPUTS values to overwrite
     */
    static constexpr void extractInputs(tetris_stats_t const& stats, std::span<data_t> out) {
        out[NEXT_PIECE] = static_cast<data_t>(*stats.get(metric::next_piece));
        out[HELD] = static_cast<data_t>(*stats.get(metric::held_piece));
        out[HOLES] = static_cast<data_t>(stats.get(metric::holes));
        out[NEW_HOLES] = static_cast<data_t>(stats.get(metric::new_holes));
        out[ROUGHNESS] = static_cast<data_t>(stats.get(metric::surface_variance));
        out[LINES_CLEARED] = static_cast<data_t>(stats.get(metric::lines_cleared));
        out[AGG_HEIGHT] = static_cast<data_t>(stats.get(metric::agg_height));
        out[MAX_HEIGHT] = static_cast<data_t>(stats.get(metric::max_height));
        out[BUMPINESS] = static_cast<data_t>(stats.get(metric::bumpiness));
        out[PIECES] = static_cast<data_t>(stats.get(metric::pieces_placed));
    }

    ModelV5() : _net{std::make_unique<v5_out>()} { init(); }
    explicit ModelV5(std::span<double const> weights);

    /** @brief overwrites the net weights in place (no re-allocation), for reuse across evaluations */
    void loadWeights(std::span<double const> weights);

    [[nodiscard]] std::array<ai::data_t, OUTPUTS> forward(std::span<ai::data_t const, INPUTS> input) const;

    /** @brief scores a packed buffer of @ref INPUTS-wide parsed inputs; the count is its size / @ref INPUTS */
    [[nodiscard]] std::vector<ai::data_t> batchForward(std::span<ai::data_t const> inputs) const;

private:
    void init() const;

    std::unique_ptr<v5_out> _net;

public:
    [[nodiscard]] size_t size() const;

    [[nodiscard]] static size_t params() {
        static auto const COUNT = ModelV5{}.size();
        return COUNT;
    }
};
}
