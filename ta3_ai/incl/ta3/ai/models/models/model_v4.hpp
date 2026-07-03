#pragma once
#include "../metrics/tetris_stats_v4.hpp"

#include "ta3/ai/model_defs.hpp"

#include "ta3/ai/lib/dlib.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/pieces/piece_defs.hpp>
#include <ta3/sim/tetris_defs.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <vector>

namespace ta3::ai {

// input layout: a top-cropped occupancy grid followed by the scalar metadata
constexpr size_t V4_BOARD_HEIGHT = std::min(ta3::sim::HEIGHT, 20uz);
constexpr size_t V4_BOARD_WIDTH = ta3::sim::WIDTH;
constexpr size_t V4_BOARD_SIZE = V4_BOARD_WIDTH * V4_BOARD_HEIGHT;

// metadata: lines cleared, holes, new holes, surfaceVar, then the piece queue (lookahead + held)
constexpr size_t V4_METADATA_SIZE = 4 + ta3::sim::PIECE_QUEUE_SIZE;
constexpr size_t V4_INPUTS = V4_BOARD_SIZE + V4_METADATA_SIZE;

using v4_board_matrix = dlib::matrix<ai::data_t, V4_BOARD_HEIGHT, V4_BOARD_WIDTH>;
using v4_metadata = std::span<ai::data_t const, V4_METADATA_SIZE>;

constexpr ta3::sim::szvec2 STAGE1_IN_DIM{V4_BOARD_WIDTH, V4_BOARD_HEIGHT};

constexpr size_t COMPRESSED_COL_SIZE = 4;

constexpr size_t CONV1_DIM = COMPRESSED_COL_SIZE;

constexpr ta3::sim::szvec2 CONV1_FILTER{1, STAGE1_IN_DIM.y}, CONV2_FILTER{5, 1}, CONV3_FILTER{1, 1};

constexpr ta3::sim::szvec2 CONV1_STRIDE = CONV1_FILTER;

using stage1_in = dlib::input<v4_board_matrix>;
using stage1_1 = dlib::relu<dlib::bn_con<dlib::con<CONV1_DIM, CONV1_FILTER.y, CONV1_FILTER.x, CONV1_STRIDE.y,
    CONV1_STRIDE.x, stage1_in>>>;
using stage1_out = stage1_1;


constexpr size_t STAGE1_OUT_SIZE = STAGE1_IN_DIM.x * CONV1_DIM;

constexpr size_t STAGE2_IN_SIZE = STAGE1_OUT_SIZE + V4_METADATA_SIZE;

constexpr size_t FC1_SIZE = 32, FC2_SIZE = 16, FC3_SIZE = 8, FC4_SIZE = 1;

using stage2_in_matrix = dlib::matrix<ai::data_t, STAGE2_IN_SIZE, 1>;
using stage2_in = dlib::input<stage2_in_matrix>;

using stage2_1 = dlib::relu<dlib::fc<FC1_SIZE, stage2_in>>;
using stage2_2 = dlib::relu<dlib::fc<FC2_SIZE, stage2_1>>;
using stage2_3 = dlib::relu<dlib::fc<FC3_SIZE, stage2_2>>;
using stage2_4 = dlib::relu<dlib::fc<FC4_SIZE, stage2_3>>;
using stage2_out = stage2_4;

constexpr size_t STAGE2_OUT_SIZE = FC4_SIZE;


/** @brief value model with a column conv stage over the board plus an fc stage over conv + metadata */
class ModelV4 {
    enum meta : size_t {
        LINES_CLEARED,
        HOLES,
        NEW_HOLES,
        ROUGHNESS,
        PIECES,
    };

public:
    static constexpr size_t INPUTS = V4_INPUTS;
    static constexpr size_t OUTPUTS = STAGE2_OUT_SIZE;
    static constexpr glm::dvec2 BOUNDS = {-2, 2};
    static constexpr size_t inputs() { return INPUTS; }
    static constexpr size_t outputs() { return OUTPUTS; }
    static constexpr glm::dvec2 bounds() { return BOUNDS; }
    using tetris_stats_t = stats_v4;

    /**
     * @brief encodes one candidate: the board occupancy grid followed by the scalar metadata
     * @param[out] out exactly @ref inputs() values to overwrite
     */
    static constexpr void extractInputs(tetris_stats_t const& stats, std::span<data_t> out) {
        // crop the hidden buffer rows above the visible field off the top
        constexpr auto topCrop = ta3::sim::HEIGHT - V4_BOARD_HEIGHT;
        auto const* board = stats.get(metric::board);
        for(auto x = 0uz; x < V4_BOARD_WIDTH; ++x) {
            auto const column = board->raw_column(x);
            for(auto y = 0uz; y < V4_BOARD_HEIGHT; ++y)
                out[y * V4_BOARD_WIDTH + x] = static_cast<data_t>((column >> (y + topCrop)) & 1u);
        }

        auto const metadata = out.subspan(V4_BOARD_SIZE);
        metadata[LINES_CLEARED] = static_cast<data_t>(stats.get(metric::lines_cleared));
        metadata[HOLES] = static_cast<data_t>(stats.get(metric::holes));
        metadata[NEW_HOLES] = static_cast<data_t>(stats.get(metric::new_holes));
        metadata[ROUGHNESS] = static_cast<data_t>(stats.get(metric::surface_variance));

        auto const queue = stats.get(metric::piece_queue);
        for(auto i = 0uz; i < queue.size(); ++i)
            metadata[PIECES + i] = static_cast<data_t>(*queue[i]);
        metadata[PIECES + queue.size()] = static_cast<data_t>(*stats.get(metric::held_piece));
    }

    ModelV4() : _stage1{std::make_unique<stage1_out>()}, _stage2{std::make_unique<stage2_out>()} { init(); }
    explicit ModelV4(std::span<double const> weights);

    /** @brief overwrites the net weights in place (no re-allocation), for reuse across evaluations */
    void loadWeights(std::span<double const> weights);

    [[nodiscard]] std::array<ai::data_t, OUTPUTS> forward(std::span<ai::data_t const, INPUTS> input) const;

    /** @brief scores a packed buffer of @ref inputs()-wide parsed inputs; the count is its size / @ref inputs() */
    [[nodiscard]] std::vector<ai::data_t> batchForward(std::span<ai::data_t const> inputs) const;

private:
    [[nodiscard]] std::array<ai::data_t, OUTPUTS> forward(v4_board_matrix board, v4_metadata metadata) const;
    [[nodiscard]] std::vector<ai::data_t> batchForward(
        std::span<v4_board_matrix const> boards,
        std::span<v4_metadata const> metadata
    ) const;

    void init() const;

    std::unique_ptr<stage1_out> _stage1;
    std::unique_ptr<stage2_out> _stage2;

public:
    [[nodiscard]] size_t size() const;

    [[nodiscard]] static size_t params() {
        static auto const COUNT = ModelV4{}.size();
        return COUNT;
    }
};
}
