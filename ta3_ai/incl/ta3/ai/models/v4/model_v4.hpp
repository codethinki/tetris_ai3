#pragma once
#include "ta3/ai/models/v4/input_data_v4.hpp"

#include "ta3/ai/model_defs.hpp"

#include "ta3/ai/lib/dlib.hpp"

#include <array>
#include <memory>
#include <span>
#include <vector>


namespace ta3::ai {
constexpr ta3::sim::szvec2 STAGE1_IN_DIM{input_data_v4::WIDTH, input_data_v4::HEIGHT};

constexpr size_t COMPRESSED_COL_SIZE = 4;

constexpr size_t CONV1_DIM = COMPRESSED_COL_SIZE;

constexpr ta3::sim::szvec2 CONV1_FILTER{1, STAGE1_IN_DIM.y}, CONV2_FILTER{5, 1}, CONV3_FILTER{1, 1};

constexpr ta3::sim::szvec2 CONV1_STRIDE = CONV1_FILTER;

using stage1_in = dlib::input<input_data_v4::board_matrix_t>;
using stage1_1 = dlib::relu<dlib::bn_con<dlib::con<CONV1_DIM, CONV1_FILTER.y, CONV1_FILTER.x, CONV1_STRIDE.y, CONV1_STRIDE.x, stage1_in>>>;
using stage1_out = stage1_1;


constexpr size_t STAGE1_OUT_SIZE = STAGE1_IN_DIM.x * CONV1_DIM;

constexpr size_t STAGE2_IN_SIZE = STAGE1_OUT_SIZE + input_data_v4::METADATA_SIZE;

static constexpr size_t FC1_SIZE = 32, FC2_SIZE = 16, FC3_SIZE = 8, FC4_SIZE = 1;

using stage2_in_matrix = dlib::matrix<ai::data_t, STAGE2_IN_SIZE, 1>;
using stage2_in = dlib::input<stage2_in_matrix>;

using stage2_1 = dlib::relu<dlib::fc<FC1_SIZE, stage2_in>>;
using stage2_2 = dlib::relu<dlib::fc<FC2_SIZE, stage2_1>>;
using stage2_3 = dlib::relu<dlib::fc<FC3_SIZE, stage2_2>>;
using stage2_4 = dlib::relu<dlib::fc<FC4_SIZE, stage2_3>>;
using stage2_out = stage2_4;

constexpr size_t STAGE2_OUT_SIZE = FC4_SIZE;


class ModelV4 {
public:
    static constexpr size_t INPUTS = input_data_v4::SIZE;
    static constexpr size_t OUTPUTS = STAGE2_OUT_SIZE;
    using input_data_t = input_data_v4;
    using stats_t = StatsV4;
    using board_matrix_t = input_data_t::board_matrix_t;
    using metadata_t = input_data_t::metadata_t;

    /** @brief encodes one candidate into @ref out (exactly @ref INPUTS values), the batch-driver entry */
    static constexpr void extractInputs(parse_inputs_t const& in, std::span<data_t> out) {
        input_data_t::extractInputs(in, out);
    }


    ModelV4() : _stage1{std::make_unique<stage1_out>()}, _stage2{std::make_unique<stage2_out>()} { init(); }
    explicit ModelV4(std::span<double const> weights);

    [[nodiscard]] std::array<ai::data_t, STAGE2_OUT_SIZE> forward(input_data_t const& input_data) const;
    [[nodiscard]] std::vector<ai::data_t> batchForward(std::span<input_data_t const> inputs) const;

    /** @brief scores a packed buffer of @ref INPUTS-wide parsed inputs; the count is its size / @ref INPUTS */
    [[nodiscard]] std::vector<ai::data_t> batchForward(std::span<ai::data_t const> inputs) const;


    [[nodiscard]] std::array<ai::data_t, STAGE2_OUT_SIZE> forward(board_matrix_t input, metadata_t metadata) const;
    [[nodiscard]] std::vector<ai::data_t> batchForward(std::span<board_matrix_t const> boards, std::span<metadata_t const> metadata) const;

private:
    void init() const;

    std::unique_ptr<stage1_out> _stage1;
    std::unique_ptr<stage2_out> _stage2;

public:
    [[nodiscard]] size_t size() const;
};
}
