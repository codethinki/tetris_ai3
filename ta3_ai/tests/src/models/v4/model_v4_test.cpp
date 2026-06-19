#include "ta3/ai/model.hpp"

#include "helpers.hpp"
#include "models/v4/test.hpp"
#include "test.hpp"

namespace ta3::ai {

namespace {

constexpr size_t EXPECTED_WEIGHT_COUNT = 2365;

sim::Board stacked_board() {
    sim::Board board = test::empty_board();
    test::fill_row(board, 22);
    return board;
}

} // namespace

V4_MODEL_TEST(constants, topology) {
    EXPECT_EQ(ModelV4::INPUTS, 209u);
    EXPECT_EQ(ModelV4::OUTPUTS, 1u);
    EXPECT_EQ(ModelV4{}.size(), EXPECTED_WEIGHT_COUNT);
}

V4_MODEL_TEST(default_construct, exposes_weight_count) {
    ModelV4 const model;
    EXPECT_EQ(model.size(), EXPECTED_WEIGHT_COUNT);
}

V4_MODEL_TEST(load, accepts_full_weight_vector) {
    auto const weights = test::zero_weights(EXPECTED_WEIGHT_COUNT);
    EXPECT_NO_THROW(ModelV4{weights});
}

V4_MODEL_TEST(forward, returns_single_score) {
    ModelV4 const model;
    sim::Board const board = test::empty_board();
    auto const input = test::make_input(board, board);

    auto const result = model.forward(input);

    EXPECT_EQ(result.size(), ModelV4::OUTPUTS);
    EXPECT_TRUE(std::isfinite(result[0]));
}

V4_MODEL_TEST(batchForward, empty_input_returns_empty_output) {
    ModelV4 const model;
    std::vector<input_data_v4> const inputs{};

    auto const out = model.batchForward(std::span<input_data_v4 const>{inputs});

    EXPECT_TRUE(out.empty());
}

V4_MODEL_TEST(batchForward, output_size_matches_batch) {
    ModelV4 const model;
    sim::Board const board = test::empty_board();
    auto const input = test::make_input(board, board);

    std::array<input_data_v4, 3> const inputs{input, input, input};
    auto const out = model.batchForward(std::span<input_data_v4 const>{inputs});

    EXPECT_EQ(out.size(), inputs.size() * ModelV4::OUTPUTS);
}

V4_MODEL_TEST(batchForward, matches_single_forward) {
    ModelV4 const model;
    sim::Board const board = test::empty_board();
    auto const input = test::make_input(board, board);

    auto const single = model.forward(input);
    std::array<input_data_v4, 1> const inputs{input};
    auto const batch = model.batchForward(std::span<input_data_v4 const>{inputs});

    ASSERT_EQ(batch.size(), single.size());
    EXPECT_NEAR(batch[0], single[0], 1e-4f);
}

V4_MODEL_TEST(batchForward, distinct_inputs_produce_finite_scores) {
    ModelV4 const model;
    sim::Board const empty = test::empty_board();
    sim::Board const stacked = stacked_board();

    auto const emptyInput = test::make_input(empty, empty);
    auto const stackedInput = test::make_input(stacked, stacked);

    std::array<input_data_v4, 2> const inputs{emptyInput, stackedInput};
    auto const out = model.batchForward(std::span<input_data_v4 const>{inputs});

    ASSERT_EQ(out.size(), 2u);
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_TRUE(std::isfinite(out[1]));
}

V4_MODEL_TEST(reload, new_weight_vector_still_runs) {
    auto weightsA = test::zero_weights(EXPECTED_WEIGHT_COUNT);
    auto weightsB = test::zero_weights(EXPECTED_WEIGHT_COUNT);
    weightsB[0] = 0.5;

    ModelV4 const modelA{weightsA};
    ModelV4 const modelB{weightsB};

    sim::Board const board = test::empty_board();
    auto const input = test::make_input(board, board);

    EXPECT_NO_THROW(modelA.forward(input));
    EXPECT_NO_THROW(modelB.forward(input));
}

}
