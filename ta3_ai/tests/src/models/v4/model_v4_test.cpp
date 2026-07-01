#include "ta3/ai/models/v4/model_v4.hpp"

#include "helpers.hpp"
#include "models/v4/test.hpp"
#include "test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

namespace ta3::ai {

namespace {

    constexpr size_t EXPECTED_WEIGHT_COUNT = 2365;

    // a board with one O placed and its stats projected, so the feature vector is non-trivial
    TetrisStatsV4 played_stats(sim::Board2& board) {
        auto const landing = board.dropLocation(sim::PieceType::O, sim::Orientation::TOP, sim::vec2{0, 0});
        board.place(sim::PieceType::O, sim::Orientation::TOP, landing);

        TetrisStatsV4 stats{};
        stats.advance(board.fullLines(), board, sim::PieceType::O, sim::PieceType::COUNT);
        return stats;
    }

    std::array<ai::data_t, ModelV4::INPUTS> make_v4_buffer() {
        sim::Board2 board{};
        auto const stats = played_stats(board);
        std::array<sim::PieceType, sim::PIECE_QUEUE_SIZE - 1> const lookahead{
            sim::PieceType::T,
            sim::PieceType::S,
            sim::PieceType::Z,
            sim::PieceType::J
        };

        std::array<ai::data_t, ModelV4::INPUTS> buffer{};
        ModelV4::extractInputs(parse_inputs_t{stats, board, lookahead}, buffer);
        return buffer;
    }

} // namespace

V4_MODEL_TEST(constants, topology) {
    EXPECT_EQ(ModelV4::INPUTS, 209u);
    EXPECT_EQ(ModelV4::OUTPUTS, 1u);
    EXPECT_EQ(ModelV4{}.size(), EXPECTED_WEIGHT_COUNT);
}

V4_MODEL_TEST(extractInputs, board_grid_then_metadata) {
    sim::Board2 board{};
    auto const stats = played_stats(board);

    std::array<sim::PieceType, 1> const lookahead{sim::PieceType::T};
    std::array<ai::data_t, ModelV4::INPUTS> out{};
    ModelV4::extractInputs(parse_inputs_t{stats, board, lookahead}, out);

    // the placed O sets some occupancy cells in the board section
    EXPECT_TRUE(std::ranges::any_of(std::span{out}.first(V4_BOARD_SIZE), [](ai::data_t v) { return v != 0.f; }));
    // holes is the second metadata scalar, right after the board section
    EXPECT_EQ(out[V4_BOARD_SIZE + 1], static_cast<ai::data_t>(stats.holes()));
}

V4_MODEL_TEST(forward, returns_single_finite_score) {
    ModelV4 const model;
    auto const buffer = make_v4_buffer();

    auto const result = model.forward(std::span<ai::data_t const, ModelV4::INPUTS>{buffer});

    EXPECT_EQ(result.size(), ModelV4::OUTPUTS);
    EXPECT_TRUE(std::isfinite(result[0]));
}

V4_MODEL_TEST(load, accepts_full_weight_vector) {
    auto const weights = test::zero_weights(EXPECTED_WEIGHT_COUNT);
    EXPECT_NO_THROW(ModelV4{weights});
}

V4_MODEL_TEST(batchForward, derives_count_from_buffer_size) {
    ModelV4 const model;
    auto const one = make_v4_buffer();

    std::array<ai::data_t, ModelV4::INPUTS * 3> buffer{};
    for(auto i = 0uz; i < 3; ++i)
        std::ranges::copy(one, buffer.begin() + i * ModelV4::INPUTS);

    auto const out = model.batchForward(buffer);

    EXPECT_EQ(out.size(), 3u * ModelV4::OUTPUTS);
}

V4_MODEL_TEST(batchForward, matches_single_forward) {
    ModelV4 const model;
    auto const buffer = make_v4_buffer();

    auto const single = model.forward(std::span<ai::data_t const, ModelV4::INPUTS>{buffer});
    auto const batch = model.batchForward(buffer);

    ASSERT_EQ(batch.size(), single.size());
    EXPECT_NEAR(batch[0], single[0], 1e-4f);
}

}
