#include "ta3/ai/models/v5/model_v5.hpp"

#include "helpers.hpp"
#include "models/v5/test.hpp"
#include "test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

namespace ta3::ai {

namespace {

    // a board with one O placed and its stats projected, so the feature vector is non-trivial
    TetrisStatsV4 played_stats(sim::Board2& board) {
        auto const landing = board.dropLocation(sim::PieceType::O, sim::Orientation::TOP, sim::vec2{0, 0});
        board.place(sim::PieceType::O, sim::Orientation::TOP, landing);

        TetrisStatsV4 stats{};
        stats.advance(board.fullLines(), board, sim::PieceType::O, sim::PieceType::COUNT);
        return stats;
    }

    std::array<ai::data_t, ModelV5::INPUTS> make_v5_buffer() {
        sim::Board2 board{};
        auto const stats = played_stats(board);
        std::array<sim::PieceType, 1> const lookahead{sim::PieceType::T};

        std::array<ai::data_t, ModelV5::INPUTS> buffer{};
        ModelV5::extractInputs(parse_inputs_t{stats, board, lookahead}, buffer);
        return buffer;
    }

} // namespace

V5_MODEL_TEST(constants, topology) {
    EXPECT_EQ(ModelV5::INPUTS, 10u);
    EXPECT_EQ(ModelV5::OUTPUTS, 1u);
    EXPECT_GT(ModelV5{}.size(), 0u);
}

V5_MODEL_TEST(extractInputs, lays_out_next_piece_held_and_stats) {
    sim::Board2 board{};
    auto const stats = played_stats(board);

    std::array<sim::PieceType, 2> const lookahead{sim::PieceType::T, sim::PieceType::S};
    std::array<ai::data_t, ModelV5::INPUTS> out{};
    ModelV5::extractInputs(parse_inputs_t{stats, board, lookahead}, out);

    EXPECT_EQ(out[0], static_cast<ai::data_t>(*sim::PieceType::T));     // next piece = lookahead front
    EXPECT_EQ(out[1], static_cast<ai::data_t>(*sim::PieceType::COUNT)); // empty hold slot
    EXPECT_EQ(out[2], static_cast<ai::data_t>(stats.holes()));
    EXPECT_EQ(out[9], static_cast<ai::data_t>(stats.pieces()));
}

V5_MODEL_TEST(forward, returns_single_finite_score) {
    ModelV5 const model;
    auto const buffer = make_v5_buffer();

    auto const result = model.forward(std::span<ai::data_t const, ModelV5::INPUTS>{buffer});

    EXPECT_EQ(result.size(), ModelV5::OUTPUTS);
    EXPECT_TRUE(std::isfinite(result[0]));
}

V5_MODEL_TEST(load, accepts_full_weight_vector) {
    auto const weights = test::zero_weights(ModelV5{}.size());
    EXPECT_NO_THROW(ModelV5{weights});
}

V5_MODEL_TEST(batchForward, derives_count_from_buffer_size) {
    ModelV5 const model;
    auto const one = make_v5_buffer();

    std::array<ai::data_t, ModelV5::INPUTS * 3> buffer{};
    for(auto i = 0uz; i < 3; ++i)
        std::ranges::copy(one, buffer.begin() + i * ModelV5::INPUTS);

    auto const out = model.batchForward(buffer);

    EXPECT_EQ(out.size(), 3u * ModelV5::OUTPUTS);
}

V5_MODEL_TEST(batchForward, matches_single_forward) {
    ModelV5 const model;
    auto const buffer = make_v5_buffer();

    auto const single = model.forward(std::span<ai::data_t const, ModelV5::INPUTS>{buffer});
    auto const batch = model.batchForward(buffer);

    ASSERT_EQ(batch.size(), single.size());
    EXPECT_NEAR(batch[0], single[0], 1e-4f);
}

}
