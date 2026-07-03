#include "ta3/ai/models/v4/input_data_v4.hpp"

#include "helpers.hpp"
#include "models/v4/test.hpp"
#include "test.hpp"

namespace ta3::ai {

namespace {

constexpr size_t METADATA_HOLES_INDEX = 1;
constexpr size_t METADATA_NEW_HOLES_INDEX = 2;
constexpr size_t METADATA_ROUGHNESS_INDEX = 3;
constexpr size_t METADATA_PIECES_INDEX = 4;

} // namespace

V4_INPUT_TEST(constants, layout) {
    EXPECT_EQ(input_data_v4::HEIGHT, 20u);
    EXPECT_EQ(input_data_v4::WIDTH, 10u);
    EXPECT_EQ(input_data_v4::BOARD_SIZE, 200u);
    EXPECT_EQ(input_data_v4::METADATA_SIZE, 9u);
    EXPECT_EQ(input_data_v4::SIZE, 209u);
}

V4_INPUT_TEST(empty_boards, zero_board_and_metadata) {
    sim::Board const board = test::empty_board();
    auto const input = test::make_input(board, board);

    EXPECT_TRUE(std::ranges::all_of(input.board_data(), [](data_t value) { return value == 0.f; }));
    EXPECT_EQ(input.metadata()[0], 0.f);
    EXPECT_EQ(input.metadata()[METADATA_HOLES_INDEX], 0.f);
    EXPECT_EQ(input.metadata()[METADATA_NEW_HOLES_INDEX], 0.f);
    EXPECT_EQ(input.metadata()[METADATA_ROUGHNESS_INDEX], 0.f);
}

V4_INPUT_TEST(board_delta, encodes_placement_change) {
    auto const prev = test::empty_board();
    sim::Board current = test::empty_board();
    current.place(sim::PieceType::O, sim::Orientation::TOP, sim::vec2{3, 21});

    auto const input = test::make_input(prev, current);

    // the O covers cols 4,5; the top crop maps sim row 21 down by (sim::HEIGHT - HEIGHT)
    constexpr size_t inputRow = 21 - (sim::HEIGHT - input_data_v4::HEIGHT);
    EXPECT_GT(input.board_data()[inputRow * input_data_v4::WIDTH + 4], 0.f);
    EXPECT_GT(input.board_data()[inputRow * input_data_v4::WIDTH + 5], 0.f);
}

V4_INPUT_TEST(holes, matches_board_hole_count) {
    sim::Board board = test::empty_board();
    board.place(sim::PieceType::O, sim::Orientation::TOP, sim::vec2{0, 20});

    auto const input = test::make_input(board, board);

    EXPECT_EQ(input.metadata()[METADATA_HOLES_INDEX], static_cast<data_t>(board.holes()));
    EXPECT_EQ(board.holes(), 2);
}

V4_INPUT_TEST(new_holes, reports_hole_delta) {
    sim::Board prev = test::empty_board();
    sim::Board current = test::empty_board();
    current.place(sim::PieceType::O, sim::Orientation::TOP, sim::vec2{0, 20});

    auto const input = test::make_input(prev, current);

    EXPECT_EQ(input.metadata()[METADATA_NEW_HOLES_INDEX], static_cast<data_t>(current.holes()));
}

V4_INPUT_TEST(roughness, uneven_stack_is_positive) {
    sim::Board board = test::empty_board();
    board.place(sim::PieceType::O, sim::Orientation::TOP, sim::vec2{0, 21});
    board.place(sim::PieceType::O, sim::Orientation::TOP, sim::vec2{4, 19});

    auto const input = test::make_input(board, board);

    EXPECT_GT(input.metadata()[METADATA_ROUGHNESS_INDEX], 0.f);
}

V4_INPUT_TEST(piece_queue, copies_queue_into_metadata) {
    sim::Board const board = test::empty_board();
    auto const queue = test::default_piece_queue();

    auto const input = test::make_input(board, board, queue);

    for(size_t i = 0; i < sim::PIECE_QUEUE_SIZE; ++i)
        EXPECT_EQ(input.metadata()[METADATA_PIECES_INDEX + i], static_cast<data_t>(*queue[i]));
}

V4_INPUT_TEST(board_matrix, has_expected_shape) {
    sim::Board const board = test::empty_board();
    auto const input = test::make_input(board, board);

    auto const matrix = input.board_matrix();
    EXPECT_EQ(matrix.nr(), static_cast<long>(input_data_v4::HEIGHT));
    EXPECT_EQ(matrix.nc(), static_cast<long>(input_data_v4::WIDTH));
}

V4_INPUT_TEST(flat_views, expose_full_input_layout) {
    sim::Board const board = test::empty_board();
    auto const input = test::make_input(board, board);

    EXPECT_EQ(input.board_data().size(), input_data_v4::BOARD_SIZE);
    EXPECT_EQ(input.metadata().size(), input_data_v4::METADATA_SIZE);
}

}
