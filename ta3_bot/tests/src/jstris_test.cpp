#include "test.hpp"

#include "ta3/bot/jstris.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ta3::bot {
namespace {
void write_uint32(std::span<std::byte, sizeof(std::uint32_t)> bytes, std::uint32_t value) {
    for(std::size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<std::byte>(value >> (i * 8));
}

std::array<std::byte, jstris::STATE_MESSAGE_SIZE> state_message() {
    std::array<std::byte, jstris::STATE_MESSAGE_SIZE> message{};
    message[0] = std::byte{'T'};
    message[1] = std::byte{'A'};
    message[2] = std::byte{'3'};
    message[3] = static_cast<std::byte>(jstris::PROTOCOL_VERSION);
    message[5] = std::byte{0};
    message[6] = std::byte{0xff};
    message[7] = std::byte{1};
    message[8] = std::byte{2};
    message[9] = std::byte{3};
    message[10] = std::byte{4};
    message[11] = std::byte{5};
    return message;
}
}

BOT_TEST(jstris, decodes_state) {
    auto message = state_message();
    write_uint32(std::span<std::byte, 4>{message.data() + 12, 4}, 1u << 22);
    write_uint32(std::span<std::byte, 4>{message.data() + 16, 4}, 1u << 3);

    auto const state = jstris::decode_state(message);

    ASSERT_TRUE(state);
    EXPECT_EQ(state->current, sim::PieceType::I);
    EXPECT_EQ(state->held, sim::NO_PIECE);
    EXPECT_EQ(
        state->lookahead,
        (std::array{
            sim::PieceType::O,
            sim::PieceType::T,
            sim::PieceType::L,
            sim::PieceType::J,
            sim::PieceType::S
        })
    );
    EXPECT_EQ(state->board.raw_column(0), 1u << 22);
    EXPECT_EQ(state->board.raw_column(1), 1u << 3);
    EXPECT_FALSE(state->gameOver);
}

BOT_TEST(jstris, rejects_invalid_state) {
    auto message = state_message();

    message[3] = static_cast<std::byte>(jstris::PROTOCOL_VERSION + 1);
    EXPECT_FALSE(jstris::decode_state(message));

    message = state_message();
    message[5] = std::byte{7};
    EXPECT_FALSE(jstris::decode_state(message));

    message = state_message();
    write_uint32(std::span<std::byte, 4>{message.data() + 12, 4}, 1u << sim::HEIGHT);
    EXPECT_FALSE(jstris::decode_state(message));
}

BOT_TEST(jstris, encodes_moves) {
    std::array moves{
        sim::Instruction::HOLD,
        sim::Instruction::RRIGHT,
        sim::Instruction::LEFT,
        sim::Instruction::PLACE
    };

    auto const message = jstris::encode_moves(moves, jstris::move_timing(std::chrono::milliseconds{80}, 200));

    ASSERT_EQ(message.size(), jstris::MOVE_MESSAGE_HEADER_SIZE + moves.size());
    EXPECT_EQ(message[0], std::byte{'T'});
    EXPECT_EQ(message[1], std::byte{'A'});
    EXPECT_EQ(message[2], std::byte{'3'});
    EXPECT_EQ(message[3], static_cast<std::byte>(jstris::PROTOCOL_VERSION));

    // little-endian uint32: initial delay (200 %), delay, deviation (a third), press duration, press deviation
    constexpr std::array<std::uint8_t, 5> fields{160, 80, 26, 20, 5};
    for(std::size_t field = 0; field < fields.size(); ++field) {
        auto const offset = jstris::MESSAGE_HEADER.size() + field * sizeof(std::uint32_t);
        EXPECT_EQ(message[offset], std::byte{fields[field]}) << "field " << field;
        for(std::size_t i = 1; i < sizeof(std::uint32_t); ++i)
            EXPECT_EQ(message[offset + i], std::byte{0}) << "field " << field;
    }
    for(std::size_t i = 0; i < moves.size(); ++i)
        EXPECT_EQ(message[jstris::MOVE_MESSAGE_HEADER_SIZE + i], static_cast<std::byte>(*moves[i]));
}

}
