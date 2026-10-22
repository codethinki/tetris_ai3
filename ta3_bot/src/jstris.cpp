#include "ta3/bot/jstris.hpp"

#include <cth/io/log.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <limits>
#include <string>
#include <utility>

namespace ta3::bot {
namespace jstris {
    namespace {
        inline constexpr std::array JSTRIS_PIECES{
            sim::PieceType::I,
            sim::PieceType::O,
            sim::PieceType::T,
            sim::PieceType::L,
            sim::PieceType::J,
            sim::PieceType::S,
            sim::PieceType::Z
        };

        inline constexpr std::size_t FLAGS_OFFSET = 4;
        inline constexpr std::size_t CURRENT_OFFSET = 5;
        inline constexpr std::size_t HELD_OFFSET = 6;
        inline constexpr std::size_t LOOKAHEAD_OFFSET = 7;

        [[nodiscard]] std::uint8_t value(std::byte byte) { return std::to_integer<std::uint8_t>(byte); }

        [[nodiscard]] std::optional<sim::PieceType> piece(std::byte byte, bool allowEmpty = false) {
            auto const id = value(byte);
            if(allowEmpty && id == 0xff)
                return sim::NO_PIECE;
            if(id >= JSTRIS_PIECES.size())
                return std::nullopt;
            return JSTRIS_PIECES[id];
        }

        [[nodiscard]] std::uint32_t read_uint32(std::span<std::byte const> message, std::size_t offset) {
            return static_cast<std::uint32_t>(value(message[offset])) |
                static_cast<std::uint32_t>(value(message[offset + 1])) << 8 |
                static_cast<std::uint32_t>(value(message[offset + 2])) << 16 |
                static_cast<std::uint32_t>(value(message[offset + 3])) << 24;
        }

        void write_uint32(std::span<std::byte> message, std::size_t offset, std::uint32_t number) {
            for(std::size_t i = 0; i < sizeof(std::uint32_t); ++i)
                message[offset + i] = static_cast<std::byte>(number >> (i * 8));
        }

        [[nodiscard]] std::string header_string(std::span<std::byte const> message) {
            std::string result;
            for(std::size_t i = 0; i < std::min(message.size(), STATE_HEADER_SIZE); ++i) {
                if(i != 0)
                    result += ' ';
                std::format_to(std::back_inserter(result), "{:02x}", value(message[i]));
            }
            return result;
        }

    }

    void move_timing_t::validate() const {
        auto const valid = [](std::chrono::milliseconds value) {
            return value.count() >= 0 &&
                static_cast<std::uintmax_t>(value.count()) <= std::numeric_limits<std::uint32_t>::max();
        };
        CTH_CRITICAL(!valid(initialDelay), "Jstris initial move delay is outside the protocol range") {}
        CTH_CRITICAL(!valid(delay), "Jstris move delay is outside the protocol range") {}
        CTH_CRITICAL(!valid(deviation), "Jstris move delay deviation is outside the protocol range") {}
        CTH_CRITICAL(!valid(pressDuration), "Jstris key press duration is outside the protocol range") {}
        CTH_CRITICAL(!valid(pressDeviation), "Jstris key press deviation is outside the protocol range") {}
    }

    move_timing_t move_timing(std::chrono::milliseconds delay, std::uint32_t initialDelayPercent) {
        return {
            .initialDelay = delay * initialDelayPercent / 100,
            .delay = delay,
            .deviation = delay / DELAY_DEVIATION_DIVISOR,
            .pressDuration = PRESS_DURATION,
            .pressDeviation = PRESS_DEVIATION
        };
    }

    std::optional<sim::TetrisState> decode_state(std::span<std::byte const> message) {
        if(message.size() != STATE_MESSAGE_SIZE ||
            !std::ranges::equal(message.first(MESSAGE_HEADER.size()), MESSAGE_HEADER) ||
            (value(message[FLAGS_OFFSET]) & ~1u) != 0)
            return std::nullopt;

        auto const current = piece(message[CURRENT_OFFSET]);
        auto const held = piece(message[HELD_OFFSET], true);
        if(!current || !held)
            return std::nullopt;

        sim::TetrisState state{
            .current = *current,
            .held = *held,
            .gameOver = (value(message[FLAGS_OFFSET]) & 1u) != 0
        };

        for(std::size_t i = 0; i < state.lookahead.size(); ++i) {
            auto const next = piece(message[LOOKAHEAD_OFFSET + i]);
            if(!next)
                return std::nullopt;
            state.lookahead[i] = *next;
        }

        std::array<std::uint32_t, sim::WIDTH> columns{};
        for(std::size_t x = 0; x < columns.size(); ++x) {
            auto const offset = STATE_HEADER_SIZE + x * sizeof(std::uint32_t);
            columns[x] = read_uint32(message, offset);
            if((columns[x] >> sim::HEIGHT) != 0)
                return std::nullopt;
        }
        state.board = sim::Board2::from_raw_columns(columns);
        return state;
    }

    std::vector<std::byte> encode_moves(
        std::span<sim::Instruction const> moves,
        move_timing_t const& timing
    ) {
        std::vector<std::byte> message(MOVE_MESSAGE_HEADER_SIZE + moves.size());
        std::ranges::copy(MESSAGE_HEADER, message.begin());

        std::array const fields{
            timing.initialDelay,
            timing.delay,
            timing.deviation,
            timing.pressDuration,
            timing.pressDeviation
        };
        static_assert(MESSAGE_HEADER.size() + fields.size() * sizeof(std::uint32_t) == MOVE_MESSAGE_HEADER_SIZE);
        for(std::size_t i = 0; i < fields.size(); ++i) {
            auto const offset = MESSAGE_HEADER.size() + i * sizeof(std::uint32_t);
            write_uint32(message, offset, static_cast<std::uint32_t>(fields[i].count()));
        }

        for(std::size_t i = 0; i < moves.size(); ++i) {
            CTH_CRITICAL(
                moves[i] == sim::Instruction::NONE || moves[i] == sim::Instruction::SIZE,
                "Jstris move sequence contains an invalid instruction"
            ) {}
            message[MOVE_MESSAGE_HEADER_SIZE + i] = static_cast<std::byte>(*moves[i]);
        }
        return message;
    }

}

Jstris::Jstris(std::uint16_t port) : _server{port, jstris::STATE_MESSAGE_SIZE} {}

auto Jstris::connect() -> cth::co::executor_task<bool> {
    while(!_server.stopped()) {
        auto const pending = _server.fence().signaled();
        if(!pending) {
            if(connected())
                co_return true;
            _connection.reset();
            cth::log::msg<cth::except::INFO>(
                "ta3_bot: waiting for Jstris bridge on ws://127.0.0.1:{}",
                _server.port().value_or(0)
            );
        }

        auto next = co_await _server.accept();
        if(!next)
            co_return false;
        if(!next->connected())
            continue;

        if(_connection)
            _connection->close();
        _connection = std::move(*next);
        cth::log::msg<cth::except::INFO>("ta3_bot: Jstris bridge connected");
        co_return true;
    }

    co_return false;
}

auto Jstris::next_board() -> cth::co::executor_task<std::optional<sim::TetrisState>> {
    while(!_server.stopped()) {
        if(!(co_await connect()))
            co_return std::nullopt;

        _connection->write_bytes(jstris::MESSAGE_HEADER);
        if(!(co_await _connection->wait())) {
            if(!_server.stopped())
                cth::log::msg<cth::except::WARNING>("ta3_bot: Jstris bridge disconnected");
            _connection.reset();
            continue;
        }

        if(_connection->got_text()) {
            [[maybe_unused]] auto const ignored = _connection->take_text();
            cth::log::msg<cth::except::WARNING>("ta3_bot: ignored text message from Jstris bridge");
            continue;
        }

        auto const message = _connection->take_bytes();
        auto state = jstris::decode_state(message);
        if(!state) {
            cth::log::msg<cth::except::WARNING>(
                "ta3_bot: ignored invalid Jstris state ({} bytes, header: {})",
                message.size(),
                jstris::header_string(message)
            );
            continue;
        }
        if(!_connection->connected()) {
            _connection.reset();
            continue;
        }
        co_return state;
    }

    co_return std::nullopt;
}

auto Jstris::move(
    std::vector<sim::Instruction> moves,
    jstris::move_timing_t timing
) -> cth::co::executor_task<bool> {
    if(!_connection || !_connection->connected())
        co_return false;

    _connection->write_bytes(jstris::encode_moves(moves, timing));
    co_return true;
}

void Jstris::stop() const noexcept { _server.stop(); }

bool Jstris::connected() const noexcept { return _connection && _connection->connected(); }

}
