#pragma once

#include <cth/coro/tasks/executor_task.hpp>
#include <cth/net/coro/websocket.hpp>
#include <ta3/sim/tetris_state.hpp>

#include <array>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ta3::bot {

namespace jstris {

inline constexpr std::uint16_t WEBSOCKET_PORT = 18473;
inline constexpr std::uint8_t PROTOCOL_VERSION = 6;
inline constexpr std::array<std::byte, 4> MESSAGE_HEADER{
    std::byte{'T'},
    std::byte{'A'},
    std::byte{'3'},
    static_cast<std::byte>(PROTOCOL_VERSION)
};
inline constexpr std::size_t VISIBLE_ROWS = 20;
inline constexpr std::size_t HIDDEN_ROWS = sim::HEIGHT - VISIBLE_ROWS;
inline constexpr std::size_t STATE_HEADER_SIZE = 12;
inline constexpr std::size_t STATE_MESSAGE_SIZE = STATE_HEADER_SIZE + sim::WIDTH * sizeof(std::uint32_t);
inline constexpr std::size_t BOARD_REQUEST_SIZE = MESSAGE_HEADER.size();
inline constexpr std::size_t MOVE_MESSAGE_HEADER_SIZE = MESSAGE_HEADER.size() + 5 * sizeof(std::uint32_t);

inline constexpr std::chrono::milliseconds PRESS_DURATION{20};
inline constexpr std::chrono::milliseconds PRESS_DEVIATION{5};
/** every delay's standard deviation is this fraction of the delay */
inline constexpr int DELAY_DEVIATION_DIVISOR = 3;

/** normally distributed timing sent with every move sequence */
struct move_timing_t {
    /** from the last hard drop to the first action */
    std::chrono::milliseconds initialDelay;
    /** between two actions */
    std::chrono::milliseconds delay;
    /** standard deviation of both delays */
    std::chrono::milliseconds deviation;
    std::chrono::milliseconds pressDuration;
    std::chrono::milliseconds pressDeviation;

    /** every value must fit the protocol's uint32 millisecond fields */
    void validate() const;
};

/** derives every timing from the delay between two actions */
[[nodiscard]] move_timing_t move_timing(std::chrono::milliseconds delay, std::uint32_t initialDelayPercent);

/** decodes one binary state frame emitted by jstris.user.js */
[[nodiscard]] std::optional<sim::TetrisState> decode_state(std::span<std::byte const> message);

/** encodes one ordered list of Jstris actions and its normally distributed per-action delay */
[[nodiscard]] std::vector<std::byte> encode_moves(
    std::span<sim::Instruction const> moves,
    move_timing_t const& timing
);

}

/** coroutine interface to one Jstris userscript connection */
class Jstris {
public:
    explicit Jstris(std::uint16_t port = jstris::WEBSOCKET_PORT);

    /** waits through rejected or closed clients until one bridge is connected; false after stop() */
    [[nodiscard]] auto connect() -> cth::co::executor_task<bool>;

    /** waits through reconnects until one valid board arrives; nullopt after stop() */
    [[nodiscard]] auto next_board() -> cth::co::executor_task<std::optional<sim::TetrisState>>;

    /** sends an ordered action sequence; false when the bridge disconnected before the send */
    [[nodiscard]] auto move(
        std::vector<sim::Instruction> moves,
        jstris::move_timing_t timing
    ) -> cth::co::executor_task<bool>;

    void stop() const noexcept;

    [[nodiscard]] bool connected() const noexcept;

private:
    cth::net::co::websocket_server _server;
    std::optional<cth::net::co::websocket_connection> _connection;
};

}
