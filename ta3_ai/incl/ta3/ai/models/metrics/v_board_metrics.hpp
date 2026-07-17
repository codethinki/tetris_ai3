#pragma once
#include "ta3/ai/model_defs.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/utility/bits.hpp>
#include <ta3/sim/utility/cuda_constant.hpp>
#include <ta3/sim/utility/tetris_defs.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ta3::ai::metric {

[[nodiscard]] constexpr float fsqrt(float x) {
    if(x <= 0)
        return 0;
    if(std::is_constant_evaluated()) {
        float c = x, p = 0;
        for(auto i = 0; i < 32 && c != p; ++i) {
            p = c;
            c = 0.5f * (c + x / c);
        }
        return c;
    }
    return std::sqrt(x);
}

namespace dev {

    TA3_CUDA_CONSTANT auto HOLE_DEPTH_PLANES = [] {
        constexpr std::size_t bitWidth = std::bit_width(static_cast<unsigned>(sim::HEIGHT - 1));
        std::array<std::uint32_t, bitWidth> planes{};
        for(std::size_t b = 0; b < bitWidth; ++b)
            for(std::uint32_t y = 0; y < sim::HEIGHT; ++y)
                if((y >> b) & 1u)
                    planes[b] |= std::uint32_t{1} << y;
        return planes;
    }();

    [[nodiscard]] constexpr std::size_t sum_set_bit_indices(std::uint32_t c) {
        std::size_t s = 0;
        for(std::size_t b = 0; b < HOLE_DEPTH_PLANES.size(); ++b)
            s += (std::size_t{1} << b) * static_cast<std::size_t>(sim::popcount(c & HOLE_DEPTH_PLANES[b]));
        return s;
    }

} // namespace dev

struct col_ctx {
    sim::Board2 const& board;
    std::size_t x;
};

struct v_board_metric_base {};

template<class T>
concept v_board_metric = std::derived_from<T, v_board_metric_base>
    && std::default_initializable<T>
    && requires(T m, T const cm, col_ctx const& c) {
        { m.step(c) };
        { cm.get() };
    };

namespace dev {

    template<class V, class... Vs>
    [[nodiscard]] consteval std::size_t index_of() {
        std::size_t i = 0;
        std::size_t found = sizeof...(Vs);
        (((std::is_same_v<V, Vs> && found == sizeof...(Vs)) ? (found = i) : found, ++i), ...);
        return found;
    }

    template<v_board_metric... Vs>
    [[nodiscard]] constexpr auto run_v_metrics(sim::Board2 const& board) {
        std::tuple<Vs...> steppers{};
        for(std::size_t x = 0; x < sim::WIDTH; ++x) {
            col_ctx const c{board, x};
            std::apply([&](Vs&... v) { (v.step(c), ...); }, steppers);
        }
        return std::apply([](Vs const&... v) { return std::tuple{v.get()...}; }, steppers);
    }

} // namespace dev

template<v_board_metric... Vs>
class fuse_v_board_metrics {
    using results_t = decltype(dev::run_v_metrics<Vs...>(std::declval<sim::Board2 const&>()));
    results_t _results;

public:
    constexpr explicit fuse_v_board_metrics(sim::Board2 const& board)
        : _results{dev::run_v_metrics<Vs...>(board)} {}

    template<class V>
    [[nodiscard]] constexpr auto get() const { return std::get<dev::index_of<V, Vs...>()>(_results); }
    template<class V>
    [[nodiscard]] constexpr auto get(V const&) const { return get<V>(); }
};

template<v_board_metric V>
inline constexpr auto board_metric = [](sim::Board2 const& board) {
    return fuse_v_board_metrics<V>{board}.template get<V>();
};

} // namespace ta3::ai::metric


namespace ta3::ai::metric::v {

struct holes : v_board_metric_base {
    std::size_t sum = 0;
    constexpr void step(col_ctx const& c) {
        sum += c.board.height(c.x) - static_cast<std::size_t>(sim::popcount(c.board.raw_column(c.x)));
    }
    [[nodiscard]] constexpr std::size_t get() const { return sum; }
};

struct agg_height : v_board_metric_base {
    std::size_t sum = 0;
    constexpr void step(col_ctx const& c) { sum += c.board.height(c.x); }
    [[nodiscard]] constexpr std::size_t get() const { return sum; }
};

struct max_height : v_board_metric_base {
    std::size_t max = 0;
    constexpr void step(col_ctx const& c) { max = std::max(max, c.board.height(c.x)); }
    [[nodiscard]] constexpr std::size_t get() const { return max; }
};

struct surface_variance : v_board_metric_base {
    static constexpr auto INVALID = std::numeric_limits<std::size_t>::max();
    std::size_t prevHeight = INVALID;
    int sum = 0;
    constexpr void step(col_ctx const& c) {
        auto const h = c.board.height(c.x);
        if(prevHeight != INVALID) {
            int const diff = static_cast<int>(h) - static_cast<int>(prevHeight);
            sum += diff * diff;
        }
        prevHeight = h;
    }
    [[nodiscard]] constexpr data_t get() const { return fsqrt(static_cast<float>(sum)); }
};

struct y_transitions : v_board_metric_base {
    static constexpr std::uint32_t FIELD = (std::uint32_t{1} << sim::HEIGHT) - 1;
    std::size_t sum = 0;
    constexpr void step(col_ctx const& c) {
        auto const col = c.board.raw_column(c.x) | (std::uint32_t{1} << sim::HEIGHT);
        sum += static_cast<std::size_t>(sim::popcount((col ^ (col >> 1)) & FIELD));
    }
    [[nodiscard]] constexpr std::size_t get() const { return sum; }
};

struct hole_depths : v_board_metric_base {
    std::size_t sum = 0;
    constexpr void step(col_ctx const& c) {
        constexpr auto h = sim::HEIGHT;
        auto const col = c.board.raw_column(c.x);
        auto const n = static_cast<std::size_t>(sim::popcount(col));
        auto const s = dev::sum_set_bit_indices(col);
        auto const pos = (h - 1) * n + n * (n + 1) / 2;
        auto const neg = s + n * n;
        sum += pos - neg;
    }
    [[nodiscard]] constexpr std::size_t get() const { return sum; }
};

struct bumpiness : v_board_metric_base {
    static constexpr auto INVALID = std::numeric_limits<std::size_t>::max();
    std::size_t prevHeight = INVALID;
    std::size_t sum = 0;
    constexpr void step(col_ctx const& c) {
        auto const h = c.board.height(c.x);
        if(prevHeight != INVALID)
            sum += h > prevHeight ? h - prevHeight : prevHeight - h;
        prevHeight = h;
    }
    [[nodiscard]] constexpr std::size_t get() const { return sum; }
};

struct wells : v_board_metric_base {
    static constexpr std::size_t WELL_DEPTH = 3;
    std::size_t count = 0;
    constexpr void step(col_ctx const& c) {
        auto const& b = c.board;
        auto const x = c.x;
        if(x == 0) {
            if(b.height(1) >= b.height(0) + WELL_DEPTH)
                ++count;
        }
        else if(x + 1 == sim::WIDTH) {
            if(b.height(sim::WIDTH - 2) >= b.height(sim::WIDTH - 1) + WELL_DEPTH)
                ++count;
        }
        else {
            auto const h = b.height(x);
            if(b.height(x - 1) >= h + WELL_DEPTH && b.height(x + 1) >= h + WELL_DEPTH)
                ++count;
        }
    }
    [[nodiscard]] constexpr std::size_t get() const { return count; }
};

} // namespace ta3::ai::metric::v


namespace ta3::ai::metric {

inline constexpr auto holes = board_metric<v::holes>;
inline constexpr auto agg_height = board_metric<v::agg_height>;
inline constexpr auto max_height = board_metric<v::max_height>;
inline constexpr auto surface_variance = board_metric<v::surface_variance>;
inline constexpr auto y_transitions = board_metric<v::y_transitions>;
inline constexpr auto hole_depths = board_metric<v::hole_depths>;
inline constexpr auto bumpiness = board_metric<v::bumpiness>;
inline constexpr auto wells = board_metric<v::wells>;

} // namespace ta3::ai::metric
