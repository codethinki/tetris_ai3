#pragma once
#include <array>
#include <cstdint>

namespace ta3::sim {

/**
 * @brief xoshiro256** -- a small, fast, constexpr-evaluable PRNG
 * @note models @c std::uniform_random_bit_generator, so it also drives the std distributions in
 *  non-constexpr code; the state is seeded through splitmix64
 */
class Xoshiro256ss {
public:
    using result_type = uint64_t;

    constexpr Xoshiro256ss() { seed(0); }
    constexpr explicit Xoshiro256ss(uint64_t s) { seed(s); }

    /** @brief reseeds the whole state from @ref s via splitmix64 */
    constexpr void seed(uint64_t s) {
        for(auto& word : _state) {
            s += 0x9e3779b97f4a7c15ULL;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            word = z ^ (z >> 31);
        }
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }

    constexpr result_type operator()() {
        result_type const result = rotl(_state[1] * 5, 7) * 9;
        result_type const t = _state[1] << 17;

        _state[2] ^= _state[0];
        _state[3] ^= _state[1];
        _state[1] ^= _state[2];
        _state[0] ^= _state[3];
        _state[2] ^= t;
        _state[3] = rotl(_state[3], 45);

        return result;
    }

    /** @brief a uniform integer in [0, @ref bound); modulo bias is negligible for the small bounds we use */
    constexpr result_type bounded(result_type bound) { return (*this)() % bound; }

    friend constexpr bool operator==(Xoshiro256ss const&, Xoshiro256ss const&) = default;

private:
    static constexpr result_type rotl(result_type x, int k) { return (x << k) | (x >> (64 - k)); }

    std::array<result_type, 4> _state{};
};

}
