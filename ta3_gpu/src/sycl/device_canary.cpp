#include "ta3/gpu/device_canary.hpp"

#include <ta3/sim/utility/bits.hpp>

#include <sycl/sycl.hpp>

#include <cstdint>

/**
 * @file device_canary.cpp
 * @brief device-side smoke test for @c ta3::sim::ctz/popcnt/bit_width32.
 * @details guards a silent hazard: MSVC's bit intrinsics dispatch through a host ISA-detection global
 *  that reads as garbage under a device compiler without MSVC CRT init, producing empty boards and
 *  constant fitness with no compiler error (see ta3_sim/incl/ta3/sim/utility/bits.hpp's __clang__ branch).
 */
namespace ta3::gpu {

namespace {

    constexpr std::uint32_t CTZ_INPUT = 0b1000u; // trailing zero count == 3
    constexpr int CTZ_EXPECT = 3;

    constexpr std::uint32_t POPCNT_INPUT = 0b1011'0110u; // 5 bits set
    constexpr int POPCNT_EXPECT = 5;

    constexpr std::uint32_t BIT_WIDTH_INPUT = 0b0010'1000u; // highest set bit at position 5 -> width 6
    constexpr int BIT_WIDTH_EXPECT = 6;

} // namespace

bool device_bits_canary() {
    sycl::queue q{sycl::default_selector_v};

    int* const result = sycl::malloc_shared<int>(1, q);
    if(result == nullptr)
        return false;
    *result = 0;

    q.single_task([=]() {
        bool ok = true;
        ok = ok && (ta3::sim::ctz(CTZ_INPUT) == CTZ_EXPECT);
        ok = ok && (ta3::sim::popcnt(POPCNT_INPUT) == POPCNT_EXPECT);
        ok = ok && (ta3::sim::bit_width32(BIT_WIDTH_INPUT) == BIT_WIDTH_EXPECT);
        *result = ok ? 1 : 0;
    });
    q.wait();

    bool const pass = (*result == 1);
    sycl::free(result, q);
    return pass;
}

} // namespace ta3::gpu
