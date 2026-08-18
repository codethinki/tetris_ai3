#pragma once

/**
 * @file device_canary.hpp
 * @brief zero-GPU-types declaration of the device bit-intrinsics smoke test.
 */
namespace ta3::gpu {

/**
 * @brief runs a trivial on-device check of ta3::sim::ctz/popcnt/bit_width32 against known inputs.
 * @details guards a silent MSVC-STL ISA-detection hazard reachable from device code (see
 *  ta3_sim/incl/ta3/sim/utility/bits.hpp); a regression here produces no build error, just empty boards
 *  and constant fitness at runtime.
 * @return true iff every checked intrinsic matched its expected value on the selected device.
 */
[[nodiscard]] bool device_bits_canary();

} // namespace ta3::gpu
