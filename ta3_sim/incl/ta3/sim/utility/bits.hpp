#pragma once
#include <bit>
#include <cstdint>
#include <type_traits>

/**
 * @file bits.hpp
 * @brief device-safe <bit> wrappers -- REQUIRED for any bit op reachable from device code.
 *
 * MSVC's std::countr_zero / std::popcount / std::bit_width dispatch through host ISA-detection globals
 * (@c __isa_available) at runtime. compiled for the device under @c --expt-relaxed-constexpr that global
 * read is undefined and the results are garbage -- observed on sm_89: countr_zero == 0 and popcount == 0
 * for EVERY input, which silently emptied every board (pieces "landed" nowhere, games never ended, all
 * fitness collapsed to one constant). libstdc++/libc++ lower to compiler builtins and are fine, which is
 * why Linux/clang builds never showed it.
 *
 * on the device these use the hardware intrinsics (__ffs/__popc/__clz); in constant evaluation (LUT
 * builders, static_asserts) and on the host they use std::, which is correct everywhere at compile time.
 *
 * AdaptiveCpp's generic SSCP compiles device (and host) code with its bundled clang++, which on Windows
 * still consumes the MSVC STL headers -- so the same __isa_available landmine can be reached from SYCL
 * kernels. the @c __clang__ branch below re-arms the fix with clang's own builtins, gated on
 * @c !is_constant_evaluated() exactly like the CUDA branch, so constant evaluation still goes through std::.
 */
namespace ta3::sim {

[[nodiscard]] constexpr int ctz(std::uint32_t v) {
#if defined(__CUDA_ARCH__)
    if(!std::is_constant_evaluated())
        return v == 0 ? 32 : __ffs(static_cast<int>(v)) - 1;
#elif defined(__clang__)
    if(!std::is_constant_evaluated())
        return v == 0 ? 32 : __builtin_ctz(v); // __builtin_ctz(0) is UB -- guard explicitly
#endif
    return std::countr_zero(v);
}

[[nodiscard]] constexpr int popcount(std::uint32_t v) {
#if defined(__CUDA_ARCH__)
    if(!std::is_constant_evaluated())
        return __popc(static_cast<int>(v));
#elif defined(__clang__)
    if(!std::is_constant_evaluated())
        return __builtin_popcount(v);
#endif
    return std::popcount(v);
}

[[nodiscard]] constexpr int bit_width32(std::uint32_t v) {
#if defined(__CUDA_ARCH__)
    if(!std::is_constant_evaluated())
        return 32 - __clz(static_cast<int>(v));
#elif defined(__clang__)
    if(!std::is_constant_evaluated())
        return v == 0 ? 0 : 32 - __builtin_clz(v); // __builtin_clz(0) is UB -- guard explicitly (CUDA's __clz(0)==32 has no clang equivalent)
#endif
    return static_cast<int>(std::bit_width(v));
}

} // namespace ta3::sim
