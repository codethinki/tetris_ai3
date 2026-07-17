#pragma once

#include "ta3/ai/model_defs.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>

namespace ta3::ai {

/** @brief parameter count of one fully-connected layer: an In*Out weight matrix plus Out biases */
[[nodiscard]] constexpr size_t fc_layer_params(size_t in, size_t out) { return in * out + out; }

/**
 * @brief one fully-connected layer, fully sized at compile time
 * @tparam In   input width
 * @tparam Out  output width
 * @tparam Relu apply relu (else linear output)
 * @param in    input activations, by direct array reference so layers chain cleanly
 * @param w     this layer's slice, fixed at Out*In weight matrix + Out biases
 * @return the Out output activations
 * @note weight layout (weights then biases) matches the stored champion files
 * @note activations are std::array, NEVER std::span: spans are pointer objects, and routing the local
 *  activation arrays through one blocked SROA -- on the device that spilled every layer's in/out to
 *  local memory inside the leaf loop. the weights stay a span on purpose: they alias a real external
 *  buffer (device: __shared__), so a pointer is exactly right there.
 */
template<size_t In, size_t Out, bool Relu>
[[nodiscard]] constexpr std::array<data_t, Out> dense(
    std::array<data_t, In> const& in,
    std::span<data_t const, In * Out + Out> w
) {
    std::array<data_t, Out> out{};
// restrict to avoid register pressure
#if defined(__CUDA_ARCH__)
#pragma unroll 4
#elif defined(__clang__)
#pragma clang loop unroll_count(4)
#endif
    for(size_t o = 0; o < Out; ++o) {
        auto v = w[In * Out + o]; // bias, stored after the matrix
        for(size_t i = 0; i < In; ++i)
            v += in[i] * w[o * In + i];
        out[o] = Relu ? std::max<data_t>(0, v) : v;
    }
    return out;
}

} // namespace ta3::ai
