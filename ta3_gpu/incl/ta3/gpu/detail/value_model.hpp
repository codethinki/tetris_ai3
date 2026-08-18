#pragma once
#include <ta3/ai/model.hpp>
#include <ta3/ai/search/placements.hpp> // clear_t, value_t

#include <ta3/sim/board2.hpp>

/**
 * @file value_model.hpp
 * @brief the search seam over @ref ta3::ai::model_t: reads weights from an external span (a host array,
 *  or the kernel's shared/local staging buffer) instead of owning them.
 */
namespace ta3::gpu {

struct net_ref {
    ai::model_t::weights_t w; ///< NUM_PARAMS weights (host array or on-device shared memory)

    [[nodiscard]] constexpr ai::search::value_t evaluate(
        ai::search::clear_t clears, sim::Board2 const& board, bool held_is_i) const {
        return static_cast<ai::search::value_t>(ai::model_t::evaluate(clears, board, held_is_i, w));
    }
};

} // namespace ta3::gpu
