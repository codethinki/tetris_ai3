#include "ta3/gpu/model_eval.hpp"

#include <ta3/gpu/detail/value_model.hpp>

#include "ta3/ai/search/beam.hpp"

#include <ta3/ai/models/metrics/tetris_stats_v4.hpp>

#include <ta3/sim/tetris_engine.hpp>

#include <cstdint>

/**
 * @file model_eval_host.cpp
 * @brief the CPU parity reference -- the exact beamed search the GPU kernel runs, via the shared
 *  constexpr core.
 * @details @ref ta3::ai::search::search_move_beam walks the identical slot spaces with the identical
 *  selection rule and float math as the device kernel of whichever backend is linked in, so a game played
 *  here matches the kernel move for move (same beam widths, same tie-breaks, same fallbacks).
 */
namespace ta3::gpu {

namespace search = ta3::ai::search;
namespace sim = ta3::sim;

std::vector<float> ModelEvaluator::eval_host(
    std::span<float const> weights,
    std::span<std::uint64_t const> seeds,
    std::uint32_t max_moves
) {
    // same value model as the kernel; here the weights live in the caller's span.
    net_ref const model{weights.subspan<0, ai::model_t::NUM_PARAMS>()};

    std::vector<float> out(seeds.size(), 0.0f);
    for(std::size_t g = 0; g < seeds.size(); ++g) {
        sim::TetrisEngine game{seeds[g]};
        ai::stats_v4 stats{}; // committed on every move, scored at the end -- identical to the kernel

        for(std::uint32_t i = 0; i < max_moves && !game.gameOver(); ++i) {
            auto const r = search::search_move_beam(game, model);
            if(r.none())
                break;

            if(r.move.hold)
                game.hold();
            auto const cleared = game.place(r.move.orientation, r.move.x);
            if(cleared == sim::TetrisEngine::DIED)
                break;
            stats.advance(game.board(), static_cast<std::uint32_t>(cleared));
        }
        out[g] = stats.piecesPlaced() == 0 ? -1.0e6f : static_cast<float>(stats.score());
    }
    return out;
}

} // namespace ta3::gpu
