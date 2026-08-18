#include "ta3/ai/models/metrics/v_board_metrics.hpp"

#include "ta3/ai/search/placements.hpp"
#include "ta3/ai/search/search.hpp"
#include "ta3/ai/search/variation_sequences.hpp"

#include <ta3/sim/board2.hpp>
#include <ta3/sim/tetris_engine.hpp>

#include "test.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#define VMET_TEST(suite, name) CTH_EX_TEST(_ai_models, suite, name)

namespace ta3::ai {

namespace {

    struct height_model {
        constexpr search::value_t operator()(search::clear_t accum, sim::Board2 const& b) const {
            std::uint32_t agg = 0;
            for(std::size_t x = 0; x < sim::WIDTH; ++x)
                agg += static_cast<std::uint32_t>(b.height(x));

            return 2.0f * static_cast<search::value_t>(accum.lines())
                - 0.3f * static_cast<search::value_t>(agg)
                - 1.0f * static_cast<search::value_t>(b.holes());
        }

        constexpr search::value_t evaluate(search::clear_t accum, sim::Board2 const& b, bool /*heldIsI*/) const {
            return (*this)(accum, b);
        }
    };

    bool board_matches(sim::Board2 const& board) {
        if(metric::board_metric<metric::holes_t>(board) != metric::holes(board))
            return false;
        if(metric::board_metric<metric::agg_height_t>(board) != metric::agg_height(board))
            return false;
        if(metric::board_metric<metric::max_height_t>(board) != metric::max_height(board))
            return false;
        if(metric::board_metric<metric::surface_variance_t>(board) != metric::surface_variance(board))
            return false;
        if(metric::board_metric<metric::y_transitions_t>(board) != metric::y_transitions(board))
            return false;
        if(metric::board_metric<metric::hole_depths_t>(board) != metric::hole_depths(board))
            return false;
        if(metric::board_metric<metric::bumpiness_t>(board) != metric::bumpiness(board))
            return false;
        if(metric::board_metric<metric::wells_t>(board) != metric::wells(board))
            return false;

        metric::fuse_v_board_metrics<
            metric::holes_t,
            metric::agg_height_t,
            metric::max_height_t,
            metric::surface_variance_t,
            metric::y_transitions_t,
            metric::hole_depths_t,
            metric::bumpiness_t,
            metric::wells_t
        > const fused{board};

        if(fused.get(metric::holes_t{}) != metric::board_metric<metric::holes_t>(board))
            return false;
        if(fused.get(metric::agg_height_t{}) != metric::board_metric<metric::agg_height_t>(board))
            return false;
        if(fused.get(metric::max_height_t{}) != metric::board_metric<metric::max_height_t>(board))
            return false;
        if(fused.get(metric::surface_variance_t{}) != metric::board_metric<metric::surface_variance_t>(board))
            return false;
        if(fused.get(metric::y_transitions_t{}) != metric::board_metric<metric::y_transitions_t>(board))
            return false;
        if(fused.get(metric::hole_depths_t{}) != metric::board_metric<metric::hole_depths_t>(board))
            return false;
        if(fused.get(metric::bumpiness_t{}) != metric::board_metric<metric::bumpiness_t>(board))
            return false;
        if(fused.get(metric::wells_t{}) != metric::board_metric<metric::wells_t>(board))
            return false;

        return true;
    }

    bool run_checks() {
        constexpr std::array<std::uint64_t, 4> seeds{1234, 0xC0FFEE, 0xBADF00D, 7};
        height_model const model;

        bool ok = true;
        for(auto const seed : seeds) {
            sim::TetrisEngine game{seed};

            for(std::uint32_t moves = 0; moves < 30 && !game.gameOver(); ++moves) {
                auto const r = search::search_move(game, model);
                if(r.none())
                    break;

                if(r.move.hold)
                    game.hold();
                if(game.place(r.move.orientation, r.move.x) == sim::TetrisEngine::DIED)
                    break;

                if(!board_matches(game.board()))
                    ok = false;
            }
        }
        return ok;
    }

} // namespace

VMET_TEST(v_board_metrics, parity_vs_legacy_and_fusion) {
    EXPECT_TRUE(run_checks());
}

} // namespace ta3::ai
