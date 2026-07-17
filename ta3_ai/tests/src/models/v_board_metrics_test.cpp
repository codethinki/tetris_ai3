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
        if(metric::board_metric<metric::v::holes>(board) != metric::holes(board))
            return false;
        if(metric::board_metric<metric::v::agg_height>(board) != metric::agg_height(board))
            return false;
        if(metric::board_metric<metric::v::max_height>(board) != metric::max_height(board))
            return false;
        if(metric::board_metric<metric::v::surface_variance>(board) != metric::surface_variance(board))
            return false;
        if(metric::board_metric<metric::v::y_transitions>(board) != metric::y_transitions(board))
            return false;
        if(metric::board_metric<metric::v::hole_depths>(board) != metric::hole_depths(board))
            return false;
        if(metric::board_metric<metric::v::bumpiness>(board) != metric::bumpiness(board))
            return false;
        if(metric::board_metric<metric::v::wells>(board) != metric::wells(board))
            return false;

        metric::fuse_v_board_metrics<
            metric::v::holes,
            metric::v::agg_height,
            metric::v::max_height,
            metric::v::surface_variance,
            metric::v::y_transitions,
            metric::v::hole_depths,
            metric::v::bumpiness,
            metric::v::wells
        > const fused{board};

        if(fused.get(metric::v::holes{}) != metric::board_metric<metric::v::holes>(board))
            return false;
        if(fused.get(metric::v::agg_height{}) != metric::board_metric<metric::v::agg_height>(board))
            return false;
        if(fused.get(metric::v::max_height{}) != metric::board_metric<metric::v::max_height>(board))
            return false;
        if(fused.get(metric::v::surface_variance{}) != metric::board_metric<metric::v::surface_variance>(board))
            return false;
        if(fused.get(metric::v::y_transitions{}) != metric::board_metric<metric::v::y_transitions>(board))
            return false;
        if(fused.get(metric::v::hole_depths{}) != metric::board_metric<metric::v::hole_depths>(board))
            return false;
        if(fused.get(metric::v::bumpiness{}) != metric::board_metric<metric::v::bumpiness>(board))
            return false;
        if(fused.get(metric::v::wells{}) != metric::board_metric<metric::v::wells>(board))
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
