#include "ta3/ai/models/metrics/v_board_metrics.hpp"

#include <ta3/sim/board2.hpp>

#include "test.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#define VMET_TEST(suite, name) CTH_EX_TEST(_ai_models, suite, name)

namespace ta3::ai {

namespace {

    bool board_matches(sim::Board2 const& board) {
        if(metric::board_metric<metric::board_holes_t>(board) != metric::board_holes(board))
            return false;
        if(metric::board_metric<metric::board_agg_height_t>(board) != metric::board_agg_height(board))
            return false;
        if(metric::board_metric<metric::board_max_height_t>(board) != metric::board_max_height(board))
            return false;
        if(metric::board_metric<metric::board_surface_var_t>(board) != metric::board_surface_var(board))
            return false;
        if(metric::board_metric<metric::board_y_transitions_t>(board) != metric::board_y_transitions(board))
            return false;
        if(metric::board_metric<metric::board_hole_depths_t>(board) != metric::board_hole_depths(board))
            return false;
        if(metric::board_metric<metric::board_bumpiness_t>(board) != metric::board_bumpiness(board))
            return false;
        if(metric::board_metric<metric::board_wells_t>(board) != metric::board_wells(board))
            return false;

        metric::fuse_v_board_metrics<
            metric::board_holes_t,
            metric::board_agg_height_t,
            metric::board_max_height_t,
            metric::board_surface_var_t,
            metric::board_y_transitions_t,
            metric::board_hole_depths_t,
            metric::board_bumpiness_t,
            metric::board_wells_t
        > const fused{board};

        if(fused.get(metric::board_holes_t{}) != metric::board_metric<metric::board_holes_t>(board))
            return false;
        if(fused.get(metric::board_agg_height_t{}) != metric::board_metric<metric::board_agg_height_t>(board))
            return false;
        if(fused.get(metric::board_max_height_t{}) != metric::board_metric<metric::board_max_height_t>(board))
            return false;
        if(fused.get(metric::board_surface_var_t{}) != metric::board_metric<metric::board_surface_var_t>(board))
            return false;
        if(fused.get(metric::board_y_transitions_t{}) != metric::board_metric<metric::board_y_transitions_t>(board))
            return false;
        if(fused.get(metric::board_hole_depths_t{}) != metric::board_metric<metric::board_hole_depths_t>(board))
            return false;
        if(fused.get(metric::board_bumpiness_t{}) != metric::board_metric<metric::board_bumpiness_t>(board))
            return false;
        if(fused.get(metric::board_wells_t{}) != metric::board_metric<metric::board_wells_t>(board))
            return false;

        return true;
    }

} // namespace

VMET_TEST(v_board_metrics, parity_vs_legacy_and_fusion) {
    constexpr auto high = (std::uint32_t{1} << sim::HEIGHT) - 1;
    std::array const boards{
        sim::Board2{},
        sim::Board2::from_raw_columns({1, 3, 7, 15, 31, 63, 127, 255, 511, 1023}),
        sim::Board2::from_raw_columns({0b10101, 0b11111, 0b10001, 0b11101, 0, 0b10111, 0b11111, 1, 3, 7}),
        sim::Board2::from_raw_columns({high, high >> 1, high >> 2, high >> 3, 0, 0, high >> 3, high >> 2, high >> 1, high})
    };

    for(auto const& board : boards)
        EXPECT_TRUE(board_matches(board));
}

} // namespace ta3::ai
