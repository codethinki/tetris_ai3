#include "ta3/ai/models/models/model_v8.hpp"

#include <ta3/sim/board2.hpp>

#include "test.hpp"

#include <array>
#include <cstddef>

#define MODEL_V8_TEST(suite, name) CTH_EX_TEST(_ai_models, suite, name)

namespace ta3::ai {

namespace {

    // ModelV8 adds exactly one input lane (HELD_I, the last of INPUTS) over the v7 feature set: it must
    // mirror the heldIsI flag exactly (1 when set, 0 otherwise) and leave every other lane untouched.
    constexpr bool held_i_lane_matches_flag() {
        std::array<data_t, ModelV8::INPUTS> outFalse{};
        std::array<data_t, ModelV8::INPUTS> outTrue{};

        sim::Board2 const board{};
        ModelV8::extractInputs(clear_hist_t{}, board, false, outFalse);
        ModelV8::extractInputs(clear_hist_t{}, board, true, outTrue);

        if(outFalse[ModelV8::INPUTS - 1] != data_t{0})
            return false;
        if(outTrue[ModelV8::INPUTS - 1] != data_t{1})
            return false;

        // every other (v7-inherited) lane is unaffected by heldIsI
        for(std::size_t i = 0; i + 1 < ModelV8::INPUTS; ++i)
            if(outFalse[i] != outTrue[i])
                return false;
        return true;
    }
    static_assert(held_i_lane_matches_flag());

} // namespace

MODEL_V8_TEST(v8, held_i_lane_matches_flag_at_runtime) {
    EXPECT_TRUE(held_i_lane_matches_flag());
}

} // namespace ta3::ai
