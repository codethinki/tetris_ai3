#include "ta3/ai/model.hpp"

#include "helpers.hpp"
#include "test.hpp"

namespace ta3::ai {

AI_TEST(model_utils, size_matches_model) {
    ModelV4 const model;
    EXPECT_GT(model.size(), 0u);
    EXPECT_EQ(model.size(), ModelV4{}.size());
}

AI_TEST(model_utils, constructor_consumes_full_weight_vector) {
    ModelV4 const model;
    auto const weights = test::zero_weights(model.size());

    EXPECT_NO_THROW(ModelV4{weights});
}

}
