#include "test.hpp"

#include "ta3/bot/model.hpp"

#include <filesystem>

namespace ta3::bot {

BOT_TEST(model, loads_champion) {
    auto const model = load_model(std::filesystem::path{TA3_TEST_MODEL_FILE});
    EXPECT_EQ(model.size(), ai::model_t::params());
}

}
