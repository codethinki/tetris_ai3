#pragma once

#include <ta3/ai/model.hpp>

#include <filesystem>

namespace ta3::bot {

/** loads one trainer-produced champion file into the selected model */
[[nodiscard]] ai::model_t load_model(std::filesystem::path const& file);

}
