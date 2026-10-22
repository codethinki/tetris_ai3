#include "ta3/bot/model.hpp"

#include <cth/io/log.hpp>

#include <cstddef>
#include <fstream>
#include <vector>

namespace ta3::bot {

ai::model_t load_model(std::filesystem::path const& file) {
    std::ifstream stream{file, std::ios::binary};
    CTH_STABLE_THROW(!stream.is_open(), "failed to open model file: {}", file.string()) {}

    std::size_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), sizeof(count));
    CTH_STABLE_THROW(!stream, "failed to read model header: {}", file.string()) {}
    CTH_STABLE_THROW(
        count != ai::model_t::params(),
        "model parameter count mismatch: expected {}, got {}",
        ai::model_t::params(),
        count
    ) {}

    std::vector<double> weights(count);
    stream.read(
        reinterpret_cast<char*>(weights.data()),
        static_cast<std::streamsize>(weights.size() * sizeof(double))
    );
    CTH_STABLE_THROW(!stream, "failed to read model parameters: {}", file.string()) {}

    return ai::model_t{weights};
}

}
