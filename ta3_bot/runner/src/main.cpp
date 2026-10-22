#include "ta3/bot/app.hpp"
#include "ta3/bot/model.hpp"

#include <cth/io/log.hpp>

#include <exception>
#include <filesystem>

namespace {
int run_bot(std::filesystem::path const& modelFile) {
    ta3::bot::BotApp app{ta3::bot::load_model(modelFile)};
    return app.run();
}
}

int main(int argc, char** argv) {
    auto const modelFile = argc > 1
        ? std::filesystem::path{argv[1]}
        : std::filesystem::path{TA3_DEFAULT_MODEL_FILE};

#ifndef NDEBUG
    return run_bot(modelFile);
#else
    try { return run_bot(modelFile); } catch(std::exception const& e) {
        cth::log::msg<cth::except::ERR>("{}", e.what());
        return 1;
    } catch(...) {
        cth::log::msg<cth::except::ERR>("exited with unknown error");
        return 1;
    }
#endif
}
