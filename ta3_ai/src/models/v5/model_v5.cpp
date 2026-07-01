#include "ta3/ai/models/v5/model_v5.hpp"

#include "ta3/ai/model_utils.hpp"

#include <algorithm>

namespace ta3::ai {

ModelV5::ModelV5(std::span<double const> weights) : ModelV5{} { loadWeights(weights); }

void ModelV5::loadWeights(std::span<double const> weights) {
    auto const used = dev::load(weights, *_net);
    CTH_CRITICAL(used != weights.size(), "weights / init size mismatch") {}
}

auto ModelV5::forward(std::span<ai::data_t const, INPUTS> input) const -> std::array<ai::data_t, OUTPUTS> {
    std::array<input_matrix_t, 1> in{dlib::mat(input.data(), INPUTS, 1)};
    auto const& out = _net->operator()(in.begin(), in.end());

    CTH_CRITICAL(out.size() != OUTPUTS, "v5 out size mismatch, expected [{}] actual [{}]", OUTPUTS, out.size()) {}

    std::array<ai::data_t, OUTPUTS> result{};
    std::copy_n(out.begin(), OUTPUTS, result.begin());
    return result;
}

std::vector<ai::data_t> ModelV5::batchForward(std::span<ai::data_t const> inputs) const {
    CTH_CRITICAL(inputs.size() % INPUTS != 0, "input buffer [{}] is not a multiple of INPUTS [{}]", inputs.size(), INPUTS) {}

    size_t const count = inputs.size() / INPUTS;
    if(count == 0)
        return {};

    std::vector<input_matrix_t> matrices{};
    matrices.reserve(count);
    for(size_t i = 0; i < count; ++i)
        matrices.emplace_back(dlib::mat(&inputs[i * INPUTS], INPUTS, 1));

    auto const& out = _net->operator()(matrices.begin(), matrices.end());

    CTH_CRITICAL(out.size() != count * OUTPUTS, "v5 batch out size mismatch") {}

    std::vector<ai::data_t> result(count * OUTPUTS);
    std::ranges::copy(out, result.begin());
    return result;
}

void ModelV5::init() const { ai::dev::init_flat<INPUTS>(*_net); }

size_t ModelV5::size() const { return dev::size(*_net); }

}
