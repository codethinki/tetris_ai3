#include "ta3/ai/models/v4/model_v4.hpp"

#include "ta3/ai/model_utils.hpp"

#include <algorithm>

namespace ta3::ai {

ModelV4::ModelV4(std::span<double const> weights) : ModelV4{} { loadWeights(weights); }

void ModelV4::loadWeights(std::span<double const> weights) {
    auto const used = dev::load(weights, *_stage1, *_stage2);
    CTH_CRITICAL(used != weights.size(), "weights / init size mismatch") {}
}

auto ModelV4::forward(std::span<ai::data_t const, V4_INPUTS> input) const -> std::array<ai::data_t, STAGE2_OUT_SIZE> {
    v4_board_matrix const board = dlib::mat(input.data(), V4_BOARD_HEIGHT, V4_BOARD_WIDTH);
    v4_metadata const metadata{input.data() + V4_BOARD_SIZE, V4_METADATA_SIZE};
    return forward(board, metadata);
}

auto ModelV4::forward(v4_board_matrix input, v4_metadata metadata) const -> std::array<ai::data_t, OUTPUTS> {
    std::span<v4_board_matrix, 1> rng{&input, 1};
    auto const& stage1Out = _stage1->operator()(rng.begin(), rng.end());

    CTH_CRITICAL(
        stage1Out.size() != STAGE1_OUT_SIZE,
        "stage 1 size mismatch, expected: [{}] actual: [{}]",
        STAGE1_OUT_SIZE,
        stage1Out.size()
    ) {}

    std::array<data_t, STAGE2_IN_SIZE> stage2InBuffer{};
    std::copy_n(stage1Out.begin(), STAGE1_OUT_SIZE, stage2InBuffer.begin());
    std::ranges::copy(metadata, stage2InBuffer.begin() + STAGE1_OUT_SIZE);

    std::array<stage2_in_matrix, 1> stage2In{dlib::mat(stage2InBuffer.data(), stage2InBuffer.size(), 1)};

    auto const& stage2Out = _stage2->operator()(stage2In.begin(), stage2In.end());

    CTH_CRITICAL(
        stage2Out.size() != STAGE2_OUT_SIZE,
        "stage 2 size mismatch, expected: [{}] actual: [{}]",
        STAGE2_OUT_SIZE,
        stage2Out.size()
    ) {}

    std::array<ai::data_t, OUTPUTS> result{};
    std::copy_n(stage2Out.begin(), OUTPUTS, result.begin());
    return result;
}

std::vector<ai::data_t> ModelV4::batchForward(std::span<ai::data_t const> inputs) const {
    CTH_CRITICAL(inputs.size() % INPUTS != 0, "input buffer [{}] is not a multiple of INPUTS [{}]", inputs.size(), INPUTS) {}

    size_t const count = inputs.size() / INPUTS;
    if(count == 0)
        return {};

    std::vector<v4_board_matrix> boards{};
    std::vector<v4_metadata> metadata{};
    boards.reserve(count);
    metadata.reserve(count);

    for(size_t i = 0; i < count; ++i) {
        auto const slice = inputs.subspan(i * INPUTS, INPUTS);
        boards.emplace_back(dlib::mat(slice.data(), V4_BOARD_HEIGHT, V4_BOARD_WIDTH));
        metadata.emplace_back(v4_metadata{slice.data() + V4_BOARD_SIZE, V4_METADATA_SIZE});
    }

    return batchForward(boards, metadata);
}

std::vector<ai::data_t> ModelV4::batchForward(
    std::span<v4_board_matrix const> boards,
    std::span<v4_metadata const> metadata
) const {
    CTH_CRITICAL(
        boards.size() != metadata.size(),
        "boards [{}], metadata [{}] size mismatch",
        boards.size(),
        metadata.size()
    ) {}

    auto const inputs = boards.size();
    if(inputs == 0)
        return {};

    auto& stage1Out = _stage1->operator()(boards.begin(), boards.end());

    auto const expectedOutSize = boards.size() * STAGE1_OUT_SIZE;

    CTH_CRITICAL(
        stage1Out.size() != expectedOutSize,
        "model output expectation mismatch, expected [{}] actual [{}]",
        expectedOutSize,
        stage1Out.size()
    ) {}

    std::vector<data_t> stage2InBuffer(inputs * STAGE2_IN_SIZE);
    std::vector<stage2_in_matrix> stage2Ins{};
    stage2Ins.reserve(inputs);

    auto const stage1HostPtr = stage1Out.host();
    for(size_t i = 0; i < inputs; i++) {
        std::span buffer{&stage2InBuffer[i * STAGE2_IN_SIZE], STAGE2_IN_SIZE};
        std::span out{stage1HostPtr + i * STAGE1_OUT_SIZE, STAGE1_OUT_SIZE};

        std::ranges::copy(out, buffer.begin());
        std::ranges::copy(metadata[i], buffer.begin() + out.size());
        stage2Ins.emplace_back(dlib::mat(buffer.data(), STAGE2_IN_SIZE, 1));
    }

    CTH_CRITICAL(stage2Ins.empty(), "must not be empty") {}

    auto const& stage2Out = _stage2->operator()(stage2Ins.begin(), stage2Ins.end());

    CTH_CRITICAL(stage2Out.size() != boards.size() * STAGE2_OUT_SIZE, "stage 2 out size mismatch") {}

    std::vector<ai::data_t> out(boards.size() * STAGE2_OUT_SIZE);
    std::ranges::copy(stage2Out, out.begin());

    return out;
}

void ModelV4::init() const {
    ai::dev::init<STAGE1_IN_DIM>(*_stage1);
    ai::dev::init_flat<STAGE2_IN_SIZE>(*_stage2);
}

size_t ModelV4::size() const { return dev::size(*_stage1) + dev::size(*_stage2); }

}
