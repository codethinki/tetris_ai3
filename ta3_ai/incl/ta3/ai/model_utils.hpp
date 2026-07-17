#pragma once
#include "ta3/ai/model_defs.hpp"

#include "ta3/ai/lib/dlib.hpp"

#include <cth/io/log.hpp>

#include <span>
#include <string_view>


namespace ta3::ai {
constexpr std::string_view SCALE_SUFFIX = "_scale";
}

namespace ta3::ai::dev {
size_t load(std::span<double const> weights, auto& model) {
    CTH_CRITICAL(weights.empty(), "model must not be empty") {}

    size_t offset = 0;

    dlib::visit_layer_parameters(
        model,
        [&offset, weights](dlib::tensor& layer) {
            auto const size = layer.size();
            if(size == 0)
                return;


            std::span dst{layer.host_write_only(), size};
            std::span src{&weights[offset], size};

            std::ranges::copy(src, dst.begin());
            layer.async_copy_to_device();

            offset += dst.size();
        }
    );

    if constexpr(cth::COMPILATION_MODE == cth::CompilationMode::DEBUG) {
        offset = 0;
        dlib::visit_layer_parameters(
            model,
            [&offset, weights](dlib::tensor const& layer) {
                auto const size = layer.size();
                if(size == 0)
                    return;
                std::span actual{layer.host(), size};
                std::span expected{&weights[offset], size};

                for(auto const [a, e] : std::views::zip(actual, expected)) {
                    CTH_CRITICAL(std::abs(a - e) > 1e-5, "model parameters set incorrectly") {}
                }
                offset += size;
            }
        );
    }
    return offset;
}

template<class... Models> requires(sizeof...(Models) > 0)
size_t load(std::span<double const> weights, auto& model, Models&... models) {
    auto const offset = dev::load(weights, model);

    CTH_CRITICAL(offset >= weights.size(), "more weights needed") {}

    return offset + dev::load(std::span{&weights[offset], weights.size() - offset}, models...);
}


template<ta3::sim::szvec2 Dim>
void init(auto& model) {
    static_assert(Dim.x != 0 && Dim.y != 0, "must have more than 0 params");
    using matrix_t = dlib::matrix<ai::data_t, Dim.y, Dim.x>;

    std::array<matrix_t, 1> rng{};

    model(rng.begin(), rng.end());
}

template<size_t S>
void init_flat(auto& model) { init<ta3::sim::szvec2{S, 1}>(model); }

size_t size(auto& model) {
    size_t sum = 0;
    dlib::visit_layer_parameters(
        model,
        [&sum](dlib::tensor const& layer) {
            auto size = layer.size();
            sum += size;
        }
    );
    return sum;
}

}
