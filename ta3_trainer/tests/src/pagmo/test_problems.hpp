#pragma once

#include <pagmo/detail/eigen.hpp>
#include <pagmo/types.hpp>

#include <cmath>
#include <random>
#include <utility>

namespace ta3::trn::test {

constexpr size_t PROBLEM_DIM = 10;
constexpr double PROBLEM_BOUND = 5.;

using bounds_t = std::pair<pagmo::vector_double, pagmo::vector_double>;

[[nodiscard]] inline bounds_t problem_bounds() {
    return {
        pagmo::vector_double(PROBLEM_DIM, -PROBLEM_BOUND),
        pagmo::vector_double(PROBLEM_DIM, PROBLEM_BOUND)
    };
}

/** f(x) = sum x_i^2, the simplest convergence check */
struct SphereProblem {
    [[nodiscard]] static bounds_t get_bounds() { return problem_bounds(); }

    [[nodiscard]] static pagmo::vector_double fitness(pagmo::vector_double const& x) {
        double sum = 0.;
        for(auto const v : x)
            sum += v * v;
        return {sum};
    }
};

/** axis-aligned ellipsoid, condition number 1e6 -- benign for diagonal covariance models */
struct SepEllipsoidProblem {
    [[nodiscard]] static bounds_t get_bounds() { return problem_bounds(); }

    [[nodiscard]] static pagmo::vector_double fitness(pagmo::vector_double const& x) {
        double sum = 0.;
        for(auto i = 0uz; i < x.size(); ++i)
            sum += std::pow(1e6, static_cast<double>(i) / (static_cast<double>(x.size()) - 1.)) * x[i] * x[i];
        return {sum};
    }
};

/** the same ellipsoid under a fixed random rotation -- solvable only by learning full correlations */
struct RotEllipsoidProblem {
    [[nodiscard]] static bounds_t get_bounds() { return problem_bounds(); }

    [[nodiscard]] static pagmo::vector_double fitness(pagmo::vector_double const& x) {
        auto const& r = rotation();
        auto const rotated = (r * Eigen::Map<Eigen::VectorXd const>{x.data(), static_cast<Eigen::Index>(x.size())}).eval();
        return SepEllipsoidProblem::fitness({rotated.data(), rotated.data() + rotated.size()});
    }

private:
    /** deterministic random rotation, built once */
    [[nodiscard]] static Eigen::MatrixXd const& rotation() {
        static Eigen::MatrixXd const r = [] {
            constexpr auto eDim = static_cast<Eigen::Index>(PROBLEM_DIM);
            std::mt19937 rng{0xdd};
            std::normal_distribution normal{0., 1.};

            Eigen::MatrixXd m{eDim, eDim};
            for(Eigen::Index i = 0; i < eDim; ++i)
                for(Eigen::Index j = 0; j < eDim; ++j)
                    m(i, j) = normal(rng);
            return Eigen::MatrixXd{Eigen::HouseholderQR<Eigen::MatrixXd>{m}.householderQ()};
        }();
        return r;
    }
};

}
