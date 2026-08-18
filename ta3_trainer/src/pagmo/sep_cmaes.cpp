#include "ta3/trainer/pagmo/sep_cmaes.hpp"

#include <cth/io/log.hpp>

#include <pagmo/detail/eigen_s11n.hpp>
#include <pagmo/problem.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>
#include <print>
#include <random>
#include <string>
#include <vector>

// NOTE: must be included *after* the other serialization headers (see pagmo's cmaes.cpp)
#include <boost/serialization/optional.hpp>

namespace ta3::trn {
namespace {
    [[nodiscard]] constexpr bool rateValid(double rate) { return rate == -1. || (rate >= 0. && rate <= 1.); }

    /** strict weak ordering with NaN last, mirrors pagmo::detail::less_than_f */
    [[nodiscard]] bool lessF(double a, double b) {
        if(std::isnan(a))
            return false;
        if(std::isnan(b))
            return true;
        return a < b;
    }
}

sep_cmaes::sep_cmaes(
    unsigned gen,
    double cc,
    double cs,
    double c1,
    double cmu,
    double sigma0,
    double ftol,
    double xtol,
    bool memory,
    bool force_bounds,
    unsigned seed
) : _gen{gen},
    _cc{cc},
    _cs{cs},
    _c1{c1},
    _cmu{cmu},
    _sigma0{sigma0},
    _ftol{ftol},
    _xtol{xtol},
    _memory{memory},
    _forceBounds{force_bounds},
    _sigma{sigma0},
    _countEval{0},
    _engine{seed},
    _seed{seed} {
    CTH_STABLE_THROW(!rateValid(cc), "cc must be in [0,1] or -1 (auto), got {}", cc) {}
    CTH_STABLE_THROW(!rateValid(cs), "cs must be in [0,1] or -1 (auto), got {}", cs) {}
    CTH_STABLE_THROW(!rateValid(c1), "c1 must be in [0,1] or -1 (auto), got {}", c1) {}
    CTH_STABLE_THROW(!rateValid(cmu), "cmu must be in [0,1] or -1 (auto), got {}", cmu) {}
}

void sep_cmaes::resetState(
    pagmo::population const& pop,
    pagmo::vector_double const& lb,
    pagmo::vector_double const& ub,
    size_t lam
) const {
    auto const dim = lb.size();
    auto const eDim = static_cast<Eigen::Index>(dim);

    if(_memory && _newpop.rows() == eDim && _newpop.cols() == static_cast<Eigen::Index>(lam))
        return;

    _sigma = _sigma0;
    _mean = Eigen::Map<Eigen::VectorXd const>{pop.get_x()[pop.best_idx()].data(), eDim};

    // the per-axis variance starts as the squared box width (pagmo::cmaes seeds D with the box width)
    _diag.resize(eDim);
    for(auto j = 0uz; j < dim; ++j)
        _diag[static_cast<Eigen::Index>(j)] = std::pow(std::max(ub[j] - lb[j], 1e-6), 2);

    _pc = Eigen::VectorXd::Zero(eDim);
    _ps = Eigen::VectorXd::Zero(eDim);
    _newpop.resize(eDim, static_cast<Eigen::Index>(lam));
    _countEval = 0;
}

pagmo::population sep_cmaes::evolve(pagmo::population pop) const {
    auto const& prob = pop.get_problem(); // const ref: seeding goes through pop.get_problem()
    auto const dim = prob.get_nx();
    auto const bounds = prob.get_bounds();
    auto const& lb = bounds.first;
    auto const& ub = bounds.second;
    auto const lam = pop.size();
    auto const mu = lam / 2;
    auto const fevals0 = prob.get_fevals(); // discount the already made fevals

    CTH_STABLE_THROW(prob.get_nc() != 0, "{} cannot deal with constraints ({})", get_name(), prob.get_name()) {}
    CTH_STABLE_THROW(prob.get_nf() != 1, "{} cannot deal with multiple objectives ({})", get_name(), prob.get_name()) {}
    CTH_STABLE_THROW(lam < 5, "{} needs at least 5 individuals in the population, got {}", get_name(), lam) {}
    auto const finite = [](double x) { return std::isfinite(x); };
    CTH_STABLE_THROW(
        !std::ranges::all_of(lb, finite) || !std::ranges::all_of(ub, finite),
        "{} requires finite box bounds ({})",
        get_name(),
        prob.get_name()
    ) {}

    if(_gen == 0)
        return pop;

    _log.clear();

    auto const n = static_cast<double>(dim);
    auto const eDim = static_cast<Eigen::Index>(dim);
    auto const eLam = static_cast<Eigen::Index>(lam);
    auto const eMu = static_cast<Eigen::Index>(mu);

    // weighted recombination coefficients
    Eigen::VectorXd weights{eMu};
    for(Eigen::Index i = 0; i < eMu; ++i)
        weights[i] = std::log(static_cast<double>(mu) + 0.5) - std::log(static_cast<double>(i) + 1.);
    weights /= weights.sum();
    auto const muEff = 1. / weights.squaredNorm(); // variance-effectiveness of sum w_i x_i

    // adaptation constants; the diagonal restriction allows boosting c1/cmu by (n+2)/3 (Ros & Hansen 2008)
    auto const sepBoost = (n + 2.) / 3.;
    auto const cc = _cc != -1 ? _cc : (4. + muEff / n) / (n + 4. + 2. * muEff / n);
    auto const cs = _cs != -1 ? _cs : (muEff + 2.) / (n + muEff + 5.);
    auto const c1 = _c1 != -1 ? _c1 : std::min(1., sepBoost * 2. / ((n + 1.3) * (n + 1.3) + muEff));
    auto const cmu = _cmu != -1
                     ? _cmu
                     : std::min(1. - c1, sepBoost * 2. * (muEff - 2. + 1. / muEff) / ((n + 2.) * (n + 2.) + muEff));
    auto const damps = 1. + 2. * std::max(0., std::sqrt((muEff - 1.) / (n + 1.)) - 1.) + cs;
    auto const chiN = std::sqrt(n) * (1. - 1. / (4. * n) + 1. / (21. * n * n)); // expectation of ||N(0, I)||

    resetState(pop, lb, ub, lam);

    if(_verbosity > 0) {
        std::println("sep-CMA-ES:");
        std::println("mu: {} - lambda: {} - mueff: {} - N: {}", mu, lam, muEff, n);
        std::println(
            "cc: {} - cs: {} - c1: {} - cmu: {} - sigma: {} - damps: {} - chiN: {}",
            cc,
            cs,
            c1,
            cmu,
            _sigma,
            damps,
            chiN
        );
    }

    std::normal_distribution normal{0., 1.};
    std::vector<size_t> order(lam);

    Eigen::VectorXd axisStd{eDim}; // per-axis standard deviations, sqrt of the C diagonal
    Eigen::VectorXd z{eDim}; // last sampled N(0, I) vector
    Eigen::VectorXd yw{eDim}; // weighted recombination of the elite steps
    Eigen::MatrixXd y{eDim, eMu}; // elite steps (x_i:lam - mean) / sigma, one per column

    auto count = 1u; // regulates the screen output

    for(auto gen = 1u; gen <= _gen; ++gen) {
        // 1 - sample lam individuals from N(mean, sigma^2 C): O(dim) per sample, no eigendecomposition
        axisStd = _diag.cwiseSqrt();
        for(Eigen::Index i = 0; i < eLam; ++i) {
            for(Eigen::Index j = 0; j < eDim; ++j)
                z[j] = normal(_engine);
            _newpop.col(i) = _mean + _sigma * axisStd.cwiseProduct(z);
        }

        // 2 - exit conditions & logs, evaluated on the previous generation (mirrors pagmo::cmaes)
        auto const dx = (_sigma * axisStd.cwiseProduct(z)).norm();
        auto const bestF = pop.get_f()[pop.best_idx()][0];
        auto const df = std::abs(bestF - pop.get_f()[pop.worst_idx()][0]);
        // early exits are rare and abort the search silently -- always log them
        if(dx < _xtol) {
            cth::log::msg<cth::except::INFO>("sep-cmaes exit at gen {}: dx {} < xtol {}", gen, dx, _xtol);
            return pop;
        }
        if(df < _ftol) {
            cth::log::msg<cth::except::INFO>("sep-cmaes exit at gen {}: df {} < ftol {}", gen, df, _ftol);
            return pop;
        }
        if(_verbosity > 0 && (gen % _verbosity == 1 || _verbosity == 1)) {
            if(count % 50 == 1)
                std::println(
                    "\n{:>7} {:>15} {:>15} {:>15} {:>15} {:>15}",
                    "Gen:",
                    "Fevals:",
                    "Best:",
                    "dx:",
                    "df:",
                    "sigma:"
                );
            std::println(
                "{:>7} {:>15} {:>15} {:>15} {:>15} {:>15}",
                gen,
                prob.get_fevals() - fevals0,
                bestF,
                dx,
                df,
                _sigma
            );
            ++count;
            _log.emplace_back(gen, prob.get_fevals() - fevals0, bestF, dx, df, _sigma);
        }

        // 3 - optionally clamp into the box bounds (worsens the covariance adaptation)
        if(_forceBounds) {
            Eigen::Map<Eigen::VectorXd const> const lbV{lb.data(), eDim};
            Eigen::Map<Eigen::VectorXd const> const ubV{ub.data(), eDim};
            for(Eigen::Index i = 0; i < eLam; ++i)
                _newpop.col(i) = _newpop.col(i).cwiseMax(lbV).cwiseMin(ubV);
        }

        // 4 - reseed stochastic problems
        if(prob.is_stochastic())
            pop.get_problem().set_seed(std::uniform_int_distribution<unsigned>{}(_engine));

        // 5 - evaluate & reinsert; the column-major sample buffer already is the flat bfe layout
        if(_bfe) {
            pagmo::vector_double const dvs(_newpop.data(), _newpop.data() + dim * lam);
            auto const fitnesses = (*_bfe)(prob, dvs);
            for(auto i = 0uz; i < lam; ++i) {
                auto const* col = _newpop.data() + i * dim;
                pop.set_xf(i, pagmo::vector_double(col, col + dim), {fitnesses[i]});
            }
        }
        else
            for(auto i = 0uz; i < lam; ++i) {
                auto const* col = _newpop.data() + i * dim;
                pop.set_x(i, pagmo::vector_double(col, col + dim));
            }
        _countEval += lam;

        // 6 - elite selection: the mu best individuals
        auto const& fs = pop.get_f();
        std::ranges::iota(order, 0uz);
        std::ranges::partial_sort(
            order,
            order.begin() + static_cast<ptrdiff_t>(mu),
            [&fs](size_t a, size_t b) { return lessF(fs[a][0], fs[b][0]); }
        );

        // 7 - recombine: y-columns are the elite steps, yw the weighted mean step
        auto const& xs = pop.get_x();
        for(Eigen::Index i = 0; i < eMu; ++i)
            y.col(i) = (Eigen::Map<Eigen::VectorXd const>{xs[order[static_cast<size_t>(i)]].data(), eDim} - _mean) /
                _sigma;
        yw.noalias() = y * weights;
        _mean += _sigma * yw;

        // 8 - update the evolution paths; diagonal C makes C^(-1/2) a per-axis division
        _ps = (1. - cs) * _ps + std::sqrt(cs * (2. - cs) * muEff) * yw.cwiseQuotient(axisStd);
        double const hsig = _ps.squaredNorm() / n
            / (1. - std::pow(1. - cs, 2. * static_cast<double>(_countEval) / static_cast<double>(lam)))
            < 2. + 4. / (n + 1.);
        _pc = (1. - cc) * _pc + hsig * std::sqrt(cc * (2. - cc) * muEff) * yw;

        // 9 - adapt the C diagonal: rank-one + rank-mu restricted to the diagonal, all elementwise
        _diag = (1. - c1 - cmu) * _diag
            + c1 * (_pc.cwiseAbs2() + (1. - hsig) * cc * (2. - cc) * _diag)
            + cmu * (y.cwiseAbs2() * weights);
        _diag = _diag.cwiseMax(1e-20);

        // 10 - adapt sigma
        _sigma *= std::exp(std::min(0.6, cs / damps * (_ps.norm() / chiN - 1.)));
    }

    if(_verbosity > 0)
        std::println("Exit condition -- generations = {}", _gen);
    return pop;
}

void sep_cmaes::set_seed(unsigned seed) {
    _engine.seed(seed);
    _seed = seed;
}

std::string sep_cmaes::get_extra_info() const {
    auto const rate = [](double value) { return value == -1. ? std::string{"auto"} : std::format("{}", value); };
    return std::format(
        "\tGenerations: {}\n\tcc: {}\n\tcs: {}\n\tc1: {}\n\tcmu: {}\n\tsigma0: {}\n"
        "\tStopping xtol: {}\n\tStopping ftol: {}\n\tMemory: {}\n\tVerbosity: {}\n\tForce bounds: {}\n\tSeed: {}",
        _gen,
        rate(_cc),
        rate(_cs),
        rate(_c1),
        rate(_cmu),
        _sigma0,
        _xtol,
        _ftol,
        _memory,
        _verbosity,
        _forceBounds,
        _seed
    );
}

template<class Archive>
void sep_cmaes::serialize(Archive& archive, unsigned) {
    pagmo::detail::archive(
        archive,
        _gen,
        _cc,
        _cs,
        _c1,
        _cmu,
        _sigma0,
        _ftol,
        _xtol,
        _memory,
        _forceBounds,
        _sigma,
        _mean,
        _diag,
        _pc,
        _ps,
        _newpop,
        _countEval,
        _engine,
        _seed,
        _verbosity,
        _log,
        _bfe
    );
}

}

PAGMO_S11N_ALGORITHM_IMPLEMENT(ta3::trn::sep_cmaes)
