#include "ta3/trainer/pagmo/dd_cmaes.hpp"

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
    /** threshold for the adaptive damping between the D and C updates */
    constexpr double BETA_THRESH = 2.;

    /** strict weak ordering with NaN last, mirrors pagmo::detail::less_than_f */
    [[nodiscard]] bool lessF(double a, double b) {
        if(std::isnan(a))
            return false;
        if(std::isnan(b))
            return true;
        return a < b;
    }
}

dd_cmaes::dd_cmaes(
    unsigned gen,
    double sigma0,
    double ftol,
    double xtol,
    bool memory,
    bool force_bounds,
    bool active_update,
    unsigned seed
) : _gen{gen},
    _sigma0{sigma0},
    _ftol{ftol},
    _xtol{xtol},
    _memory{memory},
    _forceBounds{force_bounds},
    _activeUpdate{active_update},
    _sigma{1.},
    _pcFactor{0.},
    _pdcFactor{0.},
    _psFactor{0.},
    _t{0},
    _countEval{0},
    _engine{seed},
    _seed{seed} {
    CTH_STABLE_THROW(sigma0 <= 0., "sigma0 must be positive, got {}", sigma0) {}
}

void dd_cmaes::resetState(
    pagmo::population const& pop,
    pagmo::vector_double const& lb,
    pagmo::vector_double const& ub,
    size_t lam
) const {
    auto const dim = lb.size();
    auto const eDim = static_cast<Eigen::Index>(dim);

    if(_memory && _newpop.rows() == eDim && _newpop.cols() == static_cast<Eigen::Index>(lam))
        return;

    // sigma0 is folded into the diagonal decoding: per-axis std starts as sigma0 * box width
    _sigma = 1.;
    _mean = Eigen::Map<Eigen::VectorXd const>{pop.get_x()[pop.best_idx()].data(), eDim};
    _d.resize(eDim);
    for(auto j = 0uz; j < dim; ++j)
        _d[static_cast<Eigen::Index>(j)] = _sigma0 * std::max(ub[j] - lb[j], 1e-6);

    _corrMatC = Eigen::MatrixXd::Identity(eDim, eDim);
    _eigenvectorsC = Eigen::MatrixXd::Identity(eDim, eDim);
    _sqrtEigenvectorsC = Eigen::VectorXd::Ones(eDim);
    _sqrtC = Eigen::MatrixXd::Identity(eDim, eDim);
    _invSqrtC = Eigen::MatrixXd::Identity(eDim, eDim);
    _accCUpdate = Eigen::MatrixXd::Zero(eDim, eDim);

    _pc = Eigen::VectorXd::Zero(eDim);
    _pdc = Eigen::VectorXd::Zero(eDim);
    _ps = Eigen::VectorXd::Zero(eDim);
    _pcFactor = 0.;
    _pdcFactor = 0.;
    _psFactor = 0.;

    _newpop.resize(eDim, static_cast<Eigen::Index>(lam));
    _t = 0;
    _countEval = 0;
}

pagmo::population dd_cmaes::evolve(pagmo::population pop) const {
    auto const& prob = pop.get_problem(); // const ref: seeding goes through pop.get_problem()
    auto const dim = prob.get_nx();
    auto const bounds = prob.get_bounds();
    auto const& lb = bounds.first;
    auto const& ub = bounds.second;
    auto const lam = pop.size();
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

    // recombination weights over ALL ranks: positives sum to 1, negatives to -1 (active update)
    Eigen::VectorXd w{eLam};
    for(Eigen::Index i = 0; i < eLam; ++i)
        w[i] = std::log((static_cast<double>(lam) + 1.) / 2.) - std::log(static_cast<double>(i) + 1.);
    auto const nPos = static_cast<Eigen::Index>(std::ranges::count_if(w, [](double x) { return x > 0.; }));
    auto const nRest = eLam - nPos; // non-positive weights, zeros contribute nothing
    w.head(nPos) /= w.head(nPos).sum();
    w.tail(nRest) /= std::abs(w.tail(nRest).sum());
    auto const muEffPos = 1. / w.head(nPos).squaredNorm();
    auto const muEffNeg = 1. / w.tail(nRest).squaredNorm();

    // step-size constants
    auto const cs = (muEffPos + 2.) / (n + muEffPos + 5.);
    auto const ds = 1. + cs + 2. * std::max(0., std::sqrt((muEffPos - 1.) / (n + 1.)) - 1.);
    auto const chiN = std::sqrt(n) * (1. - 1. / (4. * n) + 1. / (21. * n * n)); // expectation of ||N(0, I)||

    // covariance learning rates; C gets the slow full-matrix rate (m = n(n+1)/2 free parameters),
    // the diagonal decoding D the fast sep-like rate (m = n) -- the core idea of dd-CMA
    auto const muPrime = muEffPos + 1. / muEffPos - 2.
        + static_cast<double>(lam) / (2. * static_cast<double>(lam) + 10.);
    constexpr double EXPO = 0.75;
    auto const cone = 1. / (2. * ((n + 1.) / 2. + 1.) * std::pow(n + 1., EXPO) + muEffPos / 2.);
    auto const cmu = std::min(1. - cone, muPrime * cone);
    auto const cc = std::sqrt(muEffPos * cone) / 2.;
    auto const cdone = 1. / (4. * std::pow(n + 1., EXPO) + muEffPos / 2.);
    auto const cdmu = std::min(1. - cdone, muPrime * cdone);
    auto const cdc = std::sqrt(muEffPos * cdone) / 2.;

    // negative weights, scaled separately for the C and D updates
    auto const negLimit = 1. + 2. * muEffNeg / (muEffPos + 2.);
    Eigen::VectorXd wc = w;
    Eigen::VectorXd wd = w;
    wc.tail(nRest) *= std::min(1. + cone / cmu, negLimit);
    wd.tail(nRest) *= std::min(1. + cdone / cdmu, negLimit);

    // eigendecomposition schedule
    auto const betaEig = 10. * n;
    auto const teig = std::max<unsigned long long>(1, static_cast<unsigned long long>(1. / (betaEig * (cone + cmu))));

    resetState(pop, lb, ub, lam);

    if(_verbosity > 0) {
        std::println("dd-CMA-ES:");
        std::println("lambda: {} - mueff+: {} - mueff-: {} - N: {}", lam, muEffPos, muEffNeg, n);
        std::println(
            "cc: {} - cs: {} - cone: {} - cmu: {} - cdc: {} - cdone: {} - cdmu: {} - teig: {}",
            cc,
            cs,
            cone,
            cmu,
            cdc,
            cdone,
            cdmu,
            teig
        );
    }

    std::normal_distribution normal{0., 1.};
    std::vector<size_t> order(lam);

    Eigen::MatrixXd zMat{eDim, eLam}; // sampled N(0, I) vectors, one per column
    Eigen::MatrixXd yMat{eDim, eLam}; // sqrtC * z
    Eigen::MatrixXd zs{eDim, eLam}; // z columns in fitness order
    Eigen::MatrixXd rankUpd{eDim, eDim};
    Eigen::VectorXd scale{eDim}; // sigma * d
    Eigen::VectorXd dz{eDim}; // weighted recombination in z-space
    Eigen::VectorXd dy{eDim}; // ... in y-space (= sqrtC * dz)
    Eigen::VectorXd zp{eDim}; // evolution path pulled back to z-space
    Eigen::VectorXd dUpd{eDim};
    Eigen::VectorXd negW{nRest};
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es{eDim};

    auto count = 1u; // regulates the screen output

    for(auto gen = 1u; gen <= _gen; ++gen) {
        // 1 - sample lam individuals from N(mean, sigma^2 D C D)
        for(Eigen::Index i = 0; i < eLam; ++i)
            for(Eigen::Index j = 0; j < eDim; ++j)
                zMat(j, i) = normal(_engine);
        yMat.noalias() = _sqrtC * zMat;
        scale = _sigma * _d;
        _newpop = ((yMat.array().colwise() * scale.array()).colwise() + _mean.array()).matrix();

        // 2 - exit conditions & logs, evaluated on the previous generation (mirrors pagmo::cmaes)
        auto const dx = scale.cwiseProduct(yMat.col(eLam - 1)).norm();
        auto const bestF = pop.get_f()[pop.best_idx()][0];
        auto const df = std::abs(bestF - pop.get_f()[pop.worst_idx()][0]);
        // early exits are rare and abort the search silently -- always log them
        if(dx < _xtol) {
            cth::log::msg<cth::except::INFO>("dd-cmaes exit at gen {}: dx {} < xtol {}", gen, dx, _xtol);
            return pop;
        }
        if(df < _ftol) {
            cth::log::msg<cth::except::INFO>("dd-cmaes exit at gen {}: df {} < ftol {}", gen, df, _ftol);
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

        // 3 - optionally clamp the EVALUATED samples into the box bounds. the adaptation below keeps
        //     using the unclamped z (the reference handles bounds outside the algorithm entirely)
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

        // 6 - rank ALL individuals (the active update also learns from the worst)
        auto const& fs = pop.get_f();
        std::ranges::iota(order, 0uz);
        std::ranges::sort(order, [&fs](size_t a, size_t b) { return lessF(fs[a][0], fs[b][0]); });
        for(Eigen::Index i = 0; i < eLam; ++i)
            zs.col(i) = zMat.col(static_cast<Eigen::Index>(order[static_cast<size_t>(i)]));

        // 7 - recombine (cm = 1)
        dz.noalias() = zs.leftCols(nPos) * w.head(nPos);
        dy.noalias() = _sqrtC * dz;
        _mean += _sigma * _d.cwiseProduct(dy);

        // 8 - step-size adaptation; the exact cumulation factor replaces the asymptotic 1
        _psFactor = (1. - cs) * (1. - cs) * _psFactor + cs * (2. - cs);
        _ps = (1. - cs) * _ps + std::sqrt(cs * (2. - cs) * muEffPos) * dz;
        auto const psSq = _ps.squaredNorm();
        double const hsig = psSq / _psFactor / n < 2. + 4. / (n + 1.);
        _sigma *= std::exp((std::sqrt(psSq) / chiN - std::sqrt(_psFactor)) * cs / ds);

        // 9 - accumulate the C update in z-space (applied lazily on each eigendecomposition)
        rankUpd.noalias() = zs.leftCols(nPos) * w.head(nPos).asDiagonal() * zs.leftCols(nPos).transpose();
        auto idCoef = w.head(nPos).sum();
        if(_activeUpdate) {
            // negative samples are normalized by their z-length to bound the update
            for(Eigen::Index i = 0; i < nRest; ++i)
                negW[i] = wc[nPos + i] * n / zs.col(nPos + i).squaredNorm();
            rankUpd.noalias() += zs.rightCols(nRest) * negW.asDiagonal() * zs.rightCols(nRest).transpose();
            idCoef += wc.tail(nRest).sum();
        }
        _pc = (1. - cc) * _pc + hsig * std::sqrt(cc * (2. - cc) * muEffPos) * _d.cwiseProduct(dy);
        _pcFactor = (1. - cc) * (1. - cc) * _pcFactor + hsig * cc * (2. - cc);
        zp.noalias() = _invSqrtC * _pc.cwiseQuotient(_d);
        _accCUpdate += cmu * rankUpd + cone * (zp * zp.transpose());
        _accCUpdate.diagonal().array() -= cmu * idCoef + cone * _pcFactor;

        // 10 - adapt the diagonal decoding D with the fast rate, damped by beta when C is far from I
        _pdc = (1. - cdc) * _pdc + hsig * std::sqrt(cdc * (2. - cdc) * muEffPos) * _d.cwiseProduct(dy);
        _pdcFactor = (1. - cdc) * (1. - cdc) * _pdcFactor + hsig * cdc * (2. - cdc);
        zp.noalias() = _invSqrtC * _pdc.cwiseQuotient(_d);
        dUpd = (cdone * (zp.array().square() - _pdcFactor)).matrix();
        dUpd.noalias() += cdmu * (zs.leftCols(nPos).cwiseAbs2() * wd.head(nPos));
        if(_activeUpdate) {
            for(Eigen::Index i = 0; i < nRest; ++i)
                negW[i] = wd[nPos + i] * n / zs.col(nPos + i).squaredNorm();
            dUpd.noalias() += cdmu * (zs.rightCols(nRest).cwiseAbs2() * negW);
            dUpd.array() -= cdmu * wd.sum();
        }
        else
            dUpd.array() -= cdmu * wd.head(nPos).sum();

        auto const beta = 1. / std::max(1., _sqrtEigenvectorsC.maxCoeff() / _sqrtEigenvectorsC.minCoeff() - BETA_THRESH + 1.);
        _d.array() *= (beta / 2. * dUpd.array()).exp();

        // 11 - apply the accumulated Z to C and refresh the decomposition, every teig generations
        ++_t;
        if(_t % teig == 0) {
            es.compute(_accCUpdate, Eigen::EigenvaluesOnly);
            auto const minEig = es.eigenvalues().minCoeff();
            auto const fac = std::min(0.75 / std::abs(minEig), 1.); // keeps I + fac * Z positive definite
            _corrMatC = _sqrtC * (Eigen::MatrixXd::Identity(eDim, eDim) + fac * _accCUpdate) * _sqrtC;

            // force C back to a correlation matrix: its variances move into d
            Eigen::VectorXd const cd = _corrMatC.diagonal().cwiseSqrt();
            _d = _d.cwiseProduct(cd);
            _corrMatC = cd.cwiseInverse().asDiagonal() * _corrMatC * cd.cwiseInverse().asDiagonal();

            _corrMatC = (_corrMatC + _corrMatC.transpose()) / 2.; // enforce symmetry
            es.compute(_corrMatC);
            if(es.info() == Eigen::Success) {
                _eigenvectorsC = es.eigenvectors();
                _sqrtEigenvectorsC = es.eigenvalues().cwiseMax(1e-20).cwiseSqrt();
                _sqrtC.noalias() = _eigenvectorsC * _sqrtEigenvectorsC.asDiagonal() * _eigenvectorsC.transpose();
                _invSqrtC.noalias() = _eigenvectorsC * _sqrtEigenvectorsC.cwiseInverse().asDiagonal() * _eigenvectorsC.transpose();
            } // if the eigendecomposition fails keep the previous successful one
            _accCUpdate.setZero();
        }
    }

    if(_verbosity > 0)
        std::println("Exit condition -- generations = {}", _gen);
    return pop;
}

void dd_cmaes::set_seed(unsigned seed) {
    _engine.seed(seed);
    _seed = seed;
}

std::string dd_cmaes::get_extra_info() const {
    return std::format(
        "\tGenerations: {}\n\tsigma0: {}\n\tStopping xtol: {}\n\tStopping ftol: {}\n"
        "\tMemory: {}\n\tActive update: {}\n\tVerbosity: {}\n\tForce bounds: {}\n\tSeed: {}",
        _gen,
        _sigma0,
        _xtol,
        _ftol,
        _memory,
        _activeUpdate,
        _verbosity,
        _forceBounds,
        _seed
    );
}

template<class Archive>
void dd_cmaes::serialize(Archive& archive, unsigned) {
    pagmo::detail::archive(
        archive,
        _gen,
        _sigma0,
        _ftol,
        _xtol,
        _memory,
        _forceBounds,
        _activeUpdate,
        _sigma,
        _mean,
        _d,
        _corrMatC,
        _eigenvectorsC,
        _sqrtEigenvectorsC,
        _sqrtC,
        _invSqrtC,
        _accCUpdate,
        _pc,
        _pdc,
        _ps,
        _pcFactor,
        _pdcFactor,
        _psFactor,
        _newpop,
        _t,
        _countEval,
        _engine,
        _seed,
        _verbosity,
        _log,
        _bfe
    );
}

}

PAGMO_S11N_ALGORITHM_IMPLEMENT(ta3::trn::dd_cmaes)
