#pragma once

#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/eigen.hpp>
#include <pagmo/population.hpp>
#include <pagmo/rng.hpp>
#include <pagmo/s11n.hpp>

#include <boost/optional.hpp>

#include <string>
#include <tuple>
#include <vector>

namespace ta3::trn {

/** dd-CMA-ES: CMA-ES with diagonal decoding.
 *  the sampling covariance is factored into sigma^2 * D C D with a per-axis scaling vector D and a full
 *  correlation matrix C, adapted jointly: D learns with the fast O(1/n) rate of sep-CMA-ES, C with the
 *  slow full rate, and an adaptive damping (beta) keeps the two from fighting. converges like sep-CMA-ES
 *  on (partially) separable problems while still learning full correlations.
 *
 *  complexity per generation is O(n^2) (+ amortized eigendecomposition), state is O(n^2).
 *
 *  Akimoto & Hansen 2020, "Diagonal Acceleration for Covariance Matrix Adaptation Evolution Strategies",
 *  Evolutionary Computation 28(3). follows the author's reference implementation (ddcma.py). */
class dd_cmaes {
public:
    /** one line per logged generation: gen, fevals, best, dx, df, sigma (mirrors pagmo::cmaes) */
    using log_line_t = std::tuple<unsigned, unsigned long long, double, double, double, double>;
    using log_t = std::vector<log_line_t>;

    /** @param gen generations per evolve() call
     *  @param sigma0 initial step size, per-axis std starts as sigma0 * box width (like pagmo::cmaes)
     *  @param ftol stop when the population fitness spread falls below this
     *  @param xtol stop when the sampled variation falls below this
     *  @param memory when true the adaptation state carries over between evolve() calls
     *  @param force_bounds clamp evaluated samples into the box bounds. unlike pagmo::cmaes the
     *  adaptation keeps using the unclamped samples (the reference handles bounds outside the algorithm)
     *  @param active_update also learn from the worst individuals via negative weights (reference default)
     *  @param seed rng seed */
    dd_cmaes(
        unsigned gen = 1,
        double sigma0 = 0.5,
        double ftol = 1e-6,
        double xtol = 1e-6,
        bool memory = false,
        bool force_bounds = false,
        bool active_update = true,
        unsigned seed = pagmo::random_device::next()
    );

    /** evolves @ref pop for up to @ref _gen generations, stopping early on ftol/xtol
     *  @throws cth::except::default_exception if the problem is constrained, multi-objective,
     *  unbounded or the population holds fewer than 5 individuals */
    [[nodiscard]] pagmo::population evolve(pagmo::population pop) const;

    void set_seed(unsigned seed);
    [[nodiscard]] unsigned get_seed() const { return _seed; }

    void set_verbosity(unsigned level) { _verbosity = level; }
    [[nodiscard]] unsigned get_verbosity() const { return _verbosity; }

    [[nodiscard]] unsigned get_gen() const { return _gen; }

    void set_bfe(pagmo::bfe const& bfe) { _bfe = bfe; }

    [[nodiscard]] std::string get_name() const { return "dd-CMA-ES: Covariance Matrix Adaptation Evolutionary Strategy with Diagonal Decoding"; }
    [[nodiscard]] std::string get_extra_info() const;

    [[nodiscard]] log_t const& get_log() const { return _log; }

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive& archive, unsigned);

    /** resets the adaptation state around the current population champion,
     *  unless @ref _memory is set and the state still matches dim / lam */
    void resetState(pagmo::population const& pop, pagmo::vector_double const& lb, pagmo::vector_double const& ub, size_t lam) const;

    // config
    unsigned _gen;
    double _sigma0;
    double _ftol;
    double _xtol;
    bool _memory;
    bool _forceBounds;
    bool _activeUpdate;

    // adaptation state, only re-used across evolve() calls with _memory
    mutable double _sigma; ///< global step size (starts at 1, sigma0 is folded into _d)
    mutable Eigen::VectorXd _mean;
    mutable Eigen::VectorXd _d; ///< diagonal decoding, per-axis scaling
    mutable Eigen::MatrixXd _corrMatC; ///< correlation matrix
    mutable Eigen::MatrixXd _eigenvectorsC; ///< eigenvectors of C
    mutable Eigen::VectorXd _sqrtEigenvectorsC; ///< sqrt eigenvalues of C
    mutable Eigen::MatrixXd _sqrtC;
    mutable Eigen::MatrixXd _invSqrtC;
    mutable Eigen::MatrixXd _accCUpdate; ///< accumulated C update, applied on each eigendecomposition
    mutable Eigen::VectorXd _pc;
    mutable Eigen::VectorXd _pdc;
    mutable Eigen::VectorXd _ps;
    mutable double _pcFactor; ///< exact cumulation normalizer of _pc (replaces the asymptotic 1)
    mutable double _pdcFactor;
    mutable double _psFactor;
    mutable Eigen::MatrixXd _newpop; ///< sample buffer, one individual per column
    mutable unsigned long long _t; ///< generation counter, schedules the eigendecomposition
    mutable unsigned long long _countEval;

    mutable pagmo::detail::random_engine_type _engine;
    unsigned _seed;
    unsigned _verbosity = 0;
    mutable log_t _log;
    boost::optional<pagmo::bfe> _bfe;
};

}

PAGMO_S11N_ALGORITHM_EXPORT_KEY(ta3::trn::dd_cmaes)
