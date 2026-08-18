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

/** sep-CMA-ES: CMA-ES restricted to a diagonal covariance matrix.
 *  drop-in for pagmo::cmaes (same constructor & member interface) with linear instead of quadratic
 *  time and space complexity: sampling scales each axis independently, so no eigendecomposition and
 *  no full matrix are ever needed. the lost correlations are partly compensated by boosting the
 *  covariance learning rates c1/cmu by (n+2)/3.
 *
 *  best suited for separable / high dimensional problems, where plain CMA-ES spends most of its
 *  time on the O(n^2) covariance machinery.
 *
 *  Ros & Hansen 2008, "A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity" */
class sep_cmaes {
public:
    /** one line per logged generation: gen, fevals, best, dx, df, sigma (mirrors pagmo::cmaes) */
    using log_line_t = std::tuple<unsigned, unsigned long long, double, double, double, double>;
    using log_t = std::vector<log_line_t>;

    /** @param gen generations per evolve() call
     *  @param cc backward time horizon of the covariance evolution path ([0,1], -1 = auto)
     *  @param cs backward time horizon of the sigma evolution path ([0,1], -1 = auto)
     *  @param c1 rank-one covariance learning rate ([0,1], -1 = auto incl. the (n+2)/3 sep boost)
     *  @param cmu rank-mu covariance learning rate ([0,1], -1 = auto incl. the (n+2)/3 sep boost)
     *  @param sigma0 initial step size
     *  @param ftol stop when the population fitness spread falls below this
     *  @param xtol stop when the sampled variation falls below this
     *  @param memory when true the adaptation state carries over between evolve() calls
     *  @param force_bounds clamp samples into the box bounds (worsens the adaptation)
     *  @param seed rng seed
     *  @throws cth::except::default_exception if cc, cs, c1 or cmu is not in [0,1] or -1 */
    sep_cmaes(
        unsigned gen = 1,
        double cc = -1,
        double cs = -1,
        double c1 = -1,
        double cmu = -1,
        double sigma0 = 0.5,
        double ftol = 1e-6,
        double xtol = 1e-6,
        bool memory = false,
        bool force_bounds = false,
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

    [[nodiscard]] std::string get_name() const { return "sep-CMA-ES: Separable Covariance Matrix Adaptation Evolutionary Strategy"; }
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
    double _cc;
    double _cs;
    double _c1;
    double _cmu;
    double _sigma0;
    double _ftol;
    double _xtol;
    bool _memory;
    bool _forceBounds;

    // adaptation state, only re-used across evolve() calls with _memory (mirrors pagmo::cmaes)
    mutable double _sigma;
    mutable Eigen::VectorXd _mean;
    mutable Eigen::VectorXd _diag; ///< diagonal of C, i.e. per-axis variances
    mutable Eigen::VectorXd _pc;
    mutable Eigen::VectorXd _ps;
    mutable Eigen::MatrixXd _newpop; ///< sample buffer, one individual per column
    mutable unsigned long long _countEval;

    mutable pagmo::detail::random_engine_type _engine;
    unsigned _seed;
    unsigned _verbosity = 0;
    mutable log_t _log;
    boost::optional<pagmo::bfe> _bfe;
};

}

PAGMO_S11N_ALGORITHM_EXPORT_KEY(ta3::trn::sep_cmaes)
