#include "test_problems.hpp"

#include <ta3/trainer/pagmo/dd_cmaes.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <gtest/gtest.h>

#include <sstream>

namespace {

using namespace ta3;
using trn::dd_cmaes;

constexpr unsigned SEED = 0xdd5;
constexpr size_t POP_SIZE = 12;

[[nodiscard]] dd_cmaes make_uda(unsigned gen, unsigned seed = SEED) {
    return dd_cmaes{gen, .5, 1e-14, 1e-14, true, false, true, seed};
}

template<class Problem>
[[nodiscard]] pagmo::population evolve(dd_cmaes const& uda, unsigned pop_seed = SEED) {
    return pagmo::algorithm{uda}.evolve(pagmo::population{pagmo::problem{Problem{}}, POP_SIZE, pop_seed});
}

// ---------------------------------------------------------------------------------------------------
// convergence -- dd-CMA adapts a fast diagonal AND a full correlation matrix: it must handle both the
// axis-aligned ellipsoid (diagonal) and the rotated one (correlations, where sep-CMA-ES would stall)
// ---------------------------------------------------------------------------------------------------

TEST(dd_cmaes, ConvergesOnSphere) {
    auto const pop = evolve<trn::test::SphereProblem>(make_uda(400));
    EXPECT_LT(pop.champion_f()[0], 1e-6);
}

TEST(dd_cmaes, ConvergesOnSeparableEllipsoid) {
    auto const pop = evolve<trn::test::SepEllipsoidProblem>(make_uda(700));
    EXPECT_LT(pop.champion_f()[0], 1e-6);
}

TEST(dd_cmaes, ConvergesOnRotatedEllipsoid) {
    auto const pop = evolve<trn::test::RotEllipsoidProblem>(make_uda(1500));
    EXPECT_LT(pop.champion_f()[0], 1e-6);
}

// ---------------------------------------------------------------------------------------------------
// determinism & batch evaluation
// ---------------------------------------------------------------------------------------------------

TEST(dd_cmaes, DeterministicForSeed) {
    auto const a = evolve<trn::test::SphereProblem>(make_uda(30));
    auto const b = evolve<trn::test::SphereProblem>(make_uda(30));

    EXPECT_EQ(a.champion_x(), b.champion_x());
    EXPECT_EQ(a.champion_f(), b.champion_f());
}

TEST(dd_cmaes, BfeMatchesSerialEvaluation) {
    auto withBfe = make_uda(30);
    withBfe.set_bfe(pagmo::bfe{});

    auto const serial = evolve<trn::test::SphereProblem>(make_uda(30));
    auto const batch = evolve<trn::test::SphereProblem>(withBfe);

    EXPECT_EQ(serial.champion_x(), batch.champion_x());
    EXPECT_EQ(serial.champion_f(), batch.champion_f());
}

// ---------------------------------------------------------------------------------------------------
// serialization -- with memory = true the full adaptation state (paths, C, D, Z, rng) must roundtrip:
// a restored algorithm has to continue the search exactly like the original
// ---------------------------------------------------------------------------------------------------

TEST(dd_cmaes, SerializedStateResumesIdentically) {
    pagmo::algorithm algo{make_uda(40)};
    pagmo::population pop{pagmo::problem{trn::test::RotEllipsoidProblem{}}, POP_SIZE, SEED};
    pop = algo.evolve(pop);

    std::stringstream stream;
    {
        boost::archive::binary_oarchive archive{stream};
        archive << algo;
    }
    pagmo::algorithm restored{};
    {
        boost::archive::binary_iarchive archive{stream};
        archive >> restored;
    }
    ASSERT_NE(restored.extract<dd_cmaes>(), nullptr);

    auto const original = algo.evolve(pop);
    auto const resumed = restored.evolve(pop);

    EXPECT_EQ(original.champion_x(), resumed.champion_x());
    EXPECT_EQ(original.champion_f(), resumed.champion_f());
}

}
