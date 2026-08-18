#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

#include "TestForceFunctor.h"
#include "autopas/options/ContainerOption.h"
#include "autopas/options/DataLayoutOption.h"
#include "autopas/options/Newton3Option.h"
#include "autopas/options/TraversalOption.h"
#include "distributed_autopas/DistributedAutoPas.h"
#include "distributed_autopas/Particle.h"
#include "distributed_autopas/Runtime.h"

namespace {

using Particle = dap::Particle;

constexpr std::array<double, 3> globalMin{0., 0., 0.};
constexpr std::array<double, 3> globalMax{4., 1., 1.};
constexpr double cutoff = 0.2;

Particle makeParticle(unsigned long id, double x) {
  Particle particle;
  particle.setID(id);
  particle.setR({x, 0.5, 0.5});
  return particle;
}

auto makeConfigurator() {
  return [](auto &autoPas) {
    autoPas.setAllowedContainers({autopas::ContainerOption::directSum});
    autoPas.setAllowedTraversals({autopas::TraversalOption::ds_sequential});
    autoPas.setAllowedDataLayouts({autopas::DataLayoutOption::aos});
    autoPas.setAllowedNewton3Options({autopas::Newton3Option::enabled});
    autoPas.setVerletSkin(0.0);
  };
}

std::vector<unsigned long> ownedIds(const dap::DistributedAutoPas<Particle> &particles) {
  std::vector<unsigned long> ids;
  particles.forEachOwnedParticle([&](const auto &particle) { ids.push_back(particle.getID()); });
  std::sort(ids.begin(), ids.end());
  return ids;
}

constexpr std::array<double, 3> forceGlobalMin{0., 0., 0.};
constexpr std::array<double, 3> forceGlobalMax{8.8, 2., 2.};
constexpr double forceCutoff = 1.5;

std::vector<Particle> makeForceTestParticles() {
  constexpr std::array<double, 8> positions{0.4, 1.4, 2.7, 3.7, 4.9, 5.9, 7.2, 8.2};

  std::vector<Particle> particles;
  particles.reserve(positions.size());
  for (std::size_t id = 0; id < positions.size(); ++id) {
    auto particle = makeParticle(static_cast<unsigned long>(id), positions[id]);
    particle.setF({0., 0., 0.});
    particles.push_back(particle);
  }
  return particles;
}

void expectForceTestResult(int expectedNumberOfRanks) {
  int argc = 0;
  char **argv = nullptr;
  dap::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), expectedNumberOfRanks);

  dap::DistributedAutoPas<Particle> particles(runtime, forceGlobalMin, forceGlobalMax, forceCutoff, makeConfigurator());
  particles.addParticlesFromRoot(makeForceTestParticles());

  dap::testing::TestForceFunctor<Particle> functor(forceCutoff);
  particles.computeInteractions(&functor);

  constexpr std::array<double, 8> expectedForceX{0.0, 0.3, -0.3, 0.2, -0.2, 0.3, -0.3, 0.0};

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 8);
  EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 8 / expectedNumberOfRanks);

  std::size_t checkedParticles = 0;
  particles.forEachOwnedParticle([&](const auto &particle) {
    ASSERT_LT(particle.getID(), expectedForceX.size());
    const auto &force = particle.getF();
    EXPECT_NEAR(force[0], expectedForceX[particle.getID()], 1e-12) << "particle id " << particle.getID();
    EXPECT_NEAR(force[1], 0.0, 1e-12) << "particle id " << particle.getID();
    EXPECT_NEAR(force[2], 0.0, 1e-12) << "particle id " << particle.getID();
    ++checkedParticles;
  });

  EXPECT_EQ(checkedParticles, 8 / expectedNumberOfRanks);
  particles.finalize();
}

TEST(DistributedAutoPasTest, AddsReplicatedInputFromRootExactlyOnce) {
  int argc = 0;
  char **argv = nullptr;
  dap::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 4);

  dap::DistributedAutoPas<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  std::vector<Particle> replicatedInput;
  for (unsigned long id = 0; id < 8; ++id) {
    const auto owner = static_cast<int>(id / 2);
    const auto offsetWithinDomain = id % 2 == 0 ? 0.25 : 0.75;
    replicatedInput.push_back(makeParticle(id, static_cast<double>(owner) + offsetWithinDomain));
  }

  // Deliberately pass the same replicated collection on every rank. Only root's
  // collection may be considered by addParticlesFromRoot().
  particles.addParticlesFromRoot(replicatedInput);

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 8);
  EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 2);

  const auto firstExpectedId = static_cast<unsigned long>(2 * runtime.rank());
  EXPECT_EQ(ownedIds(particles), (std::vector<unsigned long>{firstExpectedId, firstExpectedId + 1}));

  particles.finalize();
}

TEST(DistributedAutoPasTest, RedistributesContributionsFromAllRanksToTheirOwners) {
  int argc = 0;
  char **argv = nullptr;
  dap::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 4);

  dap::DistributedAutoPas<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  const auto rank = runtime.rank();
  const auto rightRank = (rank + 1) % runtime.size();

  // Every rank contributes one particle that already belongs locally and one
  // particle whose owning rank is its right neighbor.
  const std::vector<Particle> localInput{
      makeParticle(static_cast<unsigned long>(200 + rank), static_cast<double>(rank) + 0.75),
      makeParticle(static_cast<unsigned long>(100 + rank), static_cast<double>(rightRank) + 0.25),
  };

  particles.addDistributedParticles(localInput);

  EXPECT_EQ(particles.getGlobalNumberOfOwnedParticles(), 8);
  EXPECT_EQ(particles.getLocalNumberOfOwnedParticles(), 2);

  const auto leftRank = (rank + runtime.size() - 1) % runtime.size();
  auto expectedIds =
      std::vector<unsigned long>{static_cast<unsigned long>(200 + rank), static_cast<unsigned long>(100 + leftRank)};
  std::sort(expectedIds.begin(), expectedIds.end());
  EXPECT_EQ(ownedIds(particles), expectedIds);

  particles.finalize();
}

TEST(DistributedAutoPasTest, ReducesDoubleValuesGlobally) {
  int argc = 0;
  char **argv = nullptr;
  dap::Runtime runtime(argc, argv);
  ASSERT_EQ(runtime.size(), 4);

  dap::DistributedAutoPas<Particle> particles(runtime, globalMin, globalMax, cutoff, makeConfigurator());

  const double localValue = 0.25 * static_cast<double>(runtime.rank() + 1);
  EXPECT_DOUBLE_EQ(particles.globalSum(localValue), 2.5);

  particles.finalize();
}

TEST(DistributedAutoPasTest, ComputesPeriodicForcesWithSingleRank) { expectForceTestResult(1); }

TEST(DistributedAutoPasTest, ComputesSamePeriodicForcesAcrossFourRanks) { expectForceTestResult(4); }

}  // namespace
