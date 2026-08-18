#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <vector>

#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/HaloExchange.h"
#include "distributed_autopas/Particle.h"

namespace {

using Particle = dap::Particle;

constexpr std::array<double, 3> globalMin{0., 0., 0.};
constexpr std::array<double, 3> globalMax{4., 1., 1.};
constexpr double haloWidth = 0.2;

Particle makeParticle(unsigned long id, double x) {
  Particle particle;
  particle.setID(id);
  particle.setR({x, 0.5, 0.5});
  return particle;
}

dap::DomainDecomposition makeDomain(int &rank, int &numberOfRanks) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);
  return dap::DomainDecomposition(rank, numberOfRanks, globalMin, globalMax);
}

struct TestContext {
  int rank{0};
  int numberOfRanks{0};
  dap::DomainDecomposition domain;
  dap::HaloExchange<Particle> haloExchange;

  TestContext() : domain(makeDomain(rank, numberOfRanks)), haloExchange(MPI_COMM_WORLD) {}
};

TEST(HaloExchangeTest, ExchangesLeftAndRightBoundaryParticles) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  const auto leftParticleId = static_cast<unsigned long>(1000 + 10 * context.rank);
  const auto rightParticleId = leftParticleId + 1;

  const std::vector<Particle> ownedParticles{
      makeParticle(leftParticleId, context.domain.localMin()[0] + 0.1),
      makeParticle(rightParticleId, context.domain.localMax()[0] - 0.1),
  };

  const auto halos = context.haloExchange.exchange(ownedParticles, context.domain, haloWidth);

  ASSERT_EQ(halos.size(), 2);

  const int leftSource = context.domain.leftNeighbor();
  const int rightSource = context.domain.rightNeighbor();

  EXPECT_EQ(halos[0].getID(), static_cast<unsigned long>(1000 + 10 * leftSource + 1));
  EXPECT_EQ(halos[1].getID(), static_cast<unsigned long>(1000 + 10 * rightSource));

  double expectedLeftX = static_cast<double>(leftSource) + 0.9;
  if (context.rank == 0) {
    expectedLeftX -= globalMax[0] - globalMin[0];
  }

  double expectedRightX = static_cast<double>(rightSource) + 0.1;
  if (context.rank == context.numberOfRanks - 1) {
    expectedRightX += globalMax[0] - globalMin[0];
  }

  EXPECT_NEAR(halos[0].getR()[0], expectedLeftX, 1e-12);
  EXPECT_NEAR(halos[1].getR()[0], expectedRightX, 1e-12);
}

TEST(HaloExchangeTest, IgnoresParticlesOutsideHaloWidth) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  const std::vector<Particle> ownedParticles{
      makeParticle(2000 + context.rank, context.domain.localMin()[0] + 0.5),
  };

  const auto halos = context.haloExchange.exchange(ownedParticles, context.domain, haloWidth);

  EXPECT_TRUE(halos.empty());
}

TEST(HaloExchangeTest, ShiftsPeriodicLeftHaloBelowGlobalDomain) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  std::vector<Particle> ownedParticles;
  if (context.rank == context.numberOfRanks - 1) {
    ownedParticles.push_back(makeParticle(3000, globalMax[0] - 0.1));
  }

  const auto halos = context.haloExchange.exchange(ownedParticles, context.domain, haloWidth);

  if (context.rank == 0) {
    ASSERT_EQ(halos.size(), 1);
    EXPECT_EQ(halos.front().getID(), 3000);
    EXPECT_NEAR(halos.front().getR()[0], globalMin[0] - 0.1, 1e-12);
    EXPECT_LT(halos.front().getR()[0], globalMin[0]);
  } else {
    EXPECT_TRUE(halos.empty());
  }
}

TEST(HaloExchangeTest, ShiftsPeriodicRightHaloAboveGlobalDomain) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  std::vector<Particle> ownedParticles;
  if (context.rank == 0) {
    ownedParticles.push_back(makeParticle(4000, globalMin[0] + 0.1));
  }

  const auto halos = context.haloExchange.exchange(ownedParticles, context.domain, haloWidth);

  if (context.rank == context.numberOfRanks - 1) {
    ASSERT_EQ(halos.size(), 1);
    EXPECT_EQ(halos.front().getID(), 4000);
    EXPECT_NEAR(halos.front().getR()[0], globalMax[0] + 0.1, 1e-12);
    EXPECT_GT(halos.front().getR()[0], globalMax[0]);
  } else {
    EXPECT_TRUE(halos.empty());
  }
}

}  // namespace
