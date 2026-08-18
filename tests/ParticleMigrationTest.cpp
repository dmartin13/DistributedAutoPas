#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <stdexcept>
#include <vector>

#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/Particle.h"
#include "distributed_autopas/ParticleMigration.h"

namespace {

using Particle = dap::Particle;

constexpr std::array<double, 3> globalMin{0., 0., 0.};
constexpr std::array<double, 3> globalMax{4., 1., 1.};

Particle makeParticle(unsigned long id, double x) {
  Particle particle;
  particle.setID(id);
  particle.setR({x, 0.5, 0.5});
  return particle;
}

Particle makeParticle(unsigned long id, const std::array<double, 3> &position) {
  Particle particle;
  particle.setID(id);
  particle.setR(position);
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
  dap::ParticleMigration<Particle> migration;

  TestContext() : domain(makeDomain(rank, numberOfRanks)), migration(MPI_COMM_WORLD) {}
};

TEST(ParticleMigrationTest, KeepsParticlesThatRemainInLocalDomain) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  const double x = context.domain.localMin()[0] + 0.5;
  const std::vector<Particle> emigrants{makeParticle(100 + context.rank, x)};

  const auto immigrants = context.migration.migrate(emigrants, context.domain);

  ASSERT_EQ(immigrants.size(), 1);
  EXPECT_EQ(immigrants.front().getID(), 100 + context.rank);
  EXPECT_EQ(immigrants.front().getR(), (std::array<double, 3>{x, 0.5, 0.5}));
}

TEST(ParticleMigrationTest, MigratesToLeftNeighbor) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  const double x = context.domain.localMin()[0] - 0.25;
  const std::vector<Particle> emigrants{makeParticle(200 + context.rank, x)};

  const auto immigrants = context.migration.migrate(emigrants, context.domain);

  ASSERT_EQ(immigrants.size(), 1);
  const int sourceRank = context.domain.rightNeighbor();
  EXPECT_EQ(immigrants.front().getID(), 200 + sourceRank);
  EXPECT_EQ(immigrants.front().getR(), (std::array<double, 3>{context.rank + 0.75, 0.5, 0.5}));
}

TEST(ParticleMigrationTest, MigratesToRightNeighbor) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  const double x = context.domain.localMax()[0] + 0.25;
  const std::vector<Particle> emigrants{makeParticle(300 + context.rank, x)};

  const auto immigrants = context.migration.migrate(emigrants, context.domain);

  ASSERT_EQ(immigrants.size(), 1);
  const int sourceRank = context.domain.leftNeighbor();
  EXPECT_EQ(immigrants.front().getID(), 300 + sourceRank);
  EXPECT_EQ(immigrants.front().getR(), (std::array<double, 3>{context.rank + 0.25, 0.5, 0.5}));
}

TEST(ParticleMigrationTest, MigratesAcrossPeriodicLeftBoundary) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  std::vector<Particle> emigrants;
  if (context.rank == 0) {
    emigrants.push_back(makeParticle(400, -0.25));
  }

  const auto immigrants = context.migration.migrate(emigrants, context.domain);

  if (context.rank == 3) {
    ASSERT_EQ(immigrants.size(), 1);
    EXPECT_EQ(immigrants.front().getID(), 400);
    EXPECT_EQ(immigrants.front().getR(), (std::array<double, 3>{3.75, 0.5, 0.5}));
  } else {
    EXPECT_TRUE(immigrants.empty());
  }
}

TEST(ParticleMigrationTest, MigratesAcrossPeriodicRightBoundary) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  std::vector<Particle> emigrants;
  if (context.rank == 3) {
    emigrants.push_back(makeParticle(500, 4.25));
  }

  const auto immigrants = context.migration.migrate(emigrants, context.domain);

  if (context.rank == 0) {
    ASSERT_EQ(immigrants.size(), 1);
    EXPECT_EQ(immigrants.front().getID(), 500);
    EXPECT_EQ(immigrants.front().getR(), (std::array<double, 3>{0.25, 0.5, 0.5}));
  } else {
    EXPECT_TRUE(immigrants.empty());
  }
}

TEST(ParticleMigrationTest, RejectsMovementAcrossMoreThanOneSubdomain) {
  TestContext context;
  ASSERT_EQ(context.numberOfRanks, 4);

  std::vector<Particle> emigrants;
  if (context.rank == 0) {
    emigrants.push_back(makeParticle(600, 2.5));
  }

  // The particle can move only one x-subdomain during this migration call. It
  // therefore reaches rank 1 but is still outside rank 1's local box. The error
  // is detected only after all neighbor exchanges have completed, avoiding an MPI
  // deadlock on the error path.
  if (context.rank == 1) {
    EXPECT_THROW(context.migration.migrate(emigrants, context.domain), std::runtime_error);
  } else {
    EXPECT_NO_THROW(context.migration.migrate(emigrants, context.domain));
  }
}

struct ThreeDimensionalTestContext {
  int rank{0};
  int numberOfRanks{0};
  dap::DomainDecomposition domain;
  dap::ParticleMigration<Particle> migration;

  ThreeDimensionalTestContext()
      : domain([&]() {
          MPI_Comm_rank(MPI_COMM_WORLD, &rank);
          MPI_Comm_size(MPI_COMM_WORLD, &numberOfRanks);
          return dap::DomainDecomposition(rank, numberOfRanks, {0., 0., 0.}, {2., 2., 2.}, std::array<int, 3>{2, 2, 2});
        }()),
        migration(MPI_COMM_WORLD) {}
};

TEST(ParticleMigrationTest, MigratesAcrossYFaceInThreeDimensions) {
  ThreeDimensionalTestContext context;
  ASSERT_EQ(context.numberOfRanks, 8);

  std::vector<Particle> emigrants;
  if (context.rank == 0) {
    emigrants.push_back(makeParticle(700, {0.5, 1.1, 0.5}));
  }

  const auto immigrants = context.migration.migrate(emigrants, context.domain);

  if (context.rank == 2) {
    ASSERT_EQ(immigrants.size(), 1);
    EXPECT_EQ(immigrants.front().getID(), 700);
    EXPECT_EQ(immigrants.front().getR(), (std::array<double, 3>{0.5, 1.1, 0.5}));
  } else {
    EXPECT_TRUE(immigrants.empty());
  }
}

TEST(ParticleMigrationTest, MigratesAcrossZFaceInThreeDimensions) {
  ThreeDimensionalTestContext context;
  ASSERT_EQ(context.numberOfRanks, 8);

  std::vector<Particle> emigrants;
  if (context.rank == 0) {
    emigrants.push_back(makeParticle(800, {0.5, 0.5, 1.1}));
  }

  const auto immigrants = context.migration.migrate(emigrants, context.domain);

  if (context.rank == 1) {
    ASSERT_EQ(immigrants.size(), 1);
    EXPECT_EQ(immigrants.front().getID(), 800);
    EXPECT_EQ(immigrants.front().getR(), (std::array<double, 3>{0.5, 0.5, 1.1}));
  } else {
    EXPECT_TRUE(immigrants.empty());
  }
}

TEST(ParticleMigrationTest, RoutesEdgeMigrationThroughTwoDimensions) {
  ThreeDimensionalTestContext context;
  ASSERT_EQ(context.numberOfRanks, 8);

  std::vector<Particle> emigrants;
  if (context.rank == 0) {
    emigrants.push_back(makeParticle(900, {0.5, 1.1, 1.1}));
  }

  const auto immigrants = context.migration.migrate(emigrants, context.domain);

  if (context.rank == 3) {
    ASSERT_EQ(immigrants.size(), 1);
    EXPECT_EQ(immigrants.front().getID(), 900);
    EXPECT_EQ(immigrants.front().getR(), (std::array<double, 3>{0.5, 1.1, 1.1}));
  } else {
    EXPECT_TRUE(immigrants.empty());
  }
}

TEST(ParticleMigrationTest, RoutesCornerMigrationThroughThreeDimensions) {
  ThreeDimensionalTestContext context;
  ASSERT_EQ(context.numberOfRanks, 8);

  std::vector<Particle> emigrants;
  if (context.rank == 0) {
    emigrants.push_back(makeParticle(1000, {1.1, 1.1, 1.1}));
  }

  const auto immigrants = context.migration.migrate(emigrants, context.domain);

  if (context.rank == 7) {
    ASSERT_EQ(immigrants.size(), 1);
    EXPECT_EQ(immigrants.front().getID(), 1000);
    EXPECT_EQ(immigrants.front().getR(), (std::array<double, 3>{1.1, 1.1, 1.1}));
  } else {
    EXPECT_TRUE(immigrants.empty());
  }
}

}  // namespace
