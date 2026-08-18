#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

#include "distributed_autopas/DomainDecomposition.h"

namespace {

constexpr std::array<double, 3> globalMin{0., 1., 2.};
constexpr std::array<double, 3> globalMax{10., 11., 12.};

TEST(DomainDecompositionTest, SplitsGlobalDomainAlongX) {
  constexpr int numRanks = 4;
  constexpr double subdomainLength = 2.5;

  for (int rank = 0; rank < numRanks; ++rank) {
    const dap::DomainDecomposition domain(rank, numRanks, globalMin, globalMax);

    EXPECT_EQ(domain.rank(), rank);
    EXPECT_EQ(domain.numRanks(), numRanks);
    EXPECT_EQ(domain.globalMin(), globalMin);
    EXPECT_EQ(domain.globalMax(), globalMax);

    EXPECT_DOUBLE_EQ(domain.localMin()[0], rank * subdomainLength);
    EXPECT_DOUBLE_EQ(domain.localMax()[0], (rank + 1) * subdomainLength);
    EXPECT_DOUBLE_EQ(domain.localMin()[1], globalMin[1]);
    EXPECT_DOUBLE_EQ(domain.localMax()[1], globalMax[1]);
    EXPECT_DOUBLE_EQ(domain.localMin()[2], globalMin[2]);
    EXPECT_DOUBLE_EQ(domain.localMax()[2], globalMax[2]);
  }
}

TEST(DomainDecompositionTest, RejectsInvalidRankInformation) {
  EXPECT_THROW((dap::DomainDecomposition{0, 0, globalMin, globalMax}), std::invalid_argument);
  EXPECT_THROW((dap::DomainDecomposition{-1, 4, globalMin, globalMax}), std::invalid_argument);
  EXPECT_THROW((dap::DomainDecomposition{4, 4, globalMin, globalMax}), std::invalid_argument);
}

TEST(DomainDecompositionTest, UsesPeriodicNeighbors) {
  const dap::DomainDecomposition firstRank(0, 4, globalMin, globalMax);
  EXPECT_EQ(firstRank.leftNeighbor(), 3);
  EXPECT_EQ(firstRank.rightNeighbor(), 1);

  const dap::DomainDecomposition middleRank(2, 4, globalMin, globalMax);
  EXPECT_EQ(middleRank.leftNeighbor(), 1);
  EXPECT_EQ(middleRank.rightNeighbor(), 3);

  const dap::DomainDecomposition lastRank(3, 4, globalMin, globalMax);
  EXPECT_EQ(lastRank.leftNeighbor(), 2);
  EXPECT_EQ(lastRank.rightNeighbor(), 0);
}

TEST(DomainDecompositionTest, LocalDomainUsesHalfOpenBounds) {
  const dap::DomainDecomposition domain(1, 4, globalMin, globalMax);

  EXPECT_TRUE(domain.isInsideLocalDomain({2.5, 1., 2.}));
  EXPECT_TRUE(domain.isInsideLocalDomain({4.999, 10.999, 11.999}));

  EXPECT_FALSE(domain.isInsideLocalDomain({2.499, 5., 5.}));
  EXPECT_FALSE(domain.isInsideLocalDomain({5., 5., 5.}));
  EXPECT_FALSE(domain.isInsideLocalDomain({3., 11., 5.}));
  EXPECT_FALSE(domain.isInsideLocalDomain({3., 5., 12.}));
}

TEST(DomainDecompositionTest, DetectsHaloRegionOutsideLocalDomain) {
  const dap::DomainDecomposition domain(1, 4, globalMin, globalMax);
  constexpr double haloWidth = 0.5;

  EXPECT_TRUE(domain.isInsideHaloRegion({2.25, 5., 5.}, haloWidth));
  EXPECT_TRUE(domain.isInsideHaloRegion({5.25, 5., 5.}, haloWidth));

  EXPECT_FALSE(domain.isInsideHaloRegion({2.5, 5., 5.}, haloWidth));
  EXPECT_FALSE(domain.isInsideHaloRegion({4.999, 5., 5.}, haloWidth));
  EXPECT_FALSE(domain.isInsideHaloRegion({1.999, 5., 5.}, haloWidth));
  EXPECT_FALSE(domain.isInsideHaloRegion({5.5, 5., 5.}, haloWidth));
  EXPECT_FALSE(domain.isInsideHaloRegion({2.25, 0.999, 5.}, haloWidth));
}

TEST(DomainDecompositionTest, DeterminesTargetRankFromXPosition) {
  const dap::DomainDecomposition domain(0, 4, globalMin, globalMax);

  EXPECT_EQ(domain.targetRank({0., 5., 5.}), 0);
  EXPECT_EQ(domain.targetRank({2.499, 5., 5.}), 0);
  EXPECT_EQ(domain.targetRank({2.5, 5., 5.}), 1);
  EXPECT_EQ(domain.targetRank({5., 5., 5.}), 2);
  EXPECT_EQ(domain.targetRank({7.5, 5., 5.}), 3);
  EXPECT_EQ(domain.targetRank({9.999, 5., 5.}), 3);
}

TEST(DomainDecompositionTest, AppliesPeriodicBoundaryInXOnly) {
  const dap::DomainDecomposition domain(0, 4, globalMin, globalMax);

  std::array<double, 3> below{-20.25, 4., 7.};
  domain.applyPeriodicBoundary(below);
  EXPECT_DOUBLE_EQ(below[0], 9.75);
  EXPECT_DOUBLE_EQ(below[1], 4.);
  EXPECT_DOUBLE_EQ(below[2], 7.);

  std::array<double, 3> above{30.25, 4., 7.};
  domain.applyPeriodicBoundary(above);
  EXPECT_DOUBLE_EQ(above[0], 0.25);
  EXPECT_DOUBLE_EQ(above[1], 4.);
  EXPECT_DOUBLE_EQ(above[2], 7.);

  std::array<double, 3> upperBoundary{10., 4., 7.};
  domain.applyPeriodicBoundary(upperBoundary);
  EXPECT_DOUBLE_EQ(upperBoundary[0], 0.);
}

}  // namespace
