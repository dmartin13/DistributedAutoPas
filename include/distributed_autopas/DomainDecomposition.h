#pragma once

#include <array>

#include "distributed_autopas/BoundaryType.h"

namespace dap {

/**
 * Static regular-grid domain decomposition.
 *
 * The decomposition itself is independent of the communication backend. Runtime
 * provides the process index and number of processes during construction.
 */
class DomainDecomposition {
 public:
  static constexpr int noNeighbor = -1;

  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax);

  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      const std::array<bool, 3> &subdivideDimensions);

  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      std::array<int, 3> processGrid);

  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      const std::array<bool, 3> &subdivideDimensions, const std::array<BoundaryType, 3> &boundaryTypes);

  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      std::array<int, 3> processGrid, const std::array<BoundaryType, 3> &boundaryTypes);

  [[nodiscard]] static std::array<int, 3> generateProcessGrid(int numRanks,
                                                              const std::array<bool, 3> &subdivideDimensions);

  /** Wrap coordinates in every periodic dimension into the global simulation box. */
  void applyPeriodicBoundary(std::array<double, 3> &pos) const;

  /** Wrap one coordinate if the selected dimension is periodic. */
  void applyPeriodicBoundary(std::array<double, 3> &pos, int dimension) const;

  [[nodiscard]] bool isInsideLocalDomain(const std::array<double, 3> &pos) const;

  [[nodiscard]] bool isInsideHaloRegion(const std::array<double, 3> &pos, double haloWidth) const;

  [[nodiscard]] int targetRank(const std::array<double, 3> &pos) const;

  [[nodiscard]] std::array<int, 3> rankToCoordinates(int rank) const;

  [[nodiscard]] int coordinatesToRank(const std::array<int, 3> &coordinates) const;

  /**
   * Return the neighbor rank for a Cartesian grid offset.
   *
   * Periodic dimensions wrap around the process grid. At a global boundary with
   * a non-periodic boundary type (`none` or `reflective`), no neighbor exists and
   * noNeighbor is returned.
   */
  [[nodiscard]] int neighborRank(const std::array<int, 3> &offset) const;

  [[nodiscard]] int precedingNeighbor(int dimension) const;

  [[nodiscard]] int succeedingNeighbor(int dimension) const;

  [[nodiscard]] int leftNeighbor() const;

  [[nodiscard]] int rightNeighbor() const;

  [[nodiscard]] int rank() const;

  [[nodiscard]] int numRanks() const;

  [[nodiscard]] const std::array<int, 3> &processGrid() const;

  [[nodiscard]] const std::array<int, 3> &coordinates() const;

  [[nodiscard]] const std::array<BoundaryType, 3> &boundaryTypes() const;

  [[nodiscard]] BoundaryType boundaryType(int dimension) const;

  [[nodiscard]] const std::array<double, 3> &globalMin() const;

  [[nodiscard]] const std::array<double, 3> &globalMax() const;

  [[nodiscard]] const std::array<double, 3> &localMin() const;

  [[nodiscard]] const std::array<double, 3> &localMax() const;

 private:
  int _rank{0};
  int _numRanks{1};

  std::array<int, 3> _processGrid{1, 1, 1};
  std::array<int, 3> _coordinates{0, 0, 0};
  std::array<BoundaryType, 3> _boundaryTypes{BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic};

  std::array<double, 3> _globalMin{};
  std::array<double, 3> _globalMax{};

  std::array<double, 3> _localMin{};
  std::array<double, 3> _localMax{};
};

}  // namespace dap
