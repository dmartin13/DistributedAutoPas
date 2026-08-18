#pragma once

#include <array>

namespace dap {

/**
 * Static regular-grid domain decomposition.
 *
 * The decomposition itself is independent of the communication backend. Runtime
 * provides the process index and number of processes during construction.
 *
 * The legacy constructor without a process grid keeps the current DistributedAutoPas
 * behavior and decomposes only along x. Additional constructors can represent an
 * arbitrary 3D Cartesian process grid or derive one from a subdivision mask.
 */
class DomainDecomposition {
 public:
  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax);

  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      const std::array<bool, 3> &subdivideDimensions);

  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax,
                      std::array<int, 3> processGrid);

  [[nodiscard]] static std::array<int, 3> generateProcessGrid(int numRanks,
                                                              const std::array<bool, 3> &subdivideDimensions);

  /**
   * Apply the periodic boundary used by the current 1D migration path.
   *
   * General per-dimension boundary handling will be introduced together with the
   * 3D migration and halo-exchange implementation. For now this deliberately keeps
   * the existing x-periodic behavior.
   */
  void applyPeriodicBoundary(std::array<double, 3> &pos) const;

  [[nodiscard]] bool isInsideLocalDomain(const std::array<double, 3> &pos) const;

  [[nodiscard]] bool isInsideHaloRegion(const std::array<double, 3> &pos, double haloWidth) const;

  [[nodiscard]] int targetRank(const std::array<double, 3> &pos) const;

  [[nodiscard]] std::array<int, 3> rankToCoordinates(int rank) const;

  [[nodiscard]] int coordinatesToRank(const std::array<int, 3> &coordinates) const;

  /**
   * Return the periodically wrapped neighbor rank for a Cartesian grid offset.
   * Offsets can describe faces, edges, or corners, e.g. {-1, 0, 1}.
   */
  [[nodiscard]] int neighborRank(const std::array<int, 3> &offset) const;

  [[nodiscard]] int precedingNeighbor(int dimension) const;

  [[nodiscard]] int succeedingNeighbor(int dimension) const;

  // Compatibility helpers for the current 1D migration and halo exchange.
  [[nodiscard]] int leftNeighbor() const;

  [[nodiscard]] int rightNeighbor() const;

  [[nodiscard]] int rank() const;

  [[nodiscard]] int numRanks() const;

  [[nodiscard]] const std::array<int, 3> &processGrid() const;

  [[nodiscard]] const std::array<int, 3> &coordinates() const;

  [[nodiscard]] const std::array<double, 3> &globalMin() const;

  [[nodiscard]] const std::array<double, 3> &globalMax() const;

  [[nodiscard]] const std::array<double, 3> &localMin() const;

  [[nodiscard]] const std::array<double, 3> &localMax() const;

 private:
  int _rank{0};
  int _numRanks{1};

  std::array<int, 3> _processGrid{1, 1, 1};
  std::array<int, 3> _coordinates{0, 0, 0};

  std::array<double, 3> _globalMin{};
  std::array<double, 3> _globalMax{};

  std::array<double, 3> _localMin{};
  std::array<double, 3> _localMax{};
};

}  // namespace dap
