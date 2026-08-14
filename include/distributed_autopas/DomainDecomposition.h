#pragma once

#include <array>

namespace dap {

/**
 * Static 1D domain decomposition used by the current DistributedAutoPas prototype.
 *
 * The decomposition itself is independent of the communication backend. Runtime
 * provides the process index and number of processes during construction.
 */
class DomainDecomposition {
 public:
  DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin, std::array<double, 3> globalMax);

  void applyPeriodicBoundary(std::array<double, 3> &pos) const;

  [[nodiscard]] bool isInsideLocalDomain(const std::array<double, 3> &pos) const;

  [[nodiscard]] bool isInsideHaloRegion(const std::array<double, 3> &pos, double haloWidth) const;

  [[nodiscard]] int targetRank(const std::array<double, 3> &pos) const;

  [[nodiscard]] int leftNeighbor() const;

  [[nodiscard]] int rightNeighbor() const;

  [[nodiscard]] int rank() const;

  [[nodiscard]] int numRanks() const;

  [[nodiscard]] const std::array<double, 3> &globalMin() const;

  [[nodiscard]] const std::array<double, 3> &globalMax() const;

  [[nodiscard]] const std::array<double, 3> &localMin() const;

  [[nodiscard]] const std::array<double, 3> &localMax() const;

 private:
  int _rank{0};
  int _numRanks{1};

  std::array<double, 3> _globalMin{};
  std::array<double, 3> _globalMax{};

  std::array<double, 3> _localMin{};
  std::array<double, 3> _localMax{};
};

}  // namespace dap
