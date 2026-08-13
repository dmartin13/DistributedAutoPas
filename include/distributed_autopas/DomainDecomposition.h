#pragma once

#include <mpi.h>

#include <array>

namespace dap {

class DomainDecomposition {
 public:
  DomainDecomposition(MPI_Comm comm, std::array<double, 3> globalMin, std::array<double, 3> globalMax)
      : _comm(comm), _globalMin(globalMin), _globalMax(globalMax) {
    MPI_Comm_rank(_comm, &_rank);
    MPI_Comm_size(_comm, &_numRanks);

    // MVP: 1D decomposition along x-direction.
    const double lengthX = _globalMax[0] - _globalMin[0];
    const double dx = lengthX / static_cast<double>(_numRanks);

    _localMin = _globalMin;
    _localMax = _globalMax;

    _localMin[0] = _globalMin[0] + _rank * dx;
    _localMax[0] = _globalMin[0] + (_rank + 1) * dx;
  }

  void applyPeriodicBoundary(std::array<double, 3> &pos) const {
    const double lengthX = _globalMax[0] - _globalMin[0];

    while (pos[0] < _globalMin[0]) {
      pos[0] += lengthX;
    }

    while (pos[0] >= _globalMax[0]) {
      pos[0] -= lengthX;
    }
  }

  [[nodiscard]] bool isInsideLocalDomain(const std::array<double, 3> &pos) const {
    return pos[0] >= _localMin[0] && pos[0] < _localMax[0] && pos[1] >= _localMin[1] && pos[1] < _localMax[1] &&
           pos[2] >= _localMin[2] && pos[2] < _localMax[2];
  }

  [[nodiscard]] bool isInsideHaloRegion(const std::array<double, 3> &pos, double haloWidth) const {
    return pos[0] >= _localMin[0] - haloWidth && pos[0] < _localMax[0] + haloWidth && pos[1] >= _localMin[1] &&
           pos[1] < _localMax[1] && pos[2] >= _localMin[2] && pos[2] < _localMax[2] && !isInsideLocalDomain(pos);
  }

  [[nodiscard]] int targetRank(const std::array<double, 3> &pos) const {
    const double lengthX = _globalMax[0] - _globalMin[0];
    const double relativeX = (pos[0] - _globalMin[0]) / lengthX;

    int target = static_cast<int>(relativeX * _numRanks);

    if (target < 0) {
      target = 0;
    }

    if (target >= _numRanks) {
      target = _numRanks - 1;
    }

    return target;
  }

  [[nodiscard]] int leftNeighbor() const { return (_rank - 1 + _numRanks) % _numRanks; }

  [[nodiscard]] int rightNeighbor() const { return (_rank + 1) % _numRanks; }

  [[nodiscard]] int rank() const { return _rank; }

  [[nodiscard]] int numRanks() const { return _numRanks; }

  [[nodiscard]] const std::array<double, 3> &globalMin() const { return _globalMin; }

  [[nodiscard]] const std::array<double, 3> &globalMax() const { return _globalMax; }

  [[nodiscard]] const std::array<double, 3> &localMin() const { return _localMin; }

  [[nodiscard]] const std::array<double, 3> &localMax() const { return _localMax; }

 private:
  MPI_Comm _comm{MPI_COMM_WORLD};

  int _rank{0};
  int _numRanks{1};

  std::array<double, 3> _globalMin{};
  std::array<double, 3> _globalMax{};

  std::array<double, 3> _localMin{};
  std::array<double, 3> _localMax{};
};

}  // namespace dap
