#include "distributed_autopas/DomainDecomposition.h"

#include <stdexcept>

namespace dap {

DomainDecomposition::DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin,
                                         std::array<double, 3> globalMax)
    : _rank(rank), _numRanks(numRanks), _globalMin(globalMin), _globalMax(globalMax) {
  if (_numRanks <= 0) {
    throw std::invalid_argument("DistributedAutoPas: number of ranks must be positive.");
  }

  if (_rank < 0 or _rank >= _numRanks) {
    throw std::invalid_argument("DistributedAutoPas: rank is outside the valid range.");
  }

  // MVP: 1D decomposition along x-direction.
  const double lengthX = _globalMax[0] - _globalMin[0];
  const double dx = lengthX / static_cast<double>(_numRanks);

  _localMin = _globalMin;
  _localMax = _globalMax;

  _localMin[0] = _globalMin[0] + _rank * dx;
  _localMax[0] = _globalMin[0] + (_rank + 1) * dx;
}

void DomainDecomposition::applyPeriodicBoundary(std::array<double, 3> &pos) const {
  const double lengthX = _globalMax[0] - _globalMin[0];

  while (pos[0] < _globalMin[0]) {
    pos[0] += lengthX;
  }

  while (pos[0] >= _globalMax[0]) {
    pos[0] -= lengthX;
  }
}

bool DomainDecomposition::isInsideLocalDomain(const std::array<double, 3> &pos) const {
  return pos[0] >= _localMin[0] and pos[0] < _localMax[0] and pos[1] >= _localMin[1] and pos[1] < _localMax[1] and
         pos[2] >= _localMin[2] and pos[2] < _localMax[2];
}

bool DomainDecomposition::isInsideHaloRegion(const std::array<double, 3> &pos, double haloWidth) const {
  return pos[0] >= _localMin[0] - haloWidth and pos[0] < _localMax[0] + haloWidth and pos[1] >= _localMin[1] and
         pos[1] < _localMax[1] and pos[2] >= _localMin[2] and pos[2] < _localMax[2] and not isInsideLocalDomain(pos);
}

int DomainDecomposition::targetRank(const std::array<double, 3> &pos) const {
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

int DomainDecomposition::leftNeighbor() const { return (_rank - 1 + _numRanks) % _numRanks; }

int DomainDecomposition::rightNeighbor() const { return (_rank + 1) % _numRanks; }

int DomainDecomposition::rank() const { return _rank; }

int DomainDecomposition::numRanks() const { return _numRanks; }

const std::array<double, 3> &DomainDecomposition::globalMin() const { return _globalMin; }

const std::array<double, 3> &DomainDecomposition::globalMax() const { return _globalMax; }

const std::array<double, 3> &DomainDecomposition::localMin() const { return _localMin; }

const std::array<double, 3> &DomainDecomposition::localMax() const { return _localMax; }

}  // namespace dap
