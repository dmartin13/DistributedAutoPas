#include "distributed_autopas/DomainDecomposition.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace dap {

DomainDecomposition::DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin,
                                         std::array<double, 3> globalMax)
    : DomainDecomposition(rank, numRanks, globalMin, globalMax, std::array<int, 3>{numRanks, 1, 1}) {}

DomainDecomposition::DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin,
                                         std::array<double, 3> globalMax,
                                         const std::array<bool, 3> &subdivideDimensions)
    : DomainDecomposition(rank, numRanks, globalMin, globalMax, generateProcessGrid(numRanks, subdivideDimensions)) {}

DomainDecomposition::DomainDecomposition(int rank, int numRanks, std::array<double, 3> globalMin,
                                         std::array<double, 3> globalMax, std::array<int, 3> processGrid)
    : _rank(rank), _numRanks(numRanks), _processGrid(processGrid), _globalMin(globalMin), _globalMax(globalMax) {
  if (_numRanks <= 0) {
    throw std::invalid_argument("DistributedAutoPas: number of ranks must be positive.");
  }

  if (_rank < 0 or _rank >= _numRanks) {
    throw std::invalid_argument("DistributedAutoPas: rank is outside the valid range.");
  }

  long long gridSize = 1;
  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    if (_processGrid[dimension] <= 0) {
      throw std::invalid_argument("DistributedAutoPas: process-grid dimensions must be positive.");
    }

    if (_globalMax[dimension] <= _globalMin[dimension]) {
      throw std::invalid_argument("DistributedAutoPas: global box must have positive extent in every dimension.");
    }

    gridSize *= _processGrid[dimension];
  }

  if (gridSize != _numRanks) {
    throw std::invalid_argument("DistributedAutoPas: process-grid size must equal the number of ranks.");
  }

  _coordinates = rankToCoordinates(_rank);

  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    const double globalLength = _globalMax[dimension] - _globalMin[dimension];
    const double localLength = globalLength / static_cast<double>(_processGrid[dimension]);

    _localMin[dimension] = _globalMin[dimension] + static_cast<double>(_coordinates[dimension]) * localLength;
    _localMax[dimension] = _coordinates[dimension] == _processGrid[dimension] - 1
                               ? _globalMax[dimension]
                               : _globalMin[dimension] + static_cast<double>(_coordinates[dimension] + 1) * localLength;
  }
}

std::array<int, 3> DomainDecomposition::generateProcessGrid(int numRanks,
                                                            const std::array<bool, 3> &subdivideDimensions) {
  if (numRanks <= 0) {
    throw std::invalid_argument("DistributedAutoPas: number of ranks must be positive.");
  }

  const auto dimensionsToSubdivide =
      static_cast<std::size_t>(std::count(subdivideDimensions.begin(), subdivideDimensions.end(), true));

  if (numRanks > 1 and dimensionsToSubdivide == 0) {
    throw std::invalid_argument("DistributedAutoPas: at least one dimension must be subdivided for multiple ranks.");
  }

  std::vector<int> factors;
  int remainingRanks = numRanks;

  while (remainingRanks % 2 == 0) {
    factors.push_back(2);
    remainingRanks /= 2;
  }

  for (int divisor = 3; divisor <= remainingRanks; divisor += 2) {
    while (remainingRanks % divisor == 0) {
      factors.push_back(divisor);
      remainingRanks /= divisor;
    }
  }

  while (factors.size() > dimensionsToSubdivide) {
    std::sort(factors.begin(), factors.end());
    const int smallestFactor = factors.front();
    factors.erase(factors.begin());
    factors.front() *= smallestFactor;
  }

  std::array<int, 3> processGrid{1, 1, 1};
  std::size_t factorIndex = 0;
  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    if (subdivideDimensions[dimension] and factorIndex < factors.size()) {
      processGrid[dimension] = factors[factorIndex++];
    }
  }

  return processGrid;
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
  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    if (pos[dimension] < _localMin[dimension] or pos[dimension] >= _localMax[dimension]) {
      return false;
    }
  }
  return true;
}

bool DomainDecomposition::isInsideHaloRegion(const std::array<double, 3> &pos, double haloWidth) const {
  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    const double lowerBound = _processGrid[dimension] > 1 ? _localMin[dimension] - haloWidth : _localMin[dimension];
    const double upperBound = _processGrid[dimension] > 1 ? _localMax[dimension] + haloWidth : _localMax[dimension];

    if (pos[dimension] < lowerBound or pos[dimension] >= upperBound) {
      return false;
    }
  }

  return not isInsideLocalDomain(pos);
}

int DomainDecomposition::targetRank(const std::array<double, 3> &pos) const {
  std::array<int, 3> targetCoordinates{};

  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    const double globalLength = _globalMax[dimension] - _globalMin[dimension];
    const double relativePosition = (pos[dimension] - _globalMin[dimension]) / globalLength;

    int coordinate = static_cast<int>(relativePosition * _processGrid[dimension]);
    coordinate = std::clamp(coordinate, 0, _processGrid[dimension] - 1);
    targetCoordinates[dimension] = coordinate;
  }

  return coordinatesToRank(targetCoordinates);
}

std::array<int, 3> DomainDecomposition::rankToCoordinates(int rank) const {
  if (rank < 0 or rank >= _numRanks) {
    throw std::invalid_argument("DistributedAutoPas: rank is outside the valid range.");
  }

  std::array<int, 3> coordinates{};
  int remainingRank = rank;

  const int yzPlaneSize = _processGrid[1] * _processGrid[2];
  coordinates[0] = remainingRank / yzPlaneSize;
  remainingRank -= coordinates[0] * yzPlaneSize;
  coordinates[1] = remainingRank / _processGrid[2];
  coordinates[2] = remainingRank % _processGrid[2];

  return coordinates;
}

int DomainDecomposition::coordinatesToRank(const std::array<int, 3> &coordinates) const {
  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    if (coordinates[dimension] < 0 or coordinates[dimension] >= _processGrid[dimension]) {
      throw std::invalid_argument("DistributedAutoPas: process-grid coordinates are outside the valid range.");
    }
  }

  return coordinates[0] * _processGrid[1] * _processGrid[2] + coordinates[1] * _processGrid[2] + coordinates[2];
}

int DomainDecomposition::neighborRank(const std::array<int, 3> &offset) const {
  auto neighborCoordinates = _coordinates;

  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    const int dimensionSize = _processGrid[dimension];
    const int shiftedCoordinate = neighborCoordinates[dimension] + offset[dimension];
    neighborCoordinates[dimension] = ((shiftedCoordinate % dimensionSize) + dimensionSize) % dimensionSize;
  }

  return coordinatesToRank(neighborCoordinates);
}

int DomainDecomposition::precedingNeighbor(int dimension) const {
  if (dimension < 0 or dimension >= 3) {
    throw std::invalid_argument("DistributedAutoPas: neighbor dimension must be 0, 1, or 2.");
  }

  std::array<int, 3> offset{};
  offset[dimension] = -1;
  return neighborRank(offset);
}

int DomainDecomposition::succeedingNeighbor(int dimension) const {
  if (dimension < 0 or dimension >= 3) {
    throw std::invalid_argument("DistributedAutoPas: neighbor dimension must be 0, 1, or 2.");
  }

  std::array<int, 3> offset{};
  offset[dimension] = 1;
  return neighborRank(offset);
}

int DomainDecomposition::leftNeighbor() const { return precedingNeighbor(0); }

int DomainDecomposition::rightNeighbor() const { return succeedingNeighbor(0); }

int DomainDecomposition::rank() const { return _rank; }

int DomainDecomposition::numRanks() const { return _numRanks; }

const std::array<int, 3> &DomainDecomposition::processGrid() const { return _processGrid; }

const std::array<int, 3> &DomainDecomposition::coordinates() const { return _coordinates; }

const std::array<double, 3> &DomainDecomposition::globalMin() const { return _globalMin; }

const std::array<double, 3> &DomainDecomposition::globalMax() const { return _globalMax; }

const std::array<double, 3> &DomainDecomposition::localMin() const { return _localMin; }

const std::array<double, 3> &DomainDecomposition::localMax() const { return _localMax; }

}  // namespace dap
