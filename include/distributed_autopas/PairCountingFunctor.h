#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "autopas/baseFunctors/PairwiseFunctor.h"

namespace dap {

/**
 * Small AoS-only test functor that counts interacting particle pairs.
 *
 * This is an ordinary AutoPas functor. It is templated only so the same interaction test also verifies that
 * DistributedAutoPas is no longer tied to one concrete particle type.
 */
template <class Particle>
class PairCountingFunctor : public autopas::PairwiseFunctor<Particle, PairCountingFunctor<Particle>> {
 public:
  using SoAArraysType = typename Particle::SoAArraysType;
  using SoAView = autopas::SoAView<SoAArraysType>;
  using NeighborList = std::vector<size_t, autopas::AlignedAllocator<size_t>>;

  PairCountingFunctor(double cutoff, std::array<double, 3> localMin, std::array<double, 3> localMax)
      : autopas::PairwiseFunctor<Particle, PairCountingFunctor<Particle>>(cutoff),
        _cutoffSquared(cutoff * cutoff),
        _localMin(localMin),
        _localMax(localMax) {}

  void initTraversal() override {
    _numPairs = 0;
    _numOwnedOwnedPairs = 0;
    _numOwnedHaloPairs = 0;
    _numHaloHaloPairs = 0;
    _ownedHaloPairs.clear();
  }

  void AoSFunctor(Particle &particleA, Particle &particleB, bool /*newton3*/) override {
    const auto &rA = particleA.getR();
    const auto &rB = particleB.getR();

    const double dx = rA[0] - rB[0];
    const double dy = rA[1] - rB[1];
    const double dz = rA[2] - rB[2];
    const double distanceSquared = dx * dx + dy * dy + dz * dz;

    if (distanceSquared >= _cutoffSquared) {
      return;
    }

    ++_numPairs;

    const bool aOwned = isInsideLocalDomain(rA);
    const bool bOwned = isInsideLocalDomain(rB);

    if (aOwned and bOwned) {
      ++_numOwnedOwnedPairs;
    } else if (aOwned != bOwned) {
      ++_numOwnedHaloPairs;
      _ownedHaloPairs.emplace_back(particleA.getID(), particleB.getID());
    } else {
      ++_numHaloHaloPairs;
    }
  }

  void SoAFunctorSingle(SoAView /*soa*/, bool /*newton3*/) override {
    throw std::runtime_error("PairCountingFunctor only supports AoS.");
  }

  void SoAFunctorPair(SoAView /*soa1*/, SoAView /*soa2*/, bool /*newton3*/) override {
    throw std::runtime_error("PairCountingFunctor only supports AoS.");
  }

  void SoAFunctorVerlet(SoAView /*soa*/, size_t /*indexFirst*/, const NeighborList & /*neighborList*/,
                        bool /*newton3*/) override {
    throw std::runtime_error("PairCountingFunctor only supports AoS.");
  }

  [[nodiscard]] bool allowsNewton3() override { return true; }
  [[nodiscard]] bool allowsNonNewton3() override { return false; }
  [[nodiscard]] bool isRelevantForTuning() override { return true; }
  [[nodiscard]] std::string getName() override { return "PairCountingFunctor"; }

  [[nodiscard]] size_t numPairs() const { return _numPairs; }
  [[nodiscard]] size_t numOwnedOwnedPairs() const { return _numOwnedOwnedPairs; }
  [[nodiscard]] size_t numOwnedHaloPairs() const { return _numOwnedHaloPairs; }
  [[nodiscard]] size_t numHaloHaloPairs() const { return _numHaloHaloPairs; }
  [[nodiscard]] const std::vector<std::pair<unsigned long, unsigned long>> &ownedHaloPairs() const {
    return _ownedHaloPairs;
  }

 private:
  [[nodiscard]] bool isInsideLocalDomain(const std::array<double, 3> &position) const {
    return position[0] >= _localMin[0] and position[0] < _localMax[0] and position[1] >= _localMin[1] and
           position[1] < _localMax[1] and position[2] >= _localMin[2] and position[2] < _localMax[2];
  }

  double _cutoffSquared;
  std::array<double, 3> _localMin;
  std::array<double, 3> _localMax;
  size_t _numPairs{0};
  size_t _numOwnedOwnedPairs{0};
  size_t _numOwnedHaloPairs{0};
  size_t _numHaloHaloPairs{0};

  std::vector<std::pair<unsigned long, unsigned long>> _ownedHaloPairs;
};

}  // namespace dap
