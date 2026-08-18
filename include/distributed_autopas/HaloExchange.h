#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <vector>

#include "distributed_autopas/Communication.h"
#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/ParticleSerialization.h"

namespace dap {

/**
 * Exchange halo particles for a Cartesian process grid.
 *
 * The exchange is staged dimension by dimension (x -> y -> z). Particles
 * received in an earlier dimension participate in later stages. This propagates
 * face halos to edge and corner neighbors without communicating directly with
 * all 26 possible neighbors.
 *
 * The current topology is periodic in every decomposed dimension. The legacy
 * single-rank periodic x behavior is preserved as well. General per-dimension
 * boundary conditions are handled separately at a higher level.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
class HaloExchange {
 public:
  explicit HaloExchange(MPI_Comm comm) : _comm(comm) {}

  [[nodiscard]] std::vector<Particle> exchange(const std::vector<Particle> &ownedParticles,
                                               const DomainDecomposition &domain, double haloWidth) const {
    // Candidates are the particles that may have to be forwarded in later stages.
    // Initially these are the owned particles. After every stage, newly received
    // halos are appended so that edge and corner halos are generated naturally.
    std::vector<Particle> candidates = ownedParticles;
    std::vector<Particle> haloParticles;

    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
      if (domain.processGrid()[dimension] == 1 and dimension != 0) {
        continue;
      }

      std::vector<Particle> sendPreceding;
      std::vector<Particle> sendSucceeding;

      for (const auto &particle : candidates) {
        const auto &position = particle.getR();

        if (position[dimension] < domain.localMin()[dimension] + haloWidth) {
          sendPreceding.push_back(particle);
        }

        if (position[dimension] >= domain.localMax()[dimension] - haloWidth) {
          sendSucceeding.push_back(particle);
        }
      }

      auto exchange =
          exchangeLeftRight<Particle, Serializer>(_comm, domain.precedingNeighbor(static_cast<int>(dimension)),
                                                  domain.succeedingNeighbor(static_cast<int>(dimension)), sendPreceding,
                                                  sendSucceeding, 300 + static_cast<int>(dimension) * 10);

      shiftPeriodicImages(exchange.recvFromLeft, domain, dimension, true);
      shiftPeriodicImages(exchange.recvFromRight, domain, dimension, false);

      const auto previousCandidateCount = candidates.size();
      candidates.reserve(previousCandidateCount + exchange.recvFromLeft.size() + exchange.recvFromRight.size());
      haloParticles.reserve(haloParticles.size() + exchange.recvFromLeft.size() + exchange.recvFromRight.size());

      haloParticles.insert(haloParticles.end(), exchange.recvFromLeft.begin(), exchange.recvFromLeft.end());
      haloParticles.insert(haloParticles.end(), exchange.recvFromRight.begin(), exchange.recvFromRight.end());

      candidates.insert(candidates.end(), exchange.recvFromLeft.begin(), exchange.recvFromLeft.end());
      candidates.insert(candidates.end(), exchange.recvFromRight.begin(), exchange.recvFromRight.end());
    }

    return haloParticles;
  }

 private:
  static void shiftPeriodicImages(std::vector<Particle> &particles, const DomainDecomposition &domain,
                                  std::size_t dimension, bool receivedFromPreceding) {
    const bool crossesPeriodicBoundary = receivedFromPreceding
                                             ? domain.coordinates()[dimension] == 0
                                             : domain.coordinates()[dimension] == domain.processGrid()[dimension] - 1;

    if (not crossesPeriodicBoundary) {
      return;
    }

    const double globalLength = domain.globalMax()[dimension] - domain.globalMin()[dimension];
    const double shift = receivedFromPreceding ? -globalLength : globalLength;

    for (auto &particle : particles) {
      auto position = particle.getR();
      position[dimension] += shift;
      particle.setR(position);
    }
  }

  MPI_Comm _comm{MPI_COMM_WORLD};
};

}  // namespace dap
