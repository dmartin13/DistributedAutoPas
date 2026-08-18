#pragma once

#include <mpi.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "distributed_autopas/Communication.h"
#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/ParticleSerialization.h"

namespace dap {

/**
 * Migrate particles between direct neighbors of a Cartesian process grid.
 *
 * Migration is performed dimension by dimension. Particles can move by at most
 * one subdomain in each dimension per timestep, while diagonal moves are routed
 * through multiple face-neighbor exchanges within the same migration call.
 *
 * Periodic boundaries wrap particles to the opposite side of the global box.
 * BoundaryType::none discards particles that leave the global box. Reflective
 * boundaries are currently rejected by DomainDecomposition.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
class ParticleMigration {
 public:
  explicit ParticleMigration(MPI_Comm comm) : _comm(comm) {}

  [[nodiscard]] std::vector<Particle> migrate(const std::vector<Particle> &emigrants,
                                              const DomainDecomposition &domain) const {
    auto remainingParticles = emigrants;

    for (std::size_t dimension = 0; dimension < 3; ++dimension) {
      if (domain.processGrid()[dimension] == 1) {
        std::vector<Particle> particlesInsideBoundary;
        particlesInsideBoundary.reserve(remainingParticles.size());

        for (auto particle : remainingParticles) {
          auto position = particle.getR();

          if (domain.boundaryType(static_cast<int>(dimension)) == BoundaryType::periodic) {
            domain.applyPeriodicBoundary(position, static_cast<int>(dimension));
            particle.setR(position);
            particlesInsideBoundary.push_back(std::move(particle));
          } else if (position[dimension] >= domain.globalMin()[dimension] and
                     position[dimension] < domain.globalMax()[dimension]) {
            particlesInsideBoundary.push_back(std::move(particle));
          }
          // BoundaryType::none deliberately drops particles outside the global box.
        }

        remainingParticles = std::move(particlesInsideBoundary);
        continue;
      }

      std::vector<Particle> sendPreceding;
      std::vector<Particle> sendSucceeding;
      std::vector<Particle> stayForNextDimension;

      sendPreceding.reserve(remainingParticles.size() / 3);
      sendSucceeding.reserve(remainingParticles.size() / 3);
      stayForNextDimension.reserve(remainingParticles.size());

      for (auto particle : remainingParticles) {
        auto position = particle.getR();

        if (position[dimension] < domain.localMin()[dimension]) {
          if (domain.coordinates()[dimension] == 0 and
              domain.boundaryType(static_cast<int>(dimension)) == BoundaryType::periodic) {
            domain.applyPeriodicBoundary(position, static_cast<int>(dimension));
            particle.setR(position);
          }
          sendPreceding.push_back(std::move(particle));
        } else if (position[dimension] >= domain.localMax()[dimension]) {
          if (domain.coordinates()[dimension] == domain.processGrid()[dimension] - 1 and
              domain.boundaryType(static_cast<int>(dimension)) == BoundaryType::periodic) {
            domain.applyPeriodicBoundary(position, static_cast<int>(dimension));
            particle.setR(position);
          }
          sendSucceeding.push_back(std::move(particle));
        } else {
          stayForNextDimension.push_back(std::move(particle));
        }
      }

      const auto exchange =
          exchangeLeftRight<Particle, Serializer>(_comm, domain.precedingNeighbor(static_cast<int>(dimension)),
                                                  domain.succeedingNeighbor(static_cast<int>(dimension)), sendPreceding,
                                                  sendSucceeding, 100 + static_cast<int>(dimension) * 10);

      stayForNextDimension.reserve(stayForNextDimension.size() + exchange.recvFromLeft.size() +
                                   exchange.recvFromRight.size());
      stayForNextDimension.insert(stayForNextDimension.end(), exchange.recvFromLeft.begin(),
                                  exchange.recvFromLeft.end());
      stayForNextDimension.insert(stayForNextDimension.end(), exchange.recvFromRight.begin(),
                                  exchange.recvFromRight.end());

      remainingParticles = std::move(stayForNextDimension);
    }

    for (const auto &particle : remainingParticles) {
      if (not domain.isInsideLocalDomain(particle.getR())) {
        throw std::runtime_error("Particle moved more than one subdomain in one timestep.");
      }
    }

    return remainingParticles;
  }

 private:
  MPI_Comm _comm{MPI_COMM_WORLD};
};

}  // namespace dap
