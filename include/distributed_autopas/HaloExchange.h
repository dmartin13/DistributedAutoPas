#pragma once

#include <mpi.h>

#include <vector>

#include "distributed_autopas/Communication.h"
#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/ParticleSerialization.h"

namespace dap {

template <class Particle, class Serializer = ParticleSerializer<Particle>>
class HaloExchange {
 public:
  explicit HaloExchange(MPI_Comm comm) : _comm(comm) {}

  [[nodiscard]] std::vector<Particle> exchange(const std::vector<Particle> &ownedParticles,
                                                const DomainDecomposition &domain, double haloWidth) const {
    std::vector<Particle> sendLeft;
    std::vector<Particle> sendRight;

    for (const auto &particle : ownedParticles) {
      if (particle.getR()[0] < domain.localMin()[0] + haloWidth) {
        sendLeft.push_back(particle);
      }

      if (particle.getR()[0] >= domain.localMax()[0] - haloWidth) {
        sendRight.push_back(particle);
      }
    }

    auto exchange =
        exchangeLeftRight<Particle, Serializer>(_comm, domain.leftNeighbor(), domain.rightNeighbor(), sendLeft,
                                                sendRight, 300);

    const double lengthX = domain.globalMax()[0] - domain.globalMin()[0];

    if (domain.rank() == 0) {
      for (auto &particle : exchange.recvFromLeft) {
        auto position = particle.getR();
        position[0] -= lengthX;
        particle.setR(position);
      }
    }

    if (domain.rank() == domain.numRanks() - 1) {
      for (auto &particle : exchange.recvFromRight) {
        auto position = particle.getR();
        position[0] += lengthX;
        particle.setR(position);
      }
    }

    std::vector<Particle> haloParticles;
    haloParticles.reserve(exchange.recvFromLeft.size() + exchange.recvFromRight.size());
    haloParticles.insert(haloParticles.end(), exchange.recvFromLeft.begin(), exchange.recvFromLeft.end());
    haloParticles.insert(haloParticles.end(), exchange.recvFromRight.begin(), exchange.recvFromRight.end());

    return haloParticles;
  }

 private:
  MPI_Comm _comm{MPI_COMM_WORLD};
};

}  // namespace dap
