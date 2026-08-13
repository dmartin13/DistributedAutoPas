#pragma once

#include <mpi.h>

#include <stdexcept>
#include <vector>

#include "distributed_autopas/Communication.h"
#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/ParticleSerialization.h"

namespace dap {

template <class Particle, class Serializer = ParticleSerializer<Particle>>
class ParticleMigration {
 public:
  explicit ParticleMigration(MPI_Comm comm) : _comm(comm) {}

  [[nodiscard]] std::vector<Particle> migrate(const std::vector<Particle> &emigrants,
                                               const DomainDecomposition &domain) const {
    std::vector<Particle> sendLeft;
    std::vector<Particle> sendRight;
    std::vector<Particle> immigrants;

    for (auto particle : emigrants) {
      auto position = particle.getR();
      domain.applyPeriodicBoundary(position);
      particle.setR(position);

      if (domain.isInsideLocalDomain(position)) {
        immigrants.push_back(std::move(particle));
        continue;
      }

      const int target = domain.targetRank(position);

      if (target == domain.leftNeighbor()) {
        sendLeft.push_back(std::move(particle));
      } else if (target == domain.rightNeighbor()) {
        sendRight.push_back(std::move(particle));
      } else {
        throw std::runtime_error("Particle moved more than one subdomain in one timestep.");
      }
    }

    auto exchange =
        exchangeLeftRight<Particle, Serializer>(_comm, domain.leftNeighbor(), domain.rightNeighbor(), sendLeft,
                                                sendRight, 100);

    immigrants.reserve(immigrants.size() + exchange.recvFromLeft.size() + exchange.recvFromRight.size());
    immigrants.insert(immigrants.end(), exchange.recvFromLeft.begin(), exchange.recvFromLeft.end());
    immigrants.insert(immigrants.end(), exchange.recvFromRight.begin(), exchange.recvFromRight.end());

    return immigrants;
  }

 private:
  MPI_Comm _comm{MPI_COMM_WORLD};
};

}  // namespace dap
