#include <mpi.h>

#include <array>
#include <iostream>

#include "autopas/options/ContainerOption.h"
#include "autopas/options/DataLayoutOption.h"
#include "autopas/options/IteratorBehavior.h"
#include "autopas/options/Newton3Option.h"
#include "autopas/options/TraversalOption.h"
#include "distributed_autopas/DistributedAutoPas.h"
#include "distributed_autopas/PairCountingFunctor.h"
#include "distributed_autopas/Particle.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int numRanks = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  const std::array<double, 3> globalMin{0.0, 0.0, 0.0};
  const std::array<double, 3> globalMax{10.0, 10.0, 10.0};

  constexpr double cutoff = 0.5;

  // The example uses one deterministic AoS configuration because PairCountingFunctor
  // intentionally implements only the AoS path.
  dap::DistributedAutoPas<dap::Particle> distributedAutoPas(
      MPI_COMM_WORLD, globalMin, globalMax, cutoff, [](auto &autoPas) {
        autoPas.setAllowedContainers({autopas::ContainerOption::directSum});
        autoPas.setAllowedTraversals({autopas::TraversalOption::ds_sequential});
        autoPas.setAllowedDataLayouts({autopas::DataLayoutOption::aos});
        autoPas.setAllowedNewton3Options({autopas::Newton3Option::enabled});
        autoPas.setVerletSkin(0.0);
      });

  for (int i = 0; i < 10; ++i) {
    dap::Particle particle;
    particle.setID(rank * 1000 + i);

    const double x0 = distributedAutoPas.domain().localMin()[0];
    const double x1 = distributedAutoPas.domain().localMax()[0];

    particle.setR({x0 + (x1 - x0) * (i + 0.5) / 10.0, 5.0, 5.0});
    distributedAutoPas.addParticle(particle);
  }

  // Temporary direct access to the local AutoPas container. md-flexible already
  // performs its position update through AutoPas iterators, so this is sufficient
  // for the first integration step.
  for (auto iter = distributedAutoPas.localAutoPas().begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
    auto position = iter->getR();
    position[0] += 1.0;
    iter->setR(position);
  }

  dap::PairCountingFunctor<dap::Particle> pairCounter(cutoff, distributedAutoPas.domain().localMin(),
                                                        distributedAutoPas.domain().localMax());

  // This is now the only call needed before an interaction calculation.
  // DistributedAutoPas performs migration and halo exchange internally and then
  // forwards the ordinary AutoPas functor directly to AutoPas.
  distributedAutoPas.computeInteractions(&pairCounter);

  MPI_Barrier(MPI_COMM_WORLD);

  for (int r = 0; r < numRanks; ++r) {
    if (r == rank) {
      auto &autoPas = distributedAutoPas.localAutoPas();

      std::cout << "Rank " << rank << " owns "
                << autoPas.getNumberOfParticles(autopas::IteratorBehavior::owned) << " particles and has "
                << autoPas.getNumberOfParticles(autopas::IteratorBehavior::halo) << " halo particles.\n";

      std::cout << "  owned particle ids: ";
      for (auto iter = autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
        std::cout << iter->getID() << " ";
      }
      std::cout << "\n";

      std::cout << "  halo particles (id:x): ";
      for (auto iter = autoPas.begin(autopas::IteratorBehavior::halo); iter.isValid(); ++iter) {
        std::cout << iter->getID() << ":" << iter->getR()[0] << " ";
      }
      std::cout << "\n";

      std::cout << "  interactions within cutoff: " << pairCounter.numPairs() << "\n";
      std::cout << "    owned-owned: " << pairCounter.numOwnedOwnedPairs() << "\n";
      std::cout << "    owned-halo:  " << pairCounter.numOwnedHaloPairs() << "\n";
      std::cout << "    halo-halo:   " << pairCounter.numHaloHaloPairs() << "\n";

      std::cout << "  owned-halo pairs: ";
      for (const auto &[idA, idB] : pairCounter.ownedHaloPairs()) {
        std::cout << "(" << idA << "," << idB << ") ";
      }
      std::cout << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  distributedAutoPas.finalize();
  MPI_Finalize();

  return 0;
}
