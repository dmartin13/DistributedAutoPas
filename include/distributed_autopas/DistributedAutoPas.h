#pragma once

#include <array>
#include <utility>
#include <vector>

#include "autopas/AutoPas.h"
#include "autopas/options/IteratorBehavior.h"
#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/HaloExchange.h"
#include "distributed_autopas/ParticleMigration.h"
#include "distributed_autopas/ParticleSerialization.h"

namespace dap {

/**
 * Distributed wrapper around an AutoPas container.
 *
 * Particle is intentionally a template parameter so that applications such as md-flexible can use their native
 * AutoPas-compatible particle type. Serializer controls how this particle is represented during MPI communication.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
class DistributedAutoPas {
 public:
  DistributedAutoPas(MPI_Comm comm, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff)
      : DistributedAutoPas(comm, globalMin, globalMax, cutoff, [](auto &) {}) {}

  /**
   * Construct and configure the local AutoPas instance before it is initialized.
   *
   * The configurator receives autopas::AutoPas<Particle>&. This avoids hard-coding all AutoPas configuration options
   * into DistributedAutoPas while we are still prototyping the public API.
   */
  template <class Configurator>
  DistributedAutoPas(MPI_Comm comm, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff, Configurator &&configurator)
      : _domain(comm, globalMin, globalMax), _particleMigration(comm), _haloExchange(comm), _cutoff(cutoff) {
    _autoPas.setBoxMin(_domain.localMin());
    _autoPas.setBoxMax(_domain.localMax());
    _autoPas.setCutoff(_cutoff);

    std::forward<Configurator>(configurator)(_autoPas);

    _autoPas.init();
  }

  void addParticle(const Particle &particle) { _autoPas.addParticle(particle); }

  /**
   * Synchronize the distributed particle state and execute an ordinary AutoPas functor.
   *
   * No functor adapter is used. The simulator can use the same AutoPas functors that it would use with a local
   * autopas::AutoPas<Particle> container.
   */
  template <class Functor>
  bool computeInteractions(Functor *functor) {
    synchronizeParticles();
    return _autoPas.computeInteractions(functor);
  }

  void finalize() { _autoPas.finalize(); }

  [[nodiscard]] DomainDecomposition &domain() { return _domain; }
  [[nodiscard]] const DomainDecomposition &domain() const { return _domain; }

  /**
   * Temporary access to the local AutoPas instance.
   *
   * This keeps the first md-flexible integration small. As the DistributedAutoPas API matures, commonly used AutoPas
   * operations can be forwarded directly and this escape hatch can be reduced or removed.
   */
  [[nodiscard]] autopas::AutoPas<Particle> &localAutoPas() { return _autoPas; }
  [[nodiscard]] const autopas::AutoPas<Particle> &localAutoPas() const { return _autoPas; }

 private:
  void synchronizeParticles() {
    auto emigrants = _autoPas.updateContainer();
    auto immigrants = _particleMigration.migrate(emigrants, _domain);
    _autoPas.addParticles(immigrants);

    std::vector<Particle> ownedParticles;
    ownedParticles.reserve(_autoPas.getNumberOfParticles(autopas::IteratorBehavior::owned));

    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      ownedParticles.push_back(*iter);
    }

    auto haloParticles = _haloExchange.exchange(ownedParticles, _domain, _cutoff);
    _autoPas.addHaloParticles(haloParticles);
  }

  DomainDecomposition _domain;
  ParticleMigration<Particle, Serializer> _particleMigration;
  HaloExchange<Particle, Serializer> _haloExchange;
  autopas::AutoPas<Particle> _autoPas;
  double _cutoff;
};

}  // namespace dap
