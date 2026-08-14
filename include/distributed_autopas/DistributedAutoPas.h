#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "autopas/AutoPas.h"
#include "autopas/options/IteratorBehavior.h"
#include "autopas/utils/WrapOpenMP.h"
#include "distributed_autopas/Communication.h"
#include "distributed_autopas/DomainDecomposition.h"
#include "distributed_autopas/HaloExchange.h"
#include "distributed_autopas/ParticleMigration.h"
#include "distributed_autopas/ParticleSerialization.h"
#include "distributed_autopas/Runtime.h"

namespace dap {

/**
 * Distributed particle container built around a node-local AutoPas instance.
 *
 * The public interface is intentionally expressed in terms of operations on a
 * distributed particle system rather than in terms of MPI or local AutoPas
 * iterators. The simulator describes what should happen to owned particles and
 * DistributedAutoPas decides how the local storage and communication are handled.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
class DistributedAutoPas {
 public:
  DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff)
      : DistributedAutoPas(runtime, globalMin, globalMax, cutoff, [](auto &) {}) {}

  /**
   * Construct and configure the node-local AutoPas instance before initialization.
   *
   * The configurator is currently an explicit customization point for AutoPas' local
   * tuning configuration. It is intentionally separate from all distributed concerns.
   */
  template <class Configurator>
  DistributedAutoPas(Runtime &runtime, const std::array<double, 3> &globalMin, const std::array<double, 3> &globalMax,
                     double cutoff, Configurator &&configurator)
      : _runtime(runtime),
        _domain(runtime.rank(), runtime.size(), globalMin, globalMax),
        _particleMigration(runtime.communicator()),
        _haloExchange(runtime.communicator()),
        _cutoff(cutoff) {
    _autoPas.setBoxMin(_domain.localMin());
    _autoPas.setBoxMax(_domain.localMax());
    _autoPas.setCutoff(_cutoff);

    std::forward<Configurator>(configurator)(_autoPas);

    _autoPas.init();
  }

  void addParticle(const Particle &particle) { _autoPas.addParticle(particle); }

  /**
   * Redistribute application-provided particles according to the current domain decomposition
   * and add the particles owned by this rank to the local AutoPas container.
   *
   * Every rank contributes its local input collection and must call this collective operation.
   * Particles are sent directly to their owning rank, so the application does not need a particle
   * communicator or access to the distributed communication context. This operation is intended
   * for initialization and checkpoint loading, not for timestep migration.
   *
   * Particles outside the global simulation box are ignored. The application can detect such
   * input through its global particle-count sanity check.
   */
  template <class Collection>
  void addDistributedParticles(const Collection &particles) {
    std::vector<std::vector<Particle>> particlesByDestination(static_cast<std::size_t>(_domain.numRanks()));

    for (const auto &particle : particles) {
      const auto &position = particle.getR();
      const auto &globalMin = _domain.globalMin();
      const auto &globalMax = _domain.globalMax();

      const bool insideGlobalDomain = position[0] >= globalMin[0] and position[0] < globalMax[0] and
                                      position[1] >= globalMin[1] and position[1] < globalMax[1] and
                                      position[2] >= globalMin[2] and position[2] < globalMax[2];
      if (not insideGlobalDomain) {
        continue;
      }

      particlesByDestination[static_cast<std::size_t>(_domain.targetRank(position))].push_back(particle);
    }

    auto localParticles =
        exchangeParticlesByRank<Particle, Serializer>(_runtime.communicator(), particlesByDestination);
    _autoPas.addParticles(localParticles);
  }

  /**
   * Execute a particle-local kernel for every particle owned by this process.
   *
   * The kernel must be safe to execute concurrently for different particles. Today
   * this is implemented with the node-local AutoPas iterator and OpenMP. Keeping the
   * operation at this level allows a future implementation to dispatch the same
   * operation to a GPU without exposing container iterators to the simulator.
   */
  template <class Kernel>
  void applyToOwnedParticles(Kernel &&kernel) {
    auto &&kernelRef = kernel;
    AUTOPAS_OPENMP(parallel)
    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      kernelRef(*iter);
    }
  }

  /**
   * Sum a particle-local quantity over all particles owned by this process.
   *
   * This is a local reduction. Distributed reductions are intentionally kept
   * separate via globalSum(), so callers can combine several local quantities
   * before triggering communication. The operation hides the node-local AutoPas
   * iterator and can later be implemented by a device reduction.
   */
  template <class Value, class Transform>
  [[nodiscard]] Value sumOwnedParticles(Value initialValue, Transform &&transform) const {
    auto result = initialValue;
    auto &&transformRef = transform;
    AUTOPAS_OPENMP(parallel reduction(+ : result))
    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      result += transformRef(*iter);
    }
    return result;
  }

  /**
   * Read-only traversal intended for diagnostics and output. Unlike
   * applyToOwnedParticles(), this is deliberately sequential to preserve output order.
   */
  template <class Visitor>
  void forEachOwnedParticle(Visitor &&visitor) const {
    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      visitor(*iter);
    }
  }

  /**
   * Synchronize the distributed particle state and execute an ordinary AutoPas functor.
   *
   * Migration and halo creation are implementation details of this operation. This
   * keeps the simulator independent of the concrete communication algorithm.
   */
  template <class Functor>
  bool computeInteractions(Functor *functor) {
    synchronizeParticles();
    return _autoPas.computeInteractions(functor);
  }

  [[nodiscard]] std::size_t getLocalNumberOfOwnedParticles() const {
    return _autoPas.getNumberOfParticles(autopas::IteratorBehavior::owned);
  }

  [[nodiscard]] std::size_t getLocalNumberOfHaloParticles() const {
    return _autoPas.getNumberOfParticles(autopas::IteratorBehavior::halo);
  }

  [[nodiscard]] std::size_t getLocalNumberOfParticles() const {
    return _autoPas.getNumberOfParticles(autopas::IteratorBehavior::ownedOrHalo);
  }

  [[nodiscard]] std::size_t getGlobalNumberOfOwnedParticles() const {
    return globalSum(getLocalNumberOfOwnedParticles());
  }

  [[nodiscard]] int rank() const { return _runtime.rank(); }
  [[nodiscard]] int numberOfRanks() const { return _runtime.size(); }
  [[nodiscard]] bool isRoot() const { return _runtime.isRoot(); }

  void barrier() const { _runtime.barrier(); }

  template <class T>
  [[nodiscard]] T globalSum(T localValue) const {
    return _runtime.globalSum(localValue);
  }

  [[nodiscard]] const std::array<double, 3> &localBoxMin() const { return _domain.localMin(); }
  [[nodiscard]] const std::array<double, 3> &localBoxMax() const { return _domain.localMax(); }
  [[nodiscard]] const std::array<double, 3> &globalBoxMin() const { return _domain.globalMin(); }
  [[nodiscard]] const std::array<double, 3> &globalBoxMax() const { return _domain.globalMax(); }

  // Local AutoPas metadata that is still used by md-flexible for tuning statistics.
  // These methods deliberately expose values, not the local container itself.
  [[nodiscard]] bool localSearchSpaceIsTrivial() const { return _autoPas.searchSpaceIsTrivial(); }
  [[nodiscard]] double getMeanLocalRebuildFrequency() { return _autoPas.getMeanRebuildFrequency(); }
  [[nodiscard]] auto getVerletSkin() { return _autoPas.getVerletSkin(); }
  [[nodiscard]] auto getVerletRebuildFrequency() { return _autoPas.getVerletRebuildFrequency(); }

  /**
   * Return the currently active node-local AutoPas configurations for diagnostics.
   *
   * This exposes tuning metadata without exposing the local AutoPas container itself.
   */
  [[nodiscard]] auto getCurrentLocalConfigurations() const { return _autoPas.getCurrentConfigs(); }

  void finalize() { _autoPas.finalize(); }

 private:
  void synchronizeParticles() {
    auto emigrants = _autoPas.updateContainer();
    auto immigrants = _particleMigration.migrate(emigrants, _domain);
    _autoPas.addParticles(immigrants);

    std::vector<Particle> ownedParticles;
    ownedParticles.reserve(getLocalNumberOfOwnedParticles());

    for (auto iter = _autoPas.begin(autopas::IteratorBehavior::owned); iter.isValid(); ++iter) {
      ownedParticles.push_back(*iter);
    }

    auto haloParticles = _haloExchange.exchange(ownedParticles, _domain, _cutoff);
    _autoPas.addHaloParticles(haloParticles);
  }

  Runtime &_runtime;
  DomainDecomposition _domain;
  ParticleMigration<Particle, Serializer> _particleMigration;
  HaloExchange<Particle, Serializer> _haloExchange;
  autopas::AutoPas<Particle> _autoPas;
  double _cutoff;
};

}  // namespace dap
