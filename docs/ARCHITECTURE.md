# DistributedAutoPas Architecture

## Responsibility split

The intended dependency direction is:

```text
Simulator / md-flexible
        |
        | particle operations and AutoPas interaction functors
        v
DistributedAutoPas
        |
        +-- distributed ownership / domain decomposition
        +-- migration
        +-- halo management
        +-- load balancing (future)
        +-- distributed algorithm selection (future)
        +-- communication backend
        |
        +--> node-local AutoPas
```

The simulator is executed once per process/rank, but it should not contain MPI,
NCCL, NVSHMEM, or other backend calls. It describes the physics and invokes
operations on the distributed particle system.

## Public API direction

The public API should describe operations, not local storage details. Examples:

```cpp
particles.applyToOwnedParticles(PositionUpdate{dt});
auto localKineticEnergy = particles.sumOwnedParticles(0.0, KineticEnergy{});
particles.computeInteractions(&ljFunctor);
auto globalCount = particles.getGlobalNumberOfOwnedParticles();
```

Local AutoPas iterators, halo exchange, migration, and communication contexts
are implementation details. `localAutoPas()` remains only as a temporary escape
hatch while md-flexible helper classes are migrated and is marked deprecated.

## Runtime

`dap::Runtime` owns the process communication runtime. The current backend is
MPI, but the application does not call MPI directly. Runtime creates a private
communication context for DistributedAutoPas so library messages do not collide
with communication from other components.

A future backend design may separate control communication from particle data
transport, e.g. MPI for process/topology control and CUDA-aware MPI, NCCL, or
NVSHMEM for GPU-resident particle data.

## Particle-local reductions

Simulator components such as the thermostat express local reductions through
`sumOwnedParticles()` and distributed reductions through `globalSum()`. They do not
perform MPI collectives themselves. This keeps communication policy inside
DistributedAutoPas while still allowing the simulator to define the physical
quantity that should be accumulated.

## Current limitations

- static 1D decomposition in x
- periodic migration / halo handling in x only
- no distributed load balancing
- halo exchange currently uses the cutoff width only
- `localAutoPas()` is still used by the legacy md-flexible VTK writer
- MPI is the only communication backend

## Bundled example application

The adapted md-flexible simulator lives in `examples/md-flexible`. It is built
as an application of DistributedAutoPas and may still depend on AutoPas for
particle types and interaction functors. Distributed communication is intended
to move behind the DistributedAutoPas API incrementally.

## Compiled and templated components

DistributedAutoPas is intentionally a mixed compiled/template library. Particle-dependent components remain in headers so arbitrary application particle types can instantiate them. Backend-independent, non-template implementation such as `DomainDecomposition` and non-template `Runtime` operations is compiled into the `distributed_autopas` library from `src/`.

The current `Runtime` header still contains the templated reduction helper and the private MPI communicator type. Moving the concrete communication backend fully behind an internal interface is a later communication-layer refactoring and is deliberately separate from this step.
