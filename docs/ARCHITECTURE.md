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
particles.applyToOwnedParticlesInRegion(wallMin, wallMax, WallOperation{});
auto localKineticEnergy = particles.sumOwnedParticles(0.0, KineticEnergy{});
particles.prepareInteractions();
applyApplicationBoundaryPhysics(particles);
particles.computeInteractionsPrepared(&ljFunctor);
auto globalCount = particles.getGlobalNumberOfOwnedParticles();
```

Local AutoPas iterators, halo exchange, migration, and communication contexts
are implementation details. `prepareInteractions()` exposes only the high-level
preparation phase, not the individual migration or halo operations. This allows an
application to insert physics such as a reflective wall force before invoking one or
more AutoPas functors on the prepared state. `computeInteractions()` remains the
convenience operation for the common prepare-and-compute case. The public API
intentionally provides no access to the node-local AutoPas container. Diagnostics
and output consume particle traversal and local tuning metadata through dedicated
DistributedAutoPas operations. Spatially restricted application kernels use
`applyToOwnedParticlesInRegion()`. DistributedAutoPas clips the requested region to
the rank-local ownership box and delegates the traversal to AutoPas without exposing
its region iterator to the application.


## Initial particle distribution

Particle insertion distinguishes between replicated and already distributed input. Replicated configuration input uses `addParticlesFromRoot()`, so only one logical copy is routed to owners. Checkpoint pieces that are already distributed across processes use `addDistributedParticles()`, where every process contributes its local input collection. In both cases, DistributedAutoPas determines rank ownership and the simulator does not use a particle communicator.

The current MPI implementation uses an all-to-all exchange because initialization may move a particle directly to any rank. This is intentionally separate from timestep migration, which currently assumes movement only to direct neighbors.

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
`sumOwnedParticles()` and distributed reductions through `globalSum()`. Output
coordination uses backend-neutral `Runtime` operations such as `broadcastString()`.
These components do not perform MPI collectives themselves. This keeps communication
policy inside DistributedAutoPas while still allowing the simulator to define the
physical quantity that should be accumulated or written.

## Current limitations

- `DistributedAutoPas` can construct static Cartesian process grids from a subdivision mask and uses them for local AutoPas boxes and initial particle ownership
- timestep migration supports Cartesian process grids through staged face-neighbor exchanges in x, y, and z
- halo exchange supports Cartesian process grids through staged face-neighbor exchanges in x, y, and z; halos received in earlier stages are forwarded to generate edge and corner halos
- boundary conditions are configured per dimension; `periodic`, `reflective`, and `none` are supported as distributed topology
- `reflective` is non-periodic for migration and halo exchange; crossing a reflective global boundary is reported as an error, while the physical reflective wall force remains application-level physics
- the bundled single-site md-flexible example applies its original Lennard-Jones mirror-wall model through a separate `ReflectiveBoundary` component between distributed preparation and ordinary AutoPas interaction functors; only thin wall-adjacent regions are traversed through `applyToOwnedParticlesInRegion()`
- the bundled md-flexible example forwards its `boundary-type` and `subdivide-dimension` settings to DistributedAutoPas
- no distributed load balancing
- halo exchange currently uses the cutoff width only
- MPI is the only communication backend

## Bundled example application

The adapted md-flexible simulator lives in `examples/md-flexible`. It is built
as an application of DistributedAutoPas and may still depend on AutoPas for
particle types and interaction functors. The active simulation path no longer
constructs or stores md-flexible's legacy `RegularGridDecomposition`; VTK domain
output obtains local/global bounds and the domain rank from DistributedAutoPas.
The old application-owned decomposition implementation and its dedicated tests have
been removed from the bundled md-flexible tree. DistributedAutoPas is now the only
active distributed ownership/decomposition layer in this example.

Particle insertion distinguishes between two input ownership models. For replicated
configuration input, `dap::Runtime` tells the application which process expands the
configured objects into particles. Object generation in `MDFlexConfig` therefore does
not query MPI or the current rank. Those particles are then inserted with
`addParticlesFromRoot()`, so only one logical
copy is routed to owners. Distributed checkpoint pieces use `addDistributedParticles()`,
where every rank contributes its local input collection.

## Compiled and templated components

DistributedAutoPas is intentionally a mixed compiled/template library. Particle-dependent components remain in headers so arbitrary application particle types can instantiate them. Backend-independent, non-template implementation such as `DomainDecomposition` and non-template `Runtime` operations is compiled into the `distributed_autopas` library from `src/`.

The current `Runtime` header still contains the templated reduction helper and the private MPI communicator type. Moving the concrete communication backend fully behind an internal interface is a later communication-layer refactoring and is deliberately separate from this step.


## Removed legacy md-flexible distributed layer

The bundled example no longer contains md-flexible's old `RegularGridDecomposition`,
`DomainTools`, `DomainDecomposition`, or `ParticleCommunicator` implementation. Their
dedicated domain-decomposition tests were removed together with them. Configuration
support such as `LoadBalancerOption` is kept for now because it is still referenced by
md-flexible's YAML/CLI configuration code, even though this prototype forces load
balancing to `None`.
