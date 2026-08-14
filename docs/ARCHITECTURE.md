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

## Current limitations

- static 1D decomposition in x
- periodic migration / halo handling in x only
- no distributed load balancing
- halo exchange currently uses the cutoff width only
- `localAutoPas()` is still used by legacy md-flexible helpers
- MPI is the only communication backend
