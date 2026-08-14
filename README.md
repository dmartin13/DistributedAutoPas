# DistributedAutoPas

DistributedAutoPas is a distributed particle-container layer around AutoPas.
It keeps domain ownership, migration, halo exchange, and communication out of
the simulator while reusing AutoPas for node-local particle storage and
interaction traversal.

## Repository layout

```text
DistributedAutoPas/
├── include/distributed_autopas/   # DistributedAutoPas library
├── examples/
│   ├── minimal_mpi.cpp            # small integration example
│   └── md-flexible/               # md-flexible example application
├── docs/
└── tests/
```

The expected development workspace is:

```text
<workspace>/
├── AutoPas/
└── DistributedAutoPas/
```

## Build

From the DistributedAutoPas repository:

```bash
cmake -S . -B build \
  -DAUTOPAS_SOURCE_DIR=../AutoPas \
  -DMD_FLEXIBLE_USE_MPI=ON \
  -DMD_FLEXIBLE_FUNCTOR_AUTOVEC=ON

cmake --build build -j
```

The md-flexible executable is then available at:

```text
build/examples/md-flexible/md-flexible
```

and the smaller LJ integration example at:

```text
build/examples/md-flexible/distributed-lj-example
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current architecture
and transition plan.
