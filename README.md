# DistributedAutoPas

DistributedAutoPas is a distributed particle-container layer around AutoPas.
It keeps domain ownership, migration, halo exchange, and communication out of
the simulator while reusing AutoPas for node-local particle storage and
interaction traversal.

## Repository layout

```text
DistributedAutoPas/
├── include/distributed_autopas/   # public headers and template code
├── src/                           # compiled non-template implementation
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

## Tests

DistributedAutoPas tests use GoogleTest and are enabled by default for a standalone checkout.
The communication, particle-migration, halo-exchange, and public DistributedAutoPas API tests are MPI GoogleTests. CTest launches the distributed tests with the required rank count, including one- and four-rank public-API force checks, an eight-rank 3D ownership check, and eight-rank 3D migration checks.
All tests can be run with:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The test build can be disabled with `-DDISTRIBUTED_AUTOPAS_BUILD_TESTS=OFF`.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current architecture
and transition plan.
