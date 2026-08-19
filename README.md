# DistributedAutoPas

DistributedAutoPas is a distributed particle-container layer around AutoPas.
It keeps domain ownership, migration, halo exchange, and communication out of
the simulator while reusing AutoPas for node-local particle storage and
interaction traversal.

The current static regular-grid implementation supports Cartesian decompositions in x, y, and z. Boundary conditions are selected per dimension. `periodic`, `reflective`, and `none` are represented in the distributed topology. Reflective boundaries are non-periodic for migration and halo exchange. The bundled single-site md-flexible example implements the physical reflective boundary as an application-level Lennard-Jones mirror-wall force and restricts that work to wall-adjacent particle regions through the DistributedAutoPas operation API.

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
The communication, particle-migration, halo-exchange, and public DistributedAutoPas API tests are MPI GoogleTests. CTest launches the distributed tests with the required rank count, including the split-phase non-blocking neighbor-communication API, migration through that non-blocking transport, one- and four-rank public-API force checks, a region-operation check, an eight-rank 3D ownership check, and eight-rank 3D migration and halo-exchange checks.
All tests can be run with:

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```


### md-flexible 3D force regression

A small end-to-end regression input is available at
`examples/md-flexible/input/dap_force_compare_3d.yaml`. It compares a single-rank
reference run with an eight-rank `2 x 2 x 2` decomposition and exercises internal
face, edge, and corner interactions as well as periodic boundaries in all three
dimensions.

From `build/examples/md-flexible`:

```bash
rm -rf force_compare_3d/np1 force_compare_3d/np8
mkdir -p force_compare_3d/np1 force_compare_3d/np8

./md-flexible \
  --yaml-filename input/dap_force_compare_3d.yaml \
  --vtk-output-folder force_compare_3d/np1

mpirun -np 8 ./md-flexible \
  --yaml-filename input/dap_force_compare_3d.yaml \
  --vtk-output-folder force_compare_3d/np8

python3 scripts/compare_vtk_forces.py \
  force_compare_3d/np1 \
  force_compare_3d/np8
```

The comparison should report the same owned-particle IDs and forces within the
configured tolerance.


### md-flexible reflective-boundary regression

The single-site md-flexible example contains a reflective-wall regression input at
`examples/md-flexible/input/dap_reflective_compare.yaml`. It places particles near
all six global faces and two opposite corners. With sigma = epsilon = 1 and a wall
distance of 0.5, the expected force component from each reflective wall is exactly
24. The same setup can be compared between one rank and an eight-rank `2 x 2 x 2`
decomposition.

From `build/examples/md-flexible`:

```bash
rm -rf reflective_compare/np1 reflective_compare/np8
mkdir -p reflective_compare/np1 reflective_compare/np8

./md-flexible \
  --yaml-filename input/dap_reflective_compare.yaml \
  --vtk-output-folder reflective_compare/np1

mpirun -np 8 ./md-flexible \
  --yaml-filename input/dap_reflective_compare.yaml \
  --vtk-output-folder reflective_compare/np8

python3 scripts/compare_vtk_forces.py \
  reflective_compare/np1 \
  reflective_compare/np8

python3 scripts/check_reflective_forces.py reflective_compare/np8
```

The first script checks decomposition independence. The second checks the forces
against the analytical values of the reflective regression setup.


The test build can be disabled with `-DDISTRIBUTED_AUTOPAS_BUILD_TESTS=OFF`.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current architecture
and transition plan.
