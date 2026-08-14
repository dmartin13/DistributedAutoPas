# DistributedAutoPas

DistributedAutoPas is a distributed particle-container layer around AutoPas.
It keeps domain ownership, migration, halo exchange, and communication out of
the simulator while reusing AutoPas for node-local particle storage and
interaction traversal.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current architecture
and transition plan.
