#include "distributed_autopas/Runtime.h"

#include <stdexcept>

namespace dap {

Runtime::Runtime(int &argc, char **&argv) {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (not initialized) {
    MPI_Init(&argc, &argv);
    _ownsBackendLifetime = true;
  }

  // Give DistributedAutoPas its own communication context. This prevents its
  // messages from colliding with communication performed by another library.
  if (MPI_Comm_dup(MPI_COMM_WORLD, &_communicator) != MPI_SUCCESS) {
    throw std::runtime_error("DistributedAutoPas: failed to create communication context.");
  }

  MPI_Comm_rank(_communicator, &_rank);
  MPI_Comm_size(_communicator, &_size);
}

Runtime::~Runtime() {
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized) {
    return;
  }

  if (_communicator != MPI_COMM_NULL) {
    MPI_Comm_free(&_communicator);
  }

  if (_ownsBackendLifetime) {
    MPI_Finalize();
  }
}

int Runtime::rank() const { return _rank; }

int Runtime::size() const { return _size; }

bool Runtime::isRoot() const { return _rank == 0; }

MPI_Comm Runtime::communicator() const { return _communicator; }

void Runtime::barrier() const { MPI_Barrier(_communicator); }

}  // namespace dap
