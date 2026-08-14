#pragma once

#include <mpi.h>

#include <cstddef>
#include <string>
#include <type_traits>

namespace dap {

template <class Particle, class Serializer>
class DistributedAutoPas;

namespace detail {
template <class>
inline constexpr bool alwaysFalse = false;

template <class T>
MPI_Datatype mpiDatatype() {
  if constexpr (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, int>) {
    return MPI_INT;
  } else if constexpr (std::is_same_v<T, long>) {
    return MPI_LONG;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return MPI_UNSIGNED_LONG;
  } else if constexpr (std::is_same_v<T, long long>) {
    return MPI_LONG_LONG_INT;
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    return MPI_UNSIGNED_LONG_LONG;
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (std::is_same_v<std::size_t, unsigned long>) {
      return MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<std::size_t, unsigned long long>) {
      return MPI_UNSIGNED_LONG_LONG;
    } else {
      static_assert(alwaysFalse<T>, "Unsupported std::size_t representation for MPI reduction.");
    }
  } else {
    static_assert(alwaysFalse<T>, "Unsupported type for DistributedAutoPas reduction.");
  }
}
}  // namespace detail

/**
 * Process runtime used by DistributedAutoPas.
 *
 * The application creates exactly one Runtime object. Runtime owns the communication
 * backend lifetime when it initialized it itself and provides DistributedAutoPas with
 * a private communication context. Application code therefore does not have to call
 * MPI_Init, MPI_Finalize, MPI_Comm_rank, or MPI_Comm_size directly.
 *
 * MPI is the first backend of this prototype. Backend-specific implementation details
 * that do not depend on a template type live in Runtime.cpp.
 */
class Runtime {
 public:
  Runtime(int &argc, char **&argv);

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  Runtime(Runtime &&) = delete;
  Runtime &operator=(Runtime &&) = delete;

  ~Runtime();

  [[nodiscard]] int rank() const;
  [[nodiscard]] int size() const;
  [[nodiscard]] bool isRoot() const;

  /**
   * Broadcast a string from one process to all processes in this runtime.
   *
   * This keeps backend-specific communication out of application utilities such
   * as the VTK writer.
   */
  void broadcastString(std::string &value, int rootRank = 0) const;

 private:
  template <class Particle, class Serializer>
  friend class DistributedAutoPas;

  [[nodiscard]] MPI_Comm communicator() const;

  void barrier() const;

  template <class T>
  [[nodiscard]] T globalSum(T localValue) const {
    T globalValue{};
    MPI_Allreduce(&localValue, &globalValue, 1, detail::mpiDatatype<T>(), MPI_SUM, _communicator);
    return globalValue;
  }

  MPI_Comm _communicator{MPI_COMM_NULL};
  int _rank{0};
  int _size{1};
  bool _ownsBackendLifetime{false};
};

}  // namespace dap
