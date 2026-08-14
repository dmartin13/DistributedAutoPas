#pragma once

#include <mpi.h>

#include <type_traits>
#include <vector>

#include "distributed_autopas/ParticleSerialization.h"

namespace dap {

template <class Particle>
struct LeftRightExchange {
  std::vector<Particle> recvFromLeft;
  std::vector<Particle> recvFromRight;
};

/**
 * Exchange particles with the direct left and right neighbors.
 *
 * Serialization is delegated to ParticleSerializer<Particle>. This keeps the distributed layer
 * independent of the concrete particle representation used by an application.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
LeftRightExchange<Particle> exchangeLeftRight(MPI_Comm comm, int left, int right, const std::vector<Particle> &sendLeft,
                                              const std::vector<Particle> &sendRight, int tag) {
  using Message = typename Serializer::Message;
  static_assert(std::is_trivially_copyable_v<Message>,
                "ParticleSerializer::Message must be trivially copyable for MPI_BYTE communication.");

  const auto packParticles = [](const std::vector<Particle> &particles) {
    std::vector<Message> messages;
    messages.reserve(particles.size());
    for (const auto &particle : particles) {
      messages.push_back(Serializer::pack(particle));
    }
    return messages;
  };

  const auto unpackParticles = [](const std::vector<Message> &messages) {
    std::vector<Particle> particles;
    particles.reserve(messages.size());
    for (const auto &message : messages) {
      particles.push_back(Serializer::unpack(message));
    }
    return particles;
  };

  const auto sendLeftMessages = packParticles(sendLeft);
  const auto sendRightMessages = packParticles(sendRight);

  const int sendLeftCount = static_cast<int>(sendLeftMessages.size());
  const int sendRightCount = static_cast<int>(sendRightMessages.size());

  int recvFromLeftCount = 0;
  int recvFromRightCount = 0;

  MPI_Sendrecv(&sendLeftCount, 1, MPI_INT, left, tag, &recvFromRightCount, 1, MPI_INT, right, tag, comm,
               MPI_STATUS_IGNORE);

  MPI_Sendrecv(&sendRightCount, 1, MPI_INT, right, tag + 1, &recvFromLeftCount, 1, MPI_INT, left, tag + 1, comm,
               MPI_STATUS_IGNORE);

  std::vector<Message> recvFromLeftMessages(recvFromLeftCount);
  std::vector<Message> recvFromRightMessages(recvFromRightCount);

  MPI_Sendrecv(sendLeftMessages.data(), sendLeftCount * static_cast<int>(sizeof(Message)), MPI_BYTE, left, tag + 2,
               recvFromRightMessages.data(), recvFromRightCount * static_cast<int>(sizeof(Message)), MPI_BYTE, right,
               tag + 2, comm, MPI_STATUS_IGNORE);

  MPI_Sendrecv(sendRightMessages.data(), sendRightCount * static_cast<int>(sizeof(Message)), MPI_BYTE, right, tag + 3,
               recvFromLeftMessages.data(), recvFromLeftCount * static_cast<int>(sizeof(Message)), MPI_BYTE, left,
               tag + 3, comm, MPI_STATUS_IGNORE);

  return LeftRightExchange<Particle>{unpackParticles(recvFromLeftMessages), unpackParticles(recvFromRightMessages)};
}

}  // namespace dap
