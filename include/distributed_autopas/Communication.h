#pragma once

#include <mpi.h>

#include <cstddef>
#include <stdexcept>
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

/**
 * Redistribute particles to arbitrary destination ranks.
 *
 * Each entry in particlesByDestination contains the particles that the calling rank wants to send
 * to the rank with the same index. The returned vector contains all particles sent to the calling
 * rank by all ranks, including particles routed to itself.
 *
 * This collective is primarily used for initialization and checkpoint redistribution where a
 * particle may need to move directly to any rank. Timestep migration intentionally uses the
 * cheaper direct-neighbor exchange instead.
 */
template <class Particle, class Serializer = ParticleSerializer<Particle>>
std::vector<Particle> exchangeParticlesByRank(MPI_Comm comm,
                                              const std::vector<std::vector<Particle>> &particlesByDestination) {
  using Message = typename Serializer::Message;
  static_assert(std::is_trivially_copyable_v<Message>,
                "ParticleSerializer::Message must be trivially copyable for MPI_BYTE communication.");

  int numberOfRanks = 0;
  MPI_Comm_size(comm, &numberOfRanks);

  if (particlesByDestination.size() != static_cast<std::size_t>(numberOfRanks)) {
    throw std::invalid_argument("exchangeParticlesByRank(): one destination buffer per rank is required.");
  }

  std::vector<int> sendCounts(numberOfRanks, 0);
  std::vector<int> sendDisplacements(numberOfRanks, 0);
  std::vector<Message> sendMessages;

  std::size_t totalSendMessages = 0;
  for (const auto &particles : particlesByDestination) {
    totalSendMessages += particles.size();
  }
  sendMessages.reserve(totalSendMessages);

  for (int destination = 0; destination < numberOfRanks; ++destination) {
    sendDisplacements[destination] = static_cast<int>(sendMessages.size() * sizeof(Message));
    sendCounts[destination] = static_cast<int>(particlesByDestination[destination].size() * sizeof(Message));

    for (const auto &particle : particlesByDestination[destination]) {
      sendMessages.push_back(Serializer::pack(particle));
    }
  }

  std::vector<int> receiveCounts(numberOfRanks, 0);
  MPI_Alltoall(sendCounts.data(), 1, MPI_INT, receiveCounts.data(), 1, MPI_INT, comm);

  std::vector<int> receiveDisplacements(numberOfRanks, 0);
  int totalReceiveBytes = 0;
  for (int source = 0; source < numberOfRanks; ++source) {
    receiveDisplacements[source] = totalReceiveBytes;
    totalReceiveBytes += receiveCounts[source];
  }

  std::vector<Message> receiveMessages(static_cast<std::size_t>(totalReceiveBytes) / sizeof(Message));

  MPI_Alltoallv(sendMessages.data(), sendCounts.data(), sendDisplacements.data(), MPI_BYTE, receiveMessages.data(),
                receiveCounts.data(), receiveDisplacements.data(), MPI_BYTE, comm);

  std::vector<Particle> receivedParticles;
  receivedParticles.reserve(receiveMessages.size());
  for (const auto &message : receiveMessages) {
    receivedParticles.push_back(Serializer::unpack(message));
  }

  return receivedParticles;
}

}  // namespace dap
