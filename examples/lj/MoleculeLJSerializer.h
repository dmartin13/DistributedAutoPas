#pragma once

#include <array>
#include <cstddef>

#include "distributed_autopas/ParticleSerialization.h"
#include "molecularDynamicsLibrary/MoleculeLJ.h"

namespace dap {

template <>
struct ParticleSerializer<mdLib::MoleculeLJ> {
  struct Message {
    std::array<double, 3> r{};
    std::array<double, 3> v{};
    std::array<double, 3> f{};
    unsigned long id{0};
    std::size_t typeId{0};
  };

  static Message pack(const mdLib::MoleculeLJ &particle) {
    return Message{particle.getR(), particle.getV(), particle.getF(), particle.getID(), particle.getTypeId()};
  }

  static mdLib::MoleculeLJ unpack(const Message &message) {
    mdLib::MoleculeLJ particle(message.r, message.v, message.id, message.typeId);
    particle.setF(message.f);
    particle.setOwnershipState(autopas::OwnershipState::owned);
    return particle;
  }
};

}  // namespace dap
