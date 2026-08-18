#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "autopas/utils/ArrayMath.h"
#include "autopas/utils/Math.h"
#include "distributed_autopas/BoundaryType.h"

/**
 * Application-level reflective Lennard-Jones wall used by md-flexible.
 *
 * DistributedAutoPas only models reflective boundaries as non-periodic
 * communication boundaries. The physical wall interaction remains part of the
 * simulator because it depends on Lennard-Jones particle properties.
 *
 * The current implementation covers md-flexible's single-site particle mode.
 * Only thin wall-adjacent regions are traversed through DistributedAutoPas.
 * It follows the original md-flexible model: a particle interacts with its own
 * mirror image across a reflective global boundary, but only while that
 * interaction is in the repulsive part of the Lennard-Jones potential.
 */
namespace ReflectiveBoundary {
namespace detail {

inline constexpr double sixthRootOfTwo = 1.122462048309373;

template <class Particle>
void addMirrorForce(Particle &particle, const std::array<double, 3> &position, std::size_t dimension,
                    double boundaryPosition, double sigmaSquared, double epsilon24) {
  auto mirrorPosition = position;
  mirrorPosition[dimension] = 2. * boundaryPosition - position[dimension];

  const auto displacement = autopas::utils::ArrayMath::sub(position, mirrorPosition);
  const auto distanceSquared = autopas::utils::ArrayMath::dot(displacement, displacement);
  const auto inverseDistanceSquared = 1. / distanceSquared;
  const auto lj2 = sigmaSquared * inverseDistanceSquared;
  const auto lj6 = lj2 * lj2 * lj2;
  const auto lj12 = lj6 * lj6;
  const auto lj12m6 = lj12 - lj6;
  const auto scalarMultiple = epsilon24 * (lj12 + lj12m6) * inverseDistanceSquared;

  particle.addF(autopas::utils::ArrayMath::mulScalar(displacement, scalarMultiple));
}

}  // namespace detail

template <class Container, class ParticlePropertiesLibrary>
void apply(Container &container, ParticlePropertiesLibrary &particlePropertiesLibrary, double maximumSigma) {
  const auto &boundaryTypes = container.boundaryTypes();
  const auto &globalMin = container.globalBoxMin();
  const auto &globalMax = container.globalBoxMax();
  const auto &localMin = container.localBoxMin();
  const auto &localMax = container.localBoxMax();
  const auto maximumRepulsiveWallDistance = detail::sixthRootOfTwo * maximumSigma * 0.5;

  if (maximumRepulsiveWallDistance <= 0.) {
    return;
  }

  auto applyWall = [&](std::size_t dimension, double boundaryPosition, bool lowerBoundary) {
    auto regionMin = localMin;
    auto regionMax = localMax;

    if (lowerBoundary) {
      regionMax[dimension] = std::min(localMax[dimension], boundaryPosition + maximumRepulsiveWallDistance);
    } else {
      regionMin[dimension] = std::max(localMin[dimension], boundaryPosition - maximumRepulsiveWallDistance);
    }

    container.applyToOwnedParticlesInRegion(regionMin, regionMax, [&](auto &particle) {
      const auto position = particle.getR();
      const auto siteType = particle.getTypeId();
      const auto sigma = particlePropertiesLibrary.getSigma(siteType);
      const auto particleRepulsiveWallDistance = detail::sixthRootOfTwo * sigma * 0.5;
      const auto distanceToBoundary =
          lowerBoundary ? position[dimension] - boundaryPosition : boundaryPosition - position[dimension];

      if (distanceToBoundary >= particleRepulsiveWallDistance) {
        return;
      }

      const auto sigmaSquared = particlePropertiesLibrary.getMixingSigmaSquared(siteType, siteType);
      const auto epsilon24 = particlePropertiesLibrary.getMixing24Epsilon(siteType, siteType);
      detail::addMirrorForce(particle, position, dimension, boundaryPosition, sigmaSquared, epsilon24);
    });
  };

  for (std::size_t dimension = 0; dimension < 3; ++dimension) {
    if (boundaryTypes[dimension] != dap::BoundaryType::reflective) {
      continue;
    }

    if (autopas::utils::Math::isNearRel(localMin[dimension], globalMin[dimension])) {
      applyWall(dimension, globalMin[dimension], true);
    }

    if (autopas::utils::Math::isNearRel(localMax[dimension], globalMax[dimension])) {
      applyWall(dimension, globalMax[dimension], false);
    }
  }
}

}  // namespace ReflectiveBoundary
