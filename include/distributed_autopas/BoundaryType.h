#pragma once

namespace dap {

/**
 * Boundary condition applied at the global simulation-box boundary.
 *
 * periodic   Particles and halos wrap to the opposite side of the global box.
 * reflective Non-periodic communication boundary. The physical reflection force is application-defined.
 * none       No boundary handling. Particles leaving the global box are discarded.
 */
enum class BoundaryType { periodic, reflective, none };

}  // namespace dap
