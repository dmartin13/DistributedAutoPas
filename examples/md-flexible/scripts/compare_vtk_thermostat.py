#!/usr/bin/env python3
"""
Compare the final thermostat state of a 1-rank and multi-rank md-flexible run.

Usage:
    python3 compare_vtk_thermostat.py <np1-output-root> <np4-output-root> [velocity-tolerance]

The script:
  * finds the final *_Particles_<rank>_<iteration>.vtu files,
  * joins owned particles by ID,
  * compares velocities particle-by-particle,
  * computes the global single-site temperature T = sum_i m |v_i|^2 / (3 N)
    for mass m=1, matching this test configuration,
  * checks that the final temperature is 1.5.
"""

import glob
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

FILE_RE = re.compile(r"_Particles_(\d+)_(\d+)\.vtu$")
EXPECTED_TEMPERATURE = 1.5
TEMPERATURE_TOLERANCE = 1e-5


def final_particle_files(root):
    candidates = []
    for filename in glob.glob(os.path.join(root, "**", "*_Particles_*_*.vtu"), recursive=True):
        match = FILE_RE.search(filename)
        if match:
            candidates.append((int(match.group(2)), int(match.group(1)), filename))

    if not candidates:
        raise RuntimeError(f"No particle VTU files found below: {root}")

    final_iteration = max(iteration for iteration, _, _ in candidates)
    files = sorted(
        (rank, filename)
        for iteration, rank, filename in candidates
        if iteration == final_iteration
    )
    return final_iteration, files


def data_array(piece, name):
    for element in piece.iter("DataArray"):
        if element.attrib.get("Name") == name:
            return (element.text or "").split()
    raise RuntimeError(f'DataArray "{name}" not found')


def read_velocities(root):
    iteration, files = final_particle_files(root)
    result = {}

    for rank, filename in files:
        tree = ET.parse(filename)
        piece = next(tree.getroot().iter("Piece"))

        ids = [int(x) for x in data_array(piece, "ids")]
        raw_velocities = [float(x) for x in data_array(piece, "velocities")]

        if len(raw_velocities) != 3 * len(ids):
            raise RuntimeError(
                f"Velocity/ID length mismatch in {filename}: "
                f"{len(raw_velocities)} velocity values for {len(ids)} particles"
            )

        for i, particle_id in enumerate(ids):
            velocity = tuple(raw_velocities[3 * i : 3 * i + 3])
            if particle_id in result:
                raise RuntimeError(
                    f"Particle ID {particle_id} occurs more than once in final owned output"
                )
            result[particle_id] = (velocity, rank)

    return iteration, result


def temperature(particles):
    if not particles:
        raise RuntimeError("No particles found")
    sum_v2 = 0.0
    for velocity, _ in particles.values():
        sum_v2 += sum(component * component for component in velocity)
    # This test uses one single-site component with mass = 1 and 3 translational DOFs.
    return sum_v2 / (3.0 * len(particles))


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__.strip())
        return 2

    velocity_tolerance = float(sys.argv[3]) if len(sys.argv) == 4 else 1e-5

    iteration_a, particles_a = read_velocities(sys.argv[1])
    iteration_b, particles_b = read_velocities(sys.argv[2])

    ids_a = set(particles_a)
    ids_b = set(particles_b)

    if ids_a != ids_b:
        print("FAIL: owned particle IDs differ.")
        print("  only in first run :", sorted(ids_a - ids_b))
        print("  only in second run:", sorted(ids_b - ids_a))
        return 1

    rows = []
    max_abs_diff = 0.0

    for particle_id in sorted(ids_a):
        velocity_a, rank_a = particles_a[particle_id]
        velocity_b, rank_b = particles_b[particle_id]
        diff = max(abs(a - b) for a, b in zip(velocity_a, velocity_b))
        max_abs_diff = max(max_abs_diff, diff)
        rows.append((diff, particle_id, rank_a, rank_b, velocity_a, velocity_b))

    temperature_a = temperature(particles_a)
    temperature_b = temperature(particles_b)

    print(f"First run final iteration  : {iteration_a}")
    print(f"Second run final iteration : {iteration_b}")
    print(f"Particles compared         : {len(rows)}")
    print(f"Maximum velocity difference: {max_abs_diff:.12g}")
    print(f"Velocity tolerance         : {velocity_tolerance:.12g}")
    print(f"Temperature first run      : {temperature_a:.12g}")
    print(f"Temperature second run     : {temperature_b:.12g}")
    print(f"Expected final temperature : {EXPECTED_TEMPERATURE:.12g}")
    print()
    print("Largest velocity differences:")

    for diff, particle_id, rank_a, rank_b, velocity_a, velocity_b in sorted(rows, reverse=True)[:10]:
        print(
            f"  id={particle_id:4d}  rank {rank_a}->{rank_b}  "
            f"diff={diff:.12g}  np1={velocity_a}  np4={velocity_b}"
        )

    velocity_ok = max_abs_diff <= velocity_tolerance
    temp_a_ok = abs(temperature_a - EXPECTED_TEMPERATURE) <= TEMPERATURE_TOLERANCE
    temp_b_ok = abs(temperature_b - EXPECTED_TEMPERATURE) <= TEMPERATURE_TOLERANCE

    if velocity_ok and temp_a_ok and temp_b_ok:
        print("\nPASS: thermostat result agrees across ranks and reaches the expected temperature.")
        return 0

    print("\nFAIL:")
    if not velocity_ok:
        print("  particle velocities differ between the runs")
    if not temp_a_ok:
        print(f"  first-run temperature differs from {EXPECTED_TEMPERATURE}")
    if not temp_b_ok:
        print(f"  second-run temperature differs from {EXPECTED_TEMPERATURE}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
