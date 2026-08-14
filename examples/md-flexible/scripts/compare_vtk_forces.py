#!/usr/bin/env python3
"""
Compare final owned-particle forces from two md-flexible VTK output trees.

Usage:
    python3 compare_vtk_forces.py <np1-output-root> <np4-output-root> [tolerance]

The script searches recursively for *_Particles_<rank>_<iteration>.vtu files,
selects the largest iteration in each output tree, joins particles by ID, and
compares the three force components.
"""

import glob
import math
import os
import re
import sys
import xml.etree.ElementTree as ET


FILE_RE = re.compile(r"_Particles_(\d+)_(\d+)\.vtu$")


def final_particle_files(root):
    candidates = []
    for filename in glob.glob(os.path.join(root, "**", "*_Particles_*_*.vtu"), recursive=True):
        match = FILE_RE.search(filename)
        if match:
            rank = int(match.group(1))
            iteration = int(match.group(2))
            candidates.append((iteration, rank, filename))

    if not candidates:
        raise RuntimeError(f"No particle VTU files found below: {root}")

    final_iteration = max(item[0] for item in candidates)
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


def read_forces(root):
    iteration, files = final_particle_files(root)
    result = {}

    for rank, filename in files:
        tree = ET.parse(filename)
        root_element = tree.getroot()
        piece = next(root_element.iter("Piece"))

        ids = [int(x) for x in data_array(piece, "ids")]
        raw_forces = [float(x) for x in data_array(piece, "forces")]

        if len(raw_forces) != 3 * len(ids):
            raise RuntimeError(
                f"Force/ID length mismatch in {filename}: "
                f"{len(raw_forces)} force values for {len(ids)} particles"
            )

        for i, particle_id in enumerate(ids):
            force = tuple(raw_forces[3 * i : 3 * i + 3])
            if particle_id in result:
                raise RuntimeError(
                    f"Particle ID {particle_id} occurs more than once in final owned output"
                )
            result[particle_id] = (force, rank)

    return iteration, result


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__.strip())
        return 2

    tolerance = float(sys.argv[3]) if len(sys.argv) == 4 else 1e-5

    iteration_a, forces_a = read_forces(sys.argv[1])
    iteration_b, forces_b = read_forces(sys.argv[2])

    ids_a = set(forces_a)
    ids_b = set(forces_b)

    if ids_a != ids_b:
        print("FAIL: owned particle IDs differ.")
        print("  only in first run :", sorted(ids_a - ids_b))
        print("  only in second run:", sorted(ids_b - ids_a))
        return 1

    rows = []
    max_abs_diff = 0.0

    for particle_id in sorted(ids_a):
        force_a, rank_a = forces_a[particle_id]
        force_b, rank_b = forces_b[particle_id]

        component_diffs = [abs(a - b) for a, b in zip(force_a, force_b)]
        diff = max(component_diffs)
        max_abs_diff = max(max_abs_diff, diff)
        rows.append((diff, particle_id, rank_a, rank_b, force_a, force_b))

    print(f"First run final iteration : {iteration_a}")
    print(f"Second run final iteration: {iteration_b}")
    print(f"Particles compared        : {len(rows)}")
    print(f"Maximum absolute force diff: {max_abs_diff:.12g}")
    print(f"Tolerance                  : {tolerance:.12g}")
    print()
    print("Largest differences:")

    for diff, particle_id, rank_a, rank_b, force_a, force_b in sorted(rows, reverse=True)[:10]:
        print(
            f"  id={particle_id:4d}  "
            f"rank {rank_a}->{rank_b}  "
            f"diff={diff:.12g}  "
            f"np1={force_a}  np4={force_b}"
        )

    if max_abs_diff <= tolerance:
        print("\nPASS: forces agree within tolerance.")
        return 0

    print("\nFAIL: force difference exceeds tolerance.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
