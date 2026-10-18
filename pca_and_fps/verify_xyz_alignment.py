"""
Decisive check for whether c2h6-2p_mobml_training_cds.npy still matches the
c2h6-2p_xyz/ files it was originally written alongside -- both are written
by the SAME execution of make_mobml_training_set.py (see there), so they're
internally consistent within one run; the risk is a LATER, unseeded re-run
overwriting the .npy with a new random selection while the .xyz files (and
the CCSD(T) energies already computed from them, which take days and are
never rerun casually) are left stale on disk.

Runs bare-metal, same as parse_ccsdt_energies.py -- no apptainer/PySCF
needed, this is pure numpy + file parsing:

    python3 verify_xyz_alignment.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, '/gscratch/mccoy/pot_training/pipeline_config')
from system_config import SYSTEM_TAG, ATOMS

au_to_ang = 0.529177249  # must match make_mobml_training_set.py exactly

here = os.path.dirname(os.path.abspath(__file__))
npy_path = f'{here}/output/{SYSTEM_TAG}/{SYSTEM_TAG}_mobml_training_cds.npy'
xyz_dir = f'{here}/output/{SYSTEM_TAG}/{SYSTEM_TAG}_xyz'


def read_xyz(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0])
    atoms = []
    coords = []
    for line in lines[2:2 + n]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords)


def main():
    print(f"SYSTEM_TAG={SYSTEM_TAG}")
    print(f"Loading {npy_path}")
    cds_bohr = np.load(npy_path)
    print("mobml_training_cds shape:", cds_bohr.shape)

    n_geom = len(cds_bohr)
    # Span both concatenated halves (min_cds_sample then broad_cds_sample,
    # see make_mobml_training_set.py) and the very ends, where an off-by-
    # one in either half's size would show up first.
    check_idx = sorted(set([0, 1, 2, n_geom // 4, n_geom // 2, n_geom - 3, n_geom - 2, n_geom - 1]))

    max_diff_overall = 0.0
    any_missing = False
    for i in check_idx:
        xyz_path = f'{xyz_dir}/{SYSTEM_TAG}_{i}.xyz'
        if not os.path.exists(xyz_path):
            print(f"[MISSING] index {i}: {xyz_path} does not exist")
            any_missing = True
            continue

        xyz_atoms, xyz_coords_ang = read_xyz(xyz_path)
        xyz_coords_bohr = xyz_coords_ang / au_to_ang

        atoms_match = list(xyz_atoms) == list(ATOMS)
        diff = np.max(np.abs(xyz_coords_bohr - cds_bohr[i]))
        max_diff_overall = max(max_diff_overall, diff)

        status = "PASS" if atoms_match and diff < 1e-5 else "FAIL"
        print(f"[{status}] index {i}: atoms_match={atoms_match}, max|coord diff| (Bohr) = {diff:.3e}")

    print(f"\nOverall max coordinate difference across checked indices: {max_diff_overall:.3e}")
    if any_missing:
        print("=> At least one expected .xyz file is missing -- investigate that separately.")
    elif max_diff_overall < 1e-5:
        print("=> ALIGNED: mobml_training_cds.npy matches the .xyz files at every checked index. "
              "This is not the source of the remaining test error.")
    else:
        print("=> MISALIGNED: mobml_training_cds.npy does NOT match the .xyz files these CCSD(T) "
              "energies were computed from. This confirms the hypothesis -- mob_features/hf_energies "
              "(built from the current .npy) and ccsdt_energies (computed from the stale .xyz files) "
              "are paired with different geometries than each other. Fix: regenerate everything "
              "downstream of make_mobml_training_set.py (the .xyz files, hf_mob_features, and the "
              "CCSD(T) jobs) from its CURRENT .npy output, in that order.")


if __name__ == '__main__':
    main()
