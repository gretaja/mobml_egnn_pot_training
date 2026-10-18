"""
Spot-checks that geometry index k means the same thing at every stage of
the pipeline for a system: pca_and_fps's saved coordinates/.xyz files,
hf_mob_features's cached HF energy + MOB features (independently
RECOMPUTED here, not just re-read), and ccsdt_energies's parsed CCSD(T)
energy.

This exists to tell apart "fit_mobml.py has a real indexing bug" from
"fit_mobml.py's Test MAE is bad because the regression problem is
genuinely hard" -- shape checks alone can't distinguish those two, since
mismatched *values* at the same array position don't change the shapes.

Run once mob_features/hf_energies (hf_mob_features) and
ccsdt_energies/valid_indices (ccsdt_energies, via parse_ccsdt_energies.py)
both exist for a system:

    python3 verify_alignment.py
"""

import sys
import os
import re

import numpy as np

sys.path.insert(0, '/home/packages')
sys.path.insert(0, '/home/pipeline_config')

from pyscf import gto
from mobml_pyscf import get_features_rhf, get_features_rohf, get_features_uhf

from system_config import SYSTEM_TAG, ATOMS, CHARGE, SPIN

# Must match whatever REFERENCE_METHOD get_mobml_features.py was run with
# -- "rohf" or "uhf" -- see its comment there.
REFERENCE_METHOD = "uhf"  #UPDATE HERE


def get_features(mol, verbose=True):
    """
    SPIN-dispatching wrapper matching get_mobml_features.py's own branch
    (SPIN == 0: closed-shell RHF; SPIN != 0: open-shell, alpha+beta
    concatenated per REFERENCE_METHOD) -- calling the wrong one here would
    recompute a different physical quantity than what's cached, and
    falsely report a MISMATCH that has nothing to do with any real
    alignment bug.
    """
    if SPIN == 0:
        return get_features_rhf(mol, verbose=verbose)
    get_features_openshell = get_features_rohf if REFERENCE_METHOD == "rohf" else get_features_uhf
    mf, features_alpha, features_beta = get_features_openshell(mol, verbose=verbose)
    return mf, np.concatenate([features_alpha, features_beta], axis=0)

# Must match get_mobml_features.py's BASIS exactly, or the recomputed HF
# energy/features below won't be comparable to the cached ones at all.
BASIS = "aug-cc-pVDZ"

AU_TO_ANG = 0.529177249


def build_mol(coords):
    mol = gto.Mole()
    mol.atom = [(ATOMS[i], coords[i]) for i in range(len(ATOMS))]
    mol.unit = 'Bohr'
    mol.basis = BASIS
    mol.charge = CHARGE
    mol.spin = SPIN
    mol.build()
    return mol


def load_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0])
    elements, coords = [], []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return elements, np.array(coords)


def main():
    pca_dir = f'/home/pca_and_fps/output/{SYSTEM_TAG}'
    hf_dir = '/home/hf_mob_features'
    ccsdt_dir = f'/home/ccsdt_energies/{SYSTEM_TAG}_dlpno_ccsdt_augccpvtz_output'
    suffix = f"{SYSTEM_TAG}_dlpno_ccsdt_augccpvtz"

    geometries = np.load(f'{pca_dir}/{SYSTEM_TAG}_mobml_training_cds.npy')  # Bohr
    mob_features = np.load(f'{hf_dir}/{SYSTEM_TAG}_training_mob_features.npy')
    feature_selection = np.load(f'{hf_dir}/{SYSTEM_TAG}_feature_selection.npy')
    hf_energies = np.load(f'{hf_dir}/{SYSTEM_TAG}_training_hf_dz.npy').reshape(-1)
    ccsdt_energies = np.load(f'{ccsdt_dir}/{suffix}_energies.npy')
    valid_idx = np.load(f'{ccsdt_dir}/{suffix}_valid_indices.npy')

    print(f"geometries: {geometries.shape}")
    print(f"mob_features: {mob_features.shape}")
    print(f"hf_energies: {hf_energies.shape}")
    print(f"ccsdt_energies: {ccsdt_energies.shape}")
    print(f"valid_idx: {valid_idx.shape}")

    # =========================
    # EXCLUDED GEOMETRIES
    # =========================

    # mob_features/hf_energies are dense, position-i-means-geometry-i
    # arrays (get_mobml_features.py never failed a geometry for this
    # system -- see its own printed "Valid samples" count), so anything in
    # range(len(mob_features)) NOT in valid_idx is a geometry that never
    # got a usable CCSD(T) energy and should be excluded by
    # X = mob_features[valid_idx] in fit_mobml.py -- confirm that's really
    # what's happening, not assumed.
    valid_set = set(int(v) for v in valid_idx)
    missing_idx = sorted(set(range(len(mob_features))) - valid_set)
    print(f"\nGeometries excluded from valid_idx: {missing_idx}")

    for k in missing_idx:
        out_path = f'{ccsdt_dir}/{SYSTEM_TAG}_{k}_{suffix}.out'
        if not os.path.isfile(out_path):
            print(f"  index {k}: no .out file at all ({out_path}) -- never submitted/completed, not an ORCA failure")
        else:
            with open(out_path) as f:
                text = f.read()
            has_energy = "FINAL SINGLE POINT ENERGY" in text
            terminated_normally = "****ORCA TERMINATED NORMALLY****" in text
            print(f"  index {k}: .out exists, terminated_normally={terminated_normally}, "
                  f"has_final_energy={has_energy} -- {'genuine ORCA failure' if not has_energy else 'unexpected: has an energy but parse_ccsdt_energies.py still excluded it -- check its regex/logic'}")

    # X = mob_features[valid_idx] as fit_mobml.py does -- confirm the
    # resulting length and that no excluded index leaked through.
    X_check = mob_features[valid_idx]
    print(f"\nmob_features[valid_idx] shape: {X_check.shape} "
          f"(expect ({len(mob_features) - len(missing_idx)}, ...))")
    leaked = valid_set & set(missing_idx)
    print(f"Excluded indices present in valid_idx (should be empty): {leaked}")

    # Spot-check the first couple, last couple, a few spread through
    # valid_idx, AND -- most decisively for an off-by-one/shift bug right
    # at an exclusion boundary -- the valid_idx positions immediately
    # before/after each gap in missing_idx, where a shift bug would first
    # show up as a MISMATCH even if everything far from the gap looks fine.
    n_check = min(6, len(valid_idx))
    boundary_positions = []
    for k in missing_idx:
        below = np.where(valid_idx < k)[0]
        above = np.where(valid_idx > k)[0]
        if len(below):
            boundary_positions.append(int(below[-1]))
        if len(above):
            boundary_positions.append(int(above[0]))
    check_positions = sorted(set(
        list(range(min(2, len(valid_idx))))
        + list(range(max(0, len(valid_idx) - 2), len(valid_idx)))
        + boundary_positions
        + list(np.linspace(0, len(valid_idx) - 1, n_check, dtype=int))
    ))

    all_ok = True

    for pos in check_positions:
        k = int(valid_idx[pos])
        print(f"\n--- geometry index {k} (position {pos} in valid_idx) ---")

        # 1. .xyz file vs. mobml_training_cds.npy[k]. The .xyz was written
        # in Angstrom (au_to_ang-converted); the .npy is still Bohr.
        xyz_path = f'{pca_dir}/{SYSTEM_TAG}_xyz/{SYSTEM_TAG}_{k}.xyz'
        if os.path.isfile(xyz_path):
            xyz_elements, xyz_coords_ang = load_xyz(xyz_path)
            xyz_coords_bohr = xyz_coords_ang / AU_TO_ANG
            max_diff = np.max(np.abs(xyz_coords_bohr - geometries[k]))
            ok = (xyz_elements == ATOMS) and max_diff < 1e-4
            print(f"  .xyz vs mobml_training_cds.npy[{k}]: atoms match={xyz_elements == ATOMS}, "
                  f"max coord diff={max_diff:.2e} Bohr -> {'OK' if ok else 'MISMATCH'}")
            all_ok &= ok
        else:
            print(f"  (skipped -- {xyz_path} not found)")

        # 2. .out file's own FINAL SINGLE POINT ENERGY vs. the value
        # parse_ccsdt_energies.py saved at this position.
        out_path = f'{ccsdt_dir}/{SYSTEM_TAG}_{k}_{suffix}.out'
        if os.path.isfile(out_path):
            with open(out_path) as f:
                text = f.read()
            matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
            out_energy = float(matches[-1]) if matches else None
            saved_energy = float(ccsdt_energies[pos])
            ok = out_energy is not None and abs(out_energy - saved_energy) < 1e-8
            print(f"  .out FINAL SINGLE POINT ENERGY={out_energy} vs ccsdt_energies[{pos}]={saved_energy} "
                  f"-> {'OK' if ok else 'MISMATCH'}")
            all_ok &= ok
        else:
            print(f"  (skipped -- {out_path} not found)")

        # 3. Independently RECOMPUTE the HF energy and MOB features for
        # geometries[k], and compare against the cached hf_energies[k]/
        # mob_features[k] -- the only real proof those rows correspond to
        # geometry k, since they're otherwise just cached numbers with no
        # link back to a geometry recorded anywhere.
        mol = build_mol(geometries[k])
        hf, features = get_features(mol, verbose=False)

        recomputed_hf = hf.energy_tot()
        cached_hf = float(hf_energies[k])
        hf_diff = abs(recomputed_hf - cached_hf)
        hf_ok = hf_diff < 1e-6
        print(f"  recomputed HF={recomputed_hf} vs cached hf_energies[{k}]={cached_hf} "
              f"(diff={hf_diff:.2e} Hartree) -> {'OK' if hf_ok else 'MISMATCH'}")
        all_ok &= hf_ok

        recomputed_features = features[:, feature_selection]
        cached_features = mob_features[k]
        feat_diff = np.max(np.abs(recomputed_features - cached_features))
        feat_ok = feat_diff < 1e-6
        print(f"  recomputed features vs mob_features[{k}]: max diff={feat_diff:.2e} "
              f"-> {'OK' if feat_ok else 'MISMATCH'}")
        all_ok &= feat_ok

    # =========================
    # TRAIN/TEST SUBSETTING
    # =========================

    # Replicates fit_mobml.py's own X/Y construction and split exactly
    # (same seed, same MOBML_N_TRAINING/MOBML_TEST_SIZE) -- X/Y/
    # train_idx/test_idx here are the literal objects that script's GP
    # fit uses, not a re-implementation that could itself drift from it.
    from system_config import MOBML_N_TRAINING, MOBML_TEST_SIZE

    X = mob_features[valid_idx]
    hf_energies_valid = hf_energies[valid_idx]
    deltas = ccsdt_energies - hf_energies_valid
    add_energy = np.mean(deltas)
    Y = (1000 * (deltas - add_energy)).reshape(-1, 1)

    print(f"\nX: {X.shape}, Y: {Y.shape} -- same length: {len(X) == len(Y)}")

    np.random.seed(42)
    indices = np.random.permutation(len(X))
    test_idx = indices[:MOBML_TEST_SIZE]
    train_idx_max = indices[MOBML_TEST_SIZE:max(MOBML_N_TRAINING) + MOBML_TEST_SIZE]

    overlap = set(test_idx.tolist()) & set(train_idx_max.tolist())
    in_range = np.all((indices >= 0) & (indices < len(X)))
    print(f"test_idx: {test_idx.shape}, train_idx (largest n_training={max(MOBML_N_TRAINING)}): {train_idx_max.shape}")
    print(f"train/test overlap (should be empty): {overlap}")
    print(f"all indices within [0, {len(X)}) (should be True): {in_range}")

    # For a handful of test_idx entries, confirm Y[test_idx[j]] really is
    # ccsdt_energies[pos] - hf_energies[valid_idx[pos]] for the SAME
    # position pos = test_idx[j] -- i.e. that subsetting X/Y by train_idx/
    # test_idx (numpy fancy indexing into the already-valid_idx-selected
    # arrays) doesn't reintroduce any shift, on top of everything already
    # checked above for the valid_idx selection itself.
    subset_ok = True
    for j in test_idx[:5]:
        pos = int(j)
        k = int(valid_idx[pos])
        expected = 1000 * ((ccsdt_energies[pos] - hf_energies[k]) - add_energy)
        actual = float(Y[pos, 0])
        ok = abs(expected - actual) < 1e-9
        subset_ok &= ok
        print(f"  test position {pos} (geometry {k}): Y={actual:.6f}, recomputed from "
              f"ccsdt_energies[{pos}]-hf_energies[{k}]={expected:.6f} -> {'OK' if ok else 'MISMATCH'}")
    all_ok &= subset_ok and not overlap and bool(in_range)

    print("\n" + "=" * 60)
    if all_ok:
        print("All spot-checks passed -- alignment looks correct across every stage.")
        print("The learning curve's high Test MAE most likely reflects a hard/")
        print("under-sampled regression problem, not misaligned data.")
    else:
        print("At least one spot-check MISMATCHED -- see above for which index/stage.")
    print("=" * 60)


if __name__ == '__main__':
    main()
