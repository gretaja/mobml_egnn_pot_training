"""
Extends hf_mob_features/compare_uhf_rohf.py's spot-check (10 hand-picked
geometries) into a real correlation: for EVERY test-set geometry from a
completed fit_mobml.py run, compute the ROHF/UHF energy gap and UHF spin
contamination, and correlate both against that geometry's actual test
error -- decisive evidence for whether a full UHF-features redesign is
likely to pay off, before committing to that (real) engineering cost.

Needs fit_mobml.py to have already run and saved
{SYSTEM_TAG}_test_geometry_idx.npy and {SYSTEM_TAG}_test_errors_n{N}.npy
(N = the training size to check -- defaults to the largest in
MOBML_N_TRAINING, override via N_TRAINING_TO_CHECK below).

Run with `python -i correlate_uhf_rohf_error.py` to keep everything in
scope afterward for further poking.
"""

import sys

import numpy as np
import pyscf

sys.path.insert(0, '/home/packages')
sys.path.insert(0, '/home/pipeline_config')

from pyscf import gto

from system_config import SYSTEM_TAG, ATOMS, CHARGE, SPIN, MOBML_N_TRAINING

BASIS = "aug-cc-pVDZ"

N_TRAINING_TO_CHECK = None  #UPDATE HERE -- None = largest value in MOBML_N_TRAINING

S_pure = SPIN / 2.0
expected_s2 = S_pure * (S_pure + 1)


def build_mol(coords):
    mol = gto.Mole()
    mol.atom = [(ATOMS[i], coords[i]) for i in range(len(ATOMS))]
    mol.unit = 'Bohr'
    mol.basis = BASIS
    mol.charge = CHARGE
    mol.spin = SPIN
    mol.build()
    return mol


def rohf_uhf_gap(mol):
    """Same two numbers as hf_mob_features/compare_uhf_rohf.py's compare_one,
    without the per-geometry printing -- this runs over every test point,
    not a handful."""
    rohf = pyscf.scf.ROHF(mol).density_fit()
    rohf.run(verbose=0)
    uhf = pyscf.scf.UHF(mol).density_fit()
    uhf.run(verbose=0)

    gap_cm = (rohf.e_tot - uhf.e_tot) * 219474.63136320
    s2_uhf, _ = uhf.spin_square()
    contamination = s2_uhf - expected_s2
    return gap_cm, contamination


def pearson(a, b):
    a = np.asarray(a) - np.mean(a)
    b = np.asarray(b) - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / denom) if denom > 0 else float('nan')


def main():
    n = N_TRAINING_TO_CHECK or max(MOBML_N_TRAINING)
    print(f"SYSTEM_TAG={SYSTEM_TAG}  checking test errors from n_training={n}")

    output_dir = f'/home/mobml_fitting/{SYSTEM_TAG}_models'
    geometries = np.load(f'/home/pca_and_fps/output/{SYSTEM_TAG}/{SYSTEM_TAG}_mobml_training_cds.npy')
    test_geometry_idx = np.load(f'{output_dir}/{SYSTEM_TAG}_test_geometry_idx.npy')
    signed_error = np.load(f'{output_dir}/{SYSTEM_TAG}_test_errors_n{n}.npy')
    abs_error = np.abs(signed_error)

    print(f"Test set size: {len(test_geometry_idx)}")

    gaps = []
    contaminations = []
    for j, g in enumerate(test_geometry_idx):
        mol = build_mol(geometries[int(g)])
        gap_cm, contamination = rohf_uhf_gap(mol)
        gaps.append(gap_cm)
        contaminations.append(contamination)
        if j % 20 == 0:
            print(f"  {j}/{len(test_geometry_idx)}  geometry {g}: "
                  f"gap={gap_cm:.1f} cm^-1  contamination={contamination:+.4f}  "
                  f"|error|={abs_error[j]:.1f} cm^-1")

    gaps = np.array(gaps)
    contaminations = np.array(contaminations)

    r_gap = pearson(gaps, abs_error)
    r_contam = pearson(contaminations, abs_error)
    r_gap_signed = pearson(gaps, signed_error)
    r_contam_signed = pearson(contaminations, signed_error)

    print("\n=========================")
    print("CORRELATION SUMMARY")
    print("=========================")
    print(f"ROHF-UHF gap:        mean={np.mean(gaps):.1f}, std={np.std(gaps):.1f} cm^-1")
    print(f"UHF contamination:   mean={np.mean(contaminations):.4f}, std={np.std(contaminations):.4f}")
    print(f"Test |error|:        mean={np.mean(abs_error):.1f}, std={np.std(abs_error):.1f} cm^-1")
    print(f"\nPearson r(|error|, ROHF-UHF gap)      = {r_gap:+.3f}")
    print(f"Pearson r(|error|, UHF contamination)  = {r_contam:+.3f}")
    print(f"Pearson r(signed error, ROHF-UHF gap)      = {r_gap_signed:+.3f}  (directional bias check)")
    print(f"Pearson r(signed error, UHF contamination)  = {r_contam_signed:+.3f}  (directional bias check)")

    if abs(r_gap) > 0.3 or abs(r_contam) > 0.3:
        print("\n=> Meaningful correlation -- geometries where ROHF is a weaker single-"
              "reference description (bigger ROHF-UHF gap / more UHF spin contamination) "
              "tend to have larger test error. Worth investing in UHF-based features.")
    else:
        print("\n=> Weak/no correlation -- ROHF-vs-UHF quality doesn't explain which "
              "geometries the model gets wrong. The remaining error is more likely about "
              "training-set coverage (see the learning curve) than the reference method.")

    print("\n--- Interactive prompt below. geometries, test_geometry_idx, signed_error, "
          "gaps, contaminations all in scope. ---")


if __name__ == '__main__':
    main()
