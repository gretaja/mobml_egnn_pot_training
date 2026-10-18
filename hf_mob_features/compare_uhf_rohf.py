"""
Compares ROHF vs UHF for a mix of geometries already known (from
verify_alignment.out, back when get_features_rohf_combined -- since
retired -- was still in use) to be flagged delocalized-SOMO vs clean, to
test whether ROHF's shared-spatial-orbital constraint is a genuine
electronic-structure inadequacy for the delocalized cases specifically --
not just an artifact of which orbital the top-n_somo tie-break happens to
pick.

Two numbers, neither depending on any feature-construction choice:
  - E_ROHF - E_UHF: always >= 0 (UHF has strictly more variational
    freedom). Larger on delocalized geometries would mean ROHF is a worse
    approximation specifically where the model already struggles.
  - UHF spin contamination (<S^2> vs the pure-doublet value S(S+1)=0.75):
    ROHF is spin-pure by construction (uninformative on its own); large
    UHF contamination means the true wavefunction wants to break spin/
    spatial symmetry unevenly, which a single-reference ROHF picture
    can't represent regardless of SOMO row selection.

Run with `python -i compare_uhf_rohf.py` to keep everything in scope
afterward for further poking.
"""

import sys

import numpy as np
import pyscf

sys.path.insert(0, '/home/packages')
sys.path.insert(0, '/home/pipeline_config')

from pyscf import gto

from system_config import SYSTEM_TAG, ATOMS, CHARGE, SPIN

BASIS = "aug-cc-pVDZ"

# From verify_alignment.out's spot checks, back when get_features_rohf_combined
# (since retired) was still in use: geometries it had flagged (or not) as
# delocalized-SOMO. compare_one below never calls it -- it builds ROHF/UHF
# directly -- so this list is just historical provenance for which
# geometries to look at, not a live dependency.
DELOCALIZED_GEOM_IDX = [0, 2399, 3598, 4798, 5189, 5998, 5999]
CLEAN_GEOM_IDX = [1, 1199, 5187]

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


def compare_one(mol, label):
    rohf = pyscf.scf.ROHF(mol).density_fit()
    rohf.run(verbose=0)
    uhf = pyscf.scf.UHF(mol).density_fit()
    uhf.run(verbose=0)

    e_rohf = rohf.e_tot
    e_uhf = uhf.e_tot
    gap_cm = (e_rohf - e_uhf) * 219474.63136320  # Hartree -> cm^-1

    s2_rohf, mult_rohf = rohf.spin_square()
    s2_uhf, mult_uhf = uhf.spin_square()
    contamination = s2_uhf - expected_s2

    print(f"[{label}] E_ROHF={e_rohf:.8f}  E_UHF={e_uhf:.8f}  "
          f"E_ROHF-E_UHF={gap_cm:.4f} cm^-1")
    print(f"           <S^2>_ROHF={s2_rohf:.6f} (expect {expected_s2:.4f} exactly)  "
          f"<S^2>_UHF={s2_uhf:.6f}  contamination={contamination:+.6f}")

    return {
        "label": label, "e_rohf": e_rohf, "e_uhf": e_uhf, "gap_cm": gap_cm,
        "s2_rohf": s2_rohf, "s2_uhf": s2_uhf, "contamination": contamination,
    }


def main():
    print(f"SYSTEM_TAG={SYSTEM_TAG}  SPIN={SPIN}  expected pure <S^2>={expected_s2:.4f}\n")

    geometries = np.load(f'/home/pca_and_fps/output/{SYSTEM_TAG}/{SYSTEM_TAG}_mobml_training_cds.npy')

    results = []

    print("--- delocalized-SOMO-flagged geometries ---")
    for i in DELOCALIZED_GEOM_IDX:
        mol = build_mol(geometries[i])
        results.append(compare_one(mol, f"delocalized idx={i}"))

    print("\n--- clean-SOMO geometries ---")
    for i in CLEAN_GEOM_IDX:
        mol = build_mol(geometries[i])
        results.append(compare_one(mol, f"clean idx={i}"))

    gaps_delocalized = [r["gap_cm"] for r in results if "delocalized" in r["label"]]
    gaps_clean = [r["gap_cm"] for r in results if "clean" in r["label"]]
    contam_delocalized = [r["contamination"] for r in results if "delocalized" in r["label"]]
    contam_clean = [r["contamination"] for r in results if "clean" in r["label"]]

    print("\n=========================")
    print("SUMMARY")
    print("=========================")
    print(f"E_ROHF-E_UHF (cm^-1): delocalized mean={np.mean(gaps_delocalized):.2f}, "
          f"clean mean={np.mean(gaps_clean):.2f}")
    print(f"UHF spin contamination: delocalized mean={np.mean(contam_delocalized):.4f}, "
          f"clean mean={np.mean(contam_clean):.4f}")
    if np.mean(gaps_delocalized) > 1.5 * max(np.mean(gaps_clean), 1e-6) or \
       np.mean(contam_delocalized) > 1.5 * max(np.mean(contam_clean), 1e-6):
        print("=> Delocalized geometries show a substantially bigger ROHF/UHF gap and/or "
              "UHF spin contamination -- ROHF's single-reference constraint looks like a "
              "genuine electronic-structure inadequacy there, not just a feature-selection "
              "artifact. Worth pursuing UHF-based features for real.")
    else:
        print("=> No substantial difference between delocalized and clean geometries here -- "
              "ROHF vs UHF doesn't look like the operative axis; the delocalized-SOMO cost we "
              "measured is more likely just about the feature-selection tie-break itself.")

    print("\n--- Interactive prompt below. geometries, results, build_mol, compare_one all in scope. ---")


if __name__ == '__main__':
    main()
