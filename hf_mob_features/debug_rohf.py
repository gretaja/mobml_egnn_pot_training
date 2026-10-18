"""
Diagnostic script for the ROHF/open-shell feature pipeline
(get_features_rohf_combined in mobml_pyscf) -- runs PySCF on ONE geometry
and prints every intermediate quantity needed to tell apart:

  (a) the alpha/beta density matrices themselves being wrong (wrong
      convention, wrong basis, mf.make_rdm1() not returning what we
      assumed), from
  (b) the IAO-based localized valence basis not fully spanning the true
      occupied space for this open-shell case (a harder, more structural
      problem), from
  (c) the old pyscf.scf.RHF(mol) path (before this ROHF-specific code
      existed) not actually running a real ROHF calculation for open-shell
      mol, which the reported nocc=14 there (vs. the electron-count-
      correct 18 here) suggests.

Run once, read the printed PASS/FAIL lines top to bottom -- the first FAIL
tells you which of (a)/(b)/(c) it is.
"""

import sys
import os

import numpy as np

sys.path.insert(0, '/home/packages')
sys.path.insert(0, '/home/pipeline_config')

import pyscf
from pyscf import gto

from system_config import SYSTEM_TAG, ATOMS, CHARGE, SPIN
from mobml_pyscf.prediction import get_ncore, get_nval, _localize_valence

GEOM_INDEX = 0  #UPDATE HERE -- which geometry to inspect (0 = the first one that failed)
BASIS = "aug-cc-pVDZ"  # must match get_mobml_features.py's BASIS


def build_mol(coords):
    mol = gto.Mole()
    mol.atom = [(ATOMS[i], coords[i]) for i in range(len(ATOMS))]
    mol.unit = 'Bohr'
    mol.basis = BASIS
    mol.charge = CHARGE
    mol.spin = SPIN
    mol.build()
    return mol


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def main():
    print(f"SYSTEM_TAG={SYSTEM_TAG}  ATOMS={ATOMS}  CHARGE={CHARGE}  SPIN={SPIN}  BASIS={BASIS}")

    geometries = np.load(f'/home/pca_and_fps/output/{SYSTEM_TAG}/{SYSTEM_TAG}_mobml_training_cds.npy')
    coords = geometries[GEOM_INDEX]
    mol = build_mol(coords)
    n_electrons = mol.nelectron
    print(f"geometry index {GEOM_INDEX}, {mol.nao} AOs, {n_electrons} electrons total")

    expected_n_alpha = (n_electrons + SPIN) // 2
    expected_n_beta = n_electrons - expected_n_alpha
    print(f"expected (from CHARGE/SPIN alone): n_alpha={expected_n_alpha}, n_beta={expected_n_beta}")

    print("\n--- (c) does pyscf.scf.RHF(mol) actually behave like ROHF for this open-shell mol? ---")
    rhf_probe = pyscf.scf.RHF(mol).density_fit()
    rhf_probe.run(verbose=0)
    print(f"type(pyscf.scf.RHF(mol)) = {type(rhf_probe).__name__}")
    rhf_occ = np.array(rhf_probe.get_occ())
    rhf_nocc = np.sum(rhf_occ > 0)
    check(
        "pyscf.scf.RHF(mol) occupied-orbital count matches n_alpha",
        rhf_nocc == expected_n_alpha,
        f"got nocc={rhf_nocc} via RHF(mol), expected n_alpha={expected_n_alpha} "
        f"-- if these differ, RHF(mol) is NOT correctly handling this open-shell "
        f"mol (silently doing something other than a real ROHF calculation), and "
        f"any earlier result computed via that path is unreliable.",
    )

    print("\n--- run the real thing: explicit ROHF ---")
    mf = pyscf.scf.ROHF(mol).density_fit()
    mf.run(verbose=0)
    occupations = np.array(mf.get_occ()).astype(int)
    nocc_beta = np.sum(occupations > 1)
    nocc_alpha = np.sum(occupations > 0)
    n_somo = nocc_alpha - nocc_beta
    print(f"ROHF: nocc_alpha={nocc_alpha}, nocc_beta={nocc_beta}, n_somo={n_somo}")
    check("nocc_alpha matches expected n_alpha", nocc_alpha == expected_n_alpha)
    check("nocc_beta matches expected n_beta", nocc_beta == expected_n_beta)
    check("exactly one SOMO (n_somo == SPIN)", n_somo == SPIN)

    print("\n--- (a) are dm_alpha/dm_beta from mf.make_rdm1() what we assumed? ---")
    S = mol.intor('int1e_ovlp')
    dm_alpha, dm_beta = mf.make_rdm1()
    trace_alpha = np.trace(dm_alpha @ S)
    trace_beta = np.trace(dm_beta @ S)
    trace_spin = np.trace((dm_alpha - dm_beta) @ S)
    print(f"trace(dm_alpha @ S) = {trace_alpha}  (should equal n_alpha = {expected_n_alpha})")
    print(f"trace(dm_beta @ S)  = {trace_beta}  (should equal n_beta = {expected_n_beta})")
    print(f"trace((dm_alpha - dm_beta) @ S) = {trace_spin}  (should equal SPIN = {SPIN})")
    dm_ok = check("dm_alpha trace correct", abs(trace_alpha - expected_n_alpha) < 1e-6)
    dm_ok &= check("dm_beta trace correct", abs(trace_beta - expected_n_beta) < 1e-6)
    dm_ok &= check("spin-density trace correct", abs(trace_spin - SPIN) < 1e-6)
    if not dm_ok:
        print(
            "\n=> dm_alpha/dm_beta themselves are wrong (wrong convention, wrong "
            "basis, or make_rdm1() doesn't return what get_features_rohf_combined "
            "assumed). This is the root cause -- fix the density-matrix "
            "reconstruction before looking at anything below."
        )
        return

    print("\n--- (a continued) sanity-check the reconstructed F_alpha/F_beta ---")
    hcore = mf.get_hcore()
    J, K = mf.get_jk(mol, dm=np.array([dm_alpha, dm_beta]))
    F_alpha = hcore + (J[0] + J[1]) - K[0]
    F_beta = hcore + (J[0] + J[1]) - K[1]
    check("F_alpha is symmetric", np.allclose(F_alpha, F_alpha.T, atol=1e-8))
    check("F_beta is symmetric", np.allclose(F_beta, F_beta.T, atol=1e-8))
    check("F_alpha, F_beta are finite (no NaN/Inf)", np.all(np.isfinite(F_alpha)) and np.all(np.isfinite(F_beta)))
    # HF total energy from F_alpha/F_beta should match mf.e_tot -- the
    # strongest available check that F_alpha/F_beta are actually correct,
    # not just well-formed.
    e_from_F = 0.5 * (np.sum((F_alpha + hcore) * dm_alpha) + np.sum((F_beta + hcore) * dm_beta)) + mol.energy_nuc()
    print(f"Energy from reconstructed F_alpha/F_beta: {e_from_F}")
    print(f"mf.e_tot (ROHF's own reported energy):    {mf.e_tot}")
    check("reconstructed-F energy matches mf.e_tot", abs(e_from_F - mf.e_tot) < 1e-6)

    print("\n--- (b) does the IAO-localized occupied basis fully span the true VALENCE occupied space? ---")
    F_effective = mf.get_fock()
    Cloc_val, R, ncore, nvalocc = _localize_valence(mol, F_effective, mf.mo_coeff, nocc_alpha, verbose=False)
    ncore_heuristic = get_ncore(mol.atom_charges())
    nval_heuristic = get_nval(mol.atom_charges())
    n_valence_electrons = n_electrons - 2 * ncore_heuristic
    print(f"ncore (heuristic) = {ncore_heuristic}, nval (heuristic) = {nval_heuristic}, nvalocc = {nvalocc}")
    print(f"Cloc_val shape = {Cloc_val.shape}  (columns: [0:{nvalocc}] valence-occupied, [{nvalocc}:] valence-virtual "
          f"-- Cloc_val never includes the {ncore_heuristic} core orbital(s) at all, by design)")

    # Cloc_val's occupied/virtual boundary is nvalocc = nocc_alpha - ncore,
    # NOT nocc_alpha -- Cloc_val only spans the valence space to begin
    # with (core orbitals were already excluded when _localize_valence
    # built Cvalocc), matching mobfeatures.FJK_to_features's own internal
    # nvocc = nocc - ncore convention. Slicing at nocc_alpha instead
    # silently pulls in `ncore` virtual columns in place of the `ncore`
    # core columns Cloc_val never had.
    occ_block = Cloc_val[:, :nvalocc]
    dm_total = dm_alpha + dm_beta
    spin_dm = dm_alpha - dm_beta

    # occ_block's columns are S-orthonormal (occ_block.T @ S @ occ_block ==
    # I), not orthonormal under a plain dot product -- so projecting a
    # density matrix (a contravariant C_occ @ C_occ.T object, unlike the
    # Fock/integral tensors elsewhere in prediction.py) into this basis
    # needs an S sandwich on both sides. Print both forms to show the
    # difference directly rather than asserting it.
    total_pop_nosandwich = np.einsum("pi,pq,qi->i", occ_block, dm_total, occ_block, optimize=True)
    spin_pop_nosandwich = np.einsum("pi,pq,qi->i", occ_block, spin_dm, occ_block, optimize=True)
    print(f"[no S sandwich -- WRONG] sum(total_pop) = {np.sum(total_pop_nosandwich)}, "
          f"sum(spin_pop) = {np.sum(spin_pop_nosandwich)}")

    total_pop = np.einsum("pi,pq,qi->i", occ_block, S @ dm_total @ S, occ_block, optimize=True)
    spin_pop = np.einsum("pi,pq,qi->i", occ_block, S @ spin_dm @ S, occ_block, optimize=True)
    print(f"sum(total_pop) over occupied block = {np.sum(total_pop)}  "
          f"(should equal n_valence_electrons = n_electrons - 2*ncore = {n_valence_electrons})")
    print(f"sum(spin_pop) over occupied block  = {np.sum(spin_pop)}  (should equal SPIN = {SPIN})")
    print(f"spin_pop per orbital: {spin_pop}")
    span_ok = check(
        "occupied localized block accounts for essentially all valence electrons",
        abs(np.sum(total_pop) - n_valence_electrons) < 1e-3,
    )
    span_ok &= check(
        "occupied localized block accounts for essentially all spin density",
        abs(np.sum(spin_pop) - SPIN) < 1e-3,
    )
    somo_rows = np.where(spin_pop > 0.5)[0]
    somo_ok = check(
        f"exactly {SPIN} orbital(s) cleanly carry the spin population (>0.5)",
        len(somo_rows) == SPIN,
        f"found {len(somo_rows)}",
    )

    if not span_ok:
        print(
            "\n=> Even with the S sandwich and correct nvalocc boundary, the "
            "IAO-localized occupied basis (Cloc_val's first nvalocc columns) does "
            "NOT fully span the true valence occupied space -- some electron/spin "
            "density is being lost in the IAO projection step itself for this "
            "IAO projection step itself for this open-shell case. This points at "
            "_localize_valence / pyscf.lo.iao.iao's handling of an open-shell mol, "
            "not at the Fock/density reconstruction."
        )
    elif not somo_ok:
        print(
            "\n=> Occupied space is fully captured and spin density sums to the "
            "right total, but isn't cleanly isolated on one orbital -- Boys "
            "localization is genuinely smearing the SOMO across several orbitals "
            "for this molecule/geometry, rather than a bug upstream of it."
        )
    else:
        print("\n=> Everything checks out for this geometry.")


if __name__ == '__main__':
    main()
