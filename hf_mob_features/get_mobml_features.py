import sys
import os
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
sys.path.insert(0, '/home/packages')
sys.path.insert(0, '/home/pipeline_config')

import numpy as np

from pyscf import gto
from mobml_pyscf import get_features_rhf, get_features_rohf, get_features_uhf

from system_config import SYSTEM_TAG, ATOMS, CHARGE, SPIN

# =========================
# USER SETTINGS (EDIT THESE)
# =========================

# System identity (tag, atom order, charge, spin) comes from
# system_config.py -- edit that file, not here, to point this at a
# different system. BASIS stays local since it's a per-calculation choice,
# not a system property (e.g. the CCSD(T) step uses a different basis).
BASIS = "aug-cc-pVDZ"

atoms = ATOMS

# Which open-shell reference to build features from (ignored for SPIN == 0,
# which always uses closed-shell get_features_rhf): "rohf" / "uhf" --
# get_features_rohf / get_features_uhf each return features_alpha and
# features_beta separately; this concatenates them into ONE array of
# (nocc_alpha-ncore + nocc_beta-ncore) rows per geometry. The downstream
# kernel/fit code (Summation already sums over however many rows a
# geometry has, and is invariant to their order -- see fit_mobml.py)
# doesn't need the two halves to share a localized basis or correspond
# row-for-row, so no combination scheme is needed on top of this.
REFERENCE_METHOD = "uhf"  #UPDATE HERE -- "rohf" or "uhf"

if REFERENCE_METHOD not in ("rohf", "uhf"):
    raise ValueError(f"Unknown REFERENCE_METHOD={REFERENCE_METHOD!r} -- must be 'rohf' or 'uhf'")

# Appended to every output filename below. Leave empty for the default run
# -- set this to something distinct (e.g. "_rohf") whenever you want to
# keep a second REFERENCE_METHOD's cache around at the same time without
# overwriting this one, or this run silently overwrites it. fit_mobml.py
# has a matching FEATURE_LABEL setting to read whichever set back in.
FEATURE_LABEL = ""  #UPDATE HERE

# Number of geometries used to determine which raw MOB feature columns
# are structurally always-zero (padding, see mobfeatures.resizevector)
# for this system, versus real. That padding boundary is fixed by
# nvocc/nvvir, which depend only on atom composition + charge -- not on
# any one conformation -- so it doesn't require the full dataset. Still
# use more than one geometry: a single, possibly symmetric, structure
# could have a genuinely informative column vanish by symmetry and get
# mistaken for padding.
N_SAMPLE_FOR_SELECTION = 20
FEATURE_ZERO_THRESHOLD = 1e-10

# =========================
# BUILD MOLECULE
# =========================

def build_mol(coords):
    mol = gto.Mole()
    mol.atom = [(atoms[i], coords[i]) for i in range(len(atoms))]
    mol.unit = 'Bohr'   # your geometries are in Bohr
    mol.basis = BASIS
    mol.charge = CHARGE
    mol.spin = SPIN
    mol.build()
    return mol

# =========================
# DISTRIBUTED HF + FEATURES
# =========================

def _get_features_chunk(coords_chunk, start_idx):
    """
    Build each molecule and run HF + get_features for a contiguous slice of
    geometries. This is the unit of work sent to one MPI worker rank, so it
    runs CPU-bound PySCF calculations there instead of on the manager.
    Returns (index, hf_energy, features) tuples, with hf_energy=None and the
    error string in place of features on failure -- index continues from
    start_idx so chunks from different ranks can be reassembled in order.
    """
    results = []
    for offset, coords in enumerate(coords_chunk):
        i = start_idx + offset
        try:
            mol = build_mol(coords)
            # SPIN == 0: closed-shell, plain RHF features. SPIN != 0:
            # open-shell -- which reference is picked by REFERENCE_METHOD,
            # see its comment above.
            if SPIN == 0:
                hf, features = get_features_rhf(mol, verbose=False)
            elif REFERENCE_METHOD == "rohf":
                hf, features_alpha, features_beta = get_features_rohf(mol, verbose=False)
                features = np.concatenate([features_alpha, features_beta], axis=0)
            else:
                hf, features_alpha, features_beta = get_features_uhf(mol, verbose=False)
                features = np.concatenate([features_alpha, features_beta], axis=0)
            results.append((i, hf.energy_tot(), features))
        except Exception as e:
            results.append((i, None, str(e)))
    return results

def _get_features_distributed(coords):
    """
    Run _get_features_chunk over `coords` (an (N, n_atoms, 3) array),
    splitting it into one contiguous chunk per MPI worker rank so all N
    geometries are computed concurrently across ranks -- this job already
    reserves several via --ntasks-per-node in the .sh. Falls back to a
    single local call if there are no MPI workers (e.g. running the script
    directly, without `-m mpi4py.futures`). Returns results sorted back
    into original order.
    """
    n_workers = MPI.COMM_WORLD.Get_size() - 1

    if n_workers <= 0:
        return _get_features_chunk(coords, 0)

    bounds = [b for b in np.array_split(np.arange(len(coords)), n_workers) if len(b)]
    with MPIPoolExecutor() as pool:
        futures = [pool.submit(_get_features_chunk, coords[b], int(b[0])) for b in bounds]
        results = []
        for f in futures:
            results.extend(f.result())  # also surfaces any exception raised in a worker

    results.sort(key=lambda r: r[0])
    return results

def main():
    # =========================
    # LOAD DATA
    # =========================

    print(f"REFERENCE_METHOD={REFERENCE_METHOD}  FEATURE_LABEL={FEATURE_LABEL!r}  BASIS={BASIS!r}")
    print("Loading geometries...")
    geometries = np.load(f'/home/pca_and_fps/output/{SYSTEM_TAG}/{SYSTEM_TAG}_mobml_training_cds.npy')
    print("Geometry shape:", geometries.shape)

    # =========================
    # DETERMINE FEATURE SELECTION
    # =========================

    print("Determining feature selection from a sample of geometries...")
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(geometries), size=min(N_SAMPLE_FOR_SELECTION, len(geometries)), replace=False)

    sample_results = _get_features_distributed(geometries[sample_idx])
    sample_features = np.array([features for _, energy, features in sample_results if energy is not None])

    n_failed = len(sample_results) - len(sample_features)
    if n_failed:
        print(f"Warning: {n_failed} of {len(sample_idx)} sampled geometries failed during feature selection")
        for i, energy, err in sample_results:
            if energy is None:
                print(f"  Error at index {i}: {err}")

    if len(sample_features) == 0:
        raise RuntimeError(
            "Every sampled geometry failed -- see the per-index errors printed above. "
            "Nothing usable to determine feature_selection from."
        )

    # Keep a raw column if it's non-zero for at least one occupied orbital
    # in at least one sampled geometry; columns that are zero everywhere in
    # the sample are padding (see mobfeatures.resizevector) and dropped.
    feature_selection = np.where(np.any(np.abs(sample_features) > FEATURE_ZERO_THRESHOLD, axis=(0, 1)))[0]

    print("Full feature length:", sample_features.shape[-1])
    print("Selected feature length:", len(feature_selection))

    np.save(f'/home/hf_mob_features/{SYSTEM_TAG}_feature_selection{FEATURE_LABEL}.npy', feature_selection)

    # =========================
    # PREPROCESS
    # =========================

    X = []

    E_lows = []

    valid_count = 0

    # Progress is reported once the distributed results come back, not
    # incrementally as each geometry finishes on its worker rank -- results
    # still print in original-index order (i % 10 == 0), just as a burst
    # rather than truly live, since the manager only sees them after gather.
    results = _get_features_distributed(geometries)

    for i, E_low, features in results:
        if E_low is None:
            print(f"Error at index {i}: {features}")
            continue

        E_lows.append(E_low)
        X.append(features[:, feature_selection])
        valid_count += 1

        if i % 10 == 0:
            print(f"Processed {i} samples")

    X = np.array(X)
    E_lows = np.array(E_lows).reshape(-1, 1)

    print("\n=========================")
    print("Preprocessing complete")
    print("=========================")
    print("Valid samples:", valid_count)
    print("X shape:", X.shape)
    print("E_lows:", E_lows.shape)

    mob_features_path = f'/home/hf_mob_features/{SYSTEM_TAG}_training_mob_features{FEATURE_LABEL}.npy'
    hf_energies_path = f'/home/hf_mob_features/{SYSTEM_TAG}_training_hf_dz{FEATURE_LABEL}.npy'
    np.save(mob_features_path, X)
    np.save(hf_energies_path, E_lows)
    print(f"\nSaved: {mob_features_path}\n  {hf_energies_path}")

if __name__ == '__main__':
    main()
