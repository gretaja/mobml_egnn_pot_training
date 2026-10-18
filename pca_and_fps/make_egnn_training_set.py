import numpy as np
import random
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

sys.path.insert(0, '/home/packages')
sys.path.insert(0, '/home/pipeline_config')
# dmc_sampling is already bound whole (not just output/) by this script's
# .sh, so its potential module is reachable here too.
sys.path.insert(0, '/home/dmc_sampling/scripts/potential')

import pyvibdmc as pv

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import h5py
import joblib

from system_config import SYSTEM_TAG, ATOMS, HEAVY_ONLY_MASS_SCALING
# Same mobml-corrected potential used to generate the DMC training sims --
# used below to get a consistent energy for the optimized/starting geometry.
from pyscfpot_mobml import scf_pot

au_to_ang = 0.529177249

# Must match the tag broad_mobml.py actually wrote its sim output under,
# which depends on the same HEAVY_ONLY_MASS_SCALING config value.
_MASS_SCALING_TAG = "scaled_heavy_x02" if HEAVY_ONLY_MASS_SCALING else "scaled_all_x02"

def make_total_sets(system,sim_type):

    train_energies_list = []
    train_cds_list = []

    # NOTE: verify this matches where your DMC .hdf5 sims actually live under
    # the dmc_sampling bind (/home/dmc_sampling -> /gscratch/mccoy/pot_training/dmc_sampling).
    if sim_type == 'min':
        sim = f'/home/dmc_sampling/output/{system}/mobml_sims/500w_1000ts_1dt'
        timesteps = 1000
    elif sim_type == 'broad':
        sim = f'/home/dmc_sampling/output/{system}/mobml_sims/{_MASS_SCALING_TAG}_1000w_1000ts_10dt'
        timesteps = 500
    else:
        print("Not a valid sim type")
        return None, None

    training_sim = pv.SimInfo(f'{sim}/{system}_0_sim_info.hdf5')

    for i in range(timesteps):
        cds, energies = training_sim.get_training(f'{sim}/{system}_0_training_{i}ts.hdf5',ret_ang=False, ret_cm=False)

        train_cds_list.append(cds)
        train_energies_list.append(energies)

    train_cds = np.concatenate(train_cds_list)
    train_energies = np.concatenate(train_energies_list)

    return train_cds, train_energies

def distance_spectrum(coords, atom_labels=None, heavy_only=False):
    """
    coords: (n_molecules, n_atoms, 3)
    atom_labels: sequence of element symbols, length n_atoms -- required
        when heavy_only=True so the heavy-atom subset can be selected.
    heavy_only: if True, restrict the pairwise distance matrix to heavy
        atoms only (anything other than 'H'), dropping hydrogens. If False
        (default), all pairwise distances are used, as before.
    """
    coords = np.asarray(coords)

    if heavy_only:
        if atom_labels is None:
            raise ValueError("atom_labels is required when heavy_only=True")
        heavy_idx = [i for i, a in enumerate(atom_labels) if a.upper() != 'H']
        if not heavy_idx:
            raise ValueError("heavy_only=True but atom_labels contains no heavy (non-H) atoms")
        coords = coords[:, heavy_idx, :]

    n_molecules, n_atoms, _ = coords.shape

    # ---------------------------------
    # 1. Compute batched distance matrices
    # ---------------------------------
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    D = np.linalg.norm(diff, axis=-1)

    # Zero diagonal explicitly (numerical safety)
    idx = np.arange(n_atoms)
    D[:, idx, idx] = 0.0

    # ---------------------------------
    # 2. Compute eigenvalues (batched)
    # ---------------------------------
    # eigvalsh is for symmetric matrices (faster + stable)
    eigvals = np.linalg.eigvalsh(D)  # shape (n_molecules, n_atoms)

    # ---------------------------------
    # 3. Sort eigenvalues (ascending)
    # ---------------------------------
    eigvals_sorted = np.sort(eigvals, axis=1)

    return eigvals_sorted

def save_pca(system, coords, atom_labels=None, heavy_only=False):

    X = distance_spectrum(coords, atom_labels=atom_labels, heavy_only=heavy_only)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3) #only 3 components for H2O, change for larger systems
    X_pca = pca.fit_transform(X_scaled)

    joblib.dump(scaler, f'/home/pca_and_fps/output/{system}/scaler.joblib')
    joblib.dump(pca, f'/home/pca_and_fps/output/{system}/pca.joblib')

def calc_pca(system, coords_new, atom_labels=None, heavy_only=False):

    scaler = joblib.load(f'/home/pca_and_fps/output/{system}/scaler.joblib')
    pca = joblib.load(f'/home/pca_and_fps/output/{system}/pca.joblib')

    X_new = distance_spectrum(coords_new, atom_labels=atom_labels, heavy_only=heavy_only)

    X_new_scaled = scaler.transform(X_new)
    X_new_pca = pca.transform(X_new_scaled)

    return X_new_pca

def farthest_point_sampling(X, k, start_idx=None):
    """
    X: (N, d) PCA embedding
    k: number of points to select
    start_idx: optional initial index
    """
    N = X.shape[0]
    selected = np.zeros(k, dtype=int)

    # Choose initial point
    if start_idx is None:
        selected[0] = np.random.randint(N)
    else:
        selected[0] = start_idx

    # Track minimum distance to selected set
    distances = np.linalg.norm(X - X[selected[0]], axis=1)

    for i in range(1, k):
        selected[i] = np.argmax(distances)
        new_dist = np.linalg.norm(X - X[selected[i]], axis=1)
        distances = np.minimum(distances, new_dist)

    return selected

def random_sample(cds, energies, size):
    # Generate list of indices
    indices = list(range(len(cds)))

    # Shuffle the indices randomly
    random.shuffle(indices)

    split_idx = size

    # Partition the data
    sample_indices = indices[:split_idx]

    cds_sample = cds[sample_indices]
    energies_sample = energies[sample_indices]

    return cds_sample, energies_sample

def main():

    # random_sample() (random.shuffle) and farthest_point_sampling()
    # (np.random.randint) are unseeded otherwise, so which geometries get
    # selected -- and in what order -- would change on every run. Not
    # currently called below (this main() uses every DMC sample directly
    # for the single-isomer case), but seeded here to match
    # make_mobml_training_set.py in case you wire the multi-isomer
    # PCA/FPS path in later.
    random.seed(42)
    np.random.seed(42)

    system = SYSTEM_TAG
    atoms = ATOMS
    total_training = 6000 #UPDATE HERE

    opt_geom_path = f'/home/opt_geoms/{system}_opt.npy'
    opt_geom = np.load(opt_geom_path)

    # Evaluate with the same mobml-corrected potential used to generate the
    # DMC training sims, so the anchor point is on the same energy footing
    # as min_energies/broad_energies below. scf_pot expects (N, n_atoms, 3).
    opt_energy = scf_pot(opt_geom[None])[0]

    min_cds, min_energies = make_total_sets(system, 'min')
    broad_cds, broad_energies = make_total_sets(system, 'broad')

    egnn_training_cds = np.concatenate([np.array([opt_geom]), min_cds, broad_cds])
    egnn_training_energies = np.concatenate([np.array([opt_energy]), min_energies, broad_energies])

    np.save(f'/home/pca_and_fps/output/{system}/{system}_egnn_training_cds.npy',egnn_training_cds)
    np.save(f'/home/pca_and_fps/output/{system}/{system}_egnn_training_energies.npy',egnn_training_energies)

if __name__ == '__main__':
    main()
