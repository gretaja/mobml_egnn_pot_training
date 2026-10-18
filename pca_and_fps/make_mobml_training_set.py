import numpy as np
import random
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

sys.path.insert(0, '/home/packages')
sys.path.insert(0, '/home/pipeline_config')

import pyvibdmc as pv

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import h5py
import joblib

from system_config import SYSTEM_TAG, ATOMS, HEAVY_ONLY_PCA, HEAVY_ONLY_MASS_SCALING, MOBML_TRAINING_SET_SIZE

au_to_ang = 0.529177249

# Must match the tag broad_dft.py actually wrote its sim output under,
# which depends on the same HEAVY_ONLY_MASS_SCALING config value.
_MASS_SCALING_TAG = "scaled_heavy_x02" if HEAVY_ONLY_MASS_SCALING else "scaled_all_x02"

def make_total_sets(system,sim_type):

    train_energies_list = []
    train_cds_list = []

    # NOTE: verify this matches where your DMC .hdf5 sims actually live under
    # the dmc_sampling bind (/home/dmc_sampling -> /gscratch/mccoy/pot_training/dmc_sampling).
    if sim_type == 'min':
        sim = f'/home/dmc_sampling/output/{system}/dft_sims/100w_1000ts_1dt'
        timesteps = 1000
    elif sim_type == 'broad':
        sim = f'/home/dmc_sampling/output/{system}/dft_sims/{_MASS_SCALING_TAG}_200w_500ts_10dt'
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
        atoms only (anything other than 'H' -- i.e. C or O here), dropping
        hydrogens. If False (default), all pairwise distances are used, as
        before.
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
    
def save_xyz_file(filename, atoms, coordinates):
    """
    Save an XYZ file with given atoms and coordinates.

    Parameters:
    - filename: The name of the file to save.
    - atoms: A list of atom symbols.
    - coordinates: A list of lists, each containing x, y, and z coordinates.

    Example:
    save_xyz_file('molecule.xyz', ['O', 'H', 'H'], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    """
    lines = [str(len(atoms)), ""]
    lines.extend(
        f"{atom} {x:.8f} {y:.8f} {z:.8f}"
        for atom, (x, y, z) in zip(atoms, coordinates)
    )

    # Build the whole file in memory and issue a single write() call instead
    # of one write() per atom -- much cheaper when filename lives on a
    # network-mounted scratch filesystem where each syscall has real latency.
    with open(filename, 'w') as file:
        file.write("\n".join(lines) + "\n")

def _write_xyz_chunk(xyz_dir, system, atoms, start_idx, coords_chunk, threads_per_worker=4):
    """
    Write a contiguous slice of geometries to disk. Runs inside one MPI
    worker rank (see main()); the small thread pool overlaps the
    open/write/close latency of individual files within that rank, since
    each write() call releases the GIL while it blocks on I/O.
    """
    with ThreadPoolExecutor(max_workers=threads_per_worker) as pool:
        for offset, cd in enumerate(coords_chunk):
            i = start_idx + offset
            pool.submit(save_xyz_file, f'{xyz_dir}/{system}_{i}.xyz', atoms, cd)

def main():

    # random_sample() (random.shuffle) and farthest_point_sampling()
    # (np.random.randint) were previously unseeded, so which geometries
    # got selected -- and in what order -- changed every time this script
    # ran. Anything downstream (hf_mob_features, ccsdt_energies) that reads
    # the resulting {system}_mobml_training_cds.npy/{system}_xyz/ assumes
    # a FIXED geometry-index correspondence; an unseeded rerun after those
    # already exist silently breaks that alignment instead of erroring.
    # Seeding both RNGs here (numpy for farthest_point_sampling, stdlib
    # random for random.shuffle) makes this step reproducible, matching
    # fit_mobml.py's existing np.random.seed(42) for its train/test split.
    random.seed(42)
    np.random.seed(42)

    system = SYSTEM_TAG
    atoms = ATOMS

    # If True, the PCA/FPS distance-spectrum step below only uses distances
    # between heavy atoms (anything other than H) -- makes the embedding
    # ignore hydrogen positions entirely. Set in system_config.
    heavy_only = HEAVY_ONLY_PCA

    n_training = MOBML_TRAINING_SET_SIZE

    min_cds, min_energies = make_total_sets(system, 'min')
    broad_cds, broad_energies = make_total_sets(system, 'broad')

    # round() after multiplying, not before -- round(n_training)*0.25 would
    # leave this a float (e.g. 1500.0), which random_sample's slicing and
    # farthest_point_sampling's np.zeros(k, ...) both need to be an int.
    min_cds_sample, min_energies_sample = random_sample(min_cds, min_energies, round(n_training * 0.25))

    save_pca(system, min_cds_sample, atom_labels=atoms, heavy_only=heavy_only)

    X_pca = calc_pca(system, broad_cds, atom_labels=atoms, heavy_only=heavy_only)

    subset_idx = farthest_point_sampling(X_pca, round(n_training * 0.75))

    broad_cds_sample = broad_cds[subset_idx]
    
    mobml_training_cds = np.concatenate([min_cds_sample, broad_cds_sample])
    
    np.save(f'/home/pca_and_fps/output/{system}/{system}_mobml_training_cds.npy',mobml_training_cds)

    xyz_dir = f'/home/pca_and_fps/output/{system}/{system}_xyz'
    os.makedirs(xyz_dir, exist_ok=True)

    mobml_training_cds_ang = mobml_training_cds * au_to_ang  # convert once, not per-file

    # This job already reserves several MPI ranks (--ntasks-per-node in the
    # .sh) via `python -m mpi4py.futures`, but only rank 0 (this process)
    # normally does any work -- the rest sit idle. Put them to use: split
    # the geometries into one contiguous chunk per worker rank (few large
    # messages instead of one MPI round-trip per file) and let each rank
    # write its chunk with a small internal thread pool for extra I/O
    # overlap. Falls back to local threads only if there are no MPI workers
    # (e.g. running the script directly, without `-m mpi4py.futures`).
    n_workers = MPI.COMM_WORLD.Get_size() - 1

    if n_workers > 0:
        chunks, start = [], 0
        for chunk in np.array_split(mobml_training_cds_ang, n_workers):
            if len(chunk):
                chunks.append((start, chunk))
            start += len(chunk)

        with MPIPoolExecutor() as pool:
            futures = [
                pool.submit(_write_xyz_chunk, xyz_dir, system, atoms, start_idx, chunk)
                for start_idx, chunk in chunks
            ]
            for f in futures:
                f.result()  # surface any exception raised in a worker
    else:
        with ThreadPoolExecutor(max_workers=16) as pool:
            for i, cd in enumerate(mobml_training_cds_ang):
                pool.submit(save_xyz_file, f'{xyz_dir}/{system}_{i}.xyz', atoms, cd)

if __name__ == '__main__':
    main()
