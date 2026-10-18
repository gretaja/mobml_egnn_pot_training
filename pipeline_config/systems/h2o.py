"""
Per-system values for h2o. Selected by setting PIPELINE_SYSTEM=h2o in the
.sh before running -- see system_config.py, which loads whichever file in
this directory matches PIPELINE_SYSTEM.
"""

# Atom ordering -- MUST match the order used everywhere coordinates for
# this system are generated/stored (DMC sampling starting structure,
# training-set .xyz files, EGNN checkpoints, etc.). Changing this after
# data already exists for a system will silently mismatch atom identities
# against already-stored coordinates.
ATOMS = ['O', 'H', 'H']

# Total molecular charge.
CHARGE = 0

# Spin multiplicity, 2S+1 (1 = closed-shell singlet, 2 = doublet, ...).
MULTIPLICITY = 1

# Whether the PCA/FPS training-set-selection step (pca_and_fps/) restricts
# its pairwise-distance embedding to heavy (non-H) atoms only, vs using
# every atom -- see distance_spectrum()'s heavy_only param. With only one
# heavy atom (O), heavy-only is degenerate for h2o, so False here.
HEAVY_ONLY_PCA = False

# Whether broad DMC sampling's mass-scaling (dmc_sampling/scripts/broad_*.py)
# scales only heavy (non-H) atoms' masses, leaving hydrogens at their real
# mass (True), or scales every atom's mass including hydrogens (False).
# Also picks which sim output subfolder name is used/read
# (scaled_heavy_x02 vs scaled_all_x02), so pca_and_fps's training-set
# scripts read from the one that actually matches what was scaled.
HEAVY_ONLY_MASS_SCALING = False

# Training-set sizes to try for the MOB-ML learning curve
# (mobml_fitting/fit_mobml.py) -- how many training configurations each GP
# fit uses. Must stay well under MOBML_TRAINING_SET_SIZE below (minus
# fit_mobml.py's test_size), since that's the pool it draws from.
MOBML_N_TRAINING = [50, 100, 200, 500, 800]

# Total number of MOB-ML training configurations to assemble
# (pca_and_fps/make_mobml_training_set.py): split 25% randomly-sampled
# near the minimum, 75% farthest-point-sampled from the broad DMC set.
# Capped by how much DMC-sampled data actually exists for this system.
MOBML_TRAINING_SET_SIZE = 6000

# Held-out test-set size for the MOB-ML learning curve
# (mobml_fitting/fit_mobml.py). Drawn from the same
# MOBML_TRAINING_SET_SIZE pool, so MOBML_TEST_SIZE + max(MOBML_N_TRAINING)
# must stay under it.
MOBML_TEST_SIZE = 200
