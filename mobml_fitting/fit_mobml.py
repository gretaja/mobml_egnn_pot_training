import sys
import os
sys.path.insert(0, '/home/packages')
sys.path.insert(0, '/home/pipeline_config')

import numpy as np

from GPR import GP
from GPR.kern import Matern32, Linear, SumKernel, Summation

from system_config import SYSTEM_TAG, MOBML_N_TRAINING, MOBML_TEST_SIZE

# =========================
# USER SETTINGS (EDIT THESE)
# =========================

# This script only fits a GP on features already computed by
# get_mobml_features.py -- it never runs PySCF itself, so unlike that
# script it needs no ATOMS/CHARGE/SPIN/BASIS/mol-building and no MPI pool
# (see fit_mobml.sh, which should launch this as a single plain process,
# not via `-m mpi4py.futures`).

# Appended to output_dir below. Set this to something distinct whenever
# running a comparison variant of this script (e.g. a different kernel or
# hyperparameter choice) that might run concurrently with, or before
# you're done inspecting, another run -- otherwise both runs write the
# SAME model/learning-curve filenames to the SAME directory and the
# second one silently overwrites the first's results mid-comparison.
RUN_LABEL = ""  #UPDATE HERE

# Must match whatever FEATURE_LABEL get_mobml_features.py was run with for
# the feature set you want to fit here -- selects which of its output
# files (e.g. training_mob_features{FEATURE_LABEL}.npy) this script reads.
# Independent of RUN_LABEL above: that controls where THIS script's own
# output goes, this controls which precomputed features it reads in.
FEATURE_LABEL = ""  #UPDATE HERE

# Which ground-truth energy method to train the Delta-ML correction
# against -- "ccsdt" (default) reads parse_ccsdt_energies.py's DLPNO-
# CCSD(T)/aug-cc-pVTZ output; "b2plyp" reads the B2PLYP-D3/aug-cc-pVTZ
# output from b2lyp_d3_tz_orca.sh/.inp instead. parse_ccsdt_energies.py's
# __main__ now takes a matching argument -- run
# `python3 parse_ccsdt_energies.py b2plyp` to produce the
# energies/valid_indices .npy files this reads for that case; they don't
# exist until you do. Remember to also set RUN_LABEL to something
# distinct when switching this, same as any other comparison variant, so
# the two don't overwrite each other's saved models.
ENERGY_METHOD = "ccsdt"  #UPDATE HERE -- "ccsdt" or "b2plyp"

# (output_files_dir/filename_suffix tag matching what ccsdt_tz_orca.sh /
# b2lyp_d3_tz_orca.sh actually name their output -- model-filename tag --
# print label)
_ENERGY_METHODS = {
    "ccsdt": ("dlpno_ccsdt_augccpvtz", "ccsdt_tz", "CCSD(T)"),
    "b2plyp": ("b2lyp_d3_augccpvtz", "b2plyp_tz", "B2PLYP-D3"),
}
if ENERGY_METHOD not in _ENERGY_METHODS:
    raise ValueError(f"Unsupported ENERGY_METHOD={ENERGY_METHOD!r} -- must be one of {list(_ENERGY_METHODS)}")
_energy_dir_tag, _energy_model_tag, _energy_label = _ENERGY_METHODS[ENERGY_METHOD]

def main():
    print(f"RUN_LABEL={RUN_LABEL!r}  FEATURE_LABEL={FEATURE_LABEL!r}  ENERGY_METHOD={ENERGY_METHOD!r}")

    # All output files for this system are written under here rather than
    # loose in /home/mobml_fitting, so multiple systems' outputs don't mix
    # -- and RUN_LABEL keeps concurrent/comparison runs from mixing either.
    output_dir = f'/home/mobml_fitting/{SYSTEM_TAG}_models{RUN_LABEL}'
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================

    mob_features = np.load(f'/home/hf_mob_features/{SYSTEM_TAG}_training_mob_features{FEATURE_LABEL}.npy')
    feature_selection = np.load(f'/home/hf_mob_features/{SYSTEM_TAG}_feature_selection{FEATURE_LABEL}.npy')
    print("Features shape:", mob_features.shape)
    print("Feature selection length:", len(feature_selection))

    hf_energies = np.load(f'/home/hf_mob_features/{SYSTEM_TAG}_training_hf_dz{FEATURE_LABEL}.npy').reshape(-1)
    print("HF Energies shape:", hf_energies.shape)

    energy_suffix = f'{SYSTEM_TAG}_{_energy_dir_tag}'
    ccsdt_dir = f'/home/ccsdt_energies/{energy_suffix}_output'
    ccsdt_energies = np.load(f'{ccsdt_dir}/{energy_suffix}_energies.npy')
    print(f"{_energy_label} Energies shape:", ccsdt_energies.shape)

    # hf_mob_features processes every geometry regardless of whether its
    # {_energy_label} job later succeeded, so mob_features/hf_energies can
    # be longer than ccsdt_energies if any were excluded (see
    # parse_ccsdt_energies.py). Select the same surviving geometries here
    # so all three stay aligned by geometry, not just by raw position.
    valid_idx = np.load(f'{ccsdt_dir}/{energy_suffix}_valid_indices.npy')
    if len(valid_idx) != len(mob_features):
        print(f"Excluding {len(mob_features) - len(valid_idx)} geometries with no usable {_energy_label} energy")

    X = mob_features[valid_idx]
    hf_energies = hf_energies[valid_idx]
    deltas = ccsdt_energies - hf_energies

    add_energy = np.mean(deltas)
    targets = 1000 * (deltas - add_energy)
    Y = np.array(targets).reshape(-1, 1)

    #print(deltas)

    # =========================
    # TRAIN / TEST SPLIT
    # =========================

    np.random.seed(42)
    indices = np.random.permutation(len(X))

    # Training-set sizes for the learning curve -- set in system_config.py
    # (MOBML_N_TRAINING), since how many training configurations are
    # available depends on the system's DMC-sampled dataset size.
    n_training = MOBML_N_TRAINING

    # Held-out test-set size -- set in system_config.py (MOBML_TEST_SIZE).
    test_size = MOBML_TEST_SIZE

    # Learning-curve data: one entry per n_training, filled in below and
    # saved to .npy files at the end.
    n_training_used = []
    test_mae_curve = []
    test_avg_error_curve = []

    test_idx = indices[:test_size]
    np.save(f'{output_dir}/{SYSTEM_TAG}_test_idx.npy', test_idx)

    # test_idx above is a POSITION into X (i.e. into the valid_idx-compacted
    # arrays) -- valid_idx[test_idx] is the actual GEOMETRY index each test
    # point corresponds to, same order as test_idx, fixed across every n
    # (only train_idx changes per n_training). Saved once here so an
    # external script (e.g. a ROHF/UHF-gap-vs-error correlation check) can
    # build mol objects for exactly these test geometries without having to
    # re-derive the valid_idx/test_idx chain itself.
    test_geometry_idx = valid_idx[test_idx]
    np.save(f'{output_dir}/{SYSTEM_TAG}_test_geometry_idx.npy', test_geometry_idx)

    for n in n_training:
        train_idx = indices[test_size:n+test_size]

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        print("\nTrain size:", len(X_train))
        print("Test size:", len(X_test))

        # =========================
        # STANDARDIZE FEATURES
        # =========================

        # Tried standardizing (zero mean / unit variance per raw feature
        # column, training-set statistics only) to address the isotropic
        # Matern32 / unscaled Linear kernels (see GPR/kern/stationary.py,
        # GPR/kern/polynomial.py) potentially being dominated by a few
        # large-magnitude raw columns -- empirically made test error WORSE
        # across every training size (e.g. n=500: 608 -> 978 cm^-1), not
        # better, so it's reverted here rather than kept as a regression.
        # feature_mean/feature_std are still computed (and saved with the
        # model below) for reference, but NOT applied to X_train/X_test.
        train_rows = X_train.reshape(-1, X_train.shape[-1])
        feature_mean = train_rows.mean(axis=0)
        feature_std = train_rows.std(axis=0)
        feature_std[feature_std < 1e-12] = 1.0  # avoid div-by-zero on a constant column within this split

        # =========================
        # TRAIN GP
        # =========================

        print("\nTraining GP...")

        k_matern = Matern32()
        k_matern.set_lengthscale(1.5)

        inner_k = SumKernel([k_matern, Linear()])
        k = Summation(inner_k)
        k.finalize()

        gp = GP(X_train, Y_train, k, noise=1e-4)
        gp.fit()
        gp.optimize(messages=True)

        # =========================
        # EVALUATE
        # =========================

        print("\nEvaluating...")

        pred_train = gp.predict(X_train)
        pred_test = gp.predict(X_test)

        # Convert from the (scaled, offset) training units back to cm^-1
        # before computing error statistics.
        error_train = (pred_train - Y_train) / 1000 * 219474.63136320
        error_test = (pred_test - Y_test) / 1000 * 219474.63136320

        mae_train = np.mean(np.abs(error_train))
        mae_test = np.mean(np.abs(error_test))

        # Signed mean error (bias), as distinct from MAE.
        avg_error_test = np.mean(error_test)

        # Per-geometry SIGNED test error (cm^-1), same order as test_idx/
        # test_geometry_idx above -- lets an external script correlate
        # error against anything computed per-geometry (e.g. the ROHF/UHF
        # energy gap from compare_uhf_rohf.py) without needing aggregate
        # MAE alone. Signed rather than absolute so a correlation can also
        # reveal a directional bias, not just magnitude; take np.abs() of
        # this array for magnitude-only correlations.
        signed_error_test = error_test.reshape(-1)
        np.save(f'{output_dir}/{SYSTEM_TAG}_test_errors_n{n}.npy', signed_error_test)

        print("\n=========================")
        print(f"RESULTS: {n}")
        print("=========================")
        print("Train MAE:", mae_train)
        print("Test MAE:", mae_test)
        print("Test average error:", avg_error_test)

        n_training_used.append(len(X_train))
        test_mae_curve.append(mae_test)
        test_avg_error_curve.append(avg_error_test)

        # =========================
        # SAVE MODEL
        # =========================

        model = { "model": gp.to_dict(),
                "feature_sele": feature_selection,
                "add_energy": add_energy,
                # NOT applied to this model's inputs (see STANDARDIZE
                # FEATURES above) -- saved for reference/diagnostics only.
                # Do not apply these to a new geometry's raw features
                # before calling gp.predict(); this model was trained on
                # raw, unstandardized features.
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                }

        model_filename = f"{output_dir}/mobml_{SYSTEM_TAG}_{_energy_model_tag}_{n}_matern_linear.npy"
        np.save(model_filename, model)

        print(f"\nModel saved to {model_filename}")

    # =========================
    # SAVE LEARNING CURVE DATA
    # =========================

    n_training_filename = f"{output_dir}/{SYSTEM_TAG}_learning_curve_n_training.npy"
    mae_filename = f"{output_dir}/{SYSTEM_TAG}_learning_curve_test_mae.npy"
    avg_error_filename = f"{output_dir}/{SYSTEM_TAG}_learning_curve_test_avg_error.npy"

    np.save(n_training_filename, np.array(n_training_used))
    np.save(mae_filename, np.array(test_mae_curve))
    np.save(avg_error_filename, np.array(test_avg_error_curve))

    print("\nLearning curve data saved:")
    print(" ", n_training_filename)
    print(" ", mae_filename)
    print(" ", avg_error_filename)

if __name__ == '__main__':
    main()
