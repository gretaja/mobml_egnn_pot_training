"""
Checks whether the model's remaining error is explained by CCSD(T) itself
being unreliable, not by anything on the ML side: parses the T1 diagnostic
(the standard Lee-Taylor multi-reference-character indicator; values above
roughly 0.02 are the conventional threshold for single-reference coupled
cluster becoming untrustworthy) out of the ORCA .out files already on disk
from the CCSD(T) runs, for every TEST geometry, and correlates it against
that geometry's actual per-geometry test error from a completed
fit_mobml.py run.

No new calculations of any kind -- this only reads .out files that already
exist and the test_errors_n{n}.npy/test_geometry_idx.npy fit_mobml.py
already saved. Runs bare-metal, same as parse_ccsdt_energies.py -- pure
text parsing + numpy, no apptainer/PySCF needed:

    python3 correlate_t1_diagnostic_error.py
"""

import sys
import os
import re

import numpy as np

sys.path.insert(0, '/gscratch/mccoy/pot_training/pipeline_config')
from system_config import SYSTEM_TAG, MOBML_N_TRAINING

# Which completed fit_mobml.py run to check -- must match a RUN_LABEL that
# run used (so {SYSTEM_TAG}_models{RUN_LABEL}/ exists with
# test_geometry_idx.npy / test_errors_n{n}.npy in it). Defaults to the UHF
# run since that's the best model found so far.
RUN_LABEL = "_uhf"  #UPDATE HERE
N_TRAINING_TO_CHECK = None  #UPDATE HERE -- None = largest value in MOBML_N_TRAINING

# Flexible about whitespace/punctuation around the label, strict about the
# label itself -- confirmed against this system's actual ORCA output
# (line format: "T1 diagnostic                    ...      0.019401506"),
# so the gap between the label and the number needs to allow more than a
# few characters of padding/dots.
T1_PATTERN = re.compile(r"T1\s*diagnostic.*?([0-9]+\.[0-9]+)", re.IGNORECASE)
# Same pattern parse_ccsdt_energies.py uses for the CCSD(T) energy itself.
ENERGY_PATTERN = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")

here = os.path.dirname(os.path.abspath(__file__))
output_dir = f'{here}/{SYSTEM_TAG}_models{RUN_LABEL}'
ccsdt_dir = f'/gscratch/mccoy/pot_training/ccsdt_energies/{SYSTEM_TAG}_dlpno_ccsdt_augccpvtz_output'
suffix = f"{SYSTEM_TAG}_dlpno_ccsdt_augccpvtz"


def parse_t1(out_path):
    with open(out_path) as f:
        text = f.read()
    match = T1_PATTERN.search(text)
    return float(match.group(1)) if match else None


def parse_energy(out_path):
    with open(out_path) as f:
        text = f.read()
    matches = ENERGY_PATTERN.findall(text)
    return float(matches[-1]) if matches else None


def pearson(a, b):
    a = np.asarray(a) - np.mean(a)
    b = np.asarray(b) - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / denom) if denom > 0 else float('nan')


def main():
    n = N_TRAINING_TO_CHECK or max(MOBML_N_TRAINING)
    print(f"SYSTEM_TAG={SYSTEM_TAG}  RUN_LABEL={RUN_LABEL!r}  checking test errors from n_training={n}")

    test_geometry_idx = np.load(f'{output_dir}/{SYSTEM_TAG}_test_geometry_idx.npy')
    signed_error = np.load(f'{output_dir}/{SYSTEM_TAG}_test_errors_n{n}.npy')
    abs_error = np.abs(signed_error)
    print(f"Test set size: {len(test_geometry_idx)}")

    t1_values = []
    ccsdt_energy_values = []
    missing = []
    for j, g in enumerate(test_geometry_idx):
        out_path = f'{ccsdt_dir}/{SYSTEM_TAG}_{int(g)}_{suffix}.out'
        if not os.path.isfile(out_path):
            missing.append(int(g))
            t1_values.append(np.nan)
            ccsdt_energy_values.append(np.nan)
            continue
        t1 = parse_t1(out_path)
        t1_values.append(t1 if t1 is not None else np.nan)
        energy = parse_energy(out_path)
        ccsdt_energy_values.append(energy if energy is not None else np.nan)

    t1_values = np.array(t1_values)
    ccsdt_energy_values = np.array(ccsdt_energy_values)
    n_found = int(np.sum(~np.isnan(t1_values)))
    print(f"\nT1 diagnostic found for {n_found} / {len(test_geometry_idx)} test geometries "
          f"({len(missing)} missing .out file(s))")

    # Saved in the SAME order/length as test_geometry_idx (and therefore
    # test_errors_n{n}.npy -- test_idx/test_geometry_idx is fixed across
    # every n_training, only train_idx changes, so these two files are
    # reusable against test_errors_n{n}.npy for ANY n, not just the one
    # checked below) -- NaN preserved rather than dropped, so positional
    # alignment with test_errors_n{n}.npy is never broken by a missing or
    # unparseable .out file. Load all three together to plot:
    #   test_geometry_idx = np.load(".../{SYSTEM_TAG}_test_geometry_idx.npy")
    #   t1 = np.load(".../{SYSTEM_TAG}_test_t1_diagnostic.npy")
    #   e_ccsdt = np.load(".../{SYSTEM_TAG}_test_ccsdt_energy.npy")
    #   err = np.load(".../{SYSTEM_TAG}_test_errors_n{n}.npy")
    t1_npy_path = f'{output_dir}/{SYSTEM_TAG}_test_t1_diagnostic.npy'
    ccsdt_energy_npy_path = f'{output_dir}/{SYSTEM_TAG}_test_ccsdt_energy.npy'
    np.save(t1_npy_path, t1_values)
    np.save(ccsdt_energy_npy_path, ccsdt_energy_values)
    print(f"Saved: {t1_npy_path}\n  {ccsdt_energy_npy_path}")

    if n_found == 0:
        # Fall back: show what IS near "diagnostic" in one actual .out file,
        # so the regex above can be corrected against ORCA's real format
        # rather than guessed at blindly.
        print("\nNo T1 diagnostic matched in any test-geometry .out file -- "
              "T1_PATTERN likely doesn't match this ORCA version's format. "
              "Lines containing 'diagnostic' in the first available file:")
        for g in test_geometry_idx:
            out_path = f'{ccsdt_dir}/{SYSTEM_TAG}_{int(g)}_{suffix}.out'
            if os.path.isfile(out_path):
                with open(out_path) as f:
                    for line in f:
                        if 'diagnostic' in line.lower():
                            print(f"  {line.rstrip()}")
                break
        return

    valid = ~np.isnan(t1_values)
    t1_valid = t1_values[valid]
    abs_error_valid = abs_error[valid]
    signed_error_valid = signed_error[valid]

    n_above_thres = int(np.sum(t1_valid > 0.02))
    print(f"T1 diagnostic: mean={np.mean(t1_valid):.4f}, max={np.max(t1_valid):.4f}, "
          f"{n_above_thres} / {n_found} above the conventional 0.02 multi-reference threshold")

    r_abs = pearson(t1_valid, abs_error_valid)
    r_signed = pearson(t1_valid, signed_error_valid)

    print("\n=========================")
    print("CORRELATION SUMMARY")
    print("=========================")
    print(f"Pearson r(T1 diagnostic, |test error|)    = {r_abs:+.3f}")
    print(f"Pearson r(T1 diagnostic, signed test error) = {r_signed:+.3f}  (directional bias check)")

    if n_above_thres > 0 and r_abs > 0.3:
        print(f"\n=> {n_above_thres} test geometries exceed the multi-reference threshold, AND "
              "T1 diagnostic correlates with test error -- CCSD(T) itself is plausibly "
              "unreliable on exactly the geometries the model struggles with. Part of the "
              "remaining error may be noise in the ground truth, not an ML problem -- "
              "no feature/kernel choice can fix that.")
    elif n_above_thres > 0:
        print(f"\n=> {n_above_thres} test geometries exceed the multi-reference threshold, but "
              "T1 diagnostic does NOT clearly correlate with test error -- multi-reference "
              "character is present in the dataset but isn't obviously what's driving the "
              "remaining error.")
    else:
        print("\n=> No test geometries exceed the conventional multi-reference threshold -- "
              "CCSD(T) looks reliable throughout; the remaining error is more likely an ML/"
              "data-coverage issue, not a ground-truth problem.")

    print("\n--- Interactive prompt below (if run with python -i): test_geometry_idx, "
          "t1_values, signed_error, abs_error all in scope. ---")


if __name__ == '__main__':
    main()
