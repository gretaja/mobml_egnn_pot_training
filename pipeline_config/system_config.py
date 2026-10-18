"""
Shared per-system configuration for the pot_training workflow.

Every step of the workflow -- DMC sampling (dmc_sampling/), training-set
selection (pca_and_fps/), HF + MOB feature extraction (hf_mob_features/),
MOB-ML fitting (mobml_fitting/), EGNN training (egnn_training/), and
geometry conversion (opt_geoms/) -- needs the same handful of facts about
which system it's running for. Every script that does

    sys.path.insert(0, '/home/pipeline_config')   # or the real host path,
                                                   # for scripts that run
                                                   # outside the container
    from system_config import SYSTEM_TAG, ATOMS, CHARGE, SPIN, MULTIPLICITY, \
        HEAVY_ONLY_PCA, HEAVY_ONLY_MASS_SCALING, MOBML_N_TRAINING, \
        MOBML_TRAINING_SET_SIZE, MOBML_TEST_SIZE

picks up the values below -- nothing else needs to change.

To work on MULTIPLE systems at once: this module doesn't hold the values
itself. It reads the PIPELINE_SYSTEM environment variable and loads
systems/<PIPELINE_SYSTEM>.py, so each system's values live in their own
file under systems/. Each .sh sets which system it's running by adding,
before the srun/python line:

    export PIPELINE_SYSTEM=h2o

(matching a filename in systems/, e.g. systems/h2o.py) -- apptainer passes
the calling shell's environment through by default (the same mechanism
already used for FI_LOG_LEVEL, OMP_NUM_THREADS, etc.), so no extra --env
flag is needed for the containerized steps. Scripts that run bare-metal
(egnn_training/train_egnn.sh, opt_geoms/xyz_to_npy.sh) just need the
export before their `python` line the same way.

To set up a new system: add systems/<tag>.py (copy an existing one) defining
ATOMS, CHARGE, MULTIPLICITY, HEAVY_ONLY_PCA, HEAVY_ONLY_MASS_SCALING,
MOBML_N_TRAINING, MOBML_TRAINING_SET_SIZE, MOBML_TEST_SIZE -- SYSTEM_TAG is
taken from PIPELINE_SYSTEM itself, not repeated inside the file, so it can
never drift out of sync with the filename/env var used to select it.

Not included here: things that vary by *workflow step* rather than by
*system* (basis set, EGNN architecture hyperparameters, learning-curve
training-set sizes, ...) -- those stay local to the script that uses them.
"""

import os
import importlib.util

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SYSTEMS_DIR = os.path.join(_THIS_DIR, 'systems')

SYSTEM_TAG = os.environ.get('PIPELINE_SYSTEM')
if not SYSTEM_TAG:
    raise RuntimeError(
        "PIPELINE_SYSTEM is not set. Add e.g. `export PIPELINE_SYSTEM=h2o` "
        f"to your .sh before the srun/python line, matching a filename in "
        f"{_SYSTEMS_DIR}."
    )

_system_file = os.path.join(_SYSTEMS_DIR, f"{SYSTEM_TAG}.py")
if not os.path.isfile(_system_file):
    raise FileNotFoundError(
        f"No config for PIPELINE_SYSTEM={SYSTEM_TAG!r} -- expected {_system_file} to exist."
    )

_spec = importlib.util.spec_from_file_location(f"pipeline_config.systems.{SYSTEM_TAG}", _system_file)
_system_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_system_module)

ATOMS = _system_module.ATOMS
CHARGE = _system_module.CHARGE
MULTIPLICITY = _system_module.MULTIPLICITY
HEAVY_ONLY_PCA = _system_module.HEAVY_ONLY_PCA
HEAVY_ONLY_MASS_SCALING = _system_module.HEAVY_ONLY_MASS_SCALING
MOBML_N_TRAINING = _system_module.MOBML_N_TRAINING
MOBML_TRAINING_SET_SIZE = _system_module.MOBML_TRAINING_SET_SIZE
MOBML_TEST_SIZE = _system_module.MOBML_TEST_SIZE

# PySCF's `mol.spin` is 2S (number of unpaired electrons), i.e.
# MULTIPLICITY - 1 -- derived here so it can never disagree with
# MULTIPLICITY, and so PySCF-based scripts don't have to redo this
# conversion themselves.
SPIN = MULTIPLICITY - 1
