# pot_training

Pipeline for building machine-learned potential energy surfaces (MOB-ML and
EGNN) for small molecular systems, benchmarked against DLPNO-CCSD(T)
reference energies. Developed in the McCoy Group (UW) and run on the Hyak
cluster.

## Pipeline

Each stage is a directory with one or more Python scripts and a matching
`.sh` SLURM submission script. Data flows roughly in this order:

| Stage | Directory | What it does |
|---|---|---|
| 1 | [`opt_geoms/`](opt_geoms) | Converts a DFT/MP2-optimized starting geometry (`.xyz`, Angstrom) to `.npy` (Bohr) used to seed DMC sampling. |
| 2 | [`dmc_sampling/`](dmc_sampling) | Diffusion Monte Carlo sampling with [`pyvibdmc`](https://github.com/McCoyGroup/PyVibDMC), using a DFT or MOB-ML-corrected potential ([`dmc_sampling/scripts/potential/`](dmc_sampling/scripts/potential)). `broad_*` jobs sample broadly; `min_*` jobs sample near the minimum. |
| 3 | [`pca_and_fps/`](pca_and_fps) | PCA + farthest-point sampling over the DMC-sampled configurations to assemble training sets for MOB-ML (`make_mobml_training_set.py`) and EGNN (`make_egnn_training_set.py`). |
| 4 | [`ccsdt_energies/`](ccsdt_energies) | Runs DLPNO-CCSD(T)/aug-cc-pVTZ single-point energies in ORCA on the training-set geometries, then `parse_ccsdt_energies.py` parses the output logs into reference energies. |
| 5 | [`hf_mob_features/`](hf_mob_features) | Computes HF and MOB (molecular-orbital-based) features for the training-set geometries via PySCF (`get_mobml_features.py`). `compare_uhf_rohf.py` / `debug_rohf.py` are UHF-vs-ROHF diagnostics. |
| 6 | [`mobml_fitting/`](mobml_fitting) | Fits a Gaussian Process (custom `GPR` package) on the HF/MOB features against the CCSD(T) reference energies, producing a MOB-ML learning curve. `correlate_*_error.py` / `verify_alignment.py` are sanity-check/analysis scripts. |
| 7 | [`egnn_training/`](egnn_training) | Trains an E(n)-equivariant graph neural network (PyTorch, GPU) directly on geometries and energies as an alternative potential representation. |

[`pipeline_config/`](pipeline_config) holds configuration shared across every
stage above (see below).

## Per-system configuration

Every stage needs the same handful of facts about which molecular system
it's running for (atom ordering, charge, multiplicity, training-set sizes,
...). These live in `pipeline_config/system_config.py`, which reads the
`PIPELINE_SYSTEM` environment variable and loads `pipeline_config/systems/<tag>.py`
accordingly. Each `.sh` script sets this before running, e.g.:

```bash
export PIPELINE_SYSTEM=h2o
```

To add a new system, copy an existing file in `pipeline_config/systems/`
(e.g. `h2o.py`) and fill in its values; nothing else needs to change.

## Running

Scripts are written to run on the Hyak SLURM cluster from
`/gscratch/mccoy/pot_training`. Most MPI-heavy stages (DMC sampling,
feature extraction, training-set assembly) run inside an Apptainer
container (`current.sif`) with the relevant directories bind-mounted to
`/home/...` inside it; `opt_geoms/xyz_to_npy.py` and
`egnn_training/train_egnn.py` run bare-metal instead. See each stage's
`.sh` file for the exact `sbatch`/`apptainer` invocation.

## What's tracked in git

This repo tracks code and configuration only. Generated data and logs
(`.npy`, `.out`, `.log`, `slurm-*.out`, `.joblib`, `__pycache__/`) are
gitignored since they're reproducible by re-running the pipeline — with
one exception: the optimized starting geometries under `opt_geoms/*.xyz`
are tracked, since they come from external QM geometry optimizations and
aren't reproducible by anything in this repo.
