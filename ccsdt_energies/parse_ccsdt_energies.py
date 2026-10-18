"""
Parses the energy out of the ORCA .out files produced by ccsdt_tz_orca.sh
(CCSD(T)) or b2lyp_d3_tz_orca.sh (B2PLYP-D3) -- same ENERGY_PATTERN works
for both, since ORCA prints the same "FINAL SINGLE POINT ENERGY" line
either way.

ccsdt_tz_orca.sh/b2lyp_d3_tz_orca.sh used to do this themselves at the end
of every job, but their START_IDX/END_IDX option lets you split a system's
geometries across several parallel submissions -- parsing only makes
sense once ALL of those batches have finished, not after each one. Run
this once they have:

    python3 parse_ccsdt_energies.py          # CCSD(T) (default)
    python3 parse_ccsdt_energies.py b2plyp   # B2PLYP-D3 instead -- matches
                                              # fit_mobml.py's ENERGY_METHOD

Regardless of how many ccsdt_tz_orca.sh/b2lyp_d3_tz_orca.sh submissions it
took to produce them, every batch writes into the same output_files_dir
({system_name}_dlpno_ccsdt_augccpvtz_output/ or
{system_name}_b2lyp_d3_augccpvtz_output/), so this just needs to run once
over that whole directory.
"""

import sys
import os
import glob
import re

import numpy as np

# Real host path -- this runs bare-metal, same as ccsdt_tz_orca.sh (no
# apptainer container here to provide /home/pipeline_config).
sys.path.insert(0, '/gscratch/mccoy/pot_training/pipeline_config')
from system_config import SYSTEM_TAG

ENERGY_PATTERN = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def _natural_key(path):
    """Sort '..._2.out' before '..._10.out' (plain string sort would put '..._10.out' first)."""
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", os.path.basename(path))]


def _geometry_index(path, suffix):
    """
    Extract the geometry index from a '{system_name}_{index}_{suffix}.out'
    filename (matching ccsdt_tz_orca.sh's naming). Strips the known
    "_{suffix}.out" tail, then takes the last '_'-delimited token of what's
    left -- works regardless of what characters appear in system_name
    (e.g. "c2h6-2p"), since we never need to parse system_name itself out.
    """
    stem = os.path.basename(path)[: -len(f"_{suffix}.out")]
    return int(stem.rsplit("_", 1)[-1])


def parse_ccsdt_energies(output_dir, suffix, npy_path=None, indices_npy_path=None):
    """
    Parse the FINAL SINGLE POINT ENERGY out of every '*_{suffix}.out' file
    in output_dir, in natural/numeric filename order, and optionally save
    the result to npy_path.

    output_dir: directory holding the ORCA .out files -- what
        ccsdt_tz_orca.sh calls output_files_dir.
    suffix: filename_suffix from ccsdt_tz_orca.sh, e.g.
        "{system_name}_dlpno_ccsdt_augccpvtz".
    npy_path: if given, energies are also saved here via np.save.
    indices_npy_path: if given, the geometry index each energy came from
        (parsed from its filename) is saved here -- needed downstream
        (see fit_mobml.py) since mob_features/hf_energies are computed for
        every geometry regardless of whether its CCSD(T) job succeeded, so
        anything that skips a geometry here needs this to realign them by
        geometry rather than by raw position.

    Returns: (energies, indices, out_files) -- energies as a 1D numpy
        array, the geometry index each one corresponds to (same order,
        same length), and the full sorted list of .out files found (which
        may be longer than the other two, if some had no extractable
        energy) -- so a caller can also see which geometries were skipped.
    """
    out_files = sorted(glob.glob(os.path.join(output_dir, f"*_{suffix}.out")), key=_natural_key)

    energies = []
    indices = []
    n_permission_denied = 0
    for path in out_files:
        try:
            with open(path) as f:
                text = f.read()
        except PermissionError:
            # Shared output_dir, some .out files owned by other group members
            # with permissions that don't allow us to read them -- skip and
            # keep going rather than letting the whole parse fail on one
            # file. Same effect downstream as any other skipped geometry:
            # it just won't appear in indices, so fit_mobml.py's existing
            # valid_idx-based alignment handles the gap the same way it
            # already does for a geometry with no usable CCSD(T) energy.
            print(f"Warning: permission denied reading {path} -- skipping", file=sys.stderr)
            n_permission_denied += 1
            continue
        matches = ENERGY_PATTERN.findall(text)
        if not matches:
            print(f"Warning: no final energy found in {path}", file=sys.stderr)
            continue
        energies.append(float(matches[-1]))
        indices.append(_geometry_index(path, suffix))

    if n_permission_denied:
        print(f"Skipped {n_permission_denied} of {len(out_files)} output files due to permission errors "
              "(see warnings above for which) -- ask whoever owns them to chmod/share if you need those "
              "geometries included.", file=sys.stderr)

    energies = np.array(energies)
    indices = np.array(indices)

    if npy_path is not None:
        np.save(npy_path, energies)
        print(f"Saved {len(energies)} energies from {len(out_files)} output files to {npy_path}")

    if indices_npy_path is not None:
        np.save(indices_npy_path, indices)
        print(f"Saved {len(indices)} surviving geometry indices to {indices_npy_path}")

    return energies, indices, out_files


if __name__ == '__main__':
    # Resolved relative to this file's own directory, not the caller's
    # working directory, so it works regardless of where it's invoked from.
    here = os.path.dirname(os.path.abspath(__file__))

    # Defaults to CCSD(T) (this script's only behavior before this) -- pass
    # "b2plyp" as the one command-line argument to parse the B2PLYP-D3
    # output instead (see b2lyp_d3_tz_orca.sh/.inp), matching fit_mobml.py's
    # ENERGY_METHOD setting: python3 parse_ccsdt_energies.py b2plyp
    _ENERGY_METHOD_DIR_TAGS = {"ccsdt": "dlpno_ccsdt_augccpvtz", "b2plyp": "b2lyp_d3_augccpvtz"}
    energy_method = sys.argv[1] if len(sys.argv) > 1 else "ccsdt"
    if energy_method not in _ENERGY_METHOD_DIR_TAGS:
        print(f"Error: unknown energy method {energy_method!r} -- must be one of {list(_ENERGY_METHOD_DIR_TAGS)}",
              file=sys.stderr)
        sys.exit(1)

    suffix = f"{SYSTEM_TAG}_{_ENERGY_METHOD_DIR_TAGS[energy_method]}"
    output_dir = os.path.join(here, f"{suffix}_output")
    npy_path = os.path.join(output_dir, f"{suffix}_energies.npy")
    indices_npy_path = os.path.join(output_dir, f"{suffix}_valid_indices.npy")

    if not os.path.isdir(output_dir):
        print(f"Error: output directory does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    parse_ccsdt_energies(output_dir, suffix, npy_path, indices_npy_path)
