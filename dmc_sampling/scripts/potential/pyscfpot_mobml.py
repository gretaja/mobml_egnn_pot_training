import sys
import os
sys.path.insert(0, '/home/pipeline_config')

import numpy as np
from pyscf import gto
from mobml_pyscf import get_features
from GPR import GP

from system_config import SYSTEM_TAG, ATOMS, CHARGE, SPIN

n_training = 1500 #UPDATE HERE -- which learning-curve point's model to load

# Resolved relative to this file's own directory (not the caller's working
# directory) so this module loads the model correctly regardless of which
# script imports it -- e.g. dmc_sampling's DMC drivers (cwd = dmc_sampling)
# vs. pca_and_fps/make_egnn_training_set.py (cwd = pca_and_fps), which both
# import this same module. Expects the trained model file (from
# mobml_fitting/fit_mobml.py) copied alongside this script.
_here = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(_here, f"mobml_{SYSTEM_TAG}_ccsdt_tz_{n_training}_matern_linear.npy")
model = np.load(model_path, allow_pickle=True).item()
feature_selection = model['feature_sele']
gp = GP.from_dict(model['model'])
add_energy = model['add_energy']

def scf_pot(cds):
    ans = []
    for i in range(cds.shape[0]):
        atom = [[a, tuple(cds[i, j])] for j, a in enumerate(ATOMS)]
        mol = gto.M(atom=atom, basis='aug-cc-pVDZ', charge=CHARGE, spin=SPIN,
            unit='B', verbose=0)
        hf, features = get_features(mol, verbose=False)
        energy_hf = hf.energy_tot()
        energy_pred = gp.predict(features[None, :, feature_selection])[0, 0] / 1000
        energy_tot = energy_hf + energy_pred + add_energy

        ans.append(energy_tot)
    return np.array(ans)
