import sys
sys.path.insert(0, '/home/pipeline_config')

import numpy as np
from pyscf import gto, dft

from system_config import ATOMS, CHARGE, SPIN

def scf_pot(cds):
    ans = []
    for i in range(cds.shape[0]):
        atom = [[a, tuple(cds[i, j])] for j, a in enumerate(ATOMS)]
        mol = gto.Mole()
        mol.atom = atom
        mol.basis = '6-31+g(d,p)'
        mol.charge = CHARGE
        mol.spin = SPIN
        mol.unit = 'Bohr'
        mol.verbose = 0
        mol.build()

        mf = dft.UKS(mol).density_fit()
        mf.verbose = 0

        # wB97X-D functional
        mf.xc = 'wb97x-d'

        # Faster numerical integration grid
        mf.grids.level = 1

        # Optional SCF settings
        mf.conv_tol = 1e-6
        mf.max_cycle = 100

        energy = mf.kernel()
        ans.append(energy)

    return np.array(ans)
