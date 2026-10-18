import numpy as np
import sys
pyvib = '/home/packages'
sys.path.insert(0, pyvib)
sys.path.insert(0, '/home/pipeline_config')
import pyvibdmc as pv
from pyvibdmc import potential_manager as pm
from pyvibdmc.simulation_utilities.mpi_imp_samp_manager import MPI_Potential, MPI_ImpSampManager
from system_config import SYSTEM_TAG, ATOMS

if __name__ == '__main__': #if using multiprocessing on windows / mac, you need to encapsulate using this line
    #this directory is part of the one you copied that is outside of pyvibdmc.
    pot_dir = '/home/scripts/potential/'
    py_file = 'pyscfpot_dft.py'
    pot_func = 'scf_pot'

    pot = MPI_Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir)

    # Starting Structure
    # Can also feed in an entire ensemble of walkers.
    # NOTE: /home/opt_geoms needs to be bound in the .sh for this to resolve.
    opt_geom = np.load(f'/home/opt_geoms/{SYSTEM_TAG}_opt.npy')

    start_coord = np.expand_dims(opt_geom,axis=0)
    sim_num = 0
    myDMC = pv.DMC_Sim(sim_name=f"{SYSTEM_TAG}_{sim_num}",
                            output_folder=f"/home/output/{SYSTEM_TAG}/dft_sims/100w_1000ts_1dt",
                            weighting='discrete', #or 'continuous'. 'continuous' keeps the ensemble size constant.
                            num_walkers=100, #number of geometries exploring the potential surface
                            num_timesteps=1000, #how long the simulation will go. (num_timesteps * delta_t atomic units of time)
                            equil_steps=0, #how long before we start collecting wave functions
                            chkpt_every=5, #checkpoint the simulation every "chkpt_every" time steps
                            wfn_every=999, #collect a wave function every "wfn_every" time steps
                            desc_wt_steps=250, #number of time steps you allow for descendant weighting per wave function
                            atoms=ATOMS,
                            delta_t=1, #the size of the time step in atomic units
                            potential=pot,
                            start_structures=start_coord, #can provide a single geometry, or an ensemble of geometries
                            DEBUG_save_training_every=1, #save coordinates and energies for training data
                            masses=None #can put in artificial masses, otherwise it auto-pulls values from the atoms string
    )
    myDMC.run()
