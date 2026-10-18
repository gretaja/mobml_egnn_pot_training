import sys
import numpy as np

# This script runs bare-metal (see xyz_to_npy.sh -- no apptainer), so
# system_config is reached via its real host path rather than the
# /home/pipeline_config bind the containerized steps use.
sys.path.insert(0, '/gscratch/mccoy/pot_training/pipeline_config')
from system_config import SYSTEM_TAG

def read_xyz_file(filename):
    """
    Reads in a .xyz file (in Angstrom) and returns an Nx3 numpy array of
    the coordinates in Bohr (atomic units).
    """
    au_to_ang = 0.529177249

    # Read the XYZ file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Number of atoms from the first line
    num_atoms = int(lines[0].strip())
    
    # Extracting the Cartesian coordinates
    coordinates = []
    for line in lines[2:2 + num_atoms]:
        _, x, y, z = line.split()
        coordinates.append([float(x), float(y), float(z)])
    
    # Convert to an Nx3 numpy array
    coordinates_array = np.array(coordinates)/au_to_ang
    return coordinates_array

system_name = SYSTEM_TAG
xyz_file = f'./{system_name}_opt.xyz' #UPDATE HERE if your source .xyz filename doesn't follow this pattern

cds_array = read_xyz_file(xyz_file)
np.save(f'{system_name}_opt.npy',cds_array)
