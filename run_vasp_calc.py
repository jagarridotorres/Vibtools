#| - Import Modules
from ase.io import read, write
import copy
import numpy as np
from ase.calculators.vasp import Vasp
import os
import shutil

#__|


def converged_outcar():
    if os.path.exists('./OUTCAR'):
        with open('./OUTCAR', 'r') as f:
            alltext = f.read()
            f.seek(0)
            alllines = f.readlines()
            f.close()
        if 'Voluntary' in alltext:
            return True
    return False


# Read initial structure.
atoms = read('POSCAR')

# Setup calculator for optimization.
ase_calc_opt = Vasp(istart=0,        # Start from scratch.
                gga='PE',        # Method.
                kpts  = (1,1,1), # k-points.
                gamma = True,    # Gamma-centered (defaults to Monkhorst-Pack)
                encut=400,       # Cutoff.
                ismear=1,        # Smearing
                sigma = 0.1,     # Smearing
                ediffg=-0.015,   # Convergence criteria.
                ediff=1e-6,      # Convergence criteria.
                nsw=250,         # Number of optimization steps.
                nelmin=10,       # Min. electronic steps.
                nelm=250,        # Max. electronic steps.
                prec='Accurate', # Precision.
                ibrion=2,        # Optimization algorithm.
                algo = 'Fast',   # Optimization algorithm.
                ispin=1,         # Spin-polarization.
                #npar=12,         # Parallelization.
                #ncore=4,         # Parallelization.
                lwave=False,     # Output.
                lcharg=False,    # Output.
                nfree=2,         # Degrees of freedom.
                isym=False,      # Remove symmetry.
                lreal=True       # Reciprocal space.
                )
####### 1) Geometry optimization ###############

# Create a backup of the initial structure.
write('initial_backup.traj', atoms)

# Attach calculator.
atoms.set_calculator(copy.deepcopy(ase_calc_opt))

# Optimize structure.
path_opt = './Step1-Optimisation'
if not os.path.exists(path_opt):
    os.makedirs(path_opt)
shutil.copy('./vasp.py', path_opt)
os.chdir(path_opt)
if converged_outcar() is False:
        print('Starting structural optimization..........')
        atoms.get_potential_energy()
print('The atoms structure is optimized.')

####### 2) Calculate BEC ###############

atoms = read('CONTCAR')
os.chdir('../')
ase_calc_bec = copy.deepcopy(ase_calc_opt)

# Change some flags for BEC.
ase_calc_bec.__dict__['int_params']['ibrion'] = 7
ase_calc_bec.__dict__['int_params']['nsw'] = 1
ase_calc_bec.__dict__['int_params']['nwrite'] = 3
ase_calc_bec.__dict__['bool_params']['lreal'] = False
ase_calc_bec.__dict__['bool_params']['lepsilon'] = True

atoms.set_calculator(copy.deepcopy(ase_calc_bec))

# Run BEC:
path_bec = './Step2-SecondDerivative/'
if not os.path.exists(path_bec):
    os.makedirs(path_bec)
shutil.copy('./vasp.py', path_bec)
os.chdir(path_bec)
if converged_outcar() is False:
        print('Calculating Born Effective Charges (BEC)..........')
        atoms.get_potential_energy()
print('BEC are calculated.')