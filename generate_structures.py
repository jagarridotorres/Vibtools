#| - Import Modules
from ase.io import read, write
import copy
import numpy as np
from ase.calculators.vasp import Vasp
import os
import shutil
from ase.visualize import view

#__|

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



wd = './'
opt_dir = 'Step1-Optimisation/'
bec_dir = 'Step2-SecondDerivative/'
anh_modes_dir = 'Step3-Anharmonicity-modes/'


def converged_outcar(outcar_directory='./'):
    if os.path.exists(outcar_directory+'OUTCAR'):
        with open(outcar_directory+'OUTCAR', 'r') as f:
            alltext = f.read()
            f.seek(0)
            alllines = f.readlines()
            f.close()
        if 'Voluntary' in alltext:
            return True
    return False

atoms = read(wd + opt_dir + 'OUTCAR')
n_atoms = len(atoms)

with open(wd + bec_dir + 'OUTCAR', 'r') as f:
    alltext = f.read()
    f.seek(0)
    alllines = f.readlines()
    f.close()

if 'BORN' not in alltext:
    raise Exception('Born effective charges missing. '
                    'Did you use IBRION=7 or 8?')

if 'Eigenvectors after division by SQRT(mass)' not in alltext:
    raise Exception('You must rerun with NWRITE=3 to get '
                    'sqrt(mass) weighted eigenvectors')

# Get the Eigenvectors and Eigenvalues:

for i, line in enumerate(alllines):
    if 'Eigenvectors after division by SQRT(mass)' in line:
        break
for j, line in enumerate(alllines[i:]):
    if 'f/i=' in line:
        break
for k, line in enumerate(alllines[i+5:]):
    if 'MACROSCOPIC' in line:
        break

eigenlines_normal_modes = alllines[i:i+j]
eigenlines = alllines[i:i+k]

eigenvalues_thz = []
eigenvalues_ev = []
eigenvalues_cm = []
for i in range(0, len(eigenlines)):
    if 'meV' in eigenlines[i]:
        eigenvalues_ev.append(float(eigenlines[i].split()[-2]) * 1e-3)  # eV
        eigenvalues_cm.append(float(eigenlines[i].split()[-4]))  # cm-1
        eigenvalues_thz.append(float(eigenlines[i].split()[-8]))  # THz

eigenvalues_thz_normal_modes = []
eigenvalues_ev_normal_modes = []
eigenvalues_cm_normal_modes = []

for i in range(0, len(eigenlines_normal_modes)):
    if 'meV' in eigenlines_normal_modes[i]:
        eigenvalues_ev_normal_modes.append(float(eigenlines[i].split()[-2]) * 1e-3)
        eigenvalues_cm_normal_modes.append(float(eigenlines[i].split()[-4]))
        eigenvalues_thz_normal_modes.append(float(eigenlines[i].split()[-8]))

eigenvectors = []

i = 0
for lines in eigenlines:
    i += 1
    dxdydz_i = []
    if 'dx' in lines and 'dy' in lines and 'dz' in lines:
        lines_mode_i = eigenlines[i:i+n_atoms]

        for j in lines_mode_i:
            line = j.split()
            dxdydz_i.append([float(x) for x in line[-3:]])
        eigenvectors.append(dxdydz_i)
eigenvectors_normal_modes = eigenvectors[:len(eigenvalues_cm_normal_modes)]


################### FOR THIS SCRIPT ############################
int_delta_neg = -2.0
int_delta_pos = +2.0
n_singlepoints = 10.0  # Number of single points for each curve (mode).



### Visualization of the modes: #######

# step_int = np.linspace(-4.0, 4.0, 20.0, endpoint=False)
# hbar = 6.35078e12
# for mode in range(0, len(eigenvalues_cm)):
#     traj_vib_mode = []
#     for step in step_int:
#         # Check whether this single-point has already been calc:
#         mode_i = str(int(eigenvalues_cm[mode]))
#         dir_mode = wd + anh_modes_dir + 'mode_' + mode_i + '_cm-1/'
#         dir_step = dir_mode + 'step_' + str(step)
#         struc_initial = read(wd + opt_dir + 'OUTCAR')
#         pos_initial = struc_initial.get_positions()
#         energy_sec_mode = eigenvalues_thz[mode] * 1e12
#         lfactor = (hbar / ( 2 * np.pi * energy_sec_mode))**0.5
#         normalised_step = lfactor * step
#         delta_pos = np.reshape(eigenvectors[mode], (-1,3)) * normalised_step
#         new_pos = pos_initial + delta_pos
#         struc_initial.positions = new_pos
#         traj_vib_mode.append(struc_initial)
#     write('vib_mode_'+ mode_i + '.traj', traj_vib_mode)


##########################


step_int = np.linspace(-2.0, 2.0, 11.0, endpoint=True)
step_int = np.round(step_int, 1)
hbar = 6.35078e12

n_calcs = 0
for mode in range(0, len(eigenvalues_cm)):
    for step in step_int:
        n_calcs += 1
        # Check whether this single-point has already been calc:
        mode_i = str(int(eigenvalues_cm[mode]))
        dir_mode = wd + anh_modes_dir + 'mode_' + mode_i + '_cm-1/'
        dir_step = dir_mode + 'step_' + str(step)
        if not os.path.exists(dir_step):
            os.makedirs(dir_step)
        if not converged_outcar(dir_step):
            struc_initial = read(wd + opt_dir + 'OUTCAR')

            shutil.copy(wd + opt_dir + 'vasp.py', dir_step)
            os.chdir(dir_step)
            pos_initial = struc_initial.get_positions()
            energy_sec_mode = eigenvalues_thz[mode] * 1e12
            lfactor = (hbar / (2 * np.pi * energy_sec_mode))**0.5
            normalised_step = lfactor * step
            delta_pos = np.reshape(eigenvectors[mode], (-1,3)) * normalised_step
            new_pos = pos_initial + delta_pos
            struc_initial.positions = new_pos
            ase_calc_anh = copy.deepcopy(ase_calc_opt)
            ase_calc_anh.__dict__['int_params']['nsw'] = 0
            struc_initial.set_calculator(ase_calc_anh)
            print('Single-point calculation, vibrational mode at ' + mode_i
                   + ' cm-1 and step ' + str(step) + '.')
            print('Calculation ' + str(n_calcs) + '/' + str(
                  len(eigenvalues_cm) * len(step_int)) + '.')
            if step != 0.0:
                struc_initial.get_potential_energy()
            os.chdir('../../../')
        print('Single-point calculation for the mode at ' + mode_i + ' '
          'cm-1 converged.')














