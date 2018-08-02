import numpy as np
from ase.io import read, write
import os
import copy


class Vibrations(object):
    """ Obtain vibrational spectra from atoms (in ASE format)."""
    def __init__(self, ase_calculator, atoms=None,
                 anharmonic_correction=True, working_directory='./',
                 only_normal_modes=True):
        """
        Calculate the (an)-harmonic vibrational frequencies, normal modes
        and the Born Effective Charges tensor using DFPT.

        Parameters
        ----------
        atoms : object (ASE format).
            Atoms object in ASE format.
        ase_calculator: object
            Calculator ase implmeneted in ASE. At the moment only the VASP
            calculator is supported.
        anharmonic_correction: boolean
            Flag to define whether to perform single-point calculations
            for each eigenvector. These are needed for correcting intensities
            for anharmonic effects.
        """

        #########################################################
        # 0) General setup.
        #########################################################

        # Init variables:
        self.atoms = atoms
        self.anharmonic_correction = anharmonic_correction
        self.ase_calc_opt = ase_calculator

        # Create environment for folders:
        self.wd = working_directory
        self.path_opt = '1_Optimization/'
        self.path_bec = '2_DFPT_BEC/'
        self.path_anh = '3_Anharmonic/'

        # Save vasp.py for other calculations.
        with open('./vasp.py', 'r') as f:
            self.vasp_py = f.read()
            f.close()

        if self.atoms is None:
            self.atoms = read(self.wd + 'POSCAR')
            print('You have not specified an atoms object. '
                  'Reading POSCAR from the current directory.....')

        msg = 'You need to provide an ASE calculator. ' \
              'At the moment only the VASP calculator is supported.'
        assert ase_calculator, msg

        # Create a backup of the initial structure.
        write('initial_backup.traj', self.atoms)

        ######################################################################
        # Step 1. Optimize atomic structure.
        ######################################################################

        if not os.path.exists(self.wd + self.path_opt):
            os.makedirs(self.wd + self.path_opt)
        os.chdir(self.wd + self.path_opt)
        write_vasp_py(self)
        if converged_outcar() is False:
                print('Starting structural optimization.....')
                self.atoms.get_potential_energy()
        print('The atoms structure is optimized.')

        os.chdir('../')

        ######################################################################
        # Step 2. Calculate BEC and Hessian using DFPT.
        ######################################################################

        ase_calc_bec = copy.deepcopy(self.ase_calc_opt)

        # Change some calculator flags for obtaining the BEC tensor (DFPT).
        ase_calc_bec.__dict__['int_params']['ibrion'] = 7
        ase_calc_bec.__dict__['int_params']['nsw'] = 1
        ase_calc_bec.__dict__['int_params']['nwrite'] = 3
        ase_calc_bec.__dict__['bool_params']['lreal'] = False
        ase_calc_bec.__dict__['bool_params']['lepsilon'] = True

        self.atoms.set_calculator(copy.deepcopy(ase_calc_bec))

        # Run BEC:
        if not os.path.exists(self.path_bec):
            os.makedirs(self.path_bec)
        os.chdir(self.path_bec)
        write_vasp_py(self)
        if converged_outcar() is False:
                print('Calculating Born Effective Charges (BEC)..........')
                self.atoms.get_potential_energy()
        print('DFPT calculation completed.')
        os.chdir('../')

        ######################################################################
        # Step 2.2. Get eigenvalues, eigenvectors and BEC tensor.
        ######################################################################

        n_atoms = len(self.atoms)

        with open('./' + self.path_bec + 'OUTCAR', 'r') as f:
            f.seek(0)
            alllines = f.readlines()
            f.close()

        # Get the Born charges:
        for i, line in enumerate(alllines):
            if 'BORN EFFECTIVE CHARGES' in line:
                break

        born_matrices = []
        i += 2
        for j in range(n_atoms):
            born_i = []
            i += 1
            for k in range(3):
                line = alllines[i]
                fields = line.split()
                born_i.append([float(x) for x in fields[1:4]])
                i += 1
            born_matrices.append(born_i)

        for i, line in enumerate(alllines):
            if 'Eigenvectors after division by SQRT(mass)' in line:
                break
        for k, line in enumerate(alllines[i+5:]):
            if 'MACROSCOPIC' in line:
                break

        eigenlines = alllines[i:i+k]

        # Get eigenvalues:
        e_val_thz = []
        e_val_ev = []
        e_val_cm = []

        for i in range(0, len(eigenlines)):
            if 'meV' in eigenlines[i]:
                # eigenvalues in eV:
                e_val_ev.append(float(eigenlines[i].split()[-2]) * 1e-3)
                # eigenvalues in cm-1:
                e_val_cm.append(float(eigenlines[i].split()[-4]))
                # eigenvalues in Thz:
                e_val_thz.append(float(eigenlines[i].split()[-8]))

        # Get eigenvectors:
        e_vect = []
        i = 0
        for lines in eigenlines:
            i += 1
            dxdydz_i = []
            if 'dx' in lines and 'dy' in lines and 'dz' in lines:
                lines_mode_i = eigenlines[i:i+n_atoms]

                for j in lines_mode_i:
                    line = j.split()
                    dxdydz_i.append([float(x) for x in line[-3:]])
                e_vect.append(dxdydz_i)

        # Store matrices:
        self.bec_tensor = np.array(born_matrices)
        self.eigenvectors = np.array(e_vect)
        self.eigenvalues_cm = np.array(e_val_cm)
        self.eigenvalues_ev = np.array(e_val_ev)
        self.eigenvalues_thz = np.array(e_val_thz)

        # Remove rotation and translation modes:
        if only_normal_modes:
            print('Only printing vibrational frequencies and normal modes ('
                  '3N-6):')
            self.eigenvectors = self.eigenvectors[:(3 * n_atoms - 6)]
            self.eigenvalues_cm = self.eigenvalues_cm[:(3 * n_atoms - 6)]
            self.eigenvalues_ev = self.eigenvalues_ev[:(3 * n_atoms - 6)]
            self.eigenvalues_thz = self.eigenvalues_thz[:(3 * n_atoms - 6)]
        print('Born Effective Charges (BEC) tensor:')
        print(self.bec_tensor)
        print('Eigenvectors (after division by sqrt(mass):')
        print(self.eigenvectors)
        print('Eigenvalues (in cm-1):')
        print(self.eigenvalues_cm.reshape(len(self.eigenvalues_cm), -1))

        view_modes(self)

        ######################################################################
        # Step 3. Build and run single-point calcs for anharmonic corrections.
        ######################################################################

        if self.anharmonic_correction is True:

            # Default values:
            step_int = np.linspace(-2.0, 2.0, 11.0, endpoint=True)
            step_int = np.round(step_int, 1)
            hbar = 6.35078e12
            struc_initial = read('./' + self.path_opt + 'OUTCAR')
            n_calcs = 0
            for mode in range(0, len(self.eigenvalues_cm)):
                for step in step_int:
                    n_calcs += 1
                    # Check whether this single-point has already been calc:
                    mode_i = str(int(self.eigenvalues_cm[mode]))
                    dir_mode = './' + self.path_anh + 'mode_' + mode_i + \
                               '_cm-1/'
                    dir_step = dir_mode + 'step_' + str(step) +'/'
                    if not os.path.exists(dir_step):
                        os.makedirs(dir_step)
                    if not converged_outcar(dir_step):
                        print(dir_step)
                        exit()
                        struc_i = copy.deepcopy(struc_initial)
                        os.chdir(dir_step)
                        write_vasp_py(self)
                        pos_initial = struc_i.get_positions()
                        energy_sec_mode = self.eigenvalues_thz[mode] * 1e12
                        lfactor = (hbar / (2 * np.pi * energy_sec_mode))**0.5
                        normalised_step = lfactor * step
                        delta_pos = np.reshape(self.eigenvectors[mode], (-1,
                                               3)) * normalised_step
                        new_pos = pos_initial + delta_pos
                        struc_i.positions = new_pos
                        ase_calc_anh_i = copy.deepcopy(self.ase_calc_opt)
                        ase_calc_anh_i.__dict__['int_params']['nsw'] = 0
                        struc_i.set_calculator(ase_calc_anh_i)
                        print('Single-point calculations, vibrational mode at '
                              '' + mode_i + ' cm-1 and step ' + str(step) +
                              '.')
                        print('Calculation ' + str(n_calcs) + '/' + str(
                              len(self.eigenvalues_cm) * len(step_int)) + '.')
                        if step != 0.0:
                            struc_i.get_potential_energy()
                        os.chdir('../../../')
                print('Single-point calculations for the mode at ' +
                      mode_i + ' ''cm-1 converged.')
            print('Displacements along each of the vibrational modes '
                  'completed.')

##########################################################################
# Function to visualize the modes:
##########################################################################

def view_modes(self, step_size=4.0, n_images=20):
    step_int = np.linspace(-step_size, step_size, n_images, endpoint=False)
    hbar = 6.35078e12
    struc_initial = read('./' + self.path_opt + 'OUTCAR')

    dir_results_vis = './Results/view_modes/'
    if not os.path.exists(dir_results_vis):
                os.makedirs(dir_results_vis)

    for mode in range(0, len(self.eigenvalues_cm)):
        traj_vib_mode = []
        for step in step_int:
            # Check whether this single-point has already been calc:
            mode_i = str(int(self.eigenvalues_cm[mode]))
            struc_i = copy.deepcopy(struc_initial)
            pos_initial = struc_i.get_positions()
            energy_sec_mode = self.eigenvalues_thz[mode] * 1e12
            lfactor = (hbar / ( 2 * np.pi * energy_sec_mode))**0.5
            normalised_step = lfactor * step
            delta_pos = np.reshape(self.eigenvectors[mode], (-1,
                                   3)) * normalised_step
            new_pos = pos_initial + delta_pos
            struc_i.positions = new_pos
            traj_vib_mode.append(struc_i)
        write(dir_results_vis + 'vib_mode_'+ mode_i + '.traj', traj_vib_mode)

##########################################################################
# Utils for VASP:
##########################################################################

# Check convergence in the OUTCAR file.
def converged_outcar(outcar_directory='./'):
    if os.path.exists(outcar_directory+'OUTCAR'):
        with open(outcar_directory+'OUTCAR', 'r') as f:
            alltext = f.read()
            f.seek(0)
            f.close()
        if 'Voluntary' in alltext:
            return True
    return False


def write_vasp_py(self):
    f = open('./vasp.py', "w+")
    for i in self.vasp_py:
        f.write(i)
    f.close()