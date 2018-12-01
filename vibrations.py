import numpy as np
from ase.io import read, write
import os
import copy
from scipy.optimize import curve_fit
from scipy import special
import shutil
import pickle

class Vibrations(object):
    """ Obtain vibrational spectra from atoms (in ASE format)."""
    def __init__(self, ase_calculator, atoms=None,
                 anharmonic_correction=True,
                 only_normal_modes=True):
        """
        Calculate the (an)-harmonic vibrational frequencies, normal modes
        and the Born Effective Charges tensor using DFPT.

        Parameters
        ----------
        atoms: object (ASE format).
            Atoms object in ASE format.
        ase_calculator: object
            Calculator ase implmeneted in ASE. At the moment only the VASP
            calculator is supported.
        anharmonic_correction: boolean
            Flag to define whether to perform single-point calculations
            for each eigenvector. These are needed for correcting intensities
            for anharmonic effects.
        working_directory: string
            Directory for the calculation.
        only_normal_modes: bool
            Whether to consider to compute only the normal modes or not.
        """

        #########################################################
        # 0) General setup.
        #########################################################

        # Init variables:
        self.atoms = atoms
        self.anharmonic_correction = anharmonic_correction
        self.ase_calc_opt = ase_calculator

        # Create environment for folders:
        self.wd = os.getcwd() + '/'
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
        assert self.ase_calc_opt, msg

        # Create a backup of the initial structure.
        write('initial_backup.traj', self.atoms)

        ######################################################################
        # Step 1. Optimize atomic structure.
        ######################################################################

        if not os.path.exists(self.wd + self.path_opt):
            os.makedirs(self.wd + self.path_opt)
        os.chdir(self.wd + self.path_opt)
        write_vasp_py(self)
        copy_vdw_kernel(self.wd, './')
        if converged_outcar() is False:
                print('Starting structural optimization.....')
                self.atoms.set_calculator(copy.deepcopy(self.ase_calc_opt))
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
        copy_vdw_kernel(self.wd, './')
        if converged_outcar() is False:
                print('Calculating Born Effective Charges (BEC)..........')
                self.atoms.set_calculator(copy.deepcopy(ase_calc_bec))
                self.atoms.get_potential_energy()
        print('DFPT calculation completed.')
        os.chdir('../')

        ######################################################################
        # Step 2.2. Get eigenvalues, eigenvectors and BEC tensor.
        ######################################################################

        self.n_atoms = len(self.atoms)

        with open('./' + self.path_bec + 'OUTCAR', 'r') as f:
            f.seek(0)
            alllines = f.readlines()
            f.close()

        # Get BEC tensor:
        for i, line in enumerate(alllines):
            if 'BORN EFFECTIVE CHARGES' in line:
                break

        born_matrices = []
        i += 2
        for j in range(self.n_atoms):
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
                # Eigenvalues in eV:
                e_val_ev.append(float(eigenlines[i].split()[-2]) * 1e-3)
                # Eigenvalues in cm-1:
                e_val_cm.append(float(eigenlines[i].split()[-4]))
                # Eigenvalues in Thz:
                e_val_thz.append(float(eigenlines[i].split()[-8]))

        # Get eigenvectors:
        e_vect = []
        i = 0
        for lines in eigenlines:
            i += 1
            dxdydz_i = []
            if 'dx' in lines and 'dy' in lines and 'dz' in lines:
                lines_mode_i = eigenlines[i:i+self.n_atoms]

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
            self.eigenvectors = self.eigenvectors[:(3 * self.n_atoms - 6)]
            self.eigenvalues_cm = self.eigenvalues_cm[:(3 * self.n_atoms - 6)]
            self.eigenvalues_ev = self.eigenvalues_ev[:(3 * self.n_atoms - 6)]
            self.eigenvalues_thz = self.eigenvalues_thz[:(3 * self.n_atoms -
             6)]
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
            step_int = np.linspace(-2.0, 2.0, 11.0, endpoint=True)
            step_int = np.round(step_int, 1)
            try:
                all_anh_pes = pickle.load(open("all_anh_pes.txt", "rb"))
            except:
                # Default values:
                hbar = 6.35078e12
                struc_initial = read('./' + self.path_opt + 'OUTCAR')
                n_calcs = 0
                all_anh_pes = []
                for mode in range(0, len(self.eigenvalues_cm)):
                    anh_pes = []
                    for step in step_int:
                        n_calcs += 1
                        # Check whether this single-point has already been calc:
                        mode_i = str(1 + mode)
                        dir_mode = './' + self.path_anh + 'mode' + mode_i + '/'
                        dir_step = dir_mode + 'step_' + str(step) +'/'

                        if not os.path.exists(dir_step):
                            os.makedirs(dir_step)
                        if not converged_outcar(dir_step):
                            struc_i = copy.deepcopy(struc_initial)
                            os.chdir(dir_step)
                            write_vasp_py(self)
                            copy_vdw_kernel(self.wd, './')
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
                                  '' + str(self.eigenvalues_cm[int(mode_i)-1]) + ' cm-1 and step ' + str(step) +
                                  '.')
                            print('Calculation ' + str(n_calcs) + '/' + str(
                                  len(self.eigenvalues_cm) * len(step_int)) + '.')
                            if step != 0.0:
                                struc_i.get_potential_energy()
                            os.chdir('../../../')
                        if step == 0.0:
                            energy_i = read(self.path_opt + 'OUTCAR').get_potential_energy()
                        os.chdir(dir_step)
                        # Get curves of the normal modes:
                        if step != 0.0:
                            energy_i = read('./OUTCAR').get_potential_energy()
                        anh_pes.append(energy_i)
                        os.chdir('../../../')

                    all_anh_pes.append(anh_pes)
                    print('Single-point calculations for the mode at ' +
                          str(self.eigenvalues_cm[int(mode_i)-1]) + ' ''cm-1 '
                          'converged.')
                print('Displacements along each of the vibrational modes '
                      'completed.')

                pickle.dump(all_anh_pes, open("all_anh_pes.txt", "wb"))

            #################################################################
            # Create individual PES for each mode:
            #################################################################

            # Create data arrays for the PES scans:
            normal_coordinates = []
            energy = []
            for i in range(0, len(all_anh_pes)):
                energy.append(all_anh_pes[i]-np.min(all_anh_pes[i]))
                normal_coordinates.append(step_int)

            # Flip curves:
            for i in range(0, len(energy)):
                if energy[i][0] < energy[i][-1]:
                    normal_coordinates[i] = -normal_coordinates[i]
                # # Plots:
                # plt.figure()
                # plt.plot(normal_coordinates[i],energy[i])
                # plt.show()

            #################################################################
            # Fit to Morse potential:
            #################################################################
            def morse(x, paramDe, paramA):
                return paramDe*(np.exp(-paramA*x)-1)**2

            # Obtain optimized parameters (Morse).
            paramDe_opt = []
            paramA_opt = []
            rsquared_opt = []
            f_corr = []
            mode = []

            for i in range(0, len(self.eigenvalues_cm)):

                mode.append(i+1)
                x = normal_coordinates[i]
                y = energy[i]

                hyperparameters = [1.0, 1.0]

                popt, pcov = curve_fit(morse, x, y, p0=hyperparameters,
                                       maxfev=200000,method='trf')

                ss_res = np.dot((y - morse(x, *popt)), (y - morse(x, *popt)))
                ymean = np.mean(y)
                ss_tot = np.dot((y - ymean), (y - ymean))
                rSquared = 1-ss_res/ss_tot
                rsquared_opt.append(rSquared)

                paramDe_opt.append(abs(popt[0]))
                paramA_opt.append(abs(popt[1]))

                Nparam = (((np.sqrt(2.0 * paramDe_opt[i])) / (paramA_opt[
                          i])) - (1.0/2.0))
                fCorr1 = 2.0 / (2.0 * Nparam - 1.0)
                fNum = Nparam * (Nparam - 1.0) * special.gamma(2.0 * Nparam)
                fDenom = (special.gamma(2.0 * Nparam + 1.0))
                # if np.isinf(fDenom):
                #     fDenom = 1e12
                # if np.isinf(fNum):
                #     fNum = 1e12
                fCorr2 = np.sqrt(fNum/fDenom)
                fCorrection = fCorr1 * fCorr2
                if np.isnan(fCorrection):
                    fCorrection = 1.0
                f_corr.append(fCorrection)

            anh_results = np.zeros((len(self.eigenvalues_cm), 4))
            print(["Vibrational mode (cm-1)", "De", "a", "f_corr"])
            anh_results[:, 0] = self.eigenvalues_cm
            anh_results[:, 1] = paramDe_opt
            anh_results[:, 2] = paramA_opt
            anh_results[:, 3] = f_corr
            print(anh_results)
            self.intensity_correction = np.array(f_corr)

    ##########################################################################
    # Function to plot spectra:
    ##########################################################################

    def get_spectrum(self, limits=(350, 3800), resolution=1.0, fwhm=60.0,
                     spectra_mode='surface', angle=90.0, n_transfer=0.0,
                     anharmonic_corrected_spectra=False,
                     normalized_spectrum=True):

        """
        Function to obtain the graphical representation of the spectra.

        Parameters
        ----------
        limits: tuple.
            Region for plotting the spectra (in cm-1).
        spectra_mode: string
            Type of spectrum to be calculated.
            Implemented are: 'gas_phase' and 'surface'.
        resolution: float
            Resolution of the convoluted spectrum (in cm-1).
        fwhm: float
            Full width at half maximum (in cm-1) of the peaks in the spectra.
        angle: float
            Angle of the collected variation of the polarization and the
            surface plane (only applicable when modelling spectra involving
            surfaces).
        """
        # Import matplotlib.
        import matplotlib.pyplot as plt

        # Check user input to be consistent.
        if anharmonic_corrected_spectra is True:
            print('Printing spectra including anharmonic corrections...')
            msg = 'Anharmonic corrections were not calculated before. '
            msg += 'If you want your spectrum to be corrected for ' \
                   'anharmonicities you must set the flag ' \
                   '"anharmonic_corrections" to True class ' \
                   'Vibrations'
            assert self.anharmonic_correction, msg

        Z = np.array(self.bec_tensor)
        e = np.array(self.eigenvectors)

        intensities = []

        # Decomposition for HREELS:

        # Angle resolved:
        theta = angle

        z_weight = np.cos((90. - theta) * np.pi / 180.)
        z_weight *= z_weight
        x_weight = 0.5 * (1 - z_weight)
        y_weight = x_weight
        z_weight = np.sqrt(z_weight)
        x_weight = np.sqrt(x_weight)
        y_weight = np.sqrt(y_weight)
        if spectra_mode == 'gas_phase':
            x_weight, y_weight, z_weight = 1.0, 1.0, 1.0
        norm = x_weight + y_weight + z_weight
        z_weight /= norm
        y_weight /= norm
        x_weight /= norm
        components = [x_weight, y_weight, z_weight]

        # Get intensities (decomposed):
        intensities = []
        for mode in range(0, len(self.eigenvectors)):
            S = 0.0
            for alpha in [0, 1, 2]:
                s = 0
                for l in range(0, self.n_atoms):
                    for beta in [0, 1, 2]:
                        Zab = Z[l][alpha][beta]
                        e = self.eigenvectors[mode][l][beta]
                        s += Zab * e
                S += components[alpha] * s**2
            intensities.append(S)

        if anharmonic_corrected_spectra is True:
            intensities = intensities * self.intensity_correction

        if normalized_spectrum is True:
            intensities = intensities / np.max(intensities)

        def gaus(x, a, x0, sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        list_x0 = self.eigenvalues_cm
        list_y0 = intensities

        # Transfer function:
        list_y0_tranfer_function = intensities / np.array(
                                   self.eigenvalues_cm)**n_transfer
        if n_transfer != 0.0:
            list_y0 = list_y0_tranfer_function

        x = np.arange(limits[0], limits[1], resolution)
        y_gauss = np.zeros(len(x))
        for j in range(0, len(list_x0)):
            for i in range(0, len(x)):
                y_gauss[i] += gaus(x=x[i], a=list_y0[j], x0=list_x0[j], sigma=sigma)

        plt.plot(x, y_gauss, color='navy')
        plt.show()

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
            lfactor = (hbar / (2 * np.pi * energy_sec_mode))**0.5
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


# Write vasp.py file (required to run VASP through ASE).
def write_vasp_py(self):
    f = open('./vasp.py', "w+")
    for i in self.vasp_py:
        f.write(i)
    f.close()


def copy_vdw_kernel(source_dir, target_dir):
    filename = 'vdw_kernel.bindat'
    if os.path.exists(source_dir + filename):
        shutil.copy(source_dir + filename, target_dir + filename)
