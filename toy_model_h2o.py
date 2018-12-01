from vibrations import Vibrations
from ase.calculators.vasp import Vasp
from ase.io import read

vasp_calc = Vasp(istart=0,        # Start from scratch.
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

vib = Vibrations(ase_calculator=vasp_calc,
                 anharmonic_correction=True)

vib.get_spectrum(limits=(300, 5000), spectra_mode='gas_phase',
                 anharmonic_corrected_spectra=True)
