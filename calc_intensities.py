from ase.io import read
import numpy as np
import matplotlib.pyplot as plt

wd = './examples/PBE-ethyl/'
opt_dir = 'Step1-Optimisation/'
bec_dir = 'Step2-SecondDerivative/'
anh_modes_dir = 'Step3-Anharmonicity-modes/'

# Introduce here the desired parameters:

x_axis_lim = (300, 3700)  # x axis limits for the spectrum.
fwhm = 60.0  # FWHM of the peaks.
resolution = 1.0  # Resolution of the spectrum in cm-1.

# Only for HREELS:
n_transfer = 1.0  # Set transfer function value.
angle = 89.95  # HREELS angle (0 degrees = perpendicular to the
# surface).

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

# Get the Born charges:
for i, line in enumerate(alllines):
    if 'BORN EFFECTIVE CHARGES' in line:
        break

born_matrices = []
i += 2  # skip a line
for j in range(n_atoms):
    born_i = []
    i += 1  # skips the ion count line
    for k in range(3):
        line = alllines[i]
        fields = line.split()
        born_i.append([float(x) for x in fields[1:4]])
        i += 1  # advance a line
    born_matrices.append(born_i)

# Get the Eigenvectors and Eigenvalues:
dof = n_atoms * 3 - len(atoms.constraints)  # Degrees of freedom.

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

eigenvalues_ev = []
eigenvalues_cm = []
for i in range(0, len(eigenlines)):
    if 'meV' in eigenlines[i]:
        eigenvalues_ev.append(float(eigenlines[i].split()[-2]) * 1e-3)  # eV
        eigenvalues_cm.append(float(eigenlines[i].split()[-4]))  # cm-1

eigenvalues_ev_normal_modes = []
eigenvalues_cm_normal_modes = []

for i in range(0, len(eigenlines_normal_modes)):
    if 'meV' in eigenlines_normal_modes[i]:
        eigenvalues_ev_normal_modes.append(float(eigenlines[i].split()[-2]) * 1e-3)
        eigenvalues_cm_normal_modes.append(float(eigenlines[i].split()[-4]))

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

Z = np.array(born_matrices)
e = np.array(eigenvectors)

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
norm = x_weight + y_weight + z_weight
z_weight /= norm
y_weight /= norm
x_weight /= norm
components = [x_weight, y_weight, z_weight]


# Get intensities (decomposed):
intensities = []
for mode in range(0, len(eigenvectors)):
    S = 0.0
    for alpha in [0, 1, 2]:
        s = 0
        for l in range(0, n_atoms):
            for beta in [0, 1, 2]:
                Zab =  Z[l][alpha][beta]
                e = eigenvectors[mode][l][beta]
                s +=  Zab * e
        S += components[alpha] * s**2
    intensities.append(S)

intensities_normal_modes = intensities[0:len(eigenvalues_cm_normal_modes)]
norm_intensities_normal_modes = intensities_normal_modes / np.max(intensities_normal_modes)

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

sigma = fwhm / (2 * np.sqrt(2* np.log(2)))

list_x0 = eigenvalues_cm_normal_modes
list_y0 = norm_intensities_normal_modes

# Transfer function:
list_y0_tranfer_function = norm_intensities_normal_modes /  \
                           np.array(eigenvalues_cm_normal_modes)**n_transfer
list_y0 = list_y0_tranfer_function


x = np.arange(x_axis_lim[0], x_axis_lim[1], resolution)
y_gauss = np.zeros(len(x))
for j in range(0, len(list_x0)):
    for i in range(0, len(x)):
        y_gauss[i] += gaus(x=x[i], a=list_y0[j], x0=list_x0[j], sigma=sigma)

plt.plot(x, y_gauss, color='navy')
plt.show()



