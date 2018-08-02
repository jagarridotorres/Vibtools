from scipy.optimize import curve_fit, minimize
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import os


os.remove('modes.txt')
os.system('grep "freedom DOF   =" ./RAW-VASP/step2/OUTCAR |  sed -n -e '
'"s/^.*\( = \)/\ '
'/p" > '
'modes.txt')

numberofmodes = 28

if os.path.exists('anhconstants'):
    os.remove('anhconstants')

data = []
normalcoord = []
energy = []

# Loop over all the PES Scan files
for i in range(1,numberofmodes+1):
    directory = './PES-Scans/'
    file = 'curve-anharm-mode' + str(i)
    path = directory + file
    data.append(np.genfromtxt(fname=path,delimiter=' '))

# Create data arrays for the PES Scans
for i in range(0,numberofmodes):
    normalcoord.append(data[i][:,0])
    energy.append(data[i][:,1]-np.min(data[i][:,1]))

# Flip curves
for i in range(0,numberofmodes):
    if energy[i][0] < energy[i][-1]:
        normalcoord[i] = -normalcoord[i]
    # plt.figure()
    # plt.plot(normalcoord[i],energy[i])
    # plt.show()

def morse(x, paramDe, paramA):
        return ((paramDe*(np.exp(-paramA*x)-1)**2))

def morse_fit(hyperparam, x, y):
    popt, pcov = curve_fit(morse, x, y, p0 = hyperparam,maxfev=2000000)
    # print("Mean Squared Error: ", np.mean((y-morse(x, *popt))**2))
    ss_res = np.dot((y - morse(x, *popt)),(y - morse(x, *popt)))
    ymean = np.mean(y)
    ss_tot = np.dot((y-ymean),(y-ymean))
    rSquared = 1-ss_res/ss_tot
    return rSquared

# Obtain optimised parameters (fit to Morse)

paramDe_opt = []
paramA_opt = []
rsquared_opt = []
f_corr = []
mode = []

for i in range(0,numberofmodes):

    mode.append(i+1)
    x = normalcoord[i]
    y = energy[i]
    args = (x,y)
    hyperparameters = [1.0, 1.0]
    bounds=((None,None),(None,None))

    hyper_opt = minimize(morse_fit,hyperparameters,args=args,method='TNC')
    popt, pcov = curve_fit(morse, x, y, p0 = hyperparameters,maxfev=2000000)
    ss_res = np.dot((y - morse(x, *popt)),(y - morse(x, *popt)))
    ymean = np.mean(y)
    ss_tot = np.dot((y-ymean),(y-ymean))
    rSquared = 1-ss_res/ss_tot

    rsquared_opt.append(rSquared)
    paramDe_opt.append(abs(popt[0]))
    paramA_opt.append(abs(popt[1]))
    Nparam = (((np.sqrt(2.0 * paramDe_opt[i])) / (paramA_opt[i])) - (1.0/2.0))
    fCorr1 = 2.0 / (2.0 * Nparam - 1.0)
    fNum = Nparam * (Nparam - 1.0) * special.gamma(2.0 * Nparam)
    fDenom = (special.gamma(2.0 * Nparam + 1.0))

    fCorr2 = np.sqrt(fNum/fDenom)
    fCorrection = fCorr1 * fCorr2
    if np.isnan(fCorrection):
        fCorrection = 1.0
    f_corr.append(fCorrection)

Results = np.zeros((numberofmodes,4))
Results[:,0] = mode
Results[:,1] = paramDe_opt
Results[:,2] = paramA_opt
Results[:,3] = f_corr

# Saving Data
np.savetxt('anhconstants', Results, fmt=('%d','%.5e','%.5e','%.5e'),
delimiter='  ')







