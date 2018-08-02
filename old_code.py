from scipy.optimize import curve_fit
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv

numberofModes = 3

if os.path.exists('anhconstants'):
    os.remove('anhconstants')


for mode in range(1,numberofModes+1):

    data = np.genfromtxt(fname='./PES-Scans/curve-anharm-mode' + str(mode),
    delimiter=' ')
    xData2 = np.array(data[:,0])
    yData1 = np.array(data[:,1])
    yData2 = yData1 - min(yData1)

    x=np.linspace(-2,2)

    # x must be between 0 and 10000
    # q must be between -0.01 and 0.01

    # first loop

    rSquared = 0.1

    while rSquared <= 0.999 :
        randomDe = np.random.uniform(0,10000)
        randomA = np.random.uniform(-0.01, 0.01)

        xStart = [randomDe, randomA]
        def morse(x, paramDe, paramA):
            return ((paramDe*(np.exp(-paramA*x)-1)**2))

        popt, pcov = curve_fit(morse, xData2, yData2, p0 = xStart,  maxfev=40000000)
        print(popt)


        yfit = morse(x,popt[0], popt[1])

        print("Mean Squared Error: ", np.mean((yData2-morse(xData2, *popt))**2))
        ss_res = np.dot((yData2 - morse(xData2, *popt)),(yData2 - morse(xData2, *popt)))
        ymean = np.mean(yData2)
        ss_tot = np.dot((yData2-ymean),(yData2-ymean))
        rSquared = 1-ss_res/ss_tot
        print("Mean R :", rSquared)
    else:
        print('Convergence reached')

    paramDe = abs(popt[0])
    print(paramDe)
    paramA = abs(popt[1])
    print(paramA)

    def Nfunction(paramDeforN, paramAforN):
        return (((np.sqrt(2.0 * paramDeforN)) / (paramAforN)) - (1.0 / 2.0))

    # Calc of the F Correction
    Nparam = Nfunction(paramDe, paramA)

    print(Nparam)

    fCorr1 = 2.0 / (2.0 * Nparam - 1.0)
    fNum = Nparam * (Nparam - 1.0) * special.gamma(2.0 * Nparam)
    fDenom = special.gamma(2.0 * Nparam + 1.0)
    fCorr2 = np.sqrt(fNum/fDenom)
    fCorrection = fCorr1 * fCorr2

    print(fCorrection)

    # Saving Data
    resultsData = (abs(popt[0]), abs(popt[1]), fCorrection)
    arrayResultsData = np.array(resultsData)
    arrayResultsDataReshape = arrayResultsData.reshape(1, arrayResultsData.shape[0])
    anhconstants_path = 'anhconstants'
    anhconstants_id = open(anhconstants_path, 'ab')
    np.savetxt(anhconstants_id, arrayResultsDataReshape, fmt=('%-10s','%-10s','%-10s'))



    # # Plots
    plt.figure()
    plt.plot(xData2, yData2,"ro")
    plt.plot(x, yfit)
    plt.show()
    #
    print(mode)






