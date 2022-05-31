'''
This function is going to integrate the accelaration information to obtain
displacement in three dimensions of as an integrative average dimension.

Input accelerationCorrected -> the acceleration in Earth domain
Output -> displacementX -> it will contain the movement for each of the dimensions

V1.0 Creation of the document bu David López Pérez 29.05.2020
V1.1 The selection of collapsing the data is now given to the function to
avoid unnecessary messages by David Lopez Perez 01.06.2020
V1.2 Bug fix movement1D is converted in the main script to improve
compatibility between functions by David López Pérez 14.10.2020
Python translation by Agata Kozioł
'''
import numpy as np

def calculateDisplacement(accelerationCorrected,movement1D,frequency):
    #Filter parameters
    Fs = frequency
    Ts = 1/Fs
    L = len(accelerationCorrected)
    t = np.linspace(0, L, num=L)*Ts
    fc = 0.1/Fs  # Cut off Frequency
    order = 6 # 6th Order Filter
    displacement = accelerationCorrected

    if movement1D:
        displacementAux = np.sqrt(displacement[:,0]**2 + displacement[:,1]**2 + displacement[:,2]**2)
        displacement = np.transpose(displacementAux)

    return displacement

