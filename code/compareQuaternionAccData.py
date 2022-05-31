'''
This function is going to compare the data from acceleration based
measures and quaternion measures and determine if they are comparable

Input: acceleration: movement extracted from acceleration based measures quaternion:
movement extracted from quaternion based data

Output: correlationBetweenSensors: correlation between acceleration and quaternions measures

V1.0 Creation of the document by David Lopez Perez 14.09.2020
V1.1 Bug Fix. The there was a typo in the coefficients name by David López Pérez 12.10.2020
'''
import numpy as np
from scipy import stats


def compareQuaternionAccData(acceleration, quaternion):
    #Zscores both time series.
    acceleration = stats.zscore(acceleration)
    quaternion = stats.zscore(quaternion)
    #CrossCorrelation
    coefficients = np.corrcoef(acceleration[1:], quaternion.flatten())
    correlationBetweenSensors = coefficients[0, 1]
    return correlationBetweenSensors
