'''
This function is going to shorten the activation periods to allow to
capture the real dynamics. When short hand movements are RQAed to long
manual coded periods...the estimation of RQA is not entirely accurate.
V1.0 Creation of the document by David Lopez Perez 23.07.2021
Python translation Agata Kozioł 04.05.2022
'''
import numpy as np

def timeSeriesShortenActivationPeriod(longTS,activationLength):
    # Create the shortened time series
    shortenedTS = np.zeros(longTS.shape)
    #Make sure that we have only 1s and 0s
    longTS[longTS < 1] = 0
    differential = np.diff(np.concatenate(([0], longTS), axis=None))
    positionsOfOnes = list(np.where(differential == 1))[0]

    # Find the beginning of each block
    for iPos in range(len(positionsOfOnes)):
        shortenedTS[positionsOfOnes[iPos]:positionsOfOnes[iPos]+activationLength] = 1
    return shortenedTS

