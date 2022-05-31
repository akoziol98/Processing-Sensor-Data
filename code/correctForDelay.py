'''
This funtion is going to correct the time series based on the delay
quantified in the delay estimation method.
Inputs:
     - resampledTimeSeriesCoded -> the manually coded data
     - delay -> is the delay found between sensors_wide and coding

Outputs:

     - resampledTimeSeriesCoded -< the corrected coded data

 V 1.0 Creation of the document by David Lopez Perez 01.08.2021
 V 1.1 Bug Fix validation of the input dimensions has been included by David Lopez Perez 12.08.2021
 V 1.2 The function has been extended to accept vocalisation data by David Lopez Perez 24.08.2021
Python translation and adjusting to reaching data by Agata Kozioł 03.05.2022
'''
import warnings
import numpy as np
import pandas as pd


def correctForDelay(resampledTimeSeriesCoded, delay):
    #Start the process
    if len(resampledTimeSeriesCoded.shape) == 1:
        assert not len(resampledTimeSeriesCoded) == 0

        if delay < 0:
            auxiliaryCoded_TS = resampledTimeSeriesCoded.copy()
            df = np.zeros(abs(delay))
            auxiliaryCoded_TS = np.concatenate((auxiliaryCoded_TS, df), axis=None)
            auxiliaryCoded_TS = auxiliaryCoded_TS[abs(delay)+1:].copy()
            resampledTimeSeriesCoded = auxiliaryCoded_TS.copy()
        elif delay > 0:
            auxiliaryCoded_TS = np.zeros(delay)
            sizeTCoded = len(resampledTimeSeriesCoded)
            auxiliaryCoded_TS = np.concatenate((auxiliaryCoded_TS, resampledTimeSeriesCoded), axis=None)
            auxiliaryCoded_TS = auxiliaryCoded_TS[:sizeTCoded].copy()
            resampledTimeSeriesCoded = auxiliaryCoded_TS.copy()
        else:
            return resampledTimeSeriesCoded
        return resampledTimeSeriesCoded
    elif len(resampledTimeSeriesCoded.shape) == 2:
        cols = list(resampledTimeSeriesCoded.columns)
        resampledTimeSeriesCodedOriginal = resampledTimeSeriesCoded.copy()
        assert not len(resampledTimeSeriesCoded) == 0

        if delay < 0:
            auxiliaryCoded_TS = resampledTimeSeriesCoded.copy()
            df = np.zeros((abs(delay), auxiliaryCoded_TS.shape[1]))
            auxiliaryCoded_TS = np.vstack((auxiliaryCoded_TS, df))
            auxiliaryCoded_TS = auxiliaryCoded_TS[abs(delay) + 1:, :].copy()
            resampledTimeSeriesCoded = auxiliaryCoded_TS.copy()

        elif delay > 0:
            auxiliaryCoded_TS = np.zeros((delay, resampledTimeSeriesCoded.shape[1]))
            sizeTCoded = len(resampledTimeSeriesCoded)
            auxiliaryCoded_TS = np.vstack((auxiliaryCoded_TS, resampledTimeSeriesCoded))
            auxiliaryCoded_TS = auxiliaryCoded_TS[:sizeTCoded, :]
            resampledTimeSeriesCoded = auxiliaryCoded_TS.copy()
        else:
            return resampledTimeSeriesCoded
    else:
        warnings.warn('Something went wrong. Data shape is not correct')
        return resampledTimeSeriesCoded

    resampledTimeSeriesCodedCorrected = pd.DataFrame(data=resampledTimeSeriesCoded, columns=cols)
    dataframes = [resampledTimeSeriesCodedOriginal, resampledTimeSeriesCodedCorrected]
    assert all([len(dataframes[0].columns.intersection(df.columns)) == dataframes[0].shape[1] for df in dataframes])
    return resampledTimeSeriesCodedCorrected
