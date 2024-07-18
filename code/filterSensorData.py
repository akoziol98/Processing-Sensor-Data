'''This function is going to load the data and return the interpolated data.
V1.0 Creation of the document by David López Pérez 23.11.2020
V1.1 Bug Fix. The function was always try catching a code error all the
time and therefore assuming 60Hz by David López Pérez 26.11.2020
Translation to Python and adjusting interpolation by Agata Kozioł 26.02.2022
'''

import warnings
import numpy as np
import pandas as pd
from interpolateSensorData import interpolateSensorData
from prepareDataForInterpolation import prepareDataForInterpolation

def filterSensorData(data, frequency, id):
    cols = ['PacketCounter', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']

    #We need to interpolate missing packages so the time series are comparable.
    #Prepare the data for interpolation
    try:
        assert data
        df, missingValuesArray = prepareDataForInterpolation(data)
    except AssertionError as e:
        print('There is no data!\n' + repr(e))
    #Calculate the min and max packet number
    fileLengths = [len(df[x]) for x in df]

    if not (fileLengths.count(fileLengths[0]) == len(fileLengths)):
    #Show a warning if after preparing the data something is still wrong
        warnings.warn('The length of the time series is not the same. Padding the beginning or the end with NaNs, double check that the problem is due to files being shorter at the end or the beginning')
        print('{0}: File lengths: {1}'.format(id, fileLengths))
        minPackageNumber = 10000000000
        maxPackageNumber = 0
        for sensor in df:
            if not df[sensor].empty:
                if df[sensor]['PacketCounter'].iloc[0] < minPackageNumber:
                    minPackageNumber = df[sensor]['PacketCounter'].iloc[0]

                if df[sensor]['PacketCounter'].iloc[-1] > maxPackageNumber:
                    maxPackageNumber = df[sensor]['PacketCounter'].iloc[-1]
        for sensor in df:
            if not df[sensor].empty:
                c = np.diff(df[sensor].loc[:, 'PacketCounter'].astype(int).to_numpy())
                result = np.where(c != 1)[0]

                #DataFrame.interpolate() has a problem with linear interpolating
                #without the first(forward) or last value(backward). Also packages start counting from 0 after reaching 65535
                #So I interpolate based on the beginning till number 65535 (if NaN is the first value) or from 0 to the end
                #I manually compute and input the first/last value and interpolate the rest.
                if df[sensor]['PacketCounter'].iloc[0] != minPackageNumber:

                    auxBegin = pd.DataFrame(np.nan,
                               index=range(df[sensor]['PacketCounter'].iloc[0]-minPackageNumber),
                               columns=cols)
                    df[sensor] = pd.concat([auxBegin, df[sensor]], ignore_index=True)
                    if np.isnan(df[sensor].loc[df[sensor].index[0], 'PacketCounter']) and not np.isnan(df[sensor].loc[df[sensor].index[1], 'PacketCounter']):

                        df[sensor].loc[df[sensor].index[0], 'PacketCounter'] = df[sensor].loc[df[sensor].index[1], 'PacketCounter'].astype(int) - 1

                    else:
                        packet = 0
                        packet_count = 0
                        while np.isnan(df[sensor].loc[df[sensor].index[packet], 'PacketCounter']):
                            packet_count += 1
                            packet += 1

                        df[sensor].loc[df[sensor].index[0], 'PacketCounter'] = df[sensor].loc[df[sensor].index[
                                                                                                   packet], 'PacketCounter'] - packet_count

                        df[sensor]['PacketCounter'].iloc[:result[0]+1+len(auxBegin)].interpolate(method='linear',
                                                                                         limit_direction='backward',
                                                                                         inplace = True)

                if df[sensor]['PacketCounter'].iloc[-1] != maxPackageNumber:
                    df_nan = pd.DataFrame(np.nan,
                             index=range((int(maxPackageNumber) - df[sensor]['PacketCounter'].iloc[-1].astype(int))),
                             columns=cols)
                    df[sensor] = pd.concat([df[sensor], df_nan], ignore_index=True)
                    if np.isnan(df[sensor].loc[df[sensor].index[-1], 'PacketCounter']) and not np.isnan(df[sensor].loc[df[sensor].index[-2], 'PacketCounter']):
                        df[sensor].loc[df[sensor].index[-1], 'PacketCounter'] = df[sensor].loc[df[sensor].index[-2], 'PacketCounter'].astype(int) + 1
                    else:
                        packet = -1
                        packet_count = 0
                        while np.isnan(df[sensor].loc[df[sensor].index[packet], 'PacketCounter']):
                            packet_count += 1
                            packet -= 1

                        df[sensor].loc[df[sensor].index[-1], 'PacketCounter'] = df[sensor].loc[df[sensor].index[packet], 'PacketCounter'] + packet_count

                        df[sensor].loc[:, 'PacketCounter'].interpolate(method='linear',
                                                                       limit_direction='forward',
                                                                      inplace=True)

                while np.where(np.diff(df[sensor].loc[:, 'PacketCounter'].astype(int).to_numpy()) != 1)[0].size != 0:
                    result = np.where(np.diff(df[sensor].loc[:, 'PacketCounter'].astype(int).to_numpy()) != 1)[0]
                    if result.size == 1 and df[sensor].loc[df[sensor].index[result[0]], 'PacketCounter'] == 65535:
                        break
                    if result.size != 0:
                        for i in result:
                            if df[sensor].loc[df[sensor].index[i], 'PacketCounter'] == 65535:
                                result = np.delete(result, np.where(result == i)[0])
                                continue
                            dif = int(df[sensor].loc[df[sensor].index[i+1], 'PacketCounter'] - df[sensor].loc[df[sensor].index[i], 'PacketCounter'])
                            if dif != 0:
                                    df_nan = pd.DataFrame(np.nan,
                                                          index=list(range(dif-1)),
                                                          columns=cols)
                                    df[sensor] = pd.concat(
                                        [df[sensor].iloc[:i+1, :], df_nan, df[sensor].iloc[i + 1:, :]],
                                        ignore_index=True)
                                    df[sensor].loc[:, 'PacketCounter'].iloc[i-2:].interpolate(method='linear',
                                                                                   limit_direction='forward',
                                                                                   inplace=True)
                            result = np.delete(result, np.where(result == i)[0])
                    else: break
            df[sensor].loc[:,'PacketCounter'] = df[sensor].loc[:,'PacketCounter'].astype(int)

            assert df[sensor].loc[:, 'PacketCounter'].isna().sum() == 0
            if np.where(np.diff(df[sensor].loc[:, 'PacketCounter'].astype(int).to_numpy()) != 1)[0].size > 1:
                warnings.warn('Something is wrong with package size')

    for sensor in df:
        #Interpolate labels
        if isinstance(df[sensor]['bodypart'].iloc[0], str):
            df[sensor].loc[:, ['bodypart', 'sensor', 'id']] = df[sensor].loc[:, ['bodypart', 'sensor', 'id']].fillna(method='ffill')
        else:
            df[sensor].loc[:, ['bodypart', 'sensor', 'id']] = df[sensor].loc[:, ['bodypart', 'sensor', 'id']].fillna(method='bfill')
        # Display the mean values of missing data
        if missingValuesArray[sensor] != 0:
            print('Missing data values for /' + df[sensor]['id'].iloc[0] + ': ' + df[sensor]['bodypart'].iloc[0] + '/ estimated: average of', 100*np.nanmean(missingValuesArray[sensor])/len(df[sensor]), '% missing data')

#Interpolate possible missing values and filter the time series
    dataFiltered=df.copy()

    for sensor in df:
        try:
            assert not df[sensor].empty
            # Interpolate the data. We remove the 1-2 column because it contains
            # irrelevant data that does not need to be interpolated.
            col = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z']
            dataFiltered[sensor].loc[:, col] = interpolateSensorData(df[sensor].loc[:, col], frequency)
        except AssertionError as e:
            print('The data for interpolation has not been provided\n' + repr(e))

    return dataFiltered
