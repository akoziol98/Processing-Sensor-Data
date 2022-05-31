''' This function is going to take an input array and return the data ready
for interpolation with the missing values exactly in the place that it should be.
V1.0 Creation of the document by David López Pérez 23.04.2020
V1.1 Performance improvement and number of missing samples now is returned
by the algorithm by David López Pérez 09.09.2020
V1,2 Now the function allows to send an empty array. In this case the
data wont be process and an empty array will be returned  by David Lopez Perez 10.08.2021
Translation to Python Agata Kozioł 27.02.2022 '''
import numpy as np
import pandas as pd


def prepareDataForInterpolation(data):
    cols = ['PacketCounter']
    missingValuesDict = {}
    for sensor in data:
        try:
            assert len(data[sensor]['sensor'].unique()) == 1
        except KeyError:
            print('{0} file is empty!'.format(sensor))
            continue

        #Prepare the data for interpolation
        missingValues = 0
        data[sensor]['PacketCounter'] = data[sensor]['PacketCounter'].astype(int)
        originalArray = data[sensor].copy() #.loc[:, cols]
        changedArray = data[sensor].copy() #.loc[:, cols]

        try:
            assert not originalArray.empty
        except AssertionError:
            changedArray = []
            missingValues = np.NaN
            return changedArray, missingValues

    # Calculate the differences in the packages
        diffArray = np.diff(originalArray.loc[:, 'PacketCounter'].astype(int).to_numpy())

        for i in range(len(diffArray)):

            if diffArray[i] < 1:
                #We need to account for a possible change in the numeration of the packages
                if (abs(diffArray[i]) - 65535) != 0:
                  missingValues = missingValues + 65535 - abs(diffArray[i])
                  df_nan=pd.DataFrame(np.nan, index=range(65535 - abs(diffArray[i])), columns=cols)
                  changedArray = pd.concat([changedArray, df_nan], ignore_index=True)
                  changedArray = pd.concat([changedArray, originalArray.iloc[i+1, :].to_frame().transpose()])


            elif diffArray[i] > 1:
                #Adjust for the missing values and copy the value at that position
                df_nan = pd.DataFrame(np.nan, index=range((diffArray[i]-1)), columns=cols)
                changedArray = pd.concat([originalArray.iloc[:i+1, :], df_nan, originalArray.iloc[i+1:, :]]).reset_index(drop=True)
                missingValues = missingValues + diffArray[i]-1

        if sensor in missingValuesDict:
            missingValuesDict[sensor].append(missingValues)
        else:
            missingValuesDict[sensor] = missingValues

    #Fill the interpolation codes to remove NaNs
        a = changedArray.loc[:,'PacketCounter'].isna().sum()
        changedArray.loc[:,'PacketCounter'] = changedArray.loc[:,'PacketCounter'].interpolate(limit_direction='both').astype(int) # interpolate only package numbers
        data[sensor] = changedArray.copy()

        if a != 0:
            print('Missing values for /' + data[sensor]['id'].iloc[0] + ': PacketCounter/ before interpolation: {0}, and after: {1}'.format(a, changedArray.loc[:, 'PacketCounter'].isna().sum()))
    return data, missingValuesDict