'''
This function is going to interpolate the data from the sensor data to
remove missing values. Initially a spline inteporlation is done, but in
the future further expansions can be developed.

Input:
     inputData: The original data
     frequency: frequency of the data acquisition
Ouput:
     dataInterpolated: The interpolated and filtereddata.

 V1.0 Creation of the document by David Lopez Perez 10.10.2020
 V1.1 Bug Fix the second column normally contains a full list of NaN
 values because it is not sensor-relevant data. Now this is not considered for the analysis.
 V1.2 Bug Fix. If there is a column full of Nans, that column is ignored by David Lopez Perez 02.10.2020
 V1.3 Now the function allows to send an empty array. In this case the
 data wont be process and an empty array will be returned  by David Lopez Perez 10.08.2021
 V1.4 Bug Fix. Previous to interpolation the data was being abs so no
 negative values were present. This has been removed by David Lopez Perez 02.09.2021
 Python translation Agata Kozioł 15.03.2022
'''
import warnings
import numpy as np
from scipy.signal import medfilt


def interpolateSensorData(inputData,frequency):

    dataInterpolated = inputData.copy()
    if not inputData.empty:
        for iColumn in list(inputData.columns):
            if np.isnan(inputData.loc[:, iColumn]).all():
                warnings.warn('Check it')
            else:
                #Calculate error between the median filter model and original
                x = inputData.loc[:, iColumn]
                #Interpolate them using spline interpolation

                #Before interpolating we need to make sure that the last values are
                #not NaN because that can return problems in the interpolation

                # This part is making sure first or last value isn't NaN. If it is, we need to extrapolate forward/backward
                if np.isnan(x.iloc[-1]) or np.isnan(x.iloc[0]):
                    if np.isnan(x.iloc[-1]):
                        x.interpolate(method='spline', limit_direction='forward', inplace=True, fill_value='extrapolate',
                                   order=3)
                    if np.isnan(x.iloc[0]):
                        x.interpolate(method='spline', limit_direction='backward', inplace=True, fill_value='extrapolate',
                                   order=3)

                else: #interpolation is done when the first and last value isn't NaN
                    x.interpolate(method='spline', limit_direction='forward', inplace=True, order=3)
                if not x[np.isnan(x)].empty:
                    print(x[np.isnan(x)])

                X_Interpolated = x.copy()
                #Smooth the data a bit using a median filter
                dataInterpolated.loc[:,iColumn] = medfilt(X_Interpolated, 3)

    else:
        dataInterpolated = []

    return dataInterpolated
