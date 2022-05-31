''' This snipplet is going to estimate the movement based on the sensor
filtered data and options asked to the user.

Inputs: dataFiltered -> filtered and interpolated sensor data
        compareAll -> this flag indicates that we want to accelaration
                      based data and quaternion data
        movement1D -> if we collapse the 3 coordinates of accelation
                      measures into 1 common or not
        quaternionDistance => estimate the data in quaternions format
        frequency -> the frequency rate of the sensors_wide
        positions -> columns where the acc,gyr,magnetic data is located
                     for quaternions analysis.


Outputs: displacement -> the estimated sensors_wide movement.

 V1.0 Creation of the document by David López Pérez 26.11.2020
 V1.1 The file now wont process anything if the input array is empty. This
has been added just in case data of one of the sensors_wide is missing by David López Pérez 11.08.2021
 V1.2 Performance adjustment. When quaternions and are not needed the Kalman
filter is no longer applied saving some processing time by David López Pérez 12.08.2021
 V1.3 In the quantification of quaternions now the columsn needed for the
 calculation of the data are provided to avoid errors in the data by David López Pérez 02.09.2021
 Python translation Agata Kozioł 22.03.2022
'''
import ahrs
import numpy as np
import pandas as pd

from compareQuaternionAccData import compareQuaternionAccData
from quat_distances import quat_distances
from calculateDisplacement import calculateDisplacement

def estimateSensorDisplacement(dataFiltered,compareAll,movement1D,quaternionDistances):
    disp_temp = {}
    quat_temp = {}
    for sensor in dataFiltered:
    # Start the Process
        if not isinstance(dataFiltered[sensor], pd.DataFrame):
            dataFiltered[sensor] = pd.DataFrame(dataFiltered[sensor])

        quat_angles, quat_Abs_Dist, displacement = {}, {}, {}

        if not dataFiltered[sensor].empty:
            bodypart = dataFiltered[sensor].loc[:, 'bodypart'].unique()
            assert len(bodypart) == 1
            bodypart = bodypart[0]
            acceleration = dataFiltered[sensor].loc[:, ['Acc_X', 'Acc_Y', 'Acc_Z']].to_numpy(dtype=float)
            angularVelocity = dataFiltered[sensor].loc[:, ['Gyr_X', 'Gyr_Y', 'Gyr_Z']].to_numpy(dtype=float)
            magneticField  = dataFiltered[sensor].loc[:, ['Mag_X', 'Mag_Y', 'Mag_Z']].to_numpy(dtype=float)

            frequency = 60
            if compareAll or quaternionDistances:
                qahrs = ahrs.filters.Madgwick(gyr=angularVelocity, acc=acceleration, mag=magneticField, frequency=frequency)
            if compareAll:
                    quat_Abs_Dist, quat_angles = quat_distances(qahrs.Q) #qahrs.Q - quaternions
                    #Get the Acceleration based data
                    displacement = calculateDisplacement(acceleration, 1, frequency)
                    correlationBetweenSensors = compareQuaternionAccData(displacement, quat_Abs_Dist)
                    print('The correlation between acceleration- and quaternion- based data is: ', str(correlationBetweenSensors))
            else:
                if quaternionDistances:
                    #Get the Quaternions based data
                    displacement, quat_angles = quat_distances(qahrs.Q)
                else:
                    #Get the Acceleration based data

                    displacement = calculateDisplacement(acceleration, movement1D, frequency)
        else:
            displacement = []
        disp_temp[bodypart] = displacement
        quat_temp[bodypart] = quat_angles

    return disp_temp, quat_temp
