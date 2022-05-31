'''
This function is displaying manually coded data and asks the user to pick which part of the data to crop:
beginning, end or both
Created by Agata Kozioł
'''
import warnings
from easygui import *
from matplotlib import pyplot as plt
import seaborn as sns

def cropSelection(id):
    title = id
    text = "Do you want to crop the data from the beginning or the end?"
    output = choicebox(text, title, ['beginning', 'end', 'split'])

    return output

def cropData(sensorTimeSeries, sensorFrequency, codedTimeSeries, codedFrequency, differenceMS, id):

    warnings.warn('The difference between sensor and coded data is too big. The longer data will be cropped')
    for key in sensorTimeSeries:
        sensor_len = len(sensorTimeSeries[key])
    sensor_sec = int(round(sensor_len * 1000 / sensorFrequency))

    coded_len = codedTimeSeries.shape[1]
    coded_sec = int(round(coded_len * 1000 * codedFrequency))

    print('sensor seconds: {0}, coded seconds: {1}'.format(sensor_sec, coded_sec))

    if sensor_sec > coded_sec:
        sns.lineplot(data=codedTimeSeries.transpose())
        plt.title('{0}: {1} millisecond difference. Sensor data longer'.format(id, differenceMS))
        plt.show()
        threshold = abs(int(round(differenceMS*sensorFrequency/1000)))
        crop = cropSelection(id)
        if crop == 'beginning':
            for limb in sensorTimeSeries:
                sensorTimeSeries[limb] = sensorTimeSeries[limb][threshold:].copy()
        elif crop == 'end':
            for limb in sensorTimeSeries:
                sensorTimeSeries[limb] = sensorTimeSeries[limb][:len(sensorTimeSeries[limb])-threshold+1].copy()
        elif crop == 'split':
            for limb in sensorTimeSeries:
                sensorTimeSeries[limb] = sensorTimeSeries[limb][round(threshold/2):round((len(sensorTimeSeries[limb])-(threshold/2)+1))].copy()

        print('old sensor len: {0}, difference: {1}, new sensor len: {2}'.format(sensor_len, threshold, len(sensorTimeSeries['InfantRightArm'])))

    elif sensor_sec < coded_sec:
        sns.lineplot(data=codedTimeSeries.transpose())
        plt.title('{0}: {1} millisecond difference. Coded data longer'.format(id, differenceMS))
        plt.show()
        threshold = abs(int(round(differenceMS/(codedFrequency*1000))))
        crop = cropSelection(id)
        if crop == 'beginning':
            codedTimeSeries = codedTimeSeries.iloc[:, threshold:].copy()
        elif crop == 'end':
            codedTimeSeries = codedTimeSeries.iloc[:, :codedTimeSeries.shape[1]-threshold+1].copy()
        elif crop == 'split':
            codedTimeSeries = codedTimeSeries.iloc[:, round(threshold/2):round((codedTimeSeries.shape[1]-(threshold/2)+1))].copy()

        print('old coded len: {0}, difference: {1}, new coded len: {2}'.format(coded_len, threshold, codedTimeSeries.shape[1]))

    new_differenceMS = int(round((len(sensorTimeSeries['InfantRightArm']) * 1000 / sensorFrequency)) - (codedTimeSeries.shape[1] * 1000 * codedFrequency))
    print('New difference: {0}'.format(new_differenceMS))

    return sensorTimeSeries, codedTimeSeries, new_differenceMS

