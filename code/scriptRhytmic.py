'''This script loads the data from the sensors_wide and the manualy coding of the
reaching and manages interpolation, filtering and synchronisation of the data, as well as estimates
some descriptives based on both the coding and the movement data.

V1.0 Translation to Python by Agata Koziol 11.02.2022 based on Matlab project by David Lopez Perez 23.11.2021 '''

import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import pympi
from scipy.interpolate import interp2d, interp1d
from categorisedMovementAboveMean import categorisedMovementAboveMean
from estimateDelaySide import estimateDelaySide
from estimateSensorDisplacement import estimateSensorDisplacement
from filterSensorData import filterSensorData
from SideSelectSynchronization import SideSelectSynchronization
from listSelectionDialog import listSelectionDialog
from loadCodes_BodyParts import loadCodes_BodyParts
from correctForDelay import correctForDelay
##Load and reduction of the data to the limbs of interest

#Select the parent folder where all the subfolders with the sensor data are located
from loadSensorProcessingOptions import loadSensorProcessingOptions

parent_directory = r'G:\Mój dysk\Processing-Sensor-Data\Processing-Sensor-Data\code\babygym_sensors'
os.chdir(parent_directory)
print("Current working directory: {0}".format(os.getcwd()))
files = os.listdir(parent_directory)
# Extract only those that are directories.
subFolders = [name for name in files if os.path.isdir(name)] #contains names of directories inside parent directory
#Get the names of the folder names and a reduce the number of position for further analyses
subFolders = [x for x in subFolders if x not in ['.', '..', '.DS_Store']]
#Load the conversion file
data = {}
device = []
sensorFrequency = {}

for i, file in enumerate(glob.glob(os.path.join(parent_directory, '**'), recursive=True)):
#This loop goes though all items inside parent directory and subdirectories, checks if it is a file and
#ignores files from folders with specified names
    if os.path.isfile(os.path.join(parent_directory, file)) \
            and not (any(x in file.split('/') for x in ['.', '..', '.DS_Store'])):
        file_name = file.split('\\')[-1]
        df = pd.read_csv(file, sep="\t", comment='/',
                         usecols=['PacketCounter', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y', 'Mag_Z'])

        if file.split('.')[1] == 'txt' and 'AF9' in file.split('.')[0]:
            file_name_id = file.split('\\')[-1][:7]

            with open(file) as f:
                r_freq = f.readlines()[1]
                try:
                    sensorFrequency[file_name_id] = int(r_freq .split('Update Rate: ')[1].split('.')[0])
                except IndexError:
                    sensorFrequency[file_name_id] = 60

        if file_name in data:
            print("File '"'{file}'"' already exists. It had been overwritten.".format(file=file_name))
        data[file_name] = df
        pure_file_name = file_name.split('.')[0]
        position = pure_file_name[::-1].find('_')
        device.append(pure_file_name[len(pure_file_name)-position:len(pure_file_name)-position+8])

try:
    device = np.reshape(device, (len(subFolders), len(set(device))))
except ValueError as error:
    print('Sensor data not loaded correctly\n' + repr(error))

'''Load the conversion of body parts'''
try:
    codes_path = r'G:\Mój dysk\Processing-Sensor-Data\Processing-Sensor-Data\code\Codes.txt'
    codes, body_parts = loadCodes_BodyParts(device[0], codes_path)
except TypeError as error:
    print('The device information had not been provided\n' + repr(error))

'''Load the body parts to see which one we are gonna use to reduce the data'''
listOfSelectedParts = listSelectionDialog(body_parts)

'''Modify device and data to get rid of unnecessary data'''
selected = codes.copy()

def func(x):
    return x['bodypart'] in listOfSelectedParts

selected_sensors = selected['sensor'][selected.apply(func, axis=1)].values # dataframe contains sensors_wide matching bodyparts
                                                                           # selected by user
new_device = [sensor for subject in device for sensor in subject if sensor in selected_sensors]

try:
    new_device = np.reshape(new_device, (len(subFolders), len(set(new_device)))) #rows are subjects, columns are sensors_wide
except ValueError as error:
    print('Sensor data not selected correctly\n' + repr(error))
for i in new_device:
    i = i.sort()
for i in range(len(new_device)):
    if not (new_device[0] == new_device[i]).all():
        print(new_device)
        raise ValueError('Selected sensors_wide for each subject are not the same')

'''Mapping sensor code -> bodypart'''
def sens_body(x):
    ix = codes[codes.sensor == x].first_valid_index()
    return codes.loc[ix, 'bodypart']

new_data = {}
sensor_id_list = set([])

for x in data:
    try:
        assert not data[x].empty
        data[x].insert(1, 'id', x[:7])
        pure_file_name = x.split('.')[0]
        position = pure_file_name[::-1].find('_')
        data[x].insert(2, 'sensor', pure_file_name[len(pure_file_name) - position:len(pure_file_name) - position + 8])
        data[x].insert(3, 'bodypart', sens_body(data[x].loc[0, 'sensor']))
    except AssertionError as error:
        print('{1}: Data from {0} file seems to be empty\n'.format(x, repr(error)))
        print(data[x])
    sensor_id_list.add(x[:7])
    sensor_code = x[len(x) - 12:len(x) - 4]
    if sensor_code in selected_sensors:
        if x[:7] not in new_data:
            new_data[x[:7]] = {}
            new_data[x[:7]][x] = data[x]
        else:
            new_data[x[:7]].update({x: data[x]})

'''Filter and Interpolate the data'''
dataFiltered = {}
print('Filtering and Interpolating data')

for participant in new_data:
    print(participant)
    dataFiltered[participant] = filterSensorData(new_data[participant], sensorFrequency[participant], participant)

dataFiltered_Interpolate = dataFiltered.copy()
'''Select the plot you want to perform and if you want to compare acceleration and quaternion based measures'''
compareAll, movement1D, quaternionDistances = loadSensorProcessingOptions()
'''Calculate the sensorMovement based on the selected options and filteredData'''
displacement = {}
quat_angles = {}

for participant in dataFiltered_Interpolate:
        displacement[participant], quat_angles[participant] = estimateSensorDisplacement(dataFiltered_Interpolate[participant], compareAll, movement1D, quaternionDistances)

'''Load the manual coded data
Select the parent folder where all the subfolders with the sensor data are located'''
parent_directory_codes = r"G:\Mój dysk\Processing-Sensor-Data\Processing-Sensor-Data\code\babygym coding"
os.chdir(parent_directory_codes)
print("Current working directory: {0}".format(os.getcwd()))
files_elan = os.listdir(parent_directory_codes)
filesCodingNames = [name for name in files_elan if os.path.isdir(name)]
filesCodingNames = [x for x in filesCodingNames if x not in ['.', '..', '.DS_Store']]

'''Extract the codes of the coded files'''

'''2.1 Loop through the list of unique codes and load the ones that has sensor data'''
''' Elan processing based on https://dopefishh.github.io/pympi/Elan.html '''
coded_sensors = [x for x in filesCodingNames if x in sensor_id_list and x in filesCodingNames]

elan_data = {}
listOfBehaviours = {}
totalSeconds = {}
codingFrequency = {}
all_annotations = {}
timeCoded = {}
totalLength = {}
displacement_reduced = {}

for participant in displacement:
    if participant in coded_sensors:
        displacement_reduced[participant] = displacement[participant].copy()

for i, file in enumerate(glob.glob(os.path.join(parent_directory_codes, '**'), recursive=True)):

    if os.path.isfile(os.path.join(parent_directory, file)) \
            and not (any(x in file.split('\\') for x in ['.', '..', '.DS_Store'])) and file.split('\\')[-1][:7] in coded_sensors:

        if file.split('.')[1] == 'txt':
            file_name = file.split('\\')[-1][:7]
            with open(file) as f:
                df = f.readline()
                totalSeconds[file_name] = float(df.split('duration: ')[1].split('/ ')[1])
                codingFrequency[file_name] = float(df.split('sample: ')[1].split('"')[0])/1000
                totalLength[file_name] = round(totalSeconds[file_name] / codingFrequency[file_name])

        elif file.split('.')[1] == 'eaf':
            file_name = file.split('\\')[-1][:7]
            elan_file = pympi.Elan.Eaf(file)
            if file_name in data:
                 print("File '"'{file}'"' already exists. It had been overwritten.".format(file=file_name))
            elan_data[file_name] = elan_file

            df = pd.DataFrame(columns=['id', 'label', 'StartTime', 'EndTime'])
            elan_file.get_full_time_interval()

            for tier in elan_file.get_tier_names():
                for ann in elan_file.get_annotation_data_for_tier(tier):
                    df2 = pd.DataFrame({'id': file_name, 'label': ann[2], 'StartTime': ann[0]/1000, 'EndTime': ann[1]/1000}, index=[0])
                    df = pd.concat([df, df2], ignore_index=True)
            all_annotations[file_name] = df.sort_values('StartTime').reset_index(drop=True)

            listOfBehaviours[file_name] = all_annotations[file_name].loc[:, 'label'].unique()

for file_name in all_annotations:
    timeSeriesCoded = pd.DataFrame(data=np.zeros([len(listOfBehaviours[file_name]), totalLength[file_name]]),
                      index=listOfBehaviours[file_name],
                      columns=range(totalLength[file_name]))


    for behaviour in listOfBehaviours[file_name]:
        for iValue in range(len(all_annotations[file_name][all_annotations[file_name]['label'] == behaviour].loc[:, 'StartTime'].reset_index(drop=True))):
            if all_annotations[file_name][all_annotations[file_name]['label'] == behaviour].loc[:, 'StartTime'].reset_index(drop=True)[iValue] == 0:
                timeSeriesCoded.loc[behaviour, :(round(all_annotations[file_name][all_annotations[file_name]['label'] == behaviour].loc[:, 'EndTime'].reset_index(drop=True)[iValue] / codingFrequency[file_name]))] = 1
            else:
                timeSeriesCoded.loc[behaviour,(round(all_annotations[file_name][all_annotations[file_name]['label'] == behaviour].loc[:, 'StartTime'].reset_index(drop=True)[iValue] / codingFrequency[file_name])):(round(all_annotations[file_name][all_annotations[file_name]['label'] == behaviour].loc[:, 'EndTime'].reset_index(drop=True)[iValue] / codingFrequency[file_name]))] = 1
    timeCoded[file_name] = timeSeriesCoded

'''Synchronise the manual and the movement time series'''
# 1-. We need to resample
resampledTimeSeriesCoded = {}
resampledTimeSeriesCoded1D = {}
differenceMSeconds = {}
differ = {}
# First crop the data if the difference between them is too big

for file_name in timeCoded:
    print('{3}: Milliseconds of sensor data: {0}, milliseconds of manual coded data: {1}. Difference: {2}'.format(round(np.min(list(displacement_reduced[file_name].values())[0].shape)*(1000/sensorFrequency[file_name])), round(totalSeconds[file_name]*1000), round(np.min(list(displacement_reduced[file_name].values())[0].shape)*(1000/sensorFrequency[file_name])) - totalSeconds[file_name]*1000, file_name))
    print('{2}: Sensors: {0}, manual coded: {1}'.format(np.min(list(displacement_reduced[file_name].values())[0].shape), timeCoded[file_name].shape[1], file_name))

    differenceMSeconds[file_name] = int(round(np.min(list(displacement_reduced[file_name].values())[0].shape) * (1000 / sensorFrequency[file_name])) - totalSeconds[file_name] * 1000)
    differ[file_name] = int(round(np.min(list(displacement_reduced[file_name].values())[0].shape) * (1000 / sensorFrequency[file_name])) - totalSeconds[file_name] * 1000)

for file_name in timeCoded:
    resampledTimeSeriesCoded1D[file_name] = {}
    if timeCoded[file_name].shape[0] > 1:
        resampling = interp2d(x=list(range(0, timeCoded[file_name].shape[0])),
                              y=list(range(0, timeCoded[file_name].shape[1])),
                              z=timeCoded[file_name].astype(int).transpose())
        resampled = \
            resampling(list(range(0, timeCoded[file_name].shape[0])),
                       np.linspace(start=0, stop=timeCoded[file_name].shape[1],
                                   num=np.min(list(displacement_reduced[file_name].values())[0].shape)))

    elif timeCoded[file_name].shape[0] == 1:
        resampling = interp1d(x=list(range(0, timeCoded[file_name].shape[1])),
                              y=timeCoded[file_name].astype(int), fill_value="extrapolate")
        resampled = \
            resampling(np.linspace(start=0, stop=timeCoded[file_name].shape[1],
                                   num=np.min(list(displacement_reduced[file_name].values())[0].shape))).transpose()

    resampledTimeSeriesCoded[file_name] = \
        pd.DataFrame(data=resampled,
                     columns=listOfBehaviours[file_name],
                     index=range(np.min(list(displacement_reduced[file_name].values())[0].shape)))

    resampledTimeSeriesCoded[file_name].iloc[:, 0][resampledTimeSeriesCoded[file_name].iloc[:, 0] >= 0.5] = 1
    resampledTimeSeriesCoded[file_name].iloc[:, 0][resampledTimeSeriesCoded[file_name].iloc[:, 0] < 0.5] = 0

''' 2-. Realign the time series of sensors_wide in relation to the coded data'''
W = 150

resampledTimeSeriesCodedCorrected = {}
delay = {}
labelSynch = {}

resampledTimeSeriesCodedCorrected = resampledTimeSeriesCoded.copy()
custom_sync = True
for participant in resampledTimeSeriesCoded:
    assert len(all_annotations[participant]['id'].unique()) == 1

    all_columns = all_annotations[participant]['label'].unique().tolist()
    if custom_sync:
        print('Custom synchronizing')
        r_columns = [True if x == 'R_sync' else False for x in all_annotations[participant]['label']]
        l_columns = [True if x == 'L_sync' else False for x in all_annotations[participant]['label']]
        clap = [True if x == 'clap' else False for x in all_annotations[participant]['label']]
        r_count, l_count, clap_count = 0, 0, 0

        for label, count in all_annotations[participant].groupby('label').count().loc[:, 'id'].items():
            if label == 'R_sync':
                r_count += count
            elif label == 'L_sync':
                l_count += count
            elif label== 'clap':
                clap_count += count

            label_count = [['R_sync', r_count], ['L_sync', l_count], ['clap', clap_count]]
            label_count = sorted(label_count, key=lambda x: x[1], reverse=True)

        synch_label = label_count[0][0]
        num = label_count[0][1]

        delay[participant], labelSynch[participant] = estimateDelaySide(resampledTimeSeriesCoded[participant].copy(),
                                                                        displacement_reduced[participant].copy(), W,
                                                                        synch_label, num, participant,
                                                                        differenceMSeconds[participant])
        resampledTimeSeriesCodedCorrected[participant] = correctForDelay(resampledTimeSeriesCoded[participant],
                                                                         delay[participant])
    else:
        print('Something is wrong with synchronisation')
movementsCategorised = {}
# Extract only those periods of reaching/clapping to remove movement "noise" and calculate the "reaching"/"clapping" time series

for participant in resampledTimeSeriesCodedCorrected:
    movementsCategorised[participant] = {}

    #Crop the time series and create new ones of the infants data
    for label in list(resampledTimeSeriesCodedCorrected[participant].columns):
        a = resampledTimeSeriesCodedCorrected[participant].loc[:, label].to_numpy()
        a[(a != 0) & (a < 0.5)] = 0
        a[(a != 1) & (a >= 0.5)] = 1
        difference = np.diff(np.concatenate(([0], a, [0]), axis=None))
        numberOfReachingBlocks = np.sum(difference == 1)

        startPositions = list(np.where(difference == 1)[0])
        endPositions = list(np.where(difference == -1)[0])
        labelSensor = []
        if label == "":
            print('ERROR: {0}: {1}'.format(participant, label))
            label = "L_touch"
        if label[0] == 'R': limbs = ['InfantRightArm']
        elif label[0] == 'L': limbs = ['InfantLeftArm']
        elif label[0] == 'b': limbs = ['InfantRightArm', 'InfantLeftArm']
        labelSensors = {}
        labelSensor = []
        if label.strip() == 'clap' or label.strip() == 'R_sync' or label.strip() == 'L_sync':
            continue
        elif label[0] == 'R' or label[0] == 'L':
            for limb in limbs:
                for iBlock in range(numberOfReachingBlocks):
                    labelSensor = np.concatenate((labelSensor, displacement_reduced[participant][limb][startPositions[iBlock]:endPositions[iBlock]]), axis=None)
                labelSensors[label] = labelSensor
        elif label[0] == 'b':
            labelSensors[label] = {}
            for limb in limbs:
                labelSensor = []
                for iBlock in range(numberOfReachingBlocks):
                    labelSensor = np.concatenate((labelSensor, displacement_reduced[participant][limb][startPositions[iBlock]:endPositions[iBlock]]), axis=None)
                labelSensors[label][limb] = labelSensor
        else:
            warnings.warn(('Something is wrong'))
            print('x',label,'x')
        #Categorised around the mean to create time series of reaching periods
        if label[0] == 'R' or label[0] == 'L':
            movementsCategorised[participant][label], _ = categorisedMovementAboveMean(labelSensors[label], 3, 3)
        elif label[0] == 'b':
            r, _ = categorisedMovementAboveMean(labelSensors[label][limbs[0]], 3, 3)
            l, _ = categorisedMovementAboveMean(labelSensors[label][limbs[1]], 3, 3)
            movementsCategorised[participant][label] = np.mean(np.vstack((r, l)), axis=0)

        else:
            warnings.warn(('Something is wrong'))

for participant in timeCoded:
    timeCoded[participant] = timeCoded[participant].transpose()

