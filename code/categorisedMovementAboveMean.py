'''
This function is going to convert to 1s and 0s the input time series so it
can be later used to estimate the delay between the manually coded data
and the sensor recordings.

Input: movementSeries -> the sensor measured data
Output: categorisedMovement -> the categorised version of the data

V1.0 Creation of the document by David Lopez Perez 05.10.2020
V1.1 The information about the moving average and the merging time is
asked to the user by David Lopez Perez 16.11.2020
V1.2 The information about the moving average and merging time is also
accepted by the function to asking for information many times when using
loops by David Lopez Perez 22.12.2020
V1.3 Now the function returns the moving average used in the function to
correct the time series later on by David Lopez Perez 20.07.2021
Python translation Agata Kozioł 13.04.2022
'''
from easygui import *
from scipy.ndimage.filters import uniform_filter1d
import numpy as np

def WSelectionDialog(text):
    while True:
        title = 'Pick W value:'
        output = integerbox(text, title, default=1000, lowerbound=150, upperbound=10000)
        message = "Chosen output: " + str(text) + ': ' + str(output)
        print(message)
        break
    return output

def CategorisationSelectionDialog(text, warunek):
    while True:
        title = 'Automatic Categorisation Values'
        output = integerbox(text, title, default=3)
        if warunek:
            try:
                assert output % 2 == 1
            except ValueError:
                msgbox('The moving average length should be an odd number!')
                continue

        message = "Chosen output: " + str(text) + ': ' + str(output)
        print(message)
        break
    return output

def categorisedMovementAboveMean(movementSeries,aveW=-1,mergT=-1):

    if aveW == -1:
        #Ask the user to select the length of the movement average and the window in which individual movements are detected
        answer1 = CategorisationSelectionDialog('Select the moving average length (odd number)', 1)
        if answer1 == None:
            avgWindow = 3
        else:
            avgWindow = answer1
    else:
        avgWindow = aveW
    if mergT == -1:
        answer2 = CategorisationSelectionDialog('Merging time between behaviours:', 0)
        if answer2 == None:
            mergingTime = 3 #Assuming 60Hz for now that would be .05 seconds
        else:
            mergingTime = answer2
    else:
        mergingTime = mergT

# Start the process
    movAvg = avgWindow
    moveAveraged = uniform_filter1d(movementSeries, size=avgWindow)#~50ms
    stdValue = np.std(movementSeries)
    medianValue = np.median(movementSeries)
    categorisedMovement = moveAveraged > (medianValue + stdValue)
    categorisedMovement = categorisedMovement.astype(int)

# Join those individual movements that are detected over the merging time
    differential = np.diff(categorisedMovement)
    endPositions = list(np.where(differential == -1)[0])
    startPositions = list(np.where(differential == 1)[0])

    if len(endPositions) < len(startPositions):
     #That means that the end is in the very last position
        endPositions.append(len(categorisedMovement))

    if len(endPositions) > len(startPositions):
    #That means that the start is in the very first position
        startPositions = np.concatenate(([0], startPositions), axis=None)
    for iEnd in range(len(endPositions)-1):
        if (startPositions[iEnd+1] - endPositions[iEnd]) < mergingTime:
            categorisedMovement[endPositions[iEnd]:startPositions[iEnd+1]+1] = 1

    return categorisedMovement, movAvg
