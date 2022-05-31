'''This snipplet is going to ask for the options to plot the sensorData

Outputs: compareAll: if we want to compare quaternion and gravitation data
         movement1D: collapse the acceleration data into one dimension
         quaternionDistances: convert the data into quaternions

V1.0 Creation of the document by David López Pérez 23.11.2020
Python translation Agata Kozioł 22.03.2022
Modified code from https://www.geeksforgeeks.org/python-easygui-multi-choice-box/ '''
from easygui import *

def AnalysisSelectionDialog(text, choice, default):
    title = "Analysis"
    output = choicebox(text, title, choice, preselect=default)
    message = "Chosen output: " + str(text) + ': ' + str(output)
    print(message)

    return output
def loadSensorProcessingOptions():
    measures = AnalysisSelectionDialog('Do you wanna generate quaternions and gravity correlational measures?', ['yes', 'no'], 1)
    plot = AnalysisSelectionDialog('What type of movement do you choose to plot?', ['for quaternions', 'for gravity based measures'], 1)
    if measures == 'yes':
      compareAll = 1
      movement1D = 1
    else:
      compareAll = 0

    if plot == 'for quaternions':
        quaternionDistances = 1
        movement1D = 1
    else:
        quaternionDistances = 0
        movement = AnalysisSelectionDialog('Do you want to collapse x,y and z in one dimension?', ['yes', 'no'], 0)

        if movement == 'yes':
            movement1D = 1
        else:
            movement1D = 0

    return compareAll, movement1D, quaternionDistances