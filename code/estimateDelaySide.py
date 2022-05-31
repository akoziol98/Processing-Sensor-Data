'''
Only sides (R/L/bi) are used for synchronisation, ignoring the exact label

This snipplet is going to return the delay between the coded time series
and the sensor time series.

V 1.0 Creation of the document by David Lopez Perez 28.07.2021
Python translation and adjusting to reaching data by Agata Kozioł
'''
import numpy as np
import pandas as pd
import seaborn as sns
from easygui import *
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton

from SliderInteractivePlot import SliderInteractivePlot
from categorisedMovementAboveMean import categorisedMovementAboveMean
from correctForDelay import correctForDelay
from drpdfromtsCat import drpdfromtsCat
from loadSensorProcessingOptions import AnalysisSelectionDialog
from timeSeriesShortenActivationPeriod import timeSeriesShortenActivationPeriod

def CategorisationSelectionDialog(reaches, participant):
    text = "Do you want to specify custom threshold for extracting period for synchronisation?\n{0}".format(reaches)

    title = participant
    custom = buttonbox(text, title, ['No', 'Custom threshold'])

    if custom == 'No': return False
    while True:

        title = participant
        output = enterbox('Enter the time period you want to include in sychronisation\nIdentified reaches: {0}'.format(reaches), title)
        message = "Chosen output: " + str(output)
        print(message)
        break
    return output

def estimateDelaySide(codedTimeSeries, sensorTimeSeries, W, labelName, numberOccurr, id, differenceSecond):
    #Find the start and end positions of the reaching to crop the time series for better estimation of the delay
    all_columns = codedTimeSeries.columns.values.tolist()
    try:
        if labelName == 'R':
            sensorTimeSeries = sensorTimeSeries['InfantRightArm']
            r_columns = [x for x in all_columns if x.startswith('R')]
            codedTimeSeries = codedTimeSeries.loc[:, r_columns].any(axis=1).astype(int)


        elif labelName == 'L':
            l_columns = [x for x in all_columns if x.startswith('L')]
            codedTimeSeries = codedTimeSeries.loc[:, l_columns].any(axis=1).astype(int)
            sensorTimeSeries = sensorTimeSeries['InfantLeftArm']

        elif labelName == 'right_leg':
            sensorTimeSeries = sensorTimeSeries['InfantRightLeg']
            codedTimeSeries = codedTimeSeries.any(axis=1).astype(int)

        elif labelName == 'bi':
            bi_columns = [x for x in all_columns if x.startswith('b')]
            codedTimeSeries = codedTimeSeries.loc[:, bi_columns].any(axis=1).astype(int)
            sensorTimeSeries = np.vstack((list(sensorTimeSeries['InfantRightArm']), list(sensorTimeSeries['InfantLeftArm'])))
    except KeyError as e:
        print("Sensor{0} needed for synchronisation. Restart and choose appropriate sensors_wide".format(repr(e).strip('KeyError')))
        raise KeyError


    start_end_reaches = np.diff(np.concatenate(([0], codedTimeSeries.to_numpy()), axis=None))
    idxEndOfReaches = list(np.where(start_end_reaches == -1)[0])
    idxEndOfReaches[:] = [idx - 1 for idx in idxEndOfReaches]
    idxStartOfReaches = list(np.where(start_end_reaches == 1)[0])
    reaches = list(map(lambda X: (X[0], X[1]), list(zip(idxStartOfReaches, idxEndOfReaches))))
    choice = True
    while choice:
        output = CategorisationSelectionDialog(reaches, id)
        labels = codedTimeSeries.index[codedTimeSeries == 1].tolist()  # searching for the first and last '1' in the column with the label
        codedTimeSeriesIdx = np.arange(len(codedTimeSeries)).tolist()

        if not output:
            firstLabel = np.min(labels)
            lastLabel = np.max(labels)
            reaches_thr = [firstLabel, lastLabel]
        else:
            reaches_thr = []
            try:
                threshold = int(output) + 1
                numberOccurr = len(list(np.where(start_end_reaches[:threshold] == -1)[0]))
                firstLabel = np.min(labels)
                lastLabel = np.max(codedTimeSeriesIdx[:threshold])
            except ValueError:
                if ":" in output:
                    output = output.split(":")
                    firstLabel = int(output[0])
                    lastLabel = int(output[1])+2
                    numberOccurr = len(list(np.where(start_end_reaches[firstLabel:lastLabel] == -1)[0]))
                    for x, y in reaches:
                        if x >= firstLabel and y <= lastLabel:
                            reaches_thr.append((x, y))

        # Important Assumption: max 15 seconds delay between both in terms of
        firstLabel = firstLabel - 900
        lastLabel = lastLabel + 900
        if firstLabel < 0:
            firstLabel = 0
        if labelName == 'bi':
            if lastLabel > len(sensorTimeSeries[0]):
                lastLabel = len(sensorTimeSeries[0])-1
        else:
            if lastLabel > len(sensorTimeSeries):
                lastLabel = len(sensorTimeSeries)-1

        # Plot the data to get the range in which we want to obtain the labeled reaches
        plt.figure(figsize=(15, 7), num=id)
        plt.title('{0}\n Select two points. \nSelect a point with left mouse click. \nYou can delete the last label by right click\nFinish adding point by clicking on the mouse middle button'.format(id))
        plt.xlabel('Time points')
        plt.ylabel('Acceleration m/s^{2}')

        if labelName == 'R' or labelName == 'L':
            sns.lineplot(x=range(firstLabel,lastLabel),
                         y=sensorTimeSeries[firstLabel:lastLabel])
            sns.lineplot(x=range(firstLabel, lastLabel),
                         y=codedTimeSeries[firstLabel:lastLabel] * 20)

        elif labelName == 'bi':
            sns.lineplot(x=range(firstLabel, lastLabel),
                         y=np.mean(np.vstack((sensorTimeSeries[0][firstLabel:lastLabel], sensorTimeSeries[1][firstLabel:lastLabel])), axis=0))
            sns.lineplot(x=range(firstLabel, lastLabel),
                         y=codedTimeSeries[firstLabel:lastLabel] * 20)
        points = plt.ginput(-1, mouse_stop=MouseButton.MIDDLE, show_clicks=True, mouse_add=MouseButton.LEFT, mouse_pop=MouseButton.RIGHT)
        plt.show()
        choices = AnalysisSelectionDialog('Do you want to pick another reaching period?', [True, False], 1)
        if choices == 'False':
            break
    print('{0}: Selected points: x1={1}, y1={2}, x2={3}, y2={4}'.format(id, round(points[0][0]), round(points[0][1]), round(points[1][0]), round(points[1][1])))

    if round(points[0][0]) < 0: startPos = 0
    else: startPos = round(points[0][0])+1
    if round(points[-1][0]) > lastLabel: endPos = lastLabel
    else: endPos = round(points[-1][0])


    if labelName == 'bi':
        #After the first and last clapping has been found then we croped the time series and calculate the moving average
        sensorRightHand = sensorTimeSeries[0][firstLabel:lastLabel].copy()
        sensorLeftHand = sensorTimeSeries[1][firstLabel:lastLabel].copy()
        sensorRightHand[:startPos] = np.mean(sensorRightHand)
        sensorRightHand[endPos:] = np.mean(sensorRightHand)
        sensorLeftHand[:startPos] = np.mean(sensorLeftHand)
        sensorLeftHand[endPos:] = np.mean(sensorLeftHand)
        sensorTotal, movAvg = categorisedMovementAboveMean(np.mean([sensorRightHand, sensorLeftHand], axis=0))
    elif labelName == 'R':
        sensorRightHand = sensorTimeSeries[firstLabel:lastLabel].copy()
        sensorRightHand[:startPos] = np.mean(sensorRightHand)
        sensorRightHand[endPos:] = np.mean(sensorRightHand)
        sensorTotal, movAvg = categorisedMovementAboveMean(sensorRightHand)

    elif labelName == 'L':
        sensorLeftHand = sensorTimeSeries[firstLabel:lastLabel].copy()
        sensorLeftHand[:startPos] = np.mean(sensorLeftHand)
        sensorLeftHand[endPos:] = np.mean(sensorLeftHand)
        sensorTotal, movAvg = categorisedMovementAboveMean(sensorLeftHand)

    #Convert 0s to 99s in one of the time series
    codedTimeSeriesAux = codedTimeSeries.iloc[firstLabel:lastLabel].copy().to_numpy()
    codedTimeSeriesAux = timeSeriesShortenActivationPeriod(codedTimeSeriesAux, 4) #Aprox 100ms 4
    codedTimeSeriesAux[codedTimeSeriesAux > 0.5] = 1
    codedTimeSeriesAux[codedTimeSeriesAux != 1] = 99

    recalculate = 'no'
    while recalculate != 'yes':
        if labelName == 'bi':
            sensorData = np.mean(np.vstack((sensorTimeSeries[0], sensorTimeSeries[1])), axis=0)
        elif labelName == 'R' or labelName == 'L' or labelName == 'right_leg':
            sensorData = sensorTimeSeries.copy()

        window = SliderInteractivePlot(sensorTotal, codedTimeSeriesAux, codedTimeSeries, sensorData, firstLabel, lastLabel, differenceSecond)
        profile, maxrec, maxlag = drpdfromtsCat(sensorTotal, codedTimeSeriesAux, window)
        delay = maxlag - window + 1 # one point extra due to the centre being at time 0.
        codedCorrected = correctForDelay(codedTimeSeries.to_numpy(), delay)
        df = pd.DataFrame(list(zip(sensorData, codedCorrected * 20)),
                          columns=['Sensor Data', 'Manually Coded Data'])

        plt.figure(figsize=(15, 8), num=id)
        sns.lineplot(data=df)
        plt.title('{0}: Corrected delay'.format(id))
        plt.show()

        recalculate = AnalysisSelectionDialog('Does the estimate delay corrected properly the time series?',
                                              ['yes', 'no'], 0)

    df_not_corrected = pd.DataFrame(list(zip(sensorData, codedTimeSeries * 20)),
                          columns=['Sensor Data', 'Manually Coded Data'])

    f, axs = plt.subplots(2, 1,
                          figsize=(15,8),
                          sharex=True,
                          num=id)

    sns.lineplot(data=df, ax=axs[0])
    plt.suptitle('Corrected delay')
    sns.lineplot(data=df_not_corrected, ax=axs[1])
    plt.title('Not corrected delay')
    plt.show()

    return delay, labelName
