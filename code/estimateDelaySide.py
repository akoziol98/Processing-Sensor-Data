'''
This snipplet is going to return the delay between the coded time series
and the sensor time series.

V 1.0 Creation of the document by David Lopez Perez 28.07.2021
Python translation and adjusting to reaching data by Agata Kozioł
'''
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton

from SliderInteractivePlot import SliderInteractivePlot
from categorisedMovementAboveMean import categorisedMovementAboveMean
from correctForDelay import correctForDelay
from drpdfromtsCat import drpdfromtsCat
from loadSensorProcessingOptions import AnalysisSelectionDialog
from timeSeriesShortenActivationPeriod import timeSeriesShortenActivationPeriod

def estimateDelaySide(codedTimeSeries, sensorTimeSeries, W, labelName, numberOccurr, id, differenceSecond):
    #Find the start and end positions of the reaching to crop the time series for better estimation of the delay

    all_columns = codedTimeSeries.columns.values.tolist()
    try:
        if labelName.strip() == 'R_sync':
            sensorTimeSeries = sensorTimeSeries['InfantRightLeg']
            r_columns = [x.strip() for x in all_columns if x.strip() == 'R_sync']
            codedTimeSeries = codedTimeSeries.loc[:, r_columns].any(axis=1).astype(int)
        elif labelName.strip() == 'clap':
            sensorTimeSeries = sensorTimeSeries['ParentRightHand']
            r_columns = [x.strip() for x in all_columns if x.strip() == 'clap']
            codedTimeSeries = codedTimeSeries.loc[:, r_columns].any(axis=1).astype(int)

        elif labelName.strip() == 'L_sync':
            sensorTimeSeries = sensorTimeSeries['InfantLeftLeg']
            l_columns = [x.strip() for x in all_columns if x.strip() == 'L_sync']
            codedTimeSeries = codedTimeSeries.loc[:, l_columns].any(axis=1).astype(int)

    except KeyError as e:
        print("Sensor{0} needed for synchronisation. Restart and choose appropriate sensors_wide".format(repr(e).strip('KeyError')))
        raise KeyError

    start_end_reaches = np.diff(np.concatenate(([0], codedTimeSeries.to_numpy()), axis=None))
    idxEndOfReaches = list(np.where(start_end_reaches == -1)[0])
    idxEndOfReaches[:] = [idx - 1 for idx in idxEndOfReaches]
    idxStartOfReaches = list(np.where(start_end_reaches == 1)[0])

    recalculate = 'no'
    while recalculate != 'yes':
        labels = codedTimeSeries.index[codedTimeSeries == 1].tolist()  # searching for the first and last '1' in the column with the label

        firstLabel = np.min(labels)
        lastLabel = np.max(labels)

        # Important Assumption: max 15 seconds delay between both in terms of# points = time[ms]*60[Hz]/1000
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
        plt.title('{0}\n Select two points. \nSelect a point with left mouse click. \nYou can delete the last label by right click\nFinish adding point by clicking on the mouse middle button.\nLabel:{1}: {2}'.format(id, labelName,numberOccurr))
        plt.xlabel('Time points')
        plt.ylabel('Acceleration m/s^{2}')
        print(labelName)

        sns.lineplot(x=range(firstLabel,lastLabel),
                         y=sensorTimeSeries[firstLabel:lastLabel])
        sns.lineplot(x=range(firstLabel, lastLabel),
                         y=codedTimeSeries[firstLabel:lastLabel] * 20)

        points = plt.ginput(-1, mouse_stop=MouseButton.MIDDLE, show_clicks=True, mouse_add=MouseButton.LEFT, mouse_pop=MouseButton.RIGHT)
        plt.show()

        print('{0}: Selected points: x1={1}, y1={2}, x2={3}, y2={4}'.format(id, round(points[0][0]), round(points[0][1]), round(points[1][0]), round(points[1][1])))

        if round(points[0][0]) < 0: startPos = 0
        else: startPos = round(points[0][0])+1
        if round(points[-1][0]) > lastLabel: endPos = lastLabel
        else: endPos = round(points[-1][0])

        if labelName == 'R_sync' or labelName == 'clap':
            sensorRightHand = sensorTimeSeries[firstLabel:lastLabel].copy()
            sensorRightHand[:startPos] = np.mean(sensorRightHand)
            sensorRightHand[endPos:] = np.mean(sensorRightHand)
            sensorTotal, movAvg = categorisedMovementAboveMean(sensorRightHand)

        elif labelName == 'L_sync':
            sensorLeftHand = sensorTimeSeries[firstLabel:lastLabel].copy()
            sensorLeftHand[:startPos] = np.mean(sensorLeftHand)
            sensorLeftHand[endPos:] = np.mean(sensorLeftHand)
            sensorTotal, movAvg = categorisedMovementAboveMean(sensorLeftHand)

        #Convert 0s to 99s in one of the time series
        codedTimeSeriesAux = codedTimeSeries.iloc[firstLabel:lastLabel].copy().to_numpy()
        codedTimeSeriesAux = timeSeriesShortenActivationPeriod(codedTimeSeriesAux, 4) #Aprox 100ms 4
        codedTimeSeriesAux[codedTimeSeriesAux > 0.5] = 1
        codedTimeSeriesAux[codedTimeSeriesAux != 1] = 99

        sensorData = sensorTimeSeries.copy()

        window = SliderInteractivePlot(sensorTotal, codedTimeSeriesAux, codedTimeSeries, sensorData, firstLabel, lastLabel, differenceSecond)
        profile, maxrec, maxlag = drpdfromtsCat(sensorTotal, codedTimeSeriesAux, window)
        delay = maxlag - window + 1 # one point extra due to the centre being at time 0.
        codedCorrected = correctForDelay(codedTimeSeries.to_numpy(), delay)
        df = pd.DataFrame(list(zip(sensorData, codedCorrected * 20)),
                              columns=['Sensor Data', 'Manually Coded Data'])

        recalculate = 'no'
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
