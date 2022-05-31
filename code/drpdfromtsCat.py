'''
Quick method to explore the cross-recurrence diagonal profile of two-time series.
It returns the recurrence observed for different delays profile,
the maximal recurrence observed maxrec, and the delay at which it occurred (maxlag).

Based on the R function of the R.Dale and M.Coco
 https://cran.r-project.org/web/packages/crqa/crqa.pdf

Creation of the document by David Lopez
V 1.1 Time series of different length are accepted by David Lopez
V 1.0 Creation of the document 22.12.2016
Python translation Agata Kozioł 03.05.2022
'''
import numpy as np


def drpdfromtsCat(t1, t2, ws):
    #Check the length of both time series and if they are different, shorten
    #the longest one

    if len(t1) != len(t2):
        if len(t1) > len(t2):
            t1[len(t2)+1:len(t1)] = 99 #Shorten T2
            t1[t1 == 99] = []
        else:
            t2[len(t1)+1:len(t2)] = 99 #Shorten T2
            t2[t2 == 99] = []

    #At the moment the function has been adapted only for categorical data.
    datatype = 'categorical'
    drpd = []
    #Negative window values
    for i in range(-ws-1, -1):
        ix = abs(i)
        y = t2[ix:len(t2)]
        x = t1[:len(y)]
        if datatype == 'categorical':
            drpd.append(np.sum((y == x).astype(int))/len(y))
    #Main Diagonal
    if datatype == 'categorical':
        drpd.append(np.sum((t1 == t2).astype(int))/len(t1))

    #Positive window values
    for i in range(1,(ws+2)):
        x = t1[i:len(t1)]
        y = t2[:len(x)]
        if datatype == 'categorical':
            drpd.append(np.sum((y == x).astype(int))/len(y))

    maxrec = np.max(drpd)
    maxlag = np.argmax(drpd)
    profile = drpd
    return profile, maxrec, maxlag
