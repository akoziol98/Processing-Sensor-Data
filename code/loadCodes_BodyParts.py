'''This snipplet is going to load the codes and bodyparts from the codingFile

Input: device -> the information about the device data

V1.0 Creation of the document by David Lopez Perez 23.11.2020
V1.1 This function has been rewritten to accept the path to the files
containing the codes to convert the sensor names to body parts 10.07.2021
Translation to Python by Agata Koziol 17.02.2022'''

import os

import numpy as np
import pandas as pd
from tkinter import filedialog as fd

def loadCodes_BodyParts(device, path_code_files = None):
    #Convert the sensor codes
    try:
        assert path_code_files is not None
        codes = pd.read_csv(path_code_files,
                            sep=' ',
                            usecols=[0, 1],
                            names=['sensor', 'bodypart'])
    except AssertionError:
        print('Load the files with the sensor codes conversion')
        codes_file = fd.askopenfilenames()
        codes = pd.read_csv(codes_file[0],
                            sep=' ',
                            usecols=[0, 1],
                            names=['sensor', 'bodypart'])

    #Split the cell in two columns
    body_parts = codes['bodypart']
    # Order codes and bodyParts according to the loaded files
    sensors_bodyparts = codes.loc[codes['sensor'].isin(device)].reset_index(drop=True)
    body_parts = sensors_bodyparts['bodypart']

    return codes, body_parts
