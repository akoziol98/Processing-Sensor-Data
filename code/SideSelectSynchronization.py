'''
Only sides (R/L/bi) are used for synchronisation, ignoring the exact label

Modified code from https://www.geeksforgeeks.org/python-easygui-multi-choice-box/
Created by Agata Kozioł 29.04.2022
'''
from easygui import *
def SideSelectSynchronization(limbs, id, pick_most):
    '''
    :param limbs: list of available coded limbs
    :return: label: chosen by the user limb which will be used to synchronize sensors_wide and elan coding
    '''

    if pick_most:
        label = limbs[0]
        return label[0] , int(label[1])
    else:
        title = id
        text = "Pick the label for synchronization:\n Syntax: label: number of occurrences"
        label = choicebox(text, title, limbs)
        message = '{0}: Chosen label for synchronization: {1} with {2} occurrences'.format(id, label.split("'")[1].split(",")[0], label.split("'")[-1].strip(",").strip(']').strip(" "))
        print(message)

    return label.split("'")[1].split(",")[0] , int(label.split("'")[-1].strip(",").strip(']').strip(" "))
