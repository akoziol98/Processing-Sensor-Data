''' Modified code from https://www.geeksforgeeks.org/python-easygui-multi-choice-box/
Interface allowing the user pick sensors for further analysis
by Agata Kozioł
'''
from easygui import *

def listSelectionDialog(varargin):
    text = 'Select the parts for interest for the analysis'
    title = "Window Title GfG"
    varargout = multchoicebox(text, title, varargin, preselect=[3, 9, 10, 11])
    message = "Selected items : " + str(varargout)
    print(message)

    return varargout
