'''
This script is creating an interactive plot with a slider allowing the user to adjust the window length
and observe how it affects the plot
Code modified from: https://nicoleeic.github.io/Brain_and_Code/2019/08/27/plotting-in-python.html
Created by Agata Kozioł
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from correctForDelay import correctForDelay
from drpdfromtsCat import drpdfromtsCat


def SliderInteractivePlot(sensorTotal, codedTimeSeriesAux, codedTimeSeries, sensorData, firstLabel, lastLabel, differenceSecond):
    def signal(W):
        profile, maxrec, maxlag = drpdfromtsCat(sensorTotal, codedTimeSeriesAux, W)
        delay = maxlag - W + 1 # one point extra due to the centre being at time 0.
        codedCorrected = correctForDelay(codedTimeSeries[firstLabel:lastLabel].to_numpy(), delay)
        return codedCorrected

    fig, ax = plt.subplots(figsize=(15, 8))

    fig.subplots_adjust(bottom=0.25)
    my_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    axbox = fig.add_axes([0.15, 0.1, 0.03, 0.03])
    text_box = TextBox(axbox, 'Enter window size: ', initial='', textalignment='center')
    W = 150

    my_slider = Slider(my_slider_ax, 'Window size', 100, 1000, valinit=W, valstep=1)

    def update_plot(N):
        df = pd.DataFrame(list(zip(sensorData[firstLabel:lastLabel], signal(N) * 20)),
                          columns=['Sensor Data', 'Manually Coded Data'])
        ax.clear()
        sns.lineplot(data=df, ax=ax)
        fig.canvas.draw_idle()

    def slider_action(val):
        update_plot(val)

    def submit(text):
        my_slider.set_val((int(text)))

    text_box.on_submit(submit)
    my_slider.on_changed(slider_action)
    update_plot(W)

    plt.show()
    print(my_slider.val)
    return my_slider.val

