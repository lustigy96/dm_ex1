#this data_frame includes plot functions

import matplotlib.pyplot as plt
import numpy as np


#scatter plot - for corelation detection
def scatter_plot(data_frame,x_name,y_name,color,marker,title,xlabel, ylabel, ind):
    s = 60          #size of the frame of the marker
    lw = 0          #size of the marker
    alpha = 0.07    #transparency over the edges
    axis_width = 1.5
    tick_len = 6
    fontsize = 16
    #plt.figure(ind)  # Here's the part I need

    ax = plt.scatter(data_frame[x_name].values, data_frame[y_name].values,
                     marker=marker, color=color, s=s, lw=lw,alpha=alpha)
    xrange = abs(data_frame[x_name].max() - data_frame[x_name].min())
    yrange = abs(data_frame[y_name].max() - data_frame[y_name].min())
    cushion = 0.1
    xmin = data_frame[x_name].min() - cushion * xrange
    xmax = data_frame[x_name].max() + cushion * xrange
    ymin = data_frame[y_name].min() - cushion * yrange
    ymax = data_frame[y_name].max() + cushion * yrange
    ax = plt.xlim([xmin, xmax])
    ax = plt.ylim([ymin, ymax])
    ax = plt.xlabel(xlabel, fontsize=fontsize)
    ax = plt.ylabel(ylabel, fontsize=fontsize)
    ax = plt.title(title, fontsize=fontsize+7)
    ax = plt.xticks(fontsize=fontsize)
    ax = plt.yticks(fontsize=fontsize)
    ax = plt.tick_params('both', length=tick_len, width=axis_width,
                         which='major', right=True, top=True)
    return ax

def boxplot(data_frame,data):
    plt.boxplot(data, 0, 'gD')


