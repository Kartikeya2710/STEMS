import numpy as np
import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self, path, dpi):
            self._path = path
            self._dpi = dpi


    def save_data_and_plot(self, data, ttl_data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the relative data to txt
        """
        min_val = min(min(data), min(ttl_data))
        max_val = max(max(data), max(ttl_data))

        plt.rcParams.update({'font.size': 24})  # set bigger font size
        x = np.arange(len(data))
        plt.plot(x, data, c='r', label='ours')
        plt.plot(x, ttl_data, c='b', label='ttl')
        plt.legend(loc="upper right")
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                    file.write("%s\n" % value)