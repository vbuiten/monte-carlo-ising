'''Module for plotting the evolution of certain parameters throughout simulations.'''

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
from matplotlib.ticker import AutoMinorLocator
from data.load import LatticeHistories, LatticeHistory
import numpy as np

class EvolutionPlotBase:
    def __init__(self, histories, usetex=False):
        '''
        Base class for making plots of the evolution of magnetisation or energy throughout a simulation.

        :param histories: LatticeHistories instance or LatticeHistory instance or str
                Simulation data. Either a pre-loaded set of data or a string indicating the directory in which
                the data are stored.
        :param usetex: bool
                If True, use Latex for the plot layout. Default is False.
        '''

        plt.rcParams["text.usetex"] = usetex

        if isinstance(histories, LatticeHistories):
            self.histories = histories.histories
        elif isinstance(histories, LatticeHistory):
            self.histories = [histories]
        elif isinstance(histories, str):
            self.histories = LatticeHistories(histories).histories
        else:
            raise TypeError("Expected 'histories' to be either a LatticeHistories instance,"
                            "a LatticeHistory instance, or a directory.")

        if self.histories[1].size != self.histories[0].size:
            raise ValueError("Lattices should all be the same size.")
        else:
            self.size = self.histories[0].size

        self.times = self.histories[0].times
        self.temperatures = np.array([el.temperature for el in self.histories])

        self.fig, self.ax = plt.subplots(figsize=(7,5), dpi=240)
        self.ax.set_title(r"{} \times {} lattice".format(self.size, self.size))

        self.ax.set_xlabel("Time")

        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.grid(which="major")
        self.ax.grid(which="minor", lw=.5, alpha=.5, color="grey")


    def _addHistory(self, history):
        '''
        Add the data of a single simulation to the object.

        :param history: LatticeHistory instance or str
                Data to use. Either a LatticeHistory object or a file name.
        :return: history: LatticeHistory instance
                The newly added LatticeHistory instance.
        '''

        if isinstance(history, LatticeHistory):
            pass
        elif isinstance(history, str):
            history = LatticeHistory(history)

        self.histories = self.histories.append(history)
        return history


    def show(self):
        '''
        Show the figure.

        :return:
        '''

        self.fig.show()


    def save(self, filename):
        '''
        Save the figure.

        :param filename: str
                File name at which to save the figure.
        :return:
        '''

        self.fig.savefig(filename)


class MagnetisationPlotter(EvolutionPlotBase):
    def __init__(self, histories, usetex=False):
        '''
        Class for automatically plotting the evolution of the magnetisation per spin for a series of simulations.

        :param histories: LatticeHistories instance or LatticeHistory instance or str
                Simulation data. Either a pre-loaded set of data or a string indicating the directory in which
                the data are stored.
        :param usetex: bool
                If True, use Latex for the plot layout. Default is False.
        '''

        super(MagnetisationPlotter, self).__init__(histories, usetex)

        magnetisations = np.array([history.magnetisations for history in self.histories])
        self.magnetisations = magnetisations / self.size ** 2

        self.fig.suptitle(r"Evolution of Magnetisation Per Spin")
        self.ax.set_ylabel("Magnetisation per spin")


    def addHistory(self, history):
        '''
        Add the data of a single simulation to the object.

        :param history: LatticeHistory instance or str
                Data to use. Either a LatticeHistory object or a file name.
        :return:
        '''

        history = self._addHistory(history)
        self.magnetisations = np.append(self.magnetisations, history.magnetisations / self.size**2,
                                        axis=0)


    def plot(self):
        '''
        Plot the magnetisations.

        :return:
        '''

        for i, el in enumerate(self.magnetisations):
            self.ax.plot(self.times, el, lw=.5, alpha=.8, label="$T =$ {}".format(np.around(self.temperatures[i], 2)))

        self.ax.legend()


class EnergyPlotter(EvolutionPlotBase):
    def __init__(self, histories, usetex=False):
        '''
        Class for automatically plotting the evolution of the energy per spin for a series of simulations.

        :param histories: LatticeHistories instance or LatticeHistory instance or str
                Simulation data. Either a pre-loaded set of data or a string indicating the directory in which
                the data are stored.
        :param usetex: bool
                If True, use Latex for the plot layout. Default is False.
        '''

        super(EnergyPlotter, self).__init__(histories, usetex)

        energies = np.array([history.energies for history in self.histories])
        self.energies = energies / self.size**2

        self.fig.suptitle("Evolution of Energy Per Spin")
        self.ax.set_ylabel("Energy per spin")


    def addHistory(self, history):
        '''
        Add the data of a single simulation to the object.

        :param history: LatticeHistory instance or str
                Data to use. Either a LatticeHistory object or a file name.
        :return:
        '''

        history = self._addHistory(history)
        self.energies = np.append(self.energies, history.energies / self.size**2, axis=0)


    def plot(self):
        '''
        Plot the energy per spin.

        :return:
        '''

        for i, el in enumerate(self.energies):
            self.ax.plot(self.times, el, lw=.5, alpha=.8, label="$T =$ {}".format(np.around(self.temperatures[i], 2)))

        self.ax.legend()