'''Code for measuring observables from simulation data.'''

import numpy as np
from analysis.utils import *
from data.load import LatticeHistory, LatticeHistories
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rcParams["font.family"] = "serif"

class Measurer:
    def __init__(self,
                 history=None,
                 times=None,
                 magnetisations=None,
                 energies=None,
                 correlation_time=None,
                 lattice_size=None,
                 temperature=None):
        '''
        Class for conducting measurements of the mean absolute spin, the energy per spin, the magnetic susceptibility
        and the specific heat of the lattice for a single simulation.

        :param history: LatticeHistory instance or str or NoneType
                Simulation data. Must be either a LatticeHistory object or a file name. If None, the data must be
                provided through the other arguments of the function.
        :param times: ndarray of shape (n_times,) or NoneType
                Time stamps of the simulation data. If None, argument "history" must be provided. Default is None.
        :param magnetisations: ndarray of shape (n_times,) or NoneType
                Total magnetisation of the lattice at each point in time. If None, argument "history" must be provided.
                Default is None.
        :param energies: ndarray of shape (n_times,) or NoneType
                Total energy of the lattice at each point in time. If None, argument "history" must be provided.
                Default is None.
        :param correlation_time: float or NoneType
                Correlation time of the system. If None, argument "history" must be provided. Default is None.
        :param lattice_size: int or NoneType
                Linear size of the lattice in terms of atoms on each side. If None, argument "history" must be provided.
                Default is None.
        :param temperature: float or NoneType
                Temperature of the system. If None, argument "history" must be provided. Default is None.
        '''

        if isinstance(history, LatticeHistory):
            self.history = history
            self.times = self.history.times
            self.magnetisations = self.history.magnetisations
            self.energies = self.history.energies
            self.size = self.history.size
            self.temperature = self.history.temperature

            if correlation_time is not None:
                self.correlation_time = correlation_time
            else:
                self.correlation_time = self.history.correlation_time

        elif isinstance(history, str):
            self.history = LatticeHistory(history)

            self.times = self.history.times
            self.magnetisations = self.history.magnetisations
            self.energies = self.history.energies
            self.size = self.history.size
            self.temperature = self.history.temperature

            if correlation_time is not None:
                self.correlation_time = correlation_time
            else:
                self.correlation_time = self.history.correlation_time

        elif history is None:
            if times is None or magnetisations is None or energies is None or correlation_time is None\
                    or lattice_size is None or temperature is None:
                raise ValueError("Either a history/data file must be given, or the other parameters must all be provided.")

            else:
                self.times = times
                self.magnetisations = magnetisations
                self.energies = energies
                self.correlation_time = correlation_time
                self.size = lattice_size
                self.temperature = temperature

        if self.correlation_time is None:
            raise TypeError("No correlation time found!")

        # divide times up in intervals of 1 sweep or 16 correlation times
        self.indices_sweep = np.argwhere(np.around(self.times - self.times[0], 4) % 1 == 0)[:,0]
        self.indices_16tau = np.argwhere(np.around(self.times - self.times[0], 4) % 16 * self.correlation_time == 0)[:,0]

        print ("Indices marking full sweeps:", self.indices_sweep)
        print (r"Indices marking $16\tau$ blocks:", self.indices_16tau)


    def meanAbsoluteSpin(self):
        '''
        Measure the mean absolute spin.

        :return:
            abs_spins: ndarray of shape (n_sweeps,)
                Mean absolute spin measured in each full sweep of the lattice, i.e. in intervals of length 1.
            mean: float
                Average of the mean absolute spin over all sweeps
            std: float
                Standard deviation of the mean absolute spin over all sweeps, estimated using thermal averaging.
        '''

        abs_spins = np.zeros(len(self.indices_sweep)-1)

        for i in range(len(self.indices_sweep)-1):
            magnetisations = self.magnetisations[self.indices_sweep[i]:self.indices_sweep[i+1]]
            abs_spins[i] = meanAbsoluteSpin(magnetisations, self.size)

        mean = np.mean(abs_spins)
        std = thermalAveragingStandardDeviation(self.times, abs_spins, self.correlation_time)

        return abs_spins, mean, std


    def energyPerSpin(self):
        '''
        Measure the energy per spin.

        :return:
            energies_per_spin: ndarray of shape (n_sweeps,)
                Energies per spin measured in each full sweep of the lattice, i.e. in intervals of length 1.
            mean: float
                Average of the energy per spin over all sweeps
            std: float
                Standard deviation of the energy per spin over all sweeps, estimated using thermal averaging.
        '''

        energies_per_spin = np.zeros(len(self.indices_sweep)-1)

        for i in range(len(self.indices_sweep)-1):
            energies = self.energies[self.indices_sweep[i]:self.indices_sweep[i+1]]
            energies_per_spin[i] = meanEnergyPerSpin(energies, self.size)

        mean = np.mean(energies_per_spin)
        std = thermalAveragingStandardDeviation(self.times, energies_per_spin, self.correlation_time)

        return energies_per_spin, mean, std


    def magneticSusceptibility(self):
        '''
        Measure the magnetic susceptibility of the lattice.

        :return:
            susceptibilities: ndarray of shape (n_blocks,)
                Susceptibility measured in each full block of 16 correlation times
            mean: float
                Average of the magnetic susceptibility over all blocks
            std: float
                Standard deviation of the magnetic susceptibility over all blocks
        '''

        susceptibilities = np.zeros(len(self.indices_16tau)-1)

        for i in range(len(self.indices_16tau)-1):
            magnetisations = self.magnetisations[self.indices_16tau[i]:self.indices_16tau[i+1]]
            susceptibilities[i] = magneticSusceptibility(magnetisations, self.size, self.temperature)

        mean = np.mean(susceptibilities)
        std = np.std(susceptibilities)

        return susceptibilities, mean, std


    def specificHeatPerSpin(self):
        '''
        Measure the specific heat per spin of the lattice.

        :return:
            specific_heats: ndarray of shape (n_blocks,)
                Specific heat measured in each full block of 16 correlation times
            mean: float
                Average of the specific heat over all blocks
            std: float
                Standard deviation of the specific heat over all blocks
        '''

        specific_heats = np.zeros(len(self.indices_16tau)-1)

        for i in range(len(self.indices_16tau)-1):
            energies = self.energies[self.indices_16tau[i]:self.indices_16tau[i+1]]
            specific_heats[i] = specificHeatPerSpin(energies, self.size, self.temperature)

        mean = np.mean(specific_heats)
        std = np.std(specific_heats)

        return specific_heats, mean, std


class ObservablePlotter(Measurer):
    def __init__(self,
                 history=None,
                 times=None,
                 magnetisations=None,
                 energies=None,
                 correlation_time=None,
                 lattice_size=None,
                 temperature=None,
                 usetex=False,
                 figsize=(8,8),
                 dpi=240,
                 titles=False):

        '''
        Class for plotting histograms of the mean absolute spin, the energy per spin, the magnetic susceptibility
        and the specific heat of the lattice for a single simulation.

        :param history: LatticeHistory instance or str or NoneType
                Simulation data. Must be either a LatticeHistory object or a file name. If None, the data must be
                provided through the other arguments of the function.
        :param times: ndarray of shape (n_times,) or NoneType
                Time stamps of the simulation data. If None, argument "history" must be provided. Default is None.
        :param magnetisations: ndarray of shape (n_times,) or NoneType
                Total magnetisation of the lattice at each point in time. If None, argument "history" must be provided.
                Default is None.
        :param energies: ndarray of shape (n_times,) or NoneType
                Total energy of the lattice at each point in time. If None, argument "history" must be provided.
                Default is None.
        :param correlation_time: float or NoneType
                Correlation time of the system. If None, argument "history" must be provided. Default is None.
        :param lattice_size: int or NoneType
                Linear size of the lattice in terms of atoms on each side. If None, argument "history" must be provided.
                Default is None.
        :param temperature: float or NoneType
                Temperature of the system. If None, argument "history" must be provided. Default is None.
        :param usetex: bool
                If True, Latex is used for the figure layout. Default is False.
        :param figsize: tuple of length 2
                Sets the size of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is (8,8).
        :param dpi: int
                Sets the dpi of the figure. Is directly passes on to matplotlib.pyplot.subplots.
                Default is 240.
        :param titles: bool
                If True, adds a title to each subplot. Default is False.
        '''

        super(ObservablePlotter, self).__init__(history, times, magnetisations, energies,
                                                correlation_time, lattice_size, temperature)

        self.abs_spins, self.abs_spin_mean, self.abs_spin_std = self.meanAbsoluteSpin()
        self.energies_per_spin, self.energy_per_spin_mean, self.energy_per_spin_std = self.energyPerSpin()
        self.susceptibilities, self.susceptibility_mean, self.susceptibility_std = self.magneticSusceptibility()
        self.specific_heats, self.specific_heat_mean, self.specific_heat_std = self.specificHeatPerSpin()

        plt.rcParams["text.usetex"] = usetex

        self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi)

        if titles:
            self.ax[0,0].set_title("Mean Absolute Spin")
            self.ax[0,1].set_title("Magnetic Susceptibility")
            self.ax[1,0].set_title("Energy Per Spin")
            self.ax[1,1].set_title("Specific Heat Per Spin")

        self.ax[0,0].set_xlabel(r"$<|m|>$")
        self.ax[0,1].set_xlabel(r"$\chi_M$")
        self.ax[1,0].set_xlabel(r"$E / N^2$")
        self.ax[1,1].set_xlabel(r"$C$")

        for i in range(2):
            for j in range(2):
                self.ax[i,j].set_ylabel(r"Occurrences")
                self.ax[i,j].xaxis.set_minor_locator(AutoMinorLocator(5))
                self.ax[i,j].yaxis.set_minor_locator(AutoMinorLocator(5))

        self.fig.suptitle("Measurements for Temperature $T =$ "+str(np.around(self.temperature,2)))


    def plot(self, bins=10):
        '''
        Plot the historgrams.

        :param bins: int or str
                Sets the number of bins. Is directly passed on to matplotlib.pyplot.hist.
                Default is 10.
        :return:
        '''

        self.ax[0,0].hist(self.abs_spins, bins=bins)
        self.ax[0,0].axvline(self.abs_spin_mean, color="black", ls="--", label="Mean")
        self.ax[0,0].axvline(self.abs_spin_mean-self.abs_spin_std, color="red", ls=":", label="Standard deviation")
        self.ax[0,0].axvline(self.abs_spin_mean+self.abs_spin_std, color="red", ls=":")

        self.ax[0,1].hist(self.susceptibilities, bins=bins)
        self.ax[0,1].axvline(self.susceptibility_mean, color="black", ls="--", label="Mean")
        self.ax[0,1].axvline(self.susceptibility_mean-self.susceptibility_std, color="red", ls=":",
                             label="Standard deviation")
        self.ax[0,1].axvline(self.susceptibility_mean+self.susceptibility_std, color="red", ls=":")
        self.ax[0,1].legend()

        self.ax[1,0].hist(self.energies_per_spin, bins=bins)
        self.ax[1,0].axvline(self.energy_per_spin_mean, color="black", ls="--")
        self.ax[1,0].axvline(self.energy_per_spin_mean-self.energy_per_spin_std, color="red", ls=":")
        self.ax[1,0].axvline(self.energy_per_spin_mean+self.energy_per_spin_std, color="red", ls=":")

        self.ax[1,1].hist(self.specific_heats, bins=bins)
        self.ax[1,1].axvline(self.specific_heat_mean, color="black", ls="--")
        self.ax[1,1].axvline(self.specific_heat_mean-self.specific_heat_std, color="red", ls=":")
        self.ax[1,1].axvline(self.specific_heat_mean+self.specific_heat_std, color="red", ls=":")


    def show(self):
        '''
        Show the figure.

        :return:
        '''

        self.fig.show()


    def save(self, filename, tight=False):
        '''
        Save the figure.

        :param filename: str
                File name to which the figure should be saved.
        :param tight: bool
                Whether or not to set bbox_inches="tight". Default is False.
        :return:
        '''

        if tight:
            self.fig.savefig(filename, bbox_inches="tight")
        else:
            self.fig.savefig(filename)


class DirectoryMeasurer:
    def __init__(self, directory, usetex=False):
        '''
        Conduct measurements of the mean absolute spin, the energy per spin, the magnetic susceptibility
        and the specific heat for all simulation data in a given folder. All simulation data in the folder
        must have the .hdf5 extension.

        :param directory: str
                Directory where the simulation data are stored.
        :param usetex: bool
                If True, uses Latex for the figure layout. Default is False.
        '''

        plt.rcParams["text.usetex"] = usetex

        histories_obj = LatticeHistories(directory)
        histories = histories_obj.histories
        self.n_histories = histories_obj.n_histories

        self.temperatures = np.array([history.temperature for history in histories])
        print ("Shape of temperatures: {}".format(self.temperatures.shape))
        print ("Number of simulations: {}".format(self.n_histories))
        measurers = [Measurer(history) for history in histories]

        # save all means and standard deviations
        self.corr_times = np.zeros(self.n_histories)
        self.abs_spin_means = np.zeros(self.n_histories)
        self.abs_spin_stds = np.zeros(self.n_histories)
        self.energy_per_spin_means = np.zeros(self.n_histories)
        self.energy_per_spin_stds = np.zeros(self.n_histories)
        self.susc_means = np.zeros(self.n_histories)
        self.susc_stds = np.zeros(self.n_histories)
        self.spec_heat_means = np.zeros(self.n_histories)
        self.spec_heat_stds = np.zeros(self.n_histories)

        for i, (history, measurer) in enumerate(zip(histories, measurers)):
            self.corr_times[i] = history.correlation_time
            _, self.abs_spin_means[i], self.abs_spin_stds[i] = measurer.meanAbsoluteSpin()
            _, self.energy_per_spin_means[i], self.energy_per_spin_stds[i] = measurer.energyPerSpin()
            _, self.susc_means[i], self.susc_stds[i] = measurer.magneticSusceptibility()
            _, self.spec_heat_means[i], self.spec_heat_stds[i] = measurer.specificHeatPerSpin()
            print ("Measurements finished for T = {}".format(history.temperature))


    def plotCorrelationTimes(self, figsize=(6,4), dpi=240, mec="black"):
        '''
        Plot the correlation times as a function of temperature.

        :param figsize: tuple of length 2
                Sets the size of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is (6,4).
        :param dpi: int
                Sets the dpi of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is 240.
        :param mec: str
                Marker edge color, directly passed on to matplotlib.pyplot.plot. Default is "black".
        :return:
            fig: Figure object
            ax: Axes object
        '''

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(self.temperatures, self.corr_times, marker="o", ls="", mec=mec)
        ax.set_xlabel(r"Temperature $T$")
        ax.set_ylabel(r"Correlation time $\tau$")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        fig.suptitle("Correlation Time vs. Temperature")
        fig.show()

        return fig, ax


    def plotAbsSpins(self, figsize=(6,4), dpi=240, capsize=5, fmt="s", mec="black"):
        '''
        Plot the mean absolute spin as a function of temperature.

        :param figsize: tuple of length 2
                Sets the size of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is (6,4).
        :param dpi: int
                Sets the dpi of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is 240.
        :param capsize: float
                Sets the cap size of the error bars, directly passed on to matplotlib.pyplot.errorbar.
                Default is 5.
        :param mec: str
                Marker edge color, directly passed on to matplotlib.pyplot.plot. Default is "black".
        :return:
            fig: Figure object
            ax: Axes object
        '''

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.errorbar(self.temperatures, self.abs_spin_means, yerr=self.abs_spin_stds,
                    capsize=capsize, fmt=fmt, mec=mec)
        ax.set_xlabel(r"Temperature $T$")
        ax.set_ylabel(r"Mean absolute spin $<|m|>$")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        fig.suptitle("Mean Absolute Spin vs. Temperature")
        fig.show()

        return fig, ax


    def plotEnergyPerSpin(self, figsize=(6,4), dpi=240, capsize=5, fmt="s", mec="black"):
        '''
        Plot the energy per spin as a function of temperature.

        :param figsize: tuple of length 2
                Sets the size of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is (6,4).
        :param dpi: int
                Sets the dpi of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is 240.
        :param capsize: float
                Sets the cap size of the error bars, directly passed on to matplotlib.pyplot.errorbar.
                Default is 5.
        :param mec: str
                Marker edge color, directly passed on to matplotlib.pyplot.plot. Default is "black".
        :return:
            fig: Figure object
            ax: Axes object
        '''

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.errorbar(self.temperatures, self.energy_per_spin_means, yerr=self.energy_per_spin_stds,
                    capsize=capsize, fmt=fmt, mec=mec)
        ax.set_xlabel(r"Temperature $T$")
        ax.set_ylabel(r"Energy per spin $E / N^2$")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        fig.suptitle("Energy Per Spin vs. Temperature")
        fig.show()

        return fig, ax


    def plotMagneticSusceptibility(self, figsize=(6,4), dpi=240, capsize=5, fmt="s", mec="black"):
        '''
        Plot the magnetic susceptibility as a function of temperature.

        :param figsize: tuple of length 2
                Sets the size of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is (6,4).
        :param dpi: int
                Sets the dpi of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is 240.
        :param capsize: float
                Sets the cap size of the error bars, directly passed on to matplotlib.pyplot.errorbar.
                Default is 5.
        :param mec: str
                Marker edge color, directly passed on to matplotlib.pyplot.plot. Default is "black".
        :return:
            fig: Figure object
            ax: Axes object
        '''

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.errorbar(self.temperatures, self.susc_means, yerr=self.susc_stds,
                    capsize=capsize, fmt=fmt, mec=mec)
        ax.set_xlabel(r"Temperature $T$")
        ax.set_ylabel(r"Magnetic susceptibility $\chi_M$")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        fig.suptitle("Magnetic Susceptibility vs. Temperature")
        fig.show()

        return fig, ax


    def plotSpecificHeat(self, figsize=(6,4), dpi=240, capsize=5, fmt="s", mec="black"):
        '''
        Plot the specific heat as a function of temperature.

        :param figsize: tuple of length 2
                Sets the size of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is (6,4).
        :param dpi: int
                Sets the dpi of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is 240.
        :param capsize: float
                Sets the cap size of the error bars, directly passed on to matplotlib.pyplot.errorbar.
                Default is 5.
        :param mec: str
                Marker edge color, directly passed on to matplotlib.pyplot.plot. Default is "black".
        :return:
            fig: Figure object
            ax: Axes object
        '''

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.errorbar(self.temperatures, self.spec_heat_means, yerr=self.spec_heat_stds,
                    capsize=capsize, fmt=fmt, mec=mec)
        ax.set_xlabel(r"Temperature $T$")
        ax.set_ylabel(r"Specific heat per spin $C$")
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        fig.suptitle("Specific Heat Per Spin vs. Temperature")
        fig.show()

        return fig, ax


    def plotAll(self, figsize=(6,4), dpi=240, capsize=5, fmt="s", mec="black"):
        '''
        Make plots of correlation time, mean absolute spin, energy per spin, magnetic susceptibility and
        specific heat, all as a function of temperature.

        :param figsize: tuple of length 2
                Sets the size of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is (6,4).
        :param dpi: int
                Sets the dpi of the figure. Is directly passed on to matplotlib.pyplot.subplots.
                Default is 240.
        :param capsize: float
                Sets the cap size of the error bars, directly passed on to matplotlib.pyplot.errorbar.
                Default is 5.
        :param mec: str
                Marker edge color, directly passed on to matplotlib.pyplot.plot. Default is "black".
        :return:
            [fig_tau, ax_tau]: Figure and Axes objects for the correlation time
            [fig_abs_spin, ax_abs_spin]: Figure and Axes objects for the mean absolute spin
            [fig_energy, ax_energy]: Figure and Axes objects for the energy per spin
            [fig_susc, ax_susc]: Figure and Axes objects for the magnetic susceptibility
            [fig_heat, ax_heat]: Figure and Axes objects for the specific heat
        '''

        fig_tau, ax_tau = self.plotCorrelationTimes(figsize, dpi, mec=mec)
        fig_abs_spin, ax_abs_spin = self.plotAbsSpins(figsize, dpi, capsize, fmt, mec=mec)
        fig_energy, ax_energy = self.plotEnergyPerSpin(figsize, dpi, capsize, fmt, mec=mec)
        fig_susc, ax_susc = self.plotMagneticSusceptibility(figsize, dpi, capsize, fmt, mec=mec)
        fig_heat, ax_heat = self.plotSpecificHeat(figsize, dpi, capsize, fmt, mec=mec)

        return [[fig_tau, ax_tau], [fig_abs_spin, ax_abs_spin], [fig_energy, ax_energy],
                [fig_susc, ax_susc], [fig_heat, ax_heat]]