'''Code for measuring observables from simulation data.'''

import numpy as np
from analysis.utils import *
from data.load import LatticeHistory
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
        self.indices_sweep = np.argwhere(np.around(times - times[0], 4) % 1 == 0)[:,0]
        self.indices_16tau = np.argwhere(np.around(times - times[0], 4) % 16 * self.correlation_time == 0)[:,0]

        print ("Indices marking full sweeps:", self.indices_sweep)
        print (r"Indices marking $16\tau$ blocks:", self.indices_16tau)


    def meanAbsoluteSpin(self):

        abs_spins = np.zeros(len(self.indices_sweep)-1)

        for i in range(len(self.indices_sweep)-1):
            magnetisations = self.magnetisations[self.indices_sweep[i]:self.indices_sweep[i+1]]
            abs_spins[i] = meanAbsoluteSpin(magnetisations, self.size)

        mean = np.mean(abs_spins)
        std = thermalAveragingStandardDeviation(self.times, abs_spins, self.correlation_time)

        return abs_spins, mean, std


    def energyPerSpin(self):

        energies_per_spin = np.zeros(len(self.indices_sweep)-1)

        for i in range(len(self.indices_sweep)-1):
            energies = self.energies[self.indices_sweep[i]:self.indices_sweep[i+1]]
            energies_per_spin[i] = meanEnergyPerSpin(energies, self.size)

        mean = np.mean(energies_per_spin)
        std = thermalAveragingStandardDeviation(self.times, energies_per_spin, self.correlation_time)

        return energies_per_spin, mean, std


    def magneticSusceptibility(self):

        susceptibilities = np.zeros(len(self.indices_16tau)-1)

        for i in range(len(self.indices_16tau)-1):
            magnetisations = self.magnetisations[self.indices_16tau[i]:self.indices_16tau[i+1]]
            susceptibilities[i] = magneticSusceptibility(magnetisations, self.size, self.temperature)

        mean = np.mean(susceptibilities)
        std = np.std(susceptibilities)

        return susceptibilities, mean, std


    def specificHeatPerSpin(self):

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
                 dpi=240):

        super(ObservablePlotter, self).__init__(history, times, magnetisations, energies,
                                                correlation_time, lattice_size, temperature)

        self.abs_spins, self.abs_spin_mean, self.abs_spin_std = self.meanAbsoluteSpin()
        self.energies_per_spin, self.energy_per_spin_mean, self.energy_per_spin_std = self.energyPerSpin()
        self.susceptibilities, self.susceptibility_mean, self.susceptibility_std = self.magneticSusceptibility()
        self.specific_heats, self.specific_heat_mean, self.specific_heat_std = self.specificHeatPerSpin()

        plt.rcParams["text.usetex"] = usetex

        self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi, sharey=True)

        self.ax[0,0].set_title("Mean Absolute Spin")
        self.ax[0,1].set_title("Magnetic Susceptibility")
        self.ax[1,0].set_title("Energy Per Spin")
        self.ax[1,1].set_title("Specific Heat Per Spin")

        self.ax[0,0].set_xlabel(r"$<|m|>$")
        self.ax[0,1].set_xlabel(r"$\chi_M$")
        self.ax[1,0].set_xlabel(r"$E / N^2$")
        self.ax[1,1].set_xlabel(r"$C$")

        for i in range(2):
            self.ax[i,0].set_ylabel(r"Occurrences")

            for j in range(2):
                self.ax[i,j].xaxis.set_minor_locator(AutoMinorLocator(5))
                self.ax[i,j].yaxis.set_minor_locator(AutoMinorLocator(5))

        self.fig.suptitle("Measurements for Temperature $T =$ "+str(np.around(self.temperature,2)))


    def plot(self, bins=10):

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

        self.fig.show()


    def save(self, filename, tight=False):

        if tight:
            self.fig.savefig(filename, bbox_inches="tight")
        else:
            self.fig.savefig(filename)