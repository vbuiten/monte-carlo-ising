'''Code for measuring observables from simulation data.'''

import numpy as np
from analysis.utils import *
from data.load import LatticeHistory

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
        self.indices_sweep = np.argwhere((times - times[0]) % 1 == 0)[:,0]
        self.indices_16tau = np.argwhere((times - times[0]) % 16 * self.correlation_time == 0)[:,0]


    def meanAbsoluteSpin(self):

        abs_spins = np.zeros(len(self.indices_sweep)-1)

        for i in range(len(self.indices_sweep)-1):
            magnetisations = self.magnetisations[self.indices_sweep[i]:self.indices_sweep[i+1]]
            abs_spins[i] = meanAbsoluteSpin(magnetisations, self.size)

        mean = np.mean(abs_spins)
        std = thermalAveragingStandardDeviation(self.times, abs_spins)

        return abs_spins, mean, std


    def energyPerSpin(self):

        energies_per_spin = np.zeros(len(self.indices_sweep)-1)

        for i in range(len(self.indices_sweep)-1):
            energies = self.energies[self.indices_sweep[i]:self.indices_sweep[i+1]]
            energies_per_spin[i] = meanEnergyPerSpin(energies, self.size)

        mean = np.mean(energies_per_spin)
        std = thermalAveragingStandardDeviation(self.times, energies_per_spin)

        return energies_per_spin, mean, std


    def magneticSusceptibility(self):

        susceptibilities = np.zeros(len(self.indices_16tau)-1)

        for i in range(len(self.indices_16tau)-1):
            magnetisations = self.magnetisations[self.indices_16tau[i]:self.indices_16tau[i+1]]
            susceptibilities[i] = magneticSusceptibility(magnetisations, self.size)

        mean = np.mean(susceptibilities)
        std = np.std(susceptibilities)

        return susceptibilities, mean, std


    def specificHeatPerSpin(self):

        specific_heats = np.zeros(len(self.indices_16tau)-1)

        for i in range(len(self.indices_16tau)-1):
            energies = self.energies[self.indices_16tau[i]:self.indices_16tau[i+1]]
            specific_heats[i] = specificHeatPerSpin(energies, self.size)

        mean = np.mean(specific_heats)
        std = np.std(specific_heats)

        return specific_heats, mean, std