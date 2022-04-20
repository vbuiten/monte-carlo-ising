'''Code for loading the data of a previously-run simulation.'''

from data.load import LatticeHistory
from analysis.utils import *
from simulation.utils import correlationFunction, normalisedCorrelationFunction
import numpy as np
import matplotlib.pyplot as plt

#path = r"/net/vdesk/data2/buiten/COP/"
path = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
file = path + "ising-test.hdf5"
history = LatticeHistory(file)

# plot the energies and magnetisations per spin
fig2, ax2 = plt.subplots(figsize=(7,10), dpi=240, nrows=2, sharex=True)
ax2[0].plot(history.times, history.energies/history.size**2, lw=.5)
ax2[1].plot(history.times, history.magnetisations/history.size**2, lw=.5)
ax2[1].set_xlabel("Times")

ax2[0].set_ylabel("Energy per spin")
ax2[1].set_ylabel("Magnetisation per spin")

for ax in ax2:
    ax.grid()

fig2.show()

# compute the correlation functions for energy and magnetisation
after30sweeps = history.times > 30.
before80sweeps = history.times < 80.
goodtimes = after30sweeps & before80sweeps

corr_func_energy = normalisedCorrelationFunction(history.times[after30sweeps], history.energies[after30sweeps]/history.size**2)
corr_func_magnetisation = normalisedCorrelationFunction(history.times[after30sweeps],
                                              history.magnetisations[after30sweeps]/history.size**2)

goodcorrtimes = history.times[after30sweeps][:-1] < 80.

fig, ax = plt.subplots(figsize=(7,10), dpi=240, nrows=2, sharex=True)
ax[0].plot(history.times[goodtimes], corr_func_energy[goodcorrtimes], lw=1.)
ax[1].plot(history.times[goodtimes], corr_func_magnetisation[goodcorrtimes], lw=1.)
ax[1].set_xlabel("Times")
ax[0].set_ylabel("Normalised Energy Correlation Function")
ax[1].set_ylabel("Normalised Magnetisation Correlation Function")

for axis in ax:
    axis.grid()

fig.show()