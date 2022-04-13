'''Code for loading the data of a previously-run simulation.'''

from data.load import LatticeHistory
from analysis.utils import *
from simulation.utils import correlationFunction
import numpy as np
import matplotlib.pyplot as plt

path = r"/net/vdesk/data2/buiten/COP/"
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
after50sweeps = history.times > 20.
before100sweeps = history.times < 100.
goodtimes = after50sweeps & before100sweeps

corr_func_energy = correlationFunction(history.times[goodtimes], history.energies[goodtimes]/history.size**2)
corr_func_magnetisation = correlationFunction(history.times[goodtimes],
                                              history.magnetisations[goodtimes]/history.size**2)

fig, ax = plt.subplots(figsize=(7,10), dpi=240, nrows=2, sharex=True)
ax[0].plot(history.times[goodtimes][:-1], corr_func_energy, lw=1.)
ax[1].plot(history.times[goodtimes][:-1], corr_func_magnetisation, lw=1.)
ax[1].set_xlabel("Times")
ax[0].set_ylabel("Energy Correlation Function")
ax[1].set_ylabel("Magnetisation Correlation Function")

for axis in ax:
    axis.grid()

fig.show()