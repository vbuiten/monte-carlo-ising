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

# compute the correlation function
after50sweeps = history.times > 50.
corr_func_energy = correlationFunction(history.times[after50sweeps], history.energies[after50sweeps]/history.size**2)

fig, ax = plt.subplots(figsize=(7,5), dpi=240)
ax.plot(history.times[after50sweeps][:-1], corr_func_energy, lw=1.)
ax.set_xlabel("Times")
ax.set_ylabel("Energy Correlation Function")
fig.show()