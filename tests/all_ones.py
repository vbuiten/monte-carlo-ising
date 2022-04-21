'''Algorithm test: what happens if we start in a state where each spin is +1?'''
from framework.lattice import Lattice
from simulation.simulator import Simulator
from simulation.utils import correlationFunction
from analysis.visualisation import LatticeVisual
import matplotlib.pyplot as plt
import numpy as np

temp = 2.0
N = 50

path = r"C:\\Users\\victo\\Documents\\Uni\\COP\\"
file = path+"ising-test.hdf5"

lattice = Lattice(N)
lattice.spins = np.ones((N,N))

vis = LatticeVisual(lattice)
vis.ax.set_title("Initial Configuration")
vis.show()

sim = Simulator(lattice, temp)
sim.equilibrate(reject_rate_threshold=5e-3)
times, magnetisations, energies = sim.evolve(100, None)

corr_func_magnetisation = correlationFunction(times, magnetisations)

vis.update()
vis.ax.set_title("Final Configuration")
vis.show()

fig2, ax2 = plt.subplots(figsize=(7,10), dpi=240, nrows=2, sharex=True)
ax2[0].plot(times, energies, lw=.5)
ax2[1].plot(times, magnetisations, lw=.5)
ax2[1].set_xlabel("Times")

ax2[0].set_ylabel("Energy")
ax2[1].set_ylabel("Magnetisation")

fig2.show()

fig3, ax3 = plt.subplots(figsize=(7,5), dpi=240)
ax3.plot(times[:-1], corr_func_magnetisation, lw=.5)
ax3.set_xlabel("Times")
ax3.set_ylabel("Energy Correlation Function")
fig3.show()
# compute correlation time

timestep = times[1] - times[0]
positive = corr_func_magnetisation > 0
corr_time = timestep * np.sum(corr_func_magnetisation/corr_func_magnetisation[0])
print ("Correlation time: {}".format(corr_time))