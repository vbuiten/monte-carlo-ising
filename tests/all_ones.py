'''Algorithm test: what happens if we start in a state where each spin is +1?'''
from framework.lattice import Lattice
from simulation.simulator import Simulator
from simulation.utils import normalisedCorrelationFunction
from analysis.visualisation import LatticeVisual
import matplotlib.pyplot as plt
import numpy as np

temp = 1.0
N = 50

path = r"C:\\Users\\victo\\Documents\\Uni\\COP\\"
file = path+"ising-test.hdf5"

lattice = Lattice(N)
lattice.spins = np.ones((N,N))

vis = LatticeVisual(lattice)
vis.ax.set_title("Initial Configuration")
vis.show()

sim = Simulator(lattice, temp)
times, magnetisations, energies = sim.evolve(300, None)

after100sweeps = (times > 25)
corr_func_energies = normalisedCorrelationFunction(times[after100sweeps], energies[after100sweeps])

vis.update()
vis.ax.set_title("Final Configuration")
vis.show()

fig2, ax2 = plt.subplots(figsize=(7,10), dpi=240, nrows=2, sharex=True)
ax2[0].plot(times, energies, lw=.5)
ax2[1].plot(times, magnetisations, lw=.5)
ax2[1].set_xlabel("Times")

ax2[0].set_ylabel("Energy per spin")
ax2[1].set_ylabel("Magnetisation per spin")

fig2.show()

fig3, ax3 = plt.subplots(figsize=(7,5), dpi=240)
ax3.plot(times[after100sweeps][:-1], corr_func_energies, lw=.5)
ax3.set_xlabel("Times")
ax3.set_ylabel("Energy Correlation Function")
fig3.show()

# compute correlation time
timestep = times[1] - times[0]
corr_time = timestep * np.sum(corr_func_energies)
print ("Correlation time: {}".format(corr_time))