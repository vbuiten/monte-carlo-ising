'''Algorithm test: what happens if we start in a state where each spin is +1?'''
from framework.lattice import Lattice
from simulation.simulator import Simulator
from analysis.visualisation import LatticeVisual
import matplotlib.pyplot as plt
import numpy as np

temp = 2.2
N = 50

path = r"C:\\Users\\victo\\Documents\\Uni\\COP\\"
file = path+"ising-test.hdf5"

lattice = Lattice(N)
lattice.spins = np.ones((N,N))

vis = LatticeVisual(lattice)
vis.ax.set_title("Initial Configuration")
vis.show()

sim = Simulator(lattice, temp)
times, magnetisations, energies = sim.evolve(1000, None)

fig, ax = plt.subplots(figsize=(7,5), dpi=240)
ax.plot(times, energies, lw=.5)
ax.set_xlabel("Time")
ax.set_ylabel("Energy")
fig.show()

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