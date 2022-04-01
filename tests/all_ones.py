'''Algorithm test: what happens if we start in a state where each spin is +1?'''
from framework.lattice import Lattice
from simulation.simulator import Simulator
from analysis.visualisation import LatticeVisual
import matplotlib.pyplot as plt
import numpy as np

temp = 5.0
N = 10

lattice = Lattice(N)
lattice.spins = np.ones((N,N))

vis = LatticeVisual(lattice)
vis.ax.set_title("Initial Configuration")
vis.show()

sim = Simulator(lattice, temp)
times, magnetisations, energies = sim.evolve(100)

fig, ax = plt.subplots(figsize=(7,5), dpi=240)
ax.plot(times, energies, lw=.5)
ax.set_xlabel("Time")
ax.set_ylabel("Energy")
fig.show()

vis.update()
vis.ax.set_title("Final Configuration")
vis.show()