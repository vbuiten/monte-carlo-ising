from framework.lattice import Lattice
from simulation.simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

N = 50
temp = 4.0

lattice = Lattice(N)
lattice.spins = np.ones((N,N))

sim = Simulator(lattice, temp)
times, magnetisations = sim.equilibrate(sweeps=10, reject_rate_threshold=1e-3)

fig, ax = plt.subplots(figsize=(7,5), dpi=240)
ax.plot(times, magnetisations/N**2, lw=.5)
ax.set_xlabel("Time")
ax.set_ylabel("Magnetisation per spin")
ax.grid()
fig.show()