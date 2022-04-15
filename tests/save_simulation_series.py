'''Code for running a series of simulations and saving its data.'''
from framework.lattice import Lattice
from simulation.simulator import Simulator
import numpy as np

temps = np.arange(1.0, 4.1, 0.2)
N = 50

#basepath = r"/net/vdesk/data2/buiten/COP/"
basepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
path = basepath + "ising-series\\"

for i, temp in enumerate(temps):

    file = path + "temp{}.hdf5".format(np.around(temp,2))
    lattice = Lattice(N)
    lattice.spins = np.ones((N,N))

    sim = Simulator(lattice, temp)
    times, magnetisations, energies = sim.evolve(100, file)

    print ("Finished simulation {}".format(i+1))