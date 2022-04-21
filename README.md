# monte-carlo-ising
Computational Physics 2022 Project 2: Studying the 2D Ising model with Monte Carlo methods

Author: Victorine Buiten (buiten@strw.leidenuniv.nl)
Date last updated: 21/04/2022

## What's in this repository?

This repository contains code for simulating a two-dimensional ferromagnetic material through the 2D Ising model.
The code is structured as follows:

* _framework_ contains the code required for the basic set-up of the lattice, and for the system of natural units used.
    Most notably, the Lattice class is essential for initialising the system.
* _simulation_ contains the code used for evolving the lattice. In particular, the Simulator class is used for both
    equilibration and for sampling equilibrium configurations. It also has a built-in data storage functionality.
* _data_ contains the classes LatticeHistory and LatticeHistories, which are the base classes used for loading
    simulation data from the user's disk.
* _analysis_ contains various classes for analysing and visualising the simulation results, including more elaborate
    measurements of quantities of interest.
* _final_pipeline_ contains the files that were run for performing a series of simulations at a wide range of temperatures,
    saving the data, and making the relevant plots for the investigation of the ferromagnetic behaviour of the lattice.
* _tests_ contains various test files that were run for testing the code as it was developed. They may be useful for the user
    to check for examples of how to work with the code.
    
## What are the prerequisites?

The simulations run exclusively on NumPy and Numba. For the data storage, h5py is required. For plotting the results,
the user will need the matplotlib plotting library.
    
## How to run the code?

The first step to running a simulation, is to set up the initial lattice with linear size N.
This can be done with the Lattice class:

===============

import numpy as np

from framework.lattice import Lattice

N = 50

lattice = Lattice(N)

lattice.spins = np.ones((N,N))

===============

The equilibration and evolution of the lattice are done by the Simulator class:

===============

from simulation.simulator import Simulator

from simulation.utils import normalisedCorrelationFunction, correlationTimeFromCorrelationFunction

temperature = 1.5

sim = Simulator(lattice, temperature)

times_eq, magnetisations_eq = sim.equilibrate(reject_rate_threshold=1e-3)

===============

The user may want to plot the magnetisation as a function of time to check the equilibration process:

===============

fig, ax = plt.subplots(figsize=(6,4), dpi=240)

ax.plot(times_eq, magnetisations_eq / N**2, lw=.5)

ax.set_xlabel("Time")

ax.set_ylabel("Magnetisation per spin")

fig.suptitle("Magnetisation During Equilibration")

ax.set_title("$T =$"+str(temperature))

ax.xaxis.set_minor_locator(AutoMinorLocator(5))

ax.yaxis.set_minor_locator(AutoMinorLocator(5))

ax.grid(which="major")

fig.show()

===============

After equilibration is complete, we want to compute the correlation function and correlation time:

===============

test_time_end = 400

times_test, magnetisations_test, energies_test = sim.evolve(test_time_end)

fig2, ax2 = plt.subplots(figsize=(6,4), dpi=240)

ax2.plot(times_test, magnetisations_test / N**2, lw=.5)

ax2.set_xlabel("Time")

ax2.set_ylabel("Magnetisation per spin")

fig2.suptitle("Magnetisation During Test Run")

ax2.set_title("$T =$"+str(temperature))

ax2.xaxis.set_minor_locator(AutoMinorLocator(5))

ax2.yaxis.set_minor_locator(AutoMinorLocator(5))

ax2.grid(which="major")

fig2.show()

norm_corr_func = normalisedCorrelationFunction(times_test, magnetisations_test/N**2)

corr_time = correlationTimeFromCorrelationFunction(times_test, norm_corr_func, negative_stop=True)

print ("Estimated correlation time:", corr_time)

before30percent = times_test[:-1] < 0.3*test_time_end

fig3, ax3 = plt.subplots(figsize=(6,4), dpi=240)

ax3.plot(times_test[:-1][before30percent], norm_corr_func[before30percent], lw=.5)

ax3.set_xlabel("Time")

ax3.set_ylabel(r"$\chi(t) / \chi(0)$")

ax3.set_title("$T =$"+str(temp))

fig3.suptitle("Correlation Function During Test Run")

ax3.xaxis.set_minor_locator(AutoMinorLocator(5))

ax3.yaxis.set_minor_locator(AutoMinorLocator(5))

ax3.grid(which="major")

fig3.show()

===============

Finally, we can do some measurements and plot the resulting histograms:

===============

n_it = 30

times, magnetisations, energies = sim.evolve(sim.time+n_it*16*corr_time)

plotter = ObservablePlotter(None, times, magnetisations, energies, corr_time, N, temp, usetex=True)

plotter.plot()

plotter.show()
