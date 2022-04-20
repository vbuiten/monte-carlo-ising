'''File to run for doing a series of simulations at temperatures from 1.0 to 4.0.'''

from framework.lattice import Lattice
from simulation.simulator import Simulator
from simulation.utils import normalisedCorrelationFunction, correlationTimeFromCorrelationFunction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
import os

#temperatures = np.arange(1., 4.1, .2)
temperatures = np.arange(2.6, 4.1, .2)
N = 50
basepath = "/net/vdesk/data2/buiten/COP/"
dirname = "ising-sim-data-N"+str(N)
savedir = os.path.join(basepath, dirname)

if not os.path.isdir(savedir):
    os.mkdir(savedir)

savedir = savedir + "/"

print ("Saving directory: {}".format(savedir))

for i, temp in enumerate(temperatures):

    temp_str = str(np.around(temp,3))

    lattice = Lattice(N)
    lattice.spins = np.ones((N,N))
    sim = Simulator(lattice, temp)

    # equilibrate the lattice
    times_eq, magnetisations_eq = sim.equilibrate(reject_rate_threshold=1e-3)

    fig, ax = plt.subplots(figsize=(6,4), dpi=320)
    ax.plot(times_eq, magnetisations_eq / N**2, lw=.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Magnetisation per spin")
    fig.suptitle("Magnetisation During Equilibration")
    ax.set_title("$T =$" + temp_str)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="major")
    fig.savefig(savedir+"temp"+temp_str+"equilibration.png")

    # do a test run for estimating the correlation time
    test_time_end = 400
    times_test, magnetisations_test, energies_test = sim.evolve(test_time_end)

    fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=240)
    ax2.plot(times_test, magnetisations_test / N ** 2, lw=.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Magnetisation per spin")
    fig2.suptitle("Magnetisation During Test Run")
    ax2.set_title("$T =$" + temp_str)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.grid(which="major")
    fig2.savefig(savedir+"temp"+temp_str+"test-run.png")

    # compute the correlation function and time
    norm_corr_func = normalisedCorrelationFunction(times_test, magnetisations_test)
    corr_time = correlationTimeFromCorrelationFunction(times_test, norm_corr_func)
    print ("Estimated correlation time for T = {}: {}".format(temp, corr_time))
    before80percent = times_test[:-1] < 0.8*test_time_end

    fig3, ax3 = plt.subplots(figsize=(7, 5), dpi=240)
    ax3.plot(times_test[:-1][before80percent], norm_corr_func[before80percent], lw=.5)
    ax3.set_xlabel("Time")
    ax3.set_ylabel(r"$\chi(t) / \chi(0)$")
    ax3.set_title("$T =$" + temp_str)
    fig3.suptitle("Correlation Function During Test Run")
    ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax3.grid(which="major")
    fig3.savefig(savedir+"temp"+temp_str+"correlation-function.png")

    # now run a long simulation and save the data
    n_blocks = 100
    _, _, _ = sim.evolve(sim.time + n_blocks * 16 * corr_time,
                         savefile=savedir+"temp"+temp_str+"data.hdf5", correlation_time=corr_time)

    print ("Completed for T = {}".format(temp))