from framework.lattice import Lattice
from simulation.simulator import Simulator
from simulation.utils import normalisedCorrelationFunction, correlationTimeFromCorrelationFunction
from analysis.utils import magneticSusceptibility, specificHeatPerSpin
from analysis.observables import Measurer, ObservablePlotter
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13
plt.rcParams["text.usetex"] = True

temp = 1.5
N = 50

lattice = Lattice(N)
lattice.spins = np.ones((N,N))
sim = Simulator(lattice, temp)
times_eq, magnetisations_eq = sim.equilibrate(reject_rate_threshold=1e-3)

# plot the magnetisation throughout equilibration
fig, ax = plt.subplots(figsize=(6,4), dpi=240)
ax.plot(times_eq, magnetisations_eq / N**2, lw=.5)
ax.set_xlabel("Time")
ax.set_ylabel("Magnetisation per spin")
fig.suptitle("Magnetisation During Equilibration")
ax.set_title("$T =$"+str(temp))
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(which="major")
fig.show()

# now do a test run for estimating the correlation time
test_time_end = 400
times_test, magnetisations_test, energies_test = sim.evolve(test_time_end)

fig2, ax2 = plt.subplots(figsize=(6,4), dpi=240)
ax2.plot(times_test, magnetisations_test / N**2, lw=.5)
ax2.set_xlabel("Time")
ax2.set_ylabel("Magnetisation per spin")
fig2.suptitle("Magnetisation During Test Run")
ax2.set_title("$T =$"+str(temp))
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.grid(which="major")
fig2.show()

# compute the correlation function
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

# now run a long simulation with block measurements
n_it = 30
times, magnetisations, energies = sim.evolve(sim.time+n_it*16*corr_time)

plotter = ObservablePlotter(None, times, magnetisations, energies, corr_time, N, temp, usetex=True)
plotter.plot()
plotter.show()