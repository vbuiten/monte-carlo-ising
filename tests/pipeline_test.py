from framework.lattice import Lattice
from simulation.simulator import Simulator
from simulation.utils import normalisedCorrelationFunction, correlationTimeFromCorrelationFunction
from analysis.utils import magneticSusceptibility, specificHeatPerSpin
from analysis.observables import Measurer, ObservablePlotter
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
plt.rcParams["font.family"] = "serif"

temp = 4.0
N = 50

lattice = Lattice(N)
lattice.spins = np.ones((N,N))
sim = Simulator(lattice, temp)
times_eq, magnetisations_eq = sim.equilibrate(reject_rate_threshold=1e-3)

# plot the magnetisation throughout equilibration
fig, ax = plt.subplots(figsize=(7,5), dpi=240)
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
test_time_end = 300
times_test, magnetisations_test, energies_test = sim.evolve(test_time_end)

fig2, ax2 = plt.subplots(figsize=(7,5), dpi=240)
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
before80percent = times_test[:-1] < 0.8*test_time_end

fig3, ax3 = plt.subplots(figsize=(7,5), dpi=240)
ax3.plot(times_test[:-1][before80percent], norm_corr_func[before80percent], lw=.5)
ax3.set_xlabel("Time")
ax3.set_ylabel(r"$\chi(t) / \chi(0)$")
ax3.set_title("$T =$"+str(temp))
fig3.suptitle("Correlation Function During Test Run")
ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
ax3.grid(which="major")
fig3.show()

# now run a long simulation with block measurements
n_it = 100
times, magnetisations, energies = sim.evolve(sim.time+n_it*16*corr_time)

plotter = ObservablePlotter(None, times, magnetisations, energies, corr_time, N, temp)
plotter.plot()
plotter.show()

'''
measurer = Measurer(None, times, magnetisations, energies, corr_time, N, temp)
abs_spins, mean_abs_spin, std_abs_spin = measurer.meanAbsoluteSpin()
energies_per_spin, mean_energy_per_spin, std_energy_per_spin = measurer.energyPerSpin()
susceptibilities, mean_susceptibility, std_susceptibility = measurer.magneticSusceptibility()
specific_heats, mean_specific_heat, std_specific_heat = measurer.specificHeatPerSpin()

fig4, ax4 = plt.subplots(figsize=(10,10), dpi=240, nrows=2, ncols=2, sharey=True)
ax4[0,0].hist(susceptibilities, bins="sqrt")
ax4[0,0].axvline(mean_susceptibility, color="black", ls="--", label="Mean")
ax4[0,0].axvline(mean_susceptibility-std_susceptibility, color="black", ls=":", label="Standard deviation")
ax4[0,0].axvline(mean_susceptibility+std_susceptibility, color="black", ls=":")
ax4[0,0].set_xlabel(r"$\chi_M$")
ax4[0,0].set_title("Magnetic Susceptibility")
ax4[0,0].set_ylabel("Occurrences")

ax4[1,0].hist(specific_heats, bins="sqrt")
ax4[1,0].axvline(mean_specific_heat, color="black", ls="--", label="Mean")
ax4[1,0].axvline(mean_specific_heat-std_specific_heat, color="black", ls=":", label="Standard deviation")
ax4[1,0].axvline(mean_specific_heat+std_specific_heat, color="black", ls=":")
ax4[1,0].set_xlabel(r"$C$")
ax4[1,0].set_title("Specific Heat Per Spin")
ax4[1,0].set_ylabel("Occurrences")

ax4[0,1].hist(abs_spins, bins="sqrt")
ax4[0,1].axvline(mean_abs_spin, color="black", ls="--", label="Mean")
ax4[0,1].axvline(mean_abs_spin-std_abs_spin, color="black", ls=":", label="Standard deviation")
ax4[0,1].axvline(mean_abs_spin+std_abs_spin, color="black", ls=":")
ax4[0,1].set_xlabel(r"$<|m|>$")
ax4[0,1].set_title("Mean Absolute Spin")
ax4[0,1].legend()

ax4[1,1].hist(energies_per_spin, bins="sqrt")
ax4[1,1].axvline(mean_energy_per_spin, color="black", ls="--", label="Mean")
ax4[1,1].axvline(mean_energy_per_spin-std_abs_spin, color="black", ls=":", label="Standard deviation")
ax4[1,1].axvline(mean_energy_per_spin+std_abs_spin, color="black", ls=":")
ax4[1,1].set_xlabel(r"$E / N^2$")
ax4[1,1].set_title("Energy Per Spin")

fig4.suptitle("Measurements for $T =$"+str(temp))
fig4.show()
'''