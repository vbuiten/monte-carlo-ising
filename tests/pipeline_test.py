from framework.lattice import Lattice
from simulation.simulator import Simulator
from simulation.utils import normalisedCorrelationFunction, correlationTimeFromCorrelationFunction
from analysis.utils import magneticSusceptibility, specificHeatPerSpin
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
plt.rcParams["font.family"] = "serif"

temp = 1.0
N = 20

lattice = Lattice(N)
lattice.spins = np.ones((N,N))
sim = Simulator(lattice, temp)
times_eq, magnetisations_eq = sim.equilibrate(reject_rate_threshold=5e-3)

# plot the magnetisation throughout equilibration
fig, ax = plt.subplots(figsize=(7,5), dpi=240)
ax.plot(times_eq, magnetisations_eq / N**2, lw=.5)
ax.set_xlabel("Time")
ax.set_ylabel("Magnetisation per spin")
ax.set_title("Magnetisation During Equilibration")
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(which="major")
fig.show()

# now do a test run for estimating the correlation time
test_time_end = 500
times_test, magnetisations_test, energies_test = sim.evolve(test_time_end)

fig2, ax2 = plt.subplots(figsize=(7,5), dpi=240)
ax2.plot(times_test, magnetisations_test / N**2, lw=.5)
ax2.set_xlabel("Time")
ax2.set_ylabel("Magnetisation per spin")
ax2.set_title("Magnetisation During Test Run")
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
ax2.grid(which="major")
fig2.show()

# compute the correlation function
norm_corr_func = normalisedCorrelationFunction(times_test, magnetisations_test)
corr_time = correlationTimeFromCorrelationFunction(times_test, norm_corr_func)
before80percent = times_test[:-1] < 0.8*test_time_end

fig3, ax3 = plt.subplots(figsize=(7,5), dpi=240)
ax3.plot(times_test[:-1][before80percent], norm_corr_func[before80percent], lw=.5)
ax3.set_xlabel("Time")
ax3.set_ylabel(r"$\chi(t) / \chi(0)$")
ax3.set_title("Correlation Function During Test Run")
ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
ax3.grid(which="major")
fig3.show()

# now run a long simulation with block measurements
n_it = 20
susceptibilities = np.zeros(n_it)
specific_heats = np.zeros(n_it)

for i in range(n_it):
    times, magnetisations, energies = sim.evolve(sim.time+16*corr_time)
    susceptibilities[i] = magneticSusceptibility(magnetisations, N, temp)
    specific_heats[i] = specificHeatPerSpin(energies, N, temp)
    print ("Iteration {} of {} finished.".format(i+1, n_it))
    print ("Magnetic susceptibility: {} \t Specific heat per spin: {}".format(susceptibilities[i], specific_heats[i]))

mean_susceptibility = np.mean(susceptibilities)
std_susceptibility = np.std(susceptibilities)
mean_specific_heat = np.mean(specific_heats)
std_specific_heat = np.std(specific_heats)

fig4, ax4 = plt.subplots(figsize=(7,10), dpi=240, nrows=2)
ax4[0].hist(susceptibilities, bins="sqrt")
ax4[0].axvline(mean_susceptibility, color="black", ls="--", label="Mean")
ax4[0].axvline(mean_susceptibility-std_susceptibility, color="black", ls=":", label="Standard deviation")
ax4[0].axvline(mean_susceptibility+std_susceptibility, color="black", ls=":")
ax4[0].set_xlabel(r"$\chi_M$")
ax4[0].set_title("Magnetic Susceptibility")

ax4[1].hist(specific_heats, bins="sqrt")
ax4[1].axvline(mean_specific_heat, color="black", ls="--", label="Mean")
ax4[1].axvline(mean_specific_heat-std_specific_heat, color="black", ls=":", label="Standard deviation")
ax4[1].axvline(mean_specific_heat+std_specific_heat, color="black", ls=":")
ax4[1].set_xlabel(r"$C$")
ax4[1].set_title("Specific Heat Per Spin")

for el in ax4:
    el.set_ylabel("Occurrences")

fig4.show()