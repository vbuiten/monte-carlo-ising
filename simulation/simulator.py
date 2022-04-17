import numpy as np
import h5py
import framework.lattice
from simulation.utils import meanAbsoluteSpin, energyPerSpin


class Simulator:
    def __init__(self, lattice, temperature):

        if isinstance(lattice, framework.lattice.Lattice):
            self.lattice = lattice
        else:
            raise TypeError("Parameter 'lattice' must be an instance of framework.lattice.Lattice.")

        beta = 1./temperature
        self.exp_minus_beta = np.exp(-beta)
        self.temperature = temperature

        # define a time unit
        # we define a time unit as a full sweep of the mesh
        self.flips_per_time = lattice.size**2
        self.time_per_flip = 1./self.flips_per_time
        self.time = 0.

    def step(self, current_energy=None):

        if current_energy is None:
            # measure the energy in the current state
            current_energy = self.lattice.hamiltonian()
        elif isinstance(current_energy, float) or isinstance(current_energy, int):
            pass
        else:
            raise TypeError("Parameter 'current_energy' should be a float or an integer.")

        current_spins = np.copy(self.lattice.spins)

        # randomly flip one spin
        flip_i, flip_j = self.lattice.flipRandomSpin()
        new_energy = self.lattice.updateHamiltonian((flip_i, flip_j), current_energy)

        if new_energy > current_energy:
            ratio = self.exp_minus_beta**(new_energy - current_energy)
            accept = np.random.binomial(1, ratio)

            if not accept:
                self.lattice.spins = current_spins
                new_energy = current_energy
                #print ("Not accepted. New energy: {}".format(new_energy))

        return new_energy


    def equilibrate(self, threshold=0.9, sweeps=10, reject_rate_threshold=1e-3):
        '''
        Bring the system into equilibrium using a rejection rate stability criterion.

        :param threshold:
        :param sweeps: int
                Number of sweeps after which the rejection rate is checked and reset.
        :param reject_rate_threshold: float
                Threshold for the absolute difference between rejection rates.
                If the difference in rejection rates is smaller than this number, the equilibration stops.
        :return: times: ndarray of shape (n_times,)
                Times used in the equilibration process
                magnetisations: ndarray of shape (n_times,)
                Total magnetisation of the lattice at each point in time.
        '''

        old_energy = self.lattice.hamiltonian()
        times = np.array([])
        magnetisations = np.array([])
        time = 0.
        rejected = 0
        rejection_rate = 1e-8
        rejection_rate_diff = 1.
        full_sweep = False

        # simulate until the rejection rate per sweep reaches the threshold
        #while (rejection_rate < threshold) or (not full_sweep):
        while (abs(rejection_rate_diff) > reject_rate_threshold) or not full_sweep:

            # reset the counter after one full sweep
            if (np.around(time, 8) % sweeps == 0.) and (time != 0):
                print("Time:", time)

                new_rejection_rate = rejected / (sweeps * self.lattice.size**2)
                print ("Rejection rate:", new_rejection_rate)
                rejection_rate_diff = new_rejection_rate - rejection_rate
                print ("Rejection rate difference:", rejection_rate_diff)

                rejection_rate = float(np.copy(new_rejection_rate))

                rejected = 0
                full_sweep = True

            else:
                full_sweep = False

            times = np.append(times, time)
            magnetisations = np.append(magnetisations, self.lattice.magnetisation())

            energy = self.step(current_energy=old_energy)

            # if the new state is rejected, add to the counter
            if energy == old_energy:
                rejected += 1

            old_energy = float(np.copy(energy))
            time += self.time_per_flip

        print ("Equilibration finished.")

        return times, magnetisations


    def evolve(self, time_end, savefile=None):

        times = np.arange(self.time, time_end, self.time_per_flip)
        magnetisations = np.zeros(times.shape)
        energies = np.zeros(times.shape)

        energy = self.lattice.hamiltonian()

        for i, time in enumerate(times):

            energies[i] = energy
            magnetisations[i] = self.lattice.magnetisation()
            energy = self.step(current_energy=energy)

            if (time - self.time) % 1 == 0:
                print ("Time: {}".format(time))

        print ("Simulation finished.")
        self.time = time

        if isinstance(savefile, str):
            if not savefile.endswith(".hdf5"):
                savefile = savefile+".hdf5"

            file = h5py.File(savefile, "w")
            energies_dset = file.create_dataset("energy", data=energies)
            magnetisations_dset = file.create_dataset("magnetisation", data=magnetisations)

            xgrid_dset = file.create_dataset("x_grid", data=self.lattice.x_grid)
            ygrid_dset = file.create_dataset("y_grid", data=self.lattice.y_grid)
            times_dset = file.create_dataset("times", data=times)
            file.attrs["temperature"] = self.temperature
            file.attrs["size"] = self.lattice.size

            file.close()
            print ("File created at {}.".format(savefile))

        return times, magnetisations, energies
