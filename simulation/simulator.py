import numpy as np
import h5py
import framework.lattice


class Simulator:
    def __init__(self, lattice, temperature):
        '''
        Class for evolving the 2D Ising model. The evolution is performed through the Metropolis-Hastings
        algorithm, using Boltzmann weights as the relevant probability distribution.

        :param lattice: Lattice instance
                Lattice on which to apply the evolution mechanism.
        :param temperature: float
                Dimensionless temperature of the system.
        '''

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
        '''
        Take one step of the simulation, i.e. flip a single spin on the lattice and either accept or reject
        the new state.

        :param current_energy: float or NoneType
                Energy in the current configuration, before taking the evolution step. If None, calculates the
                energy from scratch. Providing this parameter is recommended as it is computationally favourable.
                Default is None.
        :return: new_energy: float
                Energy after the evolution step has been taken.
        '''

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

        return new_energy


    def equilibrate(self, sweeps=10, reject_rate_threshold=1e-3):
        '''
        Bring the system into equilibrium using a rejection rate stability criterion.

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


    def evolve(self, time_end, savefile=None, correlation_time=None):
        '''
        Evolve the system for some "time" interval, i.e. for a given number of sweeps of the lattice.

        :param time_end: float
                "Time" for which to run the simulation.
        :param savefile: float or NoneType
                File to which the data should be saved. If None, no simulation data is stored. Default is None.
        :param correlation_time: float
                Estimated correlation time for the system. If None while "savefile" is not None, the attribute
                corresponding to the correlation time is also set to be None, but several measurements will no longer
                be possible in post-processing. Default is None.
        :return:
            times: ndarray of shape (n_times,)
                Time stamps for every evolution step
            magnetisations: ndarray of shape (n_times,)
                Total magnetisation of the lattice at every step
            energies: ndarray of shape (n_times,)
                Total energy of the lattice at every step
        '''

        times = np.arange(self.time, time_end, self.time_per_flip)
        magnetisations = np.zeros(times.shape)
        energies = np.zeros(times.shape)

        energy = self.lattice.hamiltonian()

        for i, time in enumerate(times):

            energies[i] = energy
            magnetisations[i] = self.lattice.magnetisation()
            energy = self.step(current_energy=energy)

            if np.around(time - self.time, 5) % 10 == 0:
                print ("Time: {}".format(time))

        print ("Simulation finished.")
        self.time = times[-1]

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

            if correlation_time is not None:
                file.attrs["correlation-time"] = correlation_time

            file.close()
            print ("File created at {}.".format(savefile))

        return times, magnetisations, energies
