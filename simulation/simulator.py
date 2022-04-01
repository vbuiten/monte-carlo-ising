import numpy as np

import framework.lattice


class Simulator:
    def __init__(self, lattice, temperature):

        if isinstance(lattice, framework.lattice.Lattice):
            self.lattice = lattice
        else:
            raise TypeError("Parameter 'lattice' must be an instance of framework.lattice.Lattice.")

        beta = 1./temperature
        self.exp_minus_beta = np.exp(-beta)

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


    def evolve(self, time_end):

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

        return times, magnetisations, energies