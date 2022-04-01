import numpy as np
from framework.utils import hamiltonian, updateHamiltonian

class Lattice:
    def __init__(self, size):
        '''
        size x size lattice of particles that can have spin up or spin down.

        :param size: int
                Number of spins on one side of the square lattice
        '''

        if isinstance(size, int) and size > 0:
            self.size = size
        else:
            raise TypeError("Parameter 'size' must be a positive integer!")

        x = np.arange(0, size, 1)
        y = np.arange(0, size, 1)

        self.x_grid, self.y_grid = np.meshgrid(x, y)


    @property
    def spins(self):
        return self._spins

    @spins.setter
    def spins(self, spins_grid):

        if isinstance(spins_grid, np.ndarray) and spins_grid.shape == (self.size, self.size) and np.all(np.abs(spins_grid) == 1):
            self._spins = spins_grid

        else:
            raise ValueError("Given spins must be an array of {} x {} spins, containing only 1 and -1 values".format(self.size, self.size))


    def uniformRandomSpins(self):

        random_draws = np.random.randint(2, size=(self.size, self.size))
        random_draws[random_draws == 0] = -1
        self.spins = random_draws


    def flipRandomSpin(self):
        '''
        Flips a single, random spin. The indices are drawn from a uniform random distribution.
        '''

        i, j = np.random.randint(0, self.size, size=2)
        self._spins[i,j] = -self._spins[i,j]
        return i, j


    def magnetisation(self):
        '''
        Computes the total magnetisation of the lattice, i.e. the sum of all spins.

        :return: magnetisation: float
                Total magnetisation of the lattice
        '''

        magnetisation = np.sum(self.spins)
        return float(magnetisation)


    def magnetisationPerSpin(self):
        '''
        Computer the magnetisation per spin of the lattice.

        :return: m: float
                Magnetisation per spin
        '''

        m = self.magnetisation() / self.size**2
        return m


    def hamiltonian(self):

        energy = hamiltonian(self.x_grid, self.y_grid, self.spins, self.size)

        return energy


    def updateHamiltonian(self, flip_indices, old_hamiltonian):

        new_energy = updateHamiltonian(self.x_grid, self.y_grid, self.spins, self.size, flip_indices, old_hamiltonian)

        return new_energy