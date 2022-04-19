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
        '''
        Spin configuration of the lattice.

        :return: self._spins: ndarray of shape (size, size)
                Array indicating whether a spin is up (+1) or down (-1)
        '''

        return self._spins

    @spins.setter
    def spins(self, spins_grid):
        '''
        Spin configuration of the lattice.

        :param spins_grid: ndarray of shape (size, size)
                Array indicating whether a spin is up (+1) or down (-1)
        :return:
        '''

        if isinstance(spins_grid, np.ndarray) and spins_grid.shape == (self.size, self.size) and np.all(np.abs(spins_grid) == 1):
            self._spins = spins_grid

        else:
            raise ValueError("Given spins must be an array of {} x {} spins, containing only 1 and -1 values".format(self.size, self.size))


    def uniformRandomSpins(self):
        '''
        Randomly assigns each spin in the lattice +1 or -1.

        :return:
        '''

        random_draws = np.random.randint(2, size=(self.size, self.size))
        random_draws[random_draws == 0] = -1
        self.spins = random_draws


    def flipRandomSpin(self):
        '''
        Flips a single, random spin. The indices are drawn from a uniform random distribution.

        :return: i: int
                First index of the flipped spin.
                j: int
                Second index of the flipped spin.
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
        Computes the magnetisation per spin of the lattice.

        :return: m: float
                Magnetisation per spin
        '''

        m = self.magnetisation() / self.size**2
        return m


    def hamiltonian(self):
        '''
        Computes the total energy of the lattice.

        :return: energy: int/float
                Total energy of the lattice.
        '''

        energy = hamiltonian(self.x_grid, self.y_grid, self.spins, self.size)

        return energy


    def updateHamiltonian(self, flip_indices, old_hamiltonian):
        '''
        Updates the Hamiltonian after a single spin flip.
        Use this instead of self.hamiltonian() whenever possible, as this method is much faster.

        :param flip_indices: tuple of 2 integers
                Indices of the flipped spin.
        :param old_hamiltonian: int/float
                Total energy of the lattice in the previous state, before the single spin was flipped.
        :return: new_energy: int/float
                Total energy of the lattice in the current state, after the spin flip.
        '''

        new_energy = updateHamiltonian(self.x_grid, self.y_grid, self.spins, self.size, flip_indices, old_hamiltonian)

        return new_energy