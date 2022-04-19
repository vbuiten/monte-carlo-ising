import numpy as np
from numba import jit

def periodicCoordinate(coordinate, lattice_size):
    '''
    Calculates the new position of a single spin under periodic boundary conditions.
    The given coordinate is shifted such that it stays within the lattice boundaries.

    :param coordinate: int
            Coordinate before periodic boundary conditions are applied.
    :param lattice_size: int
            Linear size of the lattice in terms of number of spins.
    :return: new_coord: int
            Coordinate after applying periodic boundary conditions.
    '''

    new_coord = coordinate % lattice_size

    return new_coord


def periodicPositions(x_grid, y_grid, lattice_size):
    '''
    Calculates new positions under periodic boundary conditions.
    Assumes that the "first" coordinate in any dimension is zero.

    :param x_grid: ndarray of shape (lattice_size, lattice_size)
            Meshgrid of x coordinates on the lattice
    :param y_grid: ndarray of shape (lattice_size, lattice_size)
            Meshgrid of y coordinates on the lattice
    :param lattice_size: int
            Linear size of the square lattice
    :return: new_x_grid: ndarray of shape (lattice_size, lattice_size)
            Meshgrid of new x coordinates
    :return new_y_grid: ndarray of shape (lattice_size, lattice_size)
            Meshgrid of new y coordinates
    '''

    new_x_grid = x_grid % lattice_size
    new_y_grid = y_grid % lattice_size

    return new_x_grid, new_y_grid


def hamiltonian(x_grid, y_grid, spins, lattice_size, J=1):
    '''
    Computes the total energy of the lattice.

    :param x_grid: ndarray of shape (lattice_size, lattice_size)
            Meshgrid of x coordinates on the lattice
    :param y_grid: ndarray of shape (lattice_size, lattice_size)
            Meshgrid of y coordinates on the lattice
    :param spins: ndarray of shape (lattice_size, lattice_size)
            Spin configuration of the lattice
    :param lattice_size: int
            Linear size of the lattice in terms of number of spins
    :param J: float/int
            Energy scale of the system. Default is 1
    :return: energy: float/int
            Total energy in the lattice.
    '''

    terms_particles = np.zeros((lattice_size, lattice_size))

    for x in range(lattice_size):
        for y in range(lattice_size):

            neighbour_spins = spins[((x_grid == periodicCoordinate(x-1, lattice_size)) & (y_grid == y)) |
                                    ((x_grid == periodicCoordinate(x+1, lattice_size)) & (y_grid == y)) |
                                    ((x_grid == x) & (y_grid == periodicCoordinate(y-1, lattice_size))) |
                                    ((x_grid == x) & (y_grid == periodicCoordinate(y+1, lattice_size)))]

            terms_particles[(x_grid == x) & (y_grid == y)] = np.sum(neighbour_spins) * spins[(x_grid == x) & (y_grid == y)]

    energy = -J * np.sum(terms_particles)

    return energy


def updateHamiltonian(x_grid, y_grid, spins, lattice_size, flip_indices, old_hamiltonian, J=1.):
    '''
    Updates the Hamiltionian after a single spin flip.

    :param x_grid: ndarray of shape (lattice_size, lattice_size)
            Meshgrid of x coordinates on the lattice
    :param y_grid: ndarray of shape (lattice_size, lattice_size)
            Meshgrid of y coordinates on the lattice
    :param spins: ndarray of shape (lattice_size, lattice_size)
            Spin configuration of the lattice
    :param lattice_size: int
            Linear size of the lattice in terms of number of spins
    :param flip_indices: tuple of 2 integers
            Indices of the flipped spin
    :param old_hamiltonian: int/float
            Total energy of the lattice in the previous state, before the single spin was flipped.
    :param J: int/float
            Energy scale of the system. Default is 1
    :return: new_hamiltonian: int/float
            Total energy of the lattice in the current state, after the spin flip.
    '''

    x, y = x_grid[flip_indices], y_grid[flip_indices]
    neighbour_spins = spins[((x_grid == periodicCoordinate(x-1, lattice_size)) & (y_grid == y)) |
                            ((x_grid == periodicCoordinate(x+1, lattice_size)) & (y_grid == y)) |
                            ((x_grid == x) & (y_grid == periodicCoordinate(y-1, lattice_size))) |
                            ((x_grid == x) & (y_grid == periodicCoordinate(y+1, lattice_size)))]

    term_flipped_spin = -J * np.sum(neighbour_spins) * spins[flip_indices]
    new_hamiltonian = old_hamiltonian + 2 * term_flipped_spin

    return new_hamiltonian