import numpy as np
from numba import jit

def periodicCoordinate(coordinate, lattice_size):

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