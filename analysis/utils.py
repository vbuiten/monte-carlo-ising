import numpy as np
from numba import jit

@jit(nopython=True)
def meanAbsoluteSpin(total_magnetisation, linear_size):
    '''
    Computes the mean absolute spin by averaging over some block of time.

    :param total_magnetisation: ndarray of shape (n_times,)
            Total magnetisation of the lattice at each point in time.
    :param linear_size: int
            Number of spins in onre row or column of the (square) lattice.
    :return: mean_abs_spin: float
            Measured mean absolute spin (per particle) in this block.
    '''

    mean_abs_spin = np.mean( np.abs(total_magnetisation) ) / linear_size**2

    return mean_abs_spin


@jit(nopython=True)
def meanEnergyPerSpin(total_energy, linear_size):
    '''
    Computes the energy per spin in the lattice, averaged over some block of time.

    :param total_energy: ndarray of shape (n_times,)
            Dimensionless total energy of the lattice at each point in time in the block.
    :param linear_size: int
            Number of spins in one row or column of the square lattice.
    :return: mean_energy_per_spin: float
            Measured mean energy per spin in this block.
    '''

    mean_energy_per_spin = np.mean(total_energy) / linear_size**2

    return mean_energy_per_spin


@jit(nopython=True)
def magneticSusceptibility(total_magnetisation, linear_size, temperature):
    '''
    Computes the dimensionless magnetic susceptibility by averaging over a block of time.

    :param total_magnetisation: ndarray of shape (n_times,)
            Total magnetisation of the lattice at each point in time.
    :param linear_size: int
            Number of spins in one row or column of the (square) lattice.
    :param temperature: float
            Dimensionless temperature of the system.
    :return: susceptibility: float
            Dimensionless magnetic susceptibility measured in this one block of time.
    '''

    prefactor = 1. / (linear_size**2 * temperature)
    susceptibility = prefactor * ( np.mean(total_magnetisation**2) - np.mean(total_magnetisation)**2 )

    return susceptibility


@jit(nopython=True)
def specificHeatPerSpin(total_energy, linear_size, temperature):
    '''
    Computes the specific heat per spin by averaging over a block of time.

    :param total_energy: ndarray of shape (n_times,)
            Dimensionless total energy of the lattice at each point in time in the block.
    :param linear_size: int
            Number of spins in one row or column of the square lattice.
    :param temperature: float
            Dimensionless temperature of the system.
    :return: specific_heat: float
            Measured specific heat per spin in this particular block.
    '''

    prefactor = 1. / (linear_size**2 * temperature**2)
    specific_heat = prefactor * ( np.mean(total_energy**2) - np.mean(total_energy)**2 )

    return specific_heat


@jit(nopython=True)
def thermalAveragingStandardDeviation(times, quantities, correlation_time):

    t_max = times[-1]
    prefactor = 2 * correlation_time / t_max
    variance = prefactor * ( np.mean(quantities**2) - np.mean(quantities)**2 )
    std = np.sqrt(variance)

    return std