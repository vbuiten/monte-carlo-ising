import numpy as np
from numba import jit

@jit(nopython=True, parallel=True)
def correlationFunction(times, quantities):

    t_max = times[-1]

    corr_func = np.zeros(len(times)-1)

    print ("Computing correlation function.")

    corr_func[0] = np.sum(quantities**2) / (t_max - times[0]) - np.sum(quantities)**2 / ((t_max - times[0])**2)

    for i in range(1, len(times)-1):

        term1 = np.sum(quantities[:-i] * quantities[i:])
        term2 = - np.sum(quantities[:-i]) * np.sum(quantities[i:])

        corr_func[i] = term1 / (t_max - times[i]) + term2 / ((t_max - times[i])**2)

        '''
        if time % 10 == 0:
            print ("Time: {} \t Correlation function: {}".format(time, corr_func[i]))
            print ("First term: {} \t Second term: {} \t Prefactor: {}".format(term1, term2, prefactor))
        '''

        if times[i] % 10 == 0:
            print ("Time:", times[i])

    return corr_func


@jit(nopython=True, parallel=True)
def normalisedCorrelationFunction(times, quantities):

    corr_func = correlationFunction(times, quantities)
    corr_func0 = corr_func[0]

    norm_corr_func = corr_func / corr_func0

    return norm_corr_func


@jit(nopython=True, parallel=True)
def correlationTimeFromCorrelationFunction(times, normalised_corr_func):

    timestep = times[1] - times[0]
    positive = normalised_corr_func > 0
    corr_time = times[0] + timestep * np.sum(normalised_corr_func[positive])

    return corr_time


@jit(nopython=True, parallel=True)
def correlationTime(times, quantities):

    norm_corr_func = normalisedCorrelationFunction(times, quantities)
    corr_time = correlationTimeFromCorrelationFunction(times, norm_corr_func)

    return corr_time


@jit(nopython=True)
def meanAbsoluteSpin(magnetisation_per_spin):

    abs_spin_per_point = np.abs(magnetisation_per_spin)

    return abs_spin_per_point


@jit(nopython=True)
def energyPerSpin(energy, n_spins):
    '''Obsolete'''

    return energy / n_spins