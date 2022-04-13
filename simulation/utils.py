import numpy as np
from numba import jit

@jit(nopython=True)
def correlationFunction(times, quantities):

    t_max = times[-1]
    corr_func = np.zeros(len(times)-1)

    print ("Computing correlation function.")

    for i, time in enumerate(times[:-1]):

        if i == 0:
            term1 = np.sum(quantities**2)
            term2 = - np.sum(quantities)**2
        else:
            term1 = np.sum(quantities[:-i] * quantities[i:])
            term2 = - np.sum(quantities[:-i]) * np.sum(quantities[i:])

        prefactor = 1. / (t_max - time)

        corr_func[i] = prefactor * (term1 + term2)

        '''
        if time % 10 == 0:
            print ("Time: {} \t Correlation function: {}".format(time, corr_func[i]))
            print ("First term: {} \t Second term: {} \t Prefactor: {}".format(term1, term2, prefactor))
        '''

        if time % 10 == 0:
            print ("Time:", time)

    return corr_func


@jit(nopython=True)
def normalisedCorrelationFunction(times, quantities):

    corr_func = correlationFunction(times, quantities)
    corr_func0 = corr_func[0]

    norm_corr_func = corr_func / corr_func0

    return norm_corr_func


@jit(nopython=True)
def meanAbsoluteSpin(magnetisation_per_spin):

    abs_spin_per_point = np.abs(magnetisation_per_spin)

    return abs_spin_per_point


@jit(nopython=True)
def energyPerSpin(energy, n_spins):
    '''Obsolete'''

    return energy / n_spins