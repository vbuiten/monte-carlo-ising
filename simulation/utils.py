import numpy as np
from numba import jit
from IPython import embed

@jit(nopython=True, parallel=True)
def correlationFunction(times, quantities, t_eval_max=None):

    t_max = times[-1]
    timestep = times[1] - times[0]
    print ("t_max:", t_max)

    if t_eval_max is None:
        t_eval_max = times[-1]

    corr_func = np.zeros(len(times)-1)

    print ("Computing correlation function.")

    corr_func[0] = timestep * np.sum(quantities**2) / (t_max - times[0]) - timestep**2 * np.sum(quantities)**2 / ((t_max - times[0])**2)

    for i in range(1, len(times)-1):

        term1 = timestep * np.sum(quantities[:-i] * quantities[i:]) / (t_max - times[i])
        term2 = -timestep**2 * np.sum(quantities[:-i]) * np.sum(quantities[i:]) / ((t_max - times[i])**2)

        corr_func[i] = term1 + term2

        if times[i] % 10 == 0:
            print("Time:", times[i])
            '''
            print ("First term:", term1)
            print ("Second term:", term2)
            '''

    return corr_func


@jit(nopython=True, parallel=True)
def normalisedCorrelationFunction(times, quantities):

    corr_func = correlationFunction(times, quantities)
    corr_func0 = corr_func[0]

    norm_corr_func = corr_func / corr_func0

    return norm_corr_func


@jit(nopython=True)
def correlationTimeFromCorrelationFunction(times, normalised_corr_func, negative_stop=True):

    timestep = times[1] - times[0]
    indices_negative = np.argwhere(normalised_corr_func < 0)

    if negative_stop:
        try:
            # this will fail if the correlation function is nonzero everywhere
            idx_first_negative = indices_negative[0,0]
            corr_time = times[0] + timestep * np.sum(normalised_corr_func[1:idx_first_negative] + normalised_corr_func[:idx_first_negative-1]) / 2
        except:
            corr_time = times[0] + timestep * np.sum(normalised_corr_func[1:] + normalised_corr_func[:-1]) / 2

    else:
        corr_time = times[0] + timestep * np.sum(normalised_corr_func[1:] + normalised_corr_func[:-1]) / 2

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