import numpy as np
from numba import jit

@jit(nopython=True)
def correlationFunction(times, quantities):

    t_max = times[-1]
    corr_func = np.zeros(len(times)-1)

    for i, time in enumerate(times[:-1]):

        term1 = np.sum(quantities[:-1-i] * quantities[1+i:])
        term2 = - np.sum(quantities[:-1-i]) * np.sum(quantities[1+i:])
        prefactor = 1. / (t_max - time)

        corr_func[i] = prefactor * (term1 + term2)

    return corr_func


@jit(nopython=True)
def normalisedCorrelationFunction(times, quantities):

    corr_func = correlationFunction(times, quantities)
    corr_func0 = corr_func[0]

    norm_corr_func = corr_func / corr_func0

    return norm_corr_func
