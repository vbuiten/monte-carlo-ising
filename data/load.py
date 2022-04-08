'''Load simulation data'''

import h5py
import numpy as np

class LatticeHistory:
    def __init__(self, filename):
        '''
        Load the data of a single simulation run of a single lattice.

        :param filename: str
        '''

        dfile = h5py.File(filename, "r")

        dset_spins = dfile["spins"]
        dset_energies = dfile["avg-energy"]
        dset_magnetisations = dfile["avg-magnetisation"]
        dset_xgrid = dfile["x_grid"]
        dset_ygrid = dfile["y_grid"]
        dset_times = dfile["times"]

        self.spins = np.copy(dset_spins)
        self.energies = np.copy(dset_energies)
        self.avg_magnetisations = np.copy(dset_magnetisations)
        self.x_grid = np.copy(dset_xgrid)
        self.y_grid = np.copy(dset_ygrid)
        self.times = np.copy(dset_times)

        self.temperature = dfile.attrs["temperature"]
        self.size = dfile.attrs["size"]

        dfile.close()