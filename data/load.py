'''Load simulation data'''

import h5py
import numpy as np
import os

class LatticeHistory:
    def __init__(self, filename):
        '''
        Load the data of a single simulation run of a single lattice.

        :param filename: str
        '''

        dfile = h5py.File(filename, "r")

        dset_energies = dfile["energy"]
        dset_magnetisations = dfile["magnetisation"]
        dset_xgrid = dfile["x_grid"]
        dset_ygrid = dfile["y_grid"]
        dset_times = dfile["times"]

        self.energies = np.copy(dset_energies)
        self.magnetisations = np.copy(dset_magnetisations)
        self.x_grid = np.copy(dset_xgrid)
        self.y_grid = np.copy(dset_ygrid)
        self.times = np.copy(dset_times)

        self.temperature = dfile.attrs["temperature"]
        self.size = dfile.attrs["size"]

        try:
            self.correlation_time = dfile.attrs["correlation-time"]
        except:
            print ("No correlation time found in file. Setting correlation time to None.")
            self.correlation_time = None

        dfile.close()


class LatticeHistories:
    def __init__(self, directory):
        '''
        Load all simulation data files that exist in a given directory, and make a list of corresponding
        LatticeHistory instances.

        :param directory: str
                Directory in which the simulation data are stored. No other .hdf5 files can be present in the folder.
        '''

        files_list = os.listdir(directory)
        self.histories = []

        for i, f in enumerate(files_list):
            if f.endswith(".hdf5"):
                history = LatticeHistory(directory+f)
                self.histories.append(history)

        self.directory = directory
        self.n_histories = len(files_list)