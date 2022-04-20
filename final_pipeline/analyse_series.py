import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
from analysis.observables import DirectoryMeasurer

datadir = "/net/vdesk/data2/buiten/COP/ising-sim-data-N50/"

dirmeasurer = DirectoryMeasurer(datadir, usetex=True)
fig, ax = dirmeasurer.plotCorrelationTimes()