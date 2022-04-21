import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 13
from analysis.observables import DirectoryMeasurer

datadir = "/net/vdesk/data2/buiten/COP/ising-sim-data-N50/"

dirmeasurer = DirectoryMeasurer(datadir, usetex=True)
plots = dirmeasurer.plotAll()

plots[0][0].savefig(datadir+"N50-corr-times.png")
plots[1][0].savefig(datadir+"N50-abs-spins.png")
plots[2][0].savefig(datadir+"N50-energy-per-spin.png")
plots[3][0].savefig(datadir+"N50-magnetic-susc.png")
plots[4][0].savefig(datadir+"N50-spec-heat.png")