from analysis.evolution import MagnetisationPlotter

basepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
path = basepath + "ising-series\\"

plotter = MagnetisationPlotter(path, usetex=False)
plotter.plot()
plotter.show()