from analysis.evolution import MagnetisationPlotter, EnergyPlotter

basepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
path = basepath + "ising-series\\"

plotter = MagnetisationPlotter(path, usetex=False)
plotter.plot()
plotter.show()

energy_plotter = EnergyPlotter(path, usetex=False)
energy_plotter.plot()
energy_plotter.show()