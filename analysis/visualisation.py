import matplotlib.pyplot as plt
import matplotlib.colors
from framework.lattice import Lattice

plt.rcParams["font.family"] = "serif"

class LatticeVisual:
    def __init__(self, lattice, usetex=False, figsize=(7,5), dpi=240):

        if usetex:
            plt.rcParams["text.tex"] = True

        if isinstance(lattice, Lattice):
            self.lattice = lattice
        else:
            raise TypeError("Parameter 'lattice' should be an instance of 'framework.lattice.Lattice'.")

        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.cmap = matplotlib.colors.ListedColormap(["dimgrey", "snow"])

        self.im = self.ax.pcolormesh(lattice.x_grid, lattice.y_grid, lattice.spins, cmap=self.cmap)
        self.cbar = self.fig.colorbar(self.im, label="Spin",
                                      ticks=(-1,1))
        self.ax.set_aspect("equal")

        self.ax.set_xlabel(r"$x$")
        self.ax.set_ylabel(r"$y$")

    def show(self):

        self.fig.show()