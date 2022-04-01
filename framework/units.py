

class UnitScaler:
    def __init__(self, k_boltzmann=1., J=1.):
        '''Scaler for converting dimensionless units to physical units and vice versa.'''

        self.k_boltzmann = k_boltzmann
        self.energy_scale = J

    def toDimlessEnergy(self, energy):

        dimless_energy = energy / self.energy_scale
        return dimless_energy

    def toEnergy(self, dimless_energy):

        energy = dimless_energy * self.energy_scale
        return energy

    def toDimlessTemperature(self, temperature):

        dimless_temp = self.k_boltzmann * temperature / self.energy_scale
        return dimless_temp

    def toTemperature(self, dimless_temperature):

        temp = self.energy_scale * dimless_temperature / self.k_boltzmann
        return temp