

class UnitScaler:
    def __init__(self, k_boltzmann=1., J=1.):
        '''Scaler for converting dimensionless units to physical units and vice versa.

        :param: k_boltzmann: float
                Boltzmann constant in units of the UnitScaler.
        :param: J: float
                Energy scale of the system in units of the UnitScaler.'''

        self.k_boltzmann = k_boltzmann
        self.energy_scale = J

    def toDimlessEnergy(self, energy):
        '''
        Convert from physical energy to dimensionless energy.

        :param energy: ndarray
                Energy in units corresponding to those of the UnitScaler.
        :return: dimless_energy: ndarray
                Corresponding dimensionless energy.
        '''

        dimless_energy = energy / self.energy_scale
        return dimless_energy

    def toEnergy(self, dimless_energy):
        '''
        Convert from dimensionless energy to physical energy.

        :param dimless_energy: ndarray
                Dimensionless energy
        :return: energy: ndarray
                Energy in units of the UnitScaler
        '''

        energy = dimless_energy * self.energy_scale
        return energy

    def toDimlessTemperature(self, temperature):
        '''
        Convert from physical temperature to dimensionless temperature.

        :param temperature: ndarray
                Temperature in units of the UnitScaler
        :return: dimless_temp: ndarray
                Corresponding dimensionless temperature
        '''

        dimless_temp = self.k_boltzmann * temperature / self.energy_scale
        return dimless_temp

    def toTemperature(self, dimless_temperature):
        '''
        Convert from dimensionless temperature to physical temperature.

        :param dimless_temperature: ndarray
                Dimensionless temperature
        :return: temp: ndarray
                Corresponding physical temperature in units of the UnitScaler
        '''

        temp = self.energy_scale * dimless_temperature / self.k_boltzmann
        return temp