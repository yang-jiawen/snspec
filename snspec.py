
import numpy as np


class spectrum:
    """[This is a spectrum class]
    """

    def __init__(self, wavelength, flux):
        """[Initialize spectrum class]

        Args:
            wavelength ([list or numpy.array]): [list of spectrum wavelength]
            flux ([list or numpy.array]): [list of spectrum flux. Need to have
            the same length as wavelength.]
        """
        self.wavelength = np.array(wavelength)
        self.flux = np.array(flux)
        return
