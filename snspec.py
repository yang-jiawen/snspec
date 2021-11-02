
import numpy as np
from copy import copy


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

    def smooth(self, order=4):
        """[Smooth spectra using wavelet transformation.]

        Args:
            order (int, optional): [Order of smoothness. The higher the order,
             the smoother the smoothed spectrum.]. Defaults to 4.
        """
        smoothed_spectrum = smooth_spectrum(self, order=order)
        self.smoothed_flux = smoothed_spectrum.flux
        self.smoothed_order = order
        return


def atrouswt(input_spec, j, wavelet=[1.0/16, 1.0/4, 3.0/8, 1.0/4, 1.0/16]):
    """[Spectra wavelet transformation.
     See https://arxiv.org/pdf/0907.3171.pdf.
     Calculate c_j from c_(j-1).
     a trous wavelet, input spectrum and j,
      output c_j(k) in the form of spectrum {'wavelengths','flux'}
     input_spectrum should be c_j]

    Args:
        input_spectrum ([spectrum class]): [class of spectrum]
        j ([float]): [order of input spectrum]
        wavelet ([numpy array]): [wavelet used to smooth the spectrum.
        Default: [1.0/16, 1.0/4, 3.0/8, 1.0/4, 1.0/16]]
    """
    wavelet = np.array(wavelet)
    spectrum = copy.deepcopy(input_spec)
    flux = list(spectrum.flux)
    flux = flux+flux[::-1]      # periodicity S(k+N)=S(N-k)

    N2 = len(flux)
    for ii in range(len(flux)):
        flux[ii] = flux[(ii-2*2**(j-1)) % N2]*wavelet[0]\
            + flux[(ii-2**(j-1)) % N2]*wavelet[1]\
            + flux[(ii) % N2]*wavelet[2] + flux[(ii+2**(j-1)) % N2] * \
            wavelet[3] + flux[(ii+2*2**(j-1)) % N2]*wavelet[4]
    spectrum.flux = np.array(flux[0:int(N2/2)])
    return(spectrum)


def smooth_spectrum(spec, order=4, **kwargs):
    """[summary]

    Args:
        spec (spectrum class): spectrum class to be smoothed
        order (int, optional): Order of smoothness. The higher the order,
             the smoother the smoothed spectrum. Defaults to 4.

    Returns:
        spectrum class: spectrum class with smoothed spectrum.
    """
    result_spec = copy.deepcopy(spec)

    for ii in range(order):
        result_spec = atrouswt(result_spec, ii+1, **kwargs)

    return result_spec
