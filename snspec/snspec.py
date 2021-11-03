
import numpy as np
from copy import copy
from scipy.stats import binned_statistic


class spectrum:
    """This is a spectrum class
    """

    def __init__(self, wavelength, flux):
        """Initialize spectrum class

        Args:
            wavelength ([list or numpy.array]): [list of spectrum wavelength]
            flux ([list or numpy.array]): [list of spectrum flux. Need to have
            the same length as wavelength.]
        """
        self.wavelength = np.array(wavelength)
        self.flux = np.array(flux)
        self.smoothed_flux = None
        self.smoothed_order = None
        return

    def smooth(self, order=4):
        """Smooth spectra using wavelet transformation.

        Args:
            order (int, optional): Order of smoothness. The higher the order,
             the smoother the smoothed spectrum. Defaults to 4.
        """
        smoothed_spectrum = smooth_spectrum(self, order=order)
        self.smoothed_flux = smoothed_spectrum.flux
        self.smoothed_order = order
        return

    # def find_continuum(self, min_range=[5820, 6000], max_range=[6200, 6540], binsize=40):
    #     """[summary]

    #     Args:
    #         min_range (list, optional): [description]. Defaults to [5820, 6000].
    #         max_range (list, optional): [description]. Defaults to [6200, 6540].
    #         binsize (int, optional): [description]. Defaults to 40.
    #     """
    #     min_wl, min_fl = find_wl_at_max_fl(
    #         self, wave_range=min_range, binsize=binsize)
    #     max_wl, max_fl = find_wl_at_max_fl(
    #         self, wave_range=max_range, binsize=binsize)
    #     self.cont_min = [min_wl, min_fl]
    #     self.cont_max = [max_wl, max_fl]
    #     self.cont_min_range = min_range
    #     self.cont_max_range = max_range
    #     return


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
    """smooth spectrum using wavelet transformation method.

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


def bin_spectrum(spec, binsize=30):
    """bin spectrum

    Args:
        spec (spectrum class): spectrum to be binned.
        spectrum.wavelength in AA.
        binsize (int, optional): binsize used for binning. Defaults to 30 (AA).

    Returns:
        spectrum class: spectrum class with binned spectrum.
    """

    fl = spec.flux
    wl = spec.wavelength
    bin_wl = np.arange(wl[0], wl[-1], binsize)
    bin_fl = binned_statistic(wl, fl, statistic='mean', bins=bin_wl)
    bin_wl = binned_statistic(wl, wl, statistic='mean', bins=bin_wl)
    return spectrum(bin_wl.statistic, bin_fl.statistic)


def find_wl_at_max_fl(spec, wave_range, binsize=30):
    """ Given spectrum class, and wave_range = [min_wl, max_wl] in AA.
    find wavelength at max flux. This is to avoid false spikes.

    Args:
        spec (spectrum class): spectrum class to find wavelength at maximum.
        wave_range (array_like of length 2): lower and upper boundaries of
        wavelength to find maximum.
        binsize (int, optional): binsize (AA) used to bin spectrum.
        Defaults to 30. If None, then no binning is done.

    Returns:
        array_like of length 2: wavelength at flux maximum, max flux.
    """

    idx = (spec.wavelength > wave_range[0]) & (spec.wavelength < wave_range[1])
    fl = np.array(spec.flux)[idx]

    wl = spec.wavelength[idx]
    if binsize is None:
        binned_spec = spec
    else:
        binned_spec = bin_spectrum(spectrum(wl, fl), binsize=binsize)

    fl = binned_spec.flux
    wl = binned_spec.wavelength
    idx = np.argmax(fl)
    return wl[idx], fl[idx]
