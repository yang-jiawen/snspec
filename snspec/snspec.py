
import numpy as np
import copy
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

        self.error = None
        return

    def smooth(self, order=2):
        """Smooth spectra using wavelet transformation.

        Args:
            order (int, optional): Order of smoothness. The higher the order,
             the smoother the smoothed spectrum. Defaults to 2.

        """
        _, smoothed_spectrum_fl = smooth_spectrum(
            self.wavelength, self.flux, order=order)
        self.smoothed_flux = smoothed_spectrum_fl
        self.smoothed_order = order
        return

    def calculate_error(self, order=2, **kwargs):
        """Calculate error from spectrum.

        Args:
            order (int, optional): ordr of smooth. The larger the smoother.
             For spectra with high SNR, order needs to be lower. Defaults to 2.
        """
        self.smooth(order=order)
        errors = find_spectrum_error(
            self.wavelength, self.flux, order=order, **kwargs)
        self.error = errors
        return

    def simulate_noise_spectrum(self, **kwargs):
        """Simulate noisy spectrum based on give spectrum.

        Args:
            order (int, optional): order of smooth. The larger the smoother.
             For spectra with high SNR, order needs to be lower. Defaults to 4.

        Returns:
            tuple of array-like: wavelength and flux of simulated spectrum
             with noise.
        """
        if self.error is None:
            self.calculate_error(**kwargs)
        noises = [np.random.normal(loc=0, scale=ii) for ii in self.error]
        return self.wavelength, noises+self.smoothed_flux


class line(spectrum):
    def __init__(self, wavelength, flux, line_begin=None, line_end=None):
        super().__init__(wavelength, flux)
        # self.begin and end defines the endpoints of this line.
        # and this will be used in calculating pEW and continuum.
        self.begin = line_begin
        self.end = line_end
        self.normalized_flux = None
        return

    def find_continuum(self, min_range, max_range, binsize=40):
        """Given wavelength ranges, find the endpoints of the continuum.

        Args:
            min_range (list): List of length 2, used to determine
             the blue endpoint. e.g. [5820, 6000] for restframe Si II 6355.
            max_range (list): List of length 2, used to determine
             the red endpoint. e.g. [6200, 6540] for restframe Si II 6355.
            binsize (int, optional): binsize used in binning to find
            maximum (AA). Defaults to 40 AA.
        """

        min_wl, min_fl = find_wl_at_max_fl(
            self.wavelength, self.flux, wave_range=min_range, binsize=binsize)
        max_wl, max_fl = find_wl_at_max_fl(
            self.wavelength, self.flux, wave_range=max_range, binsize=binsize)
        self.begin = [min_wl, min_fl]
        self.end = [max_wl, max_fl]
        self.cont_min_range = min_range
        self.cont_max_range = max_range
        return

    def normalize(self, *args, binsize=30):
        """Normalize the line feature by the continuum defined by the line
        connecting the two endpoints self.begin and self.end.
        """
        if (self.begin is None) or (self.end is None):
            self.find_continuum(*args, binsize=binsize)

        coeff = np.polyfit(*np.transpose([self.begin, self.end]), 1)
        f_line = np.poly1d(coeff)  # linear function of continuum

        # inds = (self.wavelength > self.cont_min[0]) & (
        #     self.wavelength < self.cont_max[0])
        # self.normalized_wavelength = self.wavelength[inds]
        # self.normalized_flux = self.flux[inds] / \
        #     f_line(self.normalized_wavelength)
        self.normalized_flux = self.flux/f_line(self.wavelength)
        return

    def calculate_pEW(self, *args, **kwargs):
        """Calculate pEW of line feature.

        Args:
            #min_range (array-like of length 2): min wl, min fl, the beginpoint
            #max_range (array-like of length 2): max wl, max fl, the endpoint
            *args: min_range and max_range
        Returns:
            float: pEW of the line feature
        """
        if self.normalized_flux is None:
            self.normalize(*args, **kwargs)
        inds = (self.wavelength > self.begin[0]) & (
            self.wavelength < self.end[0])
        pEW_wl = self.wavelength[inds]
        pEW_fl = self.normalized_flux[inds]
        delta_wl = pEW_wl[1:] - pEW_wl[:-1]
        fl = pEW_fl[:-1]
        pEW = np.sum((1-fl)*delta_wl)
        self.pEW = pEW
        return pEW

    def calculate_pEW_MC(self, min_range, max_range, Ntimes=100,
                         binsize=30,  **kwargs):
        """Calculate pEW and its error using Monte Carlo method.

        Args:
            min_range (array-like of length 2): wavelength range
             to find blue endpoint.
            max_range (array-like of length2): wavelength range
             to find red endpoint
            Ntimes (int, optional): number of times to run MC. Defaults to 100.
            binsize (int, optional): binsize in AA used to find endpoints.
             Defaults to 30.

        Returns:
            array-like of length 2: pEW and its error.
        """
        pEW_list = []
        for ii in range(Ntimes):
            MC_line = line(*self.simulate_noise_spectrum(**kwargs))
            pEW_list.append(MC_line.calculate_pEW(
                min_range, max_range, binsize=binsize))
        pEW, epEW = np.mean(pEW_list), np.std(pEW_list)

        return pEW, epEW


def atrouswt(wl, fl, j, wavelet=[1.0/16, 1.0/4, 3.0/8, 1.0/4, 1.0/16]):
    """Spectra wavelet transformation.
     See https://arxiv.org/pdf/0907.3171.pdf.
     Calculate c_j from c_(j-1).
     a trous wavelet, input spectrum and j,
      output c_j(k) in the form of spectrum {'wavelengths','flux'}
     input_spectrum should be c_j

    Args:
        wl (array-like): array-like of wavelength.
        fl (array-like): array-like of flux corresponding to given wavelength.
        j ([float]): [order of input spectrum]
        wavelet ([numpy array]): [wavelet used to smooth the spectrum.
        Default: [1.0/16, 1.0/4, 3.0/8, 1.0/4, 1.0/16]]

    Returns:
        array-like wavelength and array-like flux.
    """
    wavelet = np.array(wavelet)
    flux = copy.deepcopy(fl)
    flux = list(flux)+list(flux[::-1])      # periodicity S(k+N)=S(N-k)

    N2 = len(flux)
    for ii in range(len(flux)):
        flux[ii] = flux[(ii-2*2**(j-1)) % N2]*wavelet[0]\
            + flux[(ii-2**(j-1)) % N2]*wavelet[1]\
            + flux[(ii) % N2]*wavelet[2] + flux[(ii+2**(j-1)) % N2] * \
            wavelet[3] + flux[(ii+2*2**(j-1)) % N2]*wavelet[4]
    flux = np.array(flux[0:int(N2/2)])
    return(wl, flux)


def smooth_spectrum(wl, fl, order=2, **kwargs):
    """smooth spectrum using wavelet transformation method.

    Args:
        wl (array-like): array-like of wavelength.
        fl (array-like): array-like of flux corresponding to given wavelength.
        order (int, optional): Order of smoothness. The higher the order,
             the smoother the smoothed spectrum. Defaults to 4.
        **kwargs: see atrouswt.
    Returns:
        array-like wavelength and array-like flux.
    """

    for ii in range(order):
        wl, fl = atrouswt(wl, fl, ii+1, **kwargs)
    return wl, fl


def bin_spectrum(wl, fl, binsize=30):
    """bin spectrum

    Args:
        wl (array-like): array-like of wavelength in AA.
        fl (array-like): array-like of flux corresponding to given wavelength.
        binsize (int, optional): binsize used for binning. Defaults to 30 (AA).

    Returns:
        array-like wavelength and array-like flux after binning.
    """

    bin_wl = np.arange(wl[0], wl[-1], binsize)
    bin_fl = binned_statistic(wl, fl, statistic='mean', bins=bin_wl).statistic
    bin_wl = binned_statistic(wl, wl, statistic='mean', bins=bin_wl).statistic
    return bin_wl, bin_fl


def find_wl_at_max_fl(wl, fl, wave_range, binsize=30):
    """ Given spectrum, and wave_range = [min_wl, max_wl] in AA.
    find wavelength at max flux. This is to avoid false spikes.

    Args:
        wl (array-like): array-like of wavelength in AA.
        fl (array-like): array-like of flux corresponding to given wavelength.
        wave_range (array_like of length 2): lower and upper boundaries of
        wavelength to find maximum.
        binsize (int, optional): binsize (AA) used to bin spectrum.
        Defaults to 30. If None, then no binning is done.

    Returns:
        array_like of length 2: wavelength at flux maximum, max flux.
    """
    if wave_range[1]-wave_range[0] <= binsize*2:
        msg = 'Please set wave_range for find_wl_at_max_fl'\
            ' at least twice larger than binsize!'
        raise ValueError(msg)
    idx = (wl > wave_range[0]) & (wl < wave_range[1])
    fl = np.array(fl)[idx]
    wl = wl[idx]
    if binsize is not None:  # if None, no binning is done.
        wl, fl = bin_spectrum(wl, fl, binsize=binsize)

    idx = np.argmax(fl)
    return wl[idx], fl[idx]


def find_spectrum_error(wl, fl, binsize=50, order=2, **kwargs):
    """Estmiate errors of given spectrum

    Args:
        wl (array-like): list of wavelength
        fl (array-like): list of flux
        binsize (int, optional): binsize used to find the fluctuation of noise
         and use that as error. Defaults to 50.
        order (int, optional): order for smooth spectrum. Defaults to 4.
        **kwargs: see atrouswt.
    Returns:
        array-like: errors at each wavelength pixel.
    """
    _, smoothed_fl = smooth_spectrum(wl, fl, order=order, **kwargs)

    error = np.zeros(len(wl))
    noise = fl/smoothed_fl

    for ii, ww in enumerate(wl):
        idx = (wl > ww-binsize) & (wl < ww+binsize)
        error[ii] = (np.std(noise[idx]))*smoothed_fl[ii]

    return error
