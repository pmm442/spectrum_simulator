"""
Plot reference spectra for one main sequence star of each type (OBAFGKM)

Source: AJ Pickles 1998
https://ui.adsabs.harvard.edu/abs/1998PASP..110..863P/abstract
http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/PASP/110/863
"""

import math
import numpy as np
from scipy.interpolate import interp1d, interp2d

from matplotlib import pyplot as plot
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors

###############################################################################

# Constants
h = 6.626 * 10**(-34)
c = 3.0 * 10**(8)
k_B = 1.38 * 10**(-23)

# Reference wavelength and temperatures
log_stellar_temperatures = {}
log_stellar_temperatures['O'] = 4.55; log_stellar_temperatures['B'] = 4.20; # B should be 4.15
log_stellar_temperatures['A'] = 3.93; log_stellar_temperatures['F'] = 3.83; log_stellar_temperatures['G'] = 3.75
log_stellar_temperatures['K'] = 3.69; log_stellar_temperatures['M'] = 3.55;

ref_wavelength = 555.6
ref_temperature = np.power(10, log_stellar_temperatures['M'])

###############################################################################

### Interpolators ###

def get_radius_interpolator():
    # Using reference values from https://en.wikipedia.org/wiki/Main_sequence
    temperatures = np.array([2660, 3120, 3920, 4410, 5240, 5610, 5780, 5920, 6540, 7240, 8620, 10800, 16400, 30000, 38000])
    radii = np.array([0.13, 0.32, 0.63, 0.74, 0.85, 0.93, 1.0, 1.05, 1.2, 1.3, 1.7, 2.5, 3.8, 7.4, 18.0])

    interpolator = interp1d(temperatures, radii)
    return interpolator
radius_interpolator = get_radius_interpolator()

def O_star_interpolator():
    data = np.loadtxt("stellar_spectra/O9_star.txt")
    wavelengths = data[:, 0]; spectrum = data[:, 1]
    return interp1d(wavelengths, spectrum)
O_interpolator = O_star_interpolator()

def B_star_interpolator():
    data = np.loadtxt("stellar_spectra/B57_star.txt")
    wavelengths = data[:, 0]; spectrum = data[:, 1]
    return interp1d(wavelengths, spectrum)
B_interpolator = B_star_interpolator()

def A_star_interpolator():
    data = np.loadtxt("stellar_spectra/A5_star.txt")
    wavelengths = data[:, 0]; spectrum = data[:, 1]
    return interp1d(wavelengths, spectrum)
A_interpolator = A_star_interpolator()

def F_star_interpolator():
    data = np.loadtxt("stellar_spectra/F2_star.txt")
    wavelengths = data[:, 0]; spectrum = data[:, 1]
    return interp1d(wavelengths, spectrum)
F_interpolator = F_star_interpolator()

def G_star_interpolator():
    data = np.loadtxt("stellar_spectra/G2_star.txt")
    wavelengths = data[:, 0]; spectrum = data[:, 1]
    return interp1d(wavelengths, spectrum)
G_interpolator = G_star_interpolator()

def K_star_interpolator():
    data = np.loadtxt("stellar_spectra/K2_star.txt")
    wavelengths = data[:, 0]; spectrum = data[:, 1]
    return interp1d(wavelengths, spectrum)
K_interpolator = K_star_interpolator()

def M_star_interpolator():
    data = np.loadtxt("stellar_spectra/M2_star.txt")
    wavelengths = data[:, 0]; spectrum = data[:, 1]
    return interp1d(wavelengths, spectrum)
M_interpolator = M_star_interpolator()

stellar_spectra_interpolators = {}
stellar_spectra_interpolators['O'] = O_interpolator; stellar_spectra_interpolators['B'] = B_interpolator
stellar_spectra_interpolators['A'] = A_interpolator; stellar_spectra_interpolators['F'] = F_interpolator; stellar_spectra_interpolators['G'] = G_interpolator
stellar_spectra_interpolators['K'] = K_interpolator; stellar_spectra_interpolators['M'] = M_interpolator

###############################################################################

### Luminosity ###

def planck(wavelength, stellar_type):
    # Wavelength range for interpolator
    start_wavelength = 150.0; end_wavelength = 2450.0
    if stellar_type == 'M':
        start_wavelength = 300.0
    interpolator = stellar_spectra_interpolators[stellar_type]

    # Temperature and Radius
    temperature = np.power(10, log_stellar_temperatures[stellar_type])
    radius = radius_interpolator(temperature)

    # Helper for normalization
    def plancks_law(wavelength, temperature):
        wavelength_meters = wavelength * 10**-9
        return np.power(wavelength_meters, -5.0) / np.expm1(h * c / k_B / wavelength_meters / temperature)
    normalization = plancks_law(ref_wavelength, temperature) / plancks_law(ref_wavelength, ref_temperature)

    # Get spectral radiance
    if start_wavelength <= wavelength and wavelength <= end_wavelength:
        # from reference spectra
        planck_value = interpolator(wavelength) * normalization
    elif wavelength < start_wavelength:
        # from Planck's law
        normalization *= interpolator(start_wavelength) / plancks_law(start_wavelength, temperature)
        planck_value = plancks_law(wavelength, temperature) * normalization
    elif wavelength > end_wavelength:
        # from Planck's law
        normalization *= interpolator(end_wavelength) / plancks_law(end_wavelength, temperature)
        planck_value = plancks_law(wavelength, temperature) * normalization
    return planck_value

def luminosity(wavelength, stellar_type):
    # Luminosity: L ~ R^2 * B(\lambda, temperature)
    temperature = np.power(10, log_stellar_temperatures[stellar_type])
    radius = radius_interpolator(temperature)
    return 4.0 * np.pi * np.power(radius, 2) * planck(wavelength, stellar_type)
vectorized_luminosity = np.vectorize(luminosity)

###############################################################################

### PLOTTING ###

def make_plot(show = False):

    wavelengths = np.linspace(100, 4000, 4000)

    spectrum_o = vectorized_luminosity(wavelengths, 'O')
    spectrum_b = vectorized_luminosity(wavelengths, 'B')
    spectrum_a = vectorized_luminosity(wavelengths, 'A')
    spectrum_f = vectorized_luminosity(wavelengths, 'F')
    spectrum_g = vectorized_luminosity(wavelengths, 'G')
    spectrum_k = vectorized_luminosity(wavelengths, 'K')
    spectrum_m = vectorized_luminosity(wavelengths, 'M')

    plot.plot(wavelengths, spectrum_o, label = "O")
    plot.plot(wavelengths, spectrum_b, label = "B")
    plot.plot(wavelengths, spectrum_a, label = "A")
    plot.plot(wavelengths, spectrum_f, label = "F")
    plot.plot(wavelengths, spectrum_g, label = "G")
    plot.plot(wavelengths, spectrum_k, label = "K")
    plot.plot(wavelengths, spectrum_m, label = "M")

    # Decorate with spectral colors
    alpha_rainbow = 0.03; num_colors = 500

    coordinates = np.linspace(400, 700, num_colors); y_region = np.array([10**(-15), 10**(12)])
    visible_spectrum = np.zeros((num_colors, 2))
    visible_spectrum[:, 0] = coordinates; visible_spectrum[:, 1] = coordinates
    plot.pcolormesh(coordinates, y_region, np.transpose(visible_spectrum), cmap = 'nipy_spectral', alpha = 0.1)

    #plot.xscale('log')
    #plot.yscale('log')

    plot.legend()

    if show:
        plot.show()

make_plot(show = True)


