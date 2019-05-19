# Used the following demo as a base for slider widgets
# https://matplotlib.org/gallery/widgets/slider_demo.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from scipy.interpolate import interp1d, interp2d

# blackbody radiation by temperature
def planckslaw(Ts):
	c = 3*10**8 #m s^-1
	h = 6.63*10**-34 #m^2 kg s^-1
	k = 1.38*10**-23 #m^2 kg s^2 K^-1
	lmbdam = lmbda*10**-10 #m
	Ts = Ts*1000 #K
	B = 2*h*c**2 * lmbdam**-5 * np.power([np.exp(h*c/(lmbdam*k*Ts)) -1],-1)
	B = B[0] #has a weird shape with extra axis out front
	B = B/np.sum(B).reshape(-1,1)	#'normalization'
	flux_p = np.sum(B,axis=0)
	return flux_p

def gaussian(mu, sig):
	dist = np.exp(-np.power(lmbda - mu, 2.)/(2*np.power(sig,2.)))
	return dist/sum(dist)

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

# setting up initial plot for spectrum
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.275)
lmbda = np.arange(10.0, 50000.0, 10.0) #wavelength in Angstroms
flux = np.zeros(len(lmbda))
l, = plt.plot(lmbda, flux, lw=2, color='red')
plt.axis([2000, 10000, 0, 1])

## STAR STUFF ############################################################################
##########################################################################################
# star temp ranges pulled from Wikipedia, assumed star type was gaussian distributed
# about the middle of the range, used mu+3sigma=max (99.7% coverage)to calculate sigma 
# for each type
# https://en.wikipedia.org/wiki/Stellar_classification
# middle of temp ranges and sigmas that should cover all temps in range
# cold stars
mu_astrs = 8.75 #*10^3 K
sig_astrs = 0.42 #*10^3 K
mu_fstrs =  6.75 #*10^3 K
sig_fstrs = 0.25 #*10^3 K
mu_gstrs = 5.6 #*10^3 K
sig_gstrs = 0.13 #*10^3 K
mu_kstrs = 4.45 #*10^3 K
sig_kstrs = 0.25 #*10^3 K
mu_mstrs = 3.05 #*10^3 K
sig_mstrs = 0.22 #*10^3 K
# hot stars
mu_ostrs = 40 #*10^3 K
sig_ostrs = 3.3 #*10^3 K
mu_bstrs = 20 #*10^3 K
sig_bstrs = 3.3 #*10^3 K

# spectral absorption lines in angstroms
# http://cas.sdss.org/dr5/en/proj/basic/spectraltypes/lines.asp
# http://astro.uchicago.edu/~subbarao/newWeb/line.html
Ha = 6600			# Bsome, Astrong, F
Ha_dist = gaussian(Ha, 10)
Hb = 4800			# Bsome, Astrong, F
Hb_dist = gaussian(Hb, 20)
Hg = 4350			# Bsome, Astrong, F
Hg_dist = gaussian(Hg, 10)
Ca_K = 3800			# F
Ca_K_dist = gaussian(Ca_K, 10)
Ca_H = 4000			# F
Ca_H_dist = gaussian(Ca_H, 10)
Ti_O1 = 5050		# Mstrong, K, G
Ti_O1_dist = gaussian(Ti_O1, 10)
#Ti_O2 = 5200
Ti_O3 = 5550		# Mstrong, K, G
Ti_O3_dist = gaussian(Ti_O3, 10)
#Ti_O4 = 5700
Ti_O5 = 6250		# Mstrong, K, G
Ti_O5_dist = gaussian(Ti_O5, 10)
#Ti_O6 = 6300
Ti_O7 = 6800		# Mstrong, K, G
Ti_O7_dist = gaussian(Ti_O7, 10)
#Ti_O8 = 6900
Gband = 4250		# Gstrong, M, K
Gband_dist = gaussian(Gband, 10)
Na = 5800 			# Mvstrong, K
Na_dist = gaussian(Na, 10)
He_neutral = 4200	# B
He_neutral_dist = gaussian(He_neutral, 10)
He_ion = 4400		# O
He_ion_dist = gaussian(He_ion, 10)


# calculate fluxes for stars
a_planck = planckslaw(mu_astrs)
a_absorp = -0.01*(Ha_dist + Hg_dist + Hb_dist)
flux_a = a_planck + a_absorp

f_planck = planckslaw(mu_fstrs)
f_absorp = -0.01*(Ca_K_dist + Ca_H_dist)
flux_f = f_planck + f_absorp

g_planck = planckslaw(mu_gstrs)
g_absorp = -0.01*(Ti_O1_dist + Ti_O3_dist + Ti_O5_dist + Ti_O7_dist + Gband_dist)
flux_g = g_planck + g_absorp

k_planck = planckslaw(mu_kstrs)
k_absorp = -0.01*(Na_dist)
flux_k = k_planck + k_absorp

m_planck = planckslaw(mu_mstrs)
#m_absorp = -0.01*(Ti_O1_dist + Ti_O3_dist + Ti_O5_dist + Ti_O7_dist + Gband_dist + Na_dist)
flux_m = m_planck #+ m_absorp

wavelengths = lmbda / 10.0
spectrum_o = vectorized_luminosity(wavelengths, 'O')
spectrum_b = vectorized_luminosity(wavelengths, 'B')
spectrum_a = vectorized_luminosity(wavelengths, 'A')
spectrum_f = vectorized_luminosity(wavelengths, 'F')
spectrum_g = vectorized_luminosity(wavelengths, 'G')
spectrum_k = vectorized_luminosity(wavelengths, 'K')
spectrum_m = vectorized_luminosity(wavelengths, 'M')

coldfluxes = np.array([flux_a, flux_f, flux_g, flux_k, flux_m])
coldfluxes = np.array([spectrum_a, spectrum_f, spectrum_g, spectrum_k, spectrum_m])
pcoldstrtype = np.array([0.006, 0.03, 0.076, 0.121, 0.765]).reshape(-1,1)
coldfluxes = coldfluxes * pcoldstrtype
coldflux = np.sum(coldfluxes, axis=0)

o_planck = planckslaw(mu_ostrs)
o_absorp = -0.001*(He_ion_dist)
flux_o = o_planck + o_absorp

b_planck = planckslaw(mu_bstrs)
b_absorp = -0.001*(Ha_dist + Hg_dist + He_neutral_dist+ Hb_dist)
flux_b = b_planck + b_absorp

hotfluxes = np.array([flux_o, flux_b])
hotfluxes = np.array([spectrum_o, spectrum_b])
photstrtype = np.array([0.1, 0.9]).reshape(-1,1)
hotfluxes = hotfluxes * photstrtype
hotflux = np.sum(hotfluxes, axis=0)

## GAS STUFF #############################################################################
##########################################################################################
# Ha_dist same as above
# Hb_dist same as above
OII = 3730
OII_dist = gaussian(OII, 10)
OIII1 = 4960
OIII1_dist = gaussian(OIII1, 10)
OIII2 = 5010
OIII2_dist = gaussian(OIII2, 10)
SII = 6720
SII_dist = gaussian(SII, 10)
gasflux = 0.01*(8*Ha_dist + 6*Hb_dist + 4*OII_dist + OIII1_dist + 3*OIII2_dist + 2*SII_dist)

# setting up sliders
step = 1
val0 = 0
axcolor = 'lightgoldenrodyellow'
axhotstr = plt.axes([0.25, 0.075, 0.65, 0.03], facecolor=axcolor)
axcoldstr = plt.axes([0.25, 0.125, 0.65, 0.03], facecolor=axcolor)
axgas = plt.axes([0.25, 0.175, 0.65, 0.03], facecolor=axcolor)

shotstr = Slider(axhotstr, 'Hot Stars', 0, 10, valinit=val0, valstep=step)
scoldstr = Slider(axcoldstr, 'Colder Stars', 0, 10, valinit=val0, valstep=step)
sgas = Slider(axgas, 'Gas', 0, 10, valinit=val0, valstep=step)

def update(val):
	hots = 40*shotstr.val
	colds = 100*scoldstr.val
	if shotstr.val>0 and sgas.val>0:
		gas = 0.5*shotstr.val + 1.5*sgas.val
	else:
		gas = 0
	flux = colds*coldflux + hots*hotflux + gas*gasflux

	normalization_index = np.searchsorted(lmbda, 5556)
	normalization = flux[normalization_index]

	flux /= normalization
	ax.set_ylim(0, 2)
	print(min(flux), max(flux))

	l.set_ydata(flux)
	fig.canvas.draw_idle()

shotstr.on_changed(update)
scoldstr.on_changed(update)
sgas.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
	global flux
	shotstr.reset()
	scoldstr.reset()
	sgas.reset()
	flux = np.zeros(len(lmbda))
	l.set_ydata(flux)
	fig.canvas.draw_idle()
button.on_clicked(reset)

plt.show()