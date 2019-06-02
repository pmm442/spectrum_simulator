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

def get_dust_interpolator():
    fn = "dust_emission/spec_12.dat"
    data = np.loadtxt(fn)

    wavelengths = np.zeros(len(data[:, 0]) + 1)
    dust_emission_data = np.zeros(len(data[:, 0]) + 1)

    wavelengths[0] = 0.05
    dust_emission_data[0] = -20

    wavelengths[1:] = data[:, 0] * 1000.0 # Convert from um to nm
    dust_emission_data[1:] = data[:, 1] + 28 # arbitary normalization

    dust_emission_interpolator = interp1d(wavelengths, dust_emission_data)
    return dust_emission_interpolator
dust_interpolator = get_dust_interpolator()

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
    data = np.loadtxt("stellar_spectra/B3_star.txt")
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
    interpolator = stellar_spectra_interpolators[stellar_type]

    # Temperature and Radius
    temperature = np.power(10, log_stellar_temperatures[stellar_type])
    radius = radius_interpolator(temperature)

    # Helper for normalization
    def plancks_law(wavelength, temperature):
        wavelength_meters = wavelength * 10**-9
        return np.power(wavelength_meters, -5.0) / np.expm1(h * c / k_B / wavelength_meters / temperature)
    normalization = plancks_law(ref_wavelength, temperature) / plancks_law(ref_wavelength, ref_temperature)

    planck_value = interpolator(wavelength) * normalization
    return planck_value

def luminosity(wavelength, stellar_type):
    # Luminosity: L ~ R^2 * B(\lambda, temperature)
    temperature = np.power(10, log_stellar_temperatures[stellar_type])
    radius = radius_interpolator(temperature)
    return 4.0 * np.pi * np.power(radius, 2) * planck(wavelength, stellar_type)
#vectorized_luminosity = np.vectorize(luminosity)

###############################################################################

def dust_extinction(wavelength, A_v, R_v = 4.05):
    # Calzetti Extinction Curve
    # Source: http://webast.ast.obs-mip.fr/hyperz/hyperz_manual1/node10.html
    # Source: https://ned.ipac.caltech.edu/level5/Sept12/Calzetti/Calzetti1_4.html
    wavelength_u = wavelength * 10**(-3) # convert nm to um
    k_lambda = 0 # extinction curve

    one = np.floor(np.power(np.floor(wavelength_u / 0.63), 0.0001)) # if wavelength_u < 0.63, one = 1 and two = 0
    two = 1 - one # else one = 0 and two = 1

    k_lambda1 = 2.659 * (-2.156 + 1.509 / wavelength_u - 0.198 / wavelength_u**2 + 0.011 / wavelength_u**3) + R_v
    k_lambda2 = 2.659 * (-1.857 + 1.040 / wavelength_u) + R_v

    k_lambda = one * k_lambda1 + two * k_lambda2
    return np.power(10, -0.4 * k_lambda * A_v / R_v)

def dust_emission(wavelength):
	return np.power(10, dust_interpolator(wavelength)) # interpolator gives log of emission

def normalize_dust_emission(wavelengths, dust_emission, integrated_dust_emission, flux, flux_extincted):
	optical_cutoff = np.searchsorted(wavelengths, 1000)
	flux_absorbed = flux - flux_extincted

	integrated_flux_absorbed = 0
	for i, flux_absorbed_i in enumerate(flux_absorbed[:optical_cutoff]):
		d_wavelength = wavelengths[i+1] - wavelengths[i]
		integrated_flux_absorbed += flux_absorbed_i * d_wavelength

	normalization = integrated_flux_absorbed / integrated_dust_emission
	return dust_emission * normalization

###############################################################################

# setting up initial plot for spectrum
fig, ax = plt.subplots(figsize = (8, 6))
fig.canvas.set_window_title("Galaxy Spectrum Tool")
plt.subplots_adjust(left=0.30, bottom=0.375)

optical_wavelengths = np.linspace(125, 1200, 2000) 
ir_wavelengths = np.logspace(np.log10(1200), np.log10(1000000), 10000)
lmbda = np.concatenate((optical_wavelengths, ir_wavelengths))

#lmbda = np.arange(1000.0, 50000.0, 10.0) #wavelength in Angstroms
flux = np.zeros(len(lmbda))
flux_extincted = np.zeros(len(lmbda))
l_extincted, = plt.plot(lmbda, flux_extincted, lw=2, color = 'r', label = "w/ dust")
l, = plt.plot(lmbda, flux, lw=2, color='b', label = "w/o dust")

plt.legend(loc = "upper right")

start_x = 100; end_x = 1000
ax.set_xlim([start_x, end_x])
ax.set_ylim([0, 1])
scales = {}; scales['x'] = 'linear'; scales['y'] = 'linear'; scales['vary'] = True

fontsize = 16
ax.set_xlabel(r'Wavelength $\lambda$ [nm]', fontsize = fontsize)
ax.set_ylabel(r'Luminosity [$L / L_{556\ nm}$]', fontsize = fontsize + 1)

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
Ha = 660			# Bsome, Astrong, F
Ha_dist = gaussian(Ha, 1)
Hb = 480			# Bsome, Astrong, F
Hb_dist = gaussian(Hb, 2)
Hg = 435			# Bsome, Astrong, F
Hg_dist = gaussian(Hg, 1)
Ca_K = 380			# F
Ca_K_dist = gaussian(Ca_K, 1)
Ca_H = 400			# F
Ca_H_dist = gaussian(Ca_H, 1)
Ti_O1 = 505		# Mstrong, K, G
Ti_O1_dist = gaussian(Ti_O1, 1)
#Ti_O2 = 5200
Ti_O3 = 555		# Mstrong, K, G
Ti_O3_dist = gaussian(Ti_O3, 1)
#Ti_O4 = 5700
Ti_O5 = 625		# Mstrong, K, G
Ti_O5_dist = gaussian(Ti_O5, 1)
#Ti_O6 = 6300
Ti_O7 = 680		# Mstrong, K, G
Ti_O7_dist = gaussian(Ti_O7, 1)
#Ti_O8 = 6900
Gband = 4250		# Gstrong, M, K
Gband_dist = gaussian(Gband, 1)
Na = 580 			# Mvstrong, K
Na_dist = gaussian(Na, 1)
He_neutral = 420	# B
He_neutral_dist = gaussian(He_neutral, 1)
He_ion = 440		# O
He_ion_dist = gaussian(He_ion, 1)


# calculate fluxes for stars
flux_a = planckslaw(mu_astrs)
a_absorp = -0.01*(Ha_dist + Hb_dist + Hg_dist)
flux_a = flux_a + a_absorp

flux_f = planckslaw(mu_fstrs)
f_absorp = -0.01*(Ca_K_dist + Ca_H_dist)
flux_f = flux_f + f_absorp

flux_g = planckslaw(mu_gstrs)
g_absorp = -0.01*(Ti_O1_dist + Ti_O3_dist + Ti_O5_dist + Ti_O7_dist + Gband_dist)
flux_g = flux_g + g_absorp

flux_k = planckslaw(mu_kstrs)
k_absorp = -0.01*(Na_dist)
flux_k = flux_k + k_absorp

m_planck = planckslaw(mu_mstrs)
#m_absorp = -0.01*(Ti_O1_dist + Ti_O3_dist + Ti_O5_dist + Ti_O7_dist + Gband_dist + Na_dist)
flux_m = m_planck #+ m_absorp

wavelengths = lmbda #/ 10.0
#spectrum_o = vectorized_luminosity(wavelengths, 'O')
#spectrum_b = vectorized_luminosity(wavelengths, 'B')
#spectrum_a = vectorized_luminosity(wavelengths, 'A')
#spectrum_f = vectorized_luminosity(wavelengths, 'F')
#spectrum_g = vectorized_luminosity(wavelengths, 'G')
#spectrum_k = vectorized_luminosity(wavelengths, 'K')
#spectrum_m = vectorized_luminosity(wavelengths, 'M')

spectrum_o = luminosity(wavelengths, 'O')
spectrum_b = luminosity(wavelengths, 'B')
spectrum_a = luminosity(wavelengths, 'A')
spectrum_f = luminosity(wavelengths, 'F')
spectrum_g = luminosity(wavelengths, 'G')
spectrum_k = luminosity(wavelengths, 'K')
spectrum_m = luminosity(wavelengths, 'M')

### Young Stars ###

coldfluxes = np.array([flux_a, flux_f, flux_g, flux_k, flux_m])
coldfluxes = np.array([spectrum_a, spectrum_f, spectrum_g, spectrum_k, spectrum_m])

#pcoldstrtype = np.array([0.006, 0.03, 0.076, 0.121, 0.765]).reshape(-1,1)
pcoldstrtype = np.array([0.198, 0.232, 0.299, 1.160, 6.275]).reshape(-1,1)
pcoldstrtype /= np.sum(pcoldstrtype)
coldfluxes = coldfluxes * pcoldstrtype
coldflux = np.sum(coldfluxes, axis=0)

### Old Stars ###

oldfluxes = np.array([flux_g, flux_k, flux_m])
oldfluxes = np.array([spectrum_g, spectrum_k, spectrum_m])

poldstrtype = np.array([0.299, 1.160, 6.275]).reshape(-1,1)
poldstrtype /= np.sum(pcoldstrtype)
oldfluxes = oldfluxes * poldstrtype
oldflux = np.sum(oldfluxes, axis=0)

### Brand-New Stars ###

o_planck = planckslaw(mu_ostrs)
o_absorp = -0.001*(He_ion_dist)
flux_o = o_planck + o_absorp

b_planck = planckslaw(mu_bstrs)
b_absorp = -0.001*(Ha_dist + Hg_dist + He_neutral_dist+ Hb_dist)
flux_b = b_planck + b_absorp

hotfluxes = np.array([flux_o, flux_b])
hotfluxes = np.array([spectrum_o, spectrum_b])
#photstrtype = np.array([0.1, 0.9]).reshape(-1,1)
photstrtype = np.array([0.016, 0.254]).reshape(-1,1)
photstrtype /= (np.sum(photstrtype)) #+ np.sum(pcoldstrtype))
hotfluxes = hotfluxes * photstrtype
hotflux = np.sum(hotfluxes, axis=0)

## GAS STUFF #############################################################################
##########################################################################################
# Ha_dist same as above
# Hb_dist same as above

#### hot gas
OII = 373
OII_dist = gaussian(OII, 1)
OIII1 = 496
OIII1_dist = gaussian(OIII1, 1)
OIII2 = 501
OIII2_dist = gaussian(OIII2, 1)
SII = 672
SII_dist = gaussian(SII, 1)
gasflux = 2*(6*Ha_dist + 6*Hb_dist + 4*OII_dist + OIII1_dist + 3*OIII2_dist + 2*SII_dist)

#### cold gas
NaIa = 589
NaIa_dist = gaussian(NaIa, 1)
NaIb = 589.6
NaIb_dist = gaussian(NaIb, 1)
CaIIa = 393.3
CaIIa_dist = gaussian(CaIIa, 1)
CaIIb = 396.8
CaIIb_dist = gaussian(CaIIb, 1)
coldgasflux = (NaIa_dist + NaIb_dist + CaIIa_dist + CaIIb_dist)
## DUST STUFF #############################################################################
##########################################################################################

dustflux = dust_emission(wavelengths)

ir_cutoff = np.searchsorted(wavelengths, 1200)
integrated_dustflux = 0
for i, dustflux_i in enumerate(dustflux[ir_cutoff:]):
	d_wavelength = wavelengths[ir_cutoff + i] - wavelengths[ir_cutoff + i-1]
	integrated_dustflux += dustflux_i * d_wavelength

##########################################################################################

# Rainbow Region
alpha_rainbow = 0.10; num_colors = 500

coordinates = np.linspace(400, 700, num_colors); y_region = np.array([10**(-6), 10**5])
visible_spectrum = np.zeros((num_colors, 2))
visible_spectrum[:, 0] = coordinates; visible_spectrum[:, 1] = coordinates
ax.pcolormesh(coordinates, y_region, np.transpose(visible_spectrum), cmap = 'nipy_spectral', alpha = alpha_rainbow)

# setting up sliders
step = 1
val0 = 0
axcolor = 'lightgoldenrodyellow'
axhotstr = plt.axes([0.25, 0.075, 0.25, 0.03]) #, facecolor=axcolor)
axcoldstr = plt.axes([0.25, 0.125, 0.25, 0.03]) #, facecolor=axcolor)
axoldstr = plt.axes([0.25, 0.175, 0.25, 0.03]) #, facecolor=axcolor)
axhotgas = plt.axes([0.70, 0.125, 0.20, 0.03]) #, facecolor=axcolor)
axcoldgas = plt.axes([0.70, 0.075, 0.20, 0.03]) #, facecolor=axcolor)
axdust = plt.axes([0.70, 0.175, 0.20, 0.03]) #, facecolor=axcolor)

shotstr = Slider(axhotstr, 'Brand-New Stars', 0, 10, valinit=val0)
scoldstr = Slider(axcoldstr, 'Young Stars', 0, 10, valinit=val0)
soldstr = Slider(axoldstr, 'Old Stars', 0, 10, valinit=val0)
shotgas = Slider(axhotgas, 'Hot Gas', 0, 10, valinit=val0)
scoldgas = Slider(axcoldgas, 'Cold Gas', 0, 10, valinit=val0)
sdust = Slider(axdust, 'Dust', 0, 2, valinit=val0)

def update(val):
	hots = 10**shotstr.val - 1
	colds = 10**scoldstr.val - 1
	olds = 10**soldstr.val - 1

	#hots = 40*shotstr.val
	#colds = 100*scoldstr.val

	flux = olds*oldflux + colds*coldflux + hots*hotflux
	
	normalization_index = np.searchsorted(lmbda, 555.6)
	normalization = flux[normalization_index]

	if normalization != 0:
		flux /= normalization

	start_UV = np.searchsorted(lmbda, 100); end_UV = np.searchsorted(lmbda, 400)
	max_flux_UV = max(flux[start_UV:end_UV])
	if shotgas.val>0:
		frac_coefficient = 3.0 * (shotstr.val / (shotstr.val + scoldstr.val / 2.0 + soldstr.val / 2.0 + 0.0001)) 
		UV_coefficient = (max_flux_UV + 1.0)**0.15 - 1.0 # scale with UV flux (sort of)
		gas = (0.5 + 1.5*shotgas.val)*frac_coefficient*UV_coefficient
	else:
		gas = 0
	flux += gas*gasflux

	if scoldgas.val>0:
		coldgas = 0.1*scoldgas.val #arbitrary normalization
		flux -= coldgas*coldgasflux

	dust_Av = sdust.val
	flux_extincted = flux * dust_extinction(wavelengths, dust_Av)

	if dust_Av > 0:
		scaled_dust_emission = normalize_dust_emission(wavelengths, dustflux, integrated_dustflux, flux, flux_extincted)
		flux_extincted += scaled_dust_emission

	if scales['vary']:
		change_axes(flux)

	l.set_ydata(flux)
	l_extincted.set_ydata(flux_extincted)

	fig.canvas.draw_idle()

shotstr.on_changed(update)
scoldstr.on_changed(update)
soldstr.on_changed(update)
shotgas.on_changed(update)
scoldgas.on_changed(update)
sdust.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
	global flux
	shotstr.reset()
	scoldstr.reset()
	soldstr.reset()
	shotgas.reset()
	scoldgas.reset()
	sdust.reset()
	flux = np.zeros(len(lmbda))
	l.set_ydata(flux)
	l_extincted.set_ydata(flux)
	fig.canvas.draw_idle()
button.on_clicked(reset)

rax = plt.axes([0.01, 0.70, 0.15, 0.10]) #, facecolor=axcolor)
radio_x = RadioButtons(rax, (r'Linear $\lambda$', r'Log $\lambda$'), active=0)

ray = plt.axes([0.01, 0.55, 0.15, 0.10]) #, facecolor=axcolor)
radio_y = RadioButtons(ray, ('Linear L', 'Log L'), active=0)

rav = plt.axes([0.01, 0.40, 0.15, 0.10]) #, facecolor=axcolor)
radio_vary = RadioButtons(rav, ('Vary L-axis', 'Lock L-axis'), active=0)

def change_axes(flux):
	if scales['x'] == 'linear':
		ax.set_xscale('linear')
		ax.set_xlim([100, 1000])
	else:
		ax.set_xscale('log')
		ax.set_xlim([100, 100000])

	if scales['y'] == 'linear':
		ax.set_yscale('linear')
		if scales['x'] == 'linear':
			start_visible = np.searchsorted(lmbda, start_x); end_visible = np.searchsorted(lmbda, end_x)
			visible_flux = flux[start_visible:end_visible]
			max_y = max(visible_flux)
			if max_y < 1:
				max_y = 1
			ax.set_ylim([0, max_y])
		else:
			max_y = np.max(flux)
			if max_y < 1 or max_y is np.nan:
				max_y = 1
			ax.set_ylim([0, max_y])
	else:
		ax.set_yscale('log')
		ax.set_ylim([10**(-4), 10**(2)])

def changexaxis(label):
	if 'Linear' in label:
		scales['x'] = 'linear'
		#ax.set_xscale('linear')
		#ax.set_xlim([3500, 8000])
	elif 'Log' in label:
		scales['x'] = 'log'
		#ax.set_xscale('log')
		#ax.set_xlim([1000, 50000])
	change_axes(l.get_ydata())
	fig.canvas.draw_idle()

def changeyaxis(label):
	if 'Linear' in label:
		scales['y'] = 'linear'
		#ax.set_ylim([0, 2])
		#ax.set_yscale('linear')
	elif 'Log' in label:
		scales['y'] = 'log'
		#ax.set_ylim([10**(-5), 10**(2)])
		#ax.set_yscale('log')
	change_axes(l.get_ydata())
	fig.canvas.draw_idle()

def vary_yaxis(label):
	if label == 'Vary L-axis':
		scales['vary'] = True
	elif label == 'Lock L-axis':
		scales['vary'] = False

radio_x.on_clicked(changexaxis)
radio_y.on_clicked(changeyaxis)
radio_vary.on_clicked(vary_yaxis)

plt.show()