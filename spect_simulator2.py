# Used the following demo as a base for slider widgets
# https://matplotlib.org/gallery/widgets/slider_demo.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

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

# setting up initial plot for spectrum
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.275)
lmbda = np.arange(10.0, 50000.0, 10.0) #wavelength in Angstroms
flux = np.zeros(len(lmbda))
l, = plt.plot(lmbda, flux, lw=2, color='red')
plt.axis([3500, 8000, 0, 1])

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

coldfluxes = np.array([flux_a, flux_f, flux_g, flux_k, flux_m])
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