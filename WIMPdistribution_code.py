import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.special as special
from scipy.special import gamma
from scipy import interpolate

###
###  This code computes the density of WIMPs around a PBH
###  We use the setup introduced in 1607.00612, see also 1712.06383
###  We have used the code for the results in 20XX.XXXXX
###
###  Please send questions to luca.visinelli@gmail.com
###

# constants
cm = 1.
s  = 1.
g  = 1.
pi2    = 2.*np.pi
pi4    = 4.*np.pi
pi8    = 8.*np.pi
hbarc  = 1.97e-14 #conversion from GeV^-1 to cm
speedc = 3.e10    #light speed in cm/s
hbar   = hbarc/speedc # hbar in GeV s
t0     = 13.7*3.15e16 # Age of the Universe in seconds
h      = 0.7
protonmass = 1.78e-24 # Conversion from GeV to g
sigmav  = 3.e-26 # Annihilation cross section in cm^3/s
mGeV    = 100.
mDM     = mGeV*protonmass
MPl     = 1.221e19 # Planck mass in GeV
alpha   = np.sqrt(pi4**3*61.75/180.)
sKD100  = (alpha*mGeV/MPl)**0.25/gamma(0.75)
MSun    = 2.e33    # Mass of the Sun in g
MBH     = MSun;
rSun    = 3.e5     # Schwarzchild radius of the Sun in cm
pc      = 3.086e18 # conversion from parsec to cm
H0      = 1.e7/(1.e6*pc)*h # Hubble constant in s^-1
OmegaDM = 0.27
OmegaM  = 0.31
OmegaR  = 5.e-5
OmegaL  = 1.-OmegaM-OmegaR
Rhoc    = 3./(pi8)*(MPl*hbar*H0)**2 # Critical energy density in GeV^4
RhoDM   = OmegaDM*Rhoc*protonmass/hbarc**3 # DM density in g/cm^3
Rhoeq   = 5.66e-19 # Density at zeq in g/cm^3
r0      = 0.0193*pc # cm

# m is the WIMP mass in units of 100GeV
# M is the PBH mass in solar masses
# z is the redshift

def h(z):
    # Hubble rate in units of H0 as a function of z
    return np.sqrt(OmegaL+OmegaM*(1.+z)**3+OmegaR*(1.+z)**4)

def rg(M):
    # Schwarzschild radius in cm
    return rSun*M

def tdec(m):
    # WIMP decoupling time in s
    # See 1501.02233
    TKD=sKD100*mGeV*m**1.25
    return hbar*MPl/(alpha*TKD**2)

def rdec(m):
    # decoupling radius in cm
    return 3.*tdec(m)*speedc

def RhoMax(m):
    # Energy density of WIMPs today in g/cm^3
    # See V. Berezinsky, A. Gurevich, and K. Zybin, Phys. Lett. B 294, 221 (1992)
    return m*mDM/(sigmav*t0)

def rinfl(M, m):
    # Radius of influence in cm
    return (rg(M)*rdec(m)**2)**(1./3.)

def xmax(M,m):
    # extent of the profile, dimensionless
    return rinfl(M, m)/rg(M)
xmax = np.vectorize(xmax)

def rhoD(m):
    # WIMP density at kinetic decoupling in g/cm^3
    return 0.5*Rhoeq*(r0**3/(rSun*rdec(m)**2))**0.75

def rhotot(xi, M, m):
    # WIMP density profile around PBH at kinetic decoupling in g/cm^3
    # xi is dimensionless
    return rhoD(m)*min(1., (xmax(M, m)/xi)**2.25)
rhotot = np.vectorize(rhotot)

def Sigma2(xi, M, m):
    # WIMP velocity dispersion squared in units of c
    return sKD100*min(1., (xmax(M, m)/xi)**1.5)*m**0.25
Sigma2 = np.vectorize(Sigma2)

def fv2(w, xi, M, m):
    # WIMP velocity distribution
    ss = Sigma2(xi, M, m)
    return np.exp(-0.5*w/ss)/(pi2*ss)**(3./2.)
fv2 = np.vectorize(fv2)

##
##  1D integration approximation
##

def integrand1d(xi, x, M, m):
    a  = 0.
    if xi>x: #and x < rinfl(M,m)/rg(M):
      a = 8./(x**1.5)*xi*rhotot(xi,M,m)/np.sqrt(xi-x)/pi4
    return a
integrand1d = np.vectorize(integrand1d)

def rhoF1d(x, M, m):
    xM = xmax(M,m)
    rt = np.geomspace(x, 1.e40, 200)
    a  = np.trapz(integrand1d(rt, x, M, m), rt)
    return min(a,RhoMax(m))
rhoF1d = np.vectorize(rhoF1d)

g1d = lambda x, M, m: rg(M)**3*pi4*x**2*(rhoF1d(x, M, m))**2*sigmav/(m*mDM)**2
g1d = np.vectorize(g1d)
def Gamma1d(M, m):
    xt = np.geomspace(1.e-10, 1.e20, 100)
    return np.trapz(g1d(xt, M, m) , xt )
Gamma1d = np.vectorize(Gamma1d)

##
## Integrand of the WIMP density distribution
##

def integrand3d(w, xi, x, M, m):
    # w is v**2
    ffv = fv2(w,xi,M,m)
    a   = 0.
    if xi*w<1.:
      z  = xi/(1. - xi*w)
      if z>x: 
        xm = x/xi*np.sqrt((1/x-1/z)/w)
        if xm < 1.:
          ln = np.log((1.+xm)/(1.-xm))
          a  = 2.*xi/x*ffv*rhotot(xi,M,m)/z**1.5*ln
    return a
integrand3d = np.vectorize(integrand3d)

def rhoF3d(x, M, m):
    # Integration over the region xi<x
    NT = 100
    xc = x*np.geomspace(1.e-15, 1., NT)
    ll = np.zeros(NT)
    for i, xi in enumerate(xc):
      w    = np.linspace((x-xi)/x/xi, x/xi/(x+xi), NT)
      ll[i] = np.trapz(integrand3d(w, xi, x, M, m), w)
    a  = np.trapz(ll, xc)
    # Integration over the region xi>x
    xc = x*np.geomspace(1., 1.e22, NT)
    for i, xi in enumerate(xc):
      w    = np.linspace(x/xi/(x+xi), 1./xi, NT)
      ll[i] = np.trapz(integrand3d(w, xi, x, M, m), w)
    a  = a + np.trapz(ll, xc)
    return min(a,RhoMax(m))
rhoF3d = np.vectorize(rhoF3d)

def f3d(x, M, m):
    return rg(M)**3*pi4*x**2*(rhoF3d(x, M, m))**2*sigmav/(m*mDM)**2
f3d = np.vectorize(f3d)

def Gamma3d(M, m):
    # in s^-1
    a  = xmax(M, m)
    xm = 1.e3
    int = (xm*rinfl(M,m))**3*pi4*RhoMax(m)**2*sigmav/(m*mDM)**2/3.
    xt = np.geomspace(xm*a, 1.e22*a, 30)
    return int + np.trapz( f3d(xt, M, m) , xt )
Gamma3d = np.vectorize(Gamma3d)
