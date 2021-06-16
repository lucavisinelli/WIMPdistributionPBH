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
###  We have used the code for the results in:
###  2008.08077
###  2011.01930
###
###  Please send questions to:
###  luca.visinelli@sjtu.edu.cn
###

# constants
cm = 1.
s  = 1.
g  = 1.
pi2     = 2.*np.pi
pi4     = 4.*np.pi
pi8     = 8.*np.pi
eps     = 1.e-25
bigN    = 1./eps
hbarc   = 1.97e-14     #conversion from GeV^-1 to cm
speedc  = 3.e10        #light speed in cm/s
hbar    = hbarc/speedc # hbar in GeV s
t0      = 13.7*3.15e16 # Age of the Universe in seconds
h       = 0.7          # Hubble constant in units of 100 km/s/Mpc
mproton = 1.78e-24     # Conversion from GeV to g
sigmav  = 2.5e-26      # Annihilation cross section in cm^3/s
mGeV    = 100.         # Reference WIMP mass in GeV
mDM     = mGeV*mproton # Reference WIMP mass in g
MPl     = 1.221e19     # Planck mass in GeV
MSun    = 2.e33        # Mass of the Sun in g
rSun    = 3.e5         # Schwarzchild radius of the Sun in cm
pc      = 3.086e18     # conversion from parsec to cm
H0      = 10.*h/pc     # Hubble constant in s^-1
OmegaDM = 0.27         # DM fraction today
OmegaM  = 0.31         # Total matter fraction today
OmegaR  = 5.e-5        # Radiation fraction today
OmegaL  = 1.-OmegaM-OmegaR
Rhoc    = 3./(pi8)*(MPl*hbar*H0)**2     # Critical energy density in GeV^4
RhoDM   = OmegaDM*Rhoc*mproton/hbarc**3 # DM density in g/cm^3
Rhoeq   = 5.66e-19     # Density at zeq in g/cm^3
r0      = 0.0193*pc    # cm
teq     = 1.25629e12   # Age of the Universe at matter-radiation equality in s
alpha   = np.sqrt(pi4**3*61.75/180.)
sKD100  = (alpha*mGeV/MPl)**0.25/gamma(0.75)

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
    # WIMP decoupling radius in cm
    return 3.*tdec(m)*speedc

def rK(M, m):
    # Radius at which WIMP kinetic energy is negligible in cm
    # See Eq.14 in 2011.01930
    return (2.*speedc*tdec(m)*sKD100*m**0.25)**2/rg(M)

def RhoMax(m):
    # Energy density of WIMPs today in g/cm^3
    # See V. Berezinsky, A. Gurevich, K. Zybin, Phys. Lett. B 294, 221 (1992)
    return m*mDM/(sigmav*t0)

def rinfl(M, m):
    # Radius of influence in cm
    # See Eq.8 in 2011.01930
    return (rg(M)*rdec(m)**2)**(1./3.)

def xmax(M,m):
    # extent of the profile, dimensionless
    return rinfl(M, m)/rg(M)
xmax = np.vectorize(xmax)

def rhoD(m):
    # WIMP density at kinetic decoupling in g/cm^3
    return 0.5*Rhoeq*(r0**3/(rSun*rdec(m)**2))**0.75

def rhospike(xi, M, m):
    # WIMP distribution at the spike r^(-9/4)
    # See Eq.10 in 2011.01930
    return (xmax(M, m)/xi)**2.25

def rhotot(xi, M, m):
    # WIMP density profile around PBH at kinetic decoupling in g/cm^3
    # xi is dimensionless
    # See Eq.11 in 2011.01930
    return rhoD(m)*rhospike(xi, M, m)/(1.+rhospike(xi, M, m))
rhotot = np.vectorize(rhotot)

def Sigma2(xi, M, m):
    # WIMP velocity dispersion squared in units of c^2
    return sKD100*m**0.25*min(1., (xmax(M, m)/xi)**1.5)
Sigma2 = np.vectorize(Sigma2)

def fv2(w, xi, M, m):
    # WIMP velocity distribution
    # w = v_i^2
    ss = Sigma2(xi, M, m)
    return np.exp(-0.5*w/ss)/(pi2*ss)**1.5
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
    # See Eq.A29 in 2011.01930
    xM = xmax(M,m)
    rt = np.geomspace(x, 1.e40, 200)
    a  = np.trapz(integrand1d(rt, x, M, m), rt)
    return min(a,RhoMax(m))
rhoF1d = np.vectorize(rhoF1d)

##
## Integrand of the WIMP density distribution
##

def integrand3d(X, Y, x, M, m):
    # X = x*w
    # Y = xi/x
    a   = 0.
    if (Y<1. and Y<1./X and X>1./Y/(1.+Y)) or (Y>=1. and X<1./Y/(1.+Y)):
      w  = X/x
      xi = Y*x
      ffv = fv2(w,xi,M,m)
      rhi = rhotot(xi,M,m)
      xm = np.sqrt(1.+1./X-1./X/Y)/Y
      if xm > 1.:
        ln = np.log((xm+1.)/(xm-1.))
        a  = 2./np.sqrt(x**3*Y)*ffv*rhi*(1.-X*Y)**1.5*ln
    return a
integrand3d = np.vectorize(integrand3d)

def rhoF3d(x, M, m):
    # WIMP density spike around the PBH
    # See Eq.A10 in 2011.01930
    NT = 400
    ll = np.zeros(NT)
    # Integration over the region xi<x
    xc = np.geomspace(eps, 1., NT)
    for i, Y in enumerate(xc):
      X     = np.linspace(1./(1.+Y), 1., NT)/Y
      ll[i] = np.trapz(integrand3d(X, Y, x, M, m), X)
    a  = np.trapz(ll, xc)
    # Integration over the region xi>x
    xc = np.geomspace(1., 1.e30, NT)
    for i, Y in enumerate(xc):
      X     = np.linspace(eps, 1., NT)/Y/(1.+Y)
      ll[i] = np.trapz(integrand3d(X, Y, x, M, m), X)
    a  = a + np.trapz(ll, xc)
    return a*RhoMax(m)/(a+RhoMax(m))
rhoF3d = np.vectorize(rhoF3d)

def f3d(x, M, m):
    return pi4*(x*rhoF3d(x, M, m)/(m*mDM))**2*sigmav*rg(M)**3
f3d = np.vectorize(f3d)

def Gamma3d(M, m):
    # WIMP annihilation rate around the PBH in s^-1
    # See Eq.25 in 2011.01930
    a  = xmax(M, m)
    xt = np.geomspace(1.1e-8*a, 1.e20*a, 25)
    return np.trapz( f3d(xt, M, m) , xt )
Gamma3d = np.vectorize(Gamma3d)

Mt = 1.e-9
mt = 10.
a  = xmax(Mt, mt)
xt = np.geomspace(1.e-6*a, 1.e12*a, 30)
zt = rg(Mt)*xt
rt = rhoF3d(xt, Mt, mt)
rtt = rhotot(xt, Mt, mt) 
plt.loglog(zt, rt,  'ro')
plt.loglog(zt, rtt, 'k-')
plt.show()
exit()

#  MBH = 1.e-12
rm12   = [4.2e9, 3.1e5, 6.1e2, 1.3e-4, 1.2e-5, 8.8e-7]
rhom12 = [1e-22, 1.6e-16, 9.3e-7, 9.0e3, 1.e5, 8.9e5]

# MBH = 1.e-8
rm8   = [8.6e13, 6.3e12, 1.5e12, 8.5e5, 1.7e4, 1.6, 8.9e-3]
rhom8 = [1.1e-22, 4.8e-20, 6.e-19, 4.6e-9, 3.9e-3, 7.9e3, 8.9e5]

# MBH = 1.e-2
rm2   = [1.e16, 2.5e6, 1.6e6, 7.8e3]
rhom2 = [1.3e-19, 3.42e2, 2.1e3, 7.9e5]

# MBH = 1.e1
rp1   = [1.e16, 2.6e15, 1.9e7]
rhop1 = [6.e-17, 1.1e-14, 2.1e4]

abs_path  = '/Users/visinelli/Data/'

pl = 0
if pl == 1:
  Mt=1.e2
  mt=0.1
  a=xmax(Mt, mt)
  xt = np.geomspace(1.e-3*a, 1.e20*a, 30)
  rt = rg(Mt)*xt
  rht = rhoF3d(xt, Mt, mt)
  plt.loglog(rt, rht,'y-')
  data = np.array([rt, rht])
  data = data.T
  int_file = abs_path + 'SolarProfile10_m1e1.txt'
  np.savetxt(int_file, data, header="radius in cm, Density in g/cm^3")
  mt=1.
  a=xmax(Mt, mt)
  xt = np.geomspace(1.1e-1*a, 1.1e20*a, 25)
  rt = rg(Mt)*xt
  rht = rhoF3d(xt, Mt, mt)
  plt.loglog(rt, rht,'k-')
  data = np.array([rt, rht])
  data = data.T
  int_file = abs_path + 'SolarProfile10_m1e2.txt'
  np.savetxt(int_file, data, header="radius in cm, Density in g/cm^3")
  mt=10.
  a=xmax(Mt, mt)
  xt = np.geomspace(1.1e0*a, 1.1e20*a, 25)
  rt = rg(Mt)*xt
  rht = rhoF3d(xt, Mt, mt)
  plt.loglog(rt, rht,'b-')
  data = np.array([rt, rht])
  data = data.T
  int_file = abs_path + 'SolarProfile10_m1e3.txt'
  np.savetxt(int_file, data, header="radius in cm, Density in g/cm^3")
  plt.xlim(1.e3, 1.e20)
  plt.ylim(1.e-22, 1.e-12)
  plt.show()

Mt = np.geomspace(1.e-14, 1.e4, 10)
print(Mt)
GammaT = np.log10( Gamma3d(Mt, 0.1) )
print(GammaT)
data = np.array([Mt, GammaT])
data = data.T
int_file = abs_path + 'DecayRates_m1e1.txt'
np.savetxt(int_file, data, header="PBH Mass in MSun, Decay 10GeV")
GammaT = np.log10( Gamma3d(Mt, 1.) )
print(GammaT)
data = np.array([Mt, GammaT])
data = data.T
int_file = abs_path + 'DecayRates_m1e2.txt'
np.savetxt(int_file, data, header="PBH Mass in MSun, Decay 100GeV")
GammaT = np.log10( Gamma3d(Mt, 10.) )
print(GammaT)
data = np.array([Mt, GammaT])
data = data.T
int_file = abs_path + 'DecayRates_m1e3.txt'
np.savetxt(int_file, data, header="PBH Mass in MSun, Decay 1TeV")
