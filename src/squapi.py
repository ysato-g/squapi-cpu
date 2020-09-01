# -*- coding: utf-8 -*-
"""
squapi module Version 0.25
Last Updated: 8/29/2020
Author: Yoshihiro Sato
Description: This module requires
    numpy 1.11.1 or above
    scipy 1.0.0 or above
Notes:
    - Convention as adopted in Sato J.Chem.Phys.150(2019)224108.
    - Added time derivatives of g-functions for Modified Redfield Theory
      for gexp, g777, and gDL
    - Added the lognormal sepctral density
*** Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved ***
"""

#==============================================================================
#                           Definitions
#==============================================================================
import numpy as np
import scipy.linalg as LA
from scipy.special import shichi, gamma, polygamma, loggamma, factorial
from scipy.integrate import quad

'''
physical constant from nist.gov as of 1/11/2020
    c  = 299 792 458             [meter/sec]: EXACT
    h  = 6.626 070 040(81) e-34  [Joule*sec]
    kB = 1.380 648 52(79) e-23   [Joule/K]
    e  = 1.602 176 6208(98) e-19 [Coulomb]
    me = 9.109 383 7015(28)   e-31 [kg] (electron mass)
    mp = 1.672 621 923 69(51) e-27 [kg] (proton mass)
    mn = 1.674 927 498 04(95) e-27 [kg] (neutron mass)
unit conversion: 
    [cm^-1] = h * c / 1cm = 1.986 445 824 e-23 [Joule]
    [Joule] = 5.034 116 651 e22 [cm^-1]
'''

hbar  = 5.3088374598   # Planck const in [cm^-1 * ps]
kB    = 0.6950345705   # Boltzmann const in [cm^-1 / K]
Joule = 5.034116651e22 # Joule in [cm^-1]
eV    = 8065.5440062   # electron-volt in [cm^-1]
meV   = eV / 1e3       # milli-electron-volt in [cm^-1]
fs    = 1 / 1000.      # 1 fs in [ps]

## fundamental physical constant from NIST
Meter  = 1e10 # meter in [angstrom]
Second = 1e12 # second in [ps] 
Kg     = Joule / (Meter/Second)**2 # Kilo-gram in [cm^-1 (ps/angstrom)^2 ]
FineStructConst = 7.2973525693e-3 # dimensionless, ~1/137
SpeedOfLight = 299792458 * Meter / Second  # speed of light in [angstrom/ps]
Mole   = 6.02214076e23 # mole

## other physical constnt from NIST 
me = 9.1093837015e-31 * Kg # electron mass in [cm^-1 (ps/angstrom)^2]

def systeminfo(H):
    '''
    Generates arrays of eigenvalues and eigenvectors of H.
    Input:  H in site basis
    Output: energy, eket, ebra, U
    Let m the site index then
    - energy: E_0 < E_1 < E_2 < ...
    - eket[a, m] = <m|E_a>
    - ebra[m, a] = <E_a|m>
    - U[m, a] = eket.T = <m|E_a>
    Note: U is matrix-operation ready,
          rhoex = U.conjugate().T @ (rhos @ U)
    '''
    #val, vec = np.linalg.eig(H)
    val, vec = LA.eig(H)                     # scipy.linalg is faster than np.linalg
    order = val.real.argsort()               # Ascending order of energy
    energy, eket = val[order], vec.T[order]  # eket[a, site] = <site|E_a>
    ebra = eket.conjugate().T                # ebra[site, a] = <E_a|site> 
    U = eket.T                               # U[site, a]    = <site|E_a> (matrix)
    return energy, eket, ebra, U
    
def PolyGamma(n, z, rmax=10000):
    '''
    complex poly gamma function based on
    $$
        \psi^{(n)}(z) = (-1)^{n+1} n! \sum_{r=0}^\infty \frac{1}{(z+r)^{n+1}}
    $$
    with which $\psi^{(0)}(z)$ is the digamma function $\psi(z)$.
    Default rmax is 1000. Increase it for better precision
    Notes: 
    - scipy.special.polygamma does not take complex argument
    - 'import numpy as np' is assumed
    - Use sipy.special.polygamma(n, x) for x real.
    '''
#    series = 0.
#    for r in range(rmax):
#        series += 1 / (z + r) ** (n + 1)
    series = np.array([1 / (z + r)**(n + 1) for r in range(rmax)]).sum()
    return (-1) ** (n + 1) * factorial(n) * series

def HurwitzLerchPhi(z, s, a, rmax=10000):
    '''
    The Hurwitz-Lerch function defined by
    $$
        \Phi(z, s, a) = \sum_{r = 0}^\infty \frac{z^r}{r + a}^s
    $$
    for $|z| < 1$ and $a \not= 0, -1, -2, ...$.
    Default rmax is 10000. Increase it for better precision.
    Notes: 
    - scipy.special does not have this function in it
    - 'import numpy as np' is assumed
    '''
    return np.array([z**r / (r + a)**s for r in range(rmax)]).sum()


#========== Spectral densities and their associated Line Broadening Functions ===================

#---------- Drude Lorentz ------------
def JDL0(omega, kappa, Omega):
    return kappa * hbar * omega * (Omega**2 / (omega**2 + Omega**2))

def gDL0(t, kappa, Omega, T):
    eul = 0.57721566490153286061 # The Euler-gamma constant
    pi  = np.pi
    phi = (kappa * pi / 2) * (1 - np.exp(-Omega * t)) # the imaginary part
    if t <= 0:
        return 0
    elif T > 0:
        a     = hbar * Omega / (2 * pi * kB * T)
        z     = np.exp(-Omega * t / a)
        term0 = (Omega * t - (1 - np.exp(-Omega * t))) / a
        term1 = HurwitzLerchPhi(z, 1,  a)
        term2 = HurwitzLerchPhi(z, 1, -a)
        term3 = 2 * np.log(1 - z)
        term4 = 2 * eul
        term5 = (2 - (z**a)) * polygamma(0, 1 + a)
        term6 = (z**a)       * polygamma(0, 1 - a)
        return (kappa / 2) * (term0 + term1 + term2 + term3 + term4 + term5 + term6) + 1.j * phi
    elif T == 0:
        # Note: Shi(z) = shici(z)[0] and Chi(z) = shici(z)[1]
        term0 = np.sinh(Omega * t) * shichi(Omega * t)[0] - np.cosh(Omega * t) * shichi(Omega * t)[1]
        term1 = eul + np.log(Omega * t)
        return kappa * (term0 + term1) + 1.j * phi
    else:
        return 0

def dgDL0(t, kappa, Omega, T):
    pi  = np.pi
    dphi = (kappa * Omega / 2) * pi * np.exp(-Omega * t) # the imaginary part
    if T > 0:
        a     = hbar * Omega / (2 * pi * kB * T)
        z     = np.exp(-Omega * t / a)
        term0 = (1 + np.exp(-Omega * t)) / a
        term1 =   HurwitzLerchPhi(z, 1, -a) - HurwitzLerchPhi(z, 1,  a)
        term2 = z**a * (polygamma(0, 1 - a) - polygamma(0, 1 + 1))
        return (- kappa * Omega / 2) * (term0 + term1 + term2) + 1.j * dphi
    elif T == 0:
        # Note: Shi(z) = shici(z)[0] and Chi(z) = shici(z)[1]
        term0 = np.cosh(Omega * t) * shichi(Omega * t)[0] - np.sinh(Omega * t) * shichi(Omega * t)[1]
        return kappa * Omega * term0 + 1.j * dphi
    else:
        return 0

def ddgDL0(t, kappa, Omega, T):
    pi  = np.pi
    ddphi = - (kappa * Omega**2 / 2) * pi * np.exp(-Omega * t) # the imaginary part
    if T > 0:
        a     = hbar * Omega / (2 * pi * kB * T)
        z     = np.exp(-Omega * t / a)
        term0 = np.exp(-Omega * t) / a
        term1 =   HurwitzLerchPhi(z, 1, -a) + HurwitzLerchPhi(z, 1, a)
        term2 = z**a * (polygamma(0, 1 - a) - polygamma(0, 1 + a))
        return (kappa * Omega**2 / 2) * (term0 + term1 + term2) + 1.j * ddphi
    elif T == 0:
        # Note: Shi(z) = shici(z)[0] and Chi(z) = shici(z)[1]
        term0 = np.sinh(Omega * t) * shichi(Omega * t)[0] - np.cosh(Omega * t) * shichi(Omega * t)[1]
        return kappa * Omega**2 * term0 + 1.j * ddphi
    else:
        return 0

def JDL(omega, lam, Omega):
    '''
    Note: use lam in units of energy [cm^-1]
    '''
    kappa = 2 * (lam / hbar) / (np.pi * Omega)
    return JDL0(omega, kappa, Omega)

def lamDL(kappa, Omega):
    '''
    Generates reorganization energy in [cm^-1]
    '''
    return kappa * np.pi * hbar * Omega / 2

def gDL(t, lam, Omega, T):
    '''
    Note use lam in units of energy [cm^-1]
    '''
    kappa = 2 * (lam / hbar) / (np.pi * Omega)
    return gDL0(t, kappa, Omega, T)        

def dgDL(t, lam, Omega, T):
    '''
    Note use lam in units of energy [cm^-1]
    '''
    kappa = 2 * (lam / hbar) / (np.pi * Omega)
    return dgDL0(t, kappa, Omega, T)        

def ddgDL(t, lam, Omega, T):
    '''
    Note use lam in units of energy [cm^-1]
    '''
    kappa = 2 * (lam / hbar) / (np.pi * Omega)
    return ddgDL0(t, kappa, Omega, T)        

# --- below for backward compativility
def JLD0(omega, kappa, Omega): return JDL0(omega, kappa, Omega)
def gLD0(t, kappa, Omega, T): return gDL0(t, kappa, Omega, T)
def JLD(omega, lam, Omega): return JDL(omega, lam, Omega)
def lamLD(kappa, Omega): return lamDL(kappa, Omega)
def gLD(t, lam, Omega, T): return gDL(t, lam, Omega, T)

#---------- Exponential ------------
def gexp0(t, kappa, Omega, T, p = 0):
    theta = kB * T / (hbar * Omega)
    if t <= 0:
        return 0
    elif p == 0 and T > 0:
        term1 = - np.log(1 - 1j * Omega * t)
        term2 = 2 * loggamma(theta)
        term3 = - loggamma(theta * (1 + 1j * Omega * t))
        term4 = - loggamma(theta * (1 - 1j * Omega * t))
        return  kappa * term1 + kappa * (term2 + term3 + term4)
    elif p == 0 and T == 0:
        term1 = - np.log(1 - 1j * Omega * t)
        return kappa * term1
    elif p != 0 and T > 0:
        term1 = gamma(p) * ((1 - 1j * Omega * t) ** (-p) - 1)
        term2 = 2 * PolyGamma(p - 1, theta)
        term3 = - PolyGamma(p - 1, theta * (1 + 1j * Omega * t))
        term4 = - PolyGamma(p - 1, theta * (1 - 1j * Omega * t))
        return kappa * term1 + kappa * ((-theta)**p) * (term2 + term3 + term4)
    elif p != 0 and T == 0:
        term1 = gamma(p) * ((1 - 1j * Omega * t) ** (-p) - 1)
        return kappa * term1
    else:
        return 0

def dgexp0(t, kappa, Omega, T, p = 0):
    theta = kB * T / (hbar * Omega)
    if p >= 0 and T > 0:
        term1 = gamma(p + 1) * (1j * Omega) *(1 - 1j * Omega * t) ** (-(p + 1))
        term2 =   PolyGamma(p, theta * (1 + 1j * Omega * t))
        term3 = - PolyGamma(p, theta * (1 - 1j * Omega * t))
        return kappa * term1 + kappa * ((-theta)**(p + 1)) * (1j * Omega) * (term2 + term3)
    elif p >= 0 and T == 0:
        term1 = gamma(p + 1) * (1j * Omega) *(1 - 1j * Omega * t) ** (-(p + 1))
        return kappa * term1
    else:
        return 0

def ddgexp0(t, kappa, Omega, T, p = 0):
    theta = kB * T / (hbar * Omega)
    if p >= 0 and T > 0:
        term1 = gamma(p + 2) * ((1j * Omega)**2) * (1 - 1j * Omega * t) ** (-(p + 2))
        term2 = PolyGamma(p + 1, theta * (1 + 1j * Omega * t))
        term3 = PolyGamma(p + 1, theta * (1 - 1j * Omega * t))
        return kappa * term1 + kappa * ((-theta)**(p + 2)) * (Omega**2) * (term2 + term3)
    elif p >= 0 and T == 0:
        term1 = gamma(p + 2) * ((1j * Omega)**2) * (1 - 1j * Omega * t) ** (-(p + 2))
        return kappa * term1
    else:
        return 0

def Jexp0(omega, kappa, Omega, p = 0):
    if omega > 0:
        return kappa * hbar * omega * (omega / Omega)**p * np.exp(-omega / Omega)
    else:
        return 0        

def lamexp(kappa, Omega, p = 0):
    return kappa * Omega * np.gamma(1 + p)

def gexp(t, lam, Omega, T, p = 0):
    '''
        takes reorganization energy lam instead of kappa
        lam in units of [cm^-1]
    '''
    kappa = (lam / hbar) / (Omega * gamma(1 + p))
    return gexp0(t, kappa, Omega, T, p)

def dgexp(t, lam, Omega, T, p = 0):
    '''
        takes reorganization energy lam instead of kappa
        lam in units of [cm^-1]
    '''
    kappa = (lam / hbar) / (Omega * gamma(1 + p))
    return dgexp0(t, kappa, Omega, T, p)

def ddgexp(t, lam, Omega, T, p = 0):
    '''
        takes reorganization energy lam instead of kappa
        lam in units of [cm^-1]
    '''
    kappa = (lam / hbar) / (Omega * gamma(1 + p))
    return ddgexp0(t, kappa, Omega, T, p)

def Jexp(omega, lam, Omega, p = 0):
    '''
        takes reorganization energy lam instead of kappa
        lam in units of [cm^-1]
    '''
    kappa = lam / hbar / (Omega * gamma(1 + p))
    return Jexp0(omega, kappa, Omega, p)    


#----------- Approximated B777 -----------------
def g777(t, lam, T):
    '''
        lam in units of [cm^-1]
    '''
    if t <= 0:
        return 0
    else:
        term1 = gexp0(t, 0.22, 170 / hbar, T, 0)
        term2 = gexp0(t, 0.78,  34 / hbar, T, 1)
        term3 = gexp0(t, 0.31,  69 / hbar, T, 2)
        return (lam / 106.7) * (term1 + term2 + term3)

def dg777(t, lam, T):
    '''
        lam in units of [cm^-1]
    '''
    term1 = dgexp0(t, 0.22, 170 / hbar, T, 0)
    term2 = dgexp0(t, 0.78,  34 / hbar, T, 1)
    term3 = dgexp0(t, 0.31,  69 / hbar, T, 2)
    return (lam / 106.7) * (term1 + term2 + term3)

def ddg777(t, lam, T):
    '''
        lam in units of [cm^-1]
    '''
    term1 = ddgexp0(t, 0.22, 170 / hbar, T, 0)
    term2 = ddgexp0(t, 0.78,  34 / hbar, T, 1)
    term3 = ddgexp0(t, 0.31,  69 / hbar, T, 2)
    return (lam / 106.7) * (term1 + term2 + term3)

def J777(omega, lam):
    '''
        lam in units of [cm^-1]
    '''
    term1 = 0.22 * np.exp(- omega / (170 / hbar))
    term2 = 0.78 * omega /(34 / hbar) * np.exp(- omega / (34 / hbar))
    term3 = 0.31 * (omega / (69 / hbar))**2 * np.exp(- omega / (69 / hbar))
    return hbar * omega * (lam / 106.7) * (term1 + term2 + term3)


#------------ Direct Quadrature -----------------------
def fcorr_real(omega, T, t):
    return (1 / np.tanh(hbar * omega / (2 * kB * T))) * (1. - np.cos(omega * t))

def dfcorr_real(omega, T, t):
    return (1 / np.tanh(hbar * omega / (2 * kB * T))) * np.sin(omega * t)

def ddfcorr_real(omega, T, t):
    return (1 / np.tanh(hbar * omega / (2 * kB * T))) * np.cos(omega * t)

def fcorr_imag(omega, t):
    return np.sin(omega * t)

def dfcorr_imag(omega, t):
    return np.cos(omega * t)

def ddfcorr_imag(omega, t):
    return - np.sin(omega * t)

def gnum(t, T, J, mid=1e5, epsrel=1e-5):
    '''
        Note: J(omega) has the only argument omega and no other.
        mid    = 1e5  ## break up [0, np.inf] into [0, mid] + [mid, np.inf]
        epsrel = 1e-5 ## epsrel value
    '''
    if t <= 0:
        return 0.
    elif T > 0:
        re = lambda omega, T, t : (1 / hbar) * (J(omega) / omega**2) * fcorr_real(omega, T, t)
        im = lambda omega, t :    (1 / hbar) * (J(omega) / omega**2) * fcorr_imag(omega, t)
        #val_re  = quad(re, 0, np.inf, args=(T, t))[0]
        val_re  = quad(re, 0, mid, epsrel=epsrel,  args=(T, t))[0]
        val_re += quad(re, mid, np.inf, epsrel=epsrel, args=(T, t))[0]
        #val_im  = quad(im, 0, np.inf, args=t)[0]
        val_im  = quad(im, 0, mid, epsrel=epsrel,  args=t)[0]
        val_im += quad(im, mid, np.inf, epsrel=epsrel,  args=t)[0]
        return val_re + 1.j * val_im

def dgnum(t, T, J):
    '''
        Note: J(omega) has the only argument omega and no other.
    '''
    re = lambda omega, T, t : (1 / hbar) * (J(omega) / omega) * dfcorr_real(omega, T, t)
    im = lambda omega, t :    (1 / hbar) * (J(omega) / omega) * dfcorr_imag(omega, t)
    val_re = quad(re, 0, np.inf, args=(T, t))[0]
    val_im = quad(im, 0, np.inf, args=t)[0]
    return val_re + 1.j * val_im

def ddgnum(t, T, J):
    '''
        Note: J(omega) has the only argument omega and no other.
    '''
    re = lambda omega, T, t : (1 / hbar) * J(omega) * ddfcorr_real(omega, T, t)
    im = lambda omega, t :    (1 / hbar) * J(omega) * ddfcorr_imag(omega, t)
    val_re = quad(re, 0, np.inf, args=(T, t))[0]
    val_im = quad(im, 0, np.inf, args=t)[0]
    return val_re + 1.j * val_im

def lamnum(J):
    '''
        Note: J(omega) has the only argument omega and no other.
    '''
    f = lambda omega : J(omega) / omega;
    return quad(f, 0, np.inf)[0]


#------ Adolphs and Renger BiophysJ91(2006)2778 and JChemPhys116(2002)9997 -------
def JAR(omega):
    '''
        lam in units of [cm^-1]
    '''
    s1 = 0.8;
    s2 = 0.5;
    w1 = 0.069 * meV / hbar;
    w2 = 0.24  * meV / hbar;
    a1 = s1/((s1 + s2) * gamma(7 + 1) * 2);
    a2 = s2/((s1 + s2) * gamma(7 + 1) * 2);
    term1 = a1 * (omega / w1)**4 * np.exp(-(omega / w1)**0.5);
    term2 = a2 * (omega / w2)**4 * np.exp(-(omega / w2)**0.5);
    return hbar * omega * (term1 + term2)

def gAR(t, lam, T):
    #lam0 = 102; # see JChemPhys116(2002)9997
    lam0 = 78.2630756244 # from lamnum(JAR)
    return (lam / lam0) * gnum(t, T, JAR)

def dgAR(t, lam, T):
    #lam0 = 102; # see JChemPhys116(2002)9997
    lam0 = 78.2630756244 # from lamnum(JAR)
    return (lam / lam0) * dgnum(t, T, JAR)

def ddgAR(t, lam, T):
    #lam0 = 102; # see JChemPhys116(2002)9997
    lam0 = 78.2630756244 # from lamnum(JAR)
    return (lam / lam0) * ddgnum(t, T, JAR)

#----- Lognormal from J.Phys.Chem.B117(2013)7317-7323 ----------------------------
def JLN(omega, Omega, sigma, lam):
    '''
        lam = hbar * lambda, in units of [cm^-1]
        Omega in units of frequency [1/ps]
        sigma dimensionless
    '''
    f = (2 * np.pi * sigma**2)**0.5 * np.exp(sigma**2 / 2);
    K = lam / (Omega * f); # K = kappa * hbar
    return K * omega * np.exp(- np.log(omega / Omega) ** 2 / (2 * sigma**2))

def gLN(t, Omega, sigma, lam, T):
    def J(omega):
        return JLN(omega, Omega, sigma, lam)
    return gnum(t, T, J)

def dgLN(t, Omega, sigma, lam, T):
    def J(omega):
        return JLN(omega, Omega, sigma, lam)
    return dgnum(t, T, J)

def ddgLN(t, Omega, sigma, lam, T):
    def J(omega):
        return JLN(omega, Omega, sigma, lam)
    return ddgnum(t, T, J)

#----- FMO based on Lognormal --------------------------------------------------
def JFMO(omega, lam):
    '''
        lam in units of [cm^-1]
    '''
    Omega = 38 / hbar;
    sigma = 0.7;
    return JLN(omega, Omega, sigma, lam)

def gFMO(t, lam, T):
    Omega = 38 / hbar;
    sigma = 0.7;
    return gLN(t, Omega, sigma, lam, T)

def dgFMO(t, lam, T):
    Omega = 38 / hbar;
    sigma = 0.7;
    return dgLN(t, Omega, sigma, lam, T)

def ddgFMO(t, lam, T):
    Omega = 38 / hbar;
    sigma = 0.7;
    return ddgLN(t, Omega, sigma, lam, T)

#--- Intramolecular vibrational mode
def JH1(omega, omegaH, SH, gammaH):
    '''
    Lorenzian-delta type satisfying
    $\frac{J(\omega)}{\hbar\omega^2} = S_H \delta(\omega - \omega_H)$
    for $\gamma_h \rightarrow 0$.
    Correspondence:
    $\omega$   = omega     
    $\omega_H$ = omegaH
    $S_H$      = SH
    $\gamma_H$ = gammaH
    SH is the Huang-Rhys factor
    '''
    term0 = hbar * (omega**2) * SH
    term1 = (1 / np.pi) * gammaH
    term2 = ((omega - omegaH)**2 + gammaH**2)**(-1)
    return term0 * term1 * term2

#--- Intramolecular vibrational mode ver. 2
def JH2(omega, lamH, OmegaH, gammaH):
    '''
    Vibrational spectral density used in New Journal ofPhysics 15 (2013) 075013.
    Arguments:
    $\omega$       = omega  [1/ps]  
    $\hbar\lambda$ = lamH   [cm^-1]
    $\Omega_H$     = OmegaH [1/ps]
    $\gamma_H$     = gammaH [1/ps]
    '''
    term0 = 2 * 2**0.5 * lamH * omega * OmegaH**2 * gammaH
    term1 = (omega**2 - OmegaH**2)**2 + 2 * gammaH**2 * omega**2
    val = (1/np.pi) * term0 / term1
    return val

# --- below for backward compativility
def JH(omega, omegaH, SH, gammaH): return JH1(omega, omegaH, SH, gammaH)

#--- Gaussian spectral density
def Jgauss(omega, lam, tau):
    '''
    Ohmic with gaussian cutoff. 
    See Yang & Fleming Chem.Phys.275(2002)355-372. 
    $J(\omega) = \frac{\hbar\lambda \omega \tau}{\sqrt{\pi}}\exp{-(\omega\tau/2)^2}$
    Arguments:
    $\omega$       = omega [1/ps]  
    $\hbar\lambda$ = lam   [cm^-1]
    $\tau$         = tau   [ps]
    '''
    A = lam * tau / (np.pi)**0.5
    return A * omega * np.exp(-(omega * tau / 2)**2) 


#============= Reduced Influence Coefficients (RICs) ===========================
def getGamma(lam, mu, T, Dt, Dkmax, g, scaled):
    '''
    Generates the reduced influence coeffcienets
    - Set "scaled=True" if g(bath, t) = (lam[bath] / lam[0]) * g(0, t) holds 
    - Set "scaled=Fasle" if not
    '''
    nbath = len(lam)
    gm0 = np.empty(nbath, dtype=complex)
    gm1 = np.empty(nbath, dtype=complex)
    gm2 = np.empty((nbath, Dkmax), dtype=complex)
    gm3 = np.empty((nbath, Dkmax), dtype=complex)
    gm4 = np.empty((nbath, Dkmax), dtype=complex)
    #--- generate g0 and g1
    for bath in range(nbath):
        gm0[bath] = g(bath, Dt/2) - 1j * (lam[bath] - mu[bath]) * (Dt/2) / hbar
        gm1[bath] = g(bath, Dt)   - 1j * (lam[bath] - mu[bath]) * Dt / hbar
    #--- generate g2, g3, and g4
    if scaled:
        bath = 0
        for Dk in range(1, Dkmax + 1):
            gm2[bath, Dk-1] = g(bath, (Dk - 1) * Dt) - g(bath, (Dk - 1/2) * Dt)\
                              - g(bath, Dk * Dt) + g(bath, (Dk + 1/2) * Dt)  
            gm3[bath, Dk-1] = g(bath, (Dk + 1) * Dt) - 2 * g(bath, Dk * Dt)\
                              + g(bath, (Dk - 1) * Dt)
            gm4[bath, Dk-1] = g(bath, Dk * Dt) - 2 * g(bath, (Dk - 1/2) * Dt)\
                              + g(bath, (Dk - 1) * Dt)
        for bath in range(1, nbath):
            mag = lam[bath] / lam[0] ## magnification factor ##
            gm2[bath] = mag * gm2[0]
            gm3[bath] = mag * gm3[0]
            gm4[bath] = mag * gm4[0]
    else:
        for bath in range(nbath):
            for Dk in range(1, Dkmax + 1):
                gm2[bath, Dk-1] = g(bath, (Dk - 1) * Dt) - g(bath, (Dk - 1/2) * Dt)\
                                  - g(bath, Dk * Dt) + g(bath, (Dk + 1/2) * Dt)  
                gm3[bath, Dk-1] = g(bath, (Dk + 1) * Dt) - 2 * g(bath, Dk * Dt)\
                                  + g(bath, (Dk - 1) * Dt)
                gm4[bath, Dk-1] = g(bath, Dk * Dt) - 2 * g(bath, (Dk - 1/2) * Dt)\
                                  + g(bath, (Dk - 1) * Dt)
    #--- results
    return [gm0, gm1, gm2, gm3, gm4]


#============= Additonal Functions ===========================
def save_arrs(arrs, filename):
    '''
    arrs: array of numpy arrays
    filename: a data text file
    Writes the shape of each arr and flatten it into a text file
    following the format of getdata(filename, arrs, sheapes)
    in sqmodule.cpp
    '''

    file = open(filename, 'w')
    file.write(str(len(arrs))) # the total number of arrays
    file.write('\n')
    for arr in arrs:
        file.write(str(arr.ndim))
        file.write('dim ')
        for n in range(arr.ndim):
            file.write(str(arr.shape[n]))
            file.write(',')
        file.write(str('\n'))
        for af in arr.flatten():
            file.write(str(af.real))
            file.write(' ')
            file.write(str(af.imag))
            file.write('\n')
    file.close()


def save_system(H, s, lam, mu, T, Dt, Dkmax, g, scaled=False):
    '''
    Generates system.dat
    '''
    # Force the input to be numpy arrays:
    H = np.array(H, dtype=complex)
    s = np.array(s, dtype=complex)
    lam = np.array(lam, dtype=complex)
    mu = np.array(mu, dtype=complex)
    Dtc = np.array([Dt], dtype=complex)
    # Compute energy and eket:
    energy, eket, ebra, U = systeminfo(H)
    # Compute the RICs:
    gm = getGamma(lam, mu, T, Dt, Dkmax, g, scaled)

    arrs = [Dtc, energy, eket, s, gm[0], gm[1], gm[2], gm[3], gm[4]]
    # Write the shape of arr and flatten arr into a text file, 'arrs.dat'
    save_arrs(arrs, 'system.dat')


def save_init(rhos0):
    '''
    Generates init.dat
    '''
    # Force the input to be numpy array:
    rhos0 = np.array(rhos0, dtype=complex)
    arrs = [rhos0]
    # write the shape of rhos0 and flatten rhos0 into a text file, 'init.dat'
    save_arrs(arrs, 'init.dat')


def load_rhos(filename='rhos.dat'):
    '''
    Load rhos from filename=rhos.dat
    '''
    data = np.loadtxt('rhos.dat', comments='#', delimiter=',', dtype='complex')
    # extract arrays and change dtype
    Narr    = data[:,0].real.astype(int)   # array of N values. Use this for selecting
    rhosarr = data[:,1]                    # array of rhos(N) values
    
    M2 = np.count_nonzero(Narr == 0)
    M  = int(M2 ** 0.5)
    Nmax = int(data[-1, 0].real) # extract Nmax
    rhos = np.zeros((Nmax + 1, M, M), dtype=complex) 
    
    for N in range(Narr.min(), Narr.max()+1):
        select = Narr == N
        rhos[N] = rhosarr[select].reshape((M, M))

    return rhos

#=======================  EOF  ================================================
