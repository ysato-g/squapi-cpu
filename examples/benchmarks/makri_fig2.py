# -*- coding: utf-8 -*-
"""
Project: entropy 
Date Created : 9/7/20
Author: Yoshihiro Sato
Description: Reproduction of Figrue 2 of N. Makri and D. E. Makarov,
             Journal of Chemical Physics 102 4600-4610 (1995),
             page 4608.
Notes:
    - Run this by $ python3 makri_fig2.py
    - Will take about 10 seconds to complete with squapi_omp on
      a two-core system.
"""

import numpy as np
import subprocess
import sys
import matplotlib.pyplot as plt

bin_dir = "../../bin/"
sys.path.append(bin_dir)
import squapi as sq

#--- Constant (also defined in quapi):
hbar = sq.hbar    # The Planck cons in units of [cm^-1 * ps]
kB   = sq.kB      # Boltzmann const in units of [cm^-1 / K]

#==============================================================================
#                               Model
#==============================================================================
# Parameters from Makri & Makarov (1995):
#********************************************************************
Omega = 1                    # ARBITRARY FREQUENCY (any vale) [1/ps]
eps   = 0                    # $\epsilon$ [cm^-1 / hbar]
Dt    = 0.25 / Omega         # $\Delta t$, the time slice [1/ps]
wc    = 7.5 * Omega          # cut-off frequency [1/ps] 
beta  = 5.0 / (hbar * Omega) # inverse temp
xi    = 0.1                  # Kondo parameter
tmax  = 25 / Omega           # the maximum time in units of 1/Omega
Dkmax_list = [1, 5, 7]       # Dkmax values used 
#********************************************************************

# Pauli matrices:
sigma_x = np.array([[0, 1],    [1, 0]])
sigma_y = np.array([[0, -1.j], [1.j, 0]])
sigma_z = np.array([[1, 0],    [0, -1]])

# The system Hamiltonian (Makri & Makarov Eq.(29)):
H0 = hbar * Omega * sigma_x + hbar * eps * sigma_z 
# Bath temperature:
T = 1 / (beta * kB) 
# Set the eigenvalues of $s = \sigma_z$:
s1, s2 = sigma_z.diagonal() 
# the s array for single bath:
s   = [[s1, s2]]
# Reorganization energy for kappa = xi/2:
lam = [(xi / 2) * hbar * wc]
# Counter term:
mu  = [0]

def g(bath, t):
    return sq.gexp(t, lam[bath], wc, T)

#==============================================================================
#                            Computation
#==============================================================================
# squapi paramters
theta = 0              # threshold 0 to recover full iQuAPI result 
Nmax  = int(tmax / Dt) # The maximum number of the time steps

#============ Selection ==========
# select binary program, chose between 'squapi' and 'squapi_omp':
prog  = 'squapi_omp'
# set up initial density matrix:
rhos0 = [[1, 0], [0, 0]] 
#=================================

# Generate init.dat:
sq.save_init(rhos0)

rhos_list = []
# The maximum number of memory steps:
for Dkmax in Dkmax_list:
    # Generate system.dat for Dkamx value:
    sq.save_system(H0, s, lam, mu, T, Dt, Dkmax, g)
    # Run the squapi executable: 
    cmd = bin_dir + prog + ' system.dat init.dat ' + str(Nmax) + ' ' + str(theta)
    subprocess.call(cmd, shell=True)
    # Extract rhos from rhos.dat:
    rhos = sq.load_rhos('rhos.dat')
    rhos_list.append(rhos)

#==============================================================================
#                       Graphics
#==============================================================================
legends = ['b--', 'r-', 'k.']

# Make a graph:
x = np.array([k * Dt for k in range(Nmax + 1)]) 
for i in range(len(rhos_list)):
    rhos    = rhos_list[i]
    legend = legends[i]
    Dkmax  = Dkmax_list[i]
    y_z = (rhos[:, 0, 0] - rhos[:, 1, 1]).real
    y_x = (1/2) * (rhos[:, 0, 1] + rhos[:, 1, 0]).real
    plt.plot(x, y_z, legend, label = 'Dkmax=' + str(Dkmax), linewidth=2.0)
    plt.plot(x, y_x, legend, linewidth=1.0)
plt.ylim(-1, 1)
plt.xlim(0, tmax/Omega)
plt.xlabel(r'$\Omega t$')
plt.ylabel(r'$<\sigma_z>,\quad \rho_x$')
plt.grid(b = True, which = 'both', color = '0.65',linestyle = '--')
legend = plt.legend(loc='upper right', shadow=False, fontsize='x-small')
plt.show()

