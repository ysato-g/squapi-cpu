# -*- coding: utf-8 -*-
'''
Project: S-QuAPI for CPU
Date Created : 8/30/20
Date Last mod: 9/2/20
Author: Yoshihiro Sato
Description: Exciton dynamics of a pigment-protein dimer using S-QuAPI.
             The default parameters are set to reproduces Fig.4
             of New Journal of Physics 13 (2011) 063040.
             The curves in graph are extraporated curves
             just faking the HEOM result.
Usage: $ python3 benchmark3.py
Notes:
    - select computational mode with mode = 'ser', 'omp', or 'mpi'
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved *   
'''

#==============================================================================
#                           Definitions
#==============================================================================
import numpy as np
import subprocess
import sys
import matplotlib.pyplot as plt

bin_dir = '../../bin/'
sys.path.append(bin_dir)
import squapi as sq 

#--- Constant (defined in squapi.py):
hbar    = sq.hbar    # The Planck cons in units of [cm^-1 * ps]
kB      = sq.kB      # Boltzmann const in units of [cm^-1 / K]
fs      = sq.fs      # 1 fs in units of [ps]

#==============================================================================
#                                Model
#==============================================================================
#--- system ---
H = [[100, 100], [100, 0]]  # Hamiltonian in the site basis 
#--- bath ---
# Bath temperature in K:
T = 300
# independent bath model
s   = [[0, 1],[1, 0]]                             # System-bath coupling matrix
lam = [2 / np.pi * 100 / 5, 2 / np.pi * 100 /5]   # Reorganization energies
mu  = lam                                         # Counter terms  
g = lambda bath, t: sq.gexp(t, lam[bath], 53/hbar, T)

#--- initial density matirx
rhosA = np.array([[1., 0.],[0., 0.]], dtype = complex)      # The donor site excited

#==============================================================================
#                               Computation
#==============================================================================
#--- QUAPI paramters
Dkmax   = 7             # The maximum number of memory steps
theta   = 0.e-6         # threshold for propagator          
t       = 992 * fs      # The total time          
Dt      = 32 * fs       # Time slice of path integral
Nmax    = round(t / Dt) # The maximum time step

#============ Selection ==========
# select mode from 'ser' (serial) 'omp' (OpenMP) and 'mpi' (MPI):
mode     = 'mpi'
rhos0    = rhosA.copy()
system   = True 
init     = True 
squapi   = True 
cont     = False
graphics = True  
#=================================

# ----- Generate system.dat --------------------
if system == True:
    sq.save_system(H, s, lam, mu, T, Dt, Dkmax, g)

# -----Generate init.dat --------------------
if init == True:
    sq.save_init(rhos0)

#------- Processing with external program
if squapi == True:
    # to make the list of avalable binaries, get output from shell:
    output = subprocess.check_output('ls ' + bin_dir, shell=True)
    # then decode it to UTF-8 and make a list out of it:
    prog_list = output.decode('UTF-8').splitlines()
    # set up binary program based on selection:
    prog = 'squapi'
    if cont == True: prog += '_cont'
    if mode != 'ser': prog += '_' + mode
    cmd = 'echo running with ' + prog
    subprocess.call(cmd, shell=True, executable='/bin/bash')
    if prog in prog_list:
        # start computing:
        cmd = bin_dir + prog + ' system.dat init.dat ' + str(Nmax) + ' ' + str(theta)
        if mode == 'mpi':
            cmd = 'mpiexec ' + cmd
        subprocess.call(cmd, shell=True, executable='/bin/bash')
    else:
        print('**CANNOT RUN: ' + prog + ' DOES NOT EXIST IN ' + bin_dir + ' **')
        exit(0)

#==============================================================================
#                                 Graphics
#==============================================================================
if graphics == True:
    # extract rhos from rhos.dat:
    rhos = sq.load_rhos('rhos.dat')
    # Make a graph:
    from scipy.interpolate import interp1d

    time = np.array([k * Dt / fs for k in range(Nmax + 1)])
    time2 = np.linspace(0, time.max(), 300)
    
    fig = plt.figure(figsize=(7.5, 2.5))
    ax = fig.add_subplot(111)
    plt.xlabel('time [fs]', fontsize='large')
    plt.ylim(-0.6, 1.1)
    plt.xlim(0, 1000)
    plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900], fontsize='large')
    plt.yticks([-0.5, 0, 0.5, 1], fontsize='large')
    y0 = rhos[:,0,0].real.round(12)
    y1 = rhos[:,0,1].real.round(12)
    y2 = rhos[:,1,0].imag.round(12)
    f0 = interp1d(time, y0, kind = 'cubic')
    f1 = interp1d(time, y1, kind = 'cubic')
    f2 = interp1d(time, y2, kind = 'cubic')
    plt.plot(time, y0, 'o', mfc='none', mec='b', label = r'$\rho_{\rm dd}$')
    plt.plot(time, y1, 's', mfc='none', mec='k', label = r'$\rho_{\rm da}$')
    plt.plot(time, y2, 'x', mfc='none', mec='m', label = r'$\rho_{\rm ad}$')
    plt.plot(time2, f0(time2), '-r')
    plt.plot(time2, f1(time2), '-r')
    plt.plot(time2, f2(time2), '-r')
    plt.legend(loc='lower right', shadow=False, fontsize='small')
    ax.annotate(r'${\rm T=300K}$', xy=(675, 0.9), fontsize='x-large')
    ax.annotate(r'$\lambda = (2/\pi)\,{\rm J_{da}}\,/\,5$', xy=(675, 0.7), fontsize='x-large')
    plt.show()


