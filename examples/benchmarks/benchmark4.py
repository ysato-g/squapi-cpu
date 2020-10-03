# -*- coding: utf-8 -*-
"""
Project: S-QuAPI for CPU
Date Created : 8/15/20
Date Last mod: 10/2/20
Author: Yoshihiro Sato
Description: Exciton dynamics of the FMO complex using S-QuAPI. 
             The default parameters are set to reproduce Fig.3 
             of Nalbach & Thorwart, Phys.Rev.E84(2011)041926.
Usage: $ python3 benchmark4.py
Notes:
    - select computational mode with mode = 'ser', 'omp', or 'mpi' 
    - To see a quick result, reduce Dkmax value from 4 to 2.
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved *   
"""

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
meV     = sq.meV     # 1 meV in units of [cm^-1]
eV      = sq.eV 
fs      = sq.fs      # 1 fs in units of [ps]

#==============================================================================
#                               Model
#==============================================================================
#--- system ---
H = [[  240, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
     [-87.7,   315,  30.8,   8.2,   0.7,  11.8,   4.3],
     [  5.5,  30.8,     0, -53.5,  -2.2,  -9.6,   6.0],
     [ -5.9,   8.2, -53.5,   130, -70.7, -17.0, -63.3],
     [  6.7,   0.7,  -2.2, -70.7,   285,  81.1,  -1.3],
     [-13.7,  11.8,  -9.6, -17.0,  81.1,   435,  39.7],
     [ -9.9,   4.3,   6.0, -63.3,  -1.3,  39.7,   245],
    ]

#--- bath ---
T   = 77                     # Bath temperature in [K] [77]
s   = np.identity(len(H))    # System-bath coupling matrix
lam = [35] * len(H)          # Reorganization energies in [cm^-1] [35]
mu  = lam                    # Counter terms in [cm^-1] [lam]
wc  = 1/(50 * fs)            # Bath cut-off frequency in [1/ps] [1 / (50 * fs)
g   = lambda bath, t: sq.gDL(t, lam[bath], wc, T) # Bath function defined

#--- initial density matirx
rhosA = np.zeros_like(H)
rhosA[0, 0] = 1              # The donor site excited

#==============================================================================
#                               Computation
#==============================================================================
#--- S-QuAPI paramters
Dkmax = 4               # The maximum number of memory steps [4]
Nmax  = 40              # The maximu times step [40]
Dt    = 850 * fs / 40   # Time slice of path integral [850 * fs / 40]
theta = 0               # The threshold for propagator cut [0]

#============ Selection ==========
# select mode from 'ser' (serial) 'omp' (OpenMP) and 'mpi' (MPI):
mode     = 'omp'
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
    if mode != 'ser': prog += '_' + mode
    cmd = 'echo running with ' + prog 
    subprocess.call(cmd, shell=True, executable='/bin/bash')
    if prog in prog_list:
        # start computing:
        cmd = bin_dir + prog + ' system.dat init.dat ' + str(Nmax) + ' ' + str(theta)
        if mode == 'mpi':
            #cmd = 'mpiexec -np ' + str(nprocs) + ' ' + cmd 
            cmd = 'mpiexec ' + cmd 
        if cont == True: cmd += ' --cont' 
        subprocess.call(cmd, shell=True)
    else:
        print('**CANNOT RUN: ' + prog + ' DOES NOT EXIST IN ' + bin_dir + ' **')
        exit(0)

#==============================================================================
#                                Graphics
#==============================================================================
if graphics == True:
    # extract rhos from rhos.dat:
    rhos = sq.load_rhos('rhos.dat')
    # Make a graph:
    times = np.array([k * Dt / fs for k in range(Nmax + 1)])
    opts = 'o--k', 's--r', '--g', '-.b', '-y', '-c', '-.m'
    mecs = ['k', 'r', 'g', 'b', 'y', 'c', 'm'] * 3
    sites = 'BChl 1', 'BChl 2', 'BChl 3', 'BChl 4', 'BChl 5', 'BChl 6', 'BChl 7' 
    
    plt.figure(figsize=(4.8, 3.0))
    plt.xlabel('time [fs]')
    plt.ylabel('Site Population')
    plt.ylim(0,1)
    plt.xlim(0,850)
    plt.xticks([0, 125, 250, 375, 500, 625, 750], ['0', ' ', '250', ' ', '500', ' ', '750'])
    plt.yticks([0, 0.25, 0.5, 0.75, 1], ['0', ' ', '0.5', ' ', '1'])
    #plt.grid(b = True, which = 'both', color = '0.65',linestyle = '--')
    for i in range(len(H)):
        plt.plot(times, rhos[:, i, i].real, opts[i], mec=mecs[i], label = sites[i], mfc='none')
    legend = plt.legend(loc='upper right', shadow=False, fontsize='x-small')
    plt.show()
