/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.2.2 (MPI with some OpenMP for multi-node system)
 * Date Created : 8/21/20
 * Date Last mod: 10/13/20
 * Author: Yoshihiro Sato
 * Description: the main function of squapi_mpi 
 * Usage: $ mpiexec -np (nprocs) squapi_mpi system.dat init.dat (Nmax) (theta) (options)
 * Notes: 
 *      - Functions used in this program are written in sqmodule.cpp sqmodule_xxx.cpp
 *      - Based on C++11
 *      - Develped using gcc 10.2.0 and Open MPI 4.0.4 on MacOS 10.14 
 *      - size of Cn has to be lower than 2,147,483,647 (limit of int)
 *      - Supports Dkmax = 0
 *      - Takes "--cont" option for continuation to larger Nmax value
 *      - Takes "-s" option for saving D 
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved
 **********************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <deque>
#include <unordered_map>
#include <string>
#include <mpi.h>
#include "sqmodule.h"
#include "sqmodule_omp.h"
#include "sqmodule_mpi.h"
#include "hwconfig.h"
#include "opt.h"


int main(int argc, char* argv[])
{
    // --- MPI variables:
    int myid, nprocs, root;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    // --- define root id:
    root = RANK_OF_ROOT; // see sysconf.h for definition

    double time0 = MPI_Wtime(); // for time measurement using MPI 
    
    // --- global S-QuAPI variables:
    std::vector<std::complex<double>> U;
    std::vector<std::vector<std::complex<double>>> s;
    std::vector<std::complex<double>> gm0, gm1;
    std::vector<std::vector<std::complex<double>>> gm2, gm3, gm4;
    std::vector<std::complex<double>> rhos0, D;
    int Nmax, Dkmax, M, N0 = -1;
    double Dt, theta;
    
    // load data from files and store them in the S-QuAPI parameters: 
    load_data(argv, U, s, gm0, gm1, gm2, gm3, gm4, rhos0, Nmax, Dkmax, M, Dt,theta);

    // (optional) load D.dat and rhos.dat to continue from last saved point:
    if (myid == root){
        opt_load_D(argc, argv, Nmax, Dkmax, theta, "D.dat", "rhos.dat",  N0, D);    
    }

    // root broadcasts N0 to all ranks: 
    MPI_Bcast(&N0, 1, MPI_INT, root, MPI_COMM_WORLD);

    // -------- load S-QuAPI parameters -------------------------
    if (myid == root){
        std::cout << "****************************************" << std::endl;
        std::cout << "*         squapi_mpi ver 0.0           *" << std::endl;
        std::cout << "****************************************" << std::endl;
        std::cout << "----- Date and Time --------------------" << std::endl;
        std::system("date");
        std::cout << "----- parameters -----------------------" << std::endl;
        std::cout << "Dt     = " << Dt << " ps" << std::endl;
        std::cout << "Nmax   = " << Nmax   << std::endl;
        std::cout << "Dkmax  = " << Dkmax  << std::endl;
        std::cout << "M      = " << M      << std::endl;
        std::cout << "theta  = " << theta  << std::endl;
        std::cout << "nprocs = " << nprocs << std::endl;
        std::cout << "----- generate Cn and Wn ---------------" << std::endl;
    }

    /******************* STEP 1 **************************/
    // -------- Generate the paths and weights ----------------------
    std::vector<std::vector<unsigned long long>>   C;
    std::vector<std::vector<std::complex<double>>> W;
    getCW_mpi(myid, nprocs, root, theta, U, s, gm0, gm1, gm2, gm3, gm4, C, W);

    // root broadcasts CDkmax:
    int size = C[Dkmax].size();
    MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
    std::vector<unsigned long long> CDkmax(size);
    if (myid == root) CDkmax = C[Dkmax];
    MPI_Bcast(CDkmax.data(), size, MPI_UNSIGNED_LONG_LONG, root, MPI_COMM_WORLD);

    // generate Cnmap:
    double time1 = MPI_Wtime(); // for time measurement using MPI 
    if (myid == root){
        std::cout << "----- generate Cnmap -------------------" << std::endl;
    }
    std::unordered_map<unsigned long long, int> Cnmap;
    getCnmap(CDkmax, Cnmap); // serial version
    if (myid == root){
        std::cout << "  lap time = " << MPI_Wtime() - time1 << " sec" << std::endl;
    }
    
    // -------- Generate rhos ---------------------------------------
    if (myid == root){
        std::cout << "----- generate rhos --------------------" << std::endl;
    }
    std::vector<std::complex<double>> rhos(M * M);
    // ***** Start time evolution **************
    for (int N = N0 + 1; N < Nmax + 1; N++){
        time1 = MPI_Wtime(); // reset time1 for time measurement using MPI 
        int n;
        if (N == 0){
            if (myid == root) rhos = rhos0;
        }
        else if (N > 0 && Dkmax == 0){
            getrhosK(M, U, rhos);
        }
        else if (N > 0 && N < Dkmax + 1 && Dkmax > 0){
            /******************* STEP 2 **************************/
            n = N;
            getD0_mpi(N, myid, nprocs, root, C[n], rhos0, U, s, gm0, gm1, gm2, gm3, gm4, D);
            if (myid == root){
                getrhos_omp(N, U, C[n], W[n], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
            }
            // below full MPI version (not so fast):
            //getrhos_mpi(N, myid, nprocs, root, U, C[n], W[n], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
        }
        else if (N == Dkmax + 1 && Dkmax > 0){
            /******************* STEP 3 **************************/
            n = Dkmax + 1;
            getD0_mpi(N, myid, nprocs, root, C[n-1], rhos0, U, s, gm0, gm1, gm2, gm3, gm4, D);
            if (myid == root){
              getrhos_omp(N, U, C[n-1], W[n-1], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
            }
            // below full MPI version (not so fast):
            //getrhos_mpi(N, myid, nprocs, root, U, C[n-1], W[n-1], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
        }
        else{
            /******************* STEP 4 **************************/
            n = Dkmax + 1;
            getD1_mpi(N, myid, nprocs, root, C[n-1], Cnmap, U, s, gm0, gm1, gm2, gm3, gm4, D);
            if (myid == root){
              getrhos_omp(N, U, C[n-1], W[n-1], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
            }
            // below full MPI version (not so fast):
            //getrhos_mpi(N, myid, nprocs, root, U, C[n-1], W[n-1], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
        }
        // *** write N and rhos into rhos.dat ***
        if (myid == root){
            save_rhos(N, rhos, "rhos.dat");
            std::cout << "N = " << N << " of " << Nmax;
            std::cout << " lap time = " << MPI_Wtime() - time1 << " sec"; 
            std::cout << " tr = " << trace(rhos) << std::endl;
        }
    }

    if (myid == root){
        // (optional) save D to D.dat for cont 
        opt_save_D(argc, argv, Nmax, theta, D, "D.dat");

        std::cout << "----- end -----------------------------" << std::endl;
        std::cout << "    elapsed_time = " << MPI_Wtime() - time0 << " sec" << std::endl;
        std::cout << "---------------------------------------" << std::endl;
    }
    MPI_Finalize();
}

//=======================  EOF  ================================================
