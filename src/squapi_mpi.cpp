/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.2.2 (MPI with some OpenMP for multi-node system)
 * Date Created : 8/21/20
 * Date Last mod: 9/9/20
 * Author: Yoshihiro Sato
 * Description: the main function of genrhos 
 * Usage: $ mpirun -np (nprocs) genrhos system.dat init.dat (Nmax) (theta)
 * Notes: 
 *      - Functions used in this program are written in sqmodule.cpp sqmodule_xxx.cpp
 *      - Based on C++11
 *      - Develped using gcc 10.2.0 and Open MPI 4.0.4 on MacOS 10.14 
 *      - size of Cn has to be lower than 2,147,483,647 (limit of int)
 *      - Supports Dkmax = 0
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
#include "sysconf.h"
#include "cont.h"



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
    std::vector<std::complex<double>> Dtc;
    std::vector<std::complex<double>> energy;
    std::vector<std::complex<double>> eket;
    std::vector<std::complex<double>> U;
    std::vector<std::vector<std::complex<double>>> s;
    std::vector<std::complex<double>> gm0;
    std::vector<std::complex<double>> gm1;
    std::vector<std::vector<std::complex<double>>> gm2;
    std::vector<std::vector<std::complex<double>>> gm3;
    std::vector<std::vector<std::complex<double>>> gm4;
    std::vector<std::complex<double>> rhos0;
    std::vector<std::complex<double>> D;
    int Nmax, Dkmax, M, N0; // N0 not used in genrhos.cpp
    double Dt, theta;

    // load data from files and store them in the S-QuAPI parameters: 
    load_data(argv, energy, eket, U, s, 
              gm0, gm1, gm2, gm3, gm4, rhos0, Nmax, Dkmax, M, Dt,theta);

    // --- generate U for the propagators
    getU(Dt, energy, eket, U);  

    // -------- load S-QuAPI parameters -------------------------
    if (myid == root){
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
    if (myid == root){
        std::cout << "----- generate Cnmap -------------------" << std::endl;
    }
    std::unordered_map<unsigned long long, int> Cnmap;
    getCnmap(CDkmax, Cnmap); // serial version
    
    // -------- Generate rhos ---------------------------------------
    if (myid == root){
        std::cout << "----- generate rhos --------------------" << std::endl;
    }
    std::vector<std::complex<double>> rhos(M * M);
    // ***** Start time evolution **************
    for (int N = 0; N < Nmax + 1; N++){
        double time1 = MPI_Wtime(); // for time measurement using MPI 
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
            save_rhos(N, rhos);
            std::cout << "N = " << N << " of " << Nmax;
            std::cout << " lap time = " << MPI_Wtime() - time1 << " sec"; 
            std::cout << " tr = " << trace(rhos) << std::endl;
            // --- save D to D.dat for cont 
            if (N == Nmax){
                double time2 = MPI_Wtime();
                std::cout << "----- saving D to D.dat ----------------" << std::endl;
                save_D (N, theta, D);
                std::cout << "    lap time = " << MPI_Wtime() - time2 << " sec" << std::endl;
                /***********************************************
                // store backup files in zip (optional)
                std::system(
                        "now=`date '+%Y_%m_%d_%H%M%S'`;\
                        zip -rq D.dat.$now.zip D.dat;\
                        zip -rq rhos.dat.$now.zip rhos.dat;");  
                ************************************************/
            }
        }
    }
    if (myid == root){
        std::cout << "----- end -----------------------------" << std::endl;
        std::cout << "    elapsed_time = " << MPI_Wtime() - time0 << " sec" << std::endl;
        std::cout << "---------------------------------------" << std::endl;
    }
    MPI_Finalize();
}

//=======================  EOF  ================================================
