/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.1.1 (OpenMP, made for multi-core single node)
 * Date Created : 8/15/20
 * Date Last mod: 10/13/20
 * Author: Yoshihiro Sato
 * Description: the main function of squapi_omp 
 * Usage: $ squapi_omp system.dat init.dat (Nmax) (theta) (--cont)
 * Notes: 
 *      - Functions used in this program are written in sqmodule.cpp and sqmodule_omp.cpp
 *      - Based on C++11
 *      - Develped using gcc 10.2.0 on MacOS 10.14 
 *      - size of Cn has to be lower than 2,147,483,647 (limit of int)
 *      - Supports Dkmax = 0
 *      - Takes "--cont" option for continuation to larger Nmax value
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
#include <omp.h>
#include "sqmodule.h"
#include "sqmodule_omp.h"
#include "opt.h"


int main(int argc, char* argv[])
{
    double time0 = omp_get_wtime(); // for time measurement using OMP
    
    // --- global S-QuAPI variables:
    std::vector<std::complex<double>> U;
    std::vector<std::vector<std::complex<double>>> s;
    std::vector<std::complex<double>> gm0, gm1;
    std::vector<std::vector<std::complex<double>>> gm2, gm3, gm4;
    std::vector<std::complex<double>> rhos0, D;
    int Nmax, Dkmax, M, N0 = -1;
    double Dt, theta;

    // load data from files and store them in the S-QuAPI parameters: 
    load_data(argv, U, s, gm0, gm1, gm2, gm3, gm4, rhos0, Nmax, Dkmax, M, Dt, theta);

    // (optional) load D.dat and rhos.dat to continue from last saved point:
    opt_load_D(argc, argv, Nmax, Dkmax, theta, "D.dat", "rhos.dat", N0, D);    

    // -------- load S-QuAPI parameters -------------------------
    std::cout << "****************************************" << std::endl;
    std::cout << "*         squapi_omp ver 0.0           *" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "----- Date and Time --------------------" << std::endl;
    std::system("date");
    std::cout << "----- parameters -----------------------" << std::endl;
    std::cout << "Dt     = " << Dt << " ps" << std::endl;
    std::cout << "Nmax   = " << Nmax   << std::endl;
    std::cout << "Dkmax  = " << Dkmax  << std::endl;
    std::cout << "M      = " << M      << std::endl;
    std::cout << "theta  = " << theta  << std::endl;
    
    /******************* STEP 1 **************************/
    // -------- Generate the paths and weights -------------------------
    std::cout << "----- generate Cn and Wn ---------------" << std::endl;
    std::vector<std::vector<unsigned long long>>   C;
    std::vector<std::vector<std::complex<double>>> W;
    getCW_omp(theta, U, s, gm0, gm1, gm2, gm3, gm4, C, W);
    
    // --- generate Cnmap ------------
    std::cout << "----- generate Cnmap -------------------" << std::endl;
    double time1 = omp_get_wtime(); // for time measurement using OMP
    std::unordered_map<unsigned long long, int> Cnmap;
    getCnmap(C[Dkmax], Cnmap);
    //getCnmap_omp(C[Dkmax], Cnmap); // very slow. for devel only 
    std::cout << "  lap time = " << omp_get_wtime() - time1 << " sec" << std::endl;

    // -------- Generate rhos ---------------------------------------
    std::cout << "----- generate rhos --------------------" << std::endl;

    // ***** Start time evolution **************
    std::vector<std::complex<double>> rhos(M * M);
    for (int N = N0 + 1; N < Nmax + 1; N++){
        time1 = omp_get_wtime(); // reset time1 for time measurement using OMP
        int n;
        if (N == 0){
            rhos = rhos0;
        }
        else if (N > 0 && Dkmax == 0){
            getrhosK(M, U, rhos);
        }
        else if (N > 0 && N < Dkmax + 1 && Dkmax > 0){
            /******************* STEP 2 **************************/
            n = N;
            getD0_omp(N, C[n], rhos0, U, s, gm0, gm1, gm2, gm3, gm4, D);
            getrhos_omp(N, U, C[n], W[n], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
        }
        else if (N == Dkmax + 1 && Dkmax > 0){
            /******************* STEP 3 **************************/
            n = Dkmax + 1;
            getD0_omp(N, C[n-1], rhos0, U, s, gm0, gm1, gm2, gm3, gm4, D);
            getrhos_omp(N, U, C[n-1], W[n-1], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
        }
        else{
            /******************* STEP 4 **************************/
            n = Dkmax + 1;
            getD1_omp(C[n-1], Cnmap, U, s, gm0, gm1, gm2, gm3, gm4, D);
            getrhos_omp(N, U, C[n-1], W[n-1], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
        }
        // *** write N and rhos into rhos.dat ***
        save_rhos(N, rhos, "rhos.dat");

        std::cout << "N = " << N << " of " << Nmax;
        std::cout << " lap time = " << omp_get_wtime() - time1 << " sec"; 
        std::cout << " tr = " << trace(rhos) << std::endl;
    }

    // (optional) save D to D.dat for cont 
    opt_save_D(argc, argv, Nmax, theta, D, "D.dat");

    std::cout << "----- end ------------------------------" << std::endl;
    std::cout << "    elapsed_time = " << omp_get_wtime() - time0 << " sec" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

//=======================  EOF  ================================================
