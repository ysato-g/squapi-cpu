/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.0
 * Date Created : 8/15/20
 * Date Last mod: 10/8/20
 * Author: Yoshihiro Sato
 * Description: the main function of genrhos 
 * Compile: $g++ -std=c++11 squapi.cpp sqmodule.cpp -o squapi 
 * Usage: $./squapi system.dat init.dat (Nmax) (theta)
 * Notes: 
 *      - Functions used in this program are written in sqmodule.cpp
 *      - Based on C++11
 *      - Develped using MacOS 10.14 and Xcode 11.3 
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
#include <time.h>
#include "sqmodule.h"


int main(int argc, char* argv[])
{
    clock_t time0 = clock();
    
    // Define S-QuAPI quantities:
    std::vector<std::complex<double>> U;
    std::vector<std::vector<std::complex<double>>> s;
    std::vector<std::complex<double>> gm0, gm1;
    std::vector<std::vector<std::complex<double>>> gm2, gm3, gm4;
    std::vector<std::complex<double>> rhos0, D;
    int Nmax, Dkmax, M;
    double Dt, theta;
    
    // Load data from files and store them in the S-QuAPI parameters: 
    load_data(argv, U, s, gm0, gm1, gm2, gm3, gm4, rhos0, Nmax, Dkmax, M, Dt,theta);

    // Show S-QuAPI parameters:
    std::cout << "****************************************" << std::endl;
    std::cout << "*            squapi ver 0.0            *" << std::endl; 
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
    // Generate the paths and weights:
    std::cout << "----- generate Cn and Wn ---------------" << std::endl;
    std::vector<std::vector<unsigned long long>>   C;
    std::vector<std::vector<std::complex<double>>> W;
    getCW(theta, U, s, gm0, gm1, gm2, gm3, gm4, C, W);
    
    // Generate the key-value map for searching D:
    std::cout << "----- generate Cnmap -------------------" << std::endl;
    clock_t time1 = clock(); // for time measurement
    std::unordered_map<unsigned long long, int> Cnmap;
    getCnmap(C[Dkmax], Cnmap);
    std::cout << "  lap time = " << (double)(clock() - time1) / CLOCKS_PER_SEC << " sec" << std::endl;

    // Generate the density matrix by the time evolution:
    std::cout << "----- generate rhos --------------------" << std::endl;
    std::vector<std::complex<double>> rhos(M * M);
    for (int N = 0; N < Nmax + 1; N++){
        time1 = clock(); // reset time1 for time measurement
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
            getD0(N, C[n], rhos0, U, s, gm0, gm1, gm2, gm3, gm4, D);
            getrhos(N, U, C[n], W[n], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
        }
        else if (N == Dkmax + 1 && Dkmax > 0){
            /******************* STEP 3 **************************/
            n = Dkmax + 1;
            getD0(N, C[n-1], rhos0, U, s, gm0, gm1, gm2, gm3, gm4, D);
            getrhos(N, U, C[n-1], W[n-1], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
        }
        else{
            /******************* STEP 4 **************************/
            n = Dkmax + 1;
            getD1(C[n-1], Cnmap, U, s, gm0, gm1, gm2, gm3, gm4, D);
            getrhos(N, U, C[n-1], W[n-1], D, s, gm0, gm1, gm2, gm3, gm4, rhos);
        }
        // Write N and rhos into rhos.dat:
        save_rhos(N, rhos, "rhos.dat");

        std::cout << "N = " << N << " of " << Nmax;
        std::cout << " lap time = " << (double)(clock() - time1) / CLOCKS_PER_SEC  << " sec"; 
        std::cout << " tr = " << trace(rhos) << std::endl;
    }

    std::cout << "----- end -----------------------------" << std::endl;
    std::cout << "elapsed_time = " << (double)(clock() - time0) / CLOCKS_PER_SEC << " sec" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

}

//=======================  EOF  ================================================
