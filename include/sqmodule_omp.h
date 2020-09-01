/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.1.0 (OpenMP, made for multi-core single node)
 * Date Created : 8/15/20
 * Date Last mod: 8/26/20
 * Author: Yoshihiro Sato
 * Description: Header file for functions written in sqmodule_omp.cpp 
 * Notes:    
 *      - Based on C++11
 *      - Develped using gcc 10.2.0 and OpenMP included 
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved
 **********************************************************************************/

void getCW_omp(double theta,
           std::vector<std::complex<double>>& U,
           std::vector<std::vector<std::complex<double>>>& s, 
           std::vector<std::complex<double>>& gm0,
           std::vector<std::complex<double>>& gm1,
           std::vector<std::vector<std::complex<double>>>& gm2,
           std::vector<std::vector<std::complex<double>>>& gm3,
           std::vector<std::vector<std::complex<double>>>& gm4,
           std::vector<std::vector<unsigned long long>>& C,
           std::vector<std::vector<std::complex<double>>>& W);

// ******* The Density Matrix Module *******
void getD0_omp(int N, 
               std::vector<unsigned long long>& Cn, // this can be Cn or Cn_1
               std::vector<std::complex<double>>& rhos0,
               std::vector<std::complex<double>>& U,
               std::vector<std::vector<std::complex<double>>>& s, 
               std::vector<std::complex<double>>& gm0,
               std::vector<std::complex<double>>& gm1,
               std::vector<std::vector<std::complex<double>>>& gm2,
               std::vector<std::vector<std::complex<double>>>& gm3,
               std::vector<std::vector<std::complex<double>>>& gm4,
               std::vector<std::complex<double>>& D);

void getD1_omp(std::vector<unsigned long long>& Cn,
               std::unordered_map<unsigned long long, int>& Cnmap,
               std::vector<std::complex<double>>& U,
               std::vector<std::vector<std::complex<double>>>& s, 
               std::vector<std::complex<double>>& gm0,
               std::vector<std::complex<double>>& gm1,
               std::vector<std::vector<std::complex<double>>>& gm2,
               std::vector<std::vector<std::complex<double>>>& gm3,
               std::vector<std::vector<std::complex<double>>>& gm4,
               std::vector<std::complex<double>>& D);

void getCnmap_omp(std::vector<unsigned long long>& Cn,
              std::unordered_map<unsigned long long, int>& Cnmap);

void getrhos_omp(int N,
                 std::vector<std::complex<double>>& U,
                 std::vector<unsigned long long>&   Cn,
                 std::vector<std::complex<double>>& Wn,
                 std::vector<std::complex<double>>& D,
                 std::vector<std::vector<std::complex<double>>>& s, 
                 std::vector<std::complex<double>>& gm0,
                 std::vector<std::complex<double>>& gm1,
                 std::vector<std::vector<std::complex<double>>>& gm2,
                 std::vector<std::vector<std::complex<double>>>& gm3,
                 std::vector<std::vector<std::complex<double>>>& gm4,
                 std::vector<std::complex<double>>& rhos);

//=======================  EOF  ================================================
