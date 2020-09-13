/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.0.0 (serial)
 * Date Created : 8/15/20
 * Date Last mod: 8/26/20
 * Author: Yoshihiro Sato
 * Description: Header file for functions written in sqmodule.cpp 
 * Notes:  
 *      - Based on C++11
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved
 **********************************************************************************/

void getdata(std::string filename, 
             std::vector<std::vector<std::complex<double>>>& arrs, 
             std::vector<std::vector<int>>& shapes);

std::vector<std::vector<std::complex<double>>> vec2mat(std::vector<std::complex<double>>& arr, 
                                                       std::vector<int>& shape);

void load_data(char* argv[],
               std::vector<std::complex<double>>& energy,
               std::vector<std::complex<double>>& eket,
               std::vector<std::complex<double>>& U,
               std::vector<std::vector<std::complex<double>>>& s,
               std::vector<std::complex<double>>& gm0,
               std::vector<std::complex<double>>& gm1,
               std::vector<std::vector<std::complex<double>>>& gm2,
               std::vector<std::vector<std::complex<double>>>& gm3,
               std::vector<std::vector<std::complex<double>>>& gm4,
               std::vector<std::complex<double>>& rhos0,
               int& Nmax, 
               int& Dkmax, 
               int& M,
               double& Dt, 
               double& theta);

void getU(double Dt,
          std::vector<std::complex<double>>& energy,
          std::vector<std::complex<double>>& eket,
          std::vector<std::complex<double>>& U);

void getCW(double theta,
           std::vector<std::complex<double>>& U,
           std::vector<std::vector<std::complex<double>>>& s, 
           std::vector<std::complex<double>>& gm0,
           std::vector<std::complex<double>>& gm1,
           std::vector<std::vector<std::complex<double>>>& gm2,
           std::vector<std::vector<std::complex<double>>>& gm3,
           std::vector<std::vector<std::complex<double>>>& gm4,
           std::vector<std::vector<unsigned long long>>& C,
           std::vector<std::vector<std::complex<double>>>& W);

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

void getD0(int N, 
           std::vector<unsigned long long>& Cn_1,
           std::vector<std::complex<double>>& rhos0,
           std::vector<std::complex<double>>& U,
           std::vector<std::vector<std::complex<double>>>& s, 
           std::vector<std::complex<double>>& gm0,
           std::vector<std::complex<double>>& gm1,
           std::vector<std::vector<std::complex<double>>>& gm2,
           std::vector<std::vector<std::complex<double>>>& gm3,
           std::vector<std::vector<std::complex<double>>>& gm4,
           std::vector<std::complex<double>>& D);

void getD1(std::vector<unsigned long long>& Cn_1,
           std::unordered_map<unsigned long long, int>& Cnmap,
           std::vector<std::complex<double>>& U,
           std::vector<std::vector<std::complex<double>>>& s, 
           std::vector<std::complex<double>>& gm0,
           std::vector<std::complex<double>>& gm1,
           std::vector<std::vector<std::complex<double>>>& gm2,
           std::vector<std::vector<std::complex<double>>>& gm3,
           std::vector<std::vector<std::complex<double>>>& gm4,
           std::vector<std::complex<double>>& D);

void getCnmap(std::vector<unsigned long long>& Cn,
              std::unordered_map<unsigned long long, int>& Cnmap);

void getrhos(int N,
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

void getrhosK(int M,
              std::vector<std::complex<double>>& U,
              std::vector<std::complex<double>>& rhos);

double trace (std::vector<std::complex<double>>& rhos); 

void save_rhos (int N, std::vector<std::complex<double>>& rhos, std::string filename);

//=======================  EOF  ================================================
