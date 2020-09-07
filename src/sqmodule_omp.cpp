/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.1.0 (x.1.x OpenMP)
 * Date Created : 8/15/20
 * Date Last mod: 9/7/20
 * Author: Yoshihiro Sato
 * Description: Functions used in squapi_omp.cpp squapi_cont_omp.cpp 
 * Notes:
 *      - All Eq.(x) are refering to the corresponding equation numbers in 
 *        Y.Sato, Journal of Chemical Physics 150 (2019) 224108
 *      - Based on C++11
 *      - Develped using gcc 10.2.0 on MacOS 10.14 
 *      - size of Cn has to be lower than 2,147,483,647 (limit of int)
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
#include <algorithm>
#include <numeric>
#include <omp.h>
#include "inlines.h"

// ************************************* Functions  *****************************************
void getCW_omp(double theta,
               std::vector<std::complex<double>>& U,
               std::vector<std::vector<std::complex<double>>>& s, 
               std::vector<std::complex<double>>& gm0,
               std::vector<std::complex<double>>& gm1,
               std::vector<std::vector<std::complex<double>>>& gm2,
               std::vector<std::vector<std::complex<double>>>& gm3,
               std::vector<std::vector<std::complex<double>>>& gm4,
               std::vector<std::vector<unsigned long long>>& C,
               std::vector<std::vector<std::complex<double>>>& W)
{
    /**************************************************
     *  OpenMP version of getCW.
     *  - still needs a tweak for performance 
     **************************************************/
    auto M     = s[0].size();
    auto Dkmax = gm2[0].size();
    auto nmax  = Dkmax;

    std::vector<unsigned long long>   Cn;
    std::vector<std::complex<double>> Wn;
    
    // *** initialize C and W ***
    C.clear();
    W.clear();
    // *** n = 0: insert empty std::vectors for n = 0 
    Cn.clear();
    Wn.clear();
    C.emplace_back(Cn);
    W.emplace_back(Wn);
    // *** n = 1:
    Cn.resize(M * M);
    Wn.resize(M * M);
    std::iota(Cn.begin(), Cn.end(), 0); // C1 = {0, 1, 2, ..., M * M -1}
    std::fill(Wn.begin(), Wn.end(), std::complex<double>(1, 0));
    C.emplace_back(Cn);
    W.emplace_back(Wn);
    std::cout << "size of C1 = " << Cn.size() << std::endl;
    // *** n > 1:
    for(int n = 2; n < nmax + 1; n++){
        double time0 = omp_get_wtime();    
        Cn.clear();
        Wn.clear();
        int size = C[n - 1].size();
        #pragma omp parallel
        {
            std::vector<unsigned long long>   Cnbuf;
            std::vector<std::complex<double>> Wnbuf;
            #pragma omp for ordered schedule(static, 1)
            for(int i = 0; i < size; i++){
                Cnbuf.clear();
                Wnbuf.clear();
                auto aln_1 = C[n - 1][i];
                auto wn_1  = W[n - 1][i];
                for(int m1 = 0; m1 < M; m1++){
                    for(int m2 = 0; m2 < M; m2++){
                        auto aln1 = (unsigned long long)(m1) * ullpow(M, 2*n - 2);
                        auto aln2 = (unsigned long long)(m2) * ullpow(M, 2*n - 1);
                        auto aln = aln_1 + aln1 + aln2;
                        int L = 2 * n;
                        auto arg = num2arg(aln, M, L);
                        auto wn  = wn_1 * R(n, arg, U, s, gm0, gm1, gm2, gm3, gm4);
                        if(abs(wn) >= theta){
                            Cnbuf.emplace_back(aln);
                            Wnbuf.emplace_back(wn);
                        }
                    }
                }
                #pragma omp ordered
                {
                    for(auto aln: Cnbuf){ Cn.emplace_back(aln); }
                    for(auto wn:  Wnbuf){ Wn.emplace_back(wn);  }
                }
            }
        }
        // --- store Cn and Wn into C and W
        C.emplace_back(Cn);
        W.emplace_back(Wn);
        std::cout << "size of C" << n << " = " << C[n].size()
                  << "  lap time = " << omp_get_wtime() - time0 << " sec" << std::endl;
    }
}


void getCnmap_omp(std::vector<unsigned long long>& Cn,
                  std::unordered_map<unsigned long long, int>& Cnmap)
{
    /*****************************************************
     *   OpenMP version of getCnmap.
     *   NOT FOR USE YET
     *   VERY SLOW: even slower than the serial getCnmap
     *****************************************************/
    auto size = Cn.size();
    double time0 = omp_get_wtime();    
    #pragma omp parallel
    {
        #pragma omp for ordered schedule(static, 1)
        for(int i = 0; i < size; i++){
            auto key = Cn[i];
            auto val = i;
            #pragma omp ordered 
            {
                Cnmap[key] = i;
            }
        }
    }
    std::cout << "  lap time = " 
              << omp_get_wtime() - time0 
              << " sec" << std::endl;
}



// ******* The Density Matrix Modules *******
void getD0_omp(int N, 
               std::vector<unsigned long long>& Cn_1,
               std::vector<std::complex<double>>& rhos0,
               std::vector<std::complex<double>>& U,
               std::vector<std::vector<std::complex<double>>>& s, 
               std::vector<std::complex<double>>& gm0,
               std::vector<std::complex<double>>& gm1,
               std::vector<std::vector<std::complex<double>>>& gm2,
               std::vector<std::vector<std::complex<double>>>& gm3,
               std::vector<std::vector<std::complex<double>>>& gm4,
               std::vector<std::complex<double>>& D)
{
    /*****************************************************
     *  Computes D based on Eq.(19) for N < Dkamx + 1  
     *  and on Eq.(20) for N = Dkamx + 1
     *  OpenMP version of getD0.
     *  So far the line "#pragma omp parallel for" is the
     *  only difference from serial getD0.
     *****************************************************/
    int  M     = s[0].size();
    int  Dkmax = gm2[0].size();
    int  n     = std::min(N, Dkmax + 1);
    auto Dkf   = std::min(n, Dkmax); // upper limit of Dk in propagator element
    auto L     = 2 * (Dkf + 1);      // length of arguments in propagator
    auto size  = Cn_1.size();

    // reset D
    D.resize(size);
    std::fill(D.begin(), D.end(), std::complex<double>(0, 0));

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        auto aln_1 = Cn_1[i];
        auto arg = num2arg(aln_1, M, L - 2);
        // currently, arg => \alpha_{n-1} = \{s_1^\pm, s_2\pm, \cdots, s_{n-1}^\pm \}
        arg.emplace_front(0);
        arg.emplace_front(0); 
        // now arg => \{s_0^\pm, \alpha_{n-1}\} with dummy values for s_0^\pm
        for(int m0 = 0; m0 < M; m0++){
            for(int m1 = 0; m1 < M; m1++){
                arg[0] = m0;
                arg[1] = m1;
                auto p0 = P(0, n, M, Dkmax, arg, U, s, gm0, gm1, gm2, gm3, gm4);
                D[i] += p0 * rhos0[m0 * M + m1];
            }
        }
    }
}


void getD1_omp(std::vector<unsigned long long>& Cn_1,
               std::unordered_map<unsigned long long, int>& Cnmap,
               std::vector<std::complex<double>>& U,
               std::vector<std::vector<std::complex<double>>>& s, 
               std::vector<std::complex<double>>& gm0,
               std::vector<std::complex<double>>& gm1,
               std::vector<std::vector<std::complex<double>>>& gm2,
               std::vector<std::vector<std::complex<double>>>& gm3,
               std::vector<std::vector<std::complex<double>>>& gm4,
               std::vector<std::complex<double>>& D)
{
    /*****************************************************
     *  Computes D based on Eq.(21) for N > Dkamx + 1  
     *  OpenMP version of getD1.
     *  So far the line "#pragma omp parallel for" is the
     *  only difference from serial getD1.
     *****************************************************/
    auto M     = s[0].size();
    auto Dkmax = gm2[0].size();
    auto n     = Dkmax + 1;
    auto L     = 2 * n;
    auto size  = Cn_1.size();

    // --- copy D from the previous time step (N-1)
    std::vector<std::complex<double>> Dprev = D;
    
    // compute averate of D
    std::complex<double> Dave, zsize(D.size(), 0);
    auto sum = std::accumulate(D.begin(), D.end(), std::complex<double>(0, 0));
    Dave = sum / zsize;

    // --- reset D
    D.resize(size);
    std::fill(D.begin(), D.end(), std::complex<double>(0, 0));

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        auto aln_1 = Cn_1[i];
        auto arg = num2arg(aln_1, M, L - 2);  
        // currently, arg = \{s_1^\pm, s_2\pm, \cdots, s_{Dkmax}^\pm \}
        arg.emplace_front(0);
        arg.emplace_front(0);     
        // now arg = \{s_0^\pm, s_1^\pm, \cdots, s_{Dkmax}^\pm \} with dummy values for s_0^\pm
        for(int m0 = 0; m0 < M; m0++){
            for(int m1 = 0; m1 < M; m1++){
                arg[0] = m0; // arg[0] = s_0^+
                arg[1] = m1; // arg[1] = s_0^-
                // --- generate \alpha'_{Dkmax} ---
                auto argp = arg; // this is to be \alpha'_{Dkmax}
                // currently, argp = {s_0^\pm, s_1\pm, \cdots, s_{Dkmax}^\pm \} 
                argp.pop_back();
                argp.pop_back();  
                // now argp = {s_0^\pm, s_1^\pm, \cdots, s_{Dkmax-1}^\pm \}
                auto alp = arg2num(argp, M);     // alp = \alpha' compressed for search
                //********************************************************
                std::complex<double> Dval;
                try{
                    int j = Cnmap.at(alp);
                    Dval = Dprev[j];
                }
                catch(std::out_of_range&){
                    Dval = Dave;
                    //Dval = std::complex<double>(0, 0); // alternative
                }
                D[i] += P(1, n, M, Dkmax, arg, U, s, gm0, gm1, gm2, gm3, gm4) * Dval;
            }
        }
    }
}



// **** Compute rhos from C, W, and D *************
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
                 std::vector<std::complex<double>>& rhos)
{
    /**********************************************************
     *  Computes rhos from D based on Eq.(18) for N > 0. 
     *  OpenMP version of getrhos.
     * *******************************************************/
    int Dkmax = gm2[0].size();
    int M     = s[0].size();
    int size  = Cn.size();
    int n     = std::min(N, Dkmax + 1);

    // reset rhos:
    std::fill(rhos.begin(), rhos.end(), std::complex<double>(0, 0));
    #pragma omp parallel
    {
        std::vector<std::complex<double>> rhosbuf(M * M);
        std::fill(rhosbuf.begin(), rhosbuf.end(), std::complex<double>(0, 0));
        if (0 < N && N < Dkmax + 1){
            auto L = 2 * n;
            #pragma omp for
            for (int i = 0; i < size; i++){
                auto aln = Cn[i];   // \alpha_n \in C_n
                auto arg = num2arg(aln, M, L);
                auto m1 = arg[L - 2];
                auto m2 = arg[L - 1];
                auto m = m1 * M + m2;
                auto rho  = Wn[i] * D[i];
                rho *= I(0, n, n, m1, m2, m1, m2, s, gm0, gm1, gm2, gm3, gm4);  
                rhosbuf[m] += rho;  
            }
        }
        else if (N >= Dkmax + 1){
            auto L = 2 * (n - 1);
            #pragma omp for
            for (int i = 0; i < size; i++){
                auto aln = Cn[i];   // \alpha_{n-1} \in C_{n-1}
                for (int m1 = 0; m1 < M; ++m1){
                    for (int m2 = 0; m2 < M; ++m2){
                        auto arg = num2arg(aln, M, L);
                        auto m = m1 * M + m2;
                        arg.emplace_back(m1); // m1 = s_n^+
                        arg.emplace_back(m2); // m2 = s_n^-
                        auto rho = Wn[i] * D[i];
                        rho *= R(n, arg, U, s, gm0, gm1, gm2, gm3, gm4);
                        rho *= I(0, n, n, m1, m2, m1, m2, s, gm0, gm1, gm2, gm3, gm4);  
                        rhosbuf[m] += rho;  
                    }
                }
            }
        }
        // atomic operation needed: 
        #pragma omp critical
        { 
            for (int m = 0; m < M * M; m++){
                rhos[m] += rhosbuf[m];
            }
        }
    }
}


//=======================  EOF  ================================================
