/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.0.0 (x.0.x serial)
 * Date Created : 8/15/20
 * Date Last mod: 9/7/20
 * Author: Yoshihiro Sato
 * Description: Functions used in squapi.cpp, squapi_omp.cpp, squapi_mpi.cpp
 *              and  squapi_cont_xxx.cpp
 * Notes:
 *      - All Eq.(x) are refering to the corresponding equation numbers in 
 *        Y.Sato, Journal of Chemical Physics 150 (2019) 224108.
 *      - Based on C++11
 *      - Develped using MacOS 10.14 and Xcode 11.3
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
#include <time.h>
#include "inlines.h"
#include "phys.h"

// ************************************* Functions  *****************************************

void getdata(std::string filename, 
             std::vector<std::vector<std::complex<double>>>& arrs, 
             std::vector<std::vector<int>>& shapes)
{
    /******************************************************
     * Read a file ("filename") to extract its data array 
     * portion ("arrs") and its shape portion ("shape").
     * Input: filename
     ******************************************************/ 
    // open the data file:
    std::ifstream fin(filename);
    if(!fin){
        std::cout << "---------------------------" << std::endl;
        std::cout << "ERROR: Cannot open the file" << std::endl;
        std::cout << "---------------------------" << std::endl;
        exit(1);
    }
    // start reading the file:
    int narrs, ndim, i, size;    // tmp vars
    double x, y;                 // tmp vars
    std::complex<double> z;      // tmp vars
    
    fin >> narrs;     // narrs = the total number of flatten arrays written in the file
    for(int a = 0; a < narrs; a++){
        fin >> ndim;    // ndim = the actual dimension of array. this will be 1 or 2
        fin.ignore(4);  // skip four chars of "dim " portion
        shapes.emplace_back( std::vector<int>() );   // grow shapes std::vector per array
        size = 1;       // size = the total number of elements in arr
        for(int dim = 1; dim < ndim + 1; dim++){
            fin >> i;
            size *= i;
            shapes[a].emplace_back(i);
            fin.ignore(1);  // skip one char
        }
        // grow arrs std::vector per array
        arrs.emplace_back( std::vector< std::complex<double>>() );
        for(int j = 0; j < size; j++){
            fin >> x;
            fin.ignore(1);  // skip one char
            fin >> y;
            z.real(x);
            z.imag(y);
            arrs[a].emplace_back(z);
        }
    }
    // end reading the file:
    fin.close();    // close the file   
}


std::vector<std::vector<std::complex<double>>> vec2mat(std::vector<std::complex<double>>& arr, 
                                                       std::vector<int>& shape)
{
    /******************************************************
     * Transformas 1D vector ("arr") of complex values into 
     * a 2D matrix of complex values following "shape"  
     * Input: arr, shape
     ******************************************************/ 
    int rows = shape[0];
    int cols = shape[1];
    std::vector<std::complex<double>> vec(cols);           // tmp std::vector
    std::vector<std::vector<std::complex<double>>> matrix; // tmp std::vector
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++)
            vec[j] = arr[i * cols + j];
        matrix.emplace_back(vec);
    }
    return matrix;
}


void load_data(char* argv[], // argv from main()
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
               double& theta)
{
    /****************************************************
     * Loads system.dat and init.dat to extract all the
     * parameters needed for the S-QuAPI compuations. 
     * Input: argv in which
     *  argv[1] = "system.dat"
     *  argv[2] = "init.dat"
     *  argv[3] = Nmax
     *  argv[4] = theta
     * The data format of system.dat and init.dat are
     * written in squapi.py
     ****************************************************/ 
    // Read data from files and put everythin in arrs and shapes:
    std::vector< std::vector< std::complex<double>>> arrs; // data array
    std::vector< std::vector< int >> shapes;               // shape of each of the arrays
    std::vector<std::complex<double>> Dtc;                 // Dt in complex form
    // Extract std::vectors from arrs, reshaping in accordance with shapes:
    getdata(argv[1], arrs, shapes);     // argv[1] = "system.dat"
    Dtc    = arrs[0];
    energy = arrs[1];
    eket   = arrs[2];
    s      = vec2mat(arrs[3], shapes[3]);
    gm0    = arrs[4];
    gm1    = arrs[5];
    gm2    = vec2mat(arrs[6], shapes[6]);
    gm3    = vec2mat(arrs[7], shapes[7]);
    gm4    = vec2mat(arrs[8], shapes[8]);
    arrs.clear();
    shapes.clear();

    // Extract std::vectors from arrs, reshaping in accordance with shapes:
    getdata(argv[2], arrs, shapes);     // argv[2] = "init.dat"
    rhos0 = arrs[0];
    arrs.clear();
    shapes.clear();

    // Get fundamental squapi parameters for time evolution:
    Dt    = Dtc[0].real();
    Nmax  = std::stoi(argv[3]); // Nmax  = argv[3]
    theta = std::stod(argv[4]); // theta = argv[4] 
    Dkmax = gm2[0].size();
    M     = s[0].size();
}


void getU(double Dt,
          std::vector<std::complex<double>>& energy,
          std::vector<std::complex<double>>& eket,
          std::vector<std::complex<double>>& U)
{
    /****************************************************
     * Generates the time evolution operator U that is
     * needed to compute free propagator K of Eq.(5) as 
     * K(s0, s1, s2, s3) = U(s2, s0) * U(s3, s1)
     * Input: Dt, energy, eket
     ****************************************************/ 
    int M = energy.size();
    // --- initialize U ---
    U.resize(M * M);
    // --- generate U ---
    for (int m0 = 0; m0 < M; ++m0){
        for (int m1 = 0; m1 < M; ++m1){
            std::complex<double> u(0, 0);
            for (int m = 0; m < M; ++m){
                std::complex<double> cj(0, -1);  // std::conjugate of j
                auto e  = std::exp(cj * energy[m] * Dt / HBAR);
                u += eket[m * M + m0] * e * std::conj(eket[m * M + m1]);
            }
            //U[m0][m1] = u;
            U[m0 * M  + m1] = u;
        }
    }
}


// ******* Create C and W *********************
void getCW(double theta,
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
    /********************************************
     * Serial version of getCW
     ********************************************/
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
        clock_t time0 = clock(); // for time measurement
        Cn.clear();
        Wn.clear();
        int size = C[n - 1].size();
        for (int i = 0; i < size; i++){
            auto aln_1 = C[n - 1][i];
            auto wn_1  = W[n - 1][i];
            for (int m1 = 0; m1 < M; m1++){
                for (int m2 = 0; m2 < M; m2++){
                    auto aln1 = (unsigned long long)(m1) * ullpow(M, 2*n - 2);
                    auto aln2 = (unsigned long long)(m2) * ullpow(M, 2*n - 1);
                    auto aln = aln_1 + aln1 + aln2;
                    int L = 2 * n;
                    auto arg = num2arg(aln, M, L);
                    auto wn  = wn_1 * R(n, arg, U, s, gm0, gm1, gm2, gm3, gm4);
                    if(abs(wn) >= theta){
                        Cn.emplace_back(aln);
                        Wn.emplace_back(wn);
                    }
                }
            }
        }
        // --- store Cn and Wn into C and W
        C.emplace_back(Cn);
        W.emplace_back(Wn);
        std::cout << "size of C" << n << " = " << C[n].size()
                  << "  lap time = " 
                  << (double)(clock() - time0) / CLOCKS_PER_SEC 
                  << " sec" << std::endl;
    }
}



void getCnmap(std::vector<unsigned long long>& Cn,
              std::unordered_map<unsigned long long, int>& Cnmap)
{
    /*****************************************************
     *   Serial version of getCnmap.
     *   Generates Cnmap for n = Dkmax
     *****************************************************/
    clock_t time0 = clock(); // for time measurement
    auto size = Cn.size();
    for(int i = 0; i < size; i++){
        auto key = Cn[i];
        auto val = i;
        Cnmap[key] = val;
    }
    std::cout << "  lap time = " 
              << (double)(clock() - time0) / CLOCKS_PER_SEC 
              << " sec" << std::endl;
}


// ******* The Density Matrix Module *******
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
           std::vector<std::complex<double>>& D)
{
    /*****************************************************
     *  Computes D based on Eq.(19) for N < Dkamx + 1  
     *  and on Eq.(20) for N = Dkamx + 1
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



// *** The module D1 for N > Dkmax + 1:
void getD1(std::vector<unsigned long long>& Cn_1,
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
             std::vector<std::complex<double>>& rhos)
{
    /**********************************************************
     *  Computes rhos from D based on Eq.(18) for N > 0 
     *********************************************************/
    int Dkmax = gm2[0].size();
    int M     = s[0].size();
    int size  = Cn.size();
    int n     = std::min(N, Dkmax + 1);

    // reset rhos:
    std::fill(rhos.begin(), rhos.end(), std::complex<double>(0, 0));
    if (0 < N && N < Dkmax + 1){
        auto L = 2 * n;
        for (int i = 0; i < size; i++){
            auto aln = Cn[i];   // \alpha_n \in C_n
            auto arg = num2arg(aln, M, L);
            auto m1 = arg[L - 2];
            auto m2 = arg[L - 1];
            auto m = m1 * M + m2;
            auto rho  = Wn[i] * D[i];
            rho *= I(0, n, n, m1, m2, m1, m2, s, gm0, gm1, gm2, gm3, gm4);  
            rhos[m] += rho;
        }
    }
    else if (N >= Dkmax + 1){
        auto L = 2 * (n - 1);
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
                    rhos[m] += rho;  
                }
            }
        }
    }
}

double trace (std::vector<std::complex<double>>& rhos)
{
    /*************************************
     *  Computes trace from rhos
     *************************************/
    int M2 = rhos.size(); // M2 = M * M
    int M  = std::sqrt((double)M2);
    int diag  = 0;
    double tr = 0;
    for(int m = 0; m < M2; ++m){
        auto x = rhos[m].real(); 
        auto y = rhos[m].imag();
        if (m == diag * (M + 1)){
            tr += x;
            diag++;
        }
    }
    return tr;
}


void save_rhos (int N, std::vector<std::complex<double>>& rhos)
{
    /*************************************
     *  Saves rhos array as rhos.dat 
     *************************************/
    int M2 = rhos.size();   // M2 = M * M
    std::ofstream fout;
    if (N == 0){
        // open the file in 'overwirte' mode:
        fout.open("rhos.dat");
        // file formatted for numpy loadtxt:
        fout<< "# N,  rhos(N)" << std::endl;
    }
    else{
        // open the file in 'apped' mode:
        fout.open("rhos.dat", std::ios::app);
    }
    for(int m = 0; m < M2; ++m){
        auto x = rhos[m].real();
        auto y = rhos[m].imag();
        fout << N << ",";
        fout << std::scientific << std::setprecision(16)
                 << rhos[m].real() << std::showpos << rhos[m].imag() << "j" << std::endl;
        fout << std::noshowpos;
    }
    fout.close();
}

//=======================  EOF  ================================================
