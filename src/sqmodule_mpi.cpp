/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.2.1 (x.2.x MPI)
 * Date Created : 8/23/20
 * Date Last mod: 9/2/20
 * Author: Yoshihiro Sato
 * Description: Functions used in squapi_mpi.cpp and squapi_cont_mpi.cpp 
 * Notes:
 *      - All Eq.(x) are refering to the corresponding equation numbers in 
 *        Y.Sato, Journal of Chemical Physics 150 (2019) 224108
 *      - Based on C++11
 *      - Develped using gcc 10.2.0 on MacOS 10.14 
 *      - size of Cn has to be lower than 2,147,483,647 (limit of int)
 *      - getrhos_mpi added on 9/2/20
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
#include <mpi.h>
#include "inlines.h"
#include "sysconf.h"

// ************************************* Functions  *****************************************

void round_manage_cpu(int myid, int nprocs,
                      int size, int byte_per_block,
                      int& block_reg, int& block_fin,
                      int& nblocks_reg,
                      int& nranks_reg, int& nranks_sem, int& nranks_fin,
                      int& nrounds_reg, int& nrounds,
                      MPI_Comm& comm_reg,
                      MPI_Comm& comm_sem,
                      MPI_Comm& comm_fin)
{
    // --- set max block size
    int block_max = MAX_RAM_PER_CORE_MIB * MIB / byte_per_block;
    // --- determine block_reg and block_fin, the size of data block per core 
    // ********************* set up deterministic params *********************
    int scale = 2; // set this from 2 to 4 for best performance
    block_reg = block_max / std::max(1, nprocs / scale); // best so far
    nranks_reg = nprocs;   // the # of cpu cores in the regular process >= 0
    // ***********************************************************************
    if (block_reg > size / nprocs) block_reg = size / nprocs; // shrink the block size if too large
    if (block_reg < 1) block_reg = size; // shrink the block size if way too small

    nblocks_reg = size / block_reg;
    // compute quantities for the semi-final round:
    nranks_sem = nblocks_reg % nprocs;
    // compute quantities for the final round:
    block_fin = size % block_reg;
    nranks_fin = (block_fin == 0)? 0 : 1;
    // compute the # of rounds:
    nrounds_reg = nblocks_reg / nranks_reg; // the # of regular rounds
    nrounds     = nrounds_reg;              // the total # of rounds
    if (nranks_sem != 0) nrounds++;         // add the semi-final round if exists
    if (nranks_fin != 0) nrounds++;         // add the final round if exists
    // set up communicaters for the rounds:
    int color;
    // for the regular rounds:
    color = (myid < nranks_reg)? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, myid, &comm_reg);
    // for the semi final round which is done by the processes in comm_sem:
    color = (myid < nranks_sem)? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, myid, &comm_sem);
    // for the final round which is done by the root only: 
    color = (myid < nranks_fin)? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, myid, &comm_fin);

}
     


void getCW_mpi(int myid, int nprocs, int root,
          double theta,
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
    /***************************************************
     * MPI version of getCW with memory management
     * Collective calls are exclusively used for
     * performance
     ***************************************************/
    int M     = s[0].size();
    int Dkmax = gm2[0].size();
    int nmax  = Dkmax;

    std::vector<unsigned long long>   Cn;
    std::vector<std::complex<double>> Wn;
    
    // *** initialize C and W ***
    C.clear();
    W.clear();
    // *** n = 0: insert empty std::vectors for n = 0 
    Cn.clear();
    Wn.clear();
    C.push_back(Cn);
    W.push_back(Wn);
    // *** n = 1:
    Cn.resize(M * M);
    Wn.resize(M * M);
    std::iota(Cn.begin(), Cn.end(), 0); // C1 = {0, 1, 2, ..., M * M -1}
    std::fill(Wn.begin(), Wn.end(), std::complex<double>(1, 0));
    C.push_back(Cn);
    W.push_back(Wn);
    if (myid == root){
        std::cout << "size of C1 = " << Cn.size() << std::endl;
    }
    // *** n > 1:
    for(int n = 2; n < nmax + 1; n++){
        double time0 = MPI_Wtime();
        Cn.clear();
        Wn.clear();
        int size;
        if (myid == root){
            size = C[n - 1].size();
        }
        MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
        
        // bufferes used for Scatter / Gather:
        std::vector<unsigned long long>   Cn_1buf,    Cnbuf;
        std::vector<unsigned long long>   Cn_1bufall, Cnbufall;
        std::vector<std::complex<double>> Wn_1buf,    Wnbuf;
        std::vector<std::complex<double>> Wn_1bufall, Wnbufall;

        // memory size in BYTE per data block based on the following bufferes:
        // Cn_1buf, Wn_1buf, Cnbuf, Wnbuf
        int byte_per_block =  16 + 16 + M * M * 16 + M * M * 16;
        
        int block_reg, block_fin;
        int nblocks_reg;
        int nranks_reg, nranks_sem, nranks_fin;
        int nrounds_reg, nrounds;
        MPI_Comm comm_reg, comm_sem, comm_fin;

        round_manage_cpu(myid, nprocs,
                         size, byte_per_block,
                         block_reg, block_fin, nblocks_reg,
                         nranks_reg, nranks_sem, nranks_fin,
                         nrounds_reg, nrounds,
                         comm_reg, comm_sem, comm_fin);
 
        // root extends the "bufall" to the largest size:
        if (myid == root){
            Cn_1bufall.resize(block_reg * nprocs);
            Wn_1bufall.resize(block_reg * nprocs);
            Cnbufall.resize(block_reg * M * M * nprocs);
            Wnbufall.resize(block_reg * M * M * nprocs);
        }

        int i0 = 0; // the start position of C[n-1] and W[n-1] for the round
        for (int round = 0; round < nrounds; ++round){
            MPI_Comm comm;
            int block;
            // set communicator and block size for the round:
            if (round < nrounds_reg) {
                comm  = comm_reg;
                block = block_reg;
            }
            else if (round == nrounds - 1 && nranks_fin != 0) {
                comm  = comm_fin;
                block = block_fin;
            }
            else {
                comm  = comm_sem;
                block = block_reg;
            }
            if (comm != MPI_COMM_NULL) {
                // ========= process exclusive for comm BEGIN =========
                int rank, nranks;
                // get info about comm:
                MPI_Comm_rank(comm, &rank);
                MPI_Comm_size(comm, &nranks);
                // resize buffers:
                if (round == 0 || block != block_reg){
                    Cn_1buf.resize(block);
                    Wn_1buf.resize(block);
                    Cnbuf.resize(M * M * block);
                    Wnbuf.resize(M * M * block);
                }
                // root transferes data from C[n-1] and W[n-1] to buffers:
                if (rank == root) {
                    std::copy(C[n - 1].begin() + i0, C[n - 1].begin() + i0 + block * nranks, Cn_1bufall.begin());
                    std::copy(W[n - 1].begin() + i0, W[n - 1].begin() + i0 + block * nranks, Wn_1bufall.begin());
                }
                // root scatters Cn_1bufall and Wn_1bufall into all ranks in comm:
                MPI_Scatter(Cn_1bufall.data(), block, MPI_UNSIGNED_LONG_LONG,
                            Cn_1buf.data(),    block, MPI_UNSIGNED_LONG_LONG, root, comm);
                MPI_Scatter(Wn_1bufall.data(), block, MPI_C_DOUBLE_COMPLEX,
                            Wn_1buf.data(),    block, MPI_C_DOUBLE_COMPLEX,   root, comm);
                // each rank in comm generates Cnbuf and Wnbuf from Cn_1buf and Wn_1buf: 
                for (int i = 0; i < block; i++){
                    auto aln_1 = Cn_1buf[i];
                    auto wn_1  = Wn_1buf[i];
                    for (int m1 = 0; m1 < M; m1++){
                        for (int m2 = 0; m2 < M; m2++){
                            auto aln1 = (unsigned long long)(m1) * ullpow(M, 2*n - 2);
                            auto aln2 = (unsigned long long)(m2) * ullpow(M, 2*n - 1);
                            auto aln = aln_1 + aln1 + aln2;
                            int L = 2 * n;
                            auto arg = num2arg(aln, M, L);
                            auto wn  = wn_1 * R(n, arg, U, s, gm0, gm1, gm2, gm3, gm4);
                            auto m = m1 + m2 * M + i * M * M;
                            Cnbuf[m] = aln;
                            Wnbuf[m] = wn;
                        }
                    }
                }
                // root gathers Cnbuf and Wnbuf from all ranks in comm to Cnbufall and Wnbufall:  
                MPI_Gather(Cnbuf.data(),    M * M * block, MPI_UNSIGNED_LONG_LONG,
                           Cnbufall.data(), M * M * block, MPI_UNSIGNED_LONG_LONG, root, comm);
                MPI_Gather(Wnbuf.data(),    M * M * block, MPI_C_DOUBLE_COMPLEX,
                           Wnbufall.data(), M * M * block, MPI_C_DOUBLE_COMPLEX,   root, comm);
                // root stores Cnbufall and Wnbufall into Cn and Wn if wn >= theta:
                if (rank == root) {
                    for (int k = 0; k < M * M * block * nranks; ++k) {
                        auto aln = Cnbufall[k];
                        auto wn  = Wnbufall[k];
                        if (abs(wn) >= theta) {
                            Cn.push_back(aln);
                            Wn.push_back(wn);
                        }
                    }
                }
                i0 += block * nranks;
            //    // ========= process exclusive for comm END ==============
            } // comm block ends
        } // round loop ends
        // all ranks must push_back Cn and Wn or otherwise 
        // the non-existing C[n-1] and W[n-1] will cause runtime error!
        C.push_back(Cn);
        W.push_back(Wn);
        if (myid == root){
            std::cout << "size of C" << n << " = " << C[n].size()
                      << "  lap time = " << MPI_Wtime() - time0 << " sec" << std::endl;
        }
    } // n loop ends
}


void getD0_mpi(int N, 
           int myid, int nprocs, int root,
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
    int M     = s[0].size();
    int Dkmax = gm2[0].size();
    int n     = std::min(N, Dkmax + 1);
    int Dkf   = std::min(n, Dkmax); // upper limit of Dk in propagator element
    int L     = 2 * (Dkf + 1);      // length of arguments in propagator
    int size  = Cn_1.size();

    // root broadcasts size to all ranks:
    MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
    
    // root resets D:
    if (myid == root){
        D.resize(size);
        std::fill(D.begin(), D.end(), std::complex<double>(0, 0));
    }
    // --- buffers for Scatter / Gather ---
    std::vector<unsigned long long>   Cn_1buf, Cn_1bufall;
    std::vector<std::complex<double>> Dibuf,   Dibufall;

    // memory size in BYTE per data block based on the following bufferes:
    // Cn_1buf, Dibuf
    int byte_per_block =  16 + 16;
    
    int block_reg, block_fin;
    int nblocks_reg;
    int nranks_reg, nranks_sem, nranks_fin;
    int nrounds_reg, nrounds;
    MPI_Comm comm_reg, comm_sem, comm_fin;

    round_manage_cpu(myid, nprocs,
                     size, byte_per_block,
                     block_reg, block_fin, nblocks_reg,
                     nranks_reg, nranks_sem, nranks_fin,
                     nrounds_reg, nrounds,
                     comm_reg, comm_sem, comm_fin);

    // --- print information 
    if (myid == root){
        std::cout << "   * nrounds     = " << nrounds     << std::endl; 
        std::cout << "   * nrounds_reg = " << nrounds_reg << std::endl; 
        std::cout << "   * nranks_reg  = " << nranks_reg  << std::endl; 
        std::cout << "   * nranks_sem  = " << nranks_sem  << std::endl; 
        std::cout << "   * block_reg   = " << block_reg   << std::endl; 
        std::cout << "   * block_fin   = " << block_fin   << std::endl; 
    }
 
    // root extends the "bufall" to the largest size:
    if (myid == root){
        Cn_1bufall.resize(block_reg * nprocs);
        Dibufall.resize(block_reg   * nprocs);
    }
    int i0 = 0; // the start position of C[n-1] and W[n-1] for the round
    for (int round = 0; round < nrounds; ++round){
        MPI_Comm comm;
        int block;
        // set communicator and block size for the round:
        if (round < nrounds_reg) {
            comm  = comm_reg;
            block = block_reg;
        }
        else if (round == nrounds - 1 && nranks_fin != 0) {
            comm  = comm_fin;
            block = block_fin;
        }
        else {
            comm  = comm_sem;
            block = block_reg;
        }
        if (comm != MPI_COMM_NULL) {
            // ========= process exclusive for comm BEGIN =========
            int rank, nranks;
            // get info about comm:
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &nranks);
            
            // resize buffers:
            if (round == 0 || block != block_reg){
                Cn_1buf.resize(block);
                Dibuf.resize(block);
            }
            // reset Dibuf:
            std::fill(Dibuf.begin(), Dibuf.end(), std::complex<double> (0, 0));
            
            // root transferes data from Cn_1 and D to buffers:
            if (rank == root) {
                std::copy(Cn_1.begin() + i0, Cn_1.begin() + i0 + block * nranks, Cn_1bufall.begin());
            }
            // root scatters Cn_1bufall and Wn_1bufall into all ranks in comm:
            MPI_Scatter(Cn_1bufall.data(), block, MPI_UNSIGNED_LONG_LONG,
                        Cn_1buf.data(),    block, MPI_UNSIGNED_LONG_LONG, root, comm);
            // each rank in comm generates Dibuff:
            for(int i = 0; i < block; i++){
                auto aln_1 = Cn_1buf[i];
                auto arg = num2arg(aln_1, M, L - 2);
                // currently, arg => \alpha_{n-1} = \{s_1^\pm, s_2\pm, \cdots, s_{n-1}^\pm \}
                arg.push_front(0);
                arg.push_front(0); 
                // now arg => \{s_0^\pm, \alpha_{n-1}\} with dummy values for s_0^\pm
                for(int m0 = 0; m0 < M; m0++){
                    for(int m1 = 0; m1 < M; m1++){
                        arg[0] = m0;
                        arg[1] = m1;
                        auto p0 = P(0, n, M, Dkmax, arg, U, s, gm0, gm1, gm2, gm3, gm4);
                        Dibuf[i] += p0 * rhos0[m0 * M + m1];
                    }
                }
            }
            // root gathers Dibuff from all ranks in comm to Dibuff:
            MPI_Gather(Dibuf.data(),    block, MPI_C_DOUBLE_COMPLEX,
                       Dibufall.data(), block, MPI_C_DOUBLE_COMPLEX, root, comm);
            // root stores Dibuff into D: 
            if (rank == root) {
                // Note: Dibufall is NOT resized to block * nranks
                std::copy(Dibufall.begin(), Dibufall.begin() + block * nranks, D.begin() + i0);
            }
            i0 += block * nranks;
            // ========= process exclusive for comm END ==============
        } // comm block ends
    } // round loop ends
}



void getD1_mpi(int N,
           int myid, int nprocs, int root,
           std::vector<unsigned long long>& Cn_1,
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
     *  MPI version. Uses collective calls only.
     *****************************************************/
    int M     = s[0].size();
    int Dkmax = gm2[0].size();
    int n     = std::min(N, Dkmax + 1);
    int L     = 2 * n;
    int size  = Cn_1.size();

    // root broadcasts size to all ranks:
    MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
    
    std::vector<std::complex<double>> Dprev(size);
    std::complex<double> Dave;
    if (myid == root){
        // root computes the average of D:
        std::complex<double> zsize(D.size(), 0);
        auto sum = std::accumulate(D.begin(), D.end(), std::complex<double>(0, 0));
        Dave = sum / zsize;
        // root copies D from the previous time step (N-1):
        Dprev = D;
        // root resets D:
        D.resize(size);
        std::fill(D.begin(), D.end(), std::complex<double>(0, 0));
    }

    // root broadcasts Dprev and Dave to all ranks:
    MPI_Bcast(Dprev.data(), size, MPI_C_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);
    MPI_Bcast(&Dave,  1,    MPI_C_DOUBLE_COMPLEX, root, MPI_COMM_WORLD);

    // --- buffers for Scatter / Gather ---
    std::vector<unsigned long long>   Cn_1buf, Cn_1bufall;
    std::vector<std::complex<double>> Dibuf,   Dibufall;

    // memory size in BYTE per data block based on the following bufferes:
    // Cn_1buf, Dibuf
    int byte_per_block =  16 + 16;
    
    int block_reg, block_fin;
    int nblocks_reg;
    int nranks_reg, nranks_sem, nranks_fin;
    int nrounds_reg, nrounds;
    MPI_Comm comm_reg, comm_sem, comm_fin;

    round_manage_cpu(myid, nprocs,
                     size, byte_per_block,
                     block_reg, block_fin, nblocks_reg,
                     nranks_reg, nranks_sem, nranks_fin,
                     nrounds_reg, nrounds,
                     comm_reg, comm_sem, comm_fin);

    // --- print information 
    if (myid == root && N == n + 1){
        std::cout << "   * nrounds     = " << nrounds     << std::endl; 
        std::cout << "   * nrounds_reg = " << nrounds_reg << std::endl; 
        std::cout << "   * nranks_reg  = " << nranks_reg  << std::endl; 
        std::cout << "   * nranks_sem  = " << nranks_sem  << std::endl; 
        std::cout << "   * block_reg   = " << block_reg   << std::endl; 
        std::cout << "   * block_fin   = " << block_fin   << std::endl; 
    }
    // root extends the "bufall" to the largest size:
    if (myid == root){
        Cn_1bufall.resize(block_reg * nprocs);
        Dibufall.resize(block_reg   * nprocs);
    }
    int i0 = 0; // the start position of Cn_1 for the round
    for (int round = 0; round < nrounds; ++round){
        MPI_Comm comm;
        int block;
        // set communicator and block size for the round:
        if (round < nrounds_reg) {
            comm  = comm_reg;
            block = block_reg;
        }
        else if (round == nrounds - 1 && nranks_fin != 0) {
            comm  = comm_fin;
            block = block_fin;
        }
        else {
            comm  = comm_sem;
            block = block_reg;
        }
        if (comm != MPI_COMM_NULL) {
            // ========= process exclusive for comm BEGIN =========
            int rank, nranks;
            // get info about comm:
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &nranks);
            
            // resize buffers:
            if (round == 0 || block != block_reg){
                Cn_1buf.resize(block);
                Dibuf.resize(block);
            }
            // reset Dibuf:
            std::fill(Dibuf.begin(), Dibuf.end(), std::complex<double> (0, 0));
             
            // root transferes data from Cn_1 and D to buffers:
            if (rank == root) {
                std::copy(Cn_1.begin() + i0, Cn_1.begin() + i0 + block * nranks, Cn_1bufall.begin());
            }
            // root scatters Cn_1bufall and Wn_1bufall into all ranks in comm:
            MPI_Scatter(Cn_1bufall.data(), block, MPI_UNSIGNED_LONG_LONG,
                        Cn_1buf.data(),    block, MPI_UNSIGNED_LONG_LONG, root, comm);
            // each rank in comm generates Dibuf:
            for(int i = 0; i < block; i++){
                auto aln_1 = Cn_1buf[i];
                auto arg = num2arg(aln_1, M, L - 2);  
                // currently, arg = \{s_1^\pm, s_2\pm, \cdots, s_{Dkmax}^\pm \}
                arg.push_front(0);
                arg.push_front(0);     
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
                        Dibuf[i] += P(1, n, M, Dkmax, arg, U, s, gm0, gm1, gm2, gm3, gm4) * Dval;
                    }
                }
            }
            // root gathers Dibuff from all ranks in comm to Dibuf:
            MPI_Gather(Dibuf.data(),    block, MPI_C_DOUBLE_COMPLEX,
                       Dibufall.data(), block, MPI_C_DOUBLE_COMPLEX, root, comm);
            // root stores Dibuff into D: 
            if (rank == root) {
                // Note: Dibufall is NOT resized to block * nranks
                std::copy(Dibufall.begin(), Dibufall.begin() + block * nranks, D.begin() + i0);
            }
            i0 += block * nranks;
            // ========= process exclusive for comm END ==============
        } // comm block ends
    } // round loop ends
}



void getrhos_mpi(int N,
                 int myid, int nprocs, int root,
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
     *  MPI version of getrhos.
     * *******************************************************/
    int Dkmax = gm2[0].size();
    int M     = s[0].size();
    int size  = Cn.size();
    int n     = std::min(N, Dkmax + 1);

    // root broadcasts size to all ranks:
    MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);

    // --- buffers for Scatter / Gather ---
    std::vector<unsigned long long>   Cnbuf, Cnbufall;
    std::vector<std::complex<double>> Wnbuf, Wnbufall;
    std::vector<std::complex<double>> Dbuf,  Dbufall;
    std::vector<std::complex<double>> rhosbuf(M * M);

    // memory size in BYTE per data block based on the following bufferes:
    // Cnbuf, Wnbuf, and Dbuf
    int byte_per_block =  16 + 16 + 16;
    
    int block_reg, block_fin;
    int nblocks_reg;
    int nranks_reg, nranks_sem, nranks_fin;
    int nrounds_reg, nrounds;
    MPI_Comm comm_reg, comm_sem, comm_fin;

    round_manage_cpu(myid, nprocs,
                     size, byte_per_block,
                     block_reg, block_fin, nblocks_reg, 
                     nranks_reg, nranks_sem, nranks_fin,
                     nrounds_reg, nrounds,
                     comm_reg, comm_sem, comm_fin);
    // root extends the "bufall" to the largest size:
    if (myid == root){
        Cnbufall.resize(block_reg * nprocs);
        Wnbufall.resize(block_reg * nprocs);
        Dbufall.resize(block_reg  * nprocs);
    }
    // root resets rhos: (uncecessary? just in case)
    //if (myid == root){
    //    std::fill(rhos.begin(), rhos.end(), std::complex<double>(0, 0));
    //}
    // every rank resets rhosbuf:
    std::fill(rhosbuf.begin(), rhosbuf.end(), std::complex<double>(0, 0));

    int i0 = 0; // the start position of C[n] or C[n-1] 
    for (int round = 0; round < nrounds; ++round){
        MPI_Comm comm;
        int block;
        // set communicator and block size for the round:
        if (round < nrounds_reg) {
            comm  = comm_reg;
            block = block_reg;
        }
        else if (round == nrounds - 1 && nranks_fin != 0) {
            comm  = comm_fin;
            block = block_fin;
        }
        else {
            comm  = comm_sem;
            block = block_reg;
        }
        if (comm != MPI_COMM_NULL) {
            // ========= process exclusive for comm BEGIN =========
            int rank, nranks;
            // get info about comm:
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &nranks);

            // resize buffers:
            if (round == 0 || block != block_reg){    
                Cnbuf.resize(block);
                Wnbuf.resize(block);
                Dbuf.resize(block);
            }
            // root transferes data from Cn, Wn and D to buffers of each rank:
            if (rank == root) {
                std::copy(Cn.begin() + i0, Cn.begin() + i0 + block * nranks, Cnbufall.begin());
                std::copy(Wn.begin() + i0, Wn.begin() + i0 + block * nranks, Wnbufall.begin());
                std::copy(D.begin()  + i0, D.begin()  + i0 + block * nranks, Dbufall.begin());
            }
            // root scatters Cnbufall, Wnbufall, and Dbufall to all ranks in comm:
            MPI_Scatter(Cnbufall.data(), block, MPI_UNSIGNED_LONG_LONG,
                        Cnbuf.data(),    block, MPI_UNSIGNED_LONG_LONG, root, comm);
            MPI_Scatter(Wnbufall.data(), block, MPI_C_DOUBLE_COMPLEX,
                        Wnbuf.data(),    block, MPI_C_DOUBLE_COMPLEX,   root, comm);
            MPI_Scatter(Dbufall.data(),  block, MPI_C_DOUBLE_COMPLEX,
                        Dbuf.data(),     block, MPI_C_DOUBLE_COMPLEX,   root, comm);

            // each rank computes rhosbuf:
            if (0 < N && N < Dkmax + 1){
                auto L = 2 * n;
                for (int i = i0; i < block; i++){
                    auto aln = Cnbuf[i];   // \alpha_n \in C_n
                    auto arg = num2arg(aln, M, L);
                    auto m1 = arg[L - 2];
                    auto m2 = arg[L - 1];
                    auto m = m1 * M + m2;
                    auto rho  = Wnbuf[i] * Dbuf[i];
                    rho *= I(0, n, n, m1, m2, m1, m2, s, gm0, gm1, gm2, gm3, gm4);
                    rhosbuf[m] += rho;
                }
            }
            else if (N >= Dkmax + 1){
                auto L = 2 * (n - 1);
                for (int i = i0; i < block; i++){
                    auto aln = Cnbuf[i];   // \alpha_{n-1} \in C_{n-1}
                    for (int m1 = 0; m1 < M; ++m1){
                        for (int m2 = 0; m2 < M; ++m2){
                            auto arg = num2arg(aln, M, L);
                            auto m = m1 * M + m2;
                            arg.push_back(m1); // m1 = s_n^+
                            arg.push_back(m2); // m2 = s_n^-
                            auto rho = Wnbuf[i] * Dbuf[i];
                            rho *= R(n, arg, U, s, gm0, gm1, gm2, gm3, gm4);
                            rho *= I(0, n, n, m1, m2, m1, m2, s, gm0, gm1, gm2, gm3, gm4);
                            rhosbuf[m] += rho;
                        }
                    }
                }
            }
            i0 += block * nranks;
            // ========= process exclusive for comm END =========
        } // comm block ends
    } // round loop ends 
    // finally, root gets all rhosbuf reduced to rhos of root:
    MPI_Reduce(rhosbuf.data(), rhos.data(), M * M, MPI_C_DOUBLE_COMPLEX, MPI_SUM, root, MPI_COMM_WORLD);
}

//=======================  EOF  ================================================
