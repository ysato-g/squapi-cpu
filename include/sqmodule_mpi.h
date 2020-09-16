/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.2.1 (MPI with some OpenMP)
 * Date Created : 8/23/20
 * Date Last mod: 9/2/20
 * Author: Yoshihiro Sato
 * Description: Header file for functions written in sqmodule_mpi.cpp 
 * Notes:    
 *      - Based on C++11
 *      - Develped using gcc 10.2.0 and Open MPI 4.0.4 
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved
 **********************************************************************************/
void manage_round_cpu(int myid, int nprocs,
                      int size, int byte_per_block,
                      int& block_reg, int& block_fin,
                      int& nblocks_reg,
                      int& nranks_reg, int& nranks_sem, int& nranks_fin,
                      int& nrounds_reg, int& nrounds,
                      MPI_Comm& comm_reg,
                      MPI_Comm& comm_sem,
                      MPI_Comm& comm_fin);

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
           std::vector<std::vector<std::complex<double>>>& W);

void getD0_mpi(int N, 
           int myid, int nprocs, int root,
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

void getD1_mpi(int N,
           int myid, int nprocs, int root,
           std::vector<unsigned long long>& Cn,
           std::unordered_map<unsigned long long, int>& Cnmap,
           std::vector<std::complex<double>>& U,
           std::vector<std::vector<std::complex<double>>>& s, 
           std::vector<std::complex<double>>& gm0,
           std::vector<std::complex<double>>& gm1,
           std::vector<std::vector<std::complex<double>>>& gm2,
           std::vector<std::vector<std::complex<double>>>& gm3,
           std::vector<std::vector<std::complex<double>>>& gm4,
           std::vector<std::complex<double>>& D);

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
                 std::vector<std::complex<double>>& rhos);

//=======================  EOF  ================================================
