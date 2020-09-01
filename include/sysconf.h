/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.1.0 (MPI and OpenMP)
 * Date Created : 8/23/20
 * Date Last mod: 8/26/20
 * Author: Yoshihiro Sato
 * Description: header file for sqmodule_mpi.cpp, squapi_mpi.cpp, and squapi_cont_mpi.cpp 
 * Notes:
 *      - RANK_OF_ROOT and MAX_RAM_PER_CORE_MIB can be changed depending on 
 *        hardware configuration of the system 
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved
 **********************************************************************************/
// rank of root in genrhos_mpi.cpp and regenrhos_mpi.cpp
#define RANK_OF_ROOT 0

// set up the maximum ram used per core in squapi_mpi.cpp
#define MAX_RAM_PER_CORE_MIB 1000

// definition of one mega information byte in byte: DO NOT TOUCH!
#define MIB 1024 * 1024

/****************************  EOF ************************************************/ 
