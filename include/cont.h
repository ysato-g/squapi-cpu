/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.0.0 (serial)
 * Date Created : 8/23/20
 * Date Last mod: 8/26/20
 * Author: Yoshihiro Sato
 * Description: Header file of cont.cpp needed in squapi_xxx.cpp and squapi_cont_xxx.cpp 
 * Notes:
 *      - Based on C++11 (for merge of unordered_map)
 *      - Develped using gcc 10.2.0 on MacOS 10.14 
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved
 **********************************************************************************/

void save_D (int N, double theta, std::vector<std::complex<double>>& D); 

void load_D (int& N0, double& theta, std::vector<std::complex<double>>& D);

bool checkN0 (std::string filename, int N0);

//=======================  EOF  ================================================
