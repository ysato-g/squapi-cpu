/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.0.0 (x.0.x serial)
 * Date Created : 8/23/20
 * Date Last mod: 8/26/20
 * Author: Yoshihiro Sato
 * Description: Functions used in squapi_xxx.cpp and squapi_xxx.cpp 
 * Notes:
 *      - Based on C++11 (for merge of unordered_map)
 *      - Develped using gcc 10.2.0 on MacOS 10.14 
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved
 **********************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <string>

// ************************************* Functions  *****************************************
void save_D (int N, double theta, std::vector<std::complex<double>>& D) 
{
    /*************************************************
     * store D to a file, D.dat, for regenrhos
     *************************************************/
    std::ofstream fout;
    fout.open("D.dat");
    int size = (int) D.size();
    // record N and size first:
    fout << N     << std::endl;
    fout << theta << std::endl;
    fout << size  << std::endl;
    for (int i = 0; i < size; ++i) {
        auto x = D[i].real();
        auto y = D[i].imag();
        fout << std::scientific << std::setprecision(16);
        fout << std::scientific << x << std::endl;
        fout << std::scientific << y << std::endl;
        fout << std::noshowpos;
    }
    fout.close(); 
}

// ******** functions for regenrhos only ***************
// --- load D, N0, and theta from D.dat
void load_D (int& N0, double& theta, std::vector<std::complex<double>>& D)
{
    std::ifstream fin("D.dat");  // open the data file
    // --- read D.dat and store the data into N0, size, and D ---
    if(!fin){
        std::cout << "---------------------------"    << std::endl;
        std::cout << "ERROR: Cannot open D.dat   "    << std::endl;
        std::cout << "---------------------------"    << std::endl;
        exit(1);
    }
    int size;
    // get data from D.dat:
    fin >> N0;
    fin >> theta;
    fin >> size;
    D.resize(size);
    for(int i = 0; i < size; ++i){
        double x, y;
        std::complex<double> z;
        fin >> x;
        fin >> y;
        z.real(x);
        z.imag(y);
        D[i] = z;
    }
    fin.close();
}

bool checkN0 (std::string filename, int N0)
{
    // ***** read rhos.dat and check N0 ******
    std::ifstream fin(filename);  // open the data file
    if(!fin){
        std::cout << "---------------------------"    << std::endl;
        std::cout << "ERROR: Cannot open " + filename << std::endl;
        std::cout << "---------------------------"    << std::endl;
        return false;
    }

    // --- scan the entire file and crop out the last line of it
    std::string lastline;
    //fin.seekg(0, std::ios::beg); // go to the beginning of the file (just in case)
    while (true){
        std::string buf;
        std::getline(fin, buf);
        if (fin.eof()) break;
        lastline = buf;
    }
    fin.close();
    // --- extract N0 from the last line
    int delim = lastline.find(",");            // delimiter before the end of info
    int N = stod( lastline.substr(0, delim) ); // extract string from the beginning to ','
    if (N == N0){
        std::cout << "----- check consistency in N0 --------" << std::endl;
        std::cout << "  N0 in D.dat    = " << N0 << std::endl;
        std::cout << "  N0 in rhos.dat = " << N  << std::endl;
        return true;
    }
    else{
        std::cout << "**** FAILED! N0 does not match: ****" << std::endl;
        std::cout << "  N0 in D.dat    = " << N0 << std::endl;
        std::cout << "  N0 in rhos.dat = " << N  << std::endl;
        return false;
    }
}


//=======================  EOF  ================================================
