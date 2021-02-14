/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.0 (serial)
 * Date Created : 10/10/20
 * Date Last mod: 10/10/20
 * Author: Yoshihiro Sato
 * Description: Optional functions used in squapi_xxx.cpp 
 * Notes:
 *      - Based on C++11 (for merge of unordered_map)
 *      - Develped using gcc 10.2.0 on MacOS 10.14 
 *      - takes options "--save" or "-s" and "--cont" or "-c" 
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved
 **********************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <string>

// ************************************* Functions  *****************************************
void save_D (int N, double theta, std::vector<std::complex<double>>& D, std::string filename) 
{
    /*************************************************
     * store D to a file, D.dat, for squapi_cont_xxx
     *************************************************/
    std::ofstream fout;
    fout.open(filename); // filename is "D.dat"
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


void opt_save_D (int argc, char* argv[], 
                 int& N, double& theta,
                 std::vector<std::complex<double>>& D,
                 std::string filename_D)
{
    // searches if argv contains "--save_D or -d"
    for (auto i = 0; i < argc; ++i){
        std::string arg = argv[i]; 
        if (arg == "--save" || arg == "-s"){
            std::cout << "----- saving D to D.dat ----------------" << std::endl;
            if (D.size() > 1000000){
                std::cout << " This may take a while due to large D   " << std::endl;
            }
            save_D(N, theta, D, filename_D); // a good filename_D is D.dat
        }
    } 
}


void load_D (std::string filename, int& N0, double& theta, std::vector<std::complex<double>>& D)
{
    /*************************************************
     * load D, N0, and theta from D.dat
     *************************************************/
    std::ifstream fin(filename); // filename is D.dat
    // read D.dat and store the data into N0, size, and D:
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


void checkdata (std::string filename, int N0, int Nmax, int Dkmax)
{
    /*************************************************
     * read rhos.dat and check consistency and 
     * compatibility in the squapi parameters 
     *************************************************/
    std::ifstream fin(filename); // filename = rhos.dat
    if(!fin){
        std::cout << "---------------------------"    << std::endl;
        std::cout << "ERROR: Cannot open " + filename << std::endl;
        std::cout << "---------------------------"    << std::endl;
        exit(1); 
    }
    // scan the entire file and crop out the last line of it:
    std::string lastline;
    //fin.seekg(0, std::ios::beg); // go to the beginning of the file (just in case)
    while (true){
        std::string buf;
        std::getline(fin, buf);
        if (fin.eof()) break;
        lastline = buf;
    }
    fin.close();
    // extract N0 from the last line:
    int delim = lastline.find(",");               // delimiter before the end of info
    int N = std::stoi(lastline.substr(0, delim)); // extract string from the beginning to ','
    if (N != N0){
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "**** ERROR: INCONSITENT N0 VALUES ******" << std::endl;
        std::cout << "  N0 in D.dat    = " << N0 << std::endl;
        std::cout << "  N0 in rhos.dat = " << N  << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        exit(1); 
    }
    if (N0 >= Nmax){
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "******* Nmax <= N0: nothing to do ******" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        exit(1); 
    }
    if (Dkmax == 0){
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "****** DOES NOT SUPPORT Dkmax = 0 ******" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        exit(1);
    }
}


void opt_load_D (int argc, char* argv[], 
                 int& Nmax, int& Dkmax, double& theta,
                 std::string filename_D, std::string filename_rhos,
                 int& N0, std::vector<std::complex<double>>& D)
{
    // searches if argv contains "--cont"
    for (auto i = 0; i < argc; ++i){
        std::string arg = argv[i]; 
        if (arg == "--cont" || arg == "-c"){
            // overwrite N0, theta, and D by those in D.dat
            load_D(filename_D, N0, theta, D);          // filename_D is D.dat
            checkdata(filename_rhos, N0, Nmax, Dkmax); // filename_rhos is rhos.dat
        }       
    } 
}

//=======================  EOF  ================================================
