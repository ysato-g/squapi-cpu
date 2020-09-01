/***********************************************************************************
 * Project: S-QuAPI for CPU
 * Major version: 0
 * version: 0.0.0 (serial)
 * Date Created : 8/23/20
 * Date Last mod: 8/30/20
 * Author: Yoshihiro Sato
 * Description: Inline functions in sqmodule.cpp, sqmodule_xxx.cpp 
 * Notes:
 *      - All Eq.(x) are refering to the corresponding equation numbers in 
 *        Y.Sato, Journal of Chemical Physics 150 (2019) 224108.
 *      - Based on C++11
 *      - Develped using MacOS 10.14 and Xcode 11.3
 *      - size of Cn has to be lower than 2,147,483,647 (limit of int)
 * Copyright (C) 2020 Yoshihiro Sato - All Rights Reserved
 **********************************************************************************/
//#include <complex>
//#include <vector>
//#include <deque>
//#include <algorithm>
//#include <numeric>

// ******* Power function for unsigned long long *************
inline unsigned long long ullpow(int M, int p)
{
    unsigned long long num = 1, m = M;
    for(int l = 1; l <= p; l++)
        num *= m;
    return num;
} 


inline unsigned long long arg2num(std::deque<int>& arg, int M)
{
    unsigned long long num = 0;
    for(int l = 0; l < arg.size(); l++)
        num += (unsigned long long)arg[l] * ullpow(M, l);
    return num;
}


inline std::deque<int> num2arg(unsigned long long num, int M, int L)   
{
    std::deque<int> arg(L);
    unsigned long long m = (unsigned long long)M;
    for(int l = 0; l < L; l++){
        arg[l] = (int)(num % m);
        num /= m;
    }
    return arg;
}

// ***** Influence Functional Element ***********
inline std::complex<double> I(int Dk, int k, int n, int arg0, int arg1, int arg2, int arg3,
                              std::vector<std::vector<std::complex<double>>>& s, 
                              std::vector<std::complex<double>>& gm0,
                              std::vector<std::complex<double>>& gm1,
                              std::vector<std::vector<std::complex<double>>>& gm2,
                              std::vector<std::vector<std::complex<double>>>& gm3,
                              std::vector<std::vector<std::complex<double>>>& gm4)
{   
    /*******************************************************************************
     * This function computes the functional of Eq.(26), that is
     *     $I_{\Delta k}^{(n)}(s_k^\pm, s_{k+\Delta k}^\pm)$ 
     * The variables are corresponding as follows:
     * $\Delta k$  = Dk   
     * $k$         = k    
     * $n$         = n    
     * $s_k^+$     = s[bath][arg0] 
     * $s_k^-$     = s[bath][arg1] 
     * $s_{k+1}^+$ = s[bath][arg2] 
     * $s_{k+1}^-$ = s[bath][arg3] 
     * and
     * $\gamma_{\Delta k, k}^{(n)}$ = gm depending on cases as given in Eq.(9).
     * as coded below.
     *******************************************************************************/ 
    auto nbath = s.size(); // the number of baths
    auto Dkmax = gm2[0].size();
    std::complex<double> phi(0, 0);
    for (int bath = 0; bath < nbath; bath++){
        std::complex<double> gm, gmc;
        if (Dk == 0 && (k == 0 || k == n)){
            gm = gm0[bath]; // Eq.(9a)
        }
        else if (Dk == 0 && 0 < k < n - Dk){
            gm = gm1[bath]; // Eq.(9b)
        }
        else if (0 < Dk && Dk < Dkmax + 1 && (k == 0 || k == n - Dk)){
            gm = gm2[bath][Dk - 1]; // Eq.(9c)
        }
        else if (0 < Dk && Dk < Dkmax + 1 && 0 < k && k < n - Dk){
            gm = gm3[bath][Dk - 1]; // Eq.(9d)
        }
        else if (0 < Dk && Dk == n && k == 0){
            gm = gm4[bath][Dk - 1]; // Eq.(9e)
        }
        else{
            // --- This case should never happen but added for safety
            gm.real(0);
            gm.imag(0);
        }
        gmc = std::conj(gm);
        phi += (s[bath][arg2] - s[bath][arg3]) * (gm * s[bath][arg0] - gmc * s[bath][arg1]);
    }
    return std::exp(-phi);
}




// ****** Weight Renewing Module, Eq.(13a)  ********
inline std::complex<double> R(int n, 
                              std::deque<int>& arg,
                              std::vector<std::complex<double>>& U, 
                              std::vector<std::vector<std::complex<double>>>& s, 
                              std::vector<std::complex<double>>& gm0,
                              std::vector<std::complex<double>>& gm1,
                              std::vector<std::vector<std::complex<double>>>& gm2,
                              std::vector<std::vector<std::complex<double>>>& gm3,
                              std::vector<std::vector<std::complex<double>>>& gm4)
{
    /*******************************************************************************
     * Function that computes $R$ of Eq.(13b).
     *******************************************************************************/ 
    
    auto M = s[0].size();
    auto L = 2 * n;
    auto arg0 = arg[L - 4]; // $s_{n-1}^+$
    auto arg1 = arg[L - 3]; // $s_{n-1}^-$
    auto arg2 = arg[L - 2]; // $s_n^+$
    auto arg3 = arg[L - 1]; // $s_n^-$
    // below the bare propagator, Eq.(5), K(s0, s1, s2, s3) = U(s2, s0) * U(s3, s1)
    auto r = U[arg2 * M + arg0] * std::conj(U[arg3 * M + arg1]);
    r *= I(0, n - 1, n, arg0, arg1, arg0, arg1, s, gm0, gm1, gm2, gm3, gm4);
    for(int k = 1; k < n - 1; k++){
        auto Dk = n - 1 - k;
        arg0 = arg[2 * (k - 1)];     // $s_k^+$
        arg1 = arg[2 * (k - 1) + 1]; // $s_k^-$
        arg2 = arg[L - 4];           // $s_{n-1}^+$
        arg3 = arg[L - 3];           // $s_{n-1}^-$
        r *= I(Dk, k, n,     arg0, arg1, arg2, arg3, s, gm0, gm1, gm2, gm3, gm4);
        r /= I(Dk, k, n - 1, arg0, arg1, arg2, arg3, s, gm0, gm1, gm2, gm3, gm4);
    }
    for(int k = 1; k < n; k++){
        auto Dk = n - k;
        arg0 = arg[2 * (k - 1)];     // $s_k^+$
        arg1 = arg[2 * (k - 1) + 1]; // $s_k^-$
        arg2 = arg[L - 2];           // $s_n^+$
        arg3 = arg[L - 1];           // $s_n^-$
        r *= I(Dk, k, n, arg0, arg1, arg2, arg3, s, gm0, gm1, gm2, gm3, gm4);
    }
    return r;
}


// ******* The Propagator of Eq.(13b) *************
inline std::complex<double> P(int l, int n, int M, int Dkmax,
                              std::deque<int>& arg,
                              std::vector<std::complex<double>>& U,
                              std::vector<std::vector<std::complex<double>>>& s,
                              std::vector<std::complex<double>>& gm0,
                              std::vector<std::complex<double>>& gm1,
                              std::vector<std::vector<std::complex<double>>>& gm2,
                              std::vector<std::vector<std::complex<double>>>& gm3,
                              std::vector<std::vector<std::complex<double>>>& gm4)
{ 
    /*******************************************************************************
     * Function that computes $P_\ell^{(n)}$ of Eq.(13b).
     * IMPORTANT NOTE:
     * Although not explicitely described in the paper (my bad), the RICs used in 
     * $F_\ell^{(n+\ell)}$ of Eq.(13b) are understood to be Eq.(9a), (9c), and (9e) 
     * for $\ell = 0$ and Eq.(9b) and (9d) for $\ell > 0$, respectively, for a given
     * $\Delta k$ value. This is needed to compensate the relabeling the starting 
     * varialble from $k$ in Eq.(11) to $0$ in Eq.(13b).
     *******************************************************************************/ 
    int Dkf = std::min(n, Dkmax); // $\Delta(n, 0)$
    // below sets the bare propagator, Eq.(5), K(s0, s1, s2, s3) = U(s2, s0) * U(s3, s1)
    auto K = U[arg[2] * M + arg[0]] * std::conj(U[arg[3] * M + arg[1]]);
    // below computes the bath influence
    auto nbath = s.size();
    std::complex<double> phi(0, 0);  // influence phase
    for (int Dk = 0; Dk < Dkf + 1; ++Dk){
        auto arg0 = arg[0];          //  $s_0^+$
        auto arg1 = arg[1];          //  $s_0^-$
        auto arg2 = arg[2 * Dk];     //  $s_{\Delta k}^+$
        auto arg3 = arg[2 * Dk + 1]; //  $s_{\Delta k}^-$
        for (int bath = 0; bath < nbath; ++bath){
            std::complex<double> gm;
            if (Dk == 0 && l == 0){
                gm = gm0[bath]; // Eq.(9a)
            } 
            else if (Dk == 0 && l > 0){
                gm = gm1[bath]; // Eq.(9b)
            }
            else if (0 < Dk && Dk < n && l == 0){
                gm = gm2[bath][Dk - 1]; // Eq.(9c)
            }
            else if (0 < Dk && Dk < n && l > 0){
                gm = gm3[bath][Dk - 1]; // Eq.(9d)
            }
            else if (0 < Dk && Dk == n && l == 0){
                gm = gm4[bath][Dk - 1]; // Eq.(9e) 
            }
            else{
                // --- This case should never happen but added for safety
                gm.real(0);
                gm.imag(0);
            }
            auto gmc = std::conj(gm);  // complex conj of the selected RIC
            phi += (s[bath][arg2] - s[bath][arg3]) * (gm * s[bath][arg0] - gmc * s[bath][arg1]);
        }
    }
    return K * std::exp(-phi);
}


//=======================  EOF  ================================================
