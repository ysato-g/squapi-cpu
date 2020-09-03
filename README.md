# **Scalable QuAPI for CPU oriented platforms**

## What's Scalable QuAPI?
The *quasi-adiabatic propagator path integral* or QuAPI is a powerful numerical methodology to compute quantum dynamics of open quantum systems, originally developed by N. Makri and D. E. Makarov in 1994.
Scalable QuAPI (S-QuAPI) is one of the recently-developed flavors of QuAPI algorithm that is specifically targeted at scalability of parallel computation and small memory consumption.
The algorithm has been implemented on a GPU accelerated cluster and demonstraing excellent scalability as described in [Y. Sato, *Journal of Chemical Physics* **150** 224108 (2019)](http://aip.scitation.org/doi/10.1063/1.5100881).


**This repository develops squapi-cpu, an implementation of S-QuAPI algorithm for CPU oriented platforms.**


## System Requirement
**squpai-cpu** is a set of command-line tools (executables).
It requires C++ compiler to build and Python 3 to generate necessary input files.
The squapi-cpu executables and the accompanying Python script have been developed using **macOS** 10.14 (Mojave) and tested on **Linux** CenOS 7 and Ubuntu Desktop 20.
The essential requirements for all platforms are as follows:

    * Any C++ compiler supporting C++11 to build squapi, the serial version.  
    * GCC to build squapi_omp, the OpenMP version of squapi.  
    * MPI implementation to build and run squapi_mpi, the MPI version of squapi.   
    * Python 3 with numpy, scipy, and matplotlib packages.
For an easy and complete installation of Python 3, [Anaconda](https://www.anaconda.com/products/individual) is recommended.
The current version of squapi_mpi has been develped using [Open MPI](https://www.open-mpi.org/), but it should work with any other MPI implementations as well. 
Also, squapi_omp usually outperforms squapi_mpi on a single node, so MPI would not be necessary for a multi-core shared-memory platform. 

## Installation 
For **macOS**, GCC and Open MPI can readily be installed with [Homebrew](https://github.com/Homebrew).
After installing [Xcode](https://developer.apple.com/xcode/) and Homebrew, 

```
$ brew install gcc  
$ brew install open-mpi
```

For **Ubuntu**, GCC and Open MPI can be installed by 

```
$ sudo apt install gcc g++ make
$ sudo apt install openmpi-bin
```

The installed GCC is version 10.2 on my platform (macOS), so I represent the C++ compiler by ```g++-10``` in the following instructions. 
If your GCC is differnt from version 10, **edit ```CXX = g++-10``` in Makefile** to the right version (e.g., ```CXX = g++-9``` for version 9). 
For the complete installation, run the following in the root directory (where Makefile is present):

```
$ export OMPI_CXX=g++-10
$ make
```
This will compile and install all the squapi-cpu executables (squapi, squapi_omp, squapi_cont_omp, squapi_mpi, and squapi_cont_mpi) and the accompaying Python 3 sript (squapi.py) into the bin directory. 
You can clean up the root directory by

```
$ make clean
```

If just the OpenMP version is needed, then 

```
$ make omp
```

will compile and install the serial and OpenMP versions, and the MPI version will be skipped.
If only the serial version is needed, then just running

```
$ g++-10 -I include -std=c++11 src/sqmodule.cpp src/squapi.cpp -o bin/squapi
$ cp src/squapi.py bin 
```

will be enough.
Here ```g++-10``` needs to be changed, if not GCC ver. 10, to the command for your compiler.




## Usage

To test your squapi installation, try 

```
$ cd examples/benchmarks
$ python3 benchmark1.py
```

This will run squapi (serial) then generate a graph resembling Figure 2 of [P. Nalbach et al, *New Journal of Physics* **13** 063040 (2011)](https://iopscience.iop.org/article/10.1088/1367-2630/13/6/063040).
Then to test your squapi_omp installation, try 

```
$ python3 benchmark2.py
```

which makes a graph resembling Figure 3 of the same reference.
And for squapi_mpi, try

```
$ python3 benchmark3.py
```

which makes a graph resembling Figure 4 of the same reference.
For a detailed set-by-step instruction, see the JupyterLab notebook examples/how_to_use_squapi.ipynb:

```
$ cd ..
$ jupyter-lab how_to_use_squapi.ipynb 
```
