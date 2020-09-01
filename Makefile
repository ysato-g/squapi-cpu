#**********************************
# Project: S-QuAPI for CPU
# Makefile for squapi-cpu
# Date Created : 9/1/20
# Date Last mod: 9/1/20
# Author: Yoshihiro Sato
#**********************************

# define sub directories: 
SRC = src
BIN = bin
INC = include

# compilers and options:
CXX = g++-10 
MPICXX = mpicxx 
CXX_FLAGS = -I $(INC) -fopenmp -std=c++11 -O3

# programs to be made:
PROGS_OMP = squapi squapi_omp squapi_cont_omp
PROGS_MPI = $(PROGS_OMP) squapi_mpi squapi_cont_mpi
PROGS = $(PROGS_MPI) 

# object files involved:
OBJS_OMP = sqmodule.o sqmodule_omp.o cont.o  
OBJS_MPI = $(OBJS_OMP) sqmodule_mpi.o  
OBJS = $(OBJS_MPI)

# dependencies:
DEPS_OMP = $(SRC)/squapi_omp.cpp \
		   $(INC)/sqmodule.h $(INC)/sqmodule_omp.h $(INC)/cont.h \
		   $(OBJS_OMP)
DEPS_MPI = $(SRC)/squapi_mpi.cpp $(INC)/* $(OBJS_MPI) 

.PHONEY: all
all: $(OBJS) $(PROGS) install

.PHONEY: omp 
omp: $(OBJS_OMP) $(PROGS_OMP) install_omp

sqmodule.o: $(SRC)/sqmodule.cpp $(INC)/inlines.h $(INC)/phys.h
	$(CXX) $(CXX_FLAGS) -c $(SRC)/sqmodule.cpp -o sqmodule.o 

sqmodule_omp.o: $(SRC)/sqmodule_omp.cpp $(INC)/inlines.h
	$(CXX) $(CXX_FLAGS) -c $(SRC)/sqmodule_omp.cpp -o sqmodule_omp.o 

sqmodule_mpi.o: $(SRC)/sqmodule_mpi.cpp $(INC)/inlines.h $(INC)/sysconf.h
	$(MPICXX) $(CXX_FLAGS) -c $(SRC)/sqmodule_mpi.cpp -o sqmodule_mpi.o 

cont.o: $(SRC)/cont.cpp 
	$(CXX) $(CXX_FLAGS) -c $(SRC)/cont.cpp -o cont.o 

squapi: $(SRC)/squapi.cpp $(INC)/sqmodule.h sqmodule.o 
	$(CXX) $(CXX_FLAGS) sqmodule.o $(SRC)/squapi.cpp -o squapi 

squapi_omp: $(DEPS_OMP)
	$(CXX) $(CXX_FLAGS) $(OBJS_OMP) $(SRC)/squapi_omp.cpp -o squapi_omp  

squapi_cont_omp: $(DEPS_OMP)
	$(CXX) $(CXX_FLAGS) $(OBJS_OMP) $(SRC)/squapi_cont_omp.cpp -o squapi_cont_omp 

squapi_mpi: $(DEPS_MPI)
	$(MPICXX) $(CXX_FLAGS) $(OBJS_MPI) $(SRC)/squapi_mpi.cpp -o squapi_mpi 

squapi_cont_mpi: $(DEPS_MPI) 
	$(MPICXX) $(CXX_FLAGS) $(OBJS_MPI) $(SRC)/squapi_cont_mpi.cpp -o squapi_cont_mpi 

install: $(PROGS) $(SRC)/squapi.py 
	install -s $(PROGS) $(BIN)
	cp $(SRC)/squapi.py $(BIN)

install_omp: $(PROGS_OMP) $(SRC)/squapi.py 
	install -s $(PROGS_OMP) $(BIN)
	cp $(SRC)/squapi.py $(BIN)
	rm $(PROGS_OMP) $(OBJS_OMP)


.PHONEY: clean
clean:
	rm $(OBJS) $(PROGS) 

.PHONEY: uninstall
uninstall:
	mv $(BIN)/squapi* ~/.Trash
