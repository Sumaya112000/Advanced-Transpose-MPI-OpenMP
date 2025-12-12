#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

void transpose_mpi(double* A, double* AT, int local_n, int global_n)
{
    // Post receives

    // Pack data and send
    
    // Unpack data
}
void transpose_mpiOpenMP(double* A, double* AT, int local_n, int global_n)
{
    // Post recieves - mpi

    // Pack data - OpenMP
    
    // Send data - mpi

    // Unpack data - OpenMP
}


int tutorial_main(int argc, char** argv)
{
    // 1. Initialize	
    
    // 2. Transpose and time your methods

    // 3. Finalize 
    return 0;
}
