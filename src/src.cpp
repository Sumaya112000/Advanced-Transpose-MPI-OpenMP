#include "src.hpp"

bool mpi_initialized = false;
bool mpi_finalized = false;
int num_procs = 0;

int MPI_Init(int* argc, char*** argv)
{
    mpi_initialized = true;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    return MPI_SUCCESS;
}


int MPI_Finalize()
{
    mpi_finalized = true;
    return MPI_SUCCESS;
}

