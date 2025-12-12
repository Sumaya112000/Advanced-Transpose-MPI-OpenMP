#include "src.hpp"

int main(int argc, char* argv[])
{
    PMPI_Init(&argc, &argv);
    
    tutorial_main(argc, argv);

    if (mpi_finalized == false)
    {
        fprintf(stderr, "MPI was not finalized appropriately.  Make sure to call MPI_Finalize.\n");
        return -1;
    }

    PMPI_Finalize();

    return 0;
}
