#include "src.hpp"

int main(int argc, char* argv[])
{
    PMPI_Init(&argc, &argv);

    int rank, num_procs;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    tutorial_main(argc, argv, N);

    PMPI_Finalize();
    return 0;
}
