// This include is required for tutorial to pass tests
#include "src.hpp"
#include <time.h>
#include <vector>

void transposeMPI(double* A, double* AT, int local_n, int global_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double* send_buffer = new double[local_n*global_n];
    double* recv_buffer = new double[local_n*global_n];
    
    MPI_Request* send_requests = new MPI_Request[num_procs];
    MPI_Request* recv_requests = new MPI_Request[num_procs];

    int tag = 1024;

    int msg_size = local_n*local_n; // size of each msg
                                    //
    for (int i = 0; i < num_procs; i++)
    {
        MPI_Irecv(&(recv_buffer[i*msg_size]), msg_size, MPI_DOUBLE, i, tag, MPI_COMM_WORLD,
                &(recv_requests[i]));
    }

    int ctr = 0;
    for (int i = 0; i < num_procs; i++)
    {
        for (int col = i*local_n; col < (i+1)*local_n; col++)
        {
            for (int row = 0; row < local_n; row++)
            {
                send_buffer[ctr++] = A[row*global_n+col];
            }
        }
        MPI_Isend(&(send_buffer[i*msg_size]), msg_size, MPI_DOUBLE, i, tag, MPI_COMM_WORLD,
                &(send_requests[i]));
    }
    
    MPI_Waitall(num_procs, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_procs, recv_requests, MPI_STATUSES_IGNORE);

    for (int row = 0; row < local_n; row++)
    {
        for (int i = 0; i < num_procs; i++)
        {
            for (int col = 0; col < local_n; col++)
            {
                AT[row*global_n + i*local_n + col] = recv_buffer[i*local_n*local_n + row*local_n + col];
            }
        }
    }

    delete[] send_buffer;
    delete[] recv_buffer;
    delete[] send_requests;
    delete[] recv_requests;
}

void transpose_mpiOpenMP(double* A, double* AT, int local_n, int global_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int msg_size = local_n * local_n;

    double* send_buffer = (double*) malloc(local_n * global_n * sizeof(double));
    double* recv_buffer = (double*) malloc(local_n * global_n * sizeof(double));

    MPI_Request* send_requests = (MPI_Request*) malloc(num_procs * sizeof(MPI_Request));
    MPI_Request* recv_requests = (MPI_Request*) malloc(num_procs * sizeof(MPI_Request));

    int tag = 1024;

    // Post receives
    for (int i = 0; i < num_procs; i++) {
        MPI_Irecv(&(recv_buffer[i * msg_size]), msg_size, MPI_DOUBLE, i, tag,
                  MPI_COMM_WORLD, &(recv_requests[i]));
    }

    // Pack using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < num_procs; i++) {
        for (int col = i * local_n; col < (i + 1) * local_n; col++) {
            for (int row = 0; row < local_n; row++) {
                int ctr = i * msg_size + (col - i * local_n) * local_n;
                send_buffer[ctr + row] = A[row * global_n + col];
            }
        }
    }

    // Send messages
    for (int i = 0; i < num_procs; i++) {
        MPI_Isend(&(send_buffer[i * msg_size]), msg_size, MPI_DOUBLE, i, tag,
                  MPI_COMM_WORLD, &(send_requests[i]));
    }

    MPI_Waitall(num_procs, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_procs, recv_requests, MPI_STATUSES_IGNORE);

    // Unpack using OpenMP
    #pragma omp parallel for
    for (int row = 0; row < local_n; row++) {
        for (int i = 0; i < num_procs; i++) {
            for (int col = 0; col < local_n; col++) {
                AT[row * global_n + i * local_n + col] =
                    recv_buffer[i * msg_size + row * local_n + col];
            }
        }
    }

    free(send_buffer);
    free(recv_buffer);
    free(send_requests);
    free(recv_requests);
}

// Initialize, create random, finalize, and return
int tutprial_main(int argc, char** argv, int global_n)
{
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2) {
        if (rank == 0) printf("Pass the matrix size as an argument\n");
        MPI_Finalize();
        return 0;
    }

    int N = atoi(argv[1]);

    if (N % num_procs != 0) {
        if (rank == 0) printf("N must be divisible by num_procs\n");
        MPI_Finalize();
        return 0;
    }

    int local_n = N / num_procs;

    double* A  = (double*) malloc(local_n * N * sizeof(double));
    double* AT = (double*) malloc(local_n * N * sizeof(double));

    // Fill matrix with rank-dependent values
    for (int i = 0; i < local_n; i++){
        for (int j = 0; j < N; j++){
            A[i * N + j] = (double)(rank * 1000 + i * N + j);
        }
    }
    
    // Ensure all processes are ready to start timing
    MPI_Barrier(MPI_COMM_WORLD);

    // 2. Transpose

        // Time MPI-only transpose
        double t1 = MPI_Wtime();
        transpose_MPI(A, AT, local_n, N);
        double t2 = MPI_Wtime();

        // Time MPI+OpenMP transpose
        double t3 = MPI_Wtime();
        transpose_mpiOpenMP(A, AT, local_n, N);
        double t4 = MPI_Wtime();

    if (rank == 0) {
        printf("N = %d, P = %d\n", N, num_procs);
        printf("MPI only:       %lf sec\n", t2 - t1);
        printf("MPI + OpenMP:   %lf sec\n", t4 - t3);
    }

    free(A);
    free(AT);

    // 3. Finalize MPI
    MPI_Finalize();
    return 0;
}