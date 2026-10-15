#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define TOTAL_ITERATIONS 10000000LL

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(int argc, char *argv[]) {
    int rank, size;
    long long local_iterations = 0;
    long long local_count = 0;
    long long total_count = 0;
    double start_time = 0.0, end_time = 0.0;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Divide iterations evenly
    long long base_count = TOTAL_ITERATIONS / size;
    long long remainder = TOTAL_ITERATIONS % size;
    local_iterations = (rank < remainder) ? (base_count + 1) : base_count;

    // Seed PRNG uniquely per rank
    unsigned int seed = (unsigned int)(time(NULL) + rank * 104729);

    for (long long i = 0; i < local_iterations; i++) {
        double x = (double)rand_r(&seed) / (double)RAND_MAX;
        double y = (double)rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    if (rank != 0) {
        // Worker processes use Buffered Send (MPI_Bsend)
        int buffer_size = MPI_BSEND_OVERHEAD + sizeof(long long);
        char *buffer = (char *)malloc(buffer_size);
        if (buffer == NULL) {
            fprintf(stderr, "Rank %d: Failed to allocate buffer for Bsend\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Buffer_attach(buffer, buffer_size);

        // Perform buffered send to rank 0
        MPI_Bsend(&local_count, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);

        // Detach buffer (ensures message is passed to network/system buffer)
        void *detached_ptr;
        int detached_size;
        MPI_Buffer_detach(&detached_ptr, &detached_size);
        free(detached_ptr);
    } else {
        // Master process accounts for its own local count
        total_count = local_count;
        printf("Rank 0: Local count = %lld\n", local_count);

        // Receive counts using MPI_ANY_SOURCE
        for (int i = 1; i < size; i++) {
            long long worker_count = 0;
            MPI_Recv(&worker_count, 1, MPI_LONG_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            total_count += worker_count;
            printf("Rank 0: Received %lld points from Rank %d (via MPI_Bsend)\n",
                   worker_count, status.MPI_SOURCE);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        double pi_estimate = 4.0 * (double)total_count / (double)TOTAL_ITERATIONS;
        double error = fabs(pi_estimate - M_PI);

        printf("===================================================\n");
        printf("Exercise 7: Monte Carlo Pi using BSend and MPI_ANY_SOURCE\n");
        printf("Total Iterations    : %lld\n", TOTAL_ITERATIONS);
        printf("Number of Processes : %d\n", size);
        printf("Points Inside Circle: %lld\n", total_count);
        printf("Estimated Pi        : %.8f\n", pi_estimate);
        printf("Actual Pi (M_PI)    : %.8f\n", M_PI);
        printf("Absolute Error      : %.8f\n", error);
        printf("Execution Time      : %f seconds\n", end_time - start_time);
        printf("===================================================\n");
    }

    MPI_Finalize();
    return 0;
}
