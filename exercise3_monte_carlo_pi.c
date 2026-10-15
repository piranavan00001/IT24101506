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

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Synchronize processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Divide iterations evenly among ranks
    long long base_count = TOTAL_ITERATIONS / size;
    long long remainder = TOTAL_ITERATIONS % size;
    local_iterations = (rank < remainder) ? (base_count + 1) : base_count;

    // Seed the PRNG differently for each rank to ensure independent streams
    unsigned int seed = (unsigned int)(time(NULL) + rank * 104729);

    for (long long i = 0; i < local_iterations; i++) {
        double x = (double)rand_r(&seed) / (double)RAND_MAX;
        double y = (double)rand_r(&seed) / (double)RAND_MAX;
        double z = x * x + y * y;
        if (z <= 1.0) {
            local_count++;
        }
    }

    // Reduce all local counts to total_count on rank 0
    MPI_Reduce(&local_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    if (rank == 0) {
        double pi_estimate = 4.0 * (double)total_count / (double)TOTAL_ITERATIONS;
        double error = fabs(pi_estimate - M_PI);

        printf("===================================================\n");
        printf("Exercise 3: Monte Carlo Pi Estimation\n");
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
