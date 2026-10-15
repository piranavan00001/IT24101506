#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10000000LL

int main(int argc, char *argv[]) {
    int rank, size;
    long long local_sum = 0;
    long long total_sum = 0;
    double start_time = 0.0, end_time = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Synchronize processes before starting time measurement
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Divide work evenly among processes
    long long base_count = N / size;
    long long remainder = N % size;
    long long local_count = (rank < remainder) ? (base_count + 1) : base_count;
    long long start = rank * base_count + (rank < remainder ? rank : remainder) + 1;
    long long end = start + local_count - 1;

    // Compute local sum
    for (long long i = start; i <= end; i++) {
        local_sum += i;
    }

    // Reduce all local sums to total_sum on rank 0
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Rank 0 outputs the results
    if (rank == 0) {
        long long expected_sum = (N * (N + 1LL)) / 2LL;
        printf("===================================================\n");
        printf("Exercise 2: Parallel Sum of 1 to %lld\n", N);
        printf("Number of Processes : %d\n", size);
        printf("Computed Total Sum  : %lld\n", total_sum);
        printf("Expected Total Sum  : %lld\n", expected_sum);
        printf("Verification        : %s\n", (total_sum == expected_sum) ? "PASSED (MATCH)" : "FAILED");
        printf("Execution Time      : %f seconds\n", end_time - start_time);
        printf("===================================================\n");
    }

    MPI_Finalize();
    return 0;
}
