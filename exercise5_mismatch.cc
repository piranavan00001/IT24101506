#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(name, &len);

    int x = 30;
    int y = 0;

    // Source and Destination Mismatch Demonstration:
    // Rank 1 sends to Rank 2, but Rank 3 expects to receive from Rank 1 (or Rank 2 never receives).
    if (rank == 1) {
        printf("[Rank %d on %s] Attempting Ssend to Rank 2...\n", rank, name);
        fflush(stdout);
        // Destination is set to 2 instead of 3
        MPI_Ssend(&x, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
        printf("[Rank %d] Ssend completed successfully.\n", rank);
        fflush(stdout);
    }
    else if (rank == 3) {
        printf("[Rank %d on %s] Waiting in MPI_Recv for message from Rank 1...\n", rank, name);
        fflush(stdout);
        // Rank 3 expects message from Rank 1, but Rank 1 sent to Rank 2
        MPI_Recv(&y, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
        printf("[Rank %d] Successfully received y = %d\n", rank, y);
        fflush(stdout);
    }
    else {
        printf("[Rank %d on %s] Process idle / doing other work.\n", rank, name);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
