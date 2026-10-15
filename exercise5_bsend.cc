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

    int x[10], y[10];

    if (rank == 1) {
        // Populate array x without modifying/replacing the variable used to send
        for (int r = 0; r < 10; r++) {
            x[r] = 10 * r;
        }

        // Allocate and attach user-managed buffer for MPI_Bsend
        int buffer_size = MPI_BSEND_OVERHEAD + 10 * sizeof(int);
        char *buffer = (char *)malloc(buffer_size);
        if (buffer == NULL) {
            fprintf(stderr, "Rank 1: Failed to allocate buffer for Bsend\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Buffer_attach(buffer, buffer_size);

        printf("Rank 1: Sending message to rank 3 using MPI_Bsend (Buffered Send)\n");
        // Using MPI_Bsend with original variable x intact
        MPI_Bsend(x, 10, MPI_INT, 3, 0, MPI_COMM_WORLD);

        // Detach buffer (ensures message has been safely copied into network subsystem)
        void *detached_ptr;
        int detached_size;
        MPI_Buffer_detach(&detached_ptr, &detached_size);
        free(detached_ptr);
        printf("Rank 1: Buffer detached and memory freed successfully\n");
    }
    else if (rank == 3) {
        MPI_Recv(y, 10, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
        printf("Rank 3: Received values from rank 1:\n");
        for (int r = 0; r < 10; r++) {
            printf("%d ", y[r]);
        }
        printf("\n");
    }
    else {
        printf("Just a normal process From rank %d machine %s\n", rank, name);
    }

    MPI_Finalize();
    return 0;
}
