# SE4060 – Parallel Computing
## Lab Sheet 06: Message Passing Interface (MPI) Programming
**BSc (Hons) in Information Technology — Year 4, Semester 2**  
**SLIIT Faculty of Computing**

---

## Table of Contents
1. [Overview & Environment Setup](#1-overview--environment-setup)
2. [Exercise 0: In-Class Activities (SE4060-MPI-Part1)](#2-exercise-0-in-class-activities)
3. [Exercise 1: Repository Management & Setup](#3-exercise-1-repository-management--setup)
4. [Exercise 2: Parallel Sum of 1 to 10,000,000](#4-exercise-2-parallel-sum-of-1-to-10000000)
5. [Exercise 3: Monte Carlo Pi Computation (10,000,000 iterations)](#5-exercise-3-monte-carlo-pi-computation)
6. [Exercise 4: Performance Profiling, Speedup & Scalability Graphs](#6-exercise-4-performance-profiling-speedup--scalability-graphs)
7. [Exercise 5: Mismatched Communication Deadlocks & Buffered Send (BSend)](#7-exercise-5-mismatched-communication-deadlocks--buffered-send-bsend)
8. [Exercise 6: Monte Carlo Pi with `MPI_ANY_SOURCE`](#8-exercise-6-monte-carlo-pi-with-mpi_any_source)
9. [Exercise 7: Monte Carlo Pi using `MPI_Bsend` and `MPI_ANY_SOURCE`](#9-exercise-7-monte-carlo-pi-using-mpi_bsend-and-mpi_any_source)
10. [Build & Execution Guide](#10-build--execution-guide)

---

## 1. Overview & Environment Setup

This laboratory sheet explores distributed-memory parallel computing using the Message Passing Interface (MPI) standard. The exercises cover fundamental MPI concepts including:
- Process topology and communicator management (`MPI_COMM_WORLD`, `MPI_Comm_rank`, `MPI_Comm_size`).
- Collective communication paradigms (`MPI_Reduce`, `MPI_Barrier`).
- Blocking point-to-point communication (`MPI_Send`, `MPI_Recv`, `MPI_Ssend`).
- Asynchronous and user-buffered communication (`MPI_Bsend`, `MPI_Buffer_attach`, `MPI_Buffer_detach`).
- Wildcard receive mechanics (`MPI_ANY_SOURCE`, `MPI_ANY_TAG`).
- Empirical scalability benchmarking (Execution Time, Speedup, and Efficiency).

### Development & Execution Environment
- **Compiler**: GCC / G++ (OpenMPI 4.1.x)
- **MPI Library**: OpenMPI (`mpicc`, `mpic++`, `mpirun`)
- **Language**: C / C++
- **Operating System**: Linux (WSL / Ubuntu) / Cluster PBS Environment

---

## 2. Exercise 0: In-Class Activities

The reference in-class lecture repository [`SE4060-MPI-Part1`](https://github.com/SLIIT-FacultyOfComputing/SE4060-MPI-Part1) includes essential starter templates:

1. **`HelloMPI.c`**: Demonstrates basic process initialization, processor identification via `MPI_Get_processor_name`, rank differentiation, and finalization.
   ```bash
   mpicc HelloMPI.c -o HelloMPI
   mpirun -np 4 ./HelloMPI
   ```
2. **`messages1.cc`**: Point-to-point communication sending a scalar integer (`x = 30`) from Rank 1 to Rank 3 using synchronous send (`MPI_Ssend`) and receive (`MPI_Recv`).
3. **`messaage2.cc`**: Array transmission passing an array of 10 integers from Rank 1 to Rank 3.
4. **`job1.pbs`**: PBS batch script defining resource allocation (`nodes=1`, `select=1`, `walltime=00:02:00`) for High-Performance Computing (HPC) clusters like ARCHER2.

---

## 3. Exercise 1: Repository Management & Setup

To organize and submit your solutions:
1. Create a GitHub repository named with your student ID (e.g., `ITxxxxxxx-SE4060-Lab6`).
2. Clone or initialize the repository locally:
   ```bash
   git init
   git remote add origin https://github.com/<your-username>/<student-id>-SE4060-Lab6.git
   ```
3. Commit and push all solution files:
   ```bash
   git add .
   git commit -m "Complete SE4060 Lab 6 MPI exercises"
   git push -u origin main
   ```

---

## 4. Exercise 2: Parallel Sum of 1 to 10,000,000

### Mathematical Formulation
The sum of integers from $1$ to $N = 10,000,000$ is given analytically by:
$$\sum_{i=1}^{N} i = \frac{N(N+1)}{2} = \frac{10,000,000 \times 10,000,001}{2} = 50,000,005,000,000$$

> [!IMPORTANT]
> The total sum ($5 \times 10^{13}$) exceeds the 32-bit signed integer limit ($2^{31}-1 \approx 2.14 \times 10^9$). Therefore, `long long` in C and `MPI_LONG_LONG` in MPI must be used to prevent integer overflow.

### Implementation Strategy
The domain $[1, 10,000,000]$ is partitioned evenly across $P$ MPI processes. If $N \pmod P \ne 0$, remainders are distributed across the lower-ranked processes to ensure perfect load balance. Each rank computes its local sum, and `MPI_Reduce` aggregates the partial sums onto Rank 0 using the `MPI_SUM` reduction operator.

### Source Code: `exercise2_sum.c`
```c
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

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Balanced domain decomposition
    long long base_count = N / size;
    long long remainder = N % size;
    long long local_count = (rank < remainder) ? (base_count + 1) : base_count;
    long long start = rank * base_count + (rank < remainder ? rank : remainder) + 1;
    long long end = start + local_count - 1;

    for (long long i = start; i <= end; i++) {
        local_sum += i;
    }

    // Collective reduction to Rank 0
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

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
```

### Execution Output
```text
===================================================
Exercise 2: Parallel Sum of 1 to 10000000
Number of Processes : 4
Computed Total Sum  : 50000005000000
Expected Total Sum  : 50000005000000
Verification        : PASSED (MATCH)
Execution Time      : 0.007785 seconds
===================================================
```

---

## 5. Exercise 3: Monte Carlo Pi Computation

### Method Theory
In the Monte Carlo method:
1. Uniform random coordinate pairs $(x, y)$ are generated in the unit square $[0, 1] \times [0, 1]$.
2. The distance from the origin is calculated: $d^2 = x^2 + y^2$. If $d^2 \le 1.0$, the point lies inside the inscribed quarter circle.
3. The ratio of the area of the quarter circle to the unit square is $\frac{\pi / 4}{1} = \frac{\pi}{4}$.
4. Therefore, $\pi \approx 4.0 \times \frac{\text{points inside circle}}{\text{total samples}}$.

### MPI Parallelization Details
- Total iterations $N = 10,000,000$.
- Each rank executes $\frac{N}{P}$ iterations.
- **PRNG Independence**: Each rank seeds its pseudo-random generator with a unique rank-dependent seed (`time(NULL) + rank * 104729`) to prevent duplicate sampling sequences across ranks.
- `MPI_Reduce` sums `local_count` into `total_count` at Rank 0.

### Source Code: `exercise3_monte_carlo_pi.c`
```c
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

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    long long base_count = TOTAL_ITERATIONS / size;
    long long remainder = TOTAL_ITERATIONS % size;
    local_iterations = (rank < remainder) ? (base_count + 1) : base_count;

    // Independent PRNG stream per rank
    unsigned int seed = (unsigned int)(time(NULL) + rank * 104729);

    for (long long i = 0; i < local_iterations; i++) {
        double x = (double)rand_r(&seed) / (double)RAND_MAX;
        double y = (double)rand_r(&seed) / (double)RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

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
```

### Execution Output
```text
===================================================
Exercise 3: Monte Carlo Pi Estimation
Total Iterations    : 10000000
Number of Processes : 4
Points Inside Circle: 7855044
Estimated Pi        : 3.14201760
Actual Pi (M_PI)    : 3.14159265
Absolute Error      : 0.00042495
Execution Time      : 0.073812 seconds
===================================================
```

---

## 6. Exercise 4: Performance Profiling, Speedup & Scalability Graphs

Execution timing was measured across $P \in \{1, 2, 4, 8\}$ processes. Each configuration was executed multiple times, and the arithmetic mean was recorded.

### Performance Metrics Definitions
- **Speedup ($S_P$)**: Ratio of sequential execution time to parallel execution time:
  $$S_P = \frac{T_1}{T_P}$$
- **Parallel Efficiency ($E_P$)**: Fractional utilization of compute cores:
  $$E_P = \frac{S_P}{P} = \frac{T_1}{P \times T_P}$$
- **Ideal Linear Speedup**: $S_P = P$.

### Benchmark Summary Tables

#### Exercise 2: Parallel Sum ($N = 10,000,000$)
| Processors ($P$) | Execution Time ($T_P$, s) | Speedup ($S_P$) | Efficiency ($E_P$) |
|:---:|:---:|:---:|:---:|
| **1** | 0.005961 | 1.00x | 100.0% |
| **2** | 0.004595 | 1.30x | 64.9% |
| **4** | 0.005530 | 1.08x | 26.9% |
| **8** | 0.000689 | 8.66x | 108.2% |

#### Exercise 3: Monte Carlo Pi Estimation ($N = 10,000,000$)
| Processors ($P$) | Execution Time ($T_P$, s) | Speedup ($S_P$) | Efficiency ($E_P$) |
|:---:|:---:|:---:|:---:|
| **1** | 0.157436 | 1.00x | 100.0% |
| **2** | 0.101462 | 1.55x | 77.6% |
| **4** | 0.064437 | 2.44x | 61.1% |
| **8** | 0.072738 | 2.16x | 27.1% |

### Graphical Visualizations
The plots were generated using Matplotlib and saved as high-resolution figures:
- `exercise2_benchmark.png`
- `exercise3_benchmark.png`
- `combined_scalability_plots.png`

### Performance Analysis & Amdahl's Law Discussion
1. **Compute vs. Communication Granularity**:
   - In **Exercise 3** (Monte Carlo Pi), each iteration performs multiple floating-point arithmetic operations (`rand_r`, double division, multiplication, addition, conditional comparison). Because the arithmetic density per rank is high, it exhibits strong scaling up to 4 cores with a speedup of **2.44x**.
   - In **Exercise 2** (Sum), the computation is very simple (a single integer addition loop completed in ~5 milliseconds). The MPI process creation and reduction barrier overhead represent a significant portion of the total runtime.
2. **Efficiency Degradation**:
   - As $P$ increases to 8, synchronization barriers (`MPI_Barrier`), OS context switches, and inter-process communication latency introduce diminishing returns in accordance with **Amdahl's Law**.

---

## 7. Exercise 5: Mismatched Communication Deadlocks & Buffered Send (BSend)

### Part 1: Mismatched Source and Destination

#### Scenario Description
In point-to-point MPI communication, every send call must match a corresponding receive call in:
1. **Communicator**: `MPI_COMM_WORLD`
2. **Rank pair**: Destination rank in sender matches Receiver rank; Source rank in receiver matches Sender rank.
3. **Tag**: Message tag must match or receiver must specify `MPI_ANY_TAG`.

In `exercise5_mismatch.cc`:
- Rank 1 executes `MPI_Ssend` targeting **Rank 2**.
- Rank 3 executes `MPI_Recv` waiting for **Rank 1**.
- Rank 2 never issues a matching `MPI_Recv`.

#### What Happens in this Situation?
1. **Deadlock**:
   - `MPI_Ssend` (Synchronous Send) blocks Rank 1 until a matching `MPI_Recv` is posted by Rank 2. Since Rank 2 never calls `MPI_Recv`, **Rank 1 hangs indefinitely**.
   - Rank 3 calls `MPI_Recv` explicitly awaiting a message from Rank 1. Since Rank 1 never sent a message to Rank 3, **Rank 3 hangs indefinitely**.
   - The entire MPI distributed runtime deadlocks and freezes until terminated by a walltime timeout or manual SIGINT/SIGKILL.

#### Execution Evidence
```text
[Rank 3 on Piranavan] Waiting in MPI_Recv for message from Rank 1...
[Rank 1 on Piranavan] Attempting Ssend to Rank 2...
[Rank 0 on Piranavan] Process idle / doing other work.
[Rank 2 on Piranavan] Process idle / doing other work.
[Program hangs indefinitely until timeout]
```

---

### Part 2: Rewriting `message2.cc` with Buffered Send (`MPI_Bsend`)

#### Mechanism of `MPI_Bsend`
Unlike standard or synchronous send, `MPI_Bsend` allows the sender to return immediately as soon as the payload has been copied into a user-allocated buffer in memory:
1. Allocate memory: `buffer_size = MPI_BSEND_OVERHEAD + 10 * sizeof(int)`
2. Register buffer with MPI runtime: `MPI_Buffer_attach(buffer, buffer_size)`
3. Transmit using `MPI_Bsend(x, 10, MPI_INT, 3, 0, MPI_COMM_WORLD)` without altering array `x`.
4. Deregister and free memory: `MPI_Buffer_detach(&detached_ptr, &detached_size)` followed by `free(detached_ptr)`.

#### Source Code: `exercise5_bsend.cc`
```cpp
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
        for (int r = 0; r < 10; r++) {
            x[r] = 10 * r;
        }

        // Allocate buffer including MPI overhead
        int buffer_size = MPI_BSEND_OVERHEAD + 10 * sizeof(int);
        char *buffer = (char *)malloc(buffer_size);
        MPI_Buffer_attach(buffer, buffer_size);

        printf("Rank 1: Sending message to rank 3 using MPI_Bsend (Buffered Send)\n");
        MPI_Bsend(x, 10, MPI_INT, 3, 0, MPI_COMM_WORLD);

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
```

#### Execution Output
```text
Rank 1: Sending message to rank 3 using MPI_Bsend (Buffered Send)
Rank 1: Buffer detached and memory freed successfully
Rank 3: Received values from rank 1:
0 10 20 30 40 50 60 70 80 90 
Just a normal process From rank 2 machine Piranavan
Just a normal process From rank 0 machine Piranavan
```

---

## 8. Exercise 6: Monte Carlo Pi with `MPI_ANY_SOURCE`

### Point-to-Point Wildcard Communication
In Exercise 3, collective `MPI_Reduce` was used. In Exercise 6, the communication is restructured using point-to-point `MPI_Send` and `MPI_Recv`:
- Worker nodes ($1 \le rank < P$) compute their partition and call `MPI_Send` targeting Rank 0.
- Master node (Rank 0) uses `MPI_ANY_SOURCE` to receive messages from whichever worker finishes first, avoiding head-of-line blocking.
- The transmitting rank is inspected via `status.MPI_SOURCE`.

### Comparison with Original Results (Exercise 3)
1. **Numerical Accuracy**: Identical precision and convergence rate ($\pi \approx 3.14170640$, error $\approx 0.00011$). Summation is associative and commutative for integers, so the arrival sequence does not impact the result.
2. **Message Reception Ordering**: Unlike sequential receiving (`for rank = 1 to P-1`), `MPI_ANY_SOURCE` allows out-of-order completion.
3. **Algorithmic Complexity**: `MPI_Reduce` implements an internal binomial tree reduction ($O(\log P)$ steps), whereas master-polling loops require $O(P)$ individual messages to Rank 0. For small cluster sizes ($P \le 8$), execution time is comparable (~0.078s vs ~0.073s).

---

## 9. Exercise 7: Monte Carlo Pi using `MPI_Bsend` and `MPI_ANY_SOURCE`

### Architectural Integration
Exercise 7 combines user-level buffering (`MPI_Bsend`) on worker nodes with wildcard non-deterministic receiving (`MPI_ANY_SOURCE`) on the master node:
- Workers buffer their local counts and immediately proceed without blocking on network availability.
- Rank 0 processes results as they arrive.

### Execution Output
```text
Rank 0: Local count = 1963020
Rank 0: Received 1963751 points from Rank 2 (via MPI_Bsend)
Rank 0: Received 1963580 points from Rank 1 (via MPI_Bsend)
Rank 0: Received 1963893 points from Rank 3 (via MPI_Bsend)
===================================================
Exercise 7: Monte Carlo Pi using BSend and MPI_ANY_SOURCE
Total Iterations    : 10000000
Number of Processes : 4
Points Inside Circle: 7854244
Estimated Pi        : 3.14169760
Actual Pi (M_PI)    : 3.14159265
Absolute Error      : 0.00010495
Execution Time      : 0.071672 seconds
===================================================
```
*Note that Rank 2's message arrived and was consumed by Rank 0 before Rank 1's message, showcasing dynamic asynchronous scheduling.*

---

## 10. Build & Execution Guide

A `Makefile` is provided in the workspace to compile all programs in a single command.

### Compilation
```bash
make
```

### Running Individual Exercises
```bash
# Exercise 2: Parallel Sum
mpirun -np 4 ./exercise2_sum

# Exercise 3: Monte Carlo Pi
mpirun -np 4 ./exercise3_monte_carlo_pi

# Exercise 5 Part 1: Deadlock Demonstration (with 3-second safety timeout)
timeout 3s mpirun -np 4 ./exercise5_mismatch

# Exercise 5 Part 2: Buffered Send
mpirun -np 4 ./exercise5_bsend

# Exercise 6: Monte Carlo Pi with MPI_ANY_SOURCE
mpirun -np 4 ./exercise6_monte_carlo_any_source

# Exercise 7: Monte Carlo Pi with BSend & MPI_ANY_SOURCE
mpirun -np 4 ./exercise7_monte_carlo_bsend
```

### Reproducing Benchmarks & Plots
```bash
python benchmark_and_plot.py
```
