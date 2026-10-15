CC = mpicc
CXX = mpic++
CFLAGS = -O2 -Wall
CXXFLAGS = -O2 -Wall
LDFLAGS = -lm

TARGETS = exercise2_sum exercise3_monte_carlo_pi exercise5_mismatch exercise5_bsend exercise6_monte_carlo_any_source exercise7_monte_carlo_bsend

all: $(TARGETS)

exercise2_sum: exercise2_sum.c
	$(CC) $(CFLAGS) exercise2_sum.c -o exercise2_sum $(LDFLAGS)

exercise3_monte_carlo_pi: exercise3_monte_carlo_pi.c
	$(CC) $(CFLAGS) exercise3_monte_carlo_pi.c -o exercise3_monte_carlo_pi $(LDFLAGS)

exercise5_mismatch: exercise5_mismatch.cc
	$(CXX) $(CXXFLAGS) exercise5_mismatch.cc -o exercise5_mismatch

exercise5_bsend: exercise5_bsend.cc
	$(CXX) $(CXXFLAGS) exercise5_bsend.cc -o exercise5_bsend

exercise6_monte_carlo_any_source: exercise6_monte_carlo_any_source.c
	$(CC) $(CFLAGS) exercise6_monte_carlo_any_source.c -o exercise6_monte_carlo_any_source $(LDFLAGS)

exercise7_monte_carlo_bsend: exercise7_monte_carlo_bsend.c
	$(CC) $(CFLAGS) exercise7_monte_carlo_bsend.c -o exercise7_monte_carlo_bsend $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o
