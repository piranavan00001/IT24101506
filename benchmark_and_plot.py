import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

processors = [1, 2, 4, 8]
runs = 3

def run_benchmark(executable, procs):
    times = []
    print(f"\n--- Benchmarking {executable} ---")
    for p in procs:
        p_times = []
        for r in range(runs):
            cmd = f'wsl -e bash -c "cd \'/mnt/c/Users/HP/OneDrive - Sri Lanka Institute of Information Technology/Desktop/lab_08\' && mpirun --oversubscribe -np {p} ./{executable}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            match = re.search(r"Execution Time\s*:\s*([0-9.]+)\s*seconds", result.stdout)
            if match:
                t = float(match.group(1))
                p_times.append(t)
            else:
                print(f"Warning: could not parse time for P={p}, run={r}: {result.stdout} {result.stderr}")
        avg_t = np.mean(p_times) if p_times else 0.0
        times.append(avg_t)
        print(f"P={p}: Times={p_times}, Avg={avg_t:.6f}s")
    return times

# Benchmark Exercise 2 (Sum of numbers)
time_ex2 = run_benchmark("exercise2_sum", processors)

# Benchmark Exercise 3 (Monte Carlo Pi)
time_ex3 = run_benchmark("exercise3_monte_carlo_pi", processors)

# Speedup: S_p = T_1 / T_p
speedup_ex2 = [time_ex2[0] / t if t > 0 else 0 for t in time_ex2]
speedup_ex3 = [time_ex3[0] / t if t > 0 else 0 for t in time_ex3]

# Efficiency: E_p = S_p / p
eff_ex2 = [s / p for s, p in zip(speedup_ex2, processors)]
eff_ex3 = [s / p for s, p in zip(speedup_ex3, processors)]

print("\nSummary Tables:")
print("\nExercise 2 (Parallel Sum 1 to 10M):")
print("Processors | Avg Time (s) | Speedup | Efficiency")
for p, t, s, e in zip(processors, time_ex2, speedup_ex2, eff_ex2):
    print(f"{p:10d} | {t:12.6f} | {s:7.2f}x | {e*100:9.1f}%")

print("\nExercise 3 (Monte Carlo Pi 10M iterations):")
print("Processors | Avg Time (s) | Speedup | Efficiency")
for p, t, s, e in zip(processors, time_ex3, speedup_ex3, eff_ex3):
    print(f"{p:10d} | {t:12.6f} | {s:7.2f}x | {e*100:9.1f}%")

# Save detailed results to text
with open("benchmark_results.txt", "w") as f:
    f.write("Exercise 2 Benchmark (Sum 1 to 10,000,000):\n")
    f.write("Processors\tTime(s)\tSpeedup\tEfficiency(%)\n")
    for p, t, s, e in zip(processors, time_ex2, speedup_ex2, eff_ex2):
        f.write(f"{p}\t{t:.6f}\t{s:.2f}\t{e*100:.1f}\n")
    f.write("\nExercise 3 Benchmark (Monte Carlo Pi 10,000,000 iterations):\n")
    f.write("Processors\tTime(s)\tSpeedup\tEfficiency(%)\n")
    for p, t, s, e in zip(processors, time_ex3, speedup_ex3, eff_ex3):
        f.write(f"{p}\t{t:.6f}\t{s:.2f}\t{e*100:.1f}\n")

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Figure 1: Exercise 2 (Time & Speedup)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(processors, time_ex2, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='Measured Time')
ax1.set_title('Exercise 2: Execution Time vs Number of Processors', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Processors (P)', fontsize=11)
ax1.set_ylabel('Execution Time (seconds)', fontsize=11)
ax1.set_xticks(processors)
ax1.grid(True, linestyle='--', alpha=0.6)
for p, t in zip(processors, time_ex2):
    ax1.annotate(f'{t:.4f}s', (p, t), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

ax2.plot(processors, speedup_ex2, 's-', color='#2ca02c', linewidth=2.5, markersize=8, label='Measured Speedup')
ax2.plot(processors, processors, '--', color='#7f7f7f', linewidth=1.8, label='Ideal Linear Speedup')
ax2.set_title('Exercise 2: Speedup vs Number of Processors', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Processors (P)', fontsize=11)
ax2.set_ylabel('Speedup (Sp = T1 / Tp)', fontsize=11)
ax2.set_xticks(processors)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
for p, s in zip(processors, speedup_ex2):
    ax2.annotate(f'{s:.2f}x', (p, s), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('exercise2_benchmark.png', dpi=300)
plt.close()

# Figure 2: Exercise 3 (Time & Speedup)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(processors, time_ex3, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='Measured Time')
ax1.set_title('Exercise 3: Execution Time vs Number of Processors', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Processors (P)', fontsize=11)
ax1.set_ylabel('Execution Time (seconds)', fontsize=11)
ax1.set_xticks(processors)
ax1.grid(True, linestyle='--', alpha=0.6)
for p, t in zip(processors, time_ex3):
    ax1.annotate(f'{t:.4f}s', (p, t), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

ax2.plot(processors, speedup_ex3, 's-', color='#9467bd', linewidth=2.5, markersize=8, label='Measured Speedup')
ax2.plot(processors, processors, '--', color='#7f7f7f', linewidth=1.8, label='Ideal Linear Speedup')
ax2.set_title('Exercise 3: Speedup vs Number of Processors', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Processors (P)', fontsize=11)
ax2.set_ylabel('Speedup (Sp = T1 / Tp)', fontsize=11)
ax2.set_xticks(processors)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
for p, s in zip(processors, speedup_ex3):
    ax2.annotate(f'{s:.2f}x', (p, s), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('exercise3_benchmark.png', dpi=300)
plt.close()

# Figure 3: Combined Comparison Chart
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Ex2 Time
axs[0, 0].plot(processors, time_ex2, 'o-', color='#1f77b4', linewidth=2, markersize=7)
axs[0, 0].set_title('Ex 2 (Sum): Time vs Processors', fontweight='bold')
axs[0, 0].set_xlabel('Processors')
axs[0, 0].set_ylabel('Time (s)')
axs[0, 0].set_xticks(processors)
axs[0, 0].grid(True, linestyle='--', alpha=0.6)

# Ex2 Speedup
axs[0, 1].plot(processors, speedup_ex2, 's-', color='#2ca02c', linewidth=2, markersize=7, label='Measured')
axs[0, 1].plot(processors, processors, '--', color='gray', label='Ideal')
axs[0, 1].set_title('Ex 2 (Sum): Speedup vs Processors', fontweight='bold')
axs[0, 1].set_xlabel('Processors')
axs[0, 1].set_ylabel('Speedup')
axs[0, 1].set_xticks(processors)
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle='--', alpha=0.6)

# Ex3 Time
axs[1, 0].plot(processors, time_ex3, 'o-', color='#d62728', linewidth=2, markersize=7)
axs[1, 0].set_title('Ex 3 (Monte Carlo Pi): Time vs Processors', fontweight='bold')
axs[1, 0].set_xlabel('Processors')
axs[1, 0].set_ylabel('Time (s)')
axs[1, 0].set_xticks(processors)
axs[1, 0].grid(True, linestyle='--', alpha=0.6)

# Ex3 Speedup
axs[1, 1].plot(processors, speedup_ex3, 's-', color='#9467bd', linewidth=2, markersize=7, label='Measured')
axs[1, 1].plot(processors, processors, '--', color='gray', label='Ideal')
axs[1, 1].set_title('Ex 3 (Monte Carlo Pi): Speedup vs Processors', fontweight='bold')
axs[1, 1].set_xlabel('Processors')
axs[1, 1].set_ylabel('Speedup')
axs[1, 1].set_xticks(processors)
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle='--', alpha=0.6)

plt.suptitle('MPI Parallel Benchmarks: Execution Time & Speedup Scaling', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('combined_scalability_plots.png', dpi=300)
plt.close()

print("\nGraphs saved successfully: exercise2_benchmark.png, exercise3_benchmark.png, combined_scalability_plots.png")
