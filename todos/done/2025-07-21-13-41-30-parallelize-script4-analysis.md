# Script 4 Parallelization Analysis

## Analysis of src/4.py

### 1. Current Structure of the 1000 Runs

The script performs **1000 independent runs** (lines 24, 49) in a sequential for loop:

```python
# Line 49-64: Main loop
for i in tqdm(range(NUM_RUNS), desc="Progreso del Análisis Estadístico"):
    # Generate new network and initial conditions
    G = nx.barabasi_albert_graph(N_STATS, M_SCALE_FREE)  # Line 51
    omegas_gpu = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N_STATS, dtype=cp.float32)  # Line 54
    thetas_0_gpu = cp.random.uniform(0, 2 * np.pi, N_STATS, dtype=cp.float32)  # Line 55
    
    # Find Kc for this instance
    kc = find_kc_for_single_run(G, omegas_gpu, thetas_0_gpu)  # Line 58
    
    if kc is not None:
        kc_results.append(kc)
```

### 2. What Makes Each Run Independent

Each run is **completely independent** because:

1. **New network generation** (line 51): Each iteration creates a fresh Barabási-Albert network
2. **New initial frequencies** (line 54): Random normal distribution for omegas
3. **New initial phases** (line 55): Random uniform distribution for thetas
4. **No shared state**: Each run's result (Kc) depends only on its own network and initial conditions
5. **No inter-run dependencies**: Results are simply collected in a list

### 3. Exact Code Section for Parallelization

The parallelizable section is **lines 49-64**, specifically the loop body:

```python
# Lines 50-63 can be extracted into a worker function
def process_single_run(run_id):
    # Generate network and initial conditions
    G = nx.barabasi_albert_graph(N_STATS, M_SCALE_FREE)
    omegas_gpu = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N_STATS, dtype=cp.float32)
    thetas_0_gpu = cp.random.uniform(0, 2 * np.pi, N_STATS, dtype=cp.float32)
    
    # Calculate Kc
    kc = find_kc_for_single_run(G, omegas_gpu, thetas_0_gpu)
    
    return kc
```

### 4. Current Performance Bottlenecks

1. **Sequential execution**: Each of 1000 runs waits for the previous one to complete
2. **GPU underutilization**: Only one simulation runs at a time on the GPU
3. **Network generation overhead**: NetworkX graph generation is CPU-bound
4. **Memory transfers**: Each run involves CPU→GPU transfers for the adjacency matrix

The `find_kc_for_single_run` function (lines 26-40) performs:
- **50 K-value sweeps** (from `K_VALUES_SWEEP` in utils.py, line 20)
- Each sweep runs a full simulation with transient and measurement phases
- Total simulations per run: 50

### 5. Multiprocessing.Pool Implementation Strategy

Here's how `multiprocessing.Pool` could be implemented:

```python
from multiprocessing import Pool
import multiprocessing as mp

def worker_process_run(run_id):
    """Worker function for parallel processing"""
    # Set different random seed for each process
    np.random.seed(SEED + run_id if SEED else None)
    cp.random.seed(SEED + run_id if SEED else None)
    
    # Generate network and initial conditions
    G = nx.barabasi_albert_graph(N_STATS, M_SCALE_FREE)
    omegas_gpu = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N_STATS, dtype=cp.float32)
    thetas_0_gpu = cp.random.uniform(0, 2 * np.pi, N_STATS, dtype=cp.float32)
    
    # Calculate Kc
    kc = find_kc_for_single_run(G, omegas_gpu, thetas_0_gpu)
    
    return kc

# In main:
num_processes = mp.cpu_count() - 1  # Leave one CPU free
with Pool(processes=num_processes) as pool:
    # Map the worker function to all run IDs
    kc_results_raw = pool.map(worker_process_run, range(NUM_RUNS))
    
# Filter out None values
kc_results = [kc for kc in kc_results_raw if kc is not None]
```

### Key Considerations for Parallelization

1. **GPU Memory**: Multiple processes may compete for GPU memory
2. **Random Seeds**: Each process needs unique seeds to ensure different networks
3. **Progress Tracking**: tqdm won't work directly with Pool.map
4. **Error Handling**: Need to handle potential GPU allocation failures
5. **Process Count**: Optimal number depends on CPU cores and GPU memory

### Code Issues Found

1. **Undefined variable**: `N_STATS` is used but not defined (should likely be `N` from utils.py)
2. **Import issue**: The script uses `N` in line 44 but also references undefined `N_STATS`

This script is an excellent candidate for parallelization due to its embarrassingly parallel nature - each run is completely independent and the results are simply aggregated at the end.

## Summary of findings:

### 1. **Imports from utils.py**
The file `src/4.py` imports everything from `utils.py` using `from utils import *` (line 21).

### 2. **N_STATS is NOT defined**
`N_STATS` is used in `src/4.py` on lines 51, 54, 55, and 87, but it's **not defined anywhere** in the codebase. This is a bug. Looking at the code:
- Line 44 uses `N` in the print statement
- Lines 51, 54, 55, and 87 use `N_STATS`

This suggests that `N_STATS` should probably be `N`, which is defined in `utils.py` as `N = 10000` (line 10).

### 3. **GPU Memory Management Considerations**
Based on my analysis:

- **No explicit GPU memory management**: The code doesn't include any explicit GPU memory pool management or cleanup operations.
- **Potential memory leak identified**: There's a note in `/todos/todos.md` about a memory leak in Script 2 related to storing large arrays without cleaning them.
- **Memory considerations for Script 4**:
  - The script runs 1000 iterations (`NUM_RUNS = 1000`)
  - Each iteration creates new GPU arrays for the adjacency matrix, degrees, omegas, and thetas
  - The `find_kc_for_single_run` function converts the adjacency matrix to GPU memory on each call
  - No explicit memory cleanup between iterations

### 4. **Recommended GPU Memory Management**
For better GPU memory management, consider:
1. Using CuPy's memory pool to manage allocations
2. Explicitly freeing GPU memory after each iteration
3. Reusing GPU arrays when possible instead of creating new ones
4. Adding memory pool statistics monitoring

The main issue to fix immediately is replacing `N_STATS` with `N` to resolve the undefined variable error.