#!/usr/bin/env python3
"""Test only scale-free network with N=10k and realistic parameters"""

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from utils import N, K_VALUES_SWEEP, generate_random_network, prepare_sparse_matrix, batch_sweep

print(f"🚀 TESTING SCALE-FREE ONLY (N={N})")
print(f"Parameters: T_TRANSIENT=10, T_MEASURE=20, DT=0.05")

script_start = time.time()

# Generate scale-free network
print("1. Generando red scale-free...")
gen_start = time.time()
G_scale_free, omegas, thetas = generate_random_network(seed=42)  
print(f"   Network: {G_scale_free.number_of_nodes()} nodes, {G_scale_free.number_of_edges()} edges")
print(f"   Generated in {time.time() - gen_start:.2f}s")

# Analysis  
print("2. Preparando simulación...")
prep_start = time.time()
A_sparse, degrees = prepare_sparse_matrix(G_scale_free, quiet=True)
print(f"   CSR prep in {time.time() - prep_start:.2f}s")

print("3. Ejecutando batch sweep...")
sim_start = time.time()
r_results_gpu = batch_sweep(A_sparse, thetas, omegas, degrees, quiet=False)
r_results = r_results_gpu.get()
sim_time = time.time() - sim_start
print(f"   Simulation completed in {sim_time:.2f}s")

# Results
total_time = time.time() - script_start
print(f"\n✅ RESULTADO:")
print(f"   Total time: {total_time:.2f}s")
print(f"   Max r: {np.max(r_results):.3f}")
print(f"   Critical K (r>0.5): {K_VALUES_SWEEP[np.where(r_results > 0.5)[0][0] if np.any(r_results > 0.5) else -1]:.3f}")
print(f"   Speedup estimate: ~{50*30/sim_time:.0f}x vs sequential")

# Quick plot
plt.figure(figsize=(10, 6))
plt.plot(K_VALUES_SWEEP, r_results, "o-", color="crimson", markersize=4)
plt.xlabel("Coupling K")
plt.ylabel("Order parameter r")
plt.title(f"Scale-free Network Synchronization (N={N})")
plt.grid(True)
plt.savefig("scale_free_N10k.png", dpi=150, bbox_inches='tight')
print("   Plot saved: scale_free_N10k.png")
plt.close()