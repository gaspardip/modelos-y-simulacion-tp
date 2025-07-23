#!/usr/bin/env python3
"""Test complete graph with optimized parameters for dense networks"""

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from utils import N, K_VALUES_SWEEP, prepare_sparse_matrix, batch_sweep

print(f"🚀 TESTING COMPLETE GRAPH (N={N})")
print("Using optimized parameters for dense networks...")

script_start = time.time()

# Generate complete graph
print("1. Generando grafo completo...")
gen_start = time.time()
G_complete = nx.complete_graph(N)
print(f"   Network: {G_complete.number_of_nodes()} nodes, {G_complete.number_of_edges():,} edges")
print(f"   Generated in {time.time() - gen_start:.2f}s")

# Generate random initial conditions
print("2. Generando condiciones iniciales...")
init_start = time.time()
np.random.seed(42)
cp.random.seed(42)
omegas = cp.random.normal(0.0, 0.5, N, dtype=cp.float32)
thetas = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)
print(f"   Random data in {time.time() - init_start:.2f}s")

# CSR preparation
print("3. Preparando matriz CSR...")
prep_start = time.time()
A_sparse, degrees = prepare_sparse_matrix(G_complete, quiet=True)
prep_time = time.time() - prep_start
print(f"   CSR prep in {prep_time:.2f}s")

# Simulation with smaller K range for complete graphs
print("4. Ejecutando batch sweep (reduced K range for complete graphs)...")
sim_start = time.time()

# Complete graphs synchronize at much lower K values
K_complete = cp.linspace(0.0, 2.0, 25, dtype=cp.float32)  # Reduced range and count
print(f"   Using {len(K_complete)} K-values: {K_complete[0]:.2f} to {K_complete[-1]:.2f}")

r_results_gpu = batch_sweep(A_sparse, thetas, omegas, degrees, K_values=K_complete, quiet=False)
r_results = r_results_gpu.get()
sim_time = time.time() - sim_start
print(f"   Simulation completed in {sim_time:.2f}s")

# Results
total_time = time.time() - script_start
print(f"\n✅ RESULTADO:")
print(f"   Total time: {total_time:.2f}s")
print(f"   Max r: {np.max(r_results):.3f}")
critical_indices = np.where(r_results > 0.5)[0]
if len(critical_indices) > 0:
    print(f"   Critical K (r>0.5): {K_complete.get()[critical_indices[0]]:.3f}")
else:
    print(f"   No synchronization found (max r = {np.max(r_results):.3f})")
print(f"   Speedup estimate: ~{25*30/sim_time:.0f}x vs sequential")

# Quick plot
plt.figure(figsize=(10, 6))
plt.plot(K_complete.get(), r_results, "o-", color="royalblue", markersize=6)
plt.xlabel("Coupling K")
plt.ylabel("Order parameter r")
plt.title(f"Complete Graph Synchronization (N={N})")
plt.grid(True)
plt.savefig("complete_graph_N10k.png", dpi=150, bbox_inches='tight')
print("   Plot saved: complete_graph_N10k.png")
plt.close()