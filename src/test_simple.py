#!/usr/bin/env python3
"""Simplified version of 1.py to identify the problem"""

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from utils import N, K_VALUES_SWEEP, generate_random_network, prepare_sparse_matrix, batch_sweep

print(f"=== SIMPLIFIED SCRIPT 1 TEST - N={N} ===")

# Test only scale-free network first
print("Testing scale-free network only...")
start_total = time.time()

print("1. Generating scale-free network...")
start = time.time()
G_scale_free, omegas, thetas = generate_random_network(seed=42)
print(f"   Generated in {time.time() - start:.3f}s")

print("2. Running batch sweep analysis...")
start = time.time()
A_sparse, degrees = prepare_sparse_matrix(G_scale_free, quiet=True)
print(f"   CSR preparation: {time.time() - start:.3f}s")

start = time.time()
r_results_gpu = batch_sweep(A_sparse, thetas, omegas, degrees, quiet=True)
r_results = r_results_gpu.get()
print(f"   Batch sweep: {time.time() - start:.3f}s")

print("3. Testing visualization...")
start = time.time()
plt.figure(figsize=(10, 6))
plt.plot(K_VALUES_SWEEP, r_results, "o-", label=f"Scale-free (N={N})", color="crimson")
plt.xlabel("Coupling K")
plt.ylabel("Order parameter r")
plt.title("Kuramoto Synchronization")
plt.legend()
plt.grid(True)
plt.savefig("test_result.png", dpi=100, bbox_inches='tight')
plt.close()  # Don't show, just save
print(f"   Visualization: {time.time() - start:.3f}s")

total_time = time.time() - start_total
print(f"\n=== TOTAL TIME: {total_time:.3f}s ===")

# Show sample results
print(f"Sample r values: {r_results[:5]}")
print(f"Max r: {np.max(r_results):.3f}")