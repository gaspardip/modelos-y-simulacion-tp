#!/usr/bin/env python3
"""Test K-value scaling"""

import time
import cupy as cp
import numpy as np
import networkx as nx
from utils import prepare_sparse_matrix, batch_sweep, N, M_SCALE_FREE

print(f"=== K-VALUE SCALING TEST - N={N} ===")

# Setup
G = nx.barabasi_albert_graph(N, M_SCALE_FREE, seed=42)
omegas_0 = cp.random.normal(0.0, 0.5, N, dtype=cp.float32)
thetas_0 = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)
A_sparse, degrees = prepare_sparse_matrix(G, quiet=True)

# Test different numbers of K-values
k_counts = [2, 5, 10, 20, 50]

for k_count in k_counts:
    print(f"\nTesting {k_count} K-values...")
    K_test = cp.linspace(0.5, 3.0, k_count, dtype=cp.float32)
    
    start = time.time()
    try:
        r_results = batch_sweep(A_sparse, thetas_0, omegas_0, degrees, K_values=K_test, quiet=True)
        elapsed = time.time() - start
        print(f"   Time: {elapsed:.3f}s ({elapsed/k_count:.4f}s per K-value)")
        print(f"   Results shape: {r_results.shape}")
    except Exception as e:
        print(f"   ERROR: {e}")
        break

print("=== SCALING TEST COMPLETE ===")