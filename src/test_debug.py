#!/usr/bin/env python3
"""Debug script to identify bottleneck"""

import time
import cupy as cp
import numpy as np
import networkx as nx
from utils import prepare_sparse_matrix, batch_sweep, N, M_SCALE_FREE

print(f"=== DEBUG SCRIPT - N={N} ===")

# Test 1: Network generation
print("1. Testing network generation...")
start = time.time()
G = nx.barabasi_albert_graph(N, M_SCALE_FREE, seed=42)
print(f"   NetworkX BA generation: {time.time() - start:.3f}s")

start = time.time()
G_complete = nx.complete_graph(N)
print(f"   NetworkX complete generation: {time.time() - start:.3f}s")

# Test 2: Random data generation
print("2. Testing random data generation...")
start = time.time()
omegas_0 = cp.random.normal(0.0, 0.5, N, dtype=cp.float32)
thetas_0 = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)
print(f"   Random data generation: {time.time() - start:.3f}s")

# Test 3: CSR conversion
print("3. Testing CSR conversion...")
start = time.time()
A_sparse, degrees = prepare_sparse_matrix(G, quiet=True)
print(f"   CSR conversion: {time.time() - start:.3f}s")

# Test 4: Small batch sweep test
print("4. Testing VERY small batch sweep...")
start = time.time()
K_test = cp.array([1.0, 2.0], dtype=cp.float32)  # Only 2 K values
try:
    r_results = batch_sweep(A_sparse, thetas_0, omegas_0, degrees, K_values=K_test, quiet=False)
    print(f"   Small batch sweep (2 K-values): {time.time() - start:.3f}s")
    print(f"   Results: {r_results.get()}")
except Exception as e:
    print(f"   ERROR in batch sweep: {e}")

print("=== DEBUG COMPLETE ===")