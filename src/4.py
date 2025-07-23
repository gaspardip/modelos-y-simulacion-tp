# ==============================================================================
# SCRIPT 4: ANÁLISIS ESTADÍSTICO DEL UMBRAL CRÍTICO (KC)
# ==============================================================================
# Propósito: Estimar la esperanza (media) y la variabilidad (desviación estándar)
# del umbral de sincronización Kc para una clase de redes libres de escala,
# ejecutando la simulación sobre múltiples realizaciones aleatorias.
#
# Metodología:
# 1. Bucle principal que se repite N veces (análisis estadístico).
# 2. En cada iteración, se genera una nueva red y nuevas frecuencias.
# 3. Se encuentra Kc para cada red mediante un barrido de K.
# 4. Se calculan y visualizan las estadísticas de los Kc obtenidos.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cupy as cp
import networkx as nx
from utils import (N, M_SCALE_FREE, R_THRESHOLD, K_VALUES_SWEEP,
                   prepare_sparse_matrix, batch_sweep)

NUM_RUNS = 500

def find_kc_batch(G, omegas_0, thetas_0):
    """
    Find critical coupling Kc using GPU batch sweep.

    Uses single batch simulation across all K values instead of sequential search.
    """
    A_sparse, degrees = prepare_sparse_matrix(G, quiet=True)

    # Single batch simulation covering entire K-range (QUIET MODE)
    r_results = batch_sweep(A_sparse, thetas_0, omegas_0, degrees, quiet=True)

    # Find critical transition point where r > R_THRESHOLD
    r_values = r_results.get()  # Transfer to CPU for analysis
    critical_indices = np.where(r_values > R_THRESHOLD)[0]

    if len(critical_indices) > 0:
        # Return the first K-value where synchronization occurs
        kc_index = critical_indices[0]
        kc_value = K_VALUES_SWEEP[kc_index]
        return kc_value
    else:
        # No synchronization found in the K-range
        return None

def generate_networks_batch_gpu(batch_size, base_seed):
    """
    Generate batch of networks using NetworkX.

    Uses NetworkX for reliable and fast network generation.
    """
    try:
        # Generate batch using NetworkX
        seeds = [base_seed + i * 1000 for i in range(batch_size)]
        graphs = []
        for seed in seeds:
            G = nx.barabasi_albert_graph(N, M_SCALE_FREE, seed=seed)
            graphs.append(G)

        # Generate random frequencies and phases for each network
        networks = []
        for i, G in enumerate(graphs):
            # Set seed for reproducibility
            np.random.seed(seeds[i])
            cp.random.seed(seeds[i])

            omegas_0 = cp.random.normal(0.0, 1.0, N, dtype=cp.float32)  # OMEGA_MU=0, OMEGA_SIGMA=1
            thetas_0 = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)

            networks.append((G, omegas_0, thetas_0))

        return networks

    except Exception as e:
        print(f"GPU batch generation failed: {e}")
        return []

def process_networks_batch(networks):
    """
    Process a batch of networks to find their Kc values.

    This processes multiple networks sequentially but uses
    batch GPU generation for the networks themselves.
    """
    kc_results = []

    for G, omegas_0, thetas_0 in networks:
        try:
            kc = find_kc_batch(G, omegas_0, thetas_0)
            kc_results.append(kc)
        except Exception as e:
            print(f"Kc calculation failed: {e}")
            kc_results.append(None)

    return kc_results

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    print(f"Starting Kc statistical analysis: {NUM_RUNS} runs, N={N}, M={M_SCALE_FREE}")
    print("Using GPU batch processing instead of multiprocessing for massive speedup")

    start_time = time.time()
    base_seed = int(time.time())

    # GPU batch processing approach
    batch_size = min(32, NUM_RUNS)  # Ultimate generator max batch size
    num_batches = (NUM_RUNS + batch_size - 1) // batch_size

    print(f"Processing {NUM_RUNS} networks in {num_batches} batches of {batch_size}")

    kc_results_raw = []

    # Process in batches using GPU acceleration
    for batch_idx in tqdm(range(num_batches), desc="GPU Batch Processing", unit="batch"):
        # Calculate actual batch size for this iteration
        current_batch_size = min(batch_size, NUM_RUNS - batch_idx * batch_size)
        batch_base_seed = base_seed + batch_idx * batch_size * 1000

        # Generate networks in GPU batch
        print(f"Generating batch {batch_idx+1}/{num_batches} ({current_batch_size} networks)...")
        networks = generate_networks_batch_gpu(current_batch_size, batch_base_seed)

        if networks:
            # Process Kc calculations for this batch
            batch_kc_results = process_networks_batch(networks)
            kc_results_raw.extend(batch_kc_results)
        else:
            # Handle failed batch
            kc_results_raw.extend([None] * current_batch_size)

        # Clear GPU memory between batches
        cp.get_default_memory_pool().free_all_blocks()

    # Filter results
    kc_results = [kc for kc in kc_results_raw if kc is not None]
    failed_runs = len(kc_results_raw) - len(kc_results)

    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n🚀 GPU BATCH PROCESSING COMPLETED!")
    print(f"Total time: {elapsed:.1f}s ({elapsed/NUM_RUNS:.3f}s per network)")
    if failed_runs > 0:
        print(f"Failed runs: {failed_runs}/{NUM_RUNS}")
    print(f"Success rate: {len(kc_results)/NUM_RUNS*100:.1f}%")

    # --- 4. ANÁLISIS Y VISUALIZACIÓN DE RESULTADOS ---
    if kc_results:
        mean_kc = np.mean(kc_results)
        std_kc = np.std(kc_results)
        median_kc = np.median(kc_results)
        min_kc = np.min(kc_results)
        max_kc = np.max(kc_results)

        print(f"\n=== Kc Statistics (n={len(kc_results)}) ===")
        print(f"Mean:   {mean_kc:.4f} ± {std_kc:.4f}")
        print(f"Median: {median_kc:.4f}")
        print(f"Range:  [{min_kc:.4f}, {max_kc:.4f}]")

        # 95% confidence interval
        ci_lower = mean_kc - 2*std_kc
        ci_upper = mean_kc + 2*std_kc
        print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

        # Visualizar la distribución de los resultados de Kc
        plt.figure(figsize=(10, 6))
        plt.hist(kc_results, bins=20, density=True, alpha=0.7, label='Distribución de Kc', color='skyblue')
        plt.axvline(mean_kc, color='red', linestyle='--', linewidth=2, label=f'Media = {mean_kc:.3f}')
        plt.axvline(median_kc, color='green', linestyle=':', linewidth=2, label=f'Mediana = {median_kc:.3f}')
        plt.xlabel('Umbral Crítico (Kc)')
        plt.ylabel('Densidad de Probabilidad')
        plt.title(f'Distribución de Kc para Redes Libres de Escala (N={N})')
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.show()
    else:
        print("ERROR: No successful Kc calculations. Consider increasing K_VALUES_SWEEP range.")