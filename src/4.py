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
import multiprocessing as mp
from functools import partial
import cupy as cp
from utils import (N, R_THRESHOLD, K_VALUES_SWEEP, generate_random_network,
                   prepare_sparse_matrix, batch_sweep)

NUM_RUNS = 1000

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

def worker_process_run(run_id, seed):
    """
    GPU-accelerated worker for Kc analysis.
    """
    try:
        # Clear GPU memory at start
        cp.get_default_memory_pool().free_all_blocks()

        # Generate network and find Kc
        G, omegas_0, thetas_0 = generate_random_network(seed=seed + run_id, quiet=True)
        kc = find_kc_batch(G, omegas_0, thetas_0)

        return kc
    except Exception as e:
        return None
    finally:
        # Cleanup GPU memory
        cp.get_default_memory_pool().free_all_blocks()

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    print(f"Starting Kc statistical analysis: {NUM_RUNS} runs, N={N}")
    
    num_processes = min(4, mp.cpu_count())
    print(f"Using {num_processes} parallel processes")

    start_time = time.time()
    base_seed = int(time.time())
    worker_with_seed = partial(worker_process_run, seed=base_seed)

    # Run parallel processing with simplified progress tracking
    with mp.Pool(processes=num_processes) as pool:
        kc_results_raw = list(tqdm(
            pool.imap_unordered(worker_with_seed, range(NUM_RUNS)),
            total=NUM_RUNS,
            desc="Kc Analysis Progress",
            unit="run"
        ))

    # Filter results
    kc_results = [kc for kc in kc_results_raw if kc is not None]
    failed_runs = len(kc_results_raw) - len(kc_results)

    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/NUM_RUNS:.2f}s/run)")
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