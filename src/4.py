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
from utils import N, R_THRESHOLD, generate_random_network, run_simulation, prepare_sparse_matrix

NUM_RUNS = 1000

def find_kc_adaptive(G, omegas_0, thetas_0, max_iterations=8):
    """
    Finds Kc using adaptive binary search instead of full sweep.
    """
    A_sparse, degrees = prepare_sparse_matrix(G)

    k_low, k_high = 0.0, 6.0

    for _ in range(max_iterations):
        k_mid = (k_low + k_high) / 2.0

        # Run single simulation at k_mid
        r, _, _ = run_simulation(k_mid, A_sparse, omegas_0, thetas_0, degrees)
        r_cpu = float(r.get())

        if r_cpu > R_THRESHOLD:
            k_high = k_mid
        else:
            k_low = k_mid

        # Early convergence check
        if abs(k_high - k_low) < 0.05:
            break

    # Return the transition point (upper bound when synchronized)
    return k_high if r_cpu > R_THRESHOLD else None

def worker_process_run(run_id, seed):
    """
    Optimized worker with GPU memory management and adaptive Kc finding.
    """
    try:
        # Clear GPU memory at start
        cp.get_default_memory_pool().free_all_blocks()

        # Generate a new network and initial conditions
        G, omegas_0, thetas_0 = generate_random_network(seed=seed + run_id)

        # Use adaptive Kc finding (8 sims instead of 50)
        kc = find_kc_adaptive(G, omegas_0, thetas_0)

        return kc
    except Exception as e:
        print(f"Worker {run_id} failed: {e}")
        return None
    finally:
        # Cleanup GPU memory
        cp.get_default_memory_pool().free_all_blocks()

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    print(f"Iniciando análisis estadístico con {NUM_RUNS} corridas para N={N}...")

    # Use most available cores - binary search uses much less GPU memory
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    print(f"Usando {num_processes} procesos paralelos...")
    print(f"Estimated speedup: ~6x (adaptive Kc) + {num_processes}x (parallel) = ~{6*num_processes}x total")

    start_time = time.time()

    # Use a fixed seed for reproducibility (optional)
    base_seed = int(time.time())

    # Create partial function with fixed seed
    worker_with_seed = partial(worker_process_run, seed=base_seed)

    # Run parallel processing with progress bar
    with mp.Pool(processes=num_processes) as pool:
        # Use imap_unordered for better performance and progress tracking
        kc_results_raw = list(tqdm(
            pool.imap_unordered(worker_with_seed, range(NUM_RUNS)),
            total=NUM_RUNS,
            desc="Progreso del Análisis Estadístico"
        ))

    # Filter out None results
    kc_results = [kc for kc in kc_results_raw if kc is not None]
    failed_runs = len(kc_results_raw) - len(kc_results)

    if failed_runs > 0:
        print(f"Advertencia: {failed_runs} corridas no alcanzaron la sincronización en el rango de K probado.")

    end_time = time.time()
    print(f"\nAnálisis estadístico completado en {end_time - start_time:.2f} segundos.")

    # --- 4. ANÁLISIS Y VISUALIZACIÓN DE RESULTADOS ---
    if kc_results:
        mean_kc = np.mean(kc_results)
        std_kc = np.std(kc_results)
        median_kc = np.median(kc_results)
        min_kc = np.min(kc_results)
        max_kc = np.max(kc_results)

        print("\n======================================================")
        print("          RESULTADOS ESTADÍSTICOS DE KC")
        print("======================================================")
        print(f"Número de corridas exitosas: {len(kc_results)} de {NUM_RUNS}")
        print(f"Kc Promedio (Esperanza):   {mean_kc:.4f}")
        print(f"Kc Mediana:                {median_kc:.4f}")
        print(f"Desviación Estándar de Kc: {std_kc:.4f}")
        print(f"Rango: [{min_kc:.4f}, {max_kc:.4f}]")

        # Intervalo de confianza simple
        ci_lower = mean_kc - 2*std_kc
        ci_upper = mean_kc + 2*std_kc
        print(f"Intervalo de confianza del 95% (aprox.): ({ci_lower:.4f}, {ci_upper:.4f})")

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
        print("No se pudo calcular ningún Kc en las corridas. Considera aumentar el rango de K_VALUES_SWEEP.")