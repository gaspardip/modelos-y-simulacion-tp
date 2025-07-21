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
from utils import N, generate_random_network, sweep_analysis, find_kc

NUM_RUNS = 1000

def find_kc_for_single_run(G, omegas_0, thetas_0):
    """
    Realiza un barrido de K para una única red y devuelve su Kc.
    """
    r_values = sweep_analysis(G, omegas_0, thetas_0)
    # Convert list to numpy array for find_kc
    kc = find_kc(np.array(r_values))
    return kc

def worker_process_run(run_id, seed):
    """
    Worker function for parallel processing of a single run.
    Each process gets its own random seed to ensure different networks.

    Note: Each process will use GPU memory independently. If running out of
    GPU memory, reduce the number of processes or consider using
    cp.cuda.MemoryPool() for better memory management.
    """
    # Generate a new network and initial conditions
    G, omegas_0, thetas_0 = generate_random_network(seed=seed + run_id)

    # Calculate Kc for this instance
    kc = find_kc_for_single_run(G, omegas_0, thetas_0)

    return kc

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    print(f"Iniciando análisis estadístico con {NUM_RUNS} corridas para N={N}...")

    # Determine optimal number of processes
    num_processes = min(mp.cpu_count() - 1, 8)  # Leave one CPU free, max 8 processes
    print(f"Usando {num_processes} procesos paralelos...")
    print(f"Esperando reducción de tiempo de ~{NUM_RUNS}x a ~{NUM_RUNS//num_processes}x")

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

        print("\n======================================================")
        print("          RESULTADOS ESTADÍSTICOS DE KC")
        print("======================================================")
        print(f"Número de corridas exitosas: {len(kc_results)} de {NUM_RUNS}")
        print(f"Kc Promedio (Esperanza):   {mean_kc:.4f}")
        print(f"Desviación Estándar de Kc: {std_kc:.4f}")
        print(f"Intervalo de confianza del 95% (aprox.): ({mean_kc - 2*std_kc:.4f}, {mean_kc + 2*std_kc:.4f})")

        # Visualizar la distribución de los resultados de Kc
        plt.figure(figsize=(10, 6))
        plt.hist(kc_results, bins=15, density=True, alpha=0.7, label='Distribución de Kc')
        plt.axvline(mean_kc, color='red', linestyle='--', linewidth=2, label=f'Media = {mean_kc:.2f}')
        plt.xlabel('Umbral Crítico (Kc)')
        plt.ylabel('Densidad de Probabilidad')
        plt.title(f'Distribución de Kc para Redes Libres de Escala (N={N})')
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.show()
    else:
        print("No se pudo calcular ningún Kc en las corridas. Considera aumentar el rango de K_VALUES_SWEEP.")