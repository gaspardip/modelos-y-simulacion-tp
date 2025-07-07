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

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from utils import *

R_THRESHOLD = 0.5
NUM_RUNS = 1000

def find_kc_for_single_run(G, omegas_0, thetas_0):
    """
    Realiza un barrido de K para una única red y devuelve su Kc.
    """
    A_gpu = cp.asarray(nx.to_numpy_array(G), dtype=cp.float32)
    degrees_gpu = cp.sum(A_gpu, axis=1); degrees_gpu[degrees_gpu == 0] = 1

    r_values = []
    for K in K_VALUES_SWEEP:
        r, _ = run_simulation(K, A_gpu, thetas_0, omegas_0, degrees_gpu)
        r_values.append(r.get())

    # Encuentra el primer K que cruza el umbral
    indices = np.where(np.array(r_values) > R_THRESHOLD)[0]
    return K_VALUES_SWEEP[indices[0]] if len(indices) > 0 else None

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    print(f"Iniciando análisis estadístico con {NUM_RUNS} corridas para N={N}...")
    kc_results = []
    start_time = time.time()

    # Bucle principal para el análisis estadístico
    for i in tqdm(range(NUM_RUNS), desc="Progreso del Análisis Estadístico"):
        # Generar una NUEVA red y NUEVOS datos iniciales en cada iteración
        G = nx.barabasi_albert_graph(N_STATS, M_SCALE_FREE)

        # No fijamos la semilla aquí para asegurar que cada corrida sea única
        omegas_gpu = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N_STATS, dtype=cp.float32)
        thetas_0_gpu = cp.random.uniform(0, 2 * np.pi, N_STATS, dtype=cp.float32)

        # Calcular Kc para esta instancia específica
        kc = find_kc_for_single_run(G, omegas_gpu, thetas_0_gpu)

        if kc is not None:
            kc_results.append(kc)
        else:
            print(f"Advertencia: La corrida {i+1} no alcanzó la sincronización en el rango de K probado.")

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
        plt.title(f'Distribución de Kc para Redes Libres de Escala (N={N_STATS})')
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.show()
    else:
        print("No se pudo calcular ningún Kc en las corridas. Considera aumentar el rango de K_VALUES_SWEEP.")