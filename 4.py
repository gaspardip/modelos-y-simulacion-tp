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

# --- 1. PARÁMETROS GLOBALES ---
# Parámetros para el análisis estadístico
N_STATS = 1000       # Un tamaño de red manejable para múltiples corridas
NUM_RUNS = 100       # Número de realizaciones estadísticas (aumentar a 1000 para resultados finales)
M_SCALE_FREE = 3

# Parámetros de la Dinámica
OMEGA_MU = 0.0
OMEGA_SIGMA = 0.5
T_TRANSIENT = 5.0    # Tiempo transitorio (descartado)
T_MEASURE = 5.0      # Tiempo de medición para calcular r
DT = 0.01

# Parámetros del barrido y cálculo de Kc
K_VALUES_SWEEP = np.linspace(0, 5.0, 50)
R_THRESHOLD = 0.5    # Umbral para definir la sincronización

# --- 2. FUNCIONES DE SIMULACIÓN Y ANÁLISIS ---

def kuramoto_odes_gpu(thetas, K, A_sparse, omegas, degrees):
    """
    Calcula la derivada de las fases en la GPU usando operaciones sparse.
    Evita crear la matriz completa NxN de phase_diffs.
    """
    # Para redes sparse, usar multiplicación matriz-vector es más eficiente
    sin_thetas = cp.sin(thetas)
    cos_thetas = cp.cos(thetas)

    # Calcular las sumas ponderadas usando la matriz de adyacencia
    sum_sin = A_sparse @ sin_thetas
    sum_cos = A_sparse @ cos_thetas

    # Calcular la derivada usando la identidad:
    # sum(sin(theta_j - theta_i)) = cos(theta_i)*sum(sin(theta_j)) - sin(theta_i)*sum(cos(theta_j))
    interactions = cp.cos(thetas) * sum_sin - cp.sin(thetas) * sum_cos

    # Evitar división por cero
    safe_degrees = cp.maximum(degrees, 1.0)
    dthetas_dt = omegas + (K / safe_degrees) * interactions

    return dthetas_dt

def run_simulation_and_get_r(K, A_sparse, thetas_0, omegas, degrees):
    """
    Resuelve las EDOs usando el método de Euler con optimizaciones.
    Incluye fase transitoria y medición del parámetro de orden.
    """
    num_steps_transient = int(T_TRANSIENT / DT)
    num_steps_measure = int(T_MEASURE / DT)
    thetas_current = thetas_0.copy()

    # Fase transitoria (no medimos r aquí)
    for _ in range(num_steps_transient):
        dthetas = kuramoto_odes_gpu(thetas_current, K, A_sparse, omegas, degrees)
        thetas_current += dthetas * DT

    # Fase de medición - calculamos r promedio
    r_values = []
    for _ in range(num_steps_measure):
        dthetas = kuramoto_odes_gpu(thetas_current, K, A_sparse, omegas, degrees)
        thetas_current += dthetas * DT

        # Calcular r cada 10 pasos para promediar
        if _ % 10 == 0:
            exp_thetas = cp.exp(1j * thetas_current)
            r = cp.abs(cp.mean(exp_thetas))
            r_values.append(r)

    # Devolver el promedio de r en el estado estacionario
    return cp.mean(cp.array(r_values))

def find_kc_for_single_run(G, omegas_0, thetas_0):
    """
    Realiza un barrido de K para una única red y devuelve su Kc.
    """
    A_gpu = cp.asarray(nx.to_numpy_array(G), dtype=cp.float32)
    degrees_gpu = cp.sum(A_gpu, axis=1); degrees_gpu[degrees_gpu == 0] = 1

    r_values = []
    for K in K_VALUES_SWEEP:
        r = run_simulation_and_get_r(K, A_gpu, thetas_0, omegas_0, degrees_gpu)
        r_values.append(r.get())

    # Encuentra el primer K que cruza el umbral
    indices = np.where(np.array(r_values) > R_THRESHOLD)[0]
    return K_VALUES_SWEEP[indices[0]] if len(indices) > 0 else None

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    print(f"Iniciando análisis estadístico con {NUM_RUNS} corridas para N={N_STATS}...")
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