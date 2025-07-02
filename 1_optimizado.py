# ==============================================================================
# SCRIPT 1: ANÁLISIS CUANTITATIVO A GRAN ESCALA (r vs. K) - VERSIÓN OPTIMIZADA
# ==============================================================================
# Propósito: Generar el gráfico comparativo de la transición a la sincronización
# para un Grafo Completo y una Red Libre de Escala con N=10000 nodos.
#
# Tecnología:
# - CuPy para aceleración por GPU.
# - Integrador de Euler para resolver las EDOs directamente en la GPU.
# ==============================================================================

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

# --- 1. PARÁMETROS GLOBALES ---
# Parámetros de la Red
N = 10000
M_SCALE_FREE = 3  # Un poco más denso para hacerlo interesante

# Parámetros de la Dinámica
OMEGA_MU = 0.0
OMEGA_SIGMA = 0.5
T_TRANSIENT = 5.0    # Tiempo transitorio (descartado)
T_MEASURE = 5.0      # Tiempo de medición para calcular r
DT = 0.01            # Paso del tiempo para el integrador de Euler

# Rango de K para el barrido
K_VALUES = np.linspace(0, 5, 30)

# --- 2. FUNCIONES OPTIMIZADAS PARA GPU ---

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

def run_simulation_gpu(K, A_sparse, thetas_0, omegas, degrees):
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

def run_sweep_analysis_gpu(network_type, G, thetas_0, omegas):
    """
    Realiza un barrido de K para una red dada, orquestando las simulaciones en GPU.
    """
    print(f"======================================================")
    print(f"INICIANDO BARRIDO DE K PARA: {network_type.upper()} (N={G.number_of_nodes()})")
    print(f"======================================================")

    r_results = []
    start_time = time.time()

    # Para redes completas, usar matriz densa es más eficiente
    if network_type == "Grafo Completo":
        A_gpu = cp.asarray(nx.to_numpy_array(G), dtype=cp.float32)
        degrees_gpu = cp.full(N, N-1, dtype=cp.float32)  # Todos tienen grado N-1
    else:
        # Para redes sparse, convertir a formato CSR de CuPy
        A_scipy = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float32)
        A_gpu = cp.sparse.csr_matrix(A_scipy)
        degrees_gpu = cp.array(A_scipy.sum(axis=1).flatten(), dtype=cp.float32)

    for i, K in enumerate(K_VALUES):
        print(f"  Calculando... K = {K:.2f} ({i+1}/{len(K_VALUES)})")

        if network_type == "Grafo Completo":
            # Usar la versión original para grafos completos
            r = run_simulation_gpu(K, A_gpu, thetas_0, omegas, degrees_gpu)
        else:
            # Usar la versión optimizada para redes sparse
            r = run_simulation_gpu(K, A_gpu, thetas_0, omegas, degrees_gpu)

        r_results.append(r.get())  # .get() mueve el resultado de GPU a CPU

    end_time = time.time()
    print(f"Análisis completado en {end_time - start_time:.2f} segundos.")
    return np.array(r_results)

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    # Generamos los datos aleatorios una sola vez y los movemos a la GPU
    cp.random.seed(42)
    omegas_gpu = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N, dtype=cp.float32)
    thetas_0_gpu = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)

    # --- Análisis para Grafo Completo ---
    print("Generando Grafo Completo en CPU...")
    G_complete = nx.complete_graph(N)
    r_complete = run_sweep_analysis_gpu("Grafo Completo", G_complete, thetas_0_gpu, omegas_gpu)

    # --- Análisis para Red Libre de Escala ---
    print("\nGenerando Red Libre de Escala en CPU...")
    G_scale_free = nx.barabasi_albert_graph(N, M_SCALE_FREE, seed=42)
    r_scale_free = run_sweep_analysis_gpu("Red Libre de Escala", G_scale_free, thetas_0_gpu, omegas_gpu)

    # --- Visualización Comparativa Final ---
    print("\nGenerando gráfico comparativo final...")
    plt.style.use('seaborn-v0_8-whitegrid') # Estilo de gráfico profesional
    plt.figure(figsize=(14, 8))

    plt.plot(K_VALUES, r_complete, "o-", label=f"Grafo Completo (N={N})", color="royalblue", markersize=8, linewidth=2.5)
    plt.plot(K_VALUES, r_scale_free, "s-", label=f"Red Libre de Escala (N={N})", color="crimson", markersize=8, linewidth=2.5)

    plt.xlabel("Fuerza de Acoplamiento (K)", fontsize=16)
    plt.ylabel("Parámetro de Orden (r)", fontsize=16)
    plt.title("Transición a la Sincronización en Redes Grandes", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()