# ==============================================================================
# SCRIPT 1: ANÁLISIS CUANTITATIVO A GRAN ESCALA (r vs. K)
# ==============================================================================
# Propósito: Generar el gráfico comparativo de la transición a la sincronización
# para un Grafo Completo y una Red Libre de Escala con N=5000 nodos.
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
N = 5000
M_SCALE_FREE = 3  # Un poco más denso para hacerlo interesante

# Parámetros de la Dinámica
OMEGA_MU = 0.0
OMEGA_SIGMA = 0.5
T_SIMULATION = 20.0  # Tiempo de simulación
DT = 0.01  # Paso de tiempo para el integrador de Euler

# Rango de K para el barrido
K_VALUES = np.linspace(0, 5, 50)

# --- 2. FUNCIONES OPTIMIZADAS PARA GPU ---

def kuramoto_odes_gpu(thetas, K, A, omegas, degrees):
    """
    Calcula la derivada de las fases en la GPU. Es el corazón de la física.
    """
    phase_diffs = thetas - thetas[:, cp.newaxis]
    interactions = A * cp.sin(phase_diffs)
    interaction_sum = cp.sum(interactions, axis=1)
    dthetas_dt = omegas + (K / degrees) * interaction_sum
    return dthetas_dt

def run_simulation_gpu(K, A, thetas_0, omegas, degrees):
    """
    Resuelve las EDOs usando el método de Euler, enteramente en la GPU.
    Devuelve el parámetro de orden final 'r'.
    """
    num_steps = int(T_SIMULATION / DT)
    thetas_current = thetas_0.copy()

    # Bucle de integración de Euler
    for _ in range(num_steps):
        dthetas = kuramoto_odes_gpu(thetas_current, K, A, omegas, degrees)
        thetas_current += dthetas * DT

    # Calcular el parámetro de orden a partir del estado final.
    # Para un T_SIMULATION suficientemente largo, esto se aproxima al estado estacionario.
    exp_thetas = cp.exp(1j * thetas_current)
    r_final = cp.abs(cp.mean(exp_thetas))
    return r_final

def run_sweep_analysis_gpu(network_type, G, thetas_0, omegas):
    """
    Realiza un barrido de K para una red dada, orquestando las simulaciones en GPU.
    """
    print(f"======================================================")
    print(f"INICIANDO BARRIDO DE K PARA: {network_type.upper()} (N={G.number_of_nodes()})")
    print(f"======================================================")

    r_results = []
    start_time = time.time()

    # Mover la matriz de adyacencia y grados a la GPU una sola vez
    A_gpu = cp.asarray(nx.to_numpy_array(G), dtype=cp.float32)
    degrees_gpu = cp.sum(A_gpu, axis=1)
    degrees_gpu[degrees_gpu == 0] = 1

    for i, K in enumerate(K_VALUES):
        print(f"  Calculando... K = {K:.2f} ({i+1}/{len(K_VALUES)})")
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