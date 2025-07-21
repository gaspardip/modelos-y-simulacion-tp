# ==============================================================================
# SCRIPT 1: ANÁLISIS CUANTITATIVO A GRAN ESCALA (r vs. K)
# ==============================================================================
# Propósito: Generar el gráfico comparativo de la transición a la sincronización
# para un Grafo Completo y una Red Libre de Escala con N=10000 nodos.
# ==============================================================================

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from utils import N, K_VALUES_SWEEP, run_simulation, run_simulation_complete_graph, generate_random_network

def run_sweep_analysis(network_type, G, thetas, omegas):
    """
    Realiza un barrido de K para una red dada, orquestando las simulaciones en GPU.
    """
    print(f"======================================================")
    print(f"INICIANDO BARRIDO DE K PARA: {network_type.upper()} (N={G.number_of_nodes()})")
    print(f"======================================================")

    r_results = []
    start_time = time.time()

    # Para redes completas, usar la optimización analítica (sin matriz densa)
    if network_type == "Grafo Completo":
        print("    Usando optimización analítica para grafo completo (sin matriz densa)")
        for i, K in enumerate(K_VALUES_SWEEP):
            print(f"  Calculando... K = {K:.2f} ({i+1}/{len(K_VALUES_SWEEP)})")
            r, *_ = run_simulation_complete_graph(K, thetas, omegas)
            r_results.append(r.get())  # .get() mueve el resultado de GPU a CPU
    else:
        # Para redes sparse, convertir a formato CSR de CuPy
        A_scipy = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float32)
        A_gpu = cp.sparse.csr_matrix(A_scipy)
        degrees_gpu = cp.array(A_scipy.sum(axis=1).flatten(), dtype=cp.float32)
        
        for i, K in enumerate(K_VALUES_SWEEP):
            print(f"  Calculando... K = {K:.2f} ({i+1}/{len(K_VALUES_SWEEP)})")
            r, *_ = run_simulation(K, A_gpu, thetas, omegas, degrees_gpu)
            r_results.append(r.get())  # .get() mueve el resultado de GPU a CPU

    end_time = time.time()

    print(f"Análisis completado en {end_time - start_time:.2f} segundos.")

    return np.array(r_results)

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    # Generamos los datos aleatorios una sola vez y los movemos a la GPU
    G_scale_free, omegas, thetas = generate_random_network(seed=True)
    r_scale_free = run_sweep_analysis("Red Libre de Escala", G_scale_free, thetas, omegas)

    # --- Análisis para Grafo Completo ---
    print(f"Generando Grafo Completo (N={N})...")
    G_complete = nx.complete_graph(N)
    r_complete = run_sweep_analysis("Grafo Completo", G_complete, thetas, omegas)

    # --- Visualización Comparativa Final ---
    print("\nGenerando gráfico comparativo final...")
    plt.style.use('seaborn-v0_8-whitegrid') # Estilo de gráfico profesional
    plt.figure(figsize=(14, 8))

    plt.plot(K_VALUES_SWEEP, r_complete, "o-", label=f"Grafo Completo (N={N})", color="royalblue", markersize=8, linewidth=2.5)
    plt.plot(K_VALUES_SWEEP, r_scale_free, "s-", label=f"Red Libre de Escala (N={N})", color="crimson", markersize=8, linewidth=2.5)

    plt.xlabel("Fuerza de Acoplamiento (K)", fontsize=16)
    plt.ylabel("Parámetro de Orden (r)", fontsize=16)
    plt.title("Transición a la Sincronización en Redes Grandes", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()