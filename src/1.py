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
from utils import N, K_VALUES_SWEEP, generate_random_network, prepare_sparse_matrix, batch_sweep

def run_sweep_analysis(network_type, G, thetas, omegas):
    """
    Realiza un barrido de K para una red dada usando el nuevo sistema RK4 batch.
    """
    print(f"======================================================")
    print(f"INICIANDO BARRIDO DE K PARA: {network_type.upper()} (N={G.number_of_nodes()})")
    print(f"======================================================")

    start_time = time.time()

    # Preparar datos para simulación batch
    A_sparse, degrees = prepare_sparse_matrix(G, quiet=False)

    print(f"Usando GPU-accelerated RK4 batch simulation")
    print(f"Simulando {len(K_VALUES_SWEEP)} valores de K simultáneamente...")

    # Single batch simulation for all K values - MASSIVE speedup!
    r_results_gpu = batch_sweep(A_sparse, thetas, omegas, degrees, quiet=False)
    r_results = r_results_gpu.get()  # Transfer to CPU

    end_time = time.time()

    print(f"Análisis completado en {end_time - start_time:.2f} segundos.")
    print(f"Speedup: ~{len(K_VALUES_SWEEP)*15/(end_time - start_time):.1f}x vs sequential simulation")

    return r_results

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    script_start = time.time()
    print(f"🚀 INICIANDO ANÁLISIS COMPARATIVO (N={N})")
    
    # Generamos los datos aleatorios una sola vez y los movemos a la GPU
    print("1. Generando red scale-free...")
    gen_start = time.time()
    G_scale_free, omegas, thetas = generate_random_network(seed=42)
    print(f"   Completado en {time.time() - gen_start:.2f}s")
    
    print("2. Análisis de red scale-free...")
    r_scale_free = run_sweep_analysis("Red Libre de Escala", G_scale_free, thetas, omegas)

    # --- Análisis para Grafo Completo ---
    print("3. Generando grafo completo...")
    gen_start = time.time()
    G_complete = nx.complete_graph(N)
    print(f"   Completado en {time.time() - gen_start:.2f}s")
    
    print("4. Análisis de grafo completo...")
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
    
    # Save instead of show to avoid display issues
    plt.savefig("kuramoto_comparison.png", dpi=150, bbox_inches='tight')
    print("Gráfico guardado como: kuramoto_comparison.png")
    plt.close()  # Free memory
    
    total_time = time.time() - script_start
    print(f"\n✅ ANÁLISIS COMPLETO: {total_time:.2f}s total")
    print(f"   Scale-free max r: {np.max(r_scale_free):.3f}")
    print(f"   Complete max r: {np.max(r_complete):.3f}")
    print(f"   GPU batch processing enabled: {len(K_VALUES_SWEEP)} K-values simultáneos")