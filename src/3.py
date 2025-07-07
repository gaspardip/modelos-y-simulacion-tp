# ==============================================================================
# SCRIPT FINAL (DE PUBLICACIÓN): PIPELINE COMPLETO DE ANÁLISIS
# ==============================================================================
# Propósito: Realizar un análisis completo y automático de la sincronización,
# desde el barrido cuantitativo hasta la visualización cualitativa final.
#
# Metodología:
# 1. Barrido de K para obtener la curva r vs. K y los estados dinámicos.
# 2. Cálculo automático de Kc a partir de los datos del barrido.
# 3. Visualización final en tres regímenes clave (relativos a Kc) donde:
#    - El TAMAÑO del nodo es una función no lineal de su GRADO para resaltar los hubs.
#    - El COLOR del nodo representa su FRECUENCIA EFECTIVA.
# ==============================================================================

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# --- 1. PARÁMETROS GLOBALES ---
N = 10000
M_SCALE_FREE = 3
OMEGA_MU = 0.0
OMEGA_SIGMA = 0.5
T_SIMULATION = 20.0
DT = 0.01
K_VALUES_SWEEP = np.linspace(0, 5, 50)

# --- 2. FUNCIONES DE SIMULACIÓN Y ANÁLISIS ---

def kuramoto_odes_gpu(thetas, K, A, omegas, degrees):
    phase_diffs = thetas - thetas[:, cp.newaxis]
    interactions = A * cp.sin(phase_diffs)
    interaction_sum = cp.sum(interactions, axis=1)
    return omegas + (K / degrees) * interaction_sum

def run_full_sweep_and_get_dynamics(G, thetas_0, omegas):
    print("Iniciando barrido exploratorio para encontrar Kc y guardar estados...")
    r_results, final_thetas_list, effective_freqs_list = [], [], []
    A_gpu = cp.asarray(nx.to_numpy_array(G), dtype=cp.float32)
    degrees_gpu = cp.sum(A_gpu, axis=1); degrees_gpu[degrees_gpu == 0] = 1

    for K in tqdm(K_VALUES_SWEEP, desc="Barrido de K"):
        num_steps = int(T_SIMULATION / DT)
        thetas_current = thetas_0.copy()
        thetas_midpoint = None

        for step in range(num_steps):
            if step == num_steps // 2:
                thetas_midpoint = thetas_current.copy()
            dthetas = kuramoto_odes_gpu(thetas_current, K, A_gpu, omegas, degrees_gpu)
            thetas_current += dthetas * DT

        effective_freqs = (thetas_current - thetas_midpoint) / (T_SIMULATION / 2)
        effective_freqs_list.append(effective_freqs)

        final_thetas_list.append(cp.mod(thetas_current, 2 * np.pi))
        r_results.append(cp.abs(cp.mean(cp.exp(1j * final_thetas_list[-1]))).get())

    return np.array(r_results), final_thetas_list, effective_freqs_list

def find_kc_from_sweep(k_values, r_values, threshold=0.5):
    indices = np.where(r_values > threshold)[0]
    return k_values[indices[0]] if len(indices) > 0 else None

def visualize_final_state(G, effective_freqs_gpu, title, K, r_global):
    print(f"  Generando visualización final para '{title}'...")

    # --- Propiedades Visuales Mejoradas ---
    degrees = np.array([d for n, d in G.degree()])
    # Mapeo no lineal para exagerar el tamaño de los hubs
    node_sizes = degrees**1.5 + 10

    effective_freqs_cpu = cp.asnumpy(effective_freqs_gpu)
    node_colors = effective_freqs_cpu
    v_max = np.percentile(np.abs(node_colors), 99)

    # --- Generar el Gráfico ---
    plt.figure(figsize=(18, 18))
    print("    Calculando layout del grafo...")
    pos = nx.spring_layout(G, seed=42, iterations=50, k=0.8)

    print("    Dibujando la red...")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                                   cmap=plt.cm.coolwarm, vmin=-v_max, vmax=v_max, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.5, edge_color='grey')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-v_max, vmax=v_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.7)
    cbar.set_label('Frecuencia Efectiva (rad/s)', rotation=270, labelpad=25, fontsize=16)

    plt.title(f"{title}\nK={K:.2f}, Orden Global (r) = {r_global:.3f}", fontsize=28)
    plt.box(False)
    fig = plt.gca()
    fig.axes.get_xaxis().set_ticks([])
    fig.axes.get_yaxis().set_ticks([])
    plt.show()

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    # --- PASO 0: Generación de la Red y Datos Iniciales ---
    print(f"Generando Red Libre de Escala (N={N})...")
    G_visual = nx.barabasi_albert_graph(N, M_SCALE_FREE, seed=42)
    cp.random.seed(42)
    omegas_gpu = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N, dtype=cp.float32)
    thetas_0_gpu = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)

    # --- PASO 1: Barrido Exploratorio ---
    start_time = time.time()
    r_values, final_thetas_list, effective_freqs_list = run_full_sweep_and_get_dynamics(G_visual, thetas_0_gpu, omegas_gpu)
    end_time = time.time()
    print(f"Barrido exploratorio completado en {end_time - start_time:.2f} segundos.")

    # --- PASO 2: Cálculo de Kc y Análisis Dirigido ---
    Kc_calculated = find_kc_from_sweep(K_VALUES_SWEEP, r_values)

    if Kc_calculated is not None:
        print(f"\nUmbral Crítico (Kc) calculado para esta red: {Kc_calculated:.4f}")
        analysis_targets = {
            "Estado Desincronizado": Kc_calculated * 0.5,
            "Estado de Sincronización Parcial": Kc_calculated,
            "Estado de Sincronización Global": Kc_calculated * 3.0
        }

        for title, target_k in analysis_targets.items():
            print(f"\n======================================================")
            print(f"INICIANDO ANÁLISIS VISUAL PARA: {title.upper()}")
            print(f"======================================================")

            closest_k_idx = np.argmin(np.abs(K_VALUES_SWEEP - target_k))
            K_to_analyze = K_VALUES_SWEEP[closest_k_idx]
            r_global = r_values[closest_k_idx]
            effective_freqs_to_analyze = effective_freqs_list[closest_k_idx]

            visualize_final_state(G_visual, effective_freqs_to_analyze, title, K_to_analyze, r_global)
    else:
        print("No se pudo determinar un Kc.")