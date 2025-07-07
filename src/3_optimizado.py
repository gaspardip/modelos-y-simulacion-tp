# ==============================================================================
# SCRIPT OPTIMIZADO: PIPELINE COMPLETO DE ANÁLISIS CON OPTIMIZACIONES
# ==============================================================================
# Optimizaciones implementadas:
# 1. Matrices sparse para redes libres de escala
# 2. Cálculo eficiente de frecuencias sin almacenamiento intermedio
# 3. Separación de fase transitoria y de medición
# 4. Visualización filtrada (solo hubs principales)
# 5. Eliminación de cálculos innecesarios
# ==============================================================================

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from cupyx.scipy import sparse
from utils import *

# Parámetros de visualización optimizada
TOP_HUBS_PERCENT = 0.1  # Solo mostrar top 10% de hubs
MIN_NODES_TO_SHOW = 50  # Mínimo de nodos a mostrar


def run_optimized_sweep_and_get_dynamics(G, thetas_0, omegas):
    """Versión optimizada del barrido con separación de fases"""
    print("Iniciando barrido optimizado con matrices sparse...")
    r_results, final_thetas_list, effective_freqs_list = [], [], []

    # Crear matriz sparse
    adj_matrix = nx.to_scipy_sparse_array(G, format='csr')
    A_sparse = sparse.csr_matrix(adj_matrix, dtype=cp.float32)

    degrees_gpu = cp.array([d for n, d in G.degree()], dtype=cp.float32)
    degrees_gpu[degrees_gpu == 0] = 1

    num_steps_transient = int(T_TRANSIENT / DT)
    num_steps_measurement = int(T_MEASURE / DT)

    for K in tqdm(K_VALUES_SWEEP, desc="Barrido optimizado de K"):
        thetas_current = thetas_0.copy()

        # FASE 1: Transitoria (sin guardar estados)
        for step in range(num_steps_transient):
            dthetas = kuramoto_odes(thetas_current, K, A_sparse, omegas, degrees_gpu)
            thetas_current += dthetas * DT

        # FASE 2: Medición (calcular frecuencias directamente)
        thetas_start = thetas_current.copy()

        for step in range(num_steps_measurement):
            dthetas = kuramoto_odes(thetas_current, K, A_sparse, omegas, degrees_gpu)
            thetas_current += dthetas * DT

        # Calcular frecuencias efectivas directamente
        effective_freqs = (thetas_current - thetas_start) / T_MEASURE
        effective_freqs_list.append(effective_freqs)

        # No necesitamos módulo para calcular r
        final_thetas_list.append(thetas_current)
        r_results.append(cp.abs(cp.mean(cp.exp(1j * thetas_current))).get())

    return np.array(r_results), final_thetas_list, effective_freqs_list

def find_kc_from_sweep(k_values, r_values, threshold=0.5):
    """Encuentra Kc basado en el threshold de sincronización"""
    indices = np.where(r_values > threshold)[0]
    return k_values[indices[0]] if len(indices) > 0 else None

def get_filtered_nodes_for_visualization(G, top_percent=0.1, min_nodes=50):
    """Filtra nodos para visualización: solo hubs principales"""
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    num_to_show = max(int(len(sorted_nodes) * top_percent), min_nodes)
    top_nodes = [node for node, degree in sorted_nodes[:num_to_show]]

    return G.subgraph(top_nodes).copy()

def visualize_optimized_final_state(G, effective_freqs_gpu, title, K, r_global):
    """Visualización optimizada con filtrado de nodos"""
    print(f"  Generando visualización optimizada para '{title}'...")

    # Filtrar grafo para mostrar solo hubs principales
    G_filtered = get_filtered_nodes_for_visualization(G, TOP_HUBS_PERCENT, MIN_NODES_TO_SHOW)
    print(f"    Visualizando {len(G_filtered.nodes())} nodos principales de {len(G.nodes())} totales")

    # Mapear frecuencias efectivas a nodos filtrados
    effective_freqs_cpu = cp.asnumpy(effective_freqs_gpu)
    node_to_freq = {i: effective_freqs_cpu[i] for i in range(len(effective_freqs_cpu))}
    filtered_freqs = [node_to_freq[node] for node in G_filtered.nodes()]

    # Propiedades visuales optimizadas
    degrees = np.array([d for n, d in G_filtered.degree()])
    node_sizes = degrees**1.2 + 20  # Tamaño ajustado para menos nodos

    node_colors = filtered_freqs
    v_max = np.percentile(np.abs(node_colors), 95)

    # Layout optimizado para menos nodos
    plt.figure(figsize=(14, 14))
    print("    Calculando layout optimizado...")
    pos = nx.spring_layout(G_filtered, seed=42, iterations=100, k=1.5)

    print("    Dibujando red filtrada...")
    nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors, node_size=node_sizes,
                          cmap=plt.cm.coolwarm, vmin=-v_max, vmax=v_max, alpha=0.8)
    nx.draw_networkx_edges(G_filtered, pos, width=0.3, alpha=0.6, edge_color='grey')

    # Agregar etiquetas para nodos principales (top 10)
    top_10_nodes = sorted(G_filtered.degree(), key=lambda x: x[1], reverse=True)[:10]
    top_10_dict = {node: str(node) for node, _ in top_10_nodes}
    nx.draw_networkx_labels(G_filtered, pos, top_10_dict, font_size=8, font_color='black')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-v_max, vmax=v_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.7)
    cbar.set_label('Frecuencia Efectiva (rad/s)', rotation=270, labelpad=25, fontsize=14)

    plt.title(f"{title}\nK={K:.2f}, Orden Global (r) = {r_global:.3f}\n({len(G_filtered.nodes())} hubs principales de {len(G.nodes())} nodos)",
              fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_optimization_comparison():
    """Gráfico de curva r vs K con información de optimización"""
    plt.figure(figsize=(12, 8))
    plt.plot(K_VALUES_SWEEP, r_values, 'b-', linewidth=2, label='Parámetro de orden r')

    if Kc_calculated is not None:
        plt.axvline(x=Kc_calculated, color='red', linestyle='--', linewidth=2,
                   label=f'Kc crítico = {Kc_calculated:.3f}')
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='Threshold = 0.5')

    plt.xlabel('Acoplamiento K', fontsize=14)
    plt.ylabel('Parámetro de orden r', fontsize=14)
    plt.title('Curva de Sincronización Optimizada\n(Matrices Sparse + Fases Separadas)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0, 5)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# --- 3. SCRIPT PRINCIPAL OPTIMIZADO ---
if __name__ == "__main__":
    # --- PASO 0: Generación de la Red y Datos Iniciales ---
    print(f"Generando Red Libre de Escala Optimizada (N={N})...")
    G_visual = nx.barabasi_albert_graph(N, M_SCALE_FREE, seed=42)

    # Estadísticas de la red
    print(f"Red generada: {len(G_visual.nodes())} nodos, {len(G_visual.edges())} enlaces")
    print(f"Densidad de enlaces: {len(G_visual.edges())}/{N*(N-1)/2:.1%}")

    cp.random.seed(42)
    omegas_gpu = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N, dtype=cp.float32)
    thetas_0_gpu = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)

    # --- PASO 1: Barrido Exploratorio Optimizado ---
    start_time = time.time()
    r_values, final_thetas_list, effective_freqs_list = run_optimized_sweep_and_get_dynamics(
        G_visual, thetas_0_gpu, omegas_gpu)
    end_time = time.time()
    print(f"Barrido optimizado completado en {end_time - start_time:.2f} segundos.")

    # --- PASO 2: Cálculo de Kc y Análisis Dirigido ---
    Kc_calculated = find_kc_from_sweep(K_VALUES_SWEEP, r_values)

    if Kc_calculated is not None:
        print(f"\nUmbral Crítico (Kc) calculado: {Kc_calculated:.4f}")

        # Mostrar curva de sincronización
        plot_optimization_comparison()

        analysis_targets = {
            "Estado Desincronizado": Kc_calculated * 0.5,
            "Estado de Sincronización Parcial": Kc_calculated,
            "Estado de Sincronización Global": Kc_calculated * 3.0
        }

        for title, target_k in analysis_targets.items():
            print(f"\n======================================================")
            print(f"ANÁLISIS VISUAL OPTIMIZADO: {title.upper()}")
            print(f"======================================================")

            closest_k_idx = np.argmin(np.abs(K_VALUES_SWEEP - target_k))
            K_to_analyze = K_VALUES_SWEEP[closest_k_idx]
            r_global = r_values[closest_k_idx]
            effective_freqs_to_analyze = effective_freqs_list[closest_k_idx]

            visualize_optimized_final_state(G_visual, effective_freqs_to_analyze,
                                          title, K_to_analyze, r_global)
    else:
        print("No se pudo determinar un Kc con el threshold actual.")
        plot_optimization_comparison()