# ==============================================================================
# SCRIPT OPTIMIZADO: PIPELINE COMPLETO DE ANÁLISIS CON OPTIMIZACIONES
# ==============================================================================
# Propósito: Analizar frecuencias efectivas y visualización filtrada de hubs.
# Refactorizado para usar las utilidades optimizadas de utils.py.
#
# Optimizaciones implementadas:
# 1. Uso de utils.py para simulaciones optimizadas
# 2. Cálculo eficiente de frecuencias efectivas
# 3. Visualización filtrada (solo hubs principales)
# 4. Integración con funciones estándar del proyecto
# ==============================================================================

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from utils import (
    generate_random_network, run_full_analysis,
    K_VALUES_SWEEP, run_simulation, T_MEASURE, R_THRESHOLD
)

# Parámetros de visualización optimizada
TOP_HUBS_PERCENT = 0.05  # Solo mostrar top 0.5% de hubs
MIN_NODES_TO_SHOW = 20  # Mínimo de nodos a mostrar
MAX_NODES_TO_SHOW = 100  # Máximo de nodos para claridad

def run_optimized_sweep_and_get_dynamics(G, thetas_0, omegas):
    """
    Versión optimizada del barrido usando utils.py para obtener dinámicas.
    Calcula frecuencias efectivas para análisis adicional.
    """

    # Usar run_full_analysis de utils.py para obtener estados clave
    results = run_full_analysis(G, thetas_0, omegas)
    r_values = results['r_values']
    kc_value = results['kc_value']
    A_sparse = results['A_sparse']
    degrees = results['degrees']

    # Calcular frecuencias efectivas para puntos clave
    effective_freqs_dict = {}

    if kc_value is not None:
        # Calcular frecuencias efectivas para los estados clave
        key_K_values = [
            (0.5 * kc_value, "desync"),
            (1.0 * kc_value, "partial"),
            (1.8 * kc_value, "sync")
        ]

        for K, state_name in key_K_values:
            print(f"  Calculando frecuencias efectivas para estado {state_name} (K={K:.3f})...")

            # Simular y calcular frecuencias efectivas
            thetas_start = thetas_0.copy()
            r, thetas_final, _ = run_simulation(K, A_sparse, thetas_start, omegas, degrees)

            # Calcular frecuencias efectivas como diferencia de fases normalizada
            effective_freqs = (thetas_final - thetas_start) / T_MEASURE
            effective_freqs_dict[state_name] = {
                'K': K,
                'r': r.get(),
                'thetas': thetas_final,
                'effective_freqs': effective_freqs
            }

    return r_values, kc_value, effective_freqs_dict

def get_filtered_nodes_for_visualization(G, top_percent=0.1, min_nodes=50, max_nodes=MAX_NODES_TO_SHOW):
    """Filtra nodos para visualización con clasificación jerárquica de hubs"""
    degrees = dict(G.degree())
    all_degrees = list(degrees.values())

    # Clasificación jerárquica de hubs
    super_hub_threshold = np.percentile(all_degrees, 99)  # Top 1%
    major_hub_threshold = np.percentile(all_degrees, 95)  # Top 5%
    minor_hub_threshold = np.percentile(all_degrees, 85)  # Top 15%

    # Identificar diferentes tipos de hubs
    super_hubs = [n for n, d in degrees.items() if d >= super_hub_threshold]
    major_hubs = [n for n, d in degrees.items() if major_hub_threshold <= d < super_hub_threshold]
    minor_hubs = [n for n, d in degrees.items() if minor_hub_threshold <= d < major_hub_threshold]

    # Combinar todos los hubs según el porcentaje solicitado
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    num_to_show = max(int(len(sorted_nodes) * top_percent), min_nodes)
    # Limitar al máximo permitido para claridad
    num_to_show = min(num_to_show, max_nodes)
    top_nodes = [node for node, degree in sorted_nodes[:num_to_show]]

    # Crear subgrafo con los nodos seleccionados
    G_sub = G.subgraph(top_nodes).copy()

    # No añadir bordes virtuales para reducir el desorden visual
    # Solo mostrar conexiones reales entre los hubs seleccionados

    # Almacenar información de clasificación en el grafo
    for node in G_sub.nodes():
        if node in super_hubs:
            G_sub.nodes[node]['hub_type'] = 'super'
        elif node in major_hubs:
            G_sub.nodes[node]['hub_type'] = 'major'
        elif node in minor_hubs:
            G_sub.nodes[node]['hub_type'] = 'minor'
        else:
            G_sub.nodes[node]['hub_type'] = 'regular'

    print(f"    Hubs clasificados: {len(super_hubs)} super-hubs, {len(major_hubs)} major-hubs, {len(minor_hubs)} minor-hubs")

    return G_sub

def visualize_optimized_final_state(G, effective_freqs_gpu, title, K, r_global):
    """Visualización optimizada con filtrado de nodos"""
    print(f"  Generando visualización optimizada para '{title}'...")

    # Filtrar grafo para mostrar solo hubs principales
    G_filtered = get_filtered_nodes_for_visualization(G, TOP_HUBS_PERCENT, MIN_NODES_TO_SHOW, MAX_NODES_TO_SHOW)
    print(f"    Visualizando {len(G_filtered.nodes())} nodos principales de {len(G.nodes())} totales")

    # Mapear frecuencias efectivas a nodos filtrados
    effective_freqs_cpu = cp.asnumpy(effective_freqs_gpu)
    node_to_freq = {i: effective_freqs_cpu[i] for i in range(len(effective_freqs_cpu))}
    filtered_freqs = [node_to_freq[node] for node in G_filtered.nodes()]

    # Propiedades visuales optimizadas
    degrees = np.array([d for n, d in G_filtered.degree()])
    # Escalar tamaños para 30 nodos - más grandes y diferenciados
    node_sizes = (degrees**1.5) * 3 + 100  # Tamaño mayor para mejor visibilidad

    node_colors = filtered_freqs
    v_max = np.percentile(np.abs(node_colors), 95)

    # Layout optimizado para menos nodos
    plt.figure(figsize=(14, 14))
    print("    Calculando layout optimizado (Kamada-Kawai)...")
    # Kamada-Kawai produce mejores resultados para redes con hubs
    pos = nx.kamada_kawai_layout(G_filtered)

    print("    Dibujando red filtrada con estilos graduados...")
    nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors, node_size=node_sizes,
                          cmap=plt.cm.coolwarm, vmin=-v_max, vmax=v_max, alpha=0.8)

    # Dibujar todos los bordes con un estilo simple y uniforme
    # Solo variar el grosor según la importancia de los nodos conectados
    edge_widths = []
    for (u, v) in G_filtered.edges():
        u_degree = G_filtered.degree(u)
        v_degree = G_filtered.degree(v)
        # Grosor proporcional al grado mínimo de los nodos conectados
        min_degree = min(u_degree, v_degree)
        width = 0.5 + (min_degree / 10.0)  # Escalar grosor
        edge_widths.append(width)

    # Dibujar todos los bordes en gris con transparencia
    nx.draw_networkx_edges(G_filtered, pos, width=edge_widths,
                          alpha=0.3, edge_color='grey')

    # Agregar etiquetas solo para los top 5 super-hubs
    label_nodes = {}
    degrees_dict = dict(G_filtered.degree())

    # Solo etiquetar los top 5 nodos por grado
    top_5_nodes = sorted(G_filtered.nodes(),
                        key=lambda x: degrees_dict[x], reverse=True)[:5]

    for node in top_5_nodes:
        # Mostrar solo el ID del nodo
        label_nodes[node] = str(node)

    nx.draw_networkx_labels(G_filtered, pos, label_nodes, font_size=8, font_color='black',
                           font_weight='bold', bbox=dict(boxstyle="round,pad=0.3",
                                                        facecolor="white", alpha=0.7))

    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-v_max, vmax=v_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.7)
    cbar.set_label('Frecuencia Efectiva (rad/s)', rotation=270, labelpad=25, fontsize=14)

    plt.title(f"{title}\nK={K:.2f}, Orden Global (r) = {r_global:.3f}\n({len(G_filtered.nodes())} hubs principales de {len(G.nodes())} nodos)",
              fontsize=20)

    # No incluir leyenda para simplificar la visualización

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- 3. SCRIPT PRINCIPAL OPTIMIZADO ---
if __name__ == "__main__":
    # --- PASO 0: Generación de la Red y Datos Iniciales ---
    G, omegas, thetas = generate_random_network(seed=True)

    print(f"\nEstadísticas de la red:")
    print(f"- Nodos: {len(G.nodes())}")
    print(f"- Enlaces: {len(G.edges())}")
    print(f"- Grado promedio: {2*len(G.edges())/len(G.nodes()):.2f}")

    # --- PASO 1: Barrido Optimizado usando utils.py ---
    start_time = time.time()

    r_values, kc_calculated, effective_freqs_dict = run_optimized_sweep_and_get_dynamics(
        G, thetas, omegas)

    end_time = time.time()
    print(f"\nAnálisis completado en {end_time - start_time:.2f} segundos.")

    # --- PASO 2: Visualización de Resultados ---
    if kc_calculated is not None:
        print(f"\nUmbral Crítico (Kc) calculado: {kc_calculated:.4f}")

        # --- PASO 3: Análisis Visual de Estados Clave ---
        state_titles = {
            "desync": "Estado Desincronizado",
            "partial": "Estado de Sincronización Parcial",
            "sync": "Estado de Sincronización Global"
        }

        for state_key, title in state_titles.items():
            if state_key in effective_freqs_dict:
                print(f"\n{'='*60}")
                print(f"ANÁLISIS VISUAL OPTIMIZADO: {title.upper()}")
                print(f"{'='*60}")

                state_data = effective_freqs_dict[state_key]
                K_val = state_data['K']
                r_global = state_data['r']
                effective_freqs_gpu = state_data['effective_freqs']

                print(f"K = {K_val:.3f}, r = {r_global:.3f}")

                visualize_optimized_final_state(G, effective_freqs_gpu,
                                              title, K_val, r_global)
            else:
                print(f"\nNo se encontraron datos para: {title}")
    else:
        print("No se pudo determinar un Kc con el threshold actual.")