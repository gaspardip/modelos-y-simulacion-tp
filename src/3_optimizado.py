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
    generate_random_network, run_full_analysis, find_kc,
    K_VALUES_SWEEP, prepare_sparse_matrix, run_simulation
)

# Parámetros de visualización optimizada
TOP_HUBS_PERCENT = 0.1  # Solo mostrar top 10% de hubs
MIN_NODES_TO_SHOW = 50  # Mínimo de nodos a mostrar

def run_optimized_sweep_and_get_dynamics(G, thetas_0, omegas):
    """
    Versión optimizada del barrido usando utils.py para obtener dinámicas.
    Calcula frecuencias efectivas para análisis adicional.
    """
    print("Iniciando barrido optimizado usando utils.py...")
    
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
            from utils import T_MEASURE
            effective_freqs = (thetas_final - thetas_start) / T_MEASURE
            effective_freqs_dict[state_name] = {
                'K': K,
                'r': r.get(),
                'thetas': thetas_final,
                'effective_freqs': effective_freqs
            }
    
    return r_values, kc_value, effective_freqs_dict

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

def plot_optimization_comparison(r_values, kc_calculated):
    """Gráfico de curva r vs K con información de optimización"""
    plt.figure(figsize=(12, 8))
    plt.plot(K_VALUES_SWEEP, r_values, 'b-', linewidth=2, label='Parámetro de orden r')

    if kc_calculated is not None:
        plt.axvline(x=kc_calculated, color='red', linestyle='--', linewidth=2,
                   label=f'Kc crítico = {kc_calculated:.3f}')
        
        # Marcar los puntos de análisis
        analysis_points = [
            (0.5 * kc_calculated, 'Desincronizado'),
            (1.0 * kc_calculated, 'En Kc'),
            (1.8 * kc_calculated, 'Sincronizado')
        ]
        
        colors = ['green', 'red', 'purple']
        for i, (K_point, label) in enumerate(analysis_points):
            plt.axvline(x=K_point, color=colors[i], linestyle=':', alpha=0.7, 
                       label=f'{label} (K={K_point:.2f})')

    from utils import R_THRESHOLD
    plt.axhline(y=R_THRESHOLD, color='gray', linestyle=':', alpha=0.7, 
               label=f'Threshold = {R_THRESHOLD}')

    plt.xlabel('Acoplamiento K', fontsize=14)
    plt.ylabel('Parámetro de orden r', fontsize=14)
    plt.title('Curva de Sincronización Optimizada\n(Usando utils.py + Análisis de Frecuencias)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0, 5)
    plt.ylim(0, 1)
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

        # Mostrar curva de sincronización
        plot_optimization_comparison(r_values, kc_calculated)

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
        plot_optimization_comparison(r_values, None)