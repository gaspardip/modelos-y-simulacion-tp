# ==============================================================================
# SCRIPT FINAL: PIPELINE DE ANÁLISIS AUTOMÁTICO Y DIRIGIDO
# ==============================================================================
# Propósito: Realizar un análisis completo de una red grande, donde los puntos
# de visualización se eligen automáticamente en relación con el umbral
# crítico (Kc) calculado para esa misma red.
#
# Metodología:
# 1. Barrido de K para obtener la curva r vs. K y los estados de fase finales.
# 2. Cálculo automático de Kc a partir de los datos del barrido.
# 3. Análisis visual y de clustering en tres puntos clave relativos a Kc.
# ==============================================================================

import cupy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from tqdm import tqdm

# --- 1. PARÁMETROS GLOBALES ---
N = 10000
M_SCALE_FREE = 3
OMEGA_MU = 0.0
OMEGA_SIGMA = 0.5
T_SIMULATION = 20.0
DT = 0.01
K_VALUES_SWEEP = np.linspace(0, 5, 50) # Barrido para encontrar Kc

# --- 2. FUNCIONES DE SIMULACIÓN Y ANÁLISIS ---

def kuramoto_odes_gpu(thetas, K, A, omegas, degrees):
    phase_diffs = thetas - thetas[:, cp.newaxis]
    interactions = A * cp.sin(phase_diffs)
    interaction_sum = cp.sum(interactions, axis=1)
    dthetas_dt = omegas + (K / degrees) * interaction_sum
    return dthetas_dt

def run_full_sweep_and_get_states(G, thetas_0, omegas):
    """
    Realiza un barrido de K y devuelve tanto los 'r' como los estados de fase finales.
    """
    print("Iniciando barrido exploratorio para encontrar Kc y guardar estados...")
    r_results = []
    final_thetas_list = []

    A_gpu = cp.asarray(nx.to_numpy_array(G), dtype=cp.float32)
    degrees_gpu = cp.sum(A_gpu, axis=1)
    degrees_gpu[degrees_gpu == 0] = 1

    for K in tqdm(K_VALUES_SWEEP, desc="Barrido de K"):
        num_steps = int(T_SIMULATION / DT)
        thetas_current = thetas_0.copy()
        for _ in range(num_steps):
            dthetas = kuramoto_odes_gpu(thetas_current, K, A_gpu, omegas, degrees_gpu)
            thetas_current += dthetas * DT
            # Envolvemos las fases en cada paso para que se mantengan en [0, 2*pi]
            thetas_current = cp.mod(thetas_current, 2 * cp.pi)

        final_thetas_list.append(thetas_current)
        r_results.append(cp.abs(cp.mean(cp.exp(1j * thetas_current))).get())

    return np.array(r_results), final_thetas_list

def find_kc_from_sweep(k_values, r_values, threshold=0.5):
    indices_above_threshold = np.where(r_values > threshold)[0]
    return k_values[indices_above_threshold[0]] if len(indices_above_threshold) > 0 else None

def elbow_method_for_phases(final_thetas_cpu, title, max_k=10):
    print(f"  Ejecutando Método del Codo para '{title}'...")
    coords = np.stack((np.cos(final_thetas_cpu), np.sin(final_thetas_cpu)), axis=1)
    inertia = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(coords)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Número de Clusters (k)'); plt.ylabel('Inercia')
    plt.title(f'Método del Codo para {title}'); plt.xticks(k_range); plt.grid(True)
    plt.show()

def visualize_filtered_network(G, final_thetas_cpu, title, K, degree_threshold=10):
    print(f"  Generando visualización filtrada para '{title}'...")
    nodes_to_keep = [n for n, d in G.degree() if d >= degree_threshold]
    G_sub = G.subgraph(nodes_to_keep)
    node_indices = {node: i for i, node in enumerate(G.nodes())}
    subgraph_indices = [node_indices[n] for n in G_sub.nodes()]
    subgraph_thetas = final_thetas_cpu[subgraph_indices]
    node_colors = (subgraph_thetas % (2 * np.pi)) / (2 * np.pi)
    degrees = [G_sub.degree(n) for n in G_sub.nodes()]
    node_sizes = [d * 20 for d in degrees]
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G_sub, seed=42, iterations=100)
    nx.draw(G_sub, pos, node_color=node_colors, cmap=plt.cm.hsv,
            with_labels=False, node_size=node_sizes, width=0.2, edge_color='lightgray')
    r_global = np.abs(np.mean(np.exp(1j * final_thetas_cpu)))
    plt.title(f"{title}\nK={K:.2f}, Orden Global (r) = {r_global:.3f}", fontsize=24)
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
    r_values, final_thetas_list = run_full_sweep_and_get_states(G_visual, thetas_0_gpu, omegas_gpu)
    end_time = time.time()
    print(f"Barrido exploratorio completado en {end_time - start_time:.2f} segundos.")

    # --- PASO 2: Cálculo de Kc ---
    Kc_calculated = find_kc_from_sweep(K_VALUES_SWEEP, r_values)

    if Kc_calculated is None:
        print("No se pudo determinar un Kc. El sistema no cruzó el umbral de 0.5.")
    else:
        print(f"\nUmbral Crítico (Kc) calculado para esta red: {Kc_calculated:.4f}")

        # --- PASO 3: Análisis Dirigido ---
        # Definimos los puntos de análisis relativos al Kc calculado
        analysis_targets = {
            "Estado Desincronizado": Kc_calculated * 0.5,
            "Estado de Sincronización Parcial": Kc_calculated,
            "Estado de Sincronización Global": Kc_calculated * 3.0
        }

        for title, target_k in analysis_targets.items():
            print(f"\n======================================================")
            print(f"INICIANDO ANÁLISIS PARA: {title.upper()}")
            print(f"======================================================")

            # Encontramos el K más cercano del barrido y sus fases pre-calculadas
            closest_k_idx = np.argmin(np.abs(K_VALUES_SWEEP - target_k))
            K_to_analyze = K_VALUES_SWEEP[closest_k_idx]
            thetas_to_analyze = final_thetas_list[closest_k_idx]

            print(f"Analizando el estado para K={K_to_analyze:.2f} (cercano al objetivo de {target_k:.2f})")

            final_thetas_cpu = cp.asnumpy(thetas_to_analyze)

            # 1. Encontrar k óptimo
            elbow_method_for_phases(final_thetas_cpu, title)

            # 2. Visualizar la red filtrada
            visualize_filtered_network(G_visual, final_thetas_cpu, title, K_to_analyze)