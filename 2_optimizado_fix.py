# ==============================================================================
# SCRIPT FINAL OPTIMIZADO: PIPELINE DE ANÁLISIS AUTOMÁTICO Y DIRIGIDO
# ==============================================================================
# Propósito: Realizar un análisis completo de una red grande, donde los puntos
# de visualización se eligen automáticamente en relación con el umbral
# crítico (Kc) calculado para esa misma red.
#
# Metodología:
# 1. Barrido de K para obtener la curva r vs. K y los estados de fase finales.
# 2. Cálculo automático de Kc a partir de los datos del barrido.
# 3. Análisis visual y de clustering en tres puntos clave relativos a Kc.
#
# Optimizaciones implementadas:
# 1. Uso de matrices sparse para redes libres de escala
# 2. Almacenamiento selectivo de estados (solo los necesarios)
# 3. Parámetros de simulación optimizados
# 4. Visualización más eficiente
# ==============================================================================

import cupy as cp
import cupyx.scipy.sparse as sparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from tqdm import tqdm

# --- 1. PARÁMETROS GLOBALES OPTIMIZADOS ---
N = 10000
M_SCALE_FREE = 3
OMEGA_MU = 0.0
OMEGA_SIGMA = 0.5
T_TRANSIENT = 5.0    # Fase transitoria
T_MEASURE = 5.0      # Fase de medición
DT = 0.01            # Paso de tiempo más grande pero estable
K_VALUES_SWEEP = np.linspace(0, 5, 30)  # Menos puntos para encontrar Kc más rápido

# --- 2. FUNCIONES DE SIMULACIÓN Y ANÁLISIS OPTIMIZADAS ---

def kuramoto_odes_gpu_sparse(thetas, K, A_sparse, omegas, degrees):
    """
    Versión optimizada para matrices sparse que evita crear matriz NxN.
    """
    sin_thetas = cp.sin(thetas)
    cos_thetas = cp.cos(thetas)

    # Multiplicación matriz sparse × vector
    sum_sin = A_sparse @ sin_thetas
    sum_cos = A_sparse @ cos_thetas

    # Aplicar identidad trigonométrica correcta
    interactions = cos_thetas * sum_sin - sin_thetas * sum_cos

    # Evitar división por cero
    safe_degrees = cp.maximum(degrees, 1.0)
    dthetas_dt = omegas + (K / safe_degrees) * interactions

    return dthetas_dt

def run_simulation(K, A_sparse, thetas_0, omegas, degrees):
    """
    Simulación optimizada con fase transitoria y medición separadas.
    """
    num_steps_transient = int(T_TRANSIENT / DT)
    num_steps_measure = int(T_MEASURE / DT)
    thetas_current = thetas_0.copy()

    # Fase transitoria (sin medición)
    for _ in range(num_steps_transient):
        dthetas = kuramoto_odes_gpu_sparse(thetas_current, K, A_sparse, omegas, degrees)
        thetas_current += dthetas * DT

    # Fase de medición - calculamos r promedio
    r_values = []
    for step in range(num_steps_measure):
        dthetas = kuramoto_odes_gpu_sparse(thetas_current, K, A_sparse, omegas, degrees)
        thetas_current += dthetas * DT

        # Calcular r cada 10 pasos
        if step % 10 == 0:
            exp_thetas = cp.exp(1j * thetas_current)
            r = cp.abs(cp.mean(exp_thetas))
            r_values.append(r)

    # Devolver r promedio y el estado final
    r_final = cp.mean(cp.array(r_values))
    return r_final, thetas_current

def run_sweep_and_find_kc(G, thetas_0, omegas, threshold=0.5):
    """
    Barrido optimizado que encuentra Kc y almacena solo estados necesarios.
    """
    print("Iniciando barrido optimizado para encontrar Kc...")

    # Preparar matriz sparse en GPU
    A_scipy = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float32)
    A_sparse = sparse.csr_matrix(A_scipy)
    degrees = cp.array(A_scipy.sum(axis=1).flatten(), dtype=cp.float32)

    r_values = []
    kc_found = False
    kc_value = None

    # Diccionario para almacenar solo estados clave
    key_states = {}

    # Primera pasada: encontrar Kc
    for i, K in enumerate(tqdm(K_VALUES_SWEEP, desc="Barrido de K")):
        r, final_thetas = run_simulation(K, A_sparse, thetas_0, omegas, degrees)
        r_cpu = r.get()
        r_values.append(r_cpu)

        # Detectar transición a sincronización
        if not kc_found and r_cpu > threshold:
            kc_found = True
            kc_value = K
            print(f"\n¡Kc encontrado! Kc ≈ {kc_value:.3f}")

            # Almacenar estados alrededor de Kc
            if i > 0:  # Estado justo antes de Kc
                key_states["pre_kc"] = (K_VALUES_SWEEP[i-1], i-1)
            key_states["at_kc"] = (K, i)
            key_states["partial_sync_thetas"] = final_thetas

    # Segunda pasada: simular puntos específicos basados en Kc
    if kc_found:
        print("\nSimulando estados específicos...")

        # Estado desincronizado (0.5 * Kc)
        K_desync = 0.5 * kc_value
        print(f"  - Estado desincronizado (K = {K_desync:.3f})...")
        r_desync, thetas_desync = run_simulation(K_desync, A_sparse, thetas_0, omegas, degrees)
        key_states["desync"] = (K_desync, -1)  # -1 indica que no está en el barrido original
        key_states["desync_thetas"] = thetas_desync

        # Estado sincronizado (3.0 * Kc)
        K_sync = min(3.0 * kc_value, 5.0)
        print(f"  - Estado sincronizado (K = {K_sync:.3f})...")
        r_sync, thetas_sync = run_simulation(K_sync, A_sparse, thetas_0, omegas, degrees)
        key_states["sync"] = (K_sync, -1)
        key_states["sync_thetas"] = thetas_sync

    return np.array(r_values), kc_value, key_states

def visualize_transition_curve(k_values, r_values, kc, key_states):
    """
    Visualiza la curva de transición con puntos clave marcados usando valores reales de r.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, r_values, 'b-', linewidth=2, label='r vs K')

    # Marcar Kc
    if kc is not None:
        plt.axvline(x=kc, color='red', linestyle='--', label=f'Kc = {kc:.3f}')

    # Calcular valores reales de r para los puntos de análisis
    # Necesitamos simular estos puntos para obtener sus r reales
    if 'desync' in key_states and 'desync_thetas' in key_states:
        K_desync = key_states['desync'][0]
        thetas_desync = key_states['desync_thetas']
        r_desync = cp.abs(cp.mean(cp.exp(1j * thetas_desync))).get()
        plt.scatter(K_desync, r_desync, color='green', s=100, zorder=5,
                   label=f'Desincronizado (K={K_desync:.2f}, r={r_desync:.3f})')

    if 'at_kc' in key_states:
        k_val, idx = key_states['at_kc']
        if idx >= 0:  # Si está en el barrido original
            r_at_kc = r_values[idx]
        else:  # Si fue simulado por separado
            if 'partial_sync_thetas' in key_states:
                thetas_kc = key_states['partial_sync_thetas']
                r_at_kc = cp.abs(cp.mean(cp.exp(1j * thetas_kc))).get()
            else:
                r_at_kc = 0.5  # Valor por defecto en Kc
        plt.scatter(k_val, r_at_kc, color='red', s=100,
                   zorder=5, label=f'En Kc (K={k_val:.2f}, r={r_at_kc:.3f})')

    if 'sync' in key_states and 'sync_thetas' in key_states:
        K_sync = key_states['sync'][0]
        thetas_sync = key_states['sync_thetas']
        r_sync = cp.abs(cp.mean(cp.exp(1j * thetas_sync))).get()
        plt.scatter(K_sync, r_sync, color='purple', s=100, zorder=5,
                   label=f'Sincronizado (K={K_sync:.2f}, r={r_sync:.3f})')

    plt.xlabel('Fuerza de Acoplamiento (K)', fontsize=14)
    plt.ylabel('Parámetro de Orden (r)', fontsize=14)
    plt.title('Transición a la Sincronización', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

def elbow_method(final_thetas_cpu, title, max_k=8):
    """
    Método del codo optimizado con menos clusters máximos.
    """
    print(f"  Ejecutando Método del Codo para '{title}'...")
    coords = np.stack((np.cos(final_thetas_cpu), np.sin(final_thetas_cpu)), axis=1)

    # Submuestrear si hay muchos nodos para acelerar
    if len(coords) > 1000:
        indices = np.random.choice(len(coords), 1000, replace=False)
        coords_sample = coords[indices]
    else:
        coords_sample = coords

    inertia = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(coords_sample)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inercia', fontsize=12)
    plt.title(f'Método del Codo para {title}', fontsize=14)
    plt.xticks(k_range)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_network(G, final_thetas_cpu, title, K, degree_threshold=3, max_nodes=1500):
    """
    Visualización optimizada con límite de nodos y layout más rápido.
    """
    print(f"  Generando visualización optimizada para '{title}'...")

    # Obtener el componente gigante (componente conectado más grande)
    # Esto evita nodos aislados o débilmente conectados
    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc)

    # Filtrar nodos por grado dentro del componente gigante
    nodes_to_keep = [n for n, d in G_giant.degree() if d >= degree_threshold]

    # Si aún hay demasiados nodos, seleccionar los más conectados
    if len(nodes_to_keep) > max_nodes:
        node_degrees = [(n, G_giant.degree(n)) for n in nodes_to_keep]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        nodes_to_keep = [n for n, _ in node_degrees[:max_nodes]]

    # Si hay muy pocos nodos con el threshold actual, incluir más
    if len(nodes_to_keep) < 100:
        all_nodes = list(G_giant.nodes())
        node_degrees = [(n, G_giant.degree(n)) for n in all_nodes]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        nodes_to_keep = [n for n, _ in node_degrees[:min(len(all_nodes), max_nodes)]]

    G_sub = G_giant.subgraph(nodes_to_keep)

    # Obtener fases de los nodos seleccionados
    node_indices = {node: i for i, node in enumerate(G.nodes())}
    subgraph_indices = [node_indices[n] for n in G_sub.nodes()]
    subgraph_thetas = final_thetas_cpu[subgraph_indices]

    # Colores basados en fase
    node_colors = (subgraph_thetas % (2 * np.pi)) / (2 * np.pi)

    # Tamaños basados en grado (ajustados para mejor visualización)
    degrees = np.array([G_sub.degree(n) for n in G_sub.nodes()])
    # Escalar tamaños logarítmicamente para evitar nodos demasiado grandes
    node_sizes = 20 + np.log1p(degrees) * 30

    plt.figure(figsize=(15, 15))

    # Layout mejorado con más control
    # Usar Fruchterman-Reingold para mejor distribución espacial
    pos = nx.fruchterman_reingold_layout(G_sub,
                                        k=0.5,  # Distancia ideal entre nodos
                                        iterations=100,  # Más iteraciones para mejor calidad
                                        seed=42)

    # Dibujar la red con mejor estética
    # Primero los edges con transparencia
    nx.draw_networkx_edges(G_sub, pos, width=0.1, alpha=0.5, edge_color='gray')

    nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, cmap=plt.cm.hsv,
                          node_size=node_sizes)

    # Calcular y mostrar parámetro de orden
    r_global = np.abs(np.mean(np.exp(1j * final_thetas_cpu)))

    # Información adicional sobre la visualización
    avg_degree = np.mean(degrees)
    plt.title(f"{title}\nK={K:.2f}, r = {r_global:.3f}\n" +
             f"Nodos mostrados: {len(G_sub)}/{len(G)} | Grado promedio: {avg_degree:.1f}",
             fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    # --- PASO 0: Generación de la Red y Datos Iniciales ---
    print(f"Generando Red Libre de Escala (N={N}, m={M_SCALE_FREE})...")
    G = nx.barabasi_albert_graph(N, M_SCALE_FREE, seed=42)

    # Generar condiciones iniciales en GPU
    cp.random.seed(42)
    omegas_gpu = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N, dtype=cp.float32)
    thetas_0_gpu = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)

    # --- PASO 1: Barrido Optimizado y Búsqueda de Kc ---
    start_time = time.time()
    r_values, kc_calculated, key_states = run_sweep_and_find_kc(
        G, thetas_0_gpu, omegas_gpu
    )
    end_time = time.time()

    print(f"\nBarrido completado en {end_time - start_time:.2f} segundos.")

    if kc_calculated is None:
        print("No se pudo determinar un Kc. El sistema no cruzó el umbral.")
    else:
        print(f"Umbral Crítico (Kc) = {kc_calculated:.4f}")

        # --- PASO 2: Visualizar Curva de Transición ---
        visualize_transition_curve(K_VALUES_SWEEP, r_values, kc_calculated, key_states)

        # --- PASO 3: Análisis Detallado de Estados Clave ---
        analysis_states = [
            ("Estado Desincronizado", "desync_thetas"),
            ("Estado de Sincronización Parcial", "partial_sync_thetas"),
            ("Estado de Sincronización Global", "sync_thetas")
        ]

        for title, state_key in analysis_states:
            if state_key in key_states:
                print(f"\n{'='*60}")
                print(f"ANÁLISIS: {title.upper()}")
                print(f"{'='*60}")

                # Obtener datos del estado
                if state_key == "desync_thetas":
                    K_val = key_states.get("desync", (0, 0))[0]
                elif state_key == "partial_sync_thetas":
                    K_val = kc_calculated
                else:  # sync_thetas
                    K_val = key_states.get("sync", (0, 0))[0]

                thetas_cpu = cp.asnumpy(key_states[state_key])

                # 1. Método del codo
                elbow_method(thetas_cpu, title)

                # 2. Visualización de la red
                visualize_network(G, thetas_cpu, title, K_val)
            else:
                print(f"\nNo se encontraron datos para: {title}")