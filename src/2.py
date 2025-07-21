# ==============================================================================
# PIPELINE DE ANÁLISIS AUTOMÁTICO Y DIRIGIDO
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
from utils import (
    generate_random_network, run_full_analysis
)

def run_sweep_and_find_kc(G, thetas, omegas):
    """
    Barrido optimizado que encuentra Kc y almacena solo estados necesarios.
    Utiliza las funciones de utils.py para evitar duplicación.
    """
    # Ejecutar análisis completo en una sola llamada
    results = run_full_analysis(G, thetas, omegas)

    # Extraer los resultados en el formato esperado
    return results['r_values'], results['kc_value'], results['key_states']

def find_elbow_point(k_values, inertias):
    """
    Encuentra el punto del codo usando el método de máxima curvatura.

    El codo se detecta como el punto donde la segunda derivada
    (curvatura) de la curva inercia vs k es máxima.

    Args:
        k_values: Array de valores de k
        inertias: Array de valores de inercia correspondientes

    Returns:
        k_optimal: Valor de k en el codo
    """
    if len(inertias) < 4:
        # No hay suficientes puntos para calcular segunda derivada
        return k_values[1] if len(k_values) > 1 else k_values[0]

    # Convertir a arrays numpy
    k_array = np.array(k_values)
    inertia_array = np.array(inertias)

    # Método 1: Detección de codo usando distancia a la línea recta
    # Crear línea recta desde primer punto al último punto
    p1 = np.array([k_array[0], inertia_array[0]])
    p2 = np.array([k_array[-1], inertia_array[-1]])

    # Calcular distancia de cada punto a la línea recta
    distances = []
    for i in range(len(k_array)):
        point = np.array([k_array[i], inertia_array[i]])
        # Distancia de punto a línea usando fórmula: |ax + by + c| / sqrt(a² + b²)
        distance = np.abs(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)
        distances.append(distance)

    # El codo está en el punto con máxima distancia a la línea
    elbow_idx = np.argmax(distances)

    # Método 2: Validación usando segunda derivada (si hay suficientes puntos)
    if len(inertias) >= 5:
        # Calcular segunda derivada numérica
        first_deriv = np.gradient(inertia_array, k_array)
        second_deriv = np.gradient(first_deriv, k_array)

        # El codo corresponde al máximo de la segunda derivada (máxima curvatura)
        # Solo considerar la primera mitad de los datos (el codo debería estar al principio)
        mid_point = len(second_deriv) // 2
        curvature_elbow_idx = np.argmax(np.abs(second_deriv[:mid_point]))

        # Si ambos métodos están cerca, usar el promedio
        if abs(elbow_idx - curvature_elbow_idx) <= 2:
            elbow_idx = int((elbow_idx + curvature_elbow_idx) / 2)

    return k_array[elbow_idx]

def analyze_hub_clusters(G, final_thetas_cpu, title, K_val):
    """
    Análisis de clustering mejorado enfocado en hubs con métricas adaptativas.
    """
    print(f"  Ejecutando Análisis de Clustering de Hubs para '{title}'...")

    # 1. Calcular parámetro de orden para determinar estado
    r_global = np.abs(np.mean(np.exp(1j * final_thetas_cpu)))

    # 2. Seleccionar solo hubs importantes (no todos los nodos)
    degrees = dict(G.degree())
    all_degrees = list(degrees.values())
    degree_threshold = np.percentile(all_degrees, 85)  # Top 15% como hubs
    hub_nodes = [n for n, d in degrees.items() if d >= degree_threshold]

    if len(hub_nodes) < 10:
        print(f"    Muy pocos hubs ({len(hub_nodes)}) para análisis significativo.")
        return

    # 3. Obtener features según estado de sincronización
    hub_indices = [i for i, node in enumerate(G.nodes()) if node in hub_nodes]
    hub_thetas = final_thetas_cpu[hub_indices]
    hub_degrees = np.array([degrees[node] for node in hub_nodes])

    # Selección adaptativa de features
    if r_global < 0.3:  # Desincronizado
        features = np.stack((np.cos(hub_thetas), np.sin(hub_thetas)), axis=1)
        analysis_type = "Clusters de Fase en Hubs"
        max_k = min(15, len(hub_nodes) // 3)
    elif r_global < 0.7:  # Parcialmente sincronizado
        # Combinar fase + grado normalizado
        norm_degrees = (hub_degrees - hub_degrees.min()) / (hub_degrees.max() - hub_degrees.min() + 1e-6)
        features = np.stack((np.cos(hub_thetas), np.sin(hub_thetas), norm_degrees), axis=1)
        analysis_type = "Clusters Hub-Dinámicos"
        max_k = min(12, len(hub_nodes) // 4)
    else:  # Sincronizado
        # Solo estructura, la fase es uniforme
        norm_degrees = (hub_degrees - hub_degrees.min()) / (hub_degrees.max() - hub_degrees.min() + 1e-6)
        # Calcular métricas estructurales simples
        clustering_coeffs = np.array([nx.clustering(G, node) for node in hub_nodes])
        features = np.stack((norm_degrees, clustering_coeffs), axis=1)
        analysis_type = "Clusters Estructurales de Hubs"
        max_k = min(8, len(hub_nodes) // 5)

    # 4. Calcular múltiples métricas de calidad
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    k_range = range(2, max_k + 1)  # Empezar desde 2
    inertias = []
    silhouette_scores = []
    db_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(features)

        inertias.append(kmeans.inertia_)

        # Silhouette score (1 = perfecto, -1 = malo)
        sil_score = silhouette_score(features, cluster_labels)
        silhouette_scores.append(sil_score)

        # Davies-Bouldin score (menor = mejor)
        db_score = davies_bouldin_score(features, cluster_labels)
        db_scores.append(db_score)

    # 5. Encontrar número óptimo usando verdadero método del codo
    optimal_k_elbow = find_elbow_point(k_range, inertias)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_db = k_range[np.argmin(db_scores)]

    # Decisión final basada en consenso de métodos
    # Priorizar el codo si es confiable, sino usar silhouette
    if len(inertias) >= 4:  # Suficientes puntos para detectar codo
        optimal_k = optimal_k_elbow
        selection_method = "elbow"
    else:
        optimal_k = optimal_k_silhouette
        selection_method = "silhouette"

    # Validar que el resultado sea razonable
    if optimal_k < 2:
        optimal_k = max(2, optimal_k_silhouette)
        selection_method = "silhouette_fallback"

    # 6. Visualización mejorada con múltiples métricas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Inercia (método del codo mejorado)
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=6, label='Inercia')

    # Mostrar diferentes puntos óptimos
    axes[0].axvline(x=optimal_k_elbow, color='red', linestyle='--', alpha=0.8,
                   label=f'Codo: k={optimal_k_elbow}')
    if optimal_k != optimal_k_elbow:
        axes[0].axvline(x=optimal_k, color='orange', linestyle=':', alpha=0.8,
                       label=f'Seleccionado: k={optimal_k}')

    # Visualizar la línea del método de distancia
    if len(inertias) >= 4:
        line_y = [inertias[0], inertias[-1]]
        line_x = [k_range[0], k_range[-1]]
        axes[0].plot(line_x, line_y, 'gray', alpha=0.5, linestyle=':', label='Línea base')

    axes[0].set_xlabel('Número de Clusters (k)')
    axes[0].set_ylabel('Inercia')
    axes[0].set_title(f'Método del Codo ({selection_method})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Silhouette score
    axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=6, label='Silhouette')
    axes[1].axvline(x=optimal_k_silhouette, color='green', linestyle='--', alpha=0.8,
                   label=f'Max Silhouette: k={optimal_k_silhouette}')
    if optimal_k != optimal_k_silhouette:
        axes[1].axvline(x=optimal_k, color='orange', linestyle=':', alpha=0.8,
                       label=f'Seleccionado: k={optimal_k}')
    axes[1].set_xlabel('Número de Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Calidad de Separación')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Davies-Bouldin score
    axes[2].plot(k_range, db_scores, 'ro-', linewidth=2, markersize=6, label='Davies-Bouldin')
    axes[2].axvline(x=optimal_k_db, color='red', linestyle='--', alpha=0.8,
                   label=f'Min DB: k={optimal_k_db}')
    if optimal_k != optimal_k_db:
        axes[2].axvline(x=optimal_k, color='orange', linestyle=':', alpha=0.8,
                       label=f'Seleccionado: k={optimal_k}')
    axes[2].set_xlabel('Número de Clusters (k)')
    axes[2].set_ylabel('Davies-Bouldin Score')
    axes[2].set_title('Compacidad vs Separación')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Título general
    fig.suptitle(f'{analysis_type}\nK = {K_val:.3f}, r = {r_global:.3f}, Hubs = {len(hub_nodes)}',
                fontsize=14, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    # 7. Interpretación automática
    print(f"    Análisis completado:")
    print(f"    - Hubs analizados: {len(hub_nodes)}/{len(G.nodes())} ({len(hub_nodes)/len(G.nodes())*100:.1f}%)")
    print(f"    - Estado del sistema: {analysis_type}")
    print(f"    - Número óptimo de clusters: {optimal_k}")
    print(f"    - Silhouette score óptimo: {silhouette_scores[optimal_k-2]:.3f}")
    print(f"    - Davies-Bouldin score óptimo: {db_scores[optimal_k-2]:.3f}")

# Parámetros de configuración para mapeo de colores
BASE_COLOR_RANGE = 1.0      # Rango máximo de hue para estado desincronizado (espectro completo)
COMPRESSION_FACTOR = 1.5    # Factor de compresión: qué tan rápido convergen los colores con r
MIN_COLOR_RANGE = 0.2       # Rango mínimo de hue para estado altamente sincronizado

def synchronization_aware_mapping(thetas, r):
    """
    Mapeo de colores consciente de la sincronización para redes de Kuramoto.

    Los colores se vuelven más similares a medida que aumenta la sincronización (r),
    cumpliendo con el requisito de que "cada vez que aumente r (orden) los nodos
    se coloreen de mejor forma más similar".

    Args:
        thetas: Array de fases en radianes
        r: Parámetro de orden de sincronización (0 = desorden, 1 = sincronización perfecta)

    Returns:
        Array de colores normalizados [0,1] para mapeo HSV

    Metodología:
        1. Calcula la fase media usando el parámetro de orden complejo
        2. Centra las fases relativas a la fase media
        3. Comprime el rango de colores basado en el nivel de sincronización
        4. Mapea al espacio HSV con rango comprimido

    Referencias:
        - Acebrón et al. (2005). The Kuramoto model: A simple paradigm for synchronization phenomena
        - Rodrigues et al. (2016). The Kuramoto model in complex networks
    """
    # 1. Calcular la fase media usando el parámetro de orden complejo
    complex_order_param = np.mean(np.exp(1j * thetas))
    mean_phase = np.angle(complex_order_param)

    # 2. Calcular fases relativas centradas en la fase media
    relative_phases = (thetas - mean_phase) % (2 * np.pi)

    # 3. Comprimir el rango de colores basado en la sincronización con función más suave
    # Función sigmoidal suave para transición gradual
    # Cuando r ≈ 0: color_range ≈ BASE_COLOR_RANGE (espectro completo)
    # Cuando r ≈ 1: color_range ≈ MIN_COLOR_RANGE (colores similares)

    # Transición sigmoidal más suave: 1 / (1 + exp(8*(r - 0.6)))
    sigmoid_factor = 1.0 / (1.0 + np.exp(8.0 * (r - 0.6)))
    color_range = BASE_COLOR_RANGE * sigmoid_factor + MIN_COLOR_RANGE * (1 - sigmoid_factor)

    # 4. Mapear a [0, 1] con rango comprimido
    # Normalizar las fases relativas al rango [0, color_range]
    normalized_phases = (relative_phases / (2 * np.pi)) * color_range

    # 5. Centrar los colores alrededor de la fase media normalizada
    mean_color = (mean_phase % (2 * np.pi)) / (2 * np.pi)
    final_colors = (mean_color + normalized_phases - color_range/2) % 1.0

    return final_colors

def spectral_kuramoto_layout(G, max_display_nodes=120):
    """
    Layout espectral científico estándar para redes de Kuramoto.

    Utiliza los eigenvectores del Laplaciano para revelar la estructura de sincronización:
    - Fiedler vector (v2): separación natural de comunidades
    - Tercer eigenvector (v3): estructura jerárquica secundaria
    - Relaciona directamente con los modos de sincronización del sistema Kuramoto

    Basado en:
    - Kuramoto (1984). Chemical Oscillations, Waves, and Turbulence
    - Arenas et al. (2008). Synchronization in complex networks
    - Dörfler & Bullo (2014). Synchronization in complex oscillator networks
    - Rodrigues et al. (2016). The Kuramoto model in complex networks
    """
    # 1. Tomar componente gigante y preparar para análisis espectral
    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc)
    print(f"    Componente gigante: {len(G_giant)} nodos, {G_giant.number_of_edges()} edges")

    # 2. Selección inteligente de nodos para análisis espectral
    # Priorizar nodos importantes pero mantener conectividad para el Laplaciano
    degrees_dict = dict(G_giant.degree())
    all_degrees = list(degrees_dict.values())

    # Clasificación estándar de hubs
    super_hub_threshold = np.percentile(all_degrees, 99)    # Top 1%
    major_hub_threshold = np.percentile(all_degrees, 95)    # Top 5%
    minor_hub_threshold = np.percentile(all_degrees, 85)    # Top 15%

    super_hubs = [n for n, d in degrees_dict.items() if d >= super_hub_threshold]
    major_hubs = [n for n, d in degrees_dict.items() if d >= major_hub_threshold and n not in super_hubs]
    minor_hubs = [n for n, d in degrees_dict.items() if d >= minor_hub_threshold and n not in super_hubs and n not in major_hubs]

    # Seleccionar nodos manteniendo conectividad para análisis espectral válido
    selected_nodes = set(super_hubs + major_hubs)

    # Agregar nodos de grado medio para mantener estructura conectada
    medium_degree_threshold = np.percentile(all_degrees, 70)
    medium_nodes = [n for n, d in degrees_dict.items() if d >= medium_degree_threshold and n not in selected_nodes]

    # Samplear manteniendo conectividad
    remaining_space = max_display_nodes - len(selected_nodes)
    if remaining_space > 0:
        # Priorizar nodos que mantengan la conectividad
        candidate_nodes = minor_hubs + medium_nodes
        if candidate_nodes:
            selected_additional = np.random.choice(candidate_nodes,
                                                 size=min(remaining_space, len(candidate_nodes)),
                                                 replace=False)
            selected_nodes.update(selected_additional)

    # Crear subgrafo manteniendo conectividad
    selected_nodes = list(selected_nodes)[:max_display_nodes]
    G_sub = G_giant.subgraph(selected_nodes)

    # Verificar que el subgrafo sea conectado (crítico para spectral layout)
    if not nx.is_connected(G_sub):
        print(f"    Subgrafo no conectado, tomando componente principal...")
        largest_sub_cc = max(nx.connected_components(G_sub), key=len)
        G_sub = G_sub.subgraph(largest_sub_cc)
        selected_nodes = list(G_sub.nodes())

    print(f"    Subgrafo para análisis espectral: {len(G_sub)} nodos, {G_sub.number_of_edges()} edges")

    # 3. ANÁLISIS ESPECTRAL DEL LAPLACIANO - CORAZÓN DEL MÉTODO
    print(f"    Computando eigenvectores del Laplaciano normalizado...")

    try:
        # Usar el Laplaciano normalizado (estándar para redes de Kuramoto)
        L = nx.normalized_laplacian_matrix(G_sub, nodelist=list(G_sub.nodes()))

        # Computar eigenvalues y eigenvectores
        # Los más pequeños corresponden a los modos de sincronización más lentos
        eigenvals, eigenvecs = np.linalg.eigh(L.toarray())

        # Ordenar por eigenvalue (ascendente)
        sort_indices = np.argsort(eigenvals)
        eigenvals = eigenvals[sort_indices]
        eigenvecs = eigenvecs[:, sort_indices]

        print(f"    Eigenvalues del Laplaciano: λ₁={eigenvals[0]:.6f}, λ₂={eigenvals[1]:.6f}, λ₃={eigenvals[2]:.6f}")

        # 4. POSICIONAMIENTO ESPECTRAL USANDO FIEDLER VECTOR
        # Fiedler vector (segundo eigenvector más pequeño) para coordenada X
        fiedler_vector = eigenvecs[:, 1]  # λ₂

        # Tercer eigenvector para coordenada Y
        third_eigenvec = eigenvecs[:, 2]  # λ₃

        # Crear posiciones usando proyección espectral
        pos = {}
        node_list = list(G_sub.nodes())

        for i, node in enumerate(node_list):
            x = fiedler_vector[i] * 5.0  # Escalar para mejor visualización
            y = third_eigenvec[i] * 5.0
            pos[node] = (x, y)

        print(f"    Layout espectral aplicado usando v₂ (Fiedler) y v₃")

        # 5. DETECCIÓN DE COMUNIDADES BASADA EN CORTE ESPECTRAL
        # El signo del Fiedler vector indica naturalmente la bipartición óptima
        fiedler_partition = fiedler_vector >= 0

        community_1 = [node_list[i] for i in range(len(node_list)) if fiedler_partition[i]]
        community_2 = [node_list[i] for i in range(len(node_list)) if not fiedler_partition[i]]

        # Para análisis más fino, usar el tercer eigenvector también
        third_partition = third_eigenvec >= 0

        # Crear 4 comunidades espectrales basadas en signos de v₂ y v₃
        spectral_communities = []
        spectral_communities.append([node_list[i] for i in range(len(node_list))
                                   if fiedler_partition[i] and third_partition[i]])      # (+,+)
        spectral_communities.append([node_list[i] for i in range(len(node_list))
                                   if fiedler_partition[i] and not third_partition[i]])  # (+,-)
        spectral_communities.append([node_list[i] for i in range(len(node_list))
                                   if not fiedler_partition[i] and third_partition[i]])  # (-,+)
        spectral_communities.append([node_list[i] for i in range(len(node_list))
                                   if not fiedler_partition[i] and not third_partition[i]])  # (-,-)

        # Filtrar comunidades vacías
        spectral_communities = [comm for comm in spectral_communities if len(comm) > 0]

        print(f"    Comunidades espectrales detectadas: {len(spectral_communities)}")
        for i, comm in enumerate(spectral_communities):
            print(f"      Comunidad {i+1}: {len(comm)} nodos")

    except Exception as e:
        print(f"    Error en análisis espectral: {e}")
        print(f"    Fallback a spectral layout estándar de NetworkX...")
        pos = nx.spectral_layout(G_sub, scale=5.0)
        spectral_communities = []
        eigenvals = []

    return G_sub, pos, {
        'super_hubs': super_hubs,
        'major_hubs': major_hubs,
        'minor_hubs': minor_hubs,
        'spectral_communities': spectral_communities,
        'eigenvalues': eigenvals[:10] if len(eigenvals) > 0 else [],  # Primeros 10 eigenvalues
        'fiedler_vector': fiedler_vector if 'fiedler_vector' in locals() else [],
        'total_selected': len(selected_nodes),
        'total_original': len(G_giant)
    }

def visualize_network_advanced(G, final_thetas_cpu, title, K, max_nodes=100):
    """
    Visualización refinada para redes de Kuramoto con mapeo absoluto de fases.

    Args:
        G: Grafo de la red
        final_thetas_cpu: Array de fases finales en CPU
        title: Título de la visualización
        K: Valor de acoplamiento
        max_nodes: Número máximo de nodos a visualizar
    """
    print(f"  Generando visualización refinada para '{title}'...")

    # 1. Selección más selectiva de nodos para evitar solapamiento
    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc)

    # Calcular k-core y degrees
    k_core_dict = nx.core_number(G_giant)
    max_k_core = max(k_core_dict.values())
    degrees_dict = dict(G_giant.degree())
    all_degrees = list(degrees_dict.values())

    # Selección más estricta para mejor visualización
    # Solo el top 2% de hubs y k-core alto
    degree_threshold = np.percentile(all_degrees, 98)
    hub_nodes = [n for n, d in degrees_dict.items() if d >= degree_threshold]
    high_kcore_nodes = [n for n, k in k_core_dict.items() if k >= max_k_core]

    # Combinar y limitar a menos nodos
    nodes_to_keep = list(set(high_kcore_nodes) | set(hub_nodes))

    if len(nodes_to_keep) > max_nodes:
        node_importance = [(n, k_core_dict[n] * degrees_dict[n]) for n in nodes_to_keep]
        node_importance.sort(key=lambda x: x[1], reverse=True)
        nodes_to_keep = [n for n, _ in node_importance[:max_nodes]]

    # Asegurar conectividad mínima
    if len(nodes_to_keep) < 50:
        degree_threshold = np.percentile(all_degrees, 95)
        additional_nodes = [n for n, d in degrees_dict.items() if d >= degree_threshold]
        nodes_to_keep = list(set(nodes_to_keep) | set(additional_nodes[:max_nodes]))

    G_sub = G_giant.subgraph(nodes_to_keep)

    # 2. Calcular parámetro de orden global
    r_global = np.abs(np.mean(np.exp(1j * final_thetas_cpu)))

    # 3. Configurar figura con layout mejorado (sin info de red)
    fig = plt.figure(figsize=(18, 12), facecolor='white')

    # Grid layout simplificado: red principal (izquierda), histograma grande (derecha)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], hspace=0.2, wspace=0.2)

    ax_main = fig.add_subplot(gs[0, 0])  # Red ocupa 2/3 del espacio
    ax_hist = fig.add_subplot(gs[0, 1])  # Histograma grande en toda la derecha

    # 4. Layout espectral para redes de Kuramoto (análisis del Laplaciano)
    print("    Calculando layout espectral basado en el Laplaciano...")
    try:
        G_hierarchical, pos, layout_info = spectral_kuramoto_layout(G_giant, max_display_nodes=max_nodes)
        # Actualizar G_sub para usar el grafo jerárquico
        G_sub = G_hierarchical

        # Recalcular índices y colores para el nuevo subgrafo
        node_indices = {node: i for i, node in enumerate(G.nodes())}
        subgraph_indices = [node_indices[n] for n in G_sub.nodes()]
        subgraph_thetas = final_thetas_cpu[subgraph_indices]

        print(f"    Layout espectral: {layout_info['total_selected']}/{layout_info['total_original']} nodos")
        print(f"    Super hubs: {len(layout_info['super_hubs'])}, Major hubs: {len(layout_info['major_hubs'])}, Minor hubs: {len(layout_info['minor_hubs'])}")
        print(f"    Comunidades espectrales: {len(layout_info['spectral_communities'])}")
        if len(layout_info['eigenvalues']) > 0:
            eigenvals_str = ', '.join([f"λ{i+1}={val:.4f}" for i, val in enumerate(layout_info['eigenvalues'][:5])])
            print(f"    Eigenvalues del Laplaciano: {eigenvals_str}")
    except Exception as e:
        print(f"    Fallback a spring layout: {e}")
        pos = nx.spring_layout(G_sub, k=3.0, iterations=150)
        layout_info = None

        # Para el fallback, calcular subgraph_thetas del G_sub original
        node_indices = {node: i for i, node in enumerate(G.nodes())}
        subgraph_indices = [node_indices[n] for n in G_sub.nodes()]
        subgraph_thetas = final_thetas_cpu[subgraph_indices]

    # 5. Calcular parámetro de sincronización local para el subgrafo
    r_subgraph = np.abs(np.mean(np.exp(1j * subgraph_thetas)))

    # 6. Mapeo de colores consciente de la sincronización
    node_colors = synchronization_aware_mapping(subgraph_thetas, r_subgraph)
    cmap = plt.cm.hsv  # Colormap circular estándar para fases

    # Debug: verificar rango de colores y fases
    print(f"    Debug mapeo: r_global={r_global:.3f}, r_subgraph={r_subgraph:.3f}, modo=sync-aware")
    print(f"    Debug fases raw: min={subgraph_thetas.min():.3f}, max={subgraph_thetas.max():.3f}, rango={subgraph_thetas.max()-subgraph_thetas.min():.3f}")

    # Calcular fase media y rango de colores para debug
    mean_phase = np.angle(np.mean(np.exp(1j * subgraph_thetas)))
    sigmoid_factor = 1.0 / (1.0 + np.exp(8.0 * (r_subgraph - 0.6)))
    color_range = BASE_COLOR_RANGE * sigmoid_factor + MIN_COLOR_RANGE * (1 - sigmoid_factor)

    print(f"    Debug sync-aware: fase_media={mean_phase:.3f}, sigmoid_factor={sigmoid_factor:.3f}")
    print(f"    Debug color_range={color_range:.3f} (objetivo: {BASE_COLOR_RANGE:.1f}→{MIN_COLOR_RANGE:.1f})")
    print(f"    Debug colores: min={node_colors.min():.3f}, max={node_colors.max():.3f}, rango={node_colors.max()-node_colors.min():.3f}")

    # Debug adicional para verificar coherencia
    phases_std = np.std(subgraph_thetas)
    print(f"    Debug coherencia: σ_fases={phases_std:.3f}, K={K:.3f}, r_global/r_sub={r_global:.3f}/{r_subgraph:.3f}")

    # Etiqueta del colorbar actualizada
    color_label = f'Fase Relativa (r={r_subgraph:.2f})'

    # Ajustar transparencia de edges según sincronización
    if r_global < 0.3:  # Desincronizado
        edge_alpha = 0.2
    elif r_global < 0.7:  # Parcialmente sincronizado
        edge_alpha = 0.3
    else:  # Sincronizado
        edge_alpha = 0.4

    # 6. Tamaños más diferenciados
    degrees = np.array([G_sub.degree(n) for n in G_sub.nodes()])
    min_size = 30
    max_size = 800  # Reducido para evitar solapamiento

    if len(degrees) > 0:
        # Escala cuadrática para mejor diferenciación
        norm_degrees = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-6)
        node_sizes = min_size + (max_size - min_size) * (norm_degrees ** 0.5)
    else:
        node_sizes = [min_size] * len(G_sub.nodes())

    # 7. Edges jerárquicos más informativos
    if layout_info is not None:
        # Edges entre super hubs (más gruesos - backbone de la red)
        super_edges = [(u, v) for u, v in G_sub.edges()
                      if u in layout_info['super_hubs'] and v in layout_info['super_hubs']]

        # Edges de super hubs a major hubs (medianos)
        hub_super_edges = [(u, v) for u, v in G_sub.edges()
                          if (u in layout_info['super_hubs'] and v in layout_info['major_hubs']) or
                             (v in layout_info['super_hubs'] and u in layout_info['major_hubs'])]

        # Edges entre major hubs (finos)
        hub_edges = [(u, v) for u, v in G_sub.edges()
                    if (u in layout_info['major_hubs'] or u in layout_info['minor_hubs']) and
                       (v in layout_info['major_hubs'] or v in layout_info['minor_hubs'])]

        # Dibujar por capas con diferentes estilos (jerarquía científica)
        if super_edges:
            nx.draw_networkx_edges(G_sub, pos, edgelist=super_edges,
                                 width=1.5, alpha=edge_alpha*1.5, edge_color='#222222', ax=ax_main)
        if hub_super_edges:
            nx.draw_networkx_edges(G_sub, pos, edgelist=hub_super_edges,
                                 width=1.0, alpha=edge_alpha*1.2, edge_color='#444444', ax=ax_main)
        if hub_edges:
            nx.draw_networkx_edges(G_sub, pos, edgelist=hub_edges,
                                 width=0.6, alpha=edge_alpha*0.8, edge_color='#666666', ax=ax_main)
    else:
        # Fallback: edges tradicionales
        degree_threshold_edges = np.percentile(degrees, 75) if len(degrees) > 10 else 0
        important_nodes = [n for n in G_sub.nodes() if G_sub.degree(n) >= degree_threshold_edges]
        important_edges = [(u, v) for u, v in G_sub.edges()
                          if u in important_nodes or v in important_nodes]
        nx.draw_networkx_edges(G_sub, pos, edgelist=important_edges,
                              width=0.5, alpha=edge_alpha, edge_color='#333333', ax=ax_main)

    # 8. Nodos ordenados por tamaño (pequeños primero, grandes encima)
    node_order = np.argsort(node_sizes)
    ordered_nodes = [list(G_sub.nodes())[i] for i in node_order]
    ordered_colors = [node_colors[i] for i in node_order]
    ordered_sizes = [node_sizes[i] for i in node_order]

    node_collection = nx.draw_networkx_nodes(G_sub, pos, nodelist=ordered_nodes,
                                           node_color=ordered_colors,
                                           node_size=ordered_sizes,
                                           cmap=cmap,
                                           alpha=0.85,
                                           linewidths=0.5,
                                           edgecolors='black',
                                           ax=ax_main)

    # 9. Solo nodos principales sin halos

    # 10. Colorbar más visible
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_main, fraction=0.08, pad=0.02, shrink=0.8)
    cbar.set_label(color_label, fontsize=11)

    # 11. Histograma grande y mejorado (sin información de red)
    phases_deg = np.degrees(subgraph_thetas)
    n_bins = 30  # Más bins para mejor resolución en histograma grande

    # Crear histograma con mejor estética
    counts, bins, patches = ax_hist.hist(phases_deg, bins=n_bins, alpha=0.8, color='steelblue',
                                        edgecolor='black', linewidth=0.8)

    # Colorear las barras según la fase para mejor visualización
    for i, patch in enumerate(patches):
        # Asignar color basado en la fase (centro del bin)
        phase_center = (bins[i] + bins[i+1]) / 2
        hue = (phase_center / 360.0) % 1.0
        patch.set_facecolor(plt.cm.hsv(hue))
        patch.set_alpha(0.7)

    # Configuración del histograma
    ax_hist.set_xlabel('Fase (°)', fontsize=14, fontweight='bold')
    ax_hist.set_ylabel('Frecuencia', fontsize=14, fontweight='bold')
    ax_hist.set_xlim(0, 360)
    ax_hist.grid(True, alpha=0.3, linestyle='--')
    ax_hist.set_title('Distribución de Fases', fontsize=16, fontweight='bold', pad=20)
    ax_hist.tick_params(labelsize=12)

    # Añadir estadísticas importantes al histograma
    mean_phase_deg = np.degrees(np.angle(np.mean(np.exp(1j * subgraph_thetas))))
    std_phase_deg = np.degrees(np.std(subgraph_thetas))

    # Línea vertical para la fase media
    ax_hist.axvline(x=mean_phase_deg % 360, color='red', linestyle='--', linewidth=2,
                   label=f'Fase Media: {mean_phase_deg % 360:.1f}°')

    # Texto con estadísticas
    stats_text = f'r = {r_subgraph:.3f}\nσ = {std_phase_deg:.1f}°\nN = {len(subgraph_thetas)}'
    ax_hist.text(0.05, 0.95, stats_text, transform=ax_hist.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 13. Título principal mejorado con documentación de mapeo
    avg_degree = np.mean(degrees)
    std_phase = np.std(subgraph_thetas)

    title_text = (f"{title}\n"
                 f"K = {K:.3f} | r = {r_global:.3f} | σ_φ = {std_phase:.2f}\n"
                 f"Nodos: {len(G_sub)}/{len(G)} | ⟨k⟩ = {avg_degree:.1f}")

    ax_main.set_title(title_text, fontsize=14, pad=15)
    ax_main.axis('off')

    # Ajustar límites para mejor visualización
    pos_array = np.array(list(pos.values()))
    margin = 0.1
    ax_main.set_xlim(pos_array[:, 0].min() - margin, pos_array[:, 0].max() + margin)
    ax_main.set_ylim(pos_array[:, 1].min() - margin, pos_array[:, 1].max() + margin)

    plt.tight_layout()
    plt.show()

# --- 3. SCRIPT PRINCIPAL DE EJECUCIÓN ---
if __name__ == "__main__":
    # --- PASO 0: Generación de la Red y Datos Iniciales ---
    G, omegas, thetas = generate_random_network(seed=True)

    # --- PASO 1: Barrido Optimizado y Búsqueda de Kc ---
    start_time = time.time()

    r_values, kc_calculated, key_states = run_sweep_and_find_kc(
        G, thetas, omegas
    )

    end_time = time.time()

    print(f"\nBarrido completado en {end_time - start_time:.2f} segundos.")

    if kc_calculated is None:
        print("No se pudo determinar un Kc. El sistema no cruzó el umbral.")
    else:
        print(f"Umbral Crítico (Kc) = {kc_calculated:.4f}")

        # --- PASO 2: Análisis Detallado de Estados Clave ---
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

                # 1. Análisis de clustering de hubs
                analyze_hub_clusters(G, thetas_cpu, title, K_val)

                # 2. Visualización de la red
                visualize_network_advanced(G, thetas_cpu, title, K_val)
            else:
                print(f"\nNo se encontraron datos para: {title}")