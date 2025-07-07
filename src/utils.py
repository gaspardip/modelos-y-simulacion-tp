import cupy as cp
import numpy as np
import networkx as nx
from tqdm import tqdm
from cupyx.scipy import sparse

SEED = 42

# Parámetros de la Red
N = 10000
M_SCALE_FREE = 3

# Parámetros de la Dinámica
OMEGA_MU = 0.0
OMEGA_SIGMA = 0.5
T_TRANSIENT = 5.0    # Tiempo transitorio (descartado)
T_MEASURE = 5.0      # Tiempo de medición para calcular r
DT = 0.05            # Paso del tiempo para el integrador RK4

K_VALUES_SWEEP = np.linspace(0, 5, 50)
R_THRESHOLD = 0.4    # Umbral para definir la sincronización (ajustado para scale-free)

# Debug: Para redes scale-free, el threshold puede ser más bajo debido a heterogeneidad
# Threshold más bajo permite detectar Kc cuando r≈0.4 en lugar de r≈0.5

def kuramoto_odes(thetas, K, A_sparse, omegas, degrees):
    """
    Calcula la derivada de las fases en la GPU usando operaciones sparse.
    Evita crear la matriz completa NxN de phase_diffs.
    """
    # Para redes sparse, usar multiplicación matriz-vector es más eficiente
    sin_thetas = cp.sin(thetas)
    cos_thetas = cp.cos(thetas)

    # Calcular las sumas ponderadas usando la matriz de adyacencia
    sum_sin = A_sparse @ sin_thetas
    sum_cos = A_sparse @ cos_thetas

    # Calcular la derivada usando la identidad:
    # sum(sin(theta_j - theta_i)) = cos(theta_i)*sum(sin(theta_j)) - sin(theta_i)*sum(cos(theta_j))
    interactions = cp.cos(thetas) * sum_sin - cp.sin(thetas) * sum_cos

    # Evitar división por cero
    safe_degrees = cp.maximum(degrees, 1.0)
    dthetas_dt = omegas + (K / safe_degrees) * interactions

    return dthetas_dt

def rk4_step(thetas, dt, K, A_sparse, omegas, degrees):
    """
    Realiza un único paso de integración usando el método RK4 en la GPU.
    """
    dt_half = 0.5 * dt
    dt_six  = dt / 6.0

    # k1: Derivada al inicio del intervalo
    k1 = kuramoto_odes(thetas, K, A_sparse, omegas, degrees)

    # k2: Derivada en el punto medio, usando k1
    k2 = kuramoto_odes(thetas + dt_half * k1, K, A_sparse, omegas, degrees)

    # k3: Derivada en el punto medio, usando k2
    k3 = kuramoto_odes(thetas + dt_half * k2, K, A_sparse, omegas, degrees)

    # k4: Derivada al final del intervalo, usando k3
    k4 = kuramoto_odes(thetas + k3 * dt, K, A_sparse, omegas, degrees)

    # Actualización final con el promedio ponderado de las derivadas
    thetas += dt_six * (k1 + 2*k2 + 2*k3 + k4)

    # Normalizar fases al rango [0, 2π] para evitar problemas numéricos
    two_pi = thetas.dtype.type(2.0) * cp.pi
    cp.remainder(thetas, two_pi, out=thetas)

    return thetas

def run_simulation(K, A_sparse, thetas_0, omegas, degrees):
    """
    Simula con RK4 y devuelve el 'r' promedio.
    Incluye fase transitoria y medición del parámetro de orden.
    """
    num_steps_transient = int(T_TRANSIENT / DT)
    num_steps_measure = int(T_MEASURE / DT)
    thetas_current = thetas_0.copy()

    # Fase transitoria: Dejamos que el sistema "olvide" sus condiciones iniciales.
    # El integrador RK4 asegura que el camino hacia el equilibrio es preciso.
    for _ in range(num_steps_transient):
        thetas_current = rk4_step(thetas_current, DT, K, A_sparse, omegas, degrees)

    # Fase de medición: Ahora que el sistema está en su estado estacionario,
    # medimos su comportamiento de forma robusta.
    r_values = []
    for i in range(num_steps_measure):
        thetas_current = rk4_step(thetas_current, DT, K, A_sparse, omegas, degrees)

        # Calcular r cada 10 pasos para promediar
        if i % 10 == 0:
            exp_thetas = cp.exp(1j * thetas_current)
            r = cp.abs(cp.mean(exp_thetas))
            r_values.append(r)

    # Promedio de r en el estado estacionario
    r_final = cp.mean(cp.array(r_values))

    return r_final, thetas_current, r_values

def generate_random_network(seed = True):
    print(f"Generando Red Libre de Escala (N={N}, m={M_SCALE_FREE})...")

    if seed:
        cp.random.seed(SEED)
        G = nx.barabasi_albert_graph(N, M_SCALE_FREE, seed=SEED)
    else:
        G = nx.barabasi_albert_graph(N, M_SCALE_FREE)

    omegas = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N, dtype=cp.float32)
    thetas = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)

    return G, omegas, thetas

def sweep_analysis(G, thetas, omegas, store_thetas_at_kc=False, return_sparse=False):
    """
    Realiza un barrido de K y opcionalmente almacena los estados de las fases.

    Args:
        G: Grafo de la red
        thetas: Fases iniciales
        omegas: Frecuencias naturales
        store_thetas_at_kc: Si True, detecta Kc y almacena las fases en puntos clave
        return_sparse: Si True, retorna también la matriz sparse y degrees

    Returns:
        r_results: Lista de valores r para cada K
        key_states: Diccionario con estados clave (solo si store_thetas_at_kc=True)
        A_sparse, degrees: Matriz sparse y degrees (solo si return_sparse=True)
    """
    print("Iniciando barrido optimizado para encontrar Kc...")

    # Preparar matriz sparse en GPU
    A_sparse, degrees = prepare_sparse_matrix(G)

    r_results = []
    key_states = {} if store_thetas_at_kc else None
    kc_found = False
    kc_index = None

    for i, K in enumerate(tqdm(K_VALUES_SWEEP, desc="Barrido de K")):
        r, final_thetas, _ = run_simulation(K, A_sparse, thetas, omegas, degrees)
        r_cpu = r.get()
        r_results.append(r_cpu)

        # Si estamos almacenando estados y encontramos Kc
        if store_thetas_at_kc and not kc_found and r_cpu > R_THRESHOLD:
            kc_found = True
            kc_index = i
            kc_value = K
            print(f"\n¡Kc encontrado! Kc ≈ {K:.3f}")
            key_states["partial_sync_thetas"] = final_thetas
            key_states["kc_index"] = kc_index
            key_states["kc_value"] = kc_value

    # Retornar según las opciones solicitadas
    results = [r_results]
    if store_thetas_at_kc:
        results.append(key_states)

    if return_sparse:
        results.extend([A_sparse, degrees])

    return tuple(results) if len(results) > 1 else results[0]

def prepare_sparse_matrix(G):
    """
    Prepara la matriz sparse y los grados para simulación en GPU.

    Returns:
        A_sparse: Matriz de adyacencia sparse en GPU
        degrees: Array de grados de cada nodo en GPU
    """
    A_scipy = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float32)
    A_sparse = sparse.csr_matrix(A_scipy)
    degrees = cp.array(A_scipy.sum(axis=1).flatten(), dtype=cp.float32)
    return A_sparse, degrees

def find_kc(r_results):
    """Encuentra Kc basado en el threshold de sincronización"""
    indices = np.where(r_results > R_THRESHOLD)[0]
    return K_VALUES_SWEEP[indices[0]] if len(indices) > 0 else None

def run_full_analysis(G, thetas, omegas):
    """
    Ejecuta un análisis completo: barrido, encuentra Kc, y simula estados adicionales.
    Todo en una sola función optimizada para evitar preparar la matriz múltiples veces.

    Returns:
        dict con:
        - r_values: Array de valores r del barrido
        - kc_value: Valor crítico de K
        - key_states: Dict con estados clave y sus thetas
        - A_sparse: Matriz sparse (para reuso)
        - degrees: Grados de los nodos
    """
    # Hacer el barrido inicial y obtener la matriz sparse
    r_values, initial_key_states, A_sparse, degrees = sweep_analysis(
        G, thetas, omegas, store_thetas_at_kc=True, return_sparse=True
    )

    # Obtener Kc del resultado de sweep_analysis
    kc_value = initial_key_states.get("kc_value", None)

    # Preparar diccionario de resultados
    results = {
        'r_values': np.array(r_values),
        'kc_value': kc_value,
        'key_states': {},
        'A_sparse': A_sparse,
        'degrees': degrees
    }

    if kc_value is not None and "kc_index" in initial_key_states:
        kc_index = initial_key_states["kc_index"]

        # Estados alrededor de Kc
        if kc_index > 0:
            results['key_states']["pre_kc"] = (K_VALUES_SWEEP[kc_index-1], kc_index-1)
        results['key_states']["at_kc"] = (kc_value, kc_index)
        results['key_states']["partial_sync_thetas"] = initial_key_states["partial_sync_thetas"]

        print("\nSimulando estados adicionales...")

        # Simular estados adicionales con ratios mejorados para mejor progresión
        additional_K_values = [
            (0.5 * kc_value, "desync"),   # Estado desincronizado (r ≈ 0.2-0.3)
            (1.0 * kc_value, "partial"),  # Estado en Kc (r ≈ 0.5)
            (1.8 * kc_value, "sync")     # Estado sincronizado (r ≈ 0.8-0.9)
        ]
        
        print(f"    Ratios mejorados: 0.5*Kc={0.5*kc_value:.3f}, 1.0*Kc={kc_value:.3f}, 1.8*Kc={1.8*kc_value:.3f}")

        for K, state_name in additional_K_values:
            print(f"  - Estado {state_name} (K = {K:.3f})...")
            r, thetas_state, _ = run_simulation(K, A_sparse, thetas, omegas, degrees)
            results['key_states'][f"{state_name}"] = (K, -1)
            results['key_states'][f"{state_name}_thetas"] = thetas_state

        # Actualizar el estado parcial para reemplazar el estado en Kc
        if "partial_thetas" in results['key_states']:
            results['key_states']["partial_sync_thetas"] = results['key_states']["partial_thetas"]

    return results
