import cupy as cp
import numpy as np
import networkx as nx
from tqdm import tqdm
from cupyx.scipy import sparse

# Parámetros de la Red
N = 10000
M_SCALE_FREE = 5

# Parámetros de la dinámica
OMEGA_MU = 0.0
OMEGA_SIGMA = 0.5
T_TRANSIENT = 5    # Tiempo transitorio (descartado)
T_MEASURE = 10      # Tiempo de medición para calcular r
DT = 0.01            # Paso del tiempo para el integrador RK4

# Nota sobre medición del parámetro de orden:
# El parámetro de orden r(t) se muestrea cada 10 pasos temporales (cada 0.5 unidades
# de tiempo) durante la fase de medición. Luego se aplica integración ponderada por
# tiempo usando la regla trapezoidal para obtener el promedio temporal verdadero:
# ⟨r⟩_T = (1/T) ∫[0→T] r(t) dt
# Esto es especialmente importante cerca de transiciones de fase donde r(t) puede
# tener fluctuaciones lentas o deriva que sesgaría un promedio aritmético simple.

K_VALUES_SWEEP = np.linspace(0, 5, 50)
R_THRESHOLD = 0.5    # Umbral para definir la sincronización

# Debug: Para redes scale-free, el threshold puede ser más bajo debido a heterogeneidad
# Threshold más bajo permite detectar Kc cuando r≈0.4 en lugar de r≈0.5

def time_weighted_average(r_values, dt_sample):
    """
    Compute time-weighted average of order parameter using trapezoidal rule.

    This properly accounts for the continuous evolution of the order parameter
    r(t) during the measurement period, avoiding bias in slow transitions near
    the critical coupling where simple arithmetic averaging can be misleading.

    Physical motivation: The order parameter r(t) fluctuates continuously,
    and near phase transitions it may have slow oscillations or drift.
    Time integration gives the true time-averaged value: ⟨r⟩_T = (1/T) ∫ r(t) dt

    Args:
        r_values: Array of order parameter samples
        dt_sample: Time interval between samples (e.g., 10*DT = 0.5)

    Returns:
        Time-weighted average of r using trapezoidal integration
    """
    r_values = cp.asarray(r_values)

    if len(r_values) < 2:
        return r_values[0] if len(r_values) == 1 else 0.0

    # Trapezoidal integration: ∫r(t)dt ≈ dt * [r0/2 + r1 + r2 + ... + rN/2]
    integral = 0.5 * (r_values[0] + r_values[-1]) + cp.sum(r_values[1:-1])
    total_time = (len(r_values) - 1) * dt_sample

    # Return time-averaged value: (1/T) ∫ r(t) dt
    return integral * dt_sample / total_time

def time_weighted_frequency_average(freq_values, dt_sample):
    """
    Compute time-weighted average of instantaneous frequencies using trapezoidal rule.

    This function follows the same pattern as time_weighted_average() for the order
    parameter, ensuring consistent temporal averaging across all measurements.

    Physical motivation: Near phase transitions, instantaneous frequencies ω(t)
    can fluctuate or drift as oscillators adjust to network coupling. Simple
    endpoint differences (θ_final - θ_initial)/T miss this complex dynamics
    and can be misleading due to phase wrapping at 2π boundaries.

    Time integration gives the true time-averaged frequency:
    ⟨ω⟩_T = (1/T) ∫[0,T] ω(t) dt

    Args:
        freq_values: Array of instantaneous frequency samples (N_nodes x N_samples)
        dt_sample: Time interval between samples (e.g., 10*DT = 0.1)

    Returns:
        Time-weighted average frequencies for each node
    """
    freq_values = cp.asarray(freq_values)

    if freq_values.ndim == 1:
        # Single time series case
        if len(freq_values) < 2:
            return freq_values[0] if len(freq_values) == 1 else 0.0

        # Trapezoidal integration
        integral = 0.5 * (freq_values[0] + freq_values[-1]) + cp.sum(freq_values[1:-1])
        total_time = (len(freq_values) - 1) * dt_sample
        return integral * dt_sample / total_time

    else:
        # Multiple time series (N_nodes x N_samples)
        if freq_values.shape[1] < 2:
            return freq_values[:, 0] if freq_values.shape[1] == 1 else cp.zeros(freq_values.shape[0])

        # Vectorized trapezoidal integration across all nodes
        integral = 0.5 * (freq_values[:, 0] + freq_values[:, -1]) + cp.sum(freq_values[:, 1:-1], axis=1)
        total_time = (freq_values.shape[1] - 1) * dt_sample
        return integral * dt_sample / total_time

def unwrap_phases(phase_history):
    """
    Unwrap phase trajectories to handle 2π discontinuities in frequency calculations.

    When phases wrap around the 2π boundary, simple differences (θ_final - θ_initial)
    can give misleading frequency measurements. This function unwraps the phase
    evolution to ensure continuity for accurate frequency calculation.

    Physical motivation: Oscillator phases naturally evolve continuously, but
    numerical representations are bounded to [0, 2π]. Near synchronization
    transitions, oscillators may complete multiple rotations, and the cumulative
    phase change is more meaningful than the wrapped final position.

    Args:
        phase_history: Array of phase values over time (N_nodes x N_samples) or (N_samples,)

    Returns:
        Unwrapped phases with 2π discontinuities removed
    """
    phase_history = cp.asarray(phase_history)

    if phase_history.ndim == 1:
        # Single time series case
        return cp.unwrap(phase_history)
    else:
        # Multiple time series - unwrap each node separately
        unwrapped = cp.zeros_like(phase_history)
        for i in range(phase_history.shape[0]):
            unwrapped[i, :] = cp.unwrap(phase_history[i, :])
        return unwrapped

def kuramoto_odes_complete_graph(thetas, K, omegas):
    """
    Versión optimizada para grafos completos usando las identidades trigonométricas:
    sum_j sin(θj-θi) = cos(θi)*sum_j sin(θj) - sin(θi)*sum_j cos(θj)
    Evita crear la matriz densa NxN completa pero mantiene equivalencia exacta.
    """
    N = len(thetas)

    # Calcular las sumas trigonométricas una sola vez
    sum_sin = cp.sum(cp.sin(thetas))
    sum_cos = cp.sum(cp.cos(thetas))

    # Aplicar la identidad trigonométrica para cada nodo
    interactions = cp.cos(thetas) * sum_sin - cp.sin(thetas) * sum_cos

    # Para grafo completo, cada nodo tiene grado N-1
    dthetas_dt = omegas + (K / (N - 1)) * interactions

    return dthetas_dt

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

def rk4_step_complete_graph(thetas, dt, K, omegas):
    """
    Versión optimizada de RK4 para grafos completos usando fórmula analítica.
    """
    dt_half = 0.5 * dt
    dt_six  = dt / 6.0

    # k1: Derivada al inicio del intervalo
    k1 = kuramoto_odes_complete_graph(thetas, K, omegas)

    # k2: Derivada en el punto medio, usando k1
    k2 = kuramoto_odes_complete_graph(thetas + dt_half * k1, K, omegas)

    # k3: Derivada en el punto medio, usando k2
    k3 = kuramoto_odes_complete_graph(thetas + dt_half * k2, K, omegas)

    # k4: Derivada al final del intervalo, usando k3
    k4 = kuramoto_odes_complete_graph(thetas + k3 * dt, K, omegas)

    # Actualización final con el promedio ponderado de las derivadas
    thetas += dt_six * (k1 + 2*k2 + 2*k3 + k4)

    # Normalizar fases al rango [0, 2π] para evitar problemas numéricos
    two_pi = thetas.dtype.type(2.0) * cp.pi
    thetas %= two_pi

    return thetas

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
    # Usar modulo (%) en lugar de remainder para evitar discontinuidades
    two_pi = thetas.dtype.type(2.0) * cp.pi
    thetas %= two_pi

    return thetas

def run_simulation_complete_graph(K, thetas_0, omegas):
    """
    Versión optimizada de simulación para grafos completos.
    No requiere matriz de adyacencia ni array de grados.
    """
    num_steps_transient = int(T_TRANSIENT / DT)
    num_steps_measure = int(T_MEASURE / DT)
    thetas_current = thetas_0.copy()

    # Fase transitoria
    for _ in range(num_steps_transient):
        thetas_current = rk4_step_complete_graph(thetas_current, DT, K, omegas)

    # Fase de medición
    r_values = []
    for i in range(num_steps_measure):
        thetas_current = rk4_step_complete_graph(thetas_current, DT, K, omegas)

        # Calcular r cada 10 pasos para promediar
        # Nota: El muestreo cada 10 pasos es suficiente para capturar la dinámica
        # del parámetro de orden, ya que las fluctuaciones rápidas se promedian
        # naturalmente en la integración temporal posterior
        if i % 10 == 0:
            exp_thetas = cp.exp(1j * thetas_current)
            r = cp.abs(cp.mean(exp_thetas))
            r_values.append(r)

    # Promedio ponderado por tiempo de r en el estado estacionario
    # Sampling interval: 10 pasos * DT = 10 * 0.05 = 0.5 time units
    dt_sample = 10 * DT
    r_final = time_weighted_average(r_values, dt_sample)

    return r_final, thetas_current, r_values

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
        # Nota: El muestreo cada 10 pasos es suficiente para capturar la dinámica
        # del parámetro de orden, ya que las fluctuaciones rápidas se promedian
        # naturalmente en la integración temporal posterior
        if i % 10 == 0:
            exp_thetas = cp.exp(1j * thetas_current)
            r = cp.abs(cp.mean(exp_thetas))
            r_values.append(r)

    # Promedio ponderado por tiempo de r en el estado estacionario
    # Sampling interval: 10 pasos * DT = 10 * 0.05 = 0.5 time units
    dt_sample = 10 * DT
    r_final = time_weighted_average(r_values, dt_sample)

    return r_final, thetas_current, r_values

def run_simulation_with_frequency_tracking(K, A_sparse, thetas_0, omegas, degrees):
    """
    Extended version of run_simulation() that tracks instantaneous frequencies.

    This function follows the same structure as run_simulation() but additionally
    collects instantaneous frequency data during the measurement phase for proper
    time-weighted frequency averaging.

    Args:
        K: Coupling strength
        A_sparse: Sparse adjacency matrix
        thetas_0: Initial phases
        omegas: Natural frequencies
        degrees: Node degrees

    Returns:
        r_final: Time-weighted average order parameter
        thetas_final: Final phases
        effective_freqs: Time-weighted average frequencies for each node
    """
    num_steps_transient = int(T_TRANSIENT / DT)
    num_steps_measure = int(T_MEASURE / DT)
    thetas_current = thetas_0.copy()

    # Fase transitoria: Dejamos que el sistema "olvide" sus condiciones iniciales
    for _ in range(num_steps_transient):
        thetas_current = rk4_step(thetas_current, DT, K, A_sparse, omegas, degrees)

    # Fase de medición: Colectar tanto r(t) como ω(t)
    r_values = []
    freq_values = []  # Store instantaneous frequencies for all nodes

    for i in range(num_steps_measure):
        thetas_current = rk4_step(thetas_current, DT, K, A_sparse, omegas, degrees)

        # Muestrear cada 10 pasos para consistencia con run_simulation()
        if i % 10 == 0:
            # Calcular parámetro de orden
            exp_thetas = cp.exp(1j * thetas_current)
            r = cp.abs(cp.mean(exp_thetas))
            r_values.append(r)

            # Calcular frecuencias instantáneas
            instantaneous_freqs = kuramoto_odes(thetas_current, K, A_sparse, omegas, degrees)
            freq_values.append(instantaneous_freqs.copy())

    # Convertir a array para procesamiento vectorizado
    freq_values = cp.stack(freq_values, axis=1)  # Shape: (N_nodes, N_samples)

    # Promedio ponderado por tiempo
    dt_sample = 10 * DT
    r_final = time_weighted_average(r_values, dt_sample)
    effective_freqs = time_weighted_frequency_average(freq_values, dt_sample)

    return r_final, thetas_current, effective_freqs

def generate_random_network(seed = None):
    print(f"Generando Red Libre de Escala (N={N}, m={M_SCALE_FREE})...")

    if seed is not None:
        np.random.seed(seed)
        cp.random.seed(seed)
        G = nx.barabasi_albert_graph(N, M_SCALE_FREE, seed=seed)
    else:
        G = nx.barabasi_albert_graph(N, M_SCALE_FREE)

    omegas_0 = cp.random.normal(OMEGA_MU, OMEGA_SIGMA, N, dtype=cp.float32)
    thetas_0 = cp.random.uniform(0, 2 * np.pi, N, dtype=cp.float32)

    return G, omegas_0, thetas_0

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
