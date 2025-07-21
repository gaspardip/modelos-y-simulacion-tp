### **Glosario del Proyecto: Sincronización en Redes Complejas**

#### **Conceptos Fundamentales del Modelo de Kuramoto**

*   **Modelo de Kuramoto:** Un modelo matemático que describe cómo una población de osciladores acoplados puede pasar espontáneamente de un estado de desorden a uno de sincronización. Es el pilar teórico de todo el proyecto.
*   **Oscilador:** Una entidad individual que tiene un ritmo o ciclo natural. En nuestro caso, cada nodo de la red es un oscilador.
*   **Fase (\(\theta_i\)):** Un ángulo (de 0 a \(2\pi\)) que representa la posición de un oscilador en su ciclo en un instante de tiempo. Es la variable principal que simulamos.
*   **Frecuencia Natural (\(\omega_i\)):** La velocidad angular intrínseca a la que un oscilador rotaría si no estuviera conectado a ningún otro. En nuestro proyecto, las extraemos de una distribución Gaussiana.
*   **Fuerza de Acoplamiento (\(K\)):** Un parámetro global que controla la intensidad con la que los osciladores se influyen mutuamente. Es la variable que barremos para inducir la transición a la sincronización.
*   **Parámetro de Orden (\(r\)):** Una medida macroscópica (de 0 a 1) que cuantifica el nivel de coherencia global del sistema. \(r \approx 0\) es caos, \(r \approx 1\) es sincronización perfecta. Es nuestra principal métrica cuantitativa.

#### **Conceptos de Redes Complejas**

*   **Grafo Completo:** Una topología de red donde cada nodo está conectado a todos los demás. Es una red **homogénea** y "democrática" que usamos como línea de base.
*   **Red Libre de Escala (Scale-Free Network):** Una topología de red cuya distribución de grado sigue una ley de potencias. Es una red **heterogénea** y "jerárquica".
*   **Modelo de Barabási-Albert (BA):** El algoritmo que usamos para generar redes libres de escala. Se basa en dos principios: **crecimiento** y **acoplamiento preferencial**.
*   **Hubs:** Los pocos nodos en una red libre de escala que tienen un grado (número de conexiones) extremadamente alto. Son el **núcleo estructural** de la red.
*   **Periferia:** La gran mayoría de los nodos en una red libre de escala, que tienen un grado muy bajo.
*   **Laplaciano del Grafo (\(L\)):** Una matriz (\(L = D - A\)) que codifica la estructura de conectividad de una red. Sus autovalores y autovectores (su "espectro") revelan las comunidades y propiedades de conectividad de la red.
*   **Spectral Clustering:** Un algoritmo de clustering que utiliza los autovectores del Laplaciano para encontrar las comunidades estructurales "naturales" de una red.

#### **Fenómenos Dinámicos y de Sincronización**

*   **Transición de Fase:** El cambio abrupto o gradual en el comportamiento macroscópico del sistema (medido por `r`) cuando un parámetro de control (`K`) cruza un valor crítico.
*   **Umbral Crítico (\(K_c\)):** El valor específico de la fuerza de acoplamiento `K` en el cual el sistema comienza a mostrar un nivel significativo de sincronización.
*   **Cluster Sincronizado:** Un subconjunto de osciladores que se han "enganchado" (phase-locked) y oscilan a una frecuencia común.
*   **Drifters (Osciladores a la deriva):** Osciladores, generalmente en la periferia o con frecuencias naturales muy extremas, que no son capturados por el cluster principal y continúan girando a su propio ritmo.
*   **Frecuencia Efectiva (\(\Omega_i\)):** La velocidad de rotación promedio de un oscilador en el estado estacionario. Para los nodos del cluster, converge a la frecuencia media del sistema (\(\mu\)). Para los drifters, es cercana a su frecuencia natural (\(\omega_i\)).
*   **Estado Estacionario:** El comportamiento a largo plazo del sistema después de que las dinámicas transitorias iniciales se han desvanecido.
*   **Transitorio:** La fase inicial de la simulación donde el sistema evoluciona desde sus condiciones iniciales aleatorias hacia su estado de equilibrio o atractor.

#### **Metodología Computacional y Optimización**

*   **Integrador Numérico:** El algoritmo utilizado para aproximar la solución de las ecuaciones diferenciales.
    *   **Método de Euler:** Un método simple y rápido, pero de baja precisión y propenso a la inestabilidad con `DT` grandes.
    *   **Runge-Kutta 4 (RK4):** Un método de orden superior, mucho más preciso y estable, que permite usar pasos de tiempo `DT` más grandes. Fue nuestra elección final para garantizar la fiabilidad de los resultados.
*   **Paso de Tiempo (\(DT\)):** El tamaño del intervalo de tiempo discreto utilizado por el integrador. Su elección es un balance crítico entre la precisión y la velocidad de la simulación.
*   **Phase Wrapping:** El proceso de mantener las fases dentro del intervalo `[0, 2\pi]` usando el operador módulo. Descubrimos que aplicarlo **en cada paso de la integración** es crucial para obtener una representación físicamente correcta de un cluster estacionario.
*   **Matrices Dispersas (Sparse Matrices):** Una optimización de memoria fundamental. Al evitar la creación de la matriz densa \(N \times N\) de interacciones y usar la matriz de adyacencia dispersa, reducimos el consumo de memoria de \(O(N^2)\) a \(O(E)\), permitiendo simulaciones a gran escala.
*   **Visualización Filtrada:** La técnica que desarrollamos para visualizar redes grandes de forma clara, dibujando únicamente el "esqueleto" de la red compuesto por los hubs más importantes.

#### **Términos Análogos e Intuitivos**

*   **Terquedad (Stubbornness):** analogía para la **frecuencia natural (\(\omega_i\))** de un oscilador. Un oscilador con una \(\omega_i\) alta (lejos de la media) es muy "terco" y difícil de sincronizar. La `OMEGA_SIGMA` controla la diversidad de "terquedad" en la población.
*   **Presión Social (Social Pressure):** analogía para la **fuerza de acoplamiento (\(K\))**. Es la fuerza que empuja a los individuos a conformarse con el comportamiento del grupo.
*   **Inercia del Sistema (System Inertia):** analogía para el **tamaño de la red (\(N\))**. Un sistema más grande tiene más "inercia" y requiere una `K` global mayor para alcanzar el mismo nivel de sincronización efectiva.
*   **Líderes vs. Seguidores Dóciles:** conclusión clave del análisis. Los **líderes** son los hubs, que catalizan la sincronización pero no son los más perfectamente sincronizados. Los **seguidores dóciles** son los nodos con \(\omega_i \approx 0\), que se unen al cluster sin esfuerzo y exhiben la sincronización más perfecta.
*   **Bola de Pelo (Hairball):** término usado para describir el resultado de una visualización ingenua de una red grande y densa, donde la estructura es completamente ilegible.
*   **Radiografía de la Red:** metáfora para el análisis visual profundo (Script 2 y 3), que busca revelar la estructura y el mecanismo interno de la sincronización, en lugar de solo medir el resultado macroscópico.