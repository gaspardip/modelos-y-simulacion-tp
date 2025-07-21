# Sincronización en Redes Complejas: Un Análisis Computacional del Modelo de Kuramoto

## Resumen

Este repositorio contiene el código y los análisis de un proyecto de investigación computacional sobre el fenómeno de la sincronización en redes complejas, utilizando el modelo de Kuramoto como marco teórico. El objetivo principal es investigar cómo la topología de la red subyacente —específicamente la diferencia entre redes homogéneas (grafos completos) y redes heterogéneas (libres de escala)— afecta la emergencia y la naturaleza del orden colectivo.

A través de un pipeline de cuatro scripts de análisis, este proyecto explora la transición de fase del desorden a la sincronización, visualiza los mecanismos de formación de clusters, y valida los resultados mediante un análisis estadístico robusto. Las simulaciones se implementan en Python, utilizando `CuPy` para la aceleración por GPU y `NetworkX` para la manipulación de grafos, permitiendo el análisis de redes a gran escala (N >= 10,000).

El proyecto documenta no solo los resultados finales, sino también el proceso de descubrimiento, incluyendo la refutación de hipótesis iniciales y el refinamiento de la metodología computacional (ej. la transición de integradores numéricos y la optimización de memoria), culminando en una conclusión matizada sobre la compleja interacción entre la estructura y la dinámica en sistemas autoorganizados.

## 1. Introducción y Contexto

La sincronización es un fenómeno emergente fundamental en el que una población de osciladores acoplados ajusta espontáneamente sus ritmos para oscilar al unísono. Este comportamiento se observa en una vasta gama de sistemas, desde el parpadeo coordinado de luciérnagas hasta la activación neuronal en el cerebro.

El modelo de Kuramoto es el paradigma matemático para estudiar estas transiciones. En su forma original, asume un acoplamiento de campo medio donde cada oscilador interactúa con todos los demás. Sin embargo, las interacciones en sistemas reales están gobernadas por la estructura de redes complejas. Este proyecto aborda la pregunta central: **¿Cómo influye la arquitectura de una red en su capacidad para sincronizarse?**

## 2. Hipótesis

El proceso de investigación siguió una evolución de hipótesis, donde los resultados de las simulaciones nos forzaron a refinar nuestra comprensión.

#### Hipótesis Inicial (Parcialmente Incorrecta)
Nuestra primera hipótesis fue que las redes libres de escala, debido a la presencia de hubs súper-conectados, facilitarían la sincronización, resultando en un umbral de acoplamiento crítico (`Kc`) más bajo en comparación con un grafo completo.

#### Hipótesis Refinada (Confirmada)
Las primeras simulaciones (`Script 1`) revelaron un panorama más complejo. Si bien el `Kc` para el grafo completo era menor, su transición era extremadamente abrupta ("todo o nada"). La red libre de escala, en cambio, mostraba una transición mucho más suave y gradual. Esto llevó a la hipótesis refinada de que **la topología no solo afecta *si* el sistema se sincroniza, sino fundamentalmente *cómo* lo hace**, sugiriendo un mecanismo jerárquico en las redes heterogéneas.

#### Hipótesis Fallida (Descubrimiento Clave)
Inicialmente, planteamos la hipótesis de que el "núcleo estructural" (los hubs) sería idéntico al "núcleo dinámico" (los nodos más sincronizados). Intentamos probar esto comparando el top 5% de nodos por grado con el top 5% por orden local o frecuencia efectiva, usando el Índice de Jaccard. Los resultados (`Script 3` en sus primeras versiones) mostraron un solapamiento casi nulo, **refutando la hipótesis**.

#### Hipótesis Final (Confirmada)
El fracaso anterior nos llevó a la conclusión más profunda del proyecto: los hubs actúan como **catalizadores** del orden, formando el núcleo del cluster sincronizado. Sin embargo, no son necesariamente los miembros más *perfectamente* sincronizados. Debido a sus propias frecuencias naturales (su "terquedad"), mantienen una frecuencia efectiva residual. Los miembros más perfectamente sincronizados son los nodos "dóciles" con frecuencias naturales cercanas a la media del sistema. La sincronización es, por tanto, un **balance de fuerzas entre la importancia estructural y las propiedades dinámicas intrínsecas**.

## 3. Metodología y Evolución Técnica

La implementación de las simulaciones evolucionó significativamente para abordar los desafíos de precisión y rendimiento.

1.  **De `solve_ivp` a Integradores en GPU:** El enfoque inicial con `scipy.integrate.solve_ivp` demostró ser un cuello de botella insalvable para redes grandes debido a su dependencia de la CPU. La migración a `CuPy` para la computación en GPU fue el primer paso crítico.

2.  **De Euler a Runge-Kutta 4 (RK4):** La necesidad de un integrador personalizado en la GPU nos llevó primero a implementar el método de Euler. Sin embargo, para garantizar la estabilidad y precisión se adoptó un integrador **Runge-Kutta de 4º orden (RK4)**. Esta implementación, enteramente en `CuPy`, representa el estándar de oro para este tipo de simulaciones, ofreciendo un balance óptimo entre velocidad y fiabilidad.

3.  **Optimización de Memoria (Dense a Sparse):** Un desafío clave en redes grandes es el consumo de memoria del término de interacción, que escala como \(O(N^2)\). Se implementó una optimización fundamental utilizando la identidad trigonométrica \(\sin(\theta_j - \theta_i) = \cos(\theta_i)\sin(\theta_j) - \sin(\theta_i)\cos(\theta_j)\). Esto permite calcular la derivada usando multiplicaciones de la **matriz de adyacencia dispersa** por vectores, reduciendo el consumo de memoria a \(O(E)\), donde \(E\) es el número de aristas.

4.  **Refinamiento de la Visualización:** Las visualizaciones iniciales de redes grandes resultaban en "bolas de pelo" ilegibles. La metodología final se centra en visualizar un **subgrafo filtrado** que contiene solo los hubs más importantes, con el tamaño de los nodos escalado de forma no lineal con su grado para una máxima claridad.

## 4. Análisis en cuatro scripts

El proyecto está estructurado en un pipeline de cuatro scripts, cada uno con un propósito específico.

*   **`1.py` (Análisis Cuantitativo):**
    Este script genera el resultado macroscópico principal: el gráfico del parámetro de orden `r` en función de la fuerza de acoplamiento `K`. Compara el comportamiento de un grafo completo con una red libre de escala a gran escala (`N >= 10000`), utilizando el integrador RK4 y las optimizaciones de matriz dispersa.

*   **`2.py` (Análisis Visual):**
    Este script es la "radiografía" del proyecto. Toma una única red libre de escala grande, calcula su `Kc` a partir de un barrido, y genera tres visualizaciones detalladas en los regímenes clave (desordenado, parcial y fuerte). Mapea el grado del nodo a su tamaño y su frecuencia efectiva a su color, revelando el mecanismo de sincronización jerárquico.

*   **`3.py` (Análisis Profundo):**
    Este script prueba la hipótesis final. Se enfoca en el estado de sincronización parcial y compara el "núcleo estructural" (los hubs) con el "núcleo dinámico" (definido por la frecuencia efectiva). Utiliza el Índice de Jaccard y visualizaciones lado a lado para cuantificar y mostrar la compleja relación entre ambos.

*   **`4.py` (Validación y Robustez):**
    Este script aborda la naturaleza estocástica del modelo. Ejecuta la simulación cientos de veces sobre diferentes realizaciones de redes libres de escala para calcular la **esperanza (media) y la desviación estándar** del umbral crítico `Kc`. El resultado es un histograma que caracteriza la distribución de `Kc`, dando validez estadística a las conclusiones del proyecto.

## 5. Cómo Ejecutar el Código

1.  **Entorno:** Se recomienda utilizar Google Colab con un entorno de ejecución de GPU.
2.  **Instalación:** Los scripts incluyen los comandos necesarios para instalar `CuPy` y otras dependencias.
3.  **Ejecución:** Ejecutar los scripts en orden (del 1 al 4) para seguir la narrativa de la investigación. Cada script es autónomo y generará sus propios gráficos y resultados numéricos.

## 6. Referencias

-   Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators.
-   Strogatz, S. H. (2000). From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators.
-   Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks.