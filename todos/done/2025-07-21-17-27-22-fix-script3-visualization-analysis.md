Based on my analysis of `src/3.py`, here's a detailed understanding of the current implementation:

## 1. **How it currently visualizes the network**
- Uses NetworkX's `spring_layout` for positioning nodes (line 105)
- Node colors represent effective frequencies (calculated as phase differences over time)
- Node sizes are based on degrees with a power scaling: `degrees^1.2 + 20` (line 97)
- Edges are drawn with low opacity (0.3 width, 0.6 alpha) in grey
- Uses a diverging colormap (coolwarm) centered around zero frequency

## 2. **How it identifies and shows hubs**
- Identifies hubs by degree centrality - sorts all nodes by degree in descending order (line 75)
- Shows only the top N% of nodes by degree (controlled by `TOP_HUBS_PERCENT = 0.1`)
- Adds text labels for the top 10 highest-degree nodes (lines 113-115)
- Node size is proportional to degree, making hubs visually larger

## 3. **What makes nodes appear disconnected**
- The visualization creates a **subgraph** containing only the top hub nodes (line 80)
- This subgraph only includes edges between the selected hub nodes
- Lower-degree nodes and their connections are completely excluded from the visualization
- Even if two hubs are connected through intermediate nodes in the full network, they appear disconnected if those intermediate nodes aren't in the top percentage

## 4. **Current filtering logic for showing only important nodes**
The `get_filtered_nodes_for_visualization` function (lines 72-80):
- Sorts all nodes by degree (highest to lowest)
- Calculates how many nodes to show: `max(int(len(nodes) * top_percent), min_nodes)`
- Takes the top N nodes and creates a subgraph containing only these nodes
- Default: shows top 10% of nodes with a minimum of 50 nodes

## 5. **Parameters that control how many nodes are displayed**
- `TOP_HUBS_PERCENT = 0.1` (line 25): Percentage of top nodes to show (10%)
- `MIN_NODES_TO_SHOW = 50` (line 26): Minimum number of nodes to display
- The actual number shown is the maximum of these two calculations
- For a 10,000 node network: would show max(1000, 50) = 1000 nodes

## Key Issues with Current Approach:
1. **Loss of connectivity**: By creating a subgraph of only hub nodes, many connections are lost
2. **No representation of hidden connections**: Hubs that are connected through non-hub nodes appear disconnected
3. **Binary filtering**: Nodes are either fully shown or completely hidden - no middle ground
4. **Fixed percentage**: Always shows 10% regardless of network structure

The visualization prioritizes showing the most connected nodes but sacrifices network topology accuracy in the process. This explains why many hubs appear as isolated nodes when they may actually be well-connected through intermediate nodes in the full network.

## Summary

I've analyzed the visualization approaches in scripts 1, 2, and 4, and created an improved version of script 3 (`3_improved.py`) that incorporates the best techniques:

### Key Improvements Applied:

1. **Spectral Layout** (from Script 2, lines 352-498):
   - Uses Laplacian eigenvectors for optimal node positioning
   - Naturally reveals synchronization clusters
   - Handles up to 150 nodes cleanly (vs 50 in original)

2. **Synchronization-Aware Color Mapping** (from Script 2, lines 296-351):
   - Colors compress as synchronization increases
   - Uses sigmoid function for smooth transitions
   - Visually represents the synchronization state

3. **Hierarchical Edge Drawing** (from Script 2, lines 633-666):
   - Three tiers: super-hub edges (thick), hub edges (medium), other edges (thin)
   - Reduces visual clutter while emphasizing network backbone
   - Edge transparency varies with synchronization level

4. **Smart Node Filtering** (from Script 2, lines 373-417):
   - Prioritizes super hubs (top 1%), major hubs (top 5%), minor hubs (top 15%)
   - Maintains network connectivity for valid spectral analysis
   - Better preservation of network structure

5. **Professional Styling** (from Script 1):
   - Seaborn whitegrid style
   - Professional color schemes (royalblue, crimson)
   - Enhanced titles, labels, and formatting

The improved script (`/home/gaspar/src/tp-modelos-y-simulacion/src/3_improved.py`) provides much clearer visualizations for large networks while maintaining the core analysis functionality. The visualization improvements documentation is saved in `/home/gaspar/src/tp-modelos-y-simulacion/src/visualization_improvements.md`.

Based on my analysis of the codebase, here's a comprehensive understanding of how hubs are defined and identified in scale-free networks in this project:

## Hub Definition and Identification in Scale-Free Networks

### 1. **Network Generation Parameters**
- The project uses **Barabási-Albert scale-free networks** generated with `nx.barabasi_albert_graph(N, M_SCALE_FREE)`
- Key parameters:
  - `N = 10000` nodes
  - `M_SCALE_FREE = 3` (each new node connects to 3 existing nodes)
  - This creates a preferential attachment network where hubs naturally emerge

### 2. **Hub Definition by Degree**
The primary metric for identifying hubs is **node degree** (number of connections):

```python
# From script 2.py:
degree_threshold = np.percentile(all_degrees, 85)  # Top 15% as hubs
hub_nodes = [n for n, d in degrees.items() if d >= degree_threshold]
```

The project uses a **hierarchical classification** of hubs:
- **Super hubs**: Top 1% by degree (99th percentile)
- **Major hubs**: Top 5% by degree (95th percentile)
- **Minor hubs**: Top 15% by degree (85th percentile)

### 3. **Metrics Used for Node Importance**

#### a) **Degree Centrality**
- Primary metric for hub identification
- Stored in `degrees` array from adjacency matrix: `degrees = cp.array(A_scipy.sum(axis=1).flatten())`

#### b) **K-core Decomposition**
- Used for identifying structurally important nodes:
```python
k_core_dict = nx.core_number(G_giant)
high_kcore_nodes = [n for n, k in k_core_dict.items() if k >= max_k_core]
```

#### c) **Combined Importance Score**
- Combines k-core and degree for node importance:
```python
node_importance = [(n, k_core_dict[n] * degrees_dict[n]) for n in nodes_to_keep]
```

#### d) **Clustering Coefficient**
- Used in hub analysis for understanding local structure:
```python
clustering_coeffs = np.array([nx.clustering(G, node) for node in hub_nodes])
```

### 4. **Filtering Functions for Hub Visualization**

#### a) **Percentage-based Filtering**
```python
def get_filtered_nodes_for_visualization(G, top_percent=0.1, min_nodes=50):
    """Filtra nodos para visualización: solo hubs principales"""
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    num_to_show = max(int(len(sorted_nodes) * top_percent), min_nodes)
    top_nodes = [node for node, degree in sorted_nodes[:num_to_show]]
    return G.subgraph(top_nodes).copy()
```

#### b) **Visualization Parameters**
- `TOP_HUBS_PERCENT = 0.1`: Shows only top 10% of nodes by degree
- `MIN_NODES_TO_SHOW = 50`: Minimum nodes to display
- Node sizes scale with degree: `node_sizes = degrees**1.2 + 20`

### 5. **Network Generation Function**
The `generate_random_network()` function in `utils.py`:
- Creates a Barabási-Albert graph with preferential attachment
- This naturally generates a scale-free degree distribution with hubs
- The parameter `m=3` means each new node attaches to 3 existing nodes preferentially by degree

### 6. **Additional Hub-Related Features**
- **Effective frequencies**: Calculated for hubs to analyze their dynamic behavior
- **Phase clustering**: Hubs are analyzed for synchronization patterns
- **Structural vs. dynamic importance**: The code distinguishes between topological hubs (high degree) and dynamically important nodes

The project focuses on understanding how these hubs influence synchronization dynamics in Kuramoto oscillator networks, particularly around the critical coupling strength (Kc) where the network transitions from desynchronized to synchronized states.