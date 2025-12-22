import numpy as np
import heapq

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, no_update

from typing import Literal, Any
from numpy.typing import ArrayLike, NDArray

from .distance_metrics import l2_distance, cosine_distance

class HNSW:
    """
    Hierarchical Navigable Small World (HNSW) Graph Implementation.

    This class implements the HNSW algorithm for Approximate Nearest Neighbor (ANN) search.
    It constructs a multi-layer graph structure where the bottom layer (Layer 0) contains 
    all data points, and upper layers serve as sparse "expressways" for logarithmic 
    scaling of the search process.

    The algorithm approximates a randomized skip-list structure generalized for 
    high-dimensional nearest neighbor search.

    Attributes:
        dim (int): Dimensionality of the data vectors.
        distance (callable): The distance metric function (L2 or Cosine).
        max_elements (int): The hard limit on the capacity of the index.
        M (int): The target number of bi-directional links for each node at layers > 0.
            This parameter controls the trade-off between graph density/recall and memory/speed.
        maxM (int): Maximum number of connections allowed per node for layers > 0.
        maxM0 (int): Maximum number of connections allowed per node for layer 0.
            Layer 0 requires higher connectivity to ensure global navigability.
        m_L (float): Level generation normalization factor (1 / ln(M)).
            Used to determine the maximum layer of a new node based on an exponential distribution.
        ef_construction (int): Size of the dynamic candidate list (beam width) during index construction.
            Higher values improve graph quality (recall) at the cost of slower indexing.
        ef (int): Size of the dynamic candidate list during query time.
            Can be adjusted dynamically to trade off search speed for recall.
        graph (list[dict[int, dict]]): Adjacency list representation of the multi-layer graph.
            Structure: graph[level][node_id] -> {neighbor_id: distance}
        data (list[NDArray]): Internal storage for the vector data (row-major).
        level (list[int]): Stores the maximum layer assigned to each node index.
        cur_element_count (int): Current number of elements stored in the index.
        entry_point (int): The global entry point node ID (at the highest layer).
        max_level (int): The current maximum layer depth of the graph.
        rng (np.random.Generator): Random number generator for reproducible level assignment.
    """
    
    def __init__(self, space: Literal['l2', 'cosine'], dim: int) -> None:
        """
        Initializes the HNSW instance with dimensionality and metric space.

        Args:
            space (Literal['l2', 'cosine']): The metric space to use. 
                'l2' for Euclidean distance, 'cosine' for Cosine distance.
            dim (int): The dimensionality of the vectors.
        """
        self.dim = dim
        if space == 'l2':
            self.distance = l2_distance
        else:
            self.distance = cosine_distance


    def init_index(self,
                   max_elements: int,
                   M: int = 16,
                   ef_construction: int = 200,
                   random_seed: int = 12345) -> None:
        """
        Configures the index hyperparameters and initializes storage.

        Args:
            max_elements (int): The maximum number of elements this index can hold.
            M (int, optional): The number of neighbors to connect to in the graph. Defaults to 16.
            ef_construction (int, optional): The size of the candidate queue during insertion.
                Defaults to 200.
            random_seed (int, optional): Seed for random level generation. Defaults to 12345.
        """
        self.max_elements = max_elements
        self.M = M
        self.maxM = M
        # Layer 0 usually connects to more neighbors (2*M) to adhere to the Small World theory
        # ensuring the graph remains connected.
        self.maxM0 = M * 2
        
        # Normalization factor for level generation: -ln(uniform) * m_L
        # Ensures that the number of nodes decays exponentially with the layer level.
        self.m_L = 1 / np.log(M)
        
        self.ef_construction = max(ef_construction, M)
        self.ef = 10 # Default query-time parameter

        self.graph: list[dict[int, dict]] = [] # graph[level][i] is a dict of (neighbor_id, dist)
        self.data: list[NDArray] = [] # Storage for embedded vectors
        self.level: list[int] = [] # Max level for each node

        self.cur_element_count = 0
        self.entry_point = -1
        self.max_level = -1

        self.rng = np.random.default_rng(random_seed)

    def insert_items(self, data: ArrayLike) -> None:
        """
        Batch inserts multiple vectors into the index.

        Args:
            data (ArrayLike): Input data, typically an (N, dim) array.
        """
        data = np.atleast_2d(data)
        for d in data:
            self.insert(d)

    def insert(self, q: ArrayLike, logger=None) -> None:
        """
        Inserts a single vector into the HNSW graph.

        The insertion procedure follows two phases:
        1. **Zoom-in (Coarse Search):** Starts from the top layer down to the node's assigned layer `l`.
           Uses greedy search to find the nearest entry point at layer `l`.
        2. **Construction (Fine Search):** From layer `l` down to 0, finds `ef_construction` 
           nearest neighbors, selects `M` diverse neighbors using heuristics, and adds 
           bidirectional connections.

        Args:
            q (ArrayLike): The query vector to insert.
            logger (callable, optional): Callback for event logging during insertion.
        """
        q = np.array(q)

        W = []
        ep = self.entry_point
        L = self.max_level
    
        # Sample the level for the new element using an exponential distribution
        l = int(-np.log(self.rng.uniform(0.0, 1.0)) * self.m_L)

        self.data.append(q)
        self.level.append(l)
        idx = self.cur_element_count
        self.cur_element_count += 1

        if logger:
            logger({
                'event': 'init_insert',
                'node_id': idx,
                'assigned_level': l,
                'current_max_level': L,
                'entry_point': ep
            })

        if L < l:
            self.max_level = l
            for level in range(l, L, -1):
                self.graph.append({idx: {}})

                if logger:
                    logger({
                        'event': 'node_insert',
                        'level': level,
                        'id': idx
                    })

        if ep != -1:
            # Phase 1: Greedily traverse from top layer L down to l+1
            # to find the closest entry point for the insertion layer.
            for level in range(L, l, -1):
                if logger:
                    logger({
                        'event': 'zoom_in_layer',
                        'level': level,
                        'ep': ep
                    })
                W = self.search_layer(q, ep, 1, level, logger)
                ep = W[0]
                if logger:
                    logger({
                        'event': 'zoom_in_complete',
                        'level': level,
                        'new_ep': ep
                    })

            # Phase 2: Insert the element at layer l down to 0
            for level in range(min(L, l), -1, -1):
                if logger:
                    logger({
                        'event': 'construction_layer',
                        'level': level,
                        'ep': ep
                    })
                W = self.search_layer(q, ep, self.ef_construction, level, logger)
                if logger:
                    logger({
                        'event': 'candidates_found',
                        'level': level,
                        'node_id': idx,
                        'candidates': W.copy()
                    })
                neighbors, dists = self.select_neighbors(idx, W, self.M, level, logger)
                if logger:
                    logger({
                        'event': 'neighbors_selected',
                        'level': level,
                        'node_id': idx,
                        'neighbors': neighbors.copy(),
                        'distances': dists.copy()
                    })

                # Establish bidirectional connections
                for neighbor, dist in zip(neighbors, dists):
                    self._add_bidirectional_connection(idx, neighbor, dist, level, logger)

                # Enforce the maxM constraint on connections for updated neighbors
                for neighbor in neighbors:
                   self._prune_connections(neighbor, level, logger)

                # The nearest neighbor found becomes the entry point for the next layer down
                ep = W[0]
                
                # Log layer transition for the new node appearing at the next layer
                if level > 0 and logger:
                    logger({
                        'event': 'layer_transition_construction',
                        'node_id': idx,
                        'from_layer': level,
                        'to_layer': level - 1,
                        'ep': ep
                    })

        # Update the global entry point if the new node is at a higher level than current max
        if L < l:
            self.entry_point = idx

    def _add_bidirectional_connection(self, u: int, v: int, dist: float, level: int, logger=None):
        """Helper to add an undirected edge (u, v) with weight `dist` at a specific level."""
        if u not in self.graph[level]:
            self.graph[level][u] = {}
        self.graph[level][u][v] = dist
        self.graph[level][v][u] = dist

        if logger:
            logger({
                'event': 'add_connection',
                'level': level,
                'edge': (u, v)
            })

    def _prune_connections(self, u: int, level: int, logger=None):
        """
        Shrinks the connection list of node `u` if it exceeds `maxM`.

        Uses the heuristic selection algorithm to keep the most diverse/closest neighbors
        and removes the rest.
        """
        connections = list(self.graph[level][u].keys())
        maxM = self.maxM if level > 0 else self.maxM0

        if len(connections) > maxM:
            new_connections, new_dists = self.select_neighbors(u, connections, maxM, level)

            old_neighbors = set(connections)
            new_neighbors = set(new_connections)

            for n, d in zip(new_connections, new_dists):
                if n not in old_neighbors:
                    self._add_bidirectional_connection(u, n, d, level, logger)

            for n in connections:
                if n not in new_neighbors:
                    self.graph[level][u].pop(n)
                    self.graph[level][n].pop(u)

                    if logger:
                        logger({
                            'event': 'prune_connection',
                            'level': level,
                            'edge': (u, n)
                        })

    def search_layer(self,
                    q: ArrayLike,
                    ep: int,
                    ef: int,
                    level: int,
                    logger=None) -> list[int]:
        """
        Performs a greedy Best-First Search on a specific layer.

        Maintains two structures:
        1. `candidates`: A min-heap of nodes to explore (frontier).
        2. `W`: A max-heap (simulated with negative distances) of the top-`ef` found neighbors.

        The search stops when the closest candidate in the frontier is further away 
        than the furthest node in the current result set `W`.

        Args:
            q (ArrayLike): The query vector.
            ep (int): The entry point node ID for this layer.
            ef (int): Beam width (size of the dynamic candidate list).
            level (int): The graph layer level.
            logger (callable, optional): Callback for search path visualization.

        Returns:
            list[int]: The closest neighbor indices found, unsorted.
        """
        visited = set([ep])

        # Priority queues initialization
        # candidates: (distance, node_id) -> Min-Heap
        candidates = [(self.distance(self.data[ep], q), ep)]
        # W: (-distance, node_id) -> Max-Heap (using negative distance)
        W = [(-candidates[0][0], ep)] 

        if logger:
            logger({
                'event': 'init_search_layer',
                'layer': level,
                'ep': ep
            })

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            f_dist, f_id = W[0]

            # Early termination: if the closest candidate is worse than the worst 
            # node in our current top-ef list, we can't improve further.
            if c_dist > -f_dist:
                break

            if logger:
                logger({
                    'event': 'visit_node',
                    'current_node': c_id
                })

            for e_id, e_dist in self.graph[level][c_id].items():
                if e_id not in visited:
                    visited.add(e_id)
                    e_dist = self.distance(self.data[e_id], q)
                    f_dist, f_id = W[0]

                    if logger:
                        logger({
                            'event': 'consider_neighbor',
                            'current_node': c_id,
                            'neighbor': e_id
                        })

                    # If e is better than the worst in W, or W is not full yet
                    if e_dist < -f_dist or len(W) < ef:
                        heapq.heappush(candidates, (e_dist, e_id))
                        heapq.heappush(W, (-e_dist, e_id))

                        if logger:
                            logger({
                                'event': 'accept_neighbor',
                                'current_node': c_id,
                                'neighbor': e_id
                            })

                        # Keep W size fixed at ef
                        if len(W) > ef:
                            reject_dist, reject_id = heapq.heappop(W)
                            if logger:
                                logger({
                                    'event': 'reject_node',
                                    'current_node': c_id,
                                    'reject_node': reject_id
                                })
                    elif logger:
                        logger({
                            'event': 'reject_neighbor',
                            'current_node': c_id,
                            'neighbor': e_id
                        })

        nearest_neighbors = heapq.nlargest(ef, W)
        nearest_neighbors = [nn[1] for nn in nearest_neighbors]

        return nearest_neighbors

    def knn_search(self, q: ArrayLike, K: int = 1, logger=None) -> list[int]:
        """
        Performs K-Nearest Neighbors (KNN) search using the HNSW index.

        Traverses the graph from the top layer down to 0. At each layer, it finds 
        the local nearest neighbor to serve as the entry point for the next layer.
        At Layer 0, a broader search (`ef` size) is performed to find the top K results.

        Args:
            q (ArrayLike): The query/target vector.
            K (int, optional): The number of nearest neighbors to retrieve. Defaults to 1.
            logger (callable, optional): Callback for visualization logging. Defaults to None.

        Returns:
            list[int]: Indices of the K nearest neighbors, sorted by distance.
        """
        W = []
        ep = self.entry_point
        L = self.max_level

        if logger:
            logger({
                'event': 'init_knn_search',
                'entry_point': ep,
                'max_level': L
            })

        # Coarse Search: Zoom in from top layer to layer 1
        for level in range(L, 0, -1):
            if logger and level != L:
                logger({
                    'event': 'layer_transition',
                    'from_layer': level + 1,
                    'to_layer': level,
                    'ep': ep
                })
            # ef=1 implies greedy search for the single best entry point
            W = self.search_layer(q, ep, 1, level, logger=logger)
            ep = W[0]

        if logger:
            logger({
                'event': 'layer_transition',
                'from_layer': 1,
                'to_layer': 0,
                'ep': ep
            })

        # Fine Search: Perform detailed search at layer 0
        W = self.search_layer(q, ep, max(K, self.ef), 0, logger=logger)

        if logger:
            reject_nodes = []
            for i in range(K, len(W)):
                reject_nodes.append(W[i])

            logger({
                'event': 'reject_nodes',
                'reject_nodes': reject_nodes
            })

        return W[:K]

    def select_neighbors(self,
                        #  q: ArrayLike,
                         q_id: int,
                         W: list[int],
                         M: int,
                         level: int,
                         logger=None) -> tuple[list[int], list[float]]:
        """
        Wrapper to select the best neighbors for a node.
        Defaults to the Heuristic strategy which balances proximity and spatial diversity.
        """
        # proba = self.rng.uniform(0.0, 1.0)
        # if proba <= 0.5:
        #     return self.select_neighbors_simple(q_id, W, M, logger)
        return self.select_neighbors_heuristic(q_id, W, M, level, logger)

    def select_neighbors_simple(self,
                                # q: ArrayLike,
                                q_id: int,
                                C: list[int],
                                M: int,
                                logger=None) -> tuple[list[int], list[float]]:
        """
        Selects neighbors using a naive strategy (Pure Proximity).

        Simply picks the `M` closest candidates by distance. 
        Note: This method is generally inferior as it can lead to clustering 
        and poor graph navigability (Long paths).

        Args:
            q_id (int): The node ID.
            C (list[int]): Candidate neighbors.
            M (int): Number of neighbors to return.

        Returns:
            tuple[list[int], list[float]]: Tuple of (indices, distances).
        """
        candidates = []
        for c in C:
            dist = self.distance(self.data[c], self.data[q_id])
            candidates.append((dist, c))
        
        nearest = heapq.nsmallest(M, candidates)

        if logger:
            selected = set(nearest)
            for n in candidates:
                if n not in nearest:
                    logger({
                        'event': 'reject_node',
                        'id': n
                    })

        return [n[1] for n in nearest], [n[0] for n in nearest]

    def select_neighbors_heuristic(self,
                                #    q: ArrayLike,
                                   q_id: int,
                                   C: list[int],
                                   M: int,
                                   level: int,
                                   extend_candidates: bool = True,
                                   keep_pruned_connections: bool = True,
                                   logger=None) -> tuple[list[int], list[float]]:
        """
        Selects neighbors using the HNSW heuristic (Relative Neighborhood Graph approximation).

        This ensures spatial diversity by strictly accepting a neighbor `e` ONLY if 
        it is closer to the query `q` than to any already selected neighbor `r`.
        This prevents redundant connections in the same direction and maintains 
        the "Small World" navigability.

        Args:
            q_id (int): The ID of the query node.
            C (list[int]): Candidate neighbor IDs.
            M (int): Max number of neighbors to select.
            level (int): Graph level context.
            extend_candidates (bool, optional): If True, extends the candidate set by including
                neighbors of the candidates (defaults to True).
            keep_pruned_connections (bool, optional): If True, fills remaining slots in `M`
                with closest disregarded candidates to guarantee connectivity (defaults to True).

        Returns:
            tuple[list[int], list[float]]: Selected neighbor IDs and their distances.
        """
        R = []
        W = set(C)
        W_queue = []
        # Populate initial queue
        for c in C:
            heapq.heappush(W_queue, (self.distance(self.data[c], self.data[q_id]), c))

        # Heuristic: Extend candidates with their neighbors to bridge disconnected components
        if extend_candidates:
            temp_W = W.copy()
            for e in temp_W:
                if logger:
                    logger({
                        'event': 'current_node',
                        'id': e
                    })

                for en_id, en_dist in self.graph[level][e].items():
                    if en_id not in W and en_id != q_id:
                        W.add(en_id)
                        heapq.heappush(W_queue, (self.distance(self.data[en_id], self.data[q_id]), en_id))

                        if logger:
                            logger({
                                'event': 'extend_candidate',
                                'id': en_id
                            })

        W_d = [] # Queue for discarded candidates

        while len(W_queue) > 0 and len(R) < M:
            e_dist, e_id = heapq.heappop(W_queue)
            W.remove(e_id)

            if logger:
                logger({
                    'event': 'current_node',
                    'id': e_id
                })

            # Diversity Check:
            # Add e only if it is closer to q than to any node already in R.
            # This logic approximates the Relative Neighborhood Graph.
            is_good_neighbor = True
            for r_dist, r_id in R:
                dist_e_r = self.distance(self.data[e_id], self.data[r_id])
                if dist_e_r < e_dist:
                    is_good_neighbor = False
                    break

            if is_good_neighbor:
                R.append((e_dist, e_id))
                if logger:
                    logger({
                        'event': 'select_neighbor',
                        'neighbor_id': e_id,
                        'distance': e_dist
                    })
            else:
                heapq.heappush(W_d, (e_dist, e_id))

        # Fallback: If we haven't filled M connections, use the closest discarded ones
        # to ensure the node isn't isolated.
        if keep_pruned_connections:
            while len(W_d) > 0 and len(R) < M:
                heapq.heappush(R, heapq.heappop(W_d))

        return [r[1] for r in R], [r[0] for r in R]

    # ===================================================
    # =============== VISUALIZE METHODS =================
    # ===================================================
    # Note: These methods rely on heavy external libraries (Plotly, NetworkX, Dash)
    # and are strictly used for debugging, education, or analysis of small graphs.

    def visualize_layer(self, l: int):
        """
        Visualizes the topology of a specific graph layer using NetworkX and Matplotlib.

        Args:
            l (int): The level of the layer to visualize.
        """
        if l < 0 or l > self.max_level:
            print("Layer's level is out of range!")
            return

        fig = plt.figure(figsize=(10, 10))
        if fig and fig.canvas and fig.canvas.manager:
            fig.canvas.manager.set_window_title(f'HNSW Graph - Layer {l}')

        G = nx.Graph(name=f'Layer {l}')

        for node_id, node_neighbors in self.graph[l].items():
            G.add_node(node_id)
            for neighbor_id in node_neighbors.keys():
                G.add_edge(node_id, neighbor_id)

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos)

        plt.title(f'HNSW Graph - Layer {l}')
        fig.canvas.draw_idle()

    def visualize_layers(self, layers: list[int] | None = None):
        """
        Sequentially visualizes multiple layers of the HNSW graph.

        Args:
            layers (list[int] | None, optional): Specific levels to plot.
                If None, visualizes all layers from 0 to max_level.
        """
        if layers is None:
            layers = range(self.max_level + 1)

        plt.ion()
        for l in layers:
            self.visualize_layer(l)
        plt.ioff()
        plt.show(block=True)

    def _compute_layout(self):
        """
        Computes the consistent 3D layout for visualization.
        
        Uses Layer 0 connectivity (spring layout) to determine X, Y coordinates,
        and uses the layer index for the Z coordinate.
        """
        G0 = nx.Graph()
        if self.graph:
            for node_id in self.graph[0]:
                G0.add_node(node_id)
                for neighbor_id in self.graph[0][node_id]:
                    G0.add_edge(node_id, neighbor_id)

        pos_2d = nx.spring_layout(G0, dim=2, seed=42)

        node_positions = {}
        if self.graph:
            all_node_ids = set(self.graph[0].keys())
            for node_id in all_node_ids:
                x, y = pos_2d.get(node_id, (0, 0))
                max_node_level = self.level[node_id]
                for l in range(max_node_level + 1):
                    node_positions[(node_id, l)] = (x, y, l)
        
        return node_positions

    def _get_layer_color(self, level: int) -> str:
        """Returns a color string based on the layer level (Darker blue for higher levels)."""
        cmap = plt.get_cmap('Blues')
        intensity = 0.4 + 0.5 * (level / self.max_level) if self.max_level > 0 else 0.6
        rgba = cmap(intensity)
        return f'rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})'

    def _get_node_coords(self, node_positions, items, node_distances=None):
        """Extracts coordinate arrays and hover text for a set of nodes."""
        xs, ys, zs, items_text = [], [], [], []
        for node_id, l in items:
            if (node_id, l) in node_positions:
                x, y, z = node_positions[(node_id, l)]
                xs.append(x); ys.append(y); zs.append(z)
                dist_info = ""
                if node_distances:
                     dist = node_distances.get(node_id, float('inf'))
                     dist_info = f"<br>Dist: {dist:.4f}"
                items_text.append(f"ID: {node_id}<br>Layer: {l}{dist_info}")
        return xs, ys, zs, items_text

    def _get_edge_coords(self, node_positions, items):
        """Extracts line segments (including None separators) for a set of edges."""
        xs, ys, zs = [], [], []
        for (u, v), l in items:
            if (u, l) in node_positions and (v, l) in node_positions:
                p1, p2 = node_positions[(u, l)], node_positions[(v, l)]
                xs.extend([p1[0], p2[0], None]); ys.extend([p1[1], p2[1], None]); zs.extend([p1[2], p2[2], None])
        return xs, ys, zs

    def _get_vedge_coords(self, node_positions, items):
        """Extracts line segments for vertical edges (between layers)."""
        xs, ys, zs = [], [], []
        for (u1, l1), (u2, l2) in items:
            if (u1, l1) in node_positions and (u2, l2) in node_positions:
                p1, p2 = node_positions[(u1, l1)], node_positions[(u2, l2)]
                xs.extend([p1[0], p2[0], None]); ys.extend([p1[1], p2[1], None]); zs.extend([p1[2], p2[2], None])
        return xs, ys, zs

    def _reconstruct_path_coords(self, node_positions, active_nodes, parent_map):
        """Reconstructs the search path visualization from parent pointers."""
        path_x, path_y, path_z = [], [], []
        visited_edges = set()
        for node in active_nodes:
            curr = node
            while curr in parent_map:
                parent = parent_map[curr]
                edge_sig = tuple(sorted((curr, parent)))
                if edge_sig not in visited_edges:
                    visited_edges.add(edge_sig)
                    if curr in node_positions and parent in node_positions:
                        cx, cy, cz = node_positions[curr]
                        px, py, pz = node_positions[parent]
                        path_x.extend([px, cx, None]); path_y.extend([py, cy, None]); path_z.extend([pz, cz, None])
                curr = parent
        return path_x, path_y, path_z

    def _create_base_traces(self, node_positions, edge_opacity=1.0, plane_opacity=0.1, node_opacity=1.0, node_distances=None,
                          max_level=None, exclude_nodes=None, custom_edges=None, entry_point_config=None, draw_nodes=True):
        """Generates the static Plotly traces (planes, nodes, edges)."""
        traces = []
        if max_level is None: max_level = self.max_level
        if exclude_nodes is None: exclude_nodes = set()
        
        ep_id = self.entry_point; ep_level = self.max_level
        if entry_point_config: ep_id, ep_level = entry_point_config

        # Initialize edge lists
        x_edges, y_edges, z_edges = [], [], []
        x_vedges, y_vedges, z_vedges = [], [], []

        # 1. Calculate Bounds
        xs = [pos[0] for pos in node_positions.values()]
        ys = [pos[1] for pos in node_positions.values()]
        padding = 0.5
        if xs and ys:
            x_min, x_max = min(xs) - padding, max(xs) + padding
            y_min, y_max = min(ys) - padding, max(ys) + padding
        else:
            x_min, x_max, y_min, y_max = -1, 1, -1, 1

        # 2. Edges & Vertical Edges
        if custom_edges is not None:
            for ((u, v), l) in custom_edges:
                if (u, l) in node_positions and (v, l) in node_positions:
                    x1, y1, z1 = node_positions[(u, l)]
                    x2, y2, z2 = node_positions[(v, l)]
                    x_edges.extend([x1, x2, None]); y_edges.extend([y1, y2, None]); z_edges.extend([z1, z2, None])
            
            for (nid, l) in node_positions:
                 if nid not in exclude_nodes and l < max_level and (nid, l+1) in node_positions:
                     x, y, z = node_positions[(nid, l)]
                     x_vedges.extend([x, x, None]); y_vedges.extend([y, y, None]); z_vedges.extend([l, l+1, None])
        else:
            all_node_ids = set()
            if self.graph: all_node_ids = set(self.graph[0].keys())
            
            for l in range(max_level + 1):
                if l >= len(self.graph): continue
                for node_id in self.graph[l]:
                    if node_id in exclude_nodes: continue
                    # Horizontal
                    if (node_id, l) in node_positions:
                        x, y, z = node_positions[(node_id, l)]
                        if node_id in self.graph[l]:
                            for neighbor_id in self.graph[l][node_id]:
                                if neighbor_id in exclude_nodes: continue
                                if node_id < neighbor_id and (neighbor_id, l) in node_positions:
                                    xn, yn, zn = node_positions[(neighbor_id, l)]
                                    x_edges.extend([x, xn, None]); y_edges.extend([y, yn, None]); z_edges.extend([z, zn, None])
                    # Vertical
                    if l < max_level and (node_id, l+1) in node_positions:
                        if (node_id, l) in node_positions:
                             x, y, z = node_positions[(node_id, l)]
                             x_vedges.extend([x, x, None]); y_vedges.extend([y, y, None]); z_vedges.extend([l, l+1, None])

        if x_edges:
            traces.append(go.Scatter3d(x=x_edges, y=y_edges, z=z_edges, mode='lines', line=dict(color='#cccccc', width=1), opacity=edge_opacity, hoverinfo='none', name='Edges'))
        if x_vedges:
            traces.append(go.Scatter3d(x=x_vedges, y=y_vedges, z=z_vedges, mode='lines', line=dict(color='#cccccc', width=1, dash='dash'), opacity=edge_opacity, hoverinfo='none', name='Vertical Edges'))

        # 3. Planes & Nodes
        for l in range(max_level + 1):
            color_hex = self._get_layer_color(l)
            # Plane
            traces.append(go.Surface(
                x=[[x_min, x_max], [x_min, x_max]], y=[[y_min, y_min], [y_max, y_max]], z=[[l, l], [l, l]],
                opacity=plane_opacity, showscale=False, colorscale=[[0, color_hex], [1, color_hex]], hoverinfo='none', name=f'Plane L{l}'
            ))

            # Nodes
            if draw_nodes:
                xn, yn, zn, tn = [], [], [], []
                for (nid, layer), pos in node_positions.items():
                    if layer != l: continue
                    if nid in exclude_nodes: continue
                    if nid == ep_id and layer == ep_level: continue 

                    xn.append(pos[0]); yn.append(pos[1]); zn.append(pos[2])
                    dist = ""
                    if node_distances and nid in node_distances: dist = f"<br>Dist: {node_distances.get(nid, 0):.4f}"
                    tn.append(f"ID: {nid}<br>Level: {l}{dist}")
                
                if xn:
                    traces.append(go.Scatter3d(
                        x=xn, y=yn, z=zn, mode='markers',
                        marker=dict(size=5, color=color_hex, line=dict(color='#333', width=0.5), opacity=node_opacity),
                        text=tn, hoverinfo='text', name=f'Nodes L{l}'
                    ))

        # Entry Point Base
        if draw_nodes and ep_id != -1 and (ep_id, ep_level) in node_positions:
             if ep_id not in exclude_nodes:
                x, y, z = node_positions[(ep_id, ep_level)]
                dist = ""
                if node_distances: dist = f"<br>Dist: {node_distances.get(ep_id, 0):.4f}"
                traces.append(go.Scatter3d(
                    x=[x], y=[y], z=[z], mode='markers',
                    marker=dict(size=6, color='deeppink', line=dict(color='white', width=1), opacity=node_opacity),
                    text=[f"Entry Point ID: {ep_id}<br>Layer: {ep_level}{dist}"], hoverinfo='text', name='Entry Point Base'
                ))

        return traces

    def visualize_hierarchical_graph(self):
        """
        Launches a Dash application to interactively visualize the HNSW graph structure.
        Features hover effects to highlight nodes across all layers.
        """
        # Create layout and traces
        node_positions = self._compute_layout()
        base_traces = self._create_base_traces(node_positions)

        # Modify base traces to disable hover on nodes
        for trace in base_traces:
            if trace.name in ['Nodes', 'Entry Point']:
                trace.hoverinfo = 'skip'

        # Create Invisible Hitboxes for interaction
        hitbox_x, hitbox_y, hitbox_z = [], [], []
        hitbox_customdata = []

        for (node_id, level), (x, y, z) in node_positions.items():
            hitbox_x.append(x)
            hitbox_y.append(y)
            hitbox_z.append(z)
            hitbox_customdata.append(node_id)

        hitbox_trace = go.Scatter3d(
            x=hitbox_x, y=hitbox_y, z=hitbox_z,
            mode='markers',
            marker=dict(
                symbol='circle',
                size=30,  # Hitbox size
                color='white',
                opacity=0.0,   # Invisible
            ),
            customdata=hitbox_customdata,
            hovertemplate="<b>ID: %{customdata}</b><br>Level: %{z}<extra></extra>",
            name='Hitbox'
        )

        fig_traces = base_traces + [hitbox_trace]

        # Create Figure
        layout = go.Layout(
            title=f'HNSW Interactive Dashboard',
            scene=dict(
                xaxis=dict(title='', showbackground=False, showgrid=False, zeroline=False, visible=False),
                yaxis=dict(title='', showbackground=False, showgrid=False, zeroline=False, visible=False),
                zaxis=dict(title='Level', showbackground=False, showgrid=False, zeroline=False, nticks=self.max_level + 2),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode='closest',
            uirevision='constant'
        )

        base_fig = go.Figure(data=fig_traces, layout=layout)

        # Initialize Dash
        app = Dash(__name__)

        app.layout = html.Div([
            html.Title("HNSW Interactive Dashboard"),
            dcc.Graph(
                id='hnsw-viz',
                figure=base_fig,
                style={'height': '95vh'},
                clear_on_unhover=True
            )
        ])

        @app.callback(
            Output('hnsw-viz', 'figure'),
            Input('hnsw-viz', 'hoverData')
        )
        def update_highlight(hoverData):
            if hoverData is None:
                return base_fig

            try:
                point = hoverData['points'][0]
                if 'customdata' not in point: return no_update
                node_id = point['customdata']
            except:
                return no_update

            # Create Highlight
            new_fig = go.Figure(base_fig)

            if (node_id, 0) not in node_positions:
                return no_update

            x, y, _ = node_positions[(node_id, 0)]
            max_l = self.level[node_id]

            hl_x = [x] * (max_l + 1)
            hl_y = [y] * (max_l + 1)
            hl_z = list(range(max_l + 1))

            # Highlight Vertical Line
            new_fig.add_trace(go.Scatter3d(
                x=hl_x, y=hl_y, z=hl_z,
                mode='lines',
                line=dict(color='yellow', width=8),
                opacity=0.8, hoverinfo='skip', name='Highlight Line'
            ))

            # Highlight Node Halo
            new_fig.add_trace(go.Scatter3d(
                x=hl_x, y=hl_y, z=hl_z,
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=12,
                    color='yellow',
                    opacity=0.6,
                    line=dict(width=0)
                ),
                hoverinfo='skip', name='Highlight Node'
            ))

            return new_fig

        app.run(debug=True, use_reloader=False)



    def visualize_search(self, q: ArrayLike, k: int = 1) -> list[int]:
        """
        Creates a frame-by-frame 3D animation of the KNN search process.

        Visualizes:
        - The current candidate node being visited.
        - Accepted vs Rejected edges/nodes.
        - The "Frontier" (Min-heap) and "Best Results" (Max-heap).
        - The final path taken through layers.

        Args:
            q (ArrayLike): Query vector.
            k (int): Number of neighbors to find.
        
        Returns:
            list[int]: The resulting neighbor IDs (for verification).
        """
        q = np.array(q)
        search_log = []
        def logger(event): search_log.append(event)
        result_ids = self.knn_search(q, k, logger=logger)

        # 1. Graph Layout & Geometry Pre-computation
        node_distances = {}
        for idx, vec in enumerate(self.data):
            node_distances[idx] = self.distance(vec, q)

        node_positions = self._compute_layout()
        
        # 2. Generate Base Static Traces (Edges & Planes)
        base_traces = self._create_base_traces(
            node_positions, 
            edge_opacity=0.3, plane_opacity=0.05, node_opacity=0.1, node_distances=node_distances,
            draw_nodes=False
        )

        # --- Frame Generation Helper ---
        def create_traces(
            sel_e_x, sel_e_y, sel_e_z, sel_ve_x, sel_ve_y, sel_ve_z, 
            rej_e_x, rej_e_y, rej_e_z, con_e_x, con_e_y, con_e_z, 
            sel_n_x, sel_n_y, sel_n_z, sel_n_text, 
            rej_n_x, rej_n_y, rej_n_z, rej_n_text, 
            con_n_x, con_n_y, con_n_z, con_n_text, 
            cur_x, cur_y, cur_z, cur_text, 
            path_x, path_y, path_z, 
            blue_path_x, blue_path_y, blue_path_z,
            visited_history, 
            best_nodes_accumulated, 
            background_nodes,
            entry_point_config, # (id, level)
            is_final_frame=False,
            final_result_ids=None,
            active_highlight_set=None # Set of (nid, layer) currently highlighted (sel, rej, con, cur)
        ):
            if active_highlight_set is None: active_highlight_set = set()
            traces = list(base_traces)
            
            # 0. Dynamic Background Nodes (Faded)
            # Group by layer for efficiency
            bg_layer_groups = {l: [] for l in range(self.max_level + 1)}
            for nid, l in background_nodes:
                if l in bg_layer_groups: bg_layer_groups[l].append((nid, l))
            
            for l in range(self.max_level + 1):
                if bg_layer_groups[l]:
                    bx, by, bz, bt = self._get_node_coords(node_positions, bg_layer_groups[l], node_distances)
                    color = self._get_layer_color(l)
                    traces.append(go.Scatter3d(
                        x=bx, y=by, z=bz, mode='markers',
                        marker=dict(size=5, color=color, line=dict(color='#333', width=0.5), opacity=0.1),
                        text=bt, hoverinfo='text', name=f'Nodes L{l}'
                    ))
            
            # 1. Explored Nodes (Visited History)
            layer_groups = {l: [] for l in range(self.max_level + 1)}
            ep_active_coords = [] 
            
            # Explicit Entry Point Draw
            if entry_point_config:
                 eid, elvl = entry_point_config
                 if (eid, elvl) in node_positions:
                     ex, ey, ez = node_positions[(eid, elvl)]
                     edist = node_distances.get(eid, float('inf'))
                     ep_active_coords.append((ex, ey, ez, f"Entry Point ID: {eid}<br>Layer: {elvl}<br>Dist: {edist:.4f}"))

            if visited_history:
                for nid, lvl in visited_history:
                    # Logic Entry Point (skip if already handled)
                    if entry_point_config and nid == entry_point_config[0] and lvl == entry_point_config[1]:
                        continue 
                    
                    if (nid, lvl) in best_nodes_accumulated:
                        continue
                    
                    if (nid, lvl) in active_highlight_set:
                        continue

                    if lvl in layer_groups:
                        layer_groups[lvl].append((nid, lvl))
            
            for l in range(self.max_level + 1):
                vx, vy, vz, vt = [], [], [], []
                if layer_groups[l]:
                     vx, vy, vz, vt = self._get_node_coords(node_positions, layer_groups[l], node_distances)
                color = self._get_layer_color(l)
                traces.append(go.Scatter3d(
                    x=vx, y=vy, z=vz, mode='markers',
                    marker=dict(size=5, color=color, symbol='circle', line=dict(color='#333', width=0.5), opacity=1.0),
                    text=vt, hoverinfo='text', name=f'Explored L{l}'
                ))

            # 2. Persistent Best Nodes
            best_x, best_y, best_z, best_t = self._get_node_coords(node_positions, best_nodes_accumulated, node_distances)
            if best_x:
                traces.append(go.Scatter3d(
                    x=best_x, y=best_y, z=best_z, mode='markers',
                    marker=dict(size=6, color='green', symbol='circle', line=dict(color='black', width=1), opacity=1.0),
                    text=best_t, hoverinfo='text', name='Best of Layers'
                ))

            # 3. Active Entry Point
            ep_x, ep_y, ep_z, ep_t = [], [], [], []
            if ep_active_coords:
                 ep_x = [item[0] for item in ep_active_coords]
                 ep_y = [item[1] for item in ep_active_coords]
                 ep_z = [item[2] for item in ep_active_coords]
                 ep_t = [item[3] for item in ep_active_coords]

            traces.append(go.Scatter3d(
                x=ep_x, y=ep_y, z=ep_z, mode='markers',
                marker=dict(size=8, color='deeppink', symbol='circle', line=dict(color='white', width=1.5), opacity=1.0),
                text=ep_t, hoverinfo='text', name='Entry Point Active'
            ))

            # 4. Search Trajectories
            traces.append(go.Scatter3d(x=blue_path_x, y=blue_path_y, z=blue_path_z, mode='lines', line=dict(color='royalblue', width=3), opacity=0.4, name='Explored Path', hoverinfo='none'))
            traces.append(go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines', line=dict(color='#FF8C00', width=6), opacity=1.0, name='Search Path', hoverinfo='none'))
            
            # 5. Active Search State Visualization
            if not is_final_frame:
                traces.extend([
                    go.Scatter3d(x=sel_e_x, y=sel_e_y, z=sel_e_z, mode='lines', line=dict(color='green', width=4), name='Select Edge'),
                    go.Scatter3d(x=sel_ve_x, y=sel_ve_y, z=sel_ve_z, mode='lines', line=dict(color='green', width=4, dash='dash'), name='Select V-Edge'),
                    go.Scatter3d(x=rej_e_x, y=rej_e_y, z=rej_e_z, mode='lines', line=dict(color='red', width=3), name='Reject Edge'),
                    go.Scatter3d(x=con_e_x, y=con_e_y, z=con_e_z, mode='lines', line=dict(color='yellow', width=3), name='Consider Edge'),
                    
                    go.Scatter3d(x=sel_n_x, y=sel_n_y, z=sel_n_z, mode='markers', marker=dict(size=6, color='green', symbol='circle', line=dict(color='black', width=0.5)), text=sel_n_text, hoverinfo='text', name='Select Node'),
                    go.Scatter3d(x=rej_n_x, y=rej_n_y, z=rej_n_z, mode='markers', marker=dict(size=6, color='red', symbol='circle', line=dict(color='black', width=0.5)), text=rej_n_text, hoverinfo='text', name='Reject Node'),
                    go.Scatter3d(x=con_n_x, y=con_n_y, z=con_n_z, mode='markers', marker=dict(size=6, color='yellow', symbol='circle', line=dict(color='black', width=0.5)), text=con_n_text, hoverinfo='text', name='Consider Node'),
                ])
                
                # Current Focus
                if cur_x:
                    traces.append(go.Scatter3d(
                        x=cur_x, y=cur_y, z=cur_z, mode='markers',
                        marker=dict(size=12, color='green', symbol='circle', opacity=0.5, line=dict(width=0)), 
                        hoverinfo='skip', name='Current Focus Halo'
                    ))
                    traces.append(go.Scatter3d(
                        x=cur_x, y=cur_y, z=cur_z, mode='markers',
                        marker=dict(size=5, color='green', symbol='circle', line=dict(color='black', width=1), opacity=1.0),
                        text=cur_text, hoverinfo='text', name='Current Focus'
                    ))
                else:
                    traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder Halo'))
                    traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder Node'))
                    
            else:
                for _ in range(9): traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder'))

            # 6. Final Results Highlighting
            if is_final_frame and final_result_ids:
                res_items = [(rid, 0) for rid in final_result_ids]
                rx, ry, rz, rt = self._get_node_coords(node_positions, res_items, node_distances)
                if rx:
                    traces.append(go.Scatter3d(
                        x=rx, y=ry, z=rz, mode='markers', 
                        marker=dict(size=12, color='yellow', symbol='circle', opacity=0.5, line=dict(width=0)), 
                        text=rt, hoverinfo='text', hoverlabel=dict(bgcolor='green'), name='Result Halo'
                    ))
                    traces.append(go.Scatter3d(
                        x=rx, y=ry, z=rz, mode='markers', 
                        marker=dict(size=6, color='green', symbol='circle', line=dict(color='black', width=1)), 
                        hoverinfo='skip', name='Top-K Results'
                    ))
            else:
                traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder Halo'))
                traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder TopK'))

            return traces

        # --- Start Animation Frames ---
        frames = []
        all_node_pairs = set(node_positions.keys())
        global_ep = (int(self.entry_point), int(self.max_level))
        
        # 0. Initial Blank Frame (Step 0)
        # In step 0, nothing is active, so all nodes are background EXCEPT Entry Point
        empty_dynamic_args = [[] for _ in range(28)]
        frames.append(go.Frame(data=create_traces(
            *empty_dynamic_args,
            [], [], [], [], [], [],
            set(), set(),
            list(all_node_pairs - {global_ep}), # background_nodes (exclude EP)
            global_ep,
            False, None, set()
        ), name='step_0'))

        # Loop
        current_node = -1; current_layer = -1
        current_W_set = set(); node_parents = {}; visited_history = set()
        best_nodes_accumulated = set()

        for i, log in enumerate(search_log):
            event = log['event']
            current_selected_edge = set(); current_selected_vedge = set()
            current_considered_edge = set(); current_rejected_edge = set()
            current_considered_node = set(); current_rejected_node = set()

            if event == 'init_knn_search':
                current_node = log['entry_point']; current_layer = log['max_level']
                current_W_set = {(current_node, current_layer)}
                visited_history.add((current_node, current_layer))
            elif event == 'layer_transition':
                best_nodes_accumulated.update(current_W_set)
                u = log['ep']; l_from = log['from_layer']; l_to = log['to_layer']
                node_parents[(u, l_to)] = (u, l_from)
                current_selected_vedge.add(((u, l_from), (u, l_to)))
                current_layer = l_to; current_node = u
                current_W_set = {(current_node, current_layer)}
                visited_history.add((current_node, current_layer))
            elif event == 'init_search_layer':
                current_layer = log['layer']; current_node = log['ep']
                current_W_set = {(current_node, current_layer)}
                visited_history.add((current_node, current_layer))
            elif event == 'visit_node':
                current_node = log['current_node']
                visited_history.add((current_node, current_layer))
            elif event == 'consider_neighbor':
                neighbor = log['neighbor']; u, v = sorted((current_node, neighbor))
                current_considered_edge.add(((u, v), current_layer))
                current_considered_node.add((neighbor, current_layer))
                if (neighbor, current_layer) not in node_parents:
                    node_parents[(neighbor, current_layer)] = (current_node, current_layer)
                visited_history.add((neighbor, current_layer))
            elif event == 'accept_neighbor':
                neighbor = log['neighbor']; u, v = sorted((current_node, neighbor))
                current_selected_edge.add(((u, v), current_layer))
                current_W_set.add((neighbor, current_layer))
            elif event == 'reject_neighbor':
                neighbor = log['neighbor']; u, v = sorted((current_node, neighbor))
                current_rejected_node.add((neighbor, current_layer))
                current_rejected_edge.add(((u, v), current_layer))
            elif event == 'reject_node':
                reject_node = log['reject_node']
                current_rejected_node.add((reject_node, current_layer))
                current_W_set.discard((reject_node, current_layer))
            elif event == 'reject_nodes':
                for id in log['reject_nodes']:
                    current_rejected_node.add((id, current_layer))
                    current_W_set.discard((id, current_layer))

            # Coords
            sel_e_x, sel_e_y, sel_e_z = self._get_edge_coords(node_positions, current_selected_edge)
            sel_ve_x, sel_ve_y, sel_ve_z = self._get_vedge_coords(node_positions, current_selected_vedge)
            rej_e_x, rej_e_y, rej_e_z = self._get_edge_coords(node_positions, current_rejected_edge)
            con_e_x, con_e_y, con_e_z = self._get_edge_coords(node_positions, current_considered_edge)
            sel_n_x, sel_n_y, sel_n_z, sel_n_text = self._get_node_coords(node_positions, current_W_set, node_distances)
            rej_n_x, rej_n_y, rej_n_z, rej_n_text = self._get_node_coords(node_positions, current_rejected_node, node_distances)
            con_n_x, con_n_y, con_n_z, con_n_text = self._get_node_coords(node_positions, current_considered_node, node_distances)
            cur_x, cur_y, cur_z, cur_text = self._get_node_coords(node_positions, [(current_node, current_layer)], node_distances)

            active_targets = current_W_set.copy(); active_targets.add((current_node, current_layer))
            path_x, path_y, path_z = self._reconstruct_path_coords(node_positions, active_targets, node_parents)
            blue_targets = visited_history.copy()
            blue_path_x, blue_path_y, blue_path_z = self._reconstruct_path_coords(node_positions, blue_targets, node_parents)
            
            # --- Active & Background Logic ---
            active_set = set(visited_history)
            active_set.update(best_nodes_accumulated)
            active_set.update(current_W_set)
            active_set.update(current_rejected_node)
            active_set.update(current_considered_node)
            if current_node != -1: active_set.add((current_node, current_layer))
            active_set.add(global_ep) # Always active
            
            bg_nodes = list(all_node_pairs - active_set)

            # 7. Active Highlight Set Construction (Z-Order Management)
            # define set of active nodes to exclude from background/explored tracing
            highlight_set = set()
            highlight_set.update(current_W_set)
            highlight_set.update(current_rejected_node)
            highlight_set.update(current_considered_node)
            if current_node != -1: highlight_set.add((current_node, current_layer))

            frames.append(go.Frame(data=create_traces(
                sel_e_x, sel_e_y, sel_e_z, sel_ve_x, sel_ve_y, sel_ve_z,
                rej_e_x, rej_e_y, rej_e_z, con_e_x, con_e_y, con_e_z,
                sel_n_x, sel_n_y, sel_n_z, sel_n_text,
                rej_n_x, rej_n_y, rej_n_z, rej_n_text,
                con_n_x, con_n_y, con_n_z, con_n_text,
                cur_x, cur_y, cur_z, cur_text,
                path_x, path_y, path_z, blue_path_x, blue_path_y, blue_path_z,
                visited_history,
                best_nodes_accumulated.copy(),
                bg_nodes,
                global_ep,
                False, None, 
                highlight_set # Active set for Z-order filtering
            ), name=f'step_{i+1}')) # Name shifted by 1

        # Final Frame
        final_targets = set([(rid, 0) for rid in result_ids if (rid, 0) in node_positions])
        final_path_x, final_path_y, final_path_z = self._reconstruct_path_coords(node_positions, final_targets, node_parents)
        final_blue_x, final_blue_y, final_blue_z = self._reconstruct_path_coords(node_positions, visited_history, node_parents)
        best_nodes_accumulated.update(current_W_set)
        
        final_active_set = set(visited_history)
        final_active_set.update(best_nodes_accumulated)
        final_active_set.update(final_targets)
        final_active_set.add(global_ep)
        final_bg_nodes = list(all_node_pairs - final_active_set)

        # 8. Final Frame Entry Point Persistence
        # Ensure Entry Point is explicitly highlighted in the final state
        final_highlights = final_targets.copy()
        final_highlights.add(global_ep) 

        frames.append(go.Frame(data=create_traces(
            *empty_dynamic_args,
            final_path_x, final_path_y, final_path_z,
            final_blue_x, final_blue_y, final_blue_z,
            visited_history,
            best_nodes_accumulated,
            final_bg_nodes,
            global_ep,
            True,
            result_ids,
            final_highlights # Explicitly include EP to override background drawing
        ), name='final'))

        # Sliders
        sliders = [{
            "active": 0, "yanchor": "top", "xanchor": "left",
            "currentvalue": {"font": {"size": 20}, "prefix": "Step: ", "visible": True, "xanchor": "right"},
            "transition": {"duration": 0}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0,
            "steps": []
        }]

        for i, frame in enumerate(frames):
            step = {
                "args": [[frame.name], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
                "label": str(i) if frame.name != 'final' else 'Final', "method": "animate"
            }
            sliders[0]["steps"].append(step)

        # Init Figure (With Blank State)
        init_traces = create_traces(
            *empty_dynamic_args, 
            [], [], [], [], [], [],
            set(), set(),
            list(all_node_pairs - {global_ep}),
            global_ep,
            False, None, set()
        )

        fig = go.Figure(
            data=init_traces,
            layout=go.Layout(
                title=f'HNSW Search Visualization (k={k})',
                scene=dict(
                    xaxis=dict(visible=False), yaxis=dict(visible=False), 
                    zaxis=dict(title='Level', showbackground=False, nticks=self.max_level + 2),
                    aspectmode='manual', aspectratio=dict(x=1, y=1, z=1) 
                ),
                uirevision='constant', hovermode='closest', sliders=sliders,
                updatemenus=[{'type': 'buttons', 'showactive': False, 'buttons': [
                    {'label': 'Play', 'method': 'animate', 'args': [None, {
                        'frame': {'duration': 250, 'redraw': True}, 
                        'fromcurrent': True, 'transition': {'duration': 250, 'easing': 'quadratic-in-out'}
                    }]},
                    {'label': 'Pause', 'method': 'animate', 'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False}, 
                        'mode': 'immediate', 'transition': {'duration': 0}
                    }]}
                ]}]
            ),
            frames=frames
        )
        fig.show()
        return result_ids

    def visualize_insert(self, q: ArrayLike):
        """
        Creates a frame-by-frame 3D animation of the node insertion process.

        Shows the new node appearing progressively from top to bottom layers,
        with neighbors and connections highlighted only during each layer's processing.

        Args:
            q (ArrayLike): The vector to insert.

        Returns:
            int: The ID of the inserted node.
        """
        q = np.array(q)
        
        # Save state before insertion
        initial_element_count = self.cur_element_count
        initial_max_level = self.max_level
        initial_entry_point = self.entry_point
        
        # Pre-computation: Layout and Geometry (State T0)
        pre_node_positions = self._compute_layout()
        
        node_distances = {}
        for idx, vec in enumerate(self.data):
            node_distances[idx] = self.distance(vec, q)
            
        # Capture existing edges for base_traces (Graph State T0)
        pre_insert_edges = set()
        for l in range(len(self.graph)):
            for u, neighbors in self.graph[l].items():
                for v in neighbors:
                    if u < v: 
                         pre_insert_edges.add(((u, v), l))
        
        # Base traces will be created after insertion to account for new layers
        base_traces = []
        
        # Execute Insertion Process with Event Logging
        insert_log = []
        def logger(event): insert_log.append(event)
        
        self.insert(q, logger=logger)
        
        new_node_id = initial_element_count
        node_distances[new_node_id] = 0.0
        
        # Post-computation: Update Layout with New Node
        post_node_positions = self._compute_layout()
        node_positions = pre_node_positions.copy()
        for (nid, layer), pos in post_node_positions.items():
            if nid == new_node_id:
                node_positions[(nid, layer)] = pos
        
        # Create visualization
        final_max_level = self.max_level
        # Use shared helper for base traces (planes, nodes, edges)
        base_traces = self._create_base_traces(
            node_positions,
            max_level=final_max_level,
            exclude_nodes={new_node_id},
            custom_edges=pre_insert_edges,
            entry_point_config=(initial_entry_point, initial_max_level),
            node_distances=node_distances,
            draw_nodes=False
        )

        def create_frame(
            # -- Search Visualization Args (Same as visualize_search) --
            sel_e_x, sel_e_y, sel_e_z, sel_ve_x, sel_ve_y, sel_ve_z, 
            rej_e_x, rej_e_y, rej_e_z, con_e_x, con_e_y, con_e_z, 
            sel_n_x, sel_n_y, sel_n_z, sel_n_text, 
            rej_n_x, rej_n_y, rej_n_z, rej_n_text, 
            con_n_x, con_n_y, con_n_z, con_n_text, 
            cur_x, cur_y, cur_z, cur_text, 
            path_x, path_y, path_z, 
            visited_history, 
            best_nodes_accumulated, 
            background_nodes,
            entry_point_config, # (id, level)
            
            # -- Insert Visualization Args --
            new_node_layers,
            display_neighbors,
            new_edges_persistent,
            removed_edges_all,
            is_final_frame=False,
            active_highlight_set=None
        ):
            """Generates the Plotly traces for a single animation frame."""
            if active_highlight_set is None: active_highlight_set = set()
            traces = list(base_traces)
            
            # 0. Background Nodes (Inactive Elements)
            # Render faded nodes that are not part of the active search/insert context
            bg_layer_groups = {l: [] for l in range(initial_max_level + 1)}
            for nid, l in background_nodes:
                if l in bg_layer_groups: bg_layer_groups[l].append((nid, l))
            
            for l in range(initial_max_level + 1):
                if bg_layer_groups[l]:
                    bx, by, bz, bt = self._get_node_coords(node_positions, bg_layer_groups[l], node_distances)
                    color = self._get_layer_color(l)
                    traces.append(go.Scatter3d(
                        x=bx, y=by, z=bz, mode='markers',
                        marker=dict(size=5, color=color, line=dict(color='#333', width=0.5), opacity=1.0),
                        text=bt, hoverinfo='text', name=f'Nodes L{l}'
                    ))

            # 1. Explored Nodes (Visited History)
            # Render visited nodes, excluding those currently active or highlighted
            layer_groups = {l: [] for l in range(initial_max_level + 1)}
            ep_active_coords = [] 
            
            # Explicit Entry Point Draw
            if entry_point_config:
                 eid, elvl = entry_point_config
                 if (eid, elvl) in node_positions:
                     ex, ey, ez = node_positions[(eid, elvl)]
                     edist = node_distances.get(eid, float('inf'))
                     ep_active_coords.append((ex, ey, ez, f"Entry Point ID: {eid}<br>Layer: {elvl}<br>Dist: {edist:.4f}"))

            if visited_history:
                for nid, lvl in visited_history:
                    # Logic Entry Point (skip)
                    if entry_point_config and nid == entry_point_config[0] and lvl == entry_point_config[1]:
                        continue 
                    
                    if (nid, lvl) in best_nodes_accumulated:
                        continue
                    
                    # Exclude the new node from explored (it is drawn separately)
                    if nid == new_node_id:
                        continue
                        
                    # Exclude neighbors (drawn separately)
                    if (nid, lvl) in display_neighbors:
                        continue
                        
                    if (nid, lvl) in active_highlight_set:
                        continue
                        
                    if lvl in layer_groups:
                        layer_groups[lvl].append((nid, lvl))
            
            for l in range(initial_max_level + 1):
                vx, vy, vz, vt = [], [], [], []
                if layer_groups[l]:
                     vx, vy, vz, vt = self._get_node_coords(node_positions, layer_groups[l], node_distances)
                     color = self._get_layer_color(l)
                     traces.append(go.Scatter3d(
                         x=vx, y=vy, z=vz, mode='markers',
                         marker=dict(size=5, color=color, symbol='circle', line=dict(color='black', width=1), opacity=1.0),
                         text=vt, hoverinfo='text', name=f'Explored L{l}'
                     ))

            # 2. Entry Point Highlighting
            # Explicitly render the entry point to ensure visibility on top of background
            if ep_active_coords:
                 ep_x, ep_y, ep_z, ep_txt = zip(*ep_active_coords)
                 traces.append(go.Scatter3d(
                     x=ep_x, y=ep_y, z=ep_z, mode='markers',
                     marker=dict(size=8, color='deeppink', line=dict(color='white', width=1)),
                     text=ep_txt, hoverinfo='text', name='Start Entry Point'
                 ))
            elif initial_entry_point != -1 and not entry_point_config:
                # Fallback if config not passed but ID exists (should cover Init Frame)
                pass
            
            # 3. Target Node Visualization (New Insertion)
            # Render the node being inserted with halo effect and vertical connections
            if new_node_layers:
                # Filter out the layer that is acting as Entry Point (to avoid double drawing/color clash)
                draw_nn_layers = [item for item in new_node_layers if not (entry_point_config and item == entry_point_config)]
                
                nxs, nys, nzs, ntxts = self._get_node_coords(node_positions, draw_nn_layers, node_distances)
                if nxs or (not draw_nn_layers and new_node_layers and entry_point_config): 
                    # If we have dots OR we filtered everything out but still have edges to draw (e.g. top node is EP)
                    # We need to ensure we enter this block to draw edges if nxs is empty but new_node_layers is not.
                    pass
                
                if new_node_layers: # Use original list for edges check
                    # Vertical edges for new node
                    vedges = []
                    sorted_layers = sorted([l for _, l in new_node_layers])
                    for i in range(len(sorted_layers) - 1):
                        vedges.append(((new_node_id, sorted_layers[i]), (new_node_id, sorted_layers[i+1])))
                    vex, vey, vez = self._get_vedge_coords(node_positions, vedges)
                    
                    traces.append(go.Scatter3d(
                        x=vex, y=vey, z=vez, mode='lines',
                        line=dict(color='cyan', width=4, dash='dash'), name='New Node Vedges', hoverinfo='none'
                    ))
                    traces.append(go.Scatter3d(
                        x=nxs, y=nys, z=nzs, mode='markers',
                        marker=dict(size=14, color='cyan', opacity=0.4, line=dict(width=0)),
                        hoverinfo='skip', name='New Node Halo'
                    ))
                    traces.append(go.Scatter3d(
                        x=nxs, y=nys, z=nzs, mode='markers',
                        marker=dict(size=8, color='cyan', line=dict(color='white', width=2)),
                        text=ntxts, hoverinfo='text', name='New Node'
                    ))
                else:
                    for _ in range(3): traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder'))
            else:
                for _ in range(3): traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder'))

            # 4. Active Search State (Edges & Nodes)
            # Visualize the dynamic search frontier (selection, rejection, consideration)
            if not is_final_frame:
                traces.extend([
                    go.Scatter3d(x=sel_e_x, y=sel_e_y, z=sel_e_z, mode='lines', line=dict(color='green', width=4), name='Select Edge'),
                    go.Scatter3d(x=sel_ve_x, y=sel_ve_y, z=sel_ve_z, mode='lines', line=dict(color='#FF8C00', width=4), name='Select V-Edge'),

                    go.Scatter3d(x=rej_e_x, y=rej_e_y, z=rej_e_z, mode='lines', line=dict(color='red', width=3), name='Reject Edge'),
                    go.Scatter3d(x=con_e_x, y=con_e_y, z=con_e_z, mode='lines', line=dict(color='yellow', width=3), name='Consider Edge'),
                    
                    go.Scatter3d(x=sel_n_x, y=sel_n_y, z=sel_n_z, mode='markers', marker=dict(size=6, color='green', symbol='circle', line=dict(color='black', width=0.5)), text=sel_n_text, hoverinfo='text', name='Select Node'),
                    go.Scatter3d(x=rej_n_x, y=rej_n_y, z=rej_n_z, mode='markers', marker=dict(size=6, color='red', symbol='circle', line=dict(color='black', width=0.5)), text=rej_n_text, hoverinfo='text', name='Reject Node'),
                    go.Scatter3d(x=con_n_x, y=con_n_y, z=con_n_z, mode='markers', marker=dict(size=6, color='yellow', symbol='circle', line=dict(color='black', width=0.5)), text=con_n_text, hoverinfo='text', name='Consider Node'),
                ])
                
                # Current Focus
                if cur_x:
                    traces.append(go.Scatter3d(
                        x=cur_x, y=cur_y, z=cur_z, mode='markers',
                        marker=dict(size=12, color='green', symbol='circle', opacity=0.5, line=dict(width=0)), 
                        hoverinfo='skip', name='Current Focus Halo'
                    ))
                    traces.append(go.Scatter3d(
                        x=cur_x, y=cur_y, z=cur_z, mode='markers',
                        marker=dict(size=5, color='green', symbol='circle', line=dict(color='black', width=1), opacity=1.0),
                        text=cur_text, hoverinfo='text', name='Current Focus'
                    ))
                else:
                    traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder Halo'))
                    traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder Node'))
            else:
                 for _ in range(9): traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder'))

            # 5. Persistent Structural Changes
            # Highlight newly added or removed connections resulting from the insertion
            if new_edges_persistent:
                nex, ney, nez = self._get_edge_coords(node_positions, new_edges_persistent)
                traces.append(go.Scatter3d(
                    x=nex, y=ney, z=nez, mode='lines',
                    line=dict(color='limegreen', width=6), opacity=1.0, name='New Connections', hoverinfo='none'
                ))
            else:
                traces.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name='Placeholder New Edges'))
            
            if removed_edges_all:
                rex, rey, rez = self._get_edge_coords(node_positions, removed_edges_all)
                traces.append(go.Scatter3d(
                    x=rex, y=rey, z=rez, mode='lines',
                    line=dict(color='red', width=6), opacity=1.0, name='Removed Connections', hoverinfo='none'
                ))
            else:
                traces.append(go.Scatter3d(x=[], y=[], z=[], mode='lines', name='Placeholder Removed Edges'))

            # 6. Neighbor Selection Highlights
            # Emphasize final selected neighbors for the current layer
            if display_neighbors:
                sxs, sys, szs, stxts = self._get_node_coords(node_positions, display_neighbors, node_distances)
                # Removed Halo Trace per user request ("Don't increase size", "No faded nodes")
                traces.append(go.Scatter3d(
                    x=sxs, y=sys, z=szs, mode='markers',
                    marker=dict(size=5, color='green', symbol='circle', line=dict(color='black', width=1), opacity=1.0),
                    text=stxts, hoverinfo='text', name='Selected Neighbors'
                ))
            else:
                traces.append(go.Scatter3d(x=[], y=[], z=[], mode='markers', name='Placeholder Node'))
            
            # 7. Search Trajectory Visualization
            # Trace the path taken during the search process
            traces.append(go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines', line=dict(color='#FF8C00', width=6), opacity=1.0, name='Search Path', hoverinfo='none'))


            # 8. Persistent Best Candidates (Layer Optima)
            # Highlight the best-found nodes across processed layers
            # Best Nodes (Exclude Neighbors to prevent double draw/fading)
            best_nodes_to_draw = best_nodes_accumulated - display_neighbors
            best_x, best_y, best_z, best_t = self._get_node_coords(node_positions, best_nodes_to_draw, node_distances)
            if best_x:
                traces.append(go.Scatter3d(
                    x=best_x, y=best_y, z=best_z, mode='markers',
                    marker=dict(size=6, color='green', symbol='circle', line=dict(color='black', width=1), opacity=1.0),
                    text=best_t, hoverinfo='text', name='Best of Layers'
                ))

            # 9. Active Entry Point Marker
            # Ensure proper Z-order rendering for the active entry point
            ep_x, ep_y, ep_z, ep_t = [], [], [], []
            if ep_active_coords:
                 ep_x = [item[0] for item in ep_active_coords]
                 ep_y = [item[1] for item in ep_active_coords]
                 ep_z = [item[2] for item in ep_active_coords]
                 ep_t = [item[3] for item in ep_active_coords]

            traces.append(go.Scatter3d(
                x=ep_x, y=ep_y, z=ep_z, mode='markers',
                marker=dict(size=8, color='deeppink', symbol='circle', line=dict(color='white', width=1.5), opacity=1.0),
                text=ep_t, hoverinfo='text', name='Entry Point Active'
            ))

            return traces
        
        # Process events and create frames
        frames = []
        all_node_pairs = set(k for k in node_positions.keys() if k[0] != new_node_id)
        
        # Initial frame - before insertion
        empty_dynamic_args = [[] for _ in range(28)] # Corresponding to the 28 args of visualize_search trace factory
        # But we added args, so we need to be careful. create_frame has different signature.
        # Let's just pass explicit empty args
        
        initial_ep_config = (initial_entry_point, initial_max_level) if initial_entry_point != -1 else None

        frames.append(go.Frame(
            data=create_frame(
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                [], [], [],
                set(), set(),
                list(all_node_pairs - ({initial_ep_config} if initial_ep_config else set())), # background_nodes
                initial_ep_config,
                [], set(), set(), set(), False, set()
            ),
            name='step_0'
        ))
        
        # State tracking (Search State)
        current_node = -1; current_layer = -1
        current_W_set = set(); node_parents = {}; visited_history = set()
        best_nodes_accumulated = set()

        # State tracking (Insert State)
        new_node_layers = []
        new_node_level = -1
        new_edges_persistent = set()
        removed_edges_all = set()
        selected_neighbors_persistent = set()
        
        for i, log in enumerate(insert_log):
            event = log['event']
            
            # -- Search specific ephemeral states (reset every step) --
            current_selected_edge = set(); current_selected_vedge = set()
            current_considered_edge = set(); current_rejected_edge = set()
            current_considered_node = set(); current_rejected_node = set()
            
            selected_neighbors_current = set() # Reset
            
            if event == 'init_insert':
                new_node_level = log['assigned_level']
                if log['entry_point'] != -1:
                    current_layer = log['current_max_level']
                    current_node = log['entry_point']
                    current_W_set = {(current_node, current_layer)}
                    visited_history.add((current_node, current_layer))
            
            elif event == 'node_insert':
                new_node_layers.append(log['level'])
            
            elif event == 'zoom_in_layer':
                prev_layer = current_layer
                current_layer = log['level']
                current_node = log['ep']
                
                # Visual Transition
                if prev_layer > current_layer:
                    # node_parents link REMOVED to keep search path per-layer (not persistent vertically)
                    current_selected_vedge.add(((current_node, prev_layer), (current_node, current_layer)))
                    visited_history.add((current_node, current_layer))
                    
                    # Remove upper layer best nodes from accumulation (return to Visited status)
                    to_remove_upper = {node for node in best_nodes_accumulated if node[1] > current_layer}
                    best_nodes_accumulated.difference_update(to_remove_upper)
                    
                    current_W_set = {(current_node, current_layer)} 

                # Ensure New Node is visible if we are at its level
                if current_layer <= new_node_level and current_layer not in new_node_layers:
                    new_node_layers.append(current_layer)
            
            elif event == 'construction_layer':
                prev_layer = current_layer
                current_layer = log['level']
                
                # Logic: Isolate New Node focus if EP not present
                if (current_node, current_layer) not in node_positions:
                     current_node = -1
                
                if current_layer not in new_node_layers:
                    new_node_layers.append(current_layer)
                
                # ALWAYS skip frame (defer to init_search_layer)
                continue

                
            elif event == 'init_search_layer':
                current_layer = log['layer']
                current_node = log['ep']
                
                # Visual Transition (Fix Vertical Path) - Construction Phase
                if prev_layer > current_layer:
                    # node_parents link REMOVED to keep search path per-layer
                    current_selected_vedge.add(((current_node, prev_layer), (current_node, current_layer)))
                    visited_history.add((current_node, current_layer))
                    
                    # Remove upper layer best nodes from accumulation (return to Visited status)
                    to_remove_upper = {node for node in best_nodes_accumulated if node[1] > current_layer}
                    best_nodes_accumulated.difference_update(to_remove_upper)

                current_W_set = {(current_node, current_layer)}
                visited_history.add((current_node, current_layer))
            
            elif event == 'visit_node':
                current_node = log['current_node']
                visited_history.add((current_node, current_layer))
            
            elif event == 'consider_neighbor':
                neighbor = log['neighbor']; u, v = sorted((current_node, neighbor))
                current_considered_edge.add(((u, v), current_layer))
                current_considered_node.add((neighbor, current_layer))
                if (neighbor, current_layer) not in node_parents:
                    node_parents[(neighbor, current_layer)] = (current_node, current_layer)
                visited_history.add((neighbor, current_layer))
            
            elif event == 'accept_neighbor':
                neighbor = log['neighbor']; u, v = sorted((current_node, neighbor))
                current_selected_edge.add(((u, v), current_layer))
                current_W_set.add((neighbor, current_layer))
            
            elif event == 'reject_neighbor':
                neighbor = log['neighbor']; u, v = sorted((current_node, neighbor))
                current_rejected_node.add((neighbor, current_layer))
                current_rejected_edge.add(((u, v), current_layer))
            
            elif event == 'reject_node':
                reject_node = log['reject_node']
                current_rejected_node.add((reject_node, current_layer))
                current_W_set.discard((reject_node, current_layer))
                
            elif event == 'neighbors_selected':
                current_node = -1 # Hide focus
                layer = log['level']
                neighbors_set = set((nid, layer) for nid in log['neighbors'])
                
                # Filter W set
                to_remove = set()
                for item in current_W_set:
                     if item[1] == layer and item not in neighbors_set:
                          to_remove.add(item)
                current_W_set.difference_update(to_remove)
                current_rejected_node.update(to_remove)
                
                selected_neighbors_current = neighbors_set
                selected_neighbors_persistent.update(neighbors_set)

            elif event == 'add_connection':
                current_node = -1
                layer = log['level']
                u, v = log['edge']
                new_edges_persistent.add((tuple(sorted((u, v))), layer))
            
            elif event == 'prune_connection':
                current_node = -1
                layer = log['level']
                u, v = log['edge']
                removed_edges_all.add((tuple(sorted((u, v))), layer))
            
            elif event == 'layer_transition_construction':
                # Just visualization event
                pass

            elif event == 'zoom_in_complete':
                best_nodes_accumulated.update(current_W_set)
            
            # Calculate Coords (Same as visualize_search)
            sel_e_x, sel_e_y, sel_e_z = self._get_edge_coords(node_positions, current_selected_edge)
            sel_ve_x, sel_ve_y, sel_ve_z = self._get_vedge_coords(node_positions, current_selected_vedge)
            rej_e_x, rej_e_y, rej_e_z = self._get_edge_coords(node_positions, current_rejected_edge)
            con_e_x, con_e_y, con_e_z = self._get_edge_coords(node_positions, current_considered_edge)
            sel_n_x, sel_n_y, sel_n_z, sel_n_text = self._get_node_coords(node_positions, current_W_set, node_distances)
            rej_n_x, rej_n_y, rej_n_z, rej_n_text = self._get_node_coords(node_positions, current_rejected_node, node_distances)
            con_n_x, con_n_y, con_n_z, con_n_text = self._get_node_coords(node_positions, current_considered_node, node_distances)
            cur_x, cur_y, cur_z, cur_text = self._get_node_coords(node_positions, [(current_node, current_layer)], node_distances)

            active_targets = current_W_set.copy(); active_targets.add((current_node, current_layer))
            path_x, path_y, path_z = self._reconstruct_path_coords(node_positions, active_targets, node_parents)
            # Blue path removed
            
            # Build new node representation
            nn_layers = [(new_node_id, l) for l in new_node_layers]
            
            # Argument for neighbor highlights
            # Argument for neighbor highlights
            display_neighbors = selected_neighbors_current | selected_neighbors_persistent
            
            # 4. Active Highlight Set Construction (Z-Order Management)
            # define set of active nodes to exclude from background/explored tracing
            current_highlights = current_W_set | current_rejected_node | current_considered_node
            if current_node != -1: current_highlights.add((current_node, current_layer))
            
            # --- 5. Background Node Filtering ---
            active_set = set(visited_history)
            active_set.update(best_nodes_accumulated)
            active_set.update(current_W_set)
            active_set.update(current_rejected_node)
            active_set.update(current_considered_node)
            active_set.update(display_neighbors)
            for l in new_node_layers: active_set.add((new_node_id, l))
            active_set.add((current_node, current_layer))
            
            if initial_ep_config: active_set.add(initial_ep_config)
            
            bg_nodes = list(all_node_pairs - active_set)
            
            # Create frame
            frames.append(go.Frame(
                data=create_frame(
                    sel_e_x, sel_e_y, sel_e_z, sel_ve_x, sel_ve_y, sel_ve_z,
                    rej_e_x, rej_e_y, rej_e_z, con_e_x, con_e_y, con_e_z,
                    sel_n_x, sel_n_y, sel_n_z, sel_n_text,
                    rej_n_x, rej_n_y, rej_n_z, rej_n_text,
                    con_n_x, con_n_y, con_n_z, con_n_text,
                    cur_x, cur_y, cur_z, cur_text,
                    path_x, path_y, path_z, 
                    visited_history,
                    best_nodes_accumulated.copy(),
                    bg_nodes,
                    initial_ep_config,
                    
                    nn_layers,
                    display_neighbors,
                    new_edges_persistent,
                    removed_edges_all,
                    False,
                    current_highlights
                ),
                name=f'step_{i+1}'
            ))
        
        # Final frame
        final_nn_layers = [(new_node_id, l) for l in range(new_node_level + 1)]
        final_neighbors = selected_neighbors_persistent
        
        final_active_set = set(visited_history)
        final_active_set.update(best_nodes_accumulated)
        final_active_set.update(final_nn_layers)
        final_active_set.update(final_neighbors)
        if initial_ep_config: final_active_set.add(initial_ep_config)
        final_bg_nodes = list(all_node_pairs - final_active_set)
        
        # EP Update Check
        final_ep_config = initial_ep_config
        max_new_layer = max((l for _, l in final_nn_layers), default=-1)
        if max_new_layer > initial_max_level:
             final_ep_config = (new_node_id, max_new_layer)
             # Old EP falls to background/neighbor
             if initial_ep_config: final_active_set.discard(initial_ep_config) # Let it be bg if not visited
             final_bg_nodes = list(all_node_pairs - final_active_set)


        frames.append(go.Frame(
            data=create_frame(
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                [], [], [],
                visited_history,
                best_nodes_accumulated,
                final_bg_nodes,
                final_ep_config,
                final_nn_layers,
                final_neighbors,
                new_edges_persistent,
                removed_edges_all,
                is_final_frame=True,
                active_highlight_set=set()
            ),
            name='final'
        ))
        
        # Create slider
        sliders = [{
            "active": 0, "yanchor": "top", "xanchor": "left",
            "currentvalue": {"font": {"size": 20}, "prefix": "Step: ", "visible": True, "xanchor": "right"},
            "transition": {"duration": 0}, "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0,
            "steps": []
        }]
        
        for i, frame in enumerate(frames):
            sliders[0]["steps"].append({
                "args": [[frame.name], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
                "label": str(i) if frame.name != 'final' else 'Final',
                "method": "animate"
            })
        
        # Create figure
        fig = go.Figure(
            data=create_frame(
                [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                [], [], [],
                set(), set(),
                list(all_node_pairs - ({initial_ep_config} if initial_ep_config else set())),
                initial_ep_config,
                [], set(), set(), set(), False, set()
            ),
            layout=go.Layout(
                title=f'HNSW Insert Visualization (Node {new_node_id}, Level {new_node_level})',
                scene=dict(
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    zaxis=dict(title='Level', showbackground=False, nticks=self.max_level + 2),
                    aspectmode='manual', aspectratio=dict(x=1, y=1, z=1)
                ),
                uirevision='constant', hovermode='closest', sliders=sliders,
                updatemenus=[{'type': 'buttons', 'showactive': False, 'buttons': [
                    {'label': 'Play', 'method': 'animate', 'args': [None, {
                        'frame': {'duration': 300, 'redraw': True},
                        'fromcurrent': True, 'transition': {'duration': 250}
                    }]},
                    {'label': 'Pause', 'method': 'animate', 'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate', 'transition': {'duration': 0}
                    }]}
                ]}]
            ),
            frames=frames
        )
        
        fig.show()
        return new_node_id

if __name__ == '__main__':
    train_data = np.random.rand(50, 100)
    query_data = np.random.rand(3, 100)

    max_elements = 50
    dim = 100
    M = 3
    ef_construction = 6

    index = HNSW('l2', dim)
    index.init_index(max_elements, M, ef_construction)

    index.insert_items(train_data)

    W = index.knn_search(query_data[0], 10)
    dists = [l2_distance(query_data[0], train_data[w])**2 for w in W]