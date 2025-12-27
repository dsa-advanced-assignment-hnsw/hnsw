import { useEffect, useState, useRef } from 'react';
import './App.css';
import { HNSWScene } from './components/HNSWScene';
import { api } from './api';
import type { GraphState, LogEvent, NodeData } from './types';
import { getTheme } from './theme';
import { Controls } from './components/Controls';
import { ToastContainer } from './components/Toast';
import type { ToastMessage, ToastType } from './components/Toast';
import { Sun, Moon, Settings2 } from 'lucide-react';

function App() {
  const [graphState, setGraphState] = useState<GraphState>({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(false);

  // Theme State: 'light' or 'dark'
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');

  // Controls Panel Visibility
  const [showControls, setShowControls] = useState(true);

  // Advanced Controls State
  const [speed, setSpeed] = useState(500);
  const speedRef = useRef(speed); // For access inside async loops
  const [hoveredNode, setHoveredNode] = useState<number | null>(null);

  // Toast State
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const addToast = (type: ToastType, message: string) => {
    const id = Math.random().toString(36).substr(2, 9);
    setToasts(prev => [...prev, { id, type, message }]);
  };
  const removeToast = (id: string) => setToasts(prev => prev.filter(t => t.id !== id));

  // Sync ref
  useEffect(() => { speedRef.current = speed; }, [speed]);

  const currentTheme = getTheme(theme);

  useEffect(() => {
    // Initial Fetch
    refreshState();
  }, []);

  const refreshState = async () => {
    try {
      const res = await api.getState();
      setGraphState(res.data);
    } catch (e) {
      console.error("Failed to fetch state", e);
    }
  };

  const handleInit = async (max: number, m: number, ef: number, dim: number, initCount: number) => {
    setLoading(true);
    try {
      const res = await api.initRandom(max, m, ef, dim, initCount);
      // Verify log to ensure backend worked
      console.log("Init Res", res.data);

      // Clear all visualization state on reset
      setCandidateSet(new Set());
      setVisitedHistory(new Set());
      setBestNodesAccumulated(new Set());
      setCurrentNode(null);
      setFrameEdges({ selected: [], rejected: [], considered: [], selectedVertical: [], rejectedNodes: new Set() });
      setPersistentEdges([]);
      setSearchPath([]);
      setSearchPathVertical([]);
      setInsertedNodeId(null);
      setInsertSearchPath([]);
      setAddedEdges([]);
      setPrunedEdges([]);
      setPrePrunedEdges([]);

      // Fetch new state AFTER clearing old state
      await refreshState();
      addToast('success', 'Graph Initialized Successfully');
    } catch (e) {
      console.error(e);
      addToast('error', "Initialization Failed. Check console");
    } finally {
      setLoading(false);
    }
  };

  // State for visualization - matching Python's state variables
  const [candidateSet, setCandidateSet] = useState<Set<string>>(new Set()); // current_W_set (green nodes)
  const [visitedHistory, setVisitedHistory] = useState<Set<string>>(new Set()); // visited_history
  const [bestNodesAccumulated, setBestNodesAccumulated] = useState<Set<string>>(new Set()); // best from upper layers

  // Frame-specific visuals (reset each frame)
  const [currentNode, setCurrentNode] = useState<{ id: number, layer: number } | null>(null);

  // Edges for current frame only
  const [frameEdges, setFrameEdges] = useState<{
    selected: { u: number, v: number, layer: number }[];
    rejected: { u: number, v: number, layer: number }[];
    considered: { u: number, v: number, layer: number }[];
    selectedVertical: { nodeId: number, fromLayer: number, toLayer: number }[]; // Green dashed vertical edges
    rejectedNodes: Set<string>; // Red rejected nodes
  }>({ selected: [], rejected: [], considered: [], selectedVertical: [], rejectedNodes: new Set() });

  // Persistent structural edges (from insertion)
  const [persistentEdges, setPersistentEdges] = useState<{ u: number, v: number, layer: number }[]>([]);

  // Search path state (for visualization)
  const [searchPath, setSearchPath] = useState<{ u: number, v: number, layer: number }[]>([]);
  const [searchPathVertical, setSearchPathVertical] = useState<{ nodeId: number, fromLayer: number, toLayer: number }[]>([]);

  // Insert visualization state
  const [insertedNodeId, setInsertedNodeId] = useState<number | null>(null);
  const [addedEdges, setAddedEdges] = useState<{ u: number, v: number, layer: number }[]>([]); // Green persistent
  const [prunedEdges, setPrunedEdges] = useState<{ u: number, v: number, layer: number }[]>([]); // Red persistent
  const [insertSearchPath, setInsertSearchPath] = useState<{ u: number, v: number, layer: number }[]>([]); // Orange per-layer
  // Edges that will be pruned (need to show them initially as they are missing from final graph)
  const [prePrunedEdges, setPrePrunedEdges] = useState<{ u: number, v: number, layer: number }[]>([]);

  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

  // Helper to format key
  const nKey = (id: number, layer: number) => `${id}-${layer}`;

  const playLogs = async (logs: LogEvent[]) => {
    // Reset all state at start of new operation
    setCandidateSet(new Set());
    setVisitedHistory(new Set());
    setBestNodesAccumulated(new Set());
    setBestNodesAccumulated(new Set());
    setCurrentNode(null);
    setFrameEdges({ selected: [], rejected: [], considered: [], selectedVertical: [], rejectedNodes: new Set() });
    setPersistentEdges([]);
    setSearchPath([]);
    setSearchPathVertical([]);
    // Do not clear persistent entry point

    // Clear INSERT-specific states
    setInsertedNodeId(null);
    setAddedEdges([]);
    setPrunedEdges([]);
    setInsertSearchPath([]);

    // Calculate pruned edges to show them initially (since they are absent from final graph)
    // BUT only include edges that existed BEFORE insert (i.e., do not include edges involving the new node)
    let tempInsertedId = -1;
    const insertLog = logs.find(l => l.event === 'init_insert');
    if (insertLog && insertLog.event === 'init_insert') {
      tempInsertedId = insertLog.node_id;
    }

    const initialPruned: { u: number, v: number, layer: number }[] = [];
    logs.forEach(l => {
      if (l.event === 'prune_connection') {
        const [u, v] = l.edge as [number, number];
        const layer = l.level as number;
        // Only add if it doesn't involve the new node
        if (u !== tempInsertedId && v !== tempInsertedId) {
          initialPruned.push({ u, v, layer });
        }
      }
    });
    setPrePrunedEdges(initialPruned);

    // Accumulated state (matching Python)
    let current_node = -1;
    let current_layer = -1;
    let current_W_set = new Set<string>();
    let visited_history = new Set<string>();
    let best_nodes_accumulated = new Set<string>();
    let node_parents = new Map<string, string>(); // Track parent relationships for path reconstruction

    for (const log of logs) {
      if (loading) break; // Basic cancellation check (not perfect)

      // Reset frame-specific visuals
      let frame_selected_edges: { u: number, v: number, layer: number }[] = [];
      let frame_rejected_edges: { u: number, v: number, layer: number }[] = [];
      let frame_considered_edges: { u: number, v: number, layer: number }[] = [];
      let frame_selected_vedges: { nodeId: number, fromLayer: number, toLayer: number }[] = [];
      let frame_rejected_nodes = new Set<string>(); // Ephemeral rejected nodes for red highlight

      // Process event (matching Python logic exactly)
      const event = log.event;

      if (event === 'init_knn_search') {
        current_node = log.entry_point;
        current_layer = log.max_level;
        current_W_set = new Set([nKey(current_node, current_layer)]);
        visited_history.add(nKey(current_node, current_layer));
        // Visualize Entry Point
        if (current_node !== -1) {
          setCurrentNode({ id: current_node, layer: current_layer });
        }
      }

      else if (event === 'layer_transition') {
        // Save best nodes from previous layer
        best_nodes_accumulated = new Set([...best_nodes_accumulated, ...current_W_set]);

        const ep = log.ep;
        const fromLayer = current_layer;
        const toLayer = log.to_layer;

        // Track parent relationship across layers (matching Python line 1196)
        const childKey = nKey(ep, toLayer);
        const parentKey = nKey(ep, fromLayer);
        node_parents.set(childKey, parentKey);

        // Add vertical edge for this transition (ephemeral, shown this frame only)
        frame_selected_vedges.push({ nodeId: ep, fromLayer, toLayer });

        current_node = ep;
        current_layer = toLayer;
        current_W_set = new Set([nKey(current_node, current_layer)]);
        visited_history.add(nKey(current_node, current_layer));
      }
      else if (event === 'init_search_layer') {
        current_layer = log.layer;
        current_node = log.ep;
        current_W_set = new Set([nKey(current_node, current_layer)]);
        visited_history.add(nKey(current_node, current_layer));
      }
      else if (event === 'visit_node') {
        current_node = log.current_node;
        visited_history.add(nKey(current_node, current_layer));
      }
      else if (event === 'consider_neighbor') {
        const neighbor = log.neighbor;
        const u = Math.min(current_node, neighbor);
        const v = Math.max(current_node, neighbor);
        frame_considered_edges.push({ u, v, layer: current_layer });
        visited_history.add(nKey(neighbor, current_layer));
      }
      else if (event === 'accept_neighbor') {
        const neighbor = log.neighbor;
        const u = Math.min(current_node, neighbor);
        const v = Math.max(current_node, neighbor);
        frame_selected_edges.push({ u, v, layer: current_layer });
        current_W_set.add(nKey(neighbor, current_layer));
        // Track parent for path reconstruction
        const neighborKey = nKey(neighbor, current_layer);
        const currentKey = nKey(current_node, current_layer);
        if (!node_parents.has(neighborKey)) {
          node_parents.set(neighborKey, currentKey);
        }
      }
      else if (event === 'reject_neighbor') {
        const neighbor = log.neighbor;
        const u = Math.min(current_node, neighbor);
        const v = Math.max(current_node, neighbor);
        frame_rejected_edges.push({ u, v, layer: current_layer });
      }
      else if (event === 'reject_node') {
        const reject_node_id = log.reject_node ?? log.id;
        if (reject_node_id !== undefined) {
          const rejectKey = nKey(reject_node_id, current_layer);
          current_W_set.delete(rejectKey);
          // Track for red highlight this frame (Python line 1225)
          frame_rejected_nodes.add(rejectKey);
        }
      }
      else if (event === 'reject_nodes' && 'reject_nodes' in log) {
        log.reject_nodes.forEach((id: number) => {
          const rejectKey = nKey(id, current_layer);
          current_W_set.delete(rejectKey);
          // Track for red highlight this frame (Python line 1229)
          frame_rejected_nodes.add(rejectKey);
        });
      }
      // Insertion events
      else if (event === 'init_insert') {
        const newNodeId = log.node_id;
        const assignedLevel = log.assigned_level;
        setInsertedNodeId(newNodeId); // Mark inserted node for special rendering

        // Add new node to graph immediately so it's visible from start
        setGraphState(prevGraph => {
          // Add node at all layers up to assigned_level
          const newNodes: NodeData[] = [];
          for (let layer = 0; layer <= assignedLevel; layer++) {
            newNodes.push({
              id: newNodeId,
              layer,
              x: Math.random() * 10 - 5, // Random position
              y: Math.random() * 10 - 5,
              z: 0
            });
          }
          return {
            ...prevGraph,
            nodes: [...prevGraph.nodes, ...newNodes]
          };
        });

        // Refresh layout before starting search animation
        await refreshState();

        if (log.entry_point !== -1) {
          const epId = log.entry_point;
          const epLayer = log.current_max_level;

          current_node = epId;
          current_layer = epLayer;
          current_W_set = new Set([nKey(current_node, current_layer)]);
          visited_history.add(nKey(current_node, current_layer));

          setCurrentNode({ id: epId, layer: epLayer });
        }
      }
      else if (event === 'node_insert') {
        // Node is being inserted at this layer - clear search path as search phase is done for this layer
        setInsertSearchPath([]);
      }
      else if (event === 'zoom_in_layer' || event === 'construction_layer') {
        // Clear insert search path when changing layers
        setInsertSearchPath([]);

        const prevLayer = current_layer;
        current_layer = log.level;
        current_node = log.ep;

        // Track parent for vertical edge (ephemeral, 1 frame) but DON'T link for path
        // Python comment line 1734: "node_parents link REMOVED to keep search path per-layer (not persistent vertically)"
        if (prevLayer > current_layer) {
          // Vertical edge shown but parent not tracked for path reconstruction
          frame_selected_vedges.push({ nodeId: current_node, fromLayer: prevLayer, toLayer: current_layer });
        }

        current_W_set = new Set([nKey(current_node, current_layer)]);
        visited_history.add(nKey(current_node, current_layer));
      }
      else if (event === 'neighbors_selected') {
        // Filter W set to only keep selected neighbors (Python lines 1807-1821)
        current_node = -1; // Hide focus
        const layer = log.level;
        const neighbors = log.neighbors as number[];
        const neighbors_set = new Set(neighbors.map(nid => nKey(nid, layer)));

        // Find nodes in W that are not in selected neighbors
        const to_remove = new Set<string>();
        current_W_set.forEach(key => {
          const [, keyLayer] = key.split('-').map(Number);
          if (keyLayer === layer && !neighbors_set.has(key)) {
            to_remove.add(key);
          }
        });

        // Remove from W and mark as rejected (red highlight this frame)
        to_remove.forEach(key => {
          current_W_set.delete(key);
          frame_rejected_nodes.add(key);
        });
        node_parents.clear(); // Prevent path reconstruction during connection phase

        // Clear search path - neighbors selected means search phase is done
        setInsertSearchPath([]);
      }
      else if (event === 'add_connection') {
        const [u, v] = log.edge;
        const layer = log.level;
        setAddedEdges(prev => [...prev, { u, v, layer }]); // Green persistent
      }
      else if (event === 'prune_connection') {
        const [u, v] = log.edge;
        const layer = log.level;
        setPrunedEdges(prev => [...prev, { u, v, layer }]); // Red persistent
        // Also remove from addedEdges if it was just added
        setAddedEdges(prev => prev.filter(e => !(e.u === u && e.v === v && e.layer === layer)));
      }

      // Update React state for rendering
      setCandidateSet(new Set(current_W_set));
      setVisitedHistory(new Set(visited_history));
      setBestNodesAccumulated(new Set(best_nodes_accumulated));
      setBestNodesAccumulated(new Set(best_nodes_accumulated));
      setCurrentNode(current_node >= 0 ? { id: current_node, layer: current_layer } : null);
      // Path reconstruction on each frame (matching Python's _reconstruct_path_coords)
      const reconstructPath = (targetNodes: Set<string>, parents: Map<string, string>): {
        horizontal: { u: number, v: number, layer: number }[],
        vertical: { nodeId: number, fromLayer: number, toLayer: number }[]
      } => {
        const horizontalEdges: { u: number, v: number, layer: number }[] = [];
        const verticalEdges: { nodeId: number, fromLayer: number, toLayer: number }[] = [];
        const visitedEdges = new Set<string>();

        for (const nodeKey of targetNodes) {
          let curr = nodeKey;
          while (parents.has(curr)) {
            const parent = parents.get(curr)!;
            const edgeSig = [curr, parent].sort().join('|');

            if (!visitedEdges.has(edgeSig)) {
              visitedEdges.add(edgeSig);
              const [currId, currLayer] = curr.split('-').map(Number);
              const [parentId, parentLayer] = parent.split('-').map(Number);

              if (currLayer === parentLayer) {
                // Horizontal edge (same layer)
                horizontalEdges.push({ u: currId, v: parentId, layer: currLayer });
              } else {
                // Vertical edge (between layers) - same node ID, different layers
                if (currId === parentId) {
                  verticalEdges.push({
                    nodeId: currId,
                    fromLayer: Math.max(currLayer, parentLayer),
                    toLayer: Math.min(currLayer, parentLayer)
                  });
                }
              }
            }
            curr = parent;
          }
        }
        return { horizontal: horizontalEdges, vertical: verticalEdges };
      };

      // Update paths every frame for persistence
      const pathResult = reconstructPath(current_W_set, node_parents);
      setSearchPath(pathResult.horizontal);
      setSearchPathVertical(pathResult.vertical); // Persistent vertical path

      // For INSERT: also update insert search path (per-layer horizontal only)
      // But NOT during connection phase (after neighbors_selected)
      if (insertedNodeId !== null &&
        event !== 'neighbors_selected' &&
        event !== 'add_connection' &&
        event !== 'prune_connection') {
        setInsertSearchPath(pathResult.horizontal); // Will be cleared on layer change
      }

      // Merge vertical path edges with frame vertical edges for display
      // Path vertical edges are persistent, frame vertical edges are ephemeral
      // Update frame edges
      setFrameEdges({
        selected: frame_selected_edges,
        rejected: frame_rejected_edges,
        considered: frame_considered_edges,
        selectedVertical: frame_selected_vedges,
        rejectedNodes: frame_rejected_nodes
      });

      // Frame delay - dynamic
      await delay(speedRef.current);
    }


    // Final pause before showing final result
    await delay(speedRef.current);

    // Final cleanup - clear ephemeral states but keep persistent ones
    setCurrentNode(null);
    setFrameEdges({ selected: [], rejected: [], considered: [], selectedVertical: [], rejectedNodes: new Set() });
    setInsertSearchPath([]); // Clear insert search path at end (Python behavior)
  };


  const handleInsert = async (customVector?: number[]) => {
    setLoading(true);
    setCandidateSet(new Set()); // Changed from setHighlightedNodes

    // Use custom vector if provided, else random
    const vec = customVector || Array.from({ length: 3 }, () => Math.random());

    try {
      const res = await api.insert(vec);
      console.log("Insert Logs:", res.data.logs);

      // Play animation first
      await playLogs(res.data.logs);

      // THEN refresh state to show final graph with smooth transition
      await delay(speedRef.current); // Pause before layout change
      await refreshState();
      addToast('success', `Inserted Node ${res.data.logs.find((l: any) => l.event === 'init_insert')?.node_id}`);
    } catch (e) {
      console.error(e);
      addToast('error', "Insert Failed. Check console.");
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (customVector: number[] | undefined, k: number, ef: number) => {
    setLoading(true);
    setCandidateSet(new Set());

    const vec = customVector || Array.from({ length: 3 }, () => Math.random());

    try {
      // Pass ef to backend (updated signature)
      const res = await api.search(vec, k, ef);
      console.log("Search Logs:", res.data.logs);

      await playLogs(res.data.logs);
      addToast('info', 'Search Completed');
    } catch (e) {
      console.error(e);
      addToast('error', "Search Failed. Check console.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setCandidateSet(new Set());
    setVisitedHistory(new Set());
    setBestNodesAccumulated(new Set());
    setBestNodesAccumulated(new Set());
    setCurrentNode(null);
    setFrameEdges({ selected: [], rejected: [], considered: [], selectedVertical: [], rejectedNodes: new Set() });
    setPersistentEdges([]);
    setSearchPath([]);
    setSearchPathVertical([]);
    setInsertedNodeId(null);
    setAddedEdges([]);
    setPrunedEdges([]);
    setInsertSearchPath([]);

    setPrePrunedEdges([]);
  };

  const handleHover = (nodeId: number | null) => {
    setHoveredNode(nodeId);
  };

  return (
    <div className={theme}>
      <div className="relative w-full h-screen bg-slate-50 dark:bg-slate-950 overflow-hidden text-slate-900 dark:text-white selection:bg-neon-cyan selection:text-black transition-colors duration-500">

        {/* Background Scene */}
        <div className="absolute inset-0 z-0">
          <HNSWScene
            theme={currentTheme}
            graph={graphState}
            highlightedNodes={new Set()}
            activePath={[]}
            candidateSet={candidateSet}
            visitedHistory={visitedHistory}
            bestNodesAccumulated={bestNodesAccumulated}
            currentNode={currentNode}
            frameEdges={frameEdges}
            persistentEdges={persistentEdges}
            searchPath={searchPath}
            searchPathVertical={searchPathVertical}
            insertedNodeId={insertedNodeId}
            addedEdges={addedEdges}
            prunedEdges={prunedEdges}
            prePrunedEdges={prePrunedEdges}
            insertSearchPath={insertSearchPath}
            hoveredNode={hoveredNode}
            onNodeClick={(node) => console.log("Clicked:", node)}
            onNodeHover={handleHover}
          />
        </div>

        {/* Header (Floating Glass) */}
        <div className="absolute top-4 left-4 right-4 h-14 bg-white/60 dark:bg-glass-medium backdrop-blur-md rounded-xl border border-slate-200 dark:border-white/10 flex items-center justify-between px-6 z-10 shadow-lg transition-colors">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-violet-600 dark:bg-neon-cyan shadow-sm dark:shadow-neon-cyan"></div>
            <span className="font-mono font-bold text-lg tracking-wider text-slate-800 dark:text-white/90">HNSW<span className="text-violet-600 dark:text-neon-cyan">.LAB</span></span>
          </div>

          <div className="flex items-center gap-4">
            <span className="hidden md:inline text-xs text-slate-500 dark:text-slate-400 font-mono">v2.1.0-LIGHTSPEED</span>

            <div className="h-6 w-px bg-slate-200 dark:bg-white/10 mx-1"></div>

            {/* Controls Toggle */}
            <button
              onClick={() => setShowControls(!showControls)}
              className={`p-2 rounded-lg transition-all ${showControls
                ? 'bg-violet-100 text-violet-600 dark:bg-white/10 dark:text-neon-cyan'
                : 'text-slate-400 hover:text-slate-600 dark:text-slate-400 dark:hover:text-white'
                }`}
              title="Toggle Settings Panel"
            >
              <Settings2 size={18} />
            </button>

            {/* Theme Toggle */}
            <button
              onClick={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-white/5 text-slate-400 hover:text-amber-500 dark:hover:text-yellow-300 transition-all"
              title="Toggle Theme"
            >
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>

        {/* Floating Controls Panel */}
        <div className={`absolute top-24 left-4 z-10 transition-all duration-300 ease-in-out ${showControls ? 'translate-x-0 opacity-100' : '-translate-x-[120%] opacity-0'}`}>
          <div className="w-80 max-h-[calc(100vh-8rem)] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-slate-700 scrollbar-track-transparent rounded-xl shadow-2xl">
            <div className="bg-white/80 dark:bg-glass-dark backdrop-blur-xl border border-slate-200 dark:border-white/10 rounded-xl p-4 shadow-xl">
              <Controls
                onInit={handleInit}
                onInsert={handleInsert}
                onSearch={handleSearch}
                onClear={handleClear}
                speed={speed}
                setSpeed={setSpeed}
                loading={loading}
              />
            </div>
          </div>
        </div>

        {/* Footer / Status */}
        <div className="absolute bottom-4 left-4 text-xs font-mono text-slate-500 dark:text-slate-500 z-0 pointer-events-none transition-colors">
          Nodes: {graphState.nodes.length} | Edges: {graphState.edges.length} {graphState.global_entry_point !== undefined && graphState.global_entry_point !== null ? `| EP: ${graphState.global_entry_point}` : ''}
        </div>

        <ToastContainer toasts={toasts} removeToast={removeToast} />
      </div>
    </div>
  );
}

export default App;
