"use client";

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { useTheme } from 'next-themes';
import {
    Info,
    Settings,
    Maximize2,
    Layers,
    Play,
    Database,
    Network,
    Moon,
    Sun,
    ExternalLink,
    ChevronDown,
    ChevronRight,
    ArrowLeft
} from 'lucide-react';

import { GraphState, LogEvent, NodeData } from '@/types/visualization';
import { getTheme } from '@/lib/visualization-theme';
import { visualizationApi } from '@/lib/visualization-api';
import { HNSWScene } from '@/components/visualization/HNSWScene';
import { Controls } from '@/components/visualization/Controls';
import { ToastContainer, ToastMessage } from '@/components/visualization/Toast';

export default function VisualizationPage() {
    const { theme: nextTheme, setTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    // UI State
    const [showControls, setShowControls] = useState(true);
    const [loading, setLoading] = useState(false);
    const [speed, setSpeed] = useState(400);
    const [toasts, setToasts] = useState<ToastMessage[]>([]);
    const [activeOperation, setActiveOperation] = useState<string | null>(null);

    // Graph Data
    const [graphState, setGraphState] = useState<GraphState>({ nodes: [], edges: [], global_entry_point: null });

    // Animation/Ephemeral State
    const [candidateSet, setCandidateSet] = useState<Set<string>>(new Set());
    const [visitedHistory, setVisitedHistory] = useState<Set<string>>(new Set());
    const [bestNodesAccumulated, setBestNodesAccumulated] = useState<Set<string>>(new Set());
    const [currentNode, setCurrentNode] = useState<{ id: number, layer: number } | null>(null);
    const [hoveredNode, setHoveredNode] = useState<number | null>(null);

    // Edges (Ephemeral/Frame-based)
    const [frameEdges, setFrameEdges] = useState({
        selected: [] as { u: number, v: number, layer: number }[],
        rejected: [] as { u: number, v: number, layer: number }[],
        considered: [] as { u: number, v: number, layer: number }[],
        selectedVertical: [] as { nodeId: number, fromLayer: number, toLayer: number }[],
        rejectedNodes: new Set<string>()
    });

    // Persistent Animation State
    const [persistentEdges, setPersistentEdges] = useState<{ u: number, v: number, layer: number }[]>([]);
    const [searchPath, setSearchPath] = useState<{ u: number, v: number, layer: number }[]>([]);
    const [searchPathVertical, setSearchPathVertical] = useState<{ nodeId: number, fromLayer: number, toLayer: number }[]>([]);

    // Insertion Specific
    const [insertedNodeId, setInsertedNodeId] = useState<number | null>(null);
    const [addedEdges, setAddedEdges] = useState<{ u: number, v: number, layer: number }[]>([]);
    const [prunedEdges, setPrunedEdges] = useState<{ u: number, v: number, layer: number }[]>([]);
    const [prePrunedEdges, setPrePrunedEdges] = useState<{ u: number, v: number, layer: number }[]>([]);
    const [insertSearchPath, setInsertSearchPath] = useState<{ u: number, v: number, layer: number }[]>([]);

    const speedRef = useRef(speed);
    useEffect(() => { speedRef.current = speed; }, [speed]);

    useEffect(() => {
        setMounted(true);
        refreshState();
    }, []);

    const addToast = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
        const id = Math.random().toString(36).substring(2, 9);
        setToasts(prev => [...prev, { id, message, type }]);
    };

    const removeToast = (id: string) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    };

    const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

    const refreshState = async () => {
        try {
            const data = await visualizationApi.getState();
            setGraphState(data);
        } catch (e) {
            console.error("Failed to fetch state", e);
        }
    };

    const clearEphemeralState = () => {
        setCandidateSet(new Set());
        setVisitedHistory(new Set());
        setBestNodesAccumulated(new Set());
        setCurrentNode(null);
        setFrameEdges({
            selected: [],
            rejected: [],
            considered: [],
            selectedVertical: [],
            rejectedNodes: new Set()
        });
        setPersistentEdges([]);
        setSearchPath([]);
        setSearchPathVertical([]);
        setInsertedNodeId(null);
        setAddedEdges([]);
        setPrunedEdges([]);
        setPrePrunedEdges([]);
        setInsertSearchPath([]);
        setActiveOperation(null);
    };

    const playLogs = async (logs: LogEvent[]) => {
        setLoading(true);
        // Clear ephemeral states but keep the persistent graph state
        setCandidateSet(new Set());
        setVisitedHistory(new Set());
        setBestNodesAccumulated(new Set());
        setCurrentNode(null);
        setSearchPath([]);
        setSearchPathVertical([]);
        setInsertedNodeId(null);
        setAddedEdges([]);
        setPrunedEdges([]);
        setPrePrunedEdges([]);
        setInsertSearchPath([]);

        for (const log of logs) {
            const waitTime = speedRef.current;

            switch (log.event) {
                case 'init_knn_search':
                case 'init_insert':
                    if (log.event === 'init_insert') {
                        setInsertedNodeId(log.node_id);
                        setCurrentNode({ id: log.entry_point, layer: log.current_max_level });
                    } else {
                        setCurrentNode({ id: (log as any).entry_point, layer: (log as any).max_level });
                    }
                    break;

                case 'init_search_layer':
                case 'construction_layer':
                    // Do NOT clear paths here to allow full route persistence
                    // setInsertSearchPath([]); 
                    await sleep(waitTime);
                    break;

                case 'visit_node':
                case 'node_insert':
                case 'current_node':
                    {
                        const id = log.event === 'visit_node' ? log.current_node : log.id;
                        const layer = log.event === 'node_insert' ? log.level : (currentNode?.layer || 0);

                        // Accumulate search path from previous node to current node
                        if (currentNode && (log.event === 'visit_node' || log.event === 'current_node')) {
                            if (currentNode.id !== id && currentNode.layer === layer) {
                                setSearchPath(prev => [...prev, { u: currentNode.id, v: id, layer }]);
                            }
                        }

                        setCurrentNode({ id, layer });
                        setVisitedHistory(prev => new Set([...prev, `${id}-${layer}`]));
                        await sleep(waitTime);
                    }
                    break;

                case 'consider_neighbor':
                case 'select_neighbor':
                    {
                        const nid = log.event === 'consider_neighbor' ? log.neighbor : log.neighbor; // neighbor is in both
                        const nlayer = currentNode?.layer || 0;
                        setFrameEdges(prev => ({
                            ...prev,
                            considered: [{ u: currentNode?.id || 0, v: nid, layer: nlayer }]
                        }));
                        await sleep(waitTime / 2);
                    }
                    break;

                case 'accept_neighbor':
                    setCandidateSet(prev => new Set([...prev, `${log.neighbor}-${currentNode?.layer || 0}`]));
                    setFrameEdges(prev => ({ ...prev, selected: prev.considered, considered: [] }));
                    await sleep(waitTime / 2);
                    setFrameEdges(prev => ({ ...prev, selected: [] }));
                    break;

                case 'reject_neighbor':
                    setFrameEdges(prev => ({ ...prev, rejected: prev.considered, considered: [] }));
                    await sleep(waitTime / 2);
                    setFrameEdges(prev => ({ ...prev, rejected: [] }));
                    break;

                case 'layer_transition':
                case 'layer_transition_construction':
                    // Add to search path before transitioning
                    if (currentNode) {
                        const nodeId = log.event === 'layer_transition_construction' ? log.node_id : currentNode.id;
                        setSearchPathVertical(prev => [...prev, {
                            nodeId: nodeId,
                            fromLayer: log.from_layer,
                            toLayer: log.to_layer
                        }]);
                        setCurrentNode({ id: nodeId, layer: log.to_layer });
                    }
                    await sleep(waitTime);
                    break;

                case 'add_connection':
                    setAddedEdges(prev => [...prev, { u: log.edge[0], v: log.edge[1], layer: log.level }]);
                    await sleep(waitTime);
                    break;

                case 'prune_connection':
                    // First show as to-be-pruned (pre-pruned)
                    setPrePrunedEdges(prev => [...prev, { u: log.edge[0], v: log.edge[1], layer: log.level }]);
                    await sleep(waitTime / 2);
                    // Then show red highlight
                    setPrunedEdges(prev => [...prev, { u: log.edge[0], v: log.edge[1], layer: log.level }]);
                    await sleep(waitTime);
                    break;

                case 'candidates_found':
                    // Update search path for this layer
                    if (currentNode) {
                        const newEdges = log.candidates.map(c => ({ u: log.node_id, v: c, layer: log.level }));
                        // Accumulate instead of replacing
                        setInsertSearchPath(prev => [...prev, ...newEdges]);
                        setSearchPath(prev => [...prev, ...newEdges]); // Also add to main search path for consistent visualization
                    }
                    await sleep(waitTime);
                    break;
            }
        }

        setLoading(false);
        setActiveOperation(null);
        await refreshState();
        addToast("Operation complete", "success");
    };

    const handleInit = async (max: number, m: number, ef: number, dim: number, count: number) => {
        setLoading(true);
        clearEphemeralState();
        try {
            await visualizationApi.initRandom(max, m, ef, dim, count);
            await refreshState();
            addToast(`Graph initialized with ${count} nodes`, "success");
        } catch (e) {
            addToast("Initialization failed", "error");
        } finally {
            setLoading(false);
        }
    };

    const handleInsert = async (customVector?: number[]) => {
        if (loading) return;
        setActiveOperation("Inserting...");
        // Clear previous paths
        setSearchPath([]);
        setSearchPathVertical([]);
        setInsertSearchPath([]);
        try {
            const hasZ = graphState?.nodes?.[0]?.z !== undefined;
            const vector = customVector || Array.from({ length: hasZ ? 3 : 2 }, () => Math.random());
            const res = await visualizationApi.insert(vector);
            await playLogs(res.logs);
        } catch (e) {
            addToast("Insertion failed", "error");
            setLoading(false);
            setActiveOperation(null);
        }
    };

    const handleSearch = async (customVector: number[] | undefined, k: number, ef: number) => {
        if (loading) return;
        setActiveOperation("Searching...");
        // Clear previous paths
        setSearchPath([]);
        setSearchPathVertical([]);
        setInsertSearchPath([]);
        try {
            const vector = customVector || Array.from({ length: graphState.nodes[0]?.z ? 3 : 2 }, () => Math.random());
            const res = await visualizationApi.search(vector, k, ef);
            await playLogs(res.logs);
            addToast(`K-NN Search complete. Results: ${res.results.join(', ')}`, "info");
        } catch (e) {
            addToast("Search failed", "error");
            setLoading(false);
            setActiveOperation(null);
        }
    };

    if (!mounted) return null;

    const theme = getTheme(nextTheme === 'dark' ? 'dark' : 'light');

    return (
        <div className="relative w-full h-screen overflow-hidden font-sans selection:bg-indigo-500/30">
            {/* Main Canvas Area */}
            <div className="absolute inset-0 z-0">
                <HNSWScene
                    theme={theme}
                    graph={graphState}
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
                    onNodeHover={setHoveredNode}
                />
            </div>

            {/* Overlays */}
            <div className="absolute top-6 left-6 z-10 flex flex-col gap-4 pointer-events-none">
                <div className="pointer-events-auto bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl border border-slate-200 dark:border-white/10 rounded-2xl p-6 shadow-2xl max-w-sm">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                            <Link
                                href="/"
                                className="p-2 -ml-2 rounded-xl hover:bg-slate-100 dark:hover:bg-white/5 transition-colors text-slate-500 dark:text-slate-400 group"
                                title="Back to Home"
                            >
                                <ArrowLeft size={20} className="group-hover:-translate-x-0.5 transition-transform" />
                            </Link>
                            <div className="p-2 bg-indigo-500 rounded-lg shadow-lg shadow-indigo-500/20">
                                <Network size={20} className="text-white" />
                            </div>
                            <div>
                                <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-900 to-slate-600 dark:from-white dark:to-slate-400">
                                    HNSW Explorer
                                </h1>
                                <p className="text-[10px] uppercase font-bold text-indigo-500 tracking-[0.2em]">3D Visualization</p>
                            </div>
                        </div>
                        <button
                            onClick={() => setTheme(nextTheme === 'dark' ? 'light' : 'dark')}
                            className="p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-white/5 transition-colors text-slate-500 dark:text-slate-400"
                        >
                            {nextTheme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
                        </button>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-6">
                        <div className="p-3 bg-slate-50 dark:bg-white/5 rounded-xl border border-slate-100 dark:border-white/5">
                            <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Nodes</span>
                            <span className="text-lg font-mono font-bold text-slate-700 dark:text-white">
                                {graphState?.nodes?.length || 0}
                            </span>
                        </div>
                        <div className="p-3 bg-slate-50 dark:bg-white/5 rounded-xl border border-slate-100 dark:border-white/5">
                            <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Layers</span>
                            <span className="text-lg font-mono font-bold text-slate-700 dark:text-white">
                                {graphState?.nodes?.length ? Math.max(...graphState.nodes.map(n => n.layer)) + 1 : 0}
                            </span>
                        </div>
                    </div>

                    {activeOperation && (
                        <div className="flex items-center gap-3 p-3 bg-indigo-50 dark:bg-indigo-500/10 rounded-xl border border-indigo-100 dark:border-indigo-500/20 mb-4 animate-pulse">
                            <div className="w-2 h-2 rounded-full bg-indigo-500" />
                            <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400 tracking-wide uppercase">
                                {activeOperation}
                            </span>
                        </div>
                    )}

                    <div className="space-y-2">
                        <button
                            onClick={() => setShowControls(!showControls)}
                            className="w-full flex items-center justify-between px-4 py-3 bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 rounded-xl transition-all group"
                        >
                            <div className="flex items-center gap-2">
                                <Settings size={16} className="text-slate-400 group-hover:text-indigo-500 transition-colors" />
                                <span className="text-sm font-semibold text-slate-600 dark:text-slate-300">Controls</span>
                            </div>
                            {showControls ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                        </button>

                        {showControls && (
                            <div className="pt-2 animate-in fade-in slide-in-from-top-2 duration-300">
                                <Controls
                                    onInit={handleInit}
                                    onInsert={handleInsert}
                                    onSearch={handleSearch}
                                    onClear={clearEphemeralState}
                                    speed={speed}
                                    setSpeed={setSpeed}
                                    loading={loading}
                                />
                            </div>
                        )}
                    </div>
                </div>

            </div>

            <ToastContainer toasts={toasts} removeToast={removeToast} />
        </div>
    );
}
