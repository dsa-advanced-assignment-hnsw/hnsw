"use client";

import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text, Line, Html } from '@react-three/drei';
import * as THREE from 'three';
import { useSpring, animated } from '@react-spring/three';
import { GraphState, NodeData } from '@/types/visualization';
import { Theme } from '@/lib/visualization-theme';

interface SceneProps {
    graph: GraphState;
}

const LAYER_HEIGHT = 10;
const LAYER_SCALE_FACTOR = 15;

// Animated Node Component
const AnimatedNode: React.FC<{
    node: NodeData;
    inCandidate: boolean;
    inVisited: boolean;
    inBest: boolean;
    isActive: boolean;
    isRejected: boolean;
    isInserted: boolean;
    isEntryPoint: boolean;
    isHovered: boolean;
    maxLevel: number;
    theme: Theme;
    onClick: (node: NodeData) => void;
    onHover: (id: number | null) => void;
}> = ({ node, inCandidate, inVisited, inBest, isActive, isRejected, isInserted, isEntryPoint, isHovered, maxLevel, theme, onClick, onHover }) => {
    const targetPosition = useMemo(() => new THREE.Vector3(
        node.x * LAYER_SCALE_FACTOR,
        node.layer * LAYER_HEIGHT,
        node.y * LAYER_SCALE_FACTOR
    ), [node.x, node.y, node.layer]);

    const { position } = useSpring({
        position: targetPosition.toArray() as [number, number, number],
        config: { tension: 120, friction: 14 }
    });

    let color = theme.nodes.getLayerColor(node.layer, maxLevel);
    let scale = 1;

    if (isHovered) {
        color = '#FF4500';
        scale = 2.0;
    } else if (isEntryPoint) {
        color = '#FF1493';
        scale = 1.8;
    } else if (isRejected) {
        color = '#FF0000';
        scale = 1.3;
    } else if (isActive) {
        color = theme.nodes.activeColor;
        scale = 1.6;
    } else if (isInserted) {
        color = '#FFD700';
        scale = 1.4;
    } else if (inCandidate || inBest) {
        color = theme.nodes.candidateColor;
        scale = 1.3;
    } else if (inVisited) {
        scale = 1.1;
    }

    return (
        <animated.group
            position={position as any}
            scale={scale}
            onClick={(e: any) => { e.stopPropagation(); onClick(node); }}
            onPointerOver={(e: any) => { e.stopPropagation(); onHover(node.id); }}
            onPointerOut={(e: any) => { e.stopPropagation(); onHover(null); }}
        >
            <mesh castShadow>
                <sphereGeometry args={[0.3, 20, 20]} />
                <meshStandardMaterial
                    color={color}
                    metalness={0.2}
                    roughness={0.6}
                />
            </mesh>
            {isActive && (
                <mesh>
                    <sphereGeometry args={[0.55, 16, 16]} />
                    <meshBasicMaterial color={theme.nodes.activeColor} transparent opacity={0.2} side={THREE.BackSide} />
                </mesh>
            )}
            <Text
                position={[0, 0.6, 0]}
                fontSize={0.35}
                color={theme.text.primary}
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.01}
                outlineColor={theme.text.outline}
            >
                {node.id.toString()}
            </Text>
            {isHovered && (
                <Html distanceFactor={15} position={[1.5, 1.5, 0]} pointerEvents="none" zIndexRange={[100, 0]}>
                    <div className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl border-4 border-slate-300 dark:border-white/20 rounded-3xl p-8 shadow-2xl min-w-[500px] animate-in fade-in zoom-in duration-200">
                        <div className="flex items-center gap-5 mb-6 border-b-4 border-slate-200 dark:border-white/15 pb-4">
                            <div className="w-6 h-6 rounded-full bg-indigo-500 shadow-[0_0_16px_rgba(99,102,241,0.8)]" />
                            <span className="text-2xl font-bold uppercase tracking-wider text-slate-600">Node Information</span>
                        </div>
                        <div className="space-y-5">
                            <div className="flex justify-between items-center">
                                <span className="text-xl font-bold text-slate-600">Node ID</span>
                                <span className="text-3xl font-mono font-bold text-slate-900 dark:text-white bg-slate-100 dark:bg-white/15 px-6 py-2 rounded-xl shadow-md">#{node.id}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="text-xl font-bold text-slate-600">Layer</span>
                                <span className="text-3xl font-mono font-bold text-indigo-600 dark:text-indigo-400">{node.layer}</span>
                            </div>

                            {node.vector && node.vector.length > 0 && (
                                <div className="pt-5 mt-5 border-t-4 border-slate-200 dark:border-white/15">
                                    <span className="text-lg font-bold text-slate-600 block mb-3 uppercase tracking-wide">Feature Vector</span>
                                    <div className="text-lg font-mono text-slate-800 dark:text-slate-100 bg-slate-50 dark:bg-black/40 p-5 rounded-2xl border-2 border-slate-200 dark:border-white/15 break-all leading-loose shadow-inner">
                                        [{node.vector.map(v => v.toFixed(4)).join(', ')}]
                                    </div>
                                </div>
                            )}

                            <div className="pt-5 mt-5 border-t-4 border-slate-200 dark:border-white/15 space-y-4">
                                <div className="flex justify-between items-center">
                                    <span className="text-lg font-bold text-slate-600">Position X</span>
                                    <span className="text-xl font-mono font-semibold text-slate-800 dark:text-slate-100">{node.x.toFixed(4)}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-lg font-bold text-slate-600">Position Y</span>
                                    <span className="text-xl font-mono font-semibold text-slate-800 dark:text-slate-100">{node.y.toFixed(4)}</span>
                                </div>
                                {node.z !== undefined && (
                                    <div className="flex justify-between items-center">
                                        <span className="text-lg font-bold text-slate-600">Position Z</span>
                                        <span className="text-xl font-mono font-semibold text-slate-800 dark:text-slate-100">{node.z.toFixed(4)}</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </Html>
            )}
        </animated.group>
    );
};

// Layer Plane
const LayerPlane: React.FC<{ layer: number; maxLevel: number; theme: Theme }> = ({ layer, maxLevel, theme }) => {
    const planeColor = theme.planes.getColor(layer, maxLevel);

    return (
        <group position={[0, layer * LAYER_HEIGHT, 0]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow renderOrder={-1}>
                <planeGeometry args={[LAYER_SCALE_FACTOR * 3, LAYER_SCALE_FACTOR * 3]} />
                <meshBasicMaterial
                    color={planeColor}
                    transparent
                    opacity={0.05}
                    side={THREE.DoubleSide}
                    depthWrite={false}
                />
            </mesh>
            <Text
                position={[-LAYER_SCALE_FACTOR * 1.3, 0, -LAYER_SCALE_FACTOR * 1.3]}
                fontSize={0.8}
                color={theme.text.secondary}
                anchorX="center"
                anchorY="middle"
            >
                {`Layer ${layer}`}
            </Text>
        </group>
    );
};

export const HNSWScene: React.FC<SceneProps & {
    theme: Theme;
    candidateSet: Set<string>;
    visitedHistory: Set<string>;
    bestNodesAccumulated: Set<string>;
    currentNode: { id: number, layer: number } | null;
    frameEdges: {
        selected: { u: number, v: number, layer: number }[];
        rejected: { u: number, v: number, layer: number }[];
        considered: { u: number, v: number, layer: number }[];
        selectedVertical: { nodeId: number, fromLayer: number, toLayer: number }[];
        rejectedNodes: Set<string>;
    };
    persistentEdges: { u: number, v: number, layer: number }[];
    searchPath: { u: number, v: number, layer: number }[];
    searchPathVertical: { nodeId: number, fromLayer: number, toLayer: number }[];
    insertedNodeId: number | null;
    addedEdges: { u: number, v: number, layer: number }[];
    prunedEdges: { u: number, v: number, layer: number }[];
    prePrunedEdges: { u: number, v: number, layer: number }[];
    insertSearchPath: { u: number, v: number, layer: number }[];
    hoveredNode: number | null;
    onNodeClick?: (node: NodeData) => void;
    onNodeHover?: (nodeId: number | null) => void;
}> = ({ theme, graph, candidateSet, visitedHistory, bestNodesAccumulated, currentNode, frameEdges, persistentEdges, searchPath, searchPathVertical, insertedNodeId, addedEdges, prunedEdges, prePrunedEdges, insertSearchPath, hoveredNode, onNodeClick, onNodeHover }) => {
    const maxLayer = useMemo(() => {
        if (!graph?.nodes?.length) return 0;
        return Math.max(0, ...graph.nodes.map(n => n.layer));
    }, [graph?.nodes]);

    const derivedEntryPoint = useMemo(() => {
        if (graph?.global_entry_point === undefined || graph?.global_entry_point === null || !graph?.nodes) return null;
        const epNodes = graph.nodes.filter(n => n.id === graph.global_entry_point);
        if (epNodes.length === 0) return null;
        const maxEpLayer = Math.max(...epNodes.map(n => n.layer));
        return { id: graph.global_entry_point, layer: maxEpLayer };
    }, [graph?.nodes, graph?.global_entry_point]);

    const layers = Array.from({ length: maxLayer + 1 }, (_, i) => i);

    const isInSet = (nodeId: number, nodeLayer: number, keySet: Set<string>) => {
        return keySet.has(`${nodeId}-${nodeLayer}`);
    };

    const verticalEdges = useMemo(() => {
        const edges: { nodeId: number, fromLayer: number, toLayer: number }[] = [];
        const nodesByLayer = new Map<number, Set<number>>();

        graph?.nodes?.forEach(node => {
            if (!nodesByLayer.has(node.id)) {
                nodesByLayer.set(node.id, new Set());
            }
            nodesByLayer.get(node.id)!.add(node.layer);
        });

        nodesByLayer.forEach((layers, nodeId) => {
            const sortedLayers = Array.from(layers).sort((a, b) => a - b);
            for (let i = 0; i < sortedLayers.length - 1; i++) {
                edges.push({
                    nodeId,
                    fromLayer: sortedLayers[i],
                    toLayer: sortedLayers[i + 1]
                });
            }
        });

        return edges;
    }, [graph?.nodes]);

    return (
        <div className="w-full h-full" style={{ background: theme.background }}>
            <Canvas shadows camera={{ position: [25, 25, 45], fov: 50 }}>
                <color attach="background" args={[theme.canvas]} />

                <ambientLight intensity={theme.lighting.ambient} />
                <directionalLight position={[10, 20, 10]} intensity={theme.lighting.directional1} castShadow />
                <directionalLight position={[-10, 15, -10]} intensity={theme.lighting.directional2} />
                <hemisphereLight args={[theme.lighting.hemisphereTop, theme.lighting.hemisphereBottom, 0.5]} />

                <PerspectiveCamera makeDefault position={[25, 25, 45]} fov={50} />
                <OrbitControls makeDefault enableDamping dampingFactor={0.05} />

                {layers.map(l => <LayerPlane key={l} layer={l} maxLevel={maxLayer} theme={theme} />)}

                {verticalEdges.map((vedge, idx) => {
                    const bottomNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.fromLayer);
                    const topNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.toLayer);

                    if (!bottomNode || !topNode) return null;

                    const start = new THREE.Vector3(bottomNode.x * LAYER_SCALE_FACTOR, bottomNode.layer * LAYER_HEIGHT, bottomNode.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(topNode.x * LAYER_SCALE_FACTOR, topNode.layer * LAYER_HEIGHT, topNode.y * LAYER_SCALE_FACTOR);

                    return <Line key={`vedge-${idx}`} points={[start, end]} color="#cccccc" lineWidth={1} dashed dashScale={2} transparent opacity={0.4} />;
                })}

                {graph?.nodes?.map((node) => {
                    const inCandidate = isInSet(node.id, node.layer, candidateSet);
                    const inVisited = isInSet(node.id, node.layer, visitedHistory);
                    const inBest = isInSet(node.id, node.layer, bestNodesAccumulated);
                    const isActive = currentNode?.id === node.id && currentNode?.layer === node.layer;
                    const isRejected = frameEdges.rejectedNodes.has(`${node.id}-${node.layer}`);
                    const isRejectedNode = frameEdges.rejectedNodes.has(`${node.id}-${node.layer}`);
                    const isInserted = insertedNodeId === node.id;
                    const isEntryPoint = derivedEntryPoint?.id === node.id && derivedEntryPoint?.layer === node.layer;

                    return (
                        <AnimatedNode
                            key={`${node.id}-${node.layer}`}
                            node={node}
                            inCandidate={inCandidate}
                            inVisited={inVisited}
                            inBest={inBest}
                            isActive={isActive}
                            isRejected={isRejected}
                            isInserted={isInserted}
                            isEntryPoint={isEntryPoint}
                            isHovered={hoveredNode === node.id}
                            maxLevel={maxLayer}
                            theme={theme}
                            onClick={onNodeClick || (() => { })}
                            onHover={onNodeHover || (() => { })}
                        />
                    );
                })}

                {graph?.edges?.map((edge, idx) => {
                    const u = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u || !v) return null;
                    if (insertedNodeId !== null && (edge.u === insertedNodeId || edge.v === insertedNodeId)) return null;
                    const start = new THREE.Vector3(u.x * LAYER_SCALE_FACTOR, u.layer * LAYER_HEIGHT, u.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v.x * LAYER_SCALE_FACTOR, v.layer * LAYER_HEIGHT, v.y * LAYER_SCALE_FACTOR);
                    return <Line key={`edge-${idx}`} points={[start, end]} color={theme.edges.permanent} lineWidth={1.5} transparent opacity={0.6} />
                })}

                {searchPath.map((edge, idx) => {
                    const u = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u || !v) return null;
                    const start = new THREE.Vector3(u.x * LAYER_SCALE_FACTOR, u.layer * LAYER_HEIGHT, u.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v.x * LAYER_SCALE_FACTOR, v.layer * LAYER_HEIGHT, v.y * LAYER_SCALE_FACTOR);
                    return <Line key={`spath-${idx}`} points={[start, end]} color="#FF8C00" lineWidth={4} transparent opacity={1.0} renderOrder={100} />
                })}

                {searchPathVertical.map((vedge, idx) => {
                    const bottomNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.fromLayer);
                    const topNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.toLayer);
                    if (!bottomNode || !topNode) return null;
                    const start = new THREE.Vector3(bottomNode.x * LAYER_SCALE_FACTOR, bottomNode.layer * LAYER_HEIGHT, bottomNode.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(topNode.x * LAYER_SCALE_FACTOR, topNode.layer * LAYER_HEIGHT, topNode.y * LAYER_SCALE_FACTOR);
                    return <Line key={`spathv-${idx}`} points={[start, end]} color="#FF8C00" lineWidth={4} transparent opacity={1.0} renderOrder={100} />;
                })}

                {frameEdges.selectedVertical.map((vedge, idx) => {
                    const bottomNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.fromLayer);
                    const topNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.toLayer);
                    if (!bottomNode || !topNode) return null;
                    const start = new THREE.Vector3(bottomNode.x * LAYER_SCALE_FACTOR, bottomNode.layer * LAYER_HEIGHT, bottomNode.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(topNode.x * LAYER_SCALE_FACTOR, topNode.layer * LAYER_HEIGHT, topNode.y * LAYER_SCALE_FACTOR);
                    return <Line key={`vspath-${idx}`} points={[start, end]} color="#FF8C00" lineWidth={4} dashed dashScale={2} transparent opacity={0.8} renderOrder={100} />;
                })}

                {persistentEdges.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u_node || !v_node || (insertedNodeId !== null && (edge.u === insertedNodeId || edge.v === insertedNodeId))) return null;
                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    return <Line key={`perm-${idx}`} points={[start, end]} color="#4682B4" lineWidth={2} transparent opacity={0.5} renderOrder={10} />;
                })}

                {prePrunedEdges.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u_node || !v_node) return null;
                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    return <Line key={`prepruned-${idx}`} points={[start, end]} color={theme.edges.permanent} lineWidth={1.5} transparent opacity={0.6} renderOrder={10} />;
                })}

                {addedEdges.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u_node || !v_node) return null;
                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    return <Line key={`added-${idx}`} points={[start, end]} color="#32CD32" lineWidth={3} transparent opacity={1.0} renderOrder={80} />;
                })}

                {prunedEdges.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u_node || !v_node) return null;
                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    return <Line key={`pruned-${idx}`} points={[start, end]} color="#FF0000" lineWidth={3} transparent opacity={1.0} renderOrder={90} />;
                })}

                {insertSearchPath.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u_node || !v_node) return null;
                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    return <Line key={`insertpath-${idx}`} points={[start, end]} color="#FF8C00" lineWidth={4} transparent opacity={1.0} renderOrder={100} />;
                })}

                {hoveredNode !== null && (() => {
                    const edgesToRender: React.ReactNode[] = [];
                    graph.edges.filter(e => e.u === hoveredNode || e.v === hoveredNode).forEach((edge, idx) => {
                        const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                        const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                        if (u_node && v_node) {
                            const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                            const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                            edgesToRender.push(<Line key={`hover-edge-${idx}`} points={[start, end]} color="#FF4500" lineWidth={4} renderOrder={200} />);
                        }
                    });
                    const sameNodes = graph.nodes.filter(n => n.id === hoveredNode).sort((a, b) => a.layer - b.layer);
                    for (let i = 0; i < sameNodes.length - 1; i++) {
                        const n1 = sameNodes[i];
                        const n2 = sameNodes[i + 1];
                        const start = new THREE.Vector3(n1.x * LAYER_SCALE_FACTOR, n1.layer * LAYER_HEIGHT, n1.y * LAYER_SCALE_FACTOR);
                        const end = new THREE.Vector3(n2.x * LAYER_SCALE_FACTOR, n2.layer * LAYER_HEIGHT, n2.y * LAYER_SCALE_FACTOR);
                        edgesToRender.push(<Line key={`hover-vedge-${i}`} points={[start, end]} color="#FF4500" lineWidth={3} dashed dashScale={1} gapSize={0.5} renderOrder={200} />);
                    }
                    return edgesToRender;
                })()}

                {frameEdges.selected.map((edge, idx) => {
                    const u = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u || !v) return null;
                    const start = new THREE.Vector3(u.x * LAYER_SCALE_FACTOR, u.layer * LAYER_HEIGHT, u.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v.x * LAYER_SCALE_FACTOR, v.layer * LAYER_HEIGHT, v.y * LAYER_SCALE_FACTOR);
                    return <Line key={`sel-${idx}`} points={[start, end]} color={theme.edges.selected} lineWidth={2.5} renderOrder={50} />
                })}

                {frameEdges.considered.map((edge, idx) => {
                    const u = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u || !v) return null;
                    const start = new THREE.Vector3(u.x * LAYER_SCALE_FACTOR, u.layer * LAYER_HEIGHT, u.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v.x * LAYER_SCALE_FACTOR, v.layer * LAYER_HEIGHT, v.y * LAYER_SCALE_FACTOR);
                    return <Line key={`con-${idx}`} points={[start, end]} color={theme.edges.considered} lineWidth={2.5} renderOrder={50} />
                })}

                {frameEdges.rejected.map((edge, idx) => {
                    const u = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u || !v) return null;
                    const start = new THREE.Vector3(u.x * LAYER_SCALE_FACTOR, u.layer * LAYER_HEIGHT, u.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v.x * LAYER_SCALE_FACTOR, v.layer * LAYER_HEIGHT, v.y * LAYER_SCALE_FACTOR);
                    return <Line key={`rej-${idx}`} points={[start, end]} color={theme.edges.rejected} lineWidth={2.5} renderOrder={50} />
                })}
            </Canvas>
        </div>
    );
};
