import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import { Line } from '@react-three/drei';
import * as THREE from 'three';
import type { GraphState, NodeData } from '../types';
import { useSpring, animated } from '@react-spring/three';
import type { Theme } from '../theme';

interface SceneProps {
    graph: GraphState;
    highlightedNodes: Set<number>;
    activePath: number[];
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
    isRejected: boolean; // Ephemeral red highlight when rejected from W
    isInserted: boolean; // Persistent highlight for the new node during insertion
    isEntryPoint: boolean; // Highlight for entry point
    isHovered: boolean; // Hover highlight
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

    // Color using theme - Priority: Hover (Neon) > Entry Point (DeepPink) > Rejected (Red) > Active > Inserted > Candidate/Best > Visited
    let color = theme.nodes.getLayerColor(node.layer, maxLevel);
    let scale = 1;

    if (isHovered) {
        // Hover Highlight (Neon Orange/Bright) - Highest priority
        color = '#FF4500'; // OrangeRed / Neon Orange
        scale = 2.0;
    } else if (isEntryPoint) {
        // Entry Point Highlight (Deep Pink, larger)
        color = '#FF1493'; // DeepPink
        scale = 1.8;
    } else if (isRejected) {
        // Rejected nodes show red (ephemeral, this frame only)
        color = '#FF0000';
        scale = 1.3;
    } else if (isActive) {
        color = theme.nodes.activeColor;
        scale = 1.6;
    } else if (isInserted) {
        // Inserted node shows distinct color (e.g. yellow/gold)
        color = '#FFD700';
        scale = 1.4;
    } else if (inCandidate || inBest) {
        color = theme.nodes.candidateColor;
        scale = 1.3;
    } else if (inVisited) {
        // Visited nodes keep layer color but highlighted
        scale = 1.1;
    }

    return (
        <animated.group
            position={position as any}
            scale={scale}
            onClick={(e) => { e.stopPropagation(); onClick(node); }}
            onPointerOver={(e) => { e.stopPropagation(); onHover(node.id); }}
            onPointerOut={(e) => { e.stopPropagation(); onHover(null); }}
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
        </animated.group>
    );
};

// Layer Plane with grid
const LayerPlane: React.FC<{ layer: number; maxLevel: number; theme: Theme }> = ({ layer, maxLevel, theme }) => {
    const planeColor = theme.planes.getColor(layer, maxLevel);

    return (
        <group position={[0, layer * LAYER_HEIGHT, 0]}>
            {/* Semi-transparent plane with no depth write */}
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
            {/* Grid helper removed for cleaner look */}
            {/* Layer label */}
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
    searchPath: { u: number, v: number, layer: number }[];  // Orange active path (horizontal)
    searchPathVertical: { nodeId: number, fromLayer: number, toLayer: number }[]; // Orange active path (vertical)
    insertedNodeId: number | null;
    addedEdges: { u: number, v: number, layer: number }[]; // Green persistent
    prunedEdges: { u: number, v: number, layer: number }[]; // Red persistent
    prePrunedEdges: { u: number, v: number, layer: number }[]; // Initially blue (will become red)
    insertSearchPath: { u: number, v: number, layer: number }[]; // Orange per-layer
    hoveredNode: number | null;
    onNodeClick?: (node: NodeData) => void;
    onNodeHover?: (nodeId: number | null) => void;
}> = ({ theme, graph, candidateSet, visitedHistory, bestNodesAccumulated, currentNode, frameEdges, persistentEdges, searchPath, searchPathVertical, insertedNodeId, addedEdges, prunedEdges, prePrunedEdges, insertSearchPath, hoveredNode, onNodeClick, onNodeHover }) => {
    const maxLayer = useMemo(() => {
        if (!graph.nodes.length) return 0;
        return Math.max(0, ...graph.nodes.map(n => n.layer));
    }, [graph.nodes]);

    // Derive global Entry Point from graph state to ensure persistence
    const derivedEntryPoint = useMemo(() => {
        if (graph.global_entry_point === undefined || graph.global_entry_point === null) return null;
        const epNodes = graph.nodes.filter(n => n.id === graph.global_entry_point);
        if (epNodes.length === 0) return null;
        // Find instance at the highest layer
        const maxEpLayer = Math.max(...epNodes.map(n => n.layer));
        return { id: graph.global_entry_point, layer: maxEpLayer };
    }, [graph.nodes, graph.global_entry_point]);

    const layers = Array.from({ length: maxLayer + 1 }, (_, i) => i);

    const isInSet = (nodeId: number, nodeLayer: number, keySet: Set<string>) => {
        return keySet.has(`${nodeId}-${nodeLayer}`);
    };

    // Get vertical edges (connecting same node across layers)
    const verticalEdges = useMemo(() => {
        const edges: { nodeId: number, fromLayer: number, toLayer: number }[] = [];
        const nodesByLayer = new Map<number, Set<number>>();

        graph.nodes.forEach(node => {
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
    }, [graph.nodes]);

    return (
        <div className="w-full h-full" style={{ background: theme.background }}>
            <Canvas shadows camera={{ position: [25, 25, 45], fov: 50 }}>
                <color attach="background" args={[theme.canvas]} />

                {/* Lighting based on theme */}
                <ambientLight intensity={theme.lighting.ambient} />
                <directionalLight position={[10, 20, 10]} intensity={theme.lighting.directional1} castShadow />
                <directionalLight position={[-10, 15, -10]} intensity={theme.lighting.directional2} />
                <hemisphereLight args={[theme.lighting.hemisphereTop, theme.lighting.hemisphereBottom, 0.5]} />

                <PerspectiveCamera makeDefault position={[25, 25, 45]} fov={50} />
                <OrbitControls makeDefault enableDamping dampingFactor={0.05} />

                {/* Layer Planes with Grid */}
                {layers.map(l => <LayerPlane key={l} layer={l} maxLevel={maxLayer} theme={theme} />)}

                {/* Vertical Edges (between layers) */}
                {verticalEdges.map((vedge, idx) => {
                    const bottomNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.fromLayer);
                    const topNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.toLayer);

                    if (!bottomNode || !topNode) return null;

                    const start = new THREE.Vector3(
                        bottomNode.x * LAYER_SCALE_FACTOR,
                        bottomNode.layer * LAYER_HEIGHT,
                        bottomNode.y * LAYER_SCALE_FACTOR
                    );
                    const end = new THREE.Vector3(
                        topNode.x * LAYER_SCALE_FACTOR,
                        topNode.layer * LAYER_HEIGHT,
                        topNode.y * LAYER_SCALE_FACTOR
                    );

                    return (
                        <Line
                            key={`vedge-${idx}`}
                            points={[start, end]}
                            color="#cccccc"
                            lineWidth={1}
                            dashed
                            dashScale={2}
                            transparent
                            opacity={0.4}
                        />
                    );
                })}

                {/* Render Nodes */}
                {graph.nodes.map((node) => {
                    const inCandidate = isInSet(node.id, node.layer, candidateSet);
                    const inVisited = isInSet(node.id, node.layer, visitedHistory);
                    const inBest = isInSet(node.id, node.layer, bestNodesAccumulated);
                    const isActive = currentNode?.id === node.id && currentNode?.layer === node.layer;
                    const isRejected = frameEdges.rejectedNodes.has(`${node.id}-${node.layer}`); // Ephemeral red
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

                {/* Permanent Graph Edges (subtle) */}
                {graph.edges.map((edge, idx) => {
                    const u = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u || !v) return null;

                    // HNSW Insert Visualization Logic:
                    // Hide edges connected to the inserted node initially.
                    if (insertedNodeId !== null && (edge.u === insertedNodeId || edge.v === insertedNodeId)) {
                        return null;
                    }

                    const start = new THREE.Vector3(u.x * LAYER_SCALE_FACTOR, u.layer * LAYER_HEIGHT, u.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v.x * LAYER_SCALE_FACTOR, v.layer * LAYER_HEIGHT, v.y * LAYER_SCALE_FACTOR);

                    return <Line key={`edge-${idx}`} points={[start, end]} color={theme.edges.permanent} lineWidth={1.5} transparent opacity={0.6} />
                })}

                {/* Search Path Visualization - Orange path only */}

                {/* Orange Path (Active Search Path from W set) - solid orange, rendered above other elements */}
                {searchPath.map((edge, idx) => {
                    const u = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u || !v) return null;

                    const start = new THREE.Vector3(u.x * LAYER_SCALE_FACTOR, u.layer * LAYER_HEIGHT, u.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v.x * LAYER_SCALE_FACTOR, v.layer * LAYER_HEIGHT, v.y * LAYER_SCALE_FACTOR);

                    return <Line key={`spath-${idx}`} points={[start, end]} color="#FF8C00" lineWidth={4} transparent opacity={1.0} renderOrder={100} />
                })}

                {/* Orange Path Vertical Edges (persistent) - between layers */}
                {searchPathVertical.map((vedge, idx) => {
                    const bottomNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.fromLayer);
                    const topNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.toLayer);

                    if (!bottomNode || !topNode) return null;

                    const start = new THREE.Vector3(
                        bottomNode.x * LAYER_SCALE_FACTOR,
                        bottomNode.layer * LAYER_HEIGHT,
                        bottomNode.y * LAYER_SCALE_FACTOR
                    );
                    const end = new THREE.Vector3(
                        topNode.x * LAYER_SCALE_FACTOR,
                        topNode.layer * LAYER_HEIGHT,
                        topNode.y * LAYER_SCALE_FACTOR
                    );

                    return (
                        <Line
                            key={`spathv-${idx}`}
                            points={[start, end]}
                            color="#FF8C00"
                            lineWidth={4}
                            transparent
                            opacity={1.0}
                            renderOrder={100}
                        />
                    );
                })}

                {/* Frame Vertical Edges - Selected (Green dashed) - ephemeral, for layer transitions */}
                {frameEdges.selectedVertical.map((vedge, idx) => {
                    const bottomNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.fromLayer);
                    const topNode = graph.nodes.find(n => n.id === vedge.nodeId && n.layer === vedge.toLayer);

                    if (!bottomNode || !topNode) return null;

                    const start = new THREE.Vector3(
                        bottomNode.x * LAYER_SCALE_FACTOR,
                        bottomNode.layer * LAYER_HEIGHT,
                        bottomNode.y * LAYER_SCALE_FACTOR
                    );
                    const end = new THREE.Vector3(
                        topNode.x * LAYER_SCALE_FACTOR,
                        topNode.layer * LAYER_HEIGHT,
                        topNode.y * LAYER_SCALE_FACTOR
                    );

                    return (
                        <Line
                            key={`vspath-${idx}`}
                            points={[start, end]}
                            color="#FF8C00"
                            lineWidth={4}
                            dashed
                            dashScale={2}
                            transparent
                            opacity={0.8}
                            renderOrder={100}
                        />
                    );
                })}

                {/* Persistent Construction Edges (Blue) */}
                {persistentEdges.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);

                    if (!u_node || !v_node) return null;

                    // HNSW Insert Visualization Logic:
                    // Hide edges connected to the inserted node initially. 
                    // They will be revealed by 'addedEdges' (Green) when the event occurs.
                    if (insertedNodeId !== null && (edge.u === insertedNodeId || edge.v === insertedNodeId)) {
                        return null;
                    }

                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    return <Line key={`perm-${idx}`} points={[start, end]} color="#4682B4" lineWidth={2} transparent opacity={0.5} renderOrder={10} />;
                })}

                {/* Pre-Pruned Edges (Initially Blue, turn Red later via prunedEdges overlay) */}
                {prePrunedEdges.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);

                    if (!u_node || !v_node) return null;

                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    // Match graph.edges styling exactly (theme.edges.permanent, lineWidth 1.5, opacity 0.6)
                    return <Line key={`prepruned-${idx}`} points={[start, end]} color={theme.edges.permanent} lineWidth={1.5} transparent opacity={0.6} renderOrder={10} />;
                })}

                {/* Added Edges - Green Persistent (from INSERT add_connection) */}
                {addedEdges.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);

                    if (!u_node || !v_node) return null;

                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    return <Line key={`added-${idx}`} points={[start, end]} color="#32CD32" lineWidth={3} transparent opacity={1.0} renderOrder={80} />;
                })}

                {/* Pruned Edges - Red Persistent (from INSERT prune_connection) */}
                {prunedEdges.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);

                    if (!u_node || !v_node) return null;

                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    return <Line key={`pruned-${idx}`} points={[start, end]} color="#FF0000" lineWidth={3} transparent opacity={1.0} renderOrder={90} />;
                })}

                {/* Insert Search Path - Orange Per-Layer (horizontal only, cleared on layer change) */}
                {insertSearchPath.map((edge, idx) => {
                    const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);

                    if (!u_node || !v_node) return null;

                    const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                    return <Line key={`insertpath-${idx}`} points={[start, end]} color="#FF8C00" lineWidth={4} transparent opacity={1.0} renderOrder={100} />;
                })}

                {/* Hovered Edges (Neighbors & Vertical) - Neon Orange, on top of everything */}
                {hoveredNode !== null && (() => {
                    const edgesToRender: React.ReactNode[] = [];
                    // 1. Horizontal Neighbors
                    graph.edges.filter(e => e.u === hoveredNode || e.v === hoveredNode).forEach((edge, idx) => {
                        const u_node = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                        const v_node = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                        if (u_node && v_node) {
                            const start = new THREE.Vector3(u_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, u_node.y * LAYER_SCALE_FACTOR);
                            const end = new THREE.Vector3(v_node.x * LAYER_SCALE_FACTOR, edge.layer * LAYER_HEIGHT, v_node.y * LAYER_SCALE_FACTOR);
                            edgesToRender.push(<Line key={`hover-edge-${idx}`} points={[start, end]} color="#FF4500" lineWidth={4} renderOrder={200} />);
                        }
                    });

                    // 2. Vertical Connections (Same ID, different layers)
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

                {/* Frame Edges - Selected (Green) */}
                {frameEdges.selected.map((edge, idx) => {
                    const u = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u || !v) return null;

                    const start = new THREE.Vector3(u.x * LAYER_SCALE_FACTOR, u.layer * LAYER_HEIGHT, u.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v.x * LAYER_SCALE_FACTOR, v.layer * LAYER_HEIGHT, v.y * LAYER_SCALE_FACTOR);

                    return <Line key={`sel-${idx}`} points={[start, end]} color={theme.edges.selected} lineWidth={2.5} renderOrder={50} />
                })}

                {/* Frame Edges - Considered (Orange) */}
                {frameEdges.considered.map((edge, idx) => {
                    const u = graph.nodes.find(n => n.id === edge.u && n.layer === edge.layer);
                    const v = graph.nodes.find(n => n.id === edge.v && n.layer === edge.layer);
                    if (!u || !v) return null;

                    const start = new THREE.Vector3(u.x * LAYER_SCALE_FACTOR, u.layer * LAYER_HEIGHT, u.y * LAYER_SCALE_FACTOR);
                    const end = new THREE.Vector3(v.x * LAYER_SCALE_FACTOR, v.layer * LAYER_HEIGHT, v.y * LAYER_SCALE_FACTOR);

                    return <Line key={`con-${idx}`} points={[start, end]} color={theme.edges.considered} lineWidth={2.5} renderOrder={50} />
                })}

                {/* Frame Edges - Rejected (Red) */}
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
