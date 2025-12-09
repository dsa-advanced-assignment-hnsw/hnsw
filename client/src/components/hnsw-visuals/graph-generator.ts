/**
 * HNSW Graph Generator
 * 
 * Generates a representative HNSW graph structure for visualization.
 * The graph follows the HNSW probability distribution where higher layers
 * contain fewer nodes with longer-range connections.
 */

import {
  Node,
  Edge,
  HNSWGraph,
  GraphConfig,
  COLORS,
} from './types';

/**
 * Default layer probability (20% chance to appear in next layer).
 * This follows the HNSW algorithm's exponential decay.
 */
const DEFAULT_LAYER_PROBABILITY = 0.2;

/**
 * Default node radius for visualization.
 */
const DEFAULT_NODE_RADIUS = 6;

/**
 * Generates a unique node ID.
 */
function generateNodeId(layer: number, index: number): string {
  return `node-L${layer}-${index}`;
}

/**
 * Generates an edge key for tracking traversed edges.
 */
export function generateEdgeKey(from: string, to: string): string {
  // Sort to ensure consistent key regardless of direction
  return from < to ? `${from}-${to}` : `${to}-${from}`;
}

/**
 * Calculates the Y position for a layer based on canvas height.
 * Higher layers are positioned at the top.
 */
function getLayerY(layer: number, layerCount: number, canvasHeight: number): number {
  const padding = canvasHeight * 0.1;
  const usableHeight = canvasHeight - 2 * padding;
  const layerHeight = usableHeight / (layerCount - 1 || 1);
  // Invert so layer 0 is at bottom, highest layer at top
  return padding + (layerCount - 1 - layer) * layerHeight;
}


/**
 * Generates nodes for a specific layer.
 */
function generateLayerNodes(
  layer: number,
  nodeCount: number,
  layerCount: number,
  canvasWidth: number,
  canvasHeight: number
): Node[] {
  const nodes: Node[] = [];
  const layerY = getLayerY(layer, layerCount, canvasHeight);
  const padding = canvasWidth * 0.1;
  const usableWidth = canvasWidth - 2 * padding;
  
  for (let i = 0; i < nodeCount; i++) {
    // Distribute nodes horizontally with some randomness
    const baseX = padding + (usableWidth / (nodeCount + 1)) * (i + 1);
    const jitterX = (Math.random() - 0.5) * (usableWidth / nodeCount) * 0.3;
    const jitterY = (Math.random() - 0.5) * 20;
    
    nodes.push({
      id: generateNodeId(layer, i),
      x: baseX + jitterX,
      y: layerY + jitterY,
      layer,
      connections: [],
      isHighlighted: false,
      isTraversed: false,
      radius: DEFAULT_NODE_RADIUS,
      color: COLORS.NODE_DEFAULT,
    });
  }
  
  return nodes;
}

/**
 * Calculates the number of nodes for each layer based on HNSW probability.
 * Higher layers have exponentially fewer nodes.
 */
function calculateLayerNodeCounts(
  baseNodeCount: number,
  layerCount: number,
  layerProbability: number
): number[] {
  const counts: number[] = [];
  
  for (let layer = 0; layer < layerCount; layer++) {
    // Each layer has approximately layerProbability^layer of base nodes
    const expectedCount = Math.max(
      1,
      Math.round(baseNodeCount * Math.pow(layerProbability, layer))
    );
    counts.push(expectedCount);
  }
  
  return counts;
}

/**
 * Generates edges within a layer (horizontal connections).
 */
function generateIntraLayerEdges(
  layerNodes: Node[],
  connectionDensity: number,
  layer: number
): Edge[] {
  const edges: Edge[] = [];
  const nodeCount = layerNodes.length;
  
  if (nodeCount < 2) return edges;
  
  // Connect each node to its nearest neighbors
  for (let i = 0; i < nodeCount; i++) {
    const node = layerNodes[i];
    const connectionsToMake = Math.min(
      Math.ceil(connectionDensity),
      nodeCount - 1
    );
    
    // Sort other nodes by distance
    const otherNodes = layerNodes
      .filter((_, idx) => idx !== i)
      .map((other) => ({
        node: other,
        distance: Math.hypot(other.x - node.x, other.y - node.y),
      }))
      .sort((a, b) => a.distance - b.distance);
    
    // Connect to nearest neighbors
    for (let j = 0; j < connectionsToMake && j < otherNodes.length; j++) {
      const targetNode = otherNodes[j].node;
      
      // Avoid duplicate edges
      if (!node.connections.includes(targetNode.id)) {
        node.connections.push(targetNode.id);
        targetNode.connections.push(node.id);
        
        edges.push({
          from: node.id,
          to: targetNode.id,
          layer,
          isHighlighted: false,
          isTraversed: false,
        });
      }
    }
  }
  
  return edges;
}


/**
 * Generates edges between layers (vertical connections).
 * Nodes in higher layers connect to nodes in lower layers.
 */
function generateInterLayerEdges(
  upperLayerNodes: Node[],
  lowerLayerNodes: Node[],
  upperLayer: number
): Edge[] {
  const edges: Edge[] = [];
  
  // Each node in the upper layer connects to the nearest node(s) in the lower layer
  for (const upperNode of upperLayerNodes) {
    // Find nearest nodes in lower layer
    const nearestLower = lowerLayerNodes
      .map((lower) => ({
        node: lower,
        distance: Math.hypot(lower.x - upperNode.x, lower.y - upperNode.y),
      }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 2); // Connect to 2 nearest nodes in lower layer
    
    for (const { node: lowerNode } of nearestLower) {
      if (!upperNode.connections.includes(lowerNode.id)) {
        upperNode.connections.push(lowerNode.id);
        lowerNode.connections.push(upperNode.id);
        
        edges.push({
          from: upperNode.id,
          to: lowerNode.id,
          layer: upperLayer, // Edge belongs to upper layer
          isHighlighted: false,
          isTraversed: false,
        });
      }
    }
  }
  
  return edges;
}

/**
 * Generates a complete HNSW graph for visualization.
 * 
 * @param config - Configuration for the graph generation
 * @returns A complete HNSWGraph structure
 */
export function generateHNSWGraph(config: GraphConfig): HNSWGraph {
  const {
    layerCount,
    baseNodeCount,
    connectionDensity,
    canvasWidth,
    canvasHeight,
    layerProbability = DEFAULT_LAYER_PROBABILITY,
  } = config;
  
  // Validate inputs
  if (layerCount < 1) {
    throw new Error('layerCount must be at least 1');
  }
  if (baseNodeCount < 1) {
    throw new Error('baseNodeCount must be at least 1');
  }
  
  const nodes = new Map<string, Node>();
  const edges: Edge[] = [];
  const layers: string[][] = [];
  
  // Calculate node counts for each layer
  const layerNodeCounts = calculateLayerNodeCounts(
    baseNodeCount,
    layerCount,
    layerProbability
  );
  
  // Generate nodes for each layer
  const layerNodesArray: Node[][] = [];
  for (let layer = 0; layer < layerCount; layer++) {
    const layerNodes = generateLayerNodes(
      layer,
      layerNodeCounts[layer],
      layerCount,
      canvasWidth,
      canvasHeight
    );
    
    layerNodesArray.push(layerNodes);
    layers.push(layerNodes.map((n) => n.id));
    
    for (const node of layerNodes) {
      nodes.set(node.id, node);
    }
  }
  
  // Generate intra-layer edges (horizontal connections)
  for (let layer = 0; layer < layerCount; layer++) {
    const layerEdges = generateIntraLayerEdges(
      layerNodesArray[layer],
      connectionDensity,
      layer
    );
    edges.push(...layerEdges);
  }
  
  // Generate inter-layer edges (vertical connections)
  for (let layer = 1; layer < layerCount; layer++) {
    const interEdges = generateInterLayerEdges(
      layerNodesArray[layer],
      layerNodesArray[layer - 1],
      layer
    );
    edges.push(...interEdges);
  }
  
  // Entry point is a node in the highest layer
  const topLayer = layers[layerCount - 1];
  const entryPoint = topLayer[Math.floor(topLayer.length / 2)] || topLayer[0];
  
  return {
    nodes,
    edges,
    layers,
    entryPoint,
  };
}

/**
 * Creates a default graph configuration.
 */
export function createDefaultConfig(
  canvasWidth: number,
  canvasHeight: number
): GraphConfig {
  return {
    layerCount: 3,
    baseNodeCount: 50,
    connectionDensity: 3,
    canvasWidth,
    canvasHeight,
    layerProbability: DEFAULT_LAYER_PROBABILITY,
  };
}

/**
 * Resets the traversal state of all nodes and edges in the graph.
 */
export function resetGraphTraversalState(graph: HNSWGraph): void {
  for (const node of graph.nodes.values()) {
    node.isHighlighted = false;
    node.isTraversed = false;
    node.color = COLORS.NODE_DEFAULT;
  }
  
  for (const edge of graph.edges) {
    edge.isHighlighted = false;
    edge.isTraversed = false;
  }
}

/**
 * Finds a random target node in the base layer for search animation.
 */
export function selectRandomTargetNode(graph: HNSWGraph): string {
  const baseLayer = graph.layers[0];
  const randomIndex = Math.floor(Math.random() * baseLayer.length);
  return baseLayer[randomIndex];
}
