/**
 * HNSW Graph Visualization Types
 * 
 * Type definitions for the HNSW graph visualization components.
 * These types model the hierarchical structure of HNSW graphs where
 * higher layers contain fewer nodes with longer-range connections.
 */

/**
 * Represents a node in the HNSW graph visualization.
 */
export interface Node {
  /** Unique identifier for the node */
  id: string;
  /** X coordinate position on canvas */
  x: number;
  /** Y coordinate position on canvas */
  y: number;
  /** Layer index (0 = base layer, higher = upper layers) */
  layer: number;
  /** IDs of connected nodes */
  connections: string[];
  /** Whether the node is currently highlighted */
  isHighlighted: boolean;
  /** Whether the node has been traversed in search animation */
  isTraversed: boolean;
  /** Visual radius of the node */
  radius: number;
  /** Current color of the node */
  color: string;
}

/**
 * Represents an edge connecting two nodes in the graph.
 */
export interface Edge {
  /** ID of the source node */
  from: string;
  /** ID of the target node */
  to: string;
  /** Layer where this edge exists */
  layer: number;
  /** Whether the edge is currently highlighted */
  isHighlighted: boolean;
  /** Whether the edge has been traversed in search animation */
  isTraversed: boolean;
}

/**
 * Represents the complete HNSW graph structure.
 */
export interface HNSWGraph {
  /** Map of node IDs to Node objects */
  nodes: Map<string, Node>;
  /** Array of all edges in the graph */
  edges: Edge[];
  /** Node IDs organized by layer (index = layer number) */
  layers: string[][];
  /** ID of the entry point node (top layer) */
  entryPoint: string;
}


/**
 * Configuration for generating an HNSW graph.
 */
export interface GraphConfig {
  /** Number of layers in the graph (minimum 1) */
  layerCount: number;
  /** Number of nodes in the base layer (layer 0) */
  baseNodeCount: number;
  /** Average number of connections per node */
  connectionDensity: number;
  /** Width of the canvas in pixels */
  canvasWidth: number;
  /** Height of the canvas in pixels */
  canvasHeight: number;
  /** Probability of a node appearing in the next higher layer (default: 0.2) */
  layerProbability?: number;
}

/**
 * Animation state for the search traversal visualization.
 */
export interface AnimationState {
  /** Current phase of the animation */
  phase: 'idle' | 'traversing' | 'found' | 'resetting';
  /** Current layer being traversed (decreases from top to bottom) */
  currentLayer: number;
  /** ID of the currently active node */
  currentNode: string | null;
  /** Set of node IDs that have been visited */
  visitedNodes: Set<string>;
  /** Set of edge keys (from-to) that have been traversed */
  visitedEdges: Set<string>;
  /** ID of the target node being searched for */
  targetNode: string | null;
  /** Animation progress (0-1) for interpolation */
  progress: number;
}

/**
 * Props for the HNSWHeroCanvas component.
 */
export interface HNSWHeroCanvasProps {
  /** Number of layers in the graph (default: 3) */
  layerCount?: number;
  /** Number of nodes in the base layer (default: 50) */
  baseNodeCount?: number;
  /** Animation speed multiplier (default: 1.0) */
  animationSpeed?: number;
  /** Whether to show the search traversal animation (default: true) */
  showTraversal?: boolean;
  /** Whether to enable mouse interaction (default: true) */
  interactive?: boolean;
  /** Additional CSS class names */
  className?: string;
}

/**
 * Props for the GraphBackgroundPattern component.
 */
export interface GraphBackgroundPatternProps {
  /** Density of the pattern */
  density?: 'sparse' | 'medium' | 'dense';
  /** Whether to animate the pattern */
  animated?: boolean;
  /** Parallax intensity (0-1, default: 0.3) */
  parallaxIntensity?: number;
  /** Additional CSS class names */
  className?: string;
}

/**
 * Props for the FeatureCardWithNodes component.
 */
export interface FeatureCardWithNodesProps {
  /** Icon to display */
  icon: React.ReactNode;
  /** Card title */
  title: string;
  /** Card description */
  description: string;
  /** Accent color for the card */
  accentColor: string;
  /** Optional children elements */
  children?: React.ReactNode;
}

/**
 * Represents a team member for the network visualization.
 */
export interface TeamMember {
  /** Member's name */
  name: string;
  /** Member's role */
  role: string;
  /** Background color for the member's node */
  bgColor: string;
  /** GitHub profile URL */
  github: string;
  /** Optional LinkedIn profile URL */
  linkedin?: string;
}

/**
 * Props for the TeamNetworkGraph component.
 */
export interface TeamNetworkGraphProps {
  /** Array of team members to display */
  members: TeamMember[];
  /** Whether to show connecting edges (default: true) */
  showConnections?: boolean;
}

/**
 * Color constants for the visualization.
 */
export const COLORS = {
  /** Primary highlight color (cyan) */
  PRIMARY_HIGHLIGHT: '#00F0FF',
  /** Secondary highlight color (magenta) */
  SECONDARY_HIGHLIGHT: '#FF00D6',
  /** Default node fill color */
  NODE_DEFAULT: '#FFFFFF',
  /** Default node border color */
  NODE_BORDER: '#000000',
  /** Default edge color */
  EDGE_DEFAULT: 'rgba(0, 0, 0, 0.3)',
  /** Traversed edge color */
  EDGE_TRAVERSED: '#00F0FF',
  /** Layer separator line color */
  LAYER_SEPARATOR: 'rgba(0, 0, 0, 0.1)',
} as const;

/**
 * Animation timing constants (in milliseconds).
 */
export const ANIMATION_TIMING = {
  /** Duration of a single traversal step */
  TRAVERSAL_STEP: 300,
  /** Duration of layer transition */
  LAYER_TRANSITION: 500,
  /** Duration of node pulse animation */
  NODE_PULSE: 1000,
  /** Pause between animation cycles */
  CYCLE_PAUSE: 2000,
} as const;
