'use client';

/**
 * HNSWHeroCanvas Component
 * 
 * An animated HNSW graph visualization for the hero section.
 * Uses HTML5 Canvas for high-performance rendering with React.
 * 
 * Features:
 * - Multi-layer HNSW graph visualization
 * - Search traversal animation
 * - Mouse parallax interaction
 * - Viewport-based animation control
 * - Reduced motion support
 */

import React, {
  useRef,
  useEffect,
  useState,
  useCallback,
  useImperativeHandle,
  forwardRef,
} from 'react';
import {
  HNSWHeroCanvasProps,
  HNSWGraph,
  AnimationState,
  Node,
  COLORS,
  ANIMATION_TIMING,
} from './types';
import {
  generateHNSWGraph,
  createDefaultConfig,
  resetGraphTraversalState,
  selectRandomTargetNode,
  generateEdgeKey,
} from './graph-generator';

/**
 * Methods exposed via ref for programmatic control.
 */
export interface HNSWHeroCanvasRef {
  triggerTraversal: () => void;
}

/**
 * Default animation state.
 */
const createInitialAnimationState = (): AnimationState => ({
  phase: 'idle',
  currentLayer: 0,
  currentNode: null,
  visitedNodes: new Set(),
  visitedEdges: new Set(),
  targetNode: null,
  progress: 0,
});


/**
 * HNSWHeroCanvas Component
 */
const HNSWHeroCanvas = forwardRef<HNSWHeroCanvasRef, HNSWHeroCanvasProps>(
  (
    {
      layerCount = 3,
      baseNodeCount = 50,
      animationSpeed = 1.0,
      showTraversal = true,
      interactive = true,
      className = '',
    },
    ref
  ) => {
    // Canvas and context refs
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const contextRef = useRef<CanvasRenderingContext2D | null>(null);
    
    // Animation frame ref for cleanup
    const animationFrameRef = useRef<number | null>(null);
    
    // Graph state
    const graphRef = useRef<HNSWGraph | null>(null);
    const [animationState, setAnimationState] = useState<AnimationState>(
      createInitialAnimationState()
    );
    
    // Viewport visibility state
    const [isInViewport, setIsInViewport] = useState(true);
    
    // Reduced motion preference
    const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);
    
    // Mouse position for parallax
    const mousePositionRef = useRef({ x: 0, y: 0 });
    const targetMousePositionRef = useRef({ x: 0, y: 0 });
    
    // Canvas dimensions
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
    
    // Canvas support check
    const [canvasSupported, setCanvasSupported] = useState(true);
    
    // Animation timing refs
    const lastFrameTimeRef = useRef(0);
    const traversalTimerRef = useRef<number | null>(null);

    /**
     * Check for reduced motion preference.
     */
    useEffect(() => {
      const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
      setPrefersReducedMotion(mediaQuery.matches);
      
      const handleChange = (e: MediaQueryListEvent) => {
        setPrefersReducedMotion(e.matches);
      };
      
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }, []);

    /**
     * Initialize canvas and check support.
     */
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        setCanvasSupported(false);
        return;
      }
      
      contextRef.current = ctx;
      setCanvasSupported(true);
    }, []);

    /**
     * Handle canvas resize.
     */
    useEffect(() => {
      const handleResize = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const parent = canvas.parentElement;
        if (!parent) return;
        
        const rect = parent.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        
        const width = rect.width;
        const height = rect.height;
        
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        
        const ctx = contextRef.current;
        if (ctx) {
          ctx.scale(dpr, dpr);
        }
        
        setDimensions({ width, height });
      };
      
      // Debounce resize handler
      let resizeTimeout: ReturnType<typeof setTimeout>;
      const debouncedResize = () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(handleResize, 150);
      };
      
      handleResize();
      window.addEventListener('resize', debouncedResize);
      
      return () => {
        window.removeEventListener('resize', debouncedResize);
        clearTimeout(resizeTimeout);
      };
    }, []);


    /**
     * Generate graph when dimensions change.
     */
    useEffect(() => {
      if (dimensions.width === 0 || dimensions.height === 0) return;
      
      const config = createDefaultConfig(dimensions.width, dimensions.height);
      config.layerCount = layerCount;
      config.baseNodeCount = baseNodeCount;
      
      graphRef.current = generateHNSWGraph(config);
    }, [dimensions, layerCount, baseNodeCount]);

    /**
     * Viewport intersection observer for pause/resume.
     */
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const observer = new IntersectionObserver(
        (entries) => {
          const [entry] = entries;
          setIsInViewport(entry.isIntersecting);
        },
        { threshold: 0.1 }
      );
      
      observer.observe(canvas);
      
      return () => {
        observer.disconnect();
      };
    }, []);

    /**
     * Mouse move handler for parallax effect.
     */
    useEffect(() => {
      if (!interactive) return;
      
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const handleMouseMove = (e: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        // Calculate offset from center (-1 to 1)
        const offsetX = (e.clientX - rect.left - centerX) / centerX;
        const offsetY = (e.clientY - rect.top - centerY) / centerY;
        
        targetMousePositionRef.current = { x: offsetX, y: offsetY };
      };
      
      const handleMouseLeave = () => {
        targetMousePositionRef.current = { x: 0, y: 0 };
      };
      
      canvas.addEventListener('mousemove', handleMouseMove);
      canvas.addEventListener('mouseleave', handleMouseLeave);
      
      return () => {
        canvas.removeEventListener('mousemove', handleMouseMove);
        canvas.removeEventListener('mouseleave', handleMouseLeave);
      };
    }, [interactive]);

    /**
     * Trigger a new traversal animation.
     */
    const triggerTraversal = useCallback(() => {
      const graph = graphRef.current;
      if (!graph || prefersReducedMotion) return;
      
      resetGraphTraversalState(graph);
      
      const targetNode = selectRandomTargetNode(graph);
      const topLayerIndex = graph.layers.length - 1;
      
      setAnimationState({
        phase: 'traversing',
        currentLayer: topLayerIndex,
        currentNode: graph.entryPoint,
        visitedNodes: new Set([graph.entryPoint]),
        visitedEdges: new Set(),
        targetNode,
        progress: 0,
      });
    }, [prefersReducedMotion]);

    /**
     * Expose methods via ref.
     */
    useImperativeHandle(ref, () => ({
      triggerTraversal,
    }), [triggerTraversal]);


    /**
     * Draw a single node on the canvas.
     */
    const drawNode = useCallback((
      ctx: CanvasRenderingContext2D,
      node: Node,
      parallaxOffset: { x: number; y: number }
    ) => {
      const x = node.x + parallaxOffset.x * (node.layer + 1) * 5;
      const y = node.y + parallaxOffset.y * (node.layer + 1) * 3;
      
      ctx.beginPath();
      ctx.arc(x, y, node.radius, 0, Math.PI * 2);
      
      // Draw glow for highlighted/traversed nodes
      if (node.isHighlighted || node.isTraversed) {
        ctx.shadowColor = COLORS.PRIMARY_HIGHLIGHT;
        ctx.shadowBlur = 15;
        ctx.fillStyle = COLORS.PRIMARY_HIGHLIGHT;
      } else {
        ctx.shadowBlur = 0;
        ctx.fillStyle = node.color;
      }
      
      ctx.fill();
      
      // Draw border
      ctx.strokeStyle = node.isHighlighted || node.isTraversed 
        ? COLORS.PRIMARY_HIGHLIGHT 
        : COLORS.NODE_BORDER;
      ctx.lineWidth = node.isHighlighted ? 2 : 1;
      ctx.stroke();
      
      // Reset shadow
      ctx.shadowBlur = 0;
    }, []);

    /**
     * Draw edges on the canvas.
     */
    const drawEdges = useCallback((
      ctx: CanvasRenderingContext2D,
      graph: HNSWGraph,
      parallaxOffset: { x: number; y: number }
    ) => {
      for (const edge of graph.edges) {
        const fromNode = graph.nodes.get(edge.from);
        const toNode = graph.nodes.get(edge.to);
        
        if (!fromNode || !toNode) continue;
        
        const fromX = fromNode.x + parallaxOffset.x * (fromNode.layer + 1) * 5;
        const fromY = fromNode.y + parallaxOffset.y * (fromNode.layer + 1) * 3;
        const toX = toNode.x + parallaxOffset.x * (toNode.layer + 1) * 5;
        const toY = toNode.y + parallaxOffset.y * (toNode.layer + 1) * 3;
        
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        
        if (edge.isHighlighted || edge.isTraversed) {
          ctx.strokeStyle = COLORS.EDGE_TRAVERSED;
          ctx.lineWidth = 2;
          ctx.shadowColor = COLORS.PRIMARY_HIGHLIGHT;
          ctx.shadowBlur = 5;
        } else {
          // Vary opacity by layer
          const opacity = 0.1 + (edge.layer * 0.1);
          ctx.strokeStyle = `rgba(0, 0, 0, ${opacity})`;
          ctx.lineWidth = 1;
          ctx.shadowBlur = 0;
        }
        
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }, []);

    /**
     * Draw layer separator lines.
     */
    const drawLayerSeparators = useCallback((
      ctx: CanvasRenderingContext2D,
      graph: HNSWGraph
    ) => {
      const { width, height } = dimensions;
      const padding = height * 0.1;
      const usableHeight = height - 2 * padding;
      const layerHeight = usableHeight / (graph.layers.length - 1 || 1);
      
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = COLORS.LAYER_SEPARATOR;
      ctx.lineWidth = 1;
      
      for (let i = 1; i < graph.layers.length; i++) {
        const y = padding + (graph.layers.length - 1 - i) * layerHeight + layerHeight / 2;
        
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
      
      ctx.setLineDash([]);
    }, [dimensions]);


    /**
     * Update traversal animation state.
     */
    const updateTraversalAnimation = useCallback(() => {
      const graph = graphRef.current;
      if (!graph) return;
      
      setAnimationState((prev) => {
        if (prev.phase !== 'traversing') return prev;
        
        const currentNode = graph.nodes.get(prev.currentNode || '');
        if (!currentNode) return prev;
        
        // Mark current node as traversed
        currentNode.isTraversed = true;
        currentNode.color = COLORS.PRIMARY_HIGHLIGHT;
        
        // Find next node to visit
        const connections = currentNode.connections;
        let nextNode: string | null = null;
        let nextLayer = prev.currentLayer;
        
        // Prefer moving to lower layer
        for (const connId of connections) {
          const connNode = graph.nodes.get(connId);
          if (connNode && connNode.layer < prev.currentLayer && !prev.visitedNodes.has(connId)) {
            nextNode = connId;
            nextLayer = connNode.layer;
            break;
          }
        }
        
        // If no lower layer connection, try same layer
        if (!nextNode) {
          for (const connId of connections) {
            const connNode = graph.nodes.get(connId);
            if (connNode && connNode.layer === prev.currentLayer && !prev.visitedNodes.has(connId)) {
              nextNode = connId;
              break;
            }
          }
        }
        
        // Mark edge as traversed
        if (nextNode) {
          const edgeKey = generateEdgeKey(prev.currentNode!, nextNode);
          const edge = graph.edges.find(
            (e) => generateEdgeKey(e.from, e.to) === edgeKey
          );
          if (edge) {
            edge.isTraversed = true;
          }
          
          const newVisitedNodes = new Set(prev.visitedNodes);
          newVisitedNodes.add(nextNode);
          
          const newVisitedEdges = new Set(prev.visitedEdges);
          newVisitedEdges.add(edgeKey);
          
          // Check if we reached the target or base layer
          if (nextNode === prev.targetNode || nextLayer === 0) {
            // Mark target as found
            const targetNodeObj = graph.nodes.get(prev.targetNode || '');
            if (targetNodeObj) {
              targetNodeObj.isHighlighted = true;
              targetNodeObj.color = COLORS.SECONDARY_HIGHLIGHT;
            }
            
            return {
              ...prev,
              phase: 'found',
              currentNode: nextNode,
              currentLayer: nextLayer,
              visitedNodes: newVisitedNodes,
              visitedEdges: newVisitedEdges,
              progress: 1,
            };
          }
          
          return {
            ...prev,
            currentNode: nextNode,
            currentLayer: nextLayer,
            visitedNodes: newVisitedNodes,
            visitedEdges: newVisitedEdges,
            progress: prev.progress + 0.1,
          };
        }
        
        // No more nodes to visit, mark as found
        return {
          ...prev,
          phase: 'found',
          progress: 1,
        };
      });
    }, []);

    /**
     * Handle animation phase transitions.
     */
    useEffect(() => {
      if (prefersReducedMotion || !showTraversal) return;
      
      if (animationState.phase === 'idle' && isInViewport) {
        // Start a new traversal after delay
        const timer = setTimeout(() => {
          triggerTraversal();
        }, ANIMATION_TIMING.CYCLE_PAUSE);
        
        return () => clearTimeout(timer);
      }
      
      if (animationState.phase === 'traversing') {
        // Continue traversal
        const timer = setTimeout(() => {
          updateTraversalAnimation();
        }, ANIMATION_TIMING.TRAVERSAL_STEP / animationSpeed);
        
        traversalTimerRef.current = timer as unknown as number;
        
        return () => clearTimeout(timer);
      }
      
      if (animationState.phase === 'found') {
        // Reset after showing result
        const timer = setTimeout(() => {
          setAnimationState((prev) => ({
            ...prev,
            phase: 'resetting',
          }));
        }, ANIMATION_TIMING.CYCLE_PAUSE);
        
        return () => clearTimeout(timer);
      }
      
      if (animationState.phase === 'resetting') {
        // Reset graph and go to idle
        const graph = graphRef.current;
        if (graph) {
          resetGraphTraversalState(graph);
        }
        
        setAnimationState(createInitialAnimationState());
      }
    }, [
      animationState.phase,
      isInViewport,
      prefersReducedMotion,
      showTraversal,
      animationSpeed,
      triggerTraversal,
      updateTraversalAnimation,
    ]);


    /**
     * Main render loop.
     */
    useEffect(() => {
      const ctx = contextRef.current;
      const canvas = canvasRef.current;
      const graph = graphRef.current;
      
      if (!ctx || !canvas || !graph || !canvasSupported) return;
      
      // Don't animate if not in viewport or reduced motion
      if (!isInViewport && !prefersReducedMotion) {
        // Cancel any pending frame
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
        return;
      }
      
      const render = (timestamp: number) => {
        // Calculate delta time for smooth animations
        const deltaTime = timestamp - lastFrameTimeRef.current;
        lastFrameTimeRef.current = timestamp;
        
        // Smooth mouse position interpolation
        if (!prefersReducedMotion && interactive) {
          const lerp = Math.min(1, deltaTime * 0.005);
          mousePositionRef.current.x += 
            (targetMousePositionRef.current.x - mousePositionRef.current.x) * lerp;
          mousePositionRef.current.y += 
            (targetMousePositionRef.current.y - mousePositionRef.current.y) * lerp;
        }
        
        const parallaxOffset = prefersReducedMotion 
          ? { x: 0, y: 0 } 
          : mousePositionRef.current;
        
        // Clear canvas
        ctx.clearRect(0, 0, dimensions.width, dimensions.height);
        
        // Draw layer separators
        drawLayerSeparators(ctx, graph);
        
        // Draw edges
        drawEdges(ctx, graph, parallaxOffset);
        
        // Draw nodes
        for (const node of graph.nodes.values()) {
          drawNode(ctx, node, parallaxOffset);
        }
        
        // Continue animation loop if in viewport
        if (isInViewport || prefersReducedMotion) {
          animationFrameRef.current = requestAnimationFrame(render);
        }
      };
      
      // Start render loop
      animationFrameRef.current = requestAnimationFrame(render);
      
      // Cleanup
      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
      };
    }, [
      isInViewport,
      prefersReducedMotion,
      canvasSupported,
      dimensions,
      interactive,
      drawNode,
      drawEdges,
      drawLayerSeparators,
    ]);

    /**
     * Cleanup on unmount.
     */
    useEffect(() => {
      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }
        if (traversalTimerRef.current) {
          clearTimeout(traversalTimerRef.current);
          traversalTimerRef.current = null;
        }
      };
    }, []);

    // Fallback for unsupported canvas
    if (!canvasSupported) {
      return (
        <div 
          className={`hnsw-hero-fallback ${className}`}
          style={{
            width: '100%',
            height: '100%',
            background: 'linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <div style={{ opacity: 0.5 }}>HNSW Graph Visualization</div>
        </div>
      );
    }

    return (
      <canvas
        ref={canvasRef}
        className={`hnsw-hero-canvas ${className}`}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
        }}
        aria-label="HNSW graph visualization showing hierarchical search structure"
        role="img"
      />
    );
  }
);

HNSWHeroCanvas.displayName = 'HNSWHeroCanvas';

export default HNSWHeroCanvas;
export { HNSWHeroCanvas };
