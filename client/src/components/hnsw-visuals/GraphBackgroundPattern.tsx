'use client';

/**
 * GraphBackgroundPattern Component
 * 
 * An SVG-based decorative background pattern with nodes and edges.
 * Supports scroll-linked parallax effects using Framer Motion.
 * 
 * Features:
 * - Configurable density (sparse/medium/dense)
 * - Scroll-linked parallax animation
 * - Responsive design
 */

import React, { useMemo } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { GraphBackgroundPatternProps, COLORS } from './types';

/**
 * Node configuration for pattern generation.
 */
interface PatternNode {
  id: string;
  cx: number;
  cy: number;
  r: number;
}

/**
 * Edge configuration for pattern generation.
 */
interface PatternEdge {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * Density configuration mapping.
 */
const DENSITY_CONFIG = {
  sparse: { nodeCount: 8, connectionProbability: 0.2 },
  medium: { nodeCount: 15, connectionProbability: 0.3 },
  dense: { nodeCount: 25, connectionProbability: 0.4 },
} as const;

/**
 * Seeded random number generator for consistent patterns.
 */
function seededRandom(seed: number): () => number {
  return () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };
}

/**
 * Generate pattern nodes and edges based on density.
 */
function generatePattern(
  density: 'sparse' | 'medium' | 'dense',
  width: number,
  height: number,
  seed: number = 42
): { nodes: PatternNode[]; edges: PatternEdge[] } {
  const config = DENSITY_CONFIG[density];
  const random = seededRandom(seed);
  
  const nodes: PatternNode[] = [];
  const edges: PatternEdge[] = [];
  
  // Generate nodes with some padding from edges
  const padding = 20;
  for (let i = 0; i < config.nodeCount; i++) {
    nodes.push({
      id: `node-${i}`,
      cx: padding + random() * (width - 2 * padding),
      cy: padding + random() * (height - 2 * padding),
      r: 3 + random() * 4,
    });
  }
  
  // Generate edges between nearby nodes
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[j].cx - nodes[i].cx;
      const dy = nodes[j].cy - nodes[i].cy;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      // Connect nodes that are relatively close
      const maxDistance = Math.min(width, height) * 0.4;
      if (distance < maxDistance && random() < config.connectionProbability) {
        edges.push({
          id: `edge-${i}-${j}`,
          x1: nodes[i].cx,
          y1: nodes[i].cy,
          x2: nodes[j].cx,
          y2: nodes[j].cy,
        });
      }
    }
  }
  
  return { nodes, edges };
}

/**
 * GraphBackgroundPattern Component
 */
export const GraphBackgroundPattern: React.FC<GraphBackgroundPatternProps> = ({
  density = 'medium',
  animated = true,
  parallaxIntensity = 0.3,
  className = '',
}) => {
  // Pattern dimensions (will be tiled via SVG pattern)
  const patternWidth = 300;
  const patternHeight = 200;
  
  // Generate pattern data
  const { nodes, edges } = useMemo(
    () => generatePattern(density, patternWidth, patternHeight),
    [density]
  );
  
  // Scroll-linked parallax
  const { scrollYProgress } = useScroll();
  
  // Transform scroll progress to parallax offset
  const parallaxY = useTransform(
    scrollYProgress,
    [0, 1],
    [0, -100 * parallaxIntensity]
  );
  
  const parallaxX = useTransform(
    scrollYProgress,
    [0, 1],
    [0, -50 * parallaxIntensity]
  );

  // Unique pattern ID to avoid conflicts
  const patternId = useMemo(
    () => `graph-pattern-${density}-${Math.random().toString(36).substr(2, 9)}`,
    [density]
  );

  return (
    <div
      className={`graph-background-pattern absolute inset-0 overflow-hidden pointer-events-none ${className}`}
      aria-hidden="true"
    >
      <motion.svg
        className="w-full h-full"
        style={{
          x: animated ? parallaxX : 0,
          y: animated ? parallaxY : 0,
        }}
        preserveAspectRatio="xMidYMid slice"
      >
        <defs>
          <pattern
            id={patternId}
            x="0"
            y="0"
            width={patternWidth}
            height={patternHeight}
            patternUnits="userSpaceOnUse"
          >
            {/* Render edges */}
            {edges.map((edge) => (
              <line
                key={edge.id}
                x1={edge.x1}
                y1={edge.y1}
                x2={edge.x2}
                y2={edge.y2}
                stroke={COLORS.EDGE_DEFAULT}
                strokeWidth="1"
                opacity="0.3"
              />
            ))}
            
            {/* Render nodes */}
            {nodes.map((node) => (
              <g key={node.id}>
                {/* Node fill */}
                <circle
                  cx={node.cx}
                  cy={node.cy}
                  r={node.r}
                  fill={COLORS.NODE_DEFAULT}
                  stroke={COLORS.NODE_BORDER}
                  strokeWidth="1"
                  opacity="0.6"
                />
              </g>
            ))}
          </pattern>
        </defs>
        
        {/* Fill the entire SVG with the pattern */}
        <rect
          x="0"
          y="0"
          width="100%"
          height="100%"
          fill={`url(#${patternId})`}
        />
      </motion.svg>
    </div>
  );
};

export default GraphBackgroundPattern;
