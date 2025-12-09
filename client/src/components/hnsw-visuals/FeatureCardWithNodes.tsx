'use client';

/**
 * FeatureCardWithNodes Component
 * 
 * An enhanced feature card with graph-node decorations.
 * Features corner node decorations and animated edge connections on hover.
 * 
 * Features:
 * - Corner node decorations
 * - Animated edge connections on hover
 * - Customizable accent color
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FeatureCardWithNodesProps, COLORS } from './types';

/**
 * Corner node positions (relative to card bounds).
 */
const CORNER_NODES = [
  { id: 'tl', x: 0, y: 0 },     // Top-left
  { id: 'tr', x: 100, y: 0 },   // Top-right
  { id: 'bl', x: 0, y: 100 },   // Bottom-left
  { id: 'br', x: 100, y: 100 }, // Bottom-right
];

/**
 * Edge connections between corner nodes.
 */
const CORNER_EDGES = [
  { from: 'tl', to: 'tr' },
  { from: 'tr', to: 'br' },
  { from: 'br', to: 'bl' },
  { from: 'bl', to: 'tl' },
  { from: 'tl', to: 'br' }, // Diagonal
  { from: 'tr', to: 'bl' }, // Diagonal
];

/**
 * FeatureCardWithNodes Component
 */
export const FeatureCardWithNodes: React.FC<FeatureCardWithNodesProps> = ({
  icon,
  title,
  description,
  accentColor,
  children,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  // Get node position by ID
  const getNodePosition = (id: string) => {
    return CORNER_NODES.find((n) => n.id === id) || { x: 0, y: 0 };
  };

  return (
    <motion.div
      className="feature-card-with-nodes relative p-6 rounded-xl bg-white border border-gray-200 shadow-sm overflow-hidden"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.2 }}
    >
      {/* SVG overlay for nodes and edges */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none"
        preserveAspectRatio="none"
        viewBox="0 0 100 100"
        aria-hidden="true"
      >
        {/* Animated edges on hover */}
        {CORNER_EDGES.map((edge, index) => {
          const from = getNodePosition(edge.from);
          const to = getNodePosition(edge.to);
          
          return (
            <motion.line
              key={`edge-${index}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={isHovered ? accentColor : COLORS.EDGE_DEFAULT}
              strokeWidth="0.5"
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{
                pathLength: isHovered ? 1 : 0,
                opacity: isHovered ? 0.6 : 0,
              }}
              transition={{
                duration: 0.3,
                delay: index * 0.05,
              }}
            />
          );
        })}
        
        {/* Corner nodes */}
        {CORNER_NODES.map((node) => (
          <motion.circle
            key={node.id}
            cx={node.x}
            cy={node.y}
            r="2"
            fill={isHovered ? accentColor : COLORS.NODE_DEFAULT}
            stroke={isHovered ? accentColor : COLORS.NODE_BORDER}
            strokeWidth="0.5"
            initial={{ scale: 0, opacity: 0 }}
            animate={{
              scale: isHovered ? 1 : 0.5,
              opacity: isHovered ? 1 : 0.3,
            }}
            transition={{ duration: 0.2 }}
          />
        ))}
      </svg>

      {/* Card content */}
      <div className="relative z-10">
        {/* Icon */}
        <div
          className="w-12 h-12 rounded-lg flex items-center justify-center mb-4"
          style={{ backgroundColor: `${accentColor}20` }}
        >
          <div style={{ color: accentColor }}>{icon}</div>
        </div>

        {/* Title */}
        <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>

        {/* Description */}
        <p className="text-gray-600 text-sm leading-relaxed">{description}</p>

        {/* Optional children */}
        {children && <div className="mt-4">{children}</div>}
      </div>

      {/* Hover glow effect */}
      <motion.div
        className="absolute inset-0 rounded-xl pointer-events-none"
        style={{
          boxShadow: `0 0 30px ${accentColor}40`,
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: isHovered ? 1 : 0 }}
        transition={{ duration: 0.3 }}
      />
    </motion.div>
  );
};

export default FeatureCardWithNodes