'use client';

/**
 * TeamNetworkGraph Component
 * 
 * A network visualization for team members.
 * Renders team members as nodes with animated connecting edges.
 * 
 * Features:
 * - Team members displayed as nodes
 * - Animated connecting edges between members
 * - Hover interactions
 */

import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { TeamNetworkGraphProps, TeamMember, COLORS } from './types';

/**
 * Calculate node positions in a circular layout.
 */
function calculateNodePositions(
  memberCount: number,
  centerX: number,
  centerY: number,
  radius: number
): { x: number; y: number }[] {
  const positions: { x: number; y: number }[] = [];
  
  for (let i = 0; i < memberCount; i++) {
    const angle = (2 * Math.PI * i) / memberCount - Math.PI / 2;
    positions.push({
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    });
  }
  
  return positions;
}

/**
 * Generate edges between team members.
 * Creates a fully connected graph for small teams, or selective connections for larger teams.
 */
function generateEdges(memberCount: number): { from: number; to: number }[] {
  const edges: { from: number; to: number }[] = [];
  
  // For small teams (<=6), create a fully connected graph
  // For larger teams, connect each member to their neighbors and a few others
  if (memberCount <= 6) {
    for (let i = 0; i < memberCount; i++) {
      for (let j = i + 1; j < memberCount; j++) {
        edges.push({ from: i, to: j });
      }
    }
  } else {
    // Connect to immediate neighbors
    for (let i = 0; i < memberCount; i++) {
      const next = (i + 1) % memberCount;
      edges.push({ from: i, to: next });
      
      // Connect to member across (for visual interest)
      const across = (i + Math.floor(memberCount / 2)) % memberCount;
      if (across !== i && across !== next) {
        edges.push({ from: i, to: across });
      }
    }
  }
  
  return edges;
}

/**
 * TeamNetworkGraph Component
 */
export const TeamNetworkGraph: React.FC<TeamNetworkGraphProps> = ({
  members,
  showConnections = true,
}) => {
  const [hoveredMember, setHoveredMember] = useState<number | null>(null);
  
  // SVG dimensions
  const width = 600;
  const height = 400;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.35;
  const nodeRadius = 40;
  
  // Calculate positions and edges
  const positions = useMemo(
    () => calculateNodePositions(members.length, centerX, centerY, radius),
    [members.length, centerX, centerY, radius]
  );
  
  const edges = useMemo(
    () => generateEdges(members.length),
    [members.length]
  );

  return (
    <div className="team-network-graph relative w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Animated edges */}
        {showConnections && edges.map((edge, index) => {
          const from = positions[edge.from];
          const to = positions[edge.to];
          const isHighlighted = 
            hoveredMember === edge.from || hoveredMember === edge.to;
          
          return (
            <motion.line
              key={`edge-${index}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke={isHighlighted ? COLORS.PRIMARY_HIGHLIGHT : COLORS.EDGE_DEFAULT}
              strokeWidth={isHighlighted ? 2 : 1}
              initial={{ pathLength: 0 }}
              animate={{ 
                pathLength: 1,
                opacity: isHighlighted ? 1 : 0.4,
              }}
              transition={{
                pathLength: { duration: 0.5, delay: index * 0.1 },
                opacity: { duration: 0.2 },
              }}
            />
          );
        })}
        
        {/* Team member nodes */}
        {members.map((member, index) => {
          const pos = positions[index];
          const isHovered = hoveredMember === index;
          
          return (
            <g
              key={member.name}
              onMouseEnter={() => setHoveredMember(index)}
              onMouseLeave={() => setHoveredMember(null)}
              style={{ cursor: 'pointer' }}
            >
              {/* Node background circle */}
              <motion.circle
                cx={pos.x}
                cy={pos.y}
                r={nodeRadius}
                fill={member.bgColor}
                stroke={isHovered ? COLORS.PRIMARY_HIGHLIGHT : COLORS.NODE_BORDER}
                strokeWidth={isHovered ? 3 : 1}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                whileHover={{ scale: 1.1 }}
                transition={{
                  scale: { type: 'spring', stiffness: 300, damping: 20 },
                  delay: index * 0.1,
                }}
              />
              
              {/* Glow effect on hover */}
              {isHovered && (
                <motion.circle
                  cx={pos.x}
                  cy={pos.y}
                  r={nodeRadius + 5}
                  fill="none"
                  stroke={COLORS.PRIMARY_HIGHLIGHT}
                  strokeWidth={2}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.6 }}
                  transition={{ duration: 0.2 }}
                />
              )}
              
              {/* Member initials */}
              <text
                x={pos.x}
                y={pos.y}
                textAnchor="middle"
                dominantBaseline="central"
                fill="white"
                fontSize="16"
                fontWeight="bold"
                style={{ pointerEvents: 'none' }}
              >
                {member.name
                  .split(' ')
                  .map((n) => n[0])
                  .join('')
                  .toUpperCase()
                  .slice(0, 2)}
              </text>
            </g>
          );
        })}
      </svg>
      
      {/* Member info cards below the graph */}
      <div className="flex flex-wrap justify-center gap-4 mt-6">
        {members.map((member, index) => (
          <motion.div
            key={member.name}
            className="text-center p-3 rounded-lg bg-white shadow-sm border border-gray-100"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 + 0.5 }}
            onMouseEnter={() => setHoveredMember(index)}
            onMouseLeave={() => setHoveredMember(null)}
            style={{
              borderColor: hoveredMember === index ? COLORS.PRIMARY_HIGHLIGHT : undefined,
              boxShadow: hoveredMember === index 
                ? `0 0 15px ${COLORS.PRIMARY_HIGHLIGHT}40` 
                : undefined,
            }}
          >
            <div className="font-semibold text-gray-900">{member.name}</div>
            <div className="text-sm text-gray-500">{member.role}</div>
            <div className="flex justify-center gap-2 mt-2">
              {member.github && (
                <a
                  href={member.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                  aria-label={`${member.name}'s GitHub`}
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                  </svg>
                </a>
              )}
              {member.linkedin && (
                <a
                  href={member.linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-blue-600 transition-colors"
                  aria-label={`${member.name}'s LinkedIn`}
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                  </svg>
                </a>
              )}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default TeamNetworkGraph;