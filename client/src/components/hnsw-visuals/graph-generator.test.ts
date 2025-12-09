/**
 * Property-Based Tests for HNSW Graph Generator
 * 
 * **Feature: hnsw-graph-visuals, Property 1: HNSW Layer Structure Invariant**
 * **Validates: Requirements 1.1, 1.2, 5.2**
 * 
 * Tests that the generated HNSW graph maintains the layer structure invariant:
 * - Higher layers have fewer or equal nodes compared to lower layers
 * - The total layer count matches the configured layerCount prop
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { generateHNSWGraph } from './graph-generator';
import type { GraphConfig } from './types';

/**
 * Arbitrary for generating valid GraphConfig objects.
 * Constrains values to reasonable ranges for visualization.
 */
const graphConfigArbitrary = fc.record({
  layerCount: fc.integer({ min: 1, max: 5 }),
  baseNodeCount: fc.integer({ min: 5, max: 100 }),
  connectionDensity: fc.integer({ min: 1, max: 5 }),
  canvasWidth: fc.integer({ min: 400, max: 1920 }),
  canvasHeight: fc.integer({ min: 300, max: 1080 }),
  layerProbability: fc.double({ min: 0.1, max: 0.5, noNaN: true }),
});

describe('HNSW Graph Generator - Property Tests', () => {
  /**
   * **Feature: hnsw-graph-visuals, Property 1: HNSW Layer Structure Invariant**
   * **Validates: Requirements 1.1, 1.2, 5.2**
   * 
   * For any generated HNSW graph with N layers, the node count at layer[i]
   * SHALL be greater than or equal to the node count at layer[i+1] for all
   * valid layer indices, and the total layer count SHALL match the configured
   * layerCount prop.
   */
  it('should maintain layer structure invariant: lower layers have >= nodes than higher layers', () => {
    fc.assert(
      fc.property(graphConfigArbitrary, (config: GraphConfig) => {
        const graph = generateHNSWGraph(config);
        
        // Property 1a: Total layer count matches configuration
        expect(graph.layers.length).toBe(config.layerCount);
        
        // Property 1b: Each layer has at least one node
        for (let i = 0; i < graph.layers.length; i++) {
          expect(graph.layers[i].length).toBeGreaterThanOrEqual(1);
        }
        
        // Property 1c: Lower layers have >= nodes than higher layers
        // layer[0] is base (most nodes), layer[n-1] is top (fewest nodes)
        for (let i = 0; i < graph.layers.length - 1; i++) {
          const lowerLayerCount = graph.layers[i].length;
          const higherLayerCount = graph.layers[i + 1].length;
          
          expect(lowerLayerCount).toBeGreaterThanOrEqual(higherLayerCount);
        }
        
        // Property 1d: All node IDs in layers exist in the nodes map
        for (const layer of graph.layers) {
          for (const nodeId of layer) {
            expect(graph.nodes.has(nodeId)).toBe(true);
          }
        }
        
        // Property 1e: Entry point exists and is in the top layer
        expect(graph.entryPoint).toBeDefined();
        expect(graph.layers[graph.layers.length - 1]).toContain(graph.entryPoint);
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Node layer assignment is consistent.
   */
  it('should assign nodes to correct layers', () => {
    fc.assert(
      fc.property(graphConfigArbitrary, (config: GraphConfig) => {
        const graph = generateHNSWGraph(config);
        
        // Each node's layer property should match its position in the layers array
        for (let layerIndex = 0; layerIndex < graph.layers.length; layerIndex++) {
          for (const nodeId of graph.layers[layerIndex]) {
            const node = graph.nodes.get(nodeId);
            expect(node).toBeDefined();
            expect(node!.layer).toBe(layerIndex);
          }
        }
      }),
      { numRuns: 100 }
    );
  });

  /**
   * Additional property: Edges connect valid nodes.
   */
  it('should only create edges between existing nodes', () => {
    fc.assert(
      fc.property(graphConfigArbitrary, (config: GraphConfig) => {
        const graph = generateHNSWGraph(config);
        
        for (const edge of graph.edges) {
          expect(graph.nodes.has(edge.from)).toBe(true);
          expect(graph.nodes.has(edge.to)).toBe(true);
        }
      }),
      { numRuns: 100 }
    );
  });
});
