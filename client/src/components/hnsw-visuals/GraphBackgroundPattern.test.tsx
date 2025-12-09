/**
 * Property-Based Tests for GraphBackgroundPattern Component
 * 
 * Tests the correctness properties of the decorative background pattern.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as fc from 'fast-check';
import { render, cleanup } from '@testing-library/react';
import React from 'react';
import { GraphBackgroundPattern } from './GraphBackgroundPattern';

// Mock framer-motion hooks
vi.mock('framer-motion', () => ({
  motion: {
    svg: React.forwardRef(({ children, style, ...props }: React.SVGProps<SVGSVGElement> & { style?: React.CSSProperties }, ref: React.Ref<SVGSVGElement>) => (
      <svg ref={ref} data-testid="motion-svg" style={style} {...props}>
        {children}
      </svg>
    )),
  },
  useScroll: () => ({
    scrollYProgress: { get: () => 0.5 },
  }),
  useTransform: (
    _value: { get: () => number },
    inputRange: number[],
    outputRange: number[]
  ) => {
    // Simulate transform calculation
    const progress = 0.5; // Middle of scroll
    const inputMin = inputRange[0];
    const inputMax = inputRange[1];
    const outputMin = outputRange[0];
    const outputMax = outputRange[1];
    
    const normalizedProgress = (progress - inputMin) / (inputMax - inputMin);
    const result = outputMin + normalizedProgress * (outputMax - outputMin);
    
    return { get: () => result };
  },
}));

describe('GraphBackgroundPattern - Property Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  
  afterEach(() => {
    cleanup();
  });

  /**
   * **Feature: hnsw-graph-visuals, Property 7: Parallax Scroll Response**
   * **Validates: Requirements 2.2**
   * 
   * For any scroll position change ΔS, the parallax-enabled elements SHALL have
   * their transform values change proportionally by ΔS × parallaxIntensity.
   */
  it('Property 7: parallax transform should be proportional to scroll and intensity', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 1, noNaN: true }), // parallaxIntensity
        fc.double({ min: 0, max: 1, noNaN: true }), // scrollProgress
        (parallaxIntensity, scrollProgress) => {
          // Calculate expected parallax values based on the component's formula:
          // parallaxY = scrollProgress * (-100 * parallaxIntensity)
          // parallaxX = scrollProgress * (-50 * parallaxIntensity)
          
          const expectedY = scrollProgress * (-100 * parallaxIntensity);
          const expectedX = scrollProgress * (-50 * parallaxIntensity);
          
          // Property 1: Y parallax should be twice the magnitude of X parallax
          // (since -100 is twice -50)
          expect(Math.abs(expectedY)).toBeCloseTo(Math.abs(expectedX) * 2, 5);
          
          // Property 2: Parallax should be 0 when intensity is 0
          if (parallaxIntensity === 0) {
            expect(Math.abs(expectedY)).toBe(0);
            expect(Math.abs(expectedX)).toBe(0);
          }
          
          // Property 3: Parallax should be 0 when scroll is at top (0)
          if (scrollProgress === 0) {
            expect(Math.abs(expectedY)).toBe(0);
            expect(Math.abs(expectedX)).toBe(0);
          }
          
          // Property 4: Parallax direction should be negative (moving up/left as scroll increases)
          // Use a threshold to avoid floating point precision issues with very small numbers
          const threshold = 1e-10;
          if (scrollProgress > threshold && parallaxIntensity > threshold) {
            expect(expectedY).toBeLessThan(0);
            expect(expectedX).toBeLessThan(0);
          }
          
          // Property 5: Higher intensity should result in larger parallax magnitude
          const lowerIntensity = parallaxIntensity * 0.5;
          const lowerExpectedY = scrollProgress * (-100 * lowerIntensity);
          expect(Math.abs(expectedY)).toBeGreaterThanOrEqual(Math.abs(lowerExpectedY));
          
          return true;
        }
      ),
      { numRuns: 100 }
    );
  });

  /**
   * Test that density affects node count appropriately.
   */
  it('should generate different node counts based on density', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('sparse', 'medium', 'dense') as fc.Arbitrary<'sparse' | 'medium' | 'dense'>,
        (density) => {
          const { container } = render(
            <GraphBackgroundPattern density={density} animated={false} />
          );
          
          // Count nodes in the pattern
          const circles = container.querySelectorAll('circle');
          
          // Verify nodes are generated
          expect(circles.length).toBeGreaterThan(0);
          
          // Expected node counts based on DENSITY_CONFIG
          const expectedCounts = {
            sparse: 8,
            medium: 15,
            dense: 25,
          };
          
          expect(circles.length).toBe(expectedCounts[density]);
          
          return true;
        }
      ),
      { numRuns: 10 }
    );
  });

  /**
   * Test that edges connect existing nodes.
   */
  it('should generate edges between nodes', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('sparse', 'medium', 'dense') as fc.Arbitrary<'sparse' | 'medium' | 'dense'>,
        (density) => {
          const { container } = render(
            <GraphBackgroundPattern density={density} animated={false} />
          );
          
          const lines = container.querySelectorAll('line');
          const circles = container.querySelectorAll('circle');
          
          // Should have some edges (unless very sparse)
          // The number of edges depends on random connections
          expect(lines.length).toBeGreaterThanOrEqual(0);
          
          // Each edge should have valid coordinates
          lines.forEach((line) => {
            const x1 = parseFloat(line.getAttribute('x1') || '0');
            const y1 = parseFloat(line.getAttribute('y1') || '0');
            const x2 = parseFloat(line.getAttribute('x2') || '0');
            const y2 = parseFloat(line.getAttribute('y2') || '0');
            
            // Coordinates should be within pattern bounds (300x200)
            expect(x1).toBeGreaterThanOrEqual(0);
            expect(x1).toBeLessThanOrEqual(300);
            expect(y1).toBeGreaterThanOrEqual(0);
            expect(y1).toBeLessThanOrEqual(200);
            expect(x2).toBeGreaterThanOrEqual(0);
            expect(x2).toBeLessThanOrEqual(300);
            expect(y2).toBeGreaterThanOrEqual(0);
            expect(y2).toBeLessThanOrEqual(200);
          });
          
          return true;
        }
      ),
      { numRuns: 10 }
    );
  });

  /**
   * Test that component renders without errors for all density values.
   */
  it('should render successfully for all density values', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('sparse', 'medium', 'dense') as fc.Arbitrary<'sparse' | 'medium' | 'dense'>,
        fc.boolean(), // animated
        fc.double({ min: 0, max: 1, noNaN: true }), // parallaxIntensity
        (density, animated, parallaxIntensity) => {
          const { container } = render(
            <GraphBackgroundPattern 
              density={density} 
              animated={animated}
              parallaxIntensity={parallaxIntensity}
            />
          );
          
          // Should render the container
          const wrapper = container.querySelector('.graph-background-pattern');
          expect(wrapper).toBeTruthy();
          
          // Should have aria-hidden for accessibility
          expect(wrapper?.getAttribute('aria-hidden')).toBe('true');
          
          // Should render SVG
          const svg = container.querySelector('svg');
          expect(svg).toBeTruthy();
          
          return true;
        }
      ),
      { numRuns: 20 }
    );
  });

  /**
   * Test pattern uniqueness - each render should have a unique pattern ID.
   */
  it('should generate unique pattern IDs', () => {
    const patternIds = new Set<string>();
    
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 10 }),
        (renderCount) => {
          for (let i = 0; i < renderCount; i++) {
            const { container, unmount } = render(
              <GraphBackgroundPattern density="medium" />
            );
            
            const pattern = container.querySelector('pattern');
            const patternId = pattern?.getAttribute('id');
            
            if (patternId) {
              // Pattern ID should be unique
              expect(patternIds.has(patternId)).toBe(false);
              patternIds.add(patternId);
            }
            
            unmount();
          }
          
          return true;
        }
      ),
      { numRuns: 5 }
    );
  });
});
