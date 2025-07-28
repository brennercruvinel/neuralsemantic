/**
 * Base Engine Interface for Neural Semantic Compiler
 */

import {
  BaseEngine as IBaseEngine,
  CompressionResult,
  CompressionContext
} from '@neurosemantic/types';

export abstract class BaseEngine implements IBaseEngine {
  protected name: string;

  constructor(name: string) {
    this.name = name;
  }

  /**
   * Compress text using this engine's strategy
   */
  abstract compress(text: string, context: CompressionContext): Promise<CompressionResult>;

  /**
   * Get the engine name
   */
  getName(): string {
    return this.name;
  }

  /**
   * Check if this engine can handle the given context
   */
  abstract canHandle(context: CompressionContext): boolean;

  /**
   * Get engine-specific metadata
   */
  abstract getMetadata(): Record<string, any>;

  /**
   * Validate engine configuration
   */
  abstract validate(): { valid: boolean; errors: string[] };

  /**
   * Initialize engine resources
   */
  async initialize(): Promise<void> {
    // Default implementation - override if needed
  }

  /**
   * Cleanup engine resources
   */
  async cleanup(): Promise<void> {
    // Default implementation - override if needed
  }

  /**
   * Get engine performance statistics
   */
  abstract getStats(): Record<string, any>;
}