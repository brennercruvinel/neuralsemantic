/**
 * Engine Factory for Neural Semantic Compiler
 */

import {
  EngineFactory as IEngineFactory,
  EngineType,
  CompressionContext,
  CompressionLevel,
  BenchmarkResult
} from '@neurosemantic/types';

import { BaseEngine } from './base-engine';
import { SemanticEngine } from './semantic-engine';
import { ExtremeEngine } from './extreme-engine';
import { HybridEngine } from './hybrid-engine';
import { CompilerConfig } from '../config';
import { PatternManager } from '../patterns/pattern-manager';
import { VectorStore } from '../vector/vector-store';
import { QualityScorer } from '../quality/quality-scorer';
import { TokenizerManager } from '../utils/text-processing';

interface EngineFactoryDependencies {
  patternManager: PatternManager;
  vectorStore: VectorStore;
  qualityScorer: QualityScorer;
  tokenizerManager: TokenizerManager;
  config: CompilerConfig;
}

export class EngineFactory {
  private engines: Map<EngineType, BaseEngine> = new Map();
  private dependencies: EngineFactoryDependencies;

  constructor(dependencies: EngineFactoryDependencies) {
    this.dependencies = dependencies;
  }

  /**
   * Create an engine instance
   */
  createEngine(type: EngineType): BaseEngine {
    if (this.engines.has(type)) {
      return this.engines.get(type)!;
    }

    let engine: BaseEngine;

    switch (type) {
      case EngineType.SEMANTIC:
        engine = new SemanticEngine(this.dependencies);
        break;
      case EngineType.EXTREME:
        engine = new ExtremeEngine(this.dependencies);
        break;
      case EngineType.HYBRID:
        engine = new HybridEngine(this.dependencies);
        break;
      default:
        throw new Error(`Unknown engine type: ${type}`);
    }

    this.engines.set(type, engine);
    return engine;
  }

  /**
   * Get the best engine for a given context
   */
  getEngineForContext(context: CompressionContext): BaseEngine | null {
    // Handle no compression case
    if (context.level === CompressionLevel.NONE) {
      return null;
    }

    // Use engine preference if specified
    if (context.enginePreference) {
      const engine = this.createEngine(context.enginePreference);
      if (engine.canHandle(context)) {
        return engine;
      }
      // Fall through to default selection if preferred engine can't handle context
    }

    // Select based on compression level and requirements
    switch (context.level) {
      case CompressionLevel.LIGHT:
        return this.createEngine(EngineType.SEMANTIC);
      
      case CompressionLevel.BALANCED:
        return this.createEngine(EngineType.HYBRID);
      
      case CompressionLevel.AGGRESSIVE:
        return context.requiresHighQuality 
          ? this.createEngine(EngineType.HYBRID)
          : this.createEngine(EngineType.EXTREME);
      
      default:
        return this.createEngine(EngineType.HYBRID);
    }
  }

  /**
   * Get all available engine types
   */
  getAvailableEngines(): EngineType[] {
    return Object.values(EngineType);
  }

  /**
   * Warm up engines by creating instances
   */
  async warmupEngines(engineTypes: EngineType[]): Promise<void> {
    const warmupPromises = engineTypes.map(async (type) => {
      try {
        const engine = this.createEngine(type);
        await engine.initialize();
      } catch (error) {
        console.warn(`Failed to warm up engine ${type}:`, error);
      }
    });

    await Promise.all(warmupPromises);
  }

  /**
   * Validate engine configuration
   */
  validateConfiguration(): { valid: boolean; errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Test each engine type
    for (const engineType of this.getAvailableEngines()) {
      try {
        const engine = this.createEngine(engineType);
        const validation = engine.validate();
        
        if (!validation.valid) {
          const errorMsg = `${engineType} engine validation failed: ${validation.errors.join(', ')}`;
          errors.push(errorMsg);
        }
      } catch (error) {
        errors.push(`Failed to create ${engineType} engine: ${error}`);
      }
    }

    // Check dependencies
    if (!this.dependencies.patternManager) {
      errors.push('Pattern manager dependency missing');
    }

    if (!this.dependencies.tokenizerManager) {
      errors.push('Tokenizer manager dependency missing');
    }

    if (!this.dependencies.qualityScorer) {
      warnings.push('Quality scorer dependency missing - quality scores may be inaccurate');
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Get engine statistics
   */
  async getEngineStatistics(): Promise<Record<string, any>> {
    const stats: Record<string, any> = {
      total_engines: this.getAvailableEngines().length,
      instantiated_engines: this.engines.size,
      engines: {}
    };

    for (const [type, engine] of this.engines) {
      try {
        stats.engines[type] = {
          name: engine.getName(),
          metadata: engine.getMetadata(),
          stats: engine.getStats()
        };
      } catch (error) {
        stats.engines[type] = {
          error: error instanceof Error ? error.message : String(error)
        };
      }
    }

    return stats;
  }

  /**
   * Benchmark all engines
   */
  async benchmarkEngines(testTexts: string[], context: CompressionContext): Promise<Record<string, BenchmarkResult>> {
    const results: Record<string, BenchmarkResult> = {};

    for (const engineType of this.getAvailableEngines()) {
      try {
        const engine = this.createEngine(engineType);
        const startTime = Date.now();
        let totalCompressions = 0;
        let totalErrors = 0;
        let totalCompressionRatio = 0;

        for (const text of testTexts) {
          try {
            const result = await engine.compress(text, context);
            totalCompressions++;
            totalCompressionRatio += result.compressionRatio;
          } catch (error) {
            totalErrors++;
          }
        }

        const endTime = Date.now();
        const avgProcessingTime = (endTime - startTime) / testTexts.length;
        const avgCompressionRatio = totalCompressions > 0 ? totalCompressionRatio / totalCompressions : 0;
        const successRate = totalCompressions / testTexts.length;

        results[engineType] = {
          engineName: engine.getName(),
          averageCompressionRatio: avgCompressionRatio,
          averageProcessingTime: avgProcessingTime,
          successRate: successRate,
          totalCompressions,
          totalErrors
        };

      } catch (error) {
        results[engineType] = {
          engineName: engineType,
          averageCompressionRatio: 0,
          averageProcessingTime: 0,
          successRate: 0,
          totalCompressions: 0,
          totalErrors: testTexts.length
        };
      }
    }

    return results;
  }

  /**
   * Clear engine cache
   * 
   * Memory Management Strategy:
   * - Engines are cached in memory for performance
   * - Call this method to free memory when:
   *   - Application is shutting down
   *   - Memory usage is high
   *   - After processing large batches
   * - Each engine's cleanup method releases its resources
   * - The cache is completely cleared after cleanup
   */
  async clearEngineCache(): Promise<void> {
    const cleanupPromises = Array.from(this.engines.values()).map(engine => 
      engine.cleanup().catch(error => 
        console.warn(`Failed to cleanup engine ${engine.getName()}:`, error)
      )
    );

    await Promise.all(cleanupPromises);
    this.engines.clear();
  }

  /**
   * Get engine by type (creates if not exists)
   */
  getEngine(type: EngineType): BaseEngine {
    return this.createEngine(type);
  }
}