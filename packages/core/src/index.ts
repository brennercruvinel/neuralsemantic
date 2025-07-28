/**
 * Neural Semantic Compiler - Core TypeScript Implementation
 * 
 * Copyright (c) 2024 Brenner Cruvinel (@brennercruvinel)
 * All Rights Reserved.
 * 
 * PROPRIETARY AND CONFIDENTIAL
 * This software contains proprietary algorithms and trade secrets.
 * Unauthorized copying, reverse engineering, or distribution is strictly prohibited.
 * 
 * For licensing inquiries: cruvinelbrenner@gmail.com
 */

export { NeuralSemanticCompiler } from './compiler';
export { CompilerConfig, ConfigManager } from './config';

export { BaseEngine } from './engines/base-engine';
export { SemanticEngine } from './engines/semantic-engine';
export { ExtremeEngine } from './engines/extreme-engine';
export { HybridEngine } from './engines/hybrid-engine';
export { EngineFactory } from './engines/engine-factory';

export { PatternManager } from './patterns/pattern-manager';
export { ConflictResolver } from './patterns/conflict-resolver';

export { VectorStore } from './vector/vector-store';
export { EmbeddingManager } from './vector/embedding-manager';

export { TextProcessor, TokenizerManager } from './utils/text-processing';
export { MetricsCollector, PerformanceProfiler } from './utils/metrics';
export { Logger } from './utils/logger';
export { CacheManager } from './utils/cache';

export { DatabaseManager } from './database/database-manager';

export { QualityScorer } from './quality/quality-scorer';

export { AnalyticsManager } from './analytics/analytics-manager';

export * from '@neurosemantic/types';

export const VERSION = '1.0.0';
export const BUILD_DATE = new Date().toISOString();

export const createCompiler = async (config?: Partial<CompilerConfig>): Promise<NeuralSemanticCompiler> => {
  const compiler = new NeuralSemanticCompiler(config);
  await compiler.initialize();
  return compiler;
};

export const compress = async (
  text: string,
  options?: {
    level?: 'light' | 'balanced' | 'aggressive';
    domain?: string;
    preserveCode?: boolean;
  }
): Promise<{ compressed: string; ratio: number; tokens: { original: number; compressed: number } }> => {
  const compiler = await createCompiler();
  const result = await compiler.compress(text, options);
  
  return {
    compressed: result.compressedText,
    ratio: result.compressionRatio,
    tokens: {
      original: result.originalTokens,
      compressed: result.compressedTokens
    }
  };
};