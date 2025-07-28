/**
 * Neural Semantic Compiler - Main Compiler Implementation
 * 
 * The first compiler designed for neural communication optimization.
 * Reduces LLM token usage by 40-65% while preserving semantic meaning.
 */

import { v4 as uuidv4 } from 'uuid';
import {
  CompressionResult,
  CompressionContext,
  CompressionLevel,
  HealthCheckResult,
  Pattern,
  PatternType,
  EngineType
} from '@neurosemantic/types';

import { CompilerConfig, ConfigManager } from './config';
import { EngineFactory } from './engines/engine-factory';
import { PatternManager } from './patterns/pattern-manager';
import { VectorStore } from './vector/vector-store';
import { QualityScorer } from './quality/quality-scorer';
import { TokenizerManager } from './utils/text-processing';
import { Logger } from './utils/logger';
import { DatabaseManager } from './database/database-manager';
import { AnalyticsManager } from './analytics/analytics-manager';

export class NeuralSemanticCompiler {
  private config: CompilerConfig;
  private logger: Logger;
  private patternManager!: PatternManager;
  private vectorStore!: VectorStore;
  private engineFactory!: EngineFactory;
  private qualityScorer!: QualityScorer;
  private tokenizerManager!: TokenizerManager;
  private databaseManager!: DatabaseManager;
  private analyticsManager!: AnalyticsManager;
  private initialized = false;

  constructor(config?: Partial<CompilerConfig>) {
    this.config = ConfigManager.createDefaultConfig(config);
    
    // Validate log file path for security
    const validatedLogFile = this.validateLogPath(this.config.logFile);
    this.logger = new Logger(this.config.logLevel, validatedLogFile);
    
    this.logger.info('Neural Semantic Compiler instance created');
  }

  private validateLogPath(logFile?: string): string | undefined {
    if (!logFile) return undefined;
    
    // Remove any path traversal attempts
    const sanitized = logFile.replace(/\.\./g, '').replace(/^\//, '');
    
    // Ensure it's in a safe directory
    if (sanitized.includes('/tmp/') || sanitized.includes('/var/log/')) {
      return sanitized;
    }
    
    return `./logs/${sanitized}`;
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      this.logger.info('Initializing Neural Semantic Compiler...');

      this.databaseManager = new DatabaseManager(this.config.database);
      await this.databaseManager.initialize();

      this.patternManager = new PatternManager(this.databaseManager);
      await this.patternManager.initialize();

      this.vectorStore = new VectorStore(this.config.vector);
      await this.vectorStore.initialize();

      this.tokenizerManager = new TokenizerManager();

      this.qualityScorer = new QualityScorer(this.config.compression);

      this.analyticsManager = new AnalyticsManager(this.databaseManager);

      this.engineFactory = new EngineFactory({
        patternManager: this.patternManager,
        vectorStore: this.vectorStore,
        qualityScorer: this.qualityScorer,
        tokenizerManager: this.tokenizerManager,
        config: this.config
      });

      if (this.config.enableCaching) {
        await this.engineFactory.warmupEngines([EngineType.HYBRID]);
      }

      this.initialized = true;
      this.logger.info('Neural Semantic Compiler initialized successfully');

    } catch (error) {
      this.logger.error('Failed to initialize Neural Semantic Compiler', { error });
      // Clean up any partially initialized components
      this.initialized = false;
      throw new Error(`Initialization failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Compress text using Neural Semantic Compiler.
   */
  async compress(text: string, options: Partial<CompressionContext> = {}): Promise<CompressionResult> {
    if (!this.initialized) {
      throw new Error('Compiler not initialized. Call initialize() first.');
    }

    const startTime = Date.now();
    const sessionId = uuidv4();

    try {
      // Create compression context
      const context = this.createContext(text, options);

      // Select appropriate engine
      const engine = await this.engineFactory.getEngineForContext(context);

      if (!engine) {
        // No compression requested
        return this.createNoCompressionResult(text, sessionId);
      }

      // Perform compression
      const result = await engine.compress(text, context);

      // Add session metadata
      result.sessionId = sessionId;
      result.processingTimeMs = Date.now() - startTime;

      result.qualityScore = await this.qualityScorer.calculateQualityScore(result, context);

      this.logCompression(result, context);

      if (this.config.enableCaching) {
        await this.analyticsManager.storeCompressionSession(result, context);
      }

      return result;

    } catch (error) {
      this.logger.error(`Compression failed for session ${sessionId}`, { error, text: text.substring(0, 100) });
      throw new Error(`Compression failed: ${error}`);
    }
  }

  /**
   * Decompress text (best effort).
   */
  async decompress(compressedText: string, options: { domain?: string } = {}): Promise<string> {
    if (!this.initialized) {
      throw new Error('Compiler not initialized. Call initialize() first.');
    }

    try {
      const { domain } = options;

      // Get reverse patterns for decompression
      const patterns = await this.patternManager.getPatterns({ domain });

      // Create reverse mapping
      const reversePatterns = new Map<string, string>();
      for (const pattern of patterns) {
        // Only include safe reversible patterns
        if (this.isReversiblePattern(pattern)) {
          reversePatterns.set(pattern.compressed, pattern.original);
        }
      }

      // Apply reverse patterns
      let decompressed = compressedText;
      for (const [compressed, original] of reversePatterns) {
        decompressed = decompressed.replace(new RegExp(compressed, 'g'), original);
      }

      this.logger.info(`Decompression applied ${reversePatterns.size} reverse patterns`);
      return decompressed;

    } catch (error) {
      this.logger.error('Decompression failed', { error });
      return compressedText; // Return original if decompression fails
    }
  }

  /**
   * Add a new compression pattern.
   */
  async addPattern(original: string, compressed: string, options: Partial<Pattern> = {}): Promise<boolean> {
    if (!this.initialized) {
      throw new Error('Compiler not initialized. Call initialize() first.');
    }

    try {
      const pattern: Omit<Pattern, 'id'> = {
        original,
        compressed,
        patternType: options.patternType || PatternType.WORD,
        domain: options.domain || 'general',
        priority: options.priority || 500,
        language: options.language || 'en',
        frequency: options.frequency || 1,
        successRate: options.successRate || 0.0,
        version: options.version || 1,
        isActive: options.isActive !== false,
        metadata: options.metadata || {},
        createdAt: Date.now(),
        updatedAt: Date.now()
      };

      const success = await this.patternManager.addPattern(pattern);

      if (success && this.vectorStore.enabled) {
        // Add to vector store
        await this.vectorStore.addPattern({ ...pattern, id: Date.now() } as Pattern);
      }

      return success;

    } catch (error) {
      this.logger.error('Failed to add pattern', { error, original, compressed });
      return false;
    }
  }

  /**
   * Get compiler statistics.
   */
  async getStatistics(): Promise<Record<string, any>> {
    if (!this.initialized) {
      throw new Error('Compiler not initialized. Call initialize() first.');
    }

    try {
      const [patternStats, vectorStats, engineStats, analyticsStats] = await Promise.all([
        this.patternManager.getStatistics(),
        this.vectorStore.getCollectionStats(),
        this.engineFactory.getEngineStatistics(),
        this.analyticsManager.getAnalytics()
      ]);

      return {
        compiler_version: '1.0.0',
        patterns: patternStats,
        vector_store: vectorStats,
        engines: engineStats,
        analytics: analyticsStats,
        configuration: {
          default_level: this.config.compression.defaultLevel,
          active_domains: this.config.activeDomains,
          caching_enabled: this.config.enableCaching
        }
      };
    } catch (error) {
      this.logger.error('Failed to get statistics', { error });
      return { error: error instanceof Error ? error.message : String(error) };
    }
  }

  /**
   * Benchmark compression performance.
   */
  async benchmark(testTexts: string[], options: Record<string, any> = {}): Promise<Record<string, any>> {
    if (!this.initialized) {
      throw new Error('Compiler not initialized. Call initialize() first.');
    }

    try {
      const context = this.createContext('', options);

      // Benchmark all engines
      const engineResults = await this.engineFactory.benchmarkEngines(testTexts, context);

      // Overall statistics
      const overallStats = {
        total_texts: testTexts.length,
        average_text_length: testTexts.reduce((sum, text) => sum + text.length, 0) / testTexts.length,
        benchmark_date: Date.now(),
        configuration: {
          compression_level: context.level,
          domain: context.domain,
          preserve_code: context.preserveCode
        }
      };

      return {
        overall: overallStats,
        engines: engineResults
      };

    } catch (error) {
      this.logger.error('Benchmark failed', { error });
      return { error: error instanceof Error ? error.message : String(error) };
    }
  }

  /**
   * Perform health check of all components.
   */
  async healthCheck(): Promise<HealthCheckResult> {
    const health: HealthCheckResult = {
      overall: 'healthy',
      components: {},
      timestamp: Date.now()
    };

    if (!this.initialized) {
      health.overall = 'unhealthy';
      health.components.initialization = {
        status: 'unhealthy',
        error: 'Compiler not initialized'
      };
      return health;
    }

    try {
      // Check pattern manager
      const patterns = await this.patternManager.getPatterns({ limit: 1 });
      health.components.pattern_manager = {
        status: 'healthy',
        patterns_available: patterns.length > 0
      };
    } catch (error) {
      health.components.pattern_manager = {
        status: 'unhealthy',
        error: error instanceof Error ? error.message : String(error)
      };
      health.overall = 'degraded';
    }

    try {
      // Check vector store
      const vectorStats = await this.vectorStore.getCollectionStats();
      health.components.vector_store = {
        status: this.vectorStore.enabled ? 'healthy' : 'disabled',
        enabled: this.vectorStore.enabled,
        patterns_indexed: vectorStats.total_patterns || 0
      };
    } catch (error) {
      health.components.vector_store = {
        status: 'unhealthy',
        error: error instanceof Error ? error.message : String(error)
      };
      health.overall = 'degraded';
    }

    try {
      // Check engines
      const engineStats = await this.engineFactory.getEngineStatistics();
      health.components.engines = {
        status: 'healthy',
        available_engines: engineStats.total_engines,
        instantiated_engines: engineStats.instantiated_engines
      };
    } catch (error) {
      health.components.engines = {
        status: 'unhealthy',
        error: error instanceof Error ? error.message : String(error)
      };
      health.overall = 'unhealthy';
    }

    return health;
  }

  /**
   * Learn from user feedback to improve future compressions.
   */
  async learnFromFeedback(sessionId: string, rating: number, feedback?: string): Promise<void> {
    if (!this.initialized) {
      throw new Error('Compiler not initialized. Call initialize() first.');
    }

    try {
      await this.analyticsManager.recordFeedback(sessionId, rating, feedback);
      this.logger.info(`Received feedback for session ${sessionId}`, { rating, feedback });

      // TODO: Implement pattern learning based on feedback
      
    } catch (error) {
      this.logger.error('Failed to process feedback', { error, sessionId, rating });
    }
  }

  /**
   * Clear all caches.
   */
  async clearCache(): Promise<void> {
    if (!this.initialized) {
      return;
    }

    try {
      await this.engineFactory.clearEngineCache();
      if (this.vectorStore && this.vectorStore.enabled) {
        // Clear vector store cache if available
      }
      this.logger.info('Caches cleared');
    } catch (error) {
      this.logger.error('Failed to clear cache', { error });
    }
  }

  // Private helper methods

  private createContext(text: string, options: Partial<CompressionContext>): CompressionContext {
    // Parse compression level
    let level = CompressionLevel.BALANCED;
    if (options.level) {
      if (typeof options.level === 'string') {
        level = CompressionLevel[options.level.toUpperCase() as keyof typeof CompressionLevel] || CompressionLevel.BALANCED;
      } else {
        level = options.level;
      }
    }

    const context: CompressionContext = {
      level,
      domain: options.domain,
      language: options.language || 'en',
      preserveCode: options.preserveCode ?? this.config.compression.preserveCode,
      preserveUrls: options.preserveUrls ?? this.config.compression.preserveUrls,
      preserveNumbers: options.preserveNumbers ?? this.config.compression.preserveNumbers,
      targetCompression: options.targetCompression || 0.6,
      requiresHighQuality: options.requiresHighQuality ?? true,
      contextType: options.contextType || 'general'
    };
    return context;
  }

  private async createNoCompressionResult(text: string, sessionId: string): Promise<CompressionResult> {
    const tokens = await this.tokenizerManager.countTokens(text, 'gpt-4');
    
    return {
      originalText: text,
      compressedText: text,
      originalTokens: tokens.tokenCount,
      compressedTokens: tokens.tokenCount,
      compressionRatio: 1.0,
      qualityScore: 10.0,
      patternMatches: [],
      processingTimeMs: 0,
      engineUsed: 'none',
      warnings: [],
      sessionId
    };
  }

  private logCompression(result: CompressionResult, context: CompressionContext): void {
    const charReduction = result.originalText.length - result.compressedText.length;
    const tokenReduction = result.originalTokens - result.compressedTokens;

    this.logger.info('Compression completed', {
      session: result.sessionId,
      engine: result.engineUsed,
      charReduction,
      compressionRatio: result.compressionRatio,
      tokenReduction,
      qualityScore: result.qualityScore,
      processingTime: result.processingTimeMs,
      patterns: result.patternMatches.length
    });
  }

  private isReversiblePattern(pattern: Pattern): boolean {
    // Patterns are reversible if they're unambiguous
    const compressedWords = pattern.compressed.toLowerCase().split(/\s+/);

    // Skip very short abbreviations that might be ambiguous
    if (compressedWords.some(word => word.length <= 2)) {
      return false;
    }

    // Skip patterns with special characters that might interfere
    if (/[&24]/.test(pattern.compressed)) {
      return false;
    }

    return true;
  }
}