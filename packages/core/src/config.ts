/**
 * Configuration Management for Neural Semantic Compiler
 */

import {
  CompilerConfig as ICompilerConfig,
  DatabaseConfig,
  VectorConfig,
  CompressionConfig,
  LearningConfig,
  CompressionLevel
} from '@neurosemantic/types';

export interface CompilerConfig extends ICompilerConfig {}

export class ConfigManager {
  /**
   * Create default configuration
   */
  static createDefaultConfig(overrides?: Partial<CompilerConfig>): CompilerConfig {
    const defaultConfig: CompilerConfig = {
      database: {
        path: './neural_semantic.db',
        connectionPoolSize: 5,
        enableWalMode: true,
        cacheSizeMb: 64
      },
      vector: {
        modelName: 'sentence-transformers/all-MiniLM-L6-v2',
        persistDirectory: './chroma_db',
        similarityThreshold: 0.7,
        maxResults: 10,
        enableGpu: false
      },
      compression: {
        defaultLevel: CompressionLevel.BALANCED,
        preserveCode: true,
        preserveUrls: true,
        preserveNumbers: true,
        minCompressionRatio: 0.1,
        maxCompressionRatio: 0.9,
        semanticThreshold: 0.8,
        targetSemanticScore: 0.85
      },
      learning: {
        enableAutoDiscovery: true,
        minPatternFrequency: 3,
        patternQualityThreshold: 0.7,
        feedbackLearningRate: 0.1
      },
      logLevel: 'info',
      logFile: undefined,
      enableCaching: true,
      cacheTtlSeconds: 3600,
      maxCacheSize: 1000,
      activeDomains: ['general', 'web-development', 'agile'],
      domainWeights: {
        'general': 1.0,
        'web-development': 1.2,
        'agile': 1.1,
        'devops': 1.1
      }
    };

    return this.mergeConfig(defaultConfig, overrides || {});
  }

  /**
   * Load configuration from file
   */
  static async loadFromFile(filePath: string): Promise<CompilerConfig> {
    try {
      const fs = await import('fs').then(m => m.promises);
      const configData = await fs.readFile(filePath, 'utf-8');
      const parsed = JSON.parse(configData);
      return this.createDefaultConfig(parsed);
    } catch (error) {
      throw new Error(`Failed to load config from ${filePath}: ${error}`);
    }
  }

  /**
   * Load configuration from environment variables
   */
  static loadFromEnv(): Partial<CompilerConfig> {
    const env = process.env;
    
    return {
      database: {
        path: env.NSC_DB_PATH || './neural_semantic.db',
        connectionPoolSize: parseInt(env.NSC_DB_POOL_SIZE || '5'),
        enableWalMode: env.NSC_DB_WAL_MODE !== 'false',
        cacheSizeMb: parseInt(env.NSC_DB_CACHE_MB || '64')
      },
      vector: {
        modelName: env.NSC_VECTOR_MODEL || 'sentence-transformers/all-MiniLM-L6-v2',
        persistDirectory: env.NSC_VECTOR_DIR || './chroma_db',
        similarityThreshold: parseFloat(env.NSC_VECTOR_THRESHOLD || '0.7'),
        maxResults: parseInt(env.NSC_VECTOR_MAX_RESULTS || '10'),
        enableGpu: env.NSC_VECTOR_GPU === 'true'
      },
      compression: {
        defaultLevel: this.parseCompressionLevel(env.NSC_COMPRESSION_LEVEL),
        preserveCode: env.NSC_PRESERVE_CODE !== 'false',
        preserveUrls: env.NSC_PRESERVE_URLS !== 'false',
        preserveNumbers: env.NSC_PRESERVE_NUMBERS !== 'false',
        semanticThreshold: parseFloat(env.NSC_SEMANTIC_THRESHOLD || '0.8'),
        minCompressionRatio: 0.3,
        maxCompressionRatio: 0.9,
        targetSemanticScore: 0.9
      },
      logLevel: env.NSC_LOG_LEVEL || 'info',
      logFile: env.NSC_LOG_FILE,
      enableCaching: env.NSC_ENABLE_CACHING !== 'false',
      activeDomains: env.NSC_ACTIVE_DOMAINS?.split(',') || ['general', 'web-development', 'agile']
    };
  }

  /**
   * Validate configuration
   */
  static validateConfig(config: CompilerConfig): { valid: boolean; errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Database validation
    if (!config.database.path) {
      errors.push('Database path is required');
    }

    if (config.database.connectionPoolSize < 1 || config.database.connectionPoolSize > 100) {
      errors.push('Database connection pool size must be between 1 and 100');
    }

    // Vector store validation
    if (config.vector.similarityThreshold < 0 || config.vector.similarityThreshold > 1) {
      errors.push('Vector similarity threshold must be between 0 and 1');
    }

    if (config.vector.maxResults < 1 || config.vector.maxResults > 100) {
      errors.push('Vector max results must be between 1 and 100');
    }

    // Compression validation
    if (config.compression.minCompressionRatio >= config.compression.maxCompressionRatio) {
      errors.push('Min compression ratio must be less than max compression ratio');
    }

    if (config.compression.semanticThreshold < 0 || config.compression.semanticThreshold > 1) {
      errors.push('Semantic threshold must be between 0 and 1');
    }

    // Learning validation
    if (config.learning.minPatternFrequency < 1) {
      errors.push('Minimum pattern frequency must be at least 1');
    }

    if (config.learning.patternQualityThreshold < 0 || config.learning.patternQualityThreshold > 1) {
      errors.push('Pattern quality threshold must be between 0 and 1');
    }

    // Cache validation
    if (config.cacheTtlSeconds < 1) {
      warnings.push('Cache TTL is very low, may impact performance');
    }

    if (config.maxCacheSize < 10) {
      warnings.push('Cache size is very small, may impact performance');
    }

    // Domain validation
    if (config.activeDomains.length === 0) {
      warnings.push('No active domains specified');
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Save configuration to file
   */
  static async saveToFile(config: CompilerConfig, filePath: string): Promise<void> {
    try {
      const fs = await import('fs').then(m => m.promises);
      await fs.writeFile(filePath, JSON.stringify(config, null, 2));
    } catch (error) {
      throw new Error(`Failed to save config to ${filePath}: ${error}`);
    }
  }

  /**
   * Deep merge two configuration objects
   */
  private static mergeConfig(base: CompilerConfig, override: Partial<CompilerConfig>): CompilerConfig {
    const result = { ...base };

    for (const [key, value] of Object.entries(override)) {
      if (value !== undefined) {
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
          // Deep merge objects
          (result as any)[key] = {
            ...(result as any)[key],
            ...value
          };
        } else {
          // Direct assignment for primitives and arrays
          (result as any)[key] = value;
        }
      }
    }

    return result;
  }

  /**
   * Parse compression level from string
   */
  private static parseCompressionLevel(level?: string): CompressionLevel {
    if (!level) return CompressionLevel.BALANCED;
    
    const normalized = level.toUpperCase();
    return CompressionLevel[normalized as keyof typeof CompressionLevel] || CompressionLevel.BALANCED;
  }

  /**
   * Get configuration schema for validation
   */
  static getConfigSchema(): Record<string, any> {
    return {
      type: 'object',
      properties: {
        database: {
          type: 'object',
          properties: {
            path: { type: 'string' },
            connectionPoolSize: { type: 'number', minimum: 1, maximum: 100 },
            enableWalMode: { type: 'boolean' },
            cacheSizeMb: { type: 'number', minimum: 1 }
          },
          required: ['path']
        },
        vector: {
          type: 'object',
          properties: {
            modelName: { type: 'string' },
            persistDirectory: { type: 'string' },
            similarityThreshold: { type: 'number', minimum: 0, maximum: 1 },
            maxResults: { type: 'number', minimum: 1, maximum: 100 },
            enableGpu: { type: 'boolean' }
          }
        },
        compression: {
          type: 'object',
          properties: {
            defaultLevel: { enum: Object.values(CompressionLevel) },
            preserveCode: { type: 'boolean' },
            preserveUrls: { type: 'boolean' },
            preserveNumbers: { type: 'boolean' },
            minCompressionRatio: { type: 'number', minimum: 0, maximum: 1 },
            maxCompressionRatio: { type: 'number', minimum: 0, maximum: 1 },
            semanticThreshold: { type: 'number', minimum: 0, maximum: 1 }
          }
        },
        learning: {
          type: 'object',
          properties: {
            enableAutoDiscovery: { type: 'boolean' },
            minPatternFrequency: { type: 'number', minimum: 1 },
            patternQualityThreshold: { type: 'number', minimum: 0, maximum: 1 },
            feedbackLearningRate: { type: 'number', minimum: 0, maximum: 1 }
          }
        },
        logLevel: { enum: ['debug', 'info', 'warn', 'error'] },
        logFile: { type: 'string' },
        enableCaching: { type: 'boolean' },
        cacheTtlSeconds: { type: 'number', minimum: 1 },
        maxCacheSize: { type: 'number', minimum: 1 },
        activeDomains: {
          type: 'array',
          items: { type: 'string' }
        },
        domainWeights: {
          type: 'object',
          additionalProperties: { type: 'number' }
        }
      }
    };
  }
}

// Export default config factory
export const createDefaultConfig = ConfigManager.createDefaultConfig;