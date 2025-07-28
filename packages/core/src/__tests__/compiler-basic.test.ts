/**
 * Basic test suite for Neural Semantic Compiler TypeScript implementation
 */

import { NeuralSemanticCompiler } from '../compiler';
import { CompressionLevel, CompilerConfig } from '@neurosemantic/types';

describe('NeuralSemanticCompiler - Basic Tests', () => {
  let compiler: NeuralSemanticCompiler;

  beforeEach(() => {
    compiler = new NeuralSemanticCompiler();
  });

  describe('Initialization', () => {
    it('should create compiler instance', () => {
      expect(compiler).toBeDefined();
      expect(compiler).toBeInstanceOf(NeuralSemanticCompiler);
    });

    it('should initialize successfully', async () => {
      await expect(compiler.initialize()).resolves.not.toThrow();
    });

    it('should throw error if compress called before initialization', async () => {
      await expect(compiler.compress('test')).rejects.toThrow('Compiler not initialized');
    });
  });

  describe('Basic Functionality', () => {
    beforeEach(async () => {
      await compiler.initialize();
    });

    it('should perform health check', async () => {
      const health = await compiler.healthCheck();
      expect(health).toBeDefined();
      expect(health.overall).toMatch(/healthy|degraded|unhealthy/);
      expect(health.components).toBeDefined();
      expect(health.timestamp).toBeGreaterThan(0);
    });

    it('should get statistics', async () => {
      const stats = await compiler.getStatistics();
      expect(stats).toBeDefined();
      expect(stats.compiler_version).toBeDefined();
    });

    it('should add pattern', async () => {
      const result = await compiler.addPattern('test pattern', 'tp');
      expect(typeof result).toBe('boolean');
    });

    it('should clear cache without error', async () => {
      await expect(compiler.clearCache()).resolves.not.toThrow();
    });
  });

  describe('Configuration', () => {
    it('should create compiler with custom config', () => {
      const customConfig: Partial<CompilerConfig> = {
        compression: {
          defaultLevel: CompressionLevel.AGGRESSIVE,
          preserveCode: true,
          preserveUrls: true,
          preserveNumbers: true,
          minCompressionRatio: 0.5,
          maxCompressionRatio: 0.9,
          semanticThreshold: 0.8,
          targetSemanticScore: 0.85
        }
      };
      const customCompiler = new NeuralSemanticCompiler(customConfig);
      expect(customCompiler).toBeDefined();
    });
  });

  describe('Compression', () => {
    let compiler: NeuralSemanticCompiler;

    beforeEach(async () => {
      compiler = new NeuralSemanticCompiler();
      await compiler.initialize();
    });

    it('should compress simple text', async () => {
      const text = 'This is a test message for compression';
      const result = await compiler.compress(text);
      
      expect(result).toBeDefined();
      expect(result.originalText).toBe(text);
      expect(result.compressedText).toBeDefined();
      expect(result.compressionRatio).toBeGreaterThan(0);
      expect(result.compressionRatio).toBeLessThanOrEqual(1);
    });

    it('should handle empty text', async () => {
      await expect(compiler.compress('')).rejects.toThrow();
    });

    it('should preserve code blocks when configured', async () => {
      const codeText = '```javascript\nfunction test() { return true; }\n```';
      const result = await compiler.compress(codeText, { preserveCode: true });
      
      expect(result.compressedText).toContain('function test()');
    });
  });
});