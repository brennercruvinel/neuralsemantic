/**
 * MCPLOOP Integration for Claude Code CLI
 * 
 * This integration allows Neural Semantic Compiler to work seamlessly
 * with Claude Code CLI, intercepting prompts and compressing them before
 * they're sent to the LLM.
 */

import { NeuralSemanticCompiler } from '@neurosemantic/core';
import { CompressionLevel, CompressionOptions } from '@neurosemantic/types';
import {
  PromptData,
  CompressedPromptData,
  SessionAnalytics,
  ContextHints,
  CompressionStrategy,
  MCPLoopConfig,
  AnalyticsEvent
} from './types';
import { CostCalculator } from './cost-calculator';
import { AnalyticsCollector } from './analytics-collector';
import { PromptInterceptor } from './prompt-interceptor';
import { Logger } from './utils/logger';

export class MCPLoopIntegration {
  private compiler: NeuralSemanticCompiler;
  private costCalculator: CostCalculator;
  private analytics: AnalyticsCollector;
  private interceptor: PromptInterceptor;
  private logger: Logger;
  private config: MCPLoopConfig;
  private strategies: CompressionStrategy[];
  private totalCostSavings: number = 0;

  constructor(config: Partial<MCPLoopConfig> = {}) {
    this.config = this.mergeDefaultConfig(config);
    this.logger = new Logger(this.config.logging.level);
    
    this.initializeComponents();
    this.setupDefaultStrategies();

    this.logger.info('MCPLOOP Integration initialized');
  }

  /**
   * Initialize all components
   */
  private async initializeComponents(): Promise<void> {
    // Initialize core compiler
    this.compiler = NeuralSemanticCompiler.createDefault();

    // Initialize cost calculator
    this.costCalculator = new CostCalculator(this.config.costCalculation);

    // Initialize analytics collector
    this.analytics = new AnalyticsCollector(this.config.analytics);

    // Initialize prompt interceptor
    this.interceptor = new PromptInterceptor(this.strategies);
  }

  /**
   * Main method to intercept and compress Claude Code CLI prompts
   */
  async interceptPrompt(promptData: PromptData): Promise<CompressedPromptData> {
    const startTime = Date.now();

    try {
      this.logger.debug('Intercepting prompt:', {
        sessionId: promptData.sessionId,
        promptLength: promptData.prompt.length,
        files: promptData.files?.length || 0
      });

      // Extract context hints from prompt data
      const contextHints = this.extractContextHints(promptData);

      // Determine if we should compress this prompt
      const shouldCompress = this.interceptor.shouldCompress(promptData);
      
      if (!shouldCompress) {
        this.logger.debug('Prompt skipped - does not meet compression criteria');
        return this.createPassthroughResult(promptData);
      }

      // Get compression strategy
      const strategy = this.interceptor.selectStrategy(promptData, contextHints);

      // Perform compression
      const compressionOptions = this.buildCompressionOptions(strategy, contextHints);
      const result = await this.compiler.compress(promptData.prompt, compressionOptions);

      // Calculate cost savings
      const costSavings = this.costCalculator.calculateSavings(
        result.originalTokens,
        result.compressedTokens
      );

      // Track total savings
      this.totalCostSavings += costSavings.savings;

      // Create compressed prompt data
      const compressedData: CompressedPromptData = {
        ...promptData,
        prompt: result.compressedText,
        originalPrompt: promptData.prompt,
        nscMetadata: {
          originalLength: promptData.prompt.length,
          compressedLength: result.compressedText.length,
          compressionRatio: result.compressionRatio,
          tokenSavings: result.originalTokens - result.compressedTokens,
          costSavings: costSavings.savings,
          qualityScore: result.qualityScore,
          engineUsed: result.engineUsed,
          processingTimeMs: Date.now() - startTime,
          patternsApplied: result.patternMatches.length,
          warnings: result.warnings
        }
      };

      // Record analytics
      this.analytics.recordCompression({
        type: 'compression',
        sessionId: promptData.sessionId,
        data: {
          result,
          costSavings,
          strategy: strategy.name,
          contextHints
        },
        timestamp: Date.now()
      });

      this.logger.info('Prompt compressed successfully:', {
        sessionId: promptData.sessionId,
        compression: `${(result.compressionRatio * 100).toFixed(1)}%`,
        tokenSavings: result.originalTokens - result.compressedTokens,
        costSavings: `$${costSavings.savings.toFixed(4)}`,
        quality: `${result.qualityScore.toFixed(1)}/10`
      });

      return compressedData;

    } catch (error) {
      this.logger.error('Prompt compression failed:', error);

      // Record error analytics
      this.analytics.recordCompression({
        type: 'error',
        sessionId: promptData.sessionId,
        data: { error: error.message },
        timestamp: Date.now()
      });

      // Return original prompt on error
      return this.createPassthroughResult(promptData, error);
    }
  }

  /**
   * Get session analytics for a specific session
   */
  getSessionAnalytics(sessionId?: string): SessionAnalytics | Record<string, SessionAnalytics> {
    if (sessionId) {
      return this.analytics.getSessionData(sessionId);
    }
    return this.analytics.getAllSessionData();
  }

  /**
   * Generate savings report for user
   */
  generateSavingsReport(): string {
    const totalSessions = this.analytics.getTotalSessions();
    const overallStats = this.analytics.getOverallStatistics();

    if (totalSessions === 0) {
      return "No compression sessions recorded yet.";
    }

    const avgCompression = (1 - overallStats.averageCompressionRatio) * 100;
    const monthlySavings = this.totalCostSavings * 30; // Rough monthly estimate

    return `
 Neural Semantic Compiler - MCPLOOP Integration Report

 Cost Savings: $${this.totalCostSavings.toFixed(2)}
 Tokens Saved: ${overallStats.totalTokensSaved.toLocaleString()}
 Average Compression: ${avgCompression.toFixed(1)}%
Average Quality: ${overallStats.averageQualityScore.toFixed(1)}/10
 Sessions: ${totalSessions}
 Estimated Monthly Savings: $${monthlySavings.toFixed(2)}

 Keep using NSC to maximize your Claude Code CLI savings!
    `.trim();
  }

  /**
   * Add custom compression strategy
   */
  addCompressionStrategy(strategy: CompressionStrategy): void {
    this.strategies.push(strategy);
    this.strategies.sort((a, b) => b.priority - a.priority);
    this.interceptor.updateStrategies(this.strategies);
    
    this.logger.info(`Added compression strategy: ${strategy.name}`);
  }

  /**
   * Export analytics data
   */
  async exportAnalytics(format: 'json' | 'csv' = 'json'): Promise<string> {
    return this.analytics.exportData(format);
  }

  /**
   * Get real-time metrics
   */
  getMetrics() {
    return {
      totalCostSavings: this.totalCostSavings,
      totalSessions: this.analytics.getTotalSessions(),
      overallStats: this.analytics.getOverallStatistics(),
      recentActivity: this.analytics.getRecentActivity(24), // Last 24 hours
      topStrategies: this.getTopStrategies()
    };
  }

  /**
   * Health check for the integration
   */
  async healthCheck(): Promise<boolean> {
    try {
      // Test compiler
      const testResult = await this.compiler.compress("test compression", {
        level: CompressionLevel.LIGHT
      });

      // Test analytics
      const analyticsWorking = this.analytics.isHealthy();

      // Test cost calculator
      const costCalcWorking = this.costCalculator.isHealthy();

      return testResult && analyticsWorking && costCalcWorking;

    } catch (error) {
      this.logger.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * Shutdown integration and cleanup resources
   */
  async shutdown(): Promise<void> {
    this.logger.info('Shutting down MCPLOOP Integration');

    try {
      await this.compiler.close();
      await this.analytics.shutdown();
      
      this.logger.info('MCPLOOP Integration shutdown complete');
    } catch (error) {
      this.logger.error('Error during shutdown:', error);
    }
  }

  // Private helper methods

  private extractContextHints(promptData: PromptData): ContextHints {
    const hints: ContextHints = {};

    // Extract from file extensions
    if (promptData.files && promptData.files.length > 0) {
      const extensions = promptData.files.map(f => f.split('.').pop()?.toLowerCase());
      
      // Determine project type from file extensions
      if (extensions.includes('py')) hints.language = 'python';
      if (extensions.includes('js') || extensions.includes('ts')) hints.language = 'javascript';
      if (extensions.includes('java')) hints.language = 'java';
      if (extensions.includes('go')) hints.language = 'go';
      
      // Determine framework
      if (extensions.includes('tsx') || extensions.includes('jsx')) hints.framework = 'react';
      if (promptData.files.some(f => f.includes('package.json'))) hints.projectType = 'web';
      if (promptData.files.some(f => f.includes('Dockerfile'))) hints.domain = 'devops';
    }

    // Extract from prompt content
    const prompt = promptData.prompt.toLowerCase();
    
    // Domain detection
    if (prompt.includes('react') || prompt.includes('frontend') || prompt.includes('ui')) {
      hints.domain = 'web-development';
    } else if (prompt.includes('agile') || prompt.includes('scrum') || prompt.includes('sprint')) {
      hints.domain = 'agile';
    } else if (prompt.includes('docker') || prompt.includes('kubernetes') || prompt.includes('deploy')) {
      hints.domain = 'devops';
    }

    // Complexity detection
    const promptLength = promptData.prompt.length;
    if (promptLength < 500) {
      hints.complexity = 'simple';
    } else if (promptLength < 2000) {
      hints.complexity = 'medium';
    } else {
      hints.complexity = 'complex';
    }

    return hints;
  }

  private buildCompressionOptions(strategy: CompressionStrategy, hints: ContextHints): CompressionOptions {
    return {
      ...strategy.compressionOptions,
      domain: hints.domain || strategy.compressionOptions.domain,
      contextType: hints.projectType || 'general'
    };
  }

  private createPassthroughResult(promptData: PromptData, error?: Error): CompressedPromptData {
    return {
      ...promptData,
      originalPrompt: promptData.prompt,
      nscMetadata: {
        originalLength: promptData.prompt.length,
        compressedLength: promptData.prompt.length,
        compressionRatio: 1.0,
        tokenSavings: 0,
        costSavings: 0,
        qualityScore: 10.0,
        engineUsed: 'none',
        processingTimeMs: 0,
        patternsApplied: 0,
        warnings: error ? [`Compression failed: ${error.message}`] : ['Prompt skipped - no compression applied']
      }
    };
  }

  private setupDefaultStrategies(): void {
    this.strategies = [
      {
        name: 'Web Development High Compression',
        description: 'Aggressive compression for web development prompts',
        conditions: [
          { type: 'keyword', value: 'react', weight: 0.8 },
          { type: 'keyword', value: 'frontend', weight: 0.7 },
          { type: 'fileExtension', value: '.tsx', weight: 0.9 }
        ],
        compressionOptions: {
          level: CompressionLevel.AGGRESSIVE,
          domain: 'web-development',
          requiresHighQuality: false
        },
        priority: 900
      },
      {
        name: 'Agile Documentation Compression',
        description: 'Balanced compression for agile/project management content',
        conditions: [
          { type: 'keyword', value: 'sprint', weight: 0.8 },
          { type: 'keyword', value: 'agile', weight: 0.7 },
          { type: 'keyword', value: 'scrum', weight: 0.8 }
        ],
        compressionOptions: {
          level: CompressionLevel.BALANCED,
          domain: 'agile',
          requiresHighQuality: true
        },
        priority: 800
      },
      {
        name: 'General Purpose Compression',
        description: 'Default compression for general prompts',
        conditions: [],
        compressionOptions: {
          level: CompressionLevel.BALANCED,
          requiresHighQuality: true
        },
        priority: 100
      }
    ];
  }

  private getTopStrategies() {
    return this.analytics.getStrategyUsage()
      .sort((a, b) => b.usage - a.usage)
      .slice(0, 5);
  }

  private mergeDefaultConfig(config: Partial<MCPLoopConfig>): MCPLoopConfig {
    return {
      enabled: true,
      port: 8765,
      host: 'localhost',
      compressionOptions: {
        level: CompressionLevel.BALANCED,
        requiresHighQuality: true
      },
      costCalculation: {
        enabled: true,
        defaultProvider: 'anthropic',
        pricingModels: {
          'claude-3-sonnet': { inputCostPer1K: 0.003, outputCostPer1K: 0.015 },
          'claude-3-haiku': { inputCostPer1K: 0.00025, outputCostPer1K: 0.00125 },
          'gpt-4': { inputCostPer1K: 0.03, outputCostPer1K: 0.06 },
          'gpt-3.5-turbo': { inputCostPer1K: 0.001, outputCostPer1K: 0.002 }
        }
      },
      analytics: {
        enabled: true,
        sessionTimeout: 30,
        retentionDays: 30
      },
      security: {
        enableAuth: false,
        allowedOrigins: ['*']
      },
      logging: {
        level: 'info'
      },
      ...config
    };
  }
}