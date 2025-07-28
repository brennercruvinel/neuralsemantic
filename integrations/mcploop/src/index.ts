/**
 * Neural Semantic Compiler - MCPLOOP Integration
 * 
 * Provides MCP (Model Context Protocol) integration for Claude Code CLI.
 * This allows seamless integration of Neural Semantic Compiler within Claude Code sessions.
 */

import { NeuralSemanticCLI } from '@neurosemantic/cli';
import { 
  CompressionResult, 
  CompressionOptions, 
  CompressionLevel,
  Pattern 
} from '@neurosemantic/types';
import { CompressionUtils, TextAnalyzer, FormatUtils } from '@neurosemantic/utils';

export interface MCPLOOPCommand {
  name: string;
  description: string;
  parameters?: Record<string, any>;
  execute: (args: any) => Promise<any>;
}

export interface MCPLOOPContext {
  sessionId: string;
  userId?: string;
  workspaceId?: string;
  metadata?: Record<string, any>;
}

export class NeuralSemanticMCPLoop {
  private cli: NeuralSemanticCLI;
  private context: MCPLOOPContext;
  private sessionStats: {
    compressions: number;
    totalSavings: number;
    startTime: number;
  };

  constructor(context: MCPLOOPContext) {
    this.cli = new NeuralSemanticCLI({
      verbose: false,
      timeout: 60000 // 60 second timeout for MCPLOOP
    });
    
    this.context = context;
    this.sessionStats = {
      compressions: 0,
      totalSavings: 0,
      startTime: Date.now()
    };
  }

  /**
   * Get all available MCPLOOP commands
   */
  getCommands(): MCPLOOPCommand[] {
    return [
      {
        name: 'compress',
        description: 'Compress text using Neural Semantic Compiler',
        parameters: {
          text: { type: 'string', required: true, description: 'Text to compress' },
          level: { type: 'string', enum: ['light', 'balanced', 'aggressive'], default: 'balanced' },
          domain: { type: 'string', description: 'Domain context (web-dev, agile, etc.)' },
          engine: { type: 'string', enum: ['semantic', 'hybrid', 'extreme'], description: 'Compression engine' },
          preserveCode: { type: 'boolean', default: false, description: 'Preserve code blocks' },
          preserveUrls: { type: 'boolean', default: true, description: 'Preserve URLs' },
          showStats: { type: 'boolean', default: false, description: 'Include detailed statistics' }
        },
        execute: this.executeCompress.bind(this)
      },
      {
        name: 'analyze',
        description: 'Analyze text for compression potential',
        parameters: {
          text: { type: 'string', required: true, description: 'Text to analyze' }
        },
        execute: this.executeAnalyze.bind(this)
      },
      {
        name: 'compare',
        description: 'Compare compression engines',
        parameters: {
          text: { type: 'string', required: true, description: 'Text to compare' },
          level: { type: 'string', enum: ['light', 'balanced', 'aggressive'], default: 'balanced' }
        },
        execute: this.executeCompare.bind(this)
      },
      {
        name: 'patterns',
        description: 'Manage compression patterns',
        parameters: {
          action: { type: 'string', enum: ['list', 'add', 'search'], required: true },
          domain: { type: 'string', description: 'Filter by domain' },
          search: { type: 'string', description: 'Search query' },
          original: { type: 'string', description: 'Original text (for add action)' },
          compressed: { type: 'string', description: 'Compressed text (for add action)' },
          limit: { type: 'number', default: 10, description: 'Number of results' }
        },
        execute: this.executePatterns.bind(this)
      },
      {
        name: 'stats',
        description: 'Get compression statistics',
        parameters: {},
        execute: this.executeStats.bind(this)
      },
      {
        name: 'health',
        description: 'Check system health',
        parameters: {},
        execute: this.executeHealth.bind(this)
      },
      {
        name: 'optimize',
        description: 'Optimize text for LLM consumption',
        parameters: {
          text: { type: 'string', required: true, description: 'Text to optimize' },
          targetModel: { type: 'string', enum: ['gpt-4', 'claude', 'llama'], default: 'claude' },
          maxTokens: { type: 'number', description: 'Maximum token limit' }
        },
        execute: this.executeOptimize.bind(this)
      }
    ];
  }

  /**
   * Execute compression command
   */
  private async executeCompress(args: any): Promise<any> {
    try {
      const options: CompressionOptions = {
        level: args.level as CompressionLevel,
        domain: args.domain,
        engine: args.engine,
        preserveCode: args.preserveCode,
        preserveUrls: args.preserveUrls,
        preserveNumbers: true // Default for MCPLOOP
      };

      const result = await this.cli.compress(args.text, options);
      
      // Update session stats
      this.sessionStats.compressions++;
      this.sessionStats.totalSavings += result.originalTokens - result.compressedTokens;

      const response = {
        success: true,
        result: {
          originalText: result.originalText,
          compressedText: result.compressedText,
          compression: {
            ratio: result.compressionRatio,
            percentage: Math.round((1 - result.compressionRatio) * 100),
            tokenSavings: result.originalTokens - result.compressedTokens,
            characterSavings: result.originalText.length - result.compressedText.length
          },
          quality: {
            score: result.qualityScore,
            rating: this.getQualityRating(result.qualityScore)
          },
          engine: result.engineUsed,
          processingTime: result.processingTimeMs,
          warnings: result.warnings
        },
        metadata: {
          sessionId: this.context.sessionId,
          timestamp: Date.now()
        }
      };

      if (args.showStats) {
        response.result.detailedStats = {
          originalTokens: result.originalTokens,
          compressedTokens: result.compressedTokens,
          patternMatches: result.patternMatches.length,
          sessionStats: {
            totalCompressions: this.sessionStats.compressions,
            totalSavings: this.sessionStats.totalSavings,
            averageSavings: this.sessionStats.totalSavings / this.sessionStats.compressions
          }
        };
      }

      return response;

    } catch (error) {
      return {
        success: false,
        error: (error as Error).message,
        metadata: {
          sessionId: this.context.sessionId,
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Execute text analysis command
   */
  private async executeAnalyze(args: any): Promise<any> {
    try {
      const text = args.text;
      
      // Analyze text characteristics
      const complexity = TextAnalyzer.getComplexityScore(text);
      const domain = TextAnalyzer.detectDomain(text);
      const readingTime = TextAnalyzer.getReadingTime(text);
      const keyPhrases = TextAnalyzer.extractKeyPhrases(text, 5);
      
      // Estimate compression potential
      const estimatedTokens = Math.ceil(text.length / 4);
      const estimatedCompression = this.estimateCompressionPotential(text, complexity, domain);
      
      return {
        success: true,
        analysis: {
          characteristics: {
            length: text.length,
            estimatedTokens,
            complexity: Math.round(complexity * 100),
            domain,
            readingTime,
            keyPhrases
          },
          compressionPotential: {
            estimated: estimatedCompression,
            expectedTokenSavings: Math.round(estimatedTokens * (1 - estimatedCompression)),
            recommendation: this.getCompressionRecommendation(complexity, domain, text.length)
          },
          suggestions: this.getOptimizationSuggestions(text, complexity, domain)
        },
        metadata: {
          sessionId: this.context.sessionId,
          timestamp: Date.now()
        }
      };

    } catch (error) {
      return {
        success: false,
        error: (error as Error).message
      };
    }
  }

  /**
   * Execute engine comparison command
   */
  private async executeCompare(args: any): Promise<any> {
    try {
      const comparison = await this.cli.compareEngines(args.text, {
        level: args.level as CompressionLevel
      });

      const results = Object.entries(comparison).map(([engine, result]) => ({
        engine,
        success: !('error' in result),
        compression: result.compressionRatio ? Math.round((1 - result.compressionRatio) * 100) : 0,
        quality: result.qualityScore || 0,
        tokenSavings: result.tokenSavings || 0,
        processingTime: result.processingTimeMs || 0,
        error: 'error' in result ? result.error : undefined
      }));

      // Find best engine
      const successfulResults = results.filter(r => r.success);
      const bestEngine = successfulResults.reduce((best, current) => {
        const bestScore = best.compression * 0.6 + best.quality * 0.4;
        const currentScore = current.compression * 0.6 + current.quality * 0.4;
        return currentScore > bestScore ? current : best;
      }, successfulResults[0]);

      return {
        success: true,
        comparison: {
          results,
          recommendation: {
            bestEngine: bestEngine?.engine,
            reasoning: this.generateEngineRecommendationReason(bestEngine, results)
          }
        },
        metadata: {
          sessionId: this.context.sessionId,
          timestamp: Date.now()
        }
      };

    } catch (error) {
      return {
        success: false,
        error: (error as Error).message
      };
    }
  }

  /**
   * Execute patterns management command
   */
  private async executePatterns(args: any): Promise<any> {
    try {
      switch (args.action) {
        case 'list':
          const patterns = await this.cli.getPatterns({
            domain: args.domain,
            limit: args.limit
          });
          
          return {
            success: true,
            patterns: patterns.map(p => ({
              original: p.original,
              compressed: p.compressed,
              type: p.patternType,
              domain: p.domain,
              priority: p.priority,
              usage: p.frequency
            }))
          };

        case 'search':
          if (!args.search) {
            throw new Error('Search query is required');
          }
          
          const searchResults = await this.cli.getPatterns({
            search: args.search,
            limit: args.limit
          });
          
          return {
            success: true,
            patterns: searchResults,
            query: args.search
          };

        case 'add':
          if (!args.original || !args.compressed) {
            throw new Error('Both original and compressed text are required');
          }
          
          const success = await this.cli.addPattern(args.original, args.compressed, {
            domain: args.domain || 'general'
          });
          
          return {
            success,
            message: success ? 'Pattern added successfully' : 'Failed to add pattern'
          };

        default:
          throw new Error(`Unknown action: ${args.action}`);
      }

    } catch (error) {
      return {
        success: false,
        error: (error as Error).message
      };
    }
  }

  /**
   * Execute statistics command
   */
  private async executeStats(args: any): Promise<any> {
    try {
      const stats = await this.cli.getStatistics();
      
      // Calculate session duration
      const sessionDuration = Date.now() - this.sessionStats.startTime;
      
      return {
        success: true,
        statistics: {
          session: {
            ...this.sessionStats,
            duration: FormatUtils.formatDuration(sessionDuration),
            averageCompressionTime: this.sessionStats.compressions > 0 
              ? Math.round(sessionDuration / this.sessionStats.compressions) 
              : 0
          },
          system: {
            totalPatterns: stats.patterns.totalPatterns,
            availableEngines: stats.engines.availableEngines,
            vectorStoreEnabled: stats.vectorStore.enabled
          },
          performance: {
            averageCompression: Math.round(stats.session.averageCompression * 100),
            totalTokensSaved: stats.session.totalSavings,
            estimatedCostSavings: this.calculateEstimatedCostSavings(stats.session.totalSavings)
          }
        },
        metadata: {
          sessionId: this.context.sessionId,
          timestamp: Date.now()
        }
      };

    } catch (error) {
      return {
        success: false,
        error: (error as Error).message
      };
    }
  }

  /**
   * Execute health check command
   */
  private async executeHealth(args: any): Promise<any> {
    try {
      const health = await this.cli.getHealth();
      
      return {
        success: true,
        health: {
          overall: health.overallStatus,
          components: Object.entries(health.components).map(([name, details]) => ({
            name: name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
            status: details.status,
            details: 'error' in details ? details.error : 'OK'
          })),
          issues: health.issues,
          recommendation: this.getHealthRecommendation(health)
        },
        metadata: {
          sessionId: this.context.sessionId,
          timestamp: Date.now()
        }
      };

    } catch (error) {
      return {
        success: false,
        error: (error as Error).message
      };
    }
  }

  /**
   * Execute optimization command
   */
  private async executeOptimize(args: any): Promise<any> {
    try {
      const text = args.text;
      const targetModel = args.targetModel || 'claude';
      const maxTokens = args.maxTokens;

      // First, analyze the text
      const analysis = await this.executeAnalyze({ text });
      
      if (!analysis.success) {
        return analysis;
      }

      // Determine optimal compression strategy
      const strategy = this.getOptimalStrategy(text, targetModel, maxTokens, analysis.analysis);
      
      // Apply compression with optimal settings
      const result = await this.cli.compress(text, strategy.options);
      
      // Check if we need further optimization
      const finalResult = maxTokens && result.compressedTokens > maxTokens
        ? await this.applyTokenLimitOptimization(result, maxTokens)
        : result;

      return {
        success: true,
        optimization: {
          strategy: strategy.description,
          originalTokens: result.originalTokens,
          optimizedTokens: finalResult.compressedTokens,
          tokenReduction: result.originalTokens - finalResult.compressedTokens,
          compressionRatio: finalResult.compressionRatio,
          quality: finalResult.qualityScore,
          optimizedText: finalResult.compressedText,
          fitsTokenLimit: !maxTokens || finalResult.compressedTokens <= maxTokens,
          targetModel
        },
        metadata: {
          sessionId: this.context.sessionId,
          timestamp: Date.now()
        }
      };

    } catch (error) {
      return {
        success: false,
        error: (error as Error).message
      };
    }
  }

  // Helper methods

  private getQualityRating(score: number): string {
    if (score >= 9) return 'Excellent';
    if (score >= 8) return 'Very Good';
    if (score >= 7) return 'Good';
    if (score >= 6) return 'Fair';
    if (score >= 5) return 'Poor';
    return 'Very Poor';
  }

  private estimateCompressionPotential(text: string, complexity: number, domain: string): number {
    let baseCompression = 0.7; // 30% compression

    // Adjust based on complexity
    if (complexity > 0.8) baseCompression += 0.1; // Less compression for complex text
    if (complexity < 0.3) baseCompression -= 0.1; // More compression for simple text

    // Adjust based on domain
    const domainMultipliers = {
      'web-development': 0.95,
      'agile': 0.9,
      'business': 0.85,
      'technical': 0.9,
      'general': 1.0
    };

    baseCompression *= domainMultipliers[domain as keyof typeof domainMultipliers] || 1.0;

    // Adjust based on text length
    if (text.length > 2000) baseCompression -= 0.05; // Better compression for longer text
    if (text.length < 200) baseCompression += 0.1; // Conservative for short text

    return Math.max(0.5, Math.min(0.95, baseCompression));
  }

  private getCompressionRecommendation(complexity: number, domain: string, textLength: number): string {
    if (complexity > 0.8) {
      return 'Use light compression to preserve complex information';
    } else if (textLength > 2000 && domain !== 'technical') {
      return 'Aggressive compression recommended for large non-technical text';
    } else if (domain === 'web-development' || domain === 'technical') {
      return 'Balanced compression with code preservation enabled';
    } else {
      return 'Balanced compression recommended';
    }
  }

  private getOptimizationSuggestions(text: string, complexity: number, domain: string): string[] {
    const suggestions: string[] = [];

    if (text.includes('  ')) {
      suggestions.push('Remove extra whitespace for better compression');
    }

    if (complexity > 0.8) {
      suggestions.push('Consider breaking complex sentences into simpler ones');
    }

    if (domain === 'technical' && text.includes('```')) {
      suggestions.push('Enable code preservation to maintain syntax integrity');
    }

    if (text.length > 5000) {
      suggestions.push('Consider splitting very long texts for better processing');
    }

    const redundantPhrases = ['in order to', 'it is important to note that', 'please note that'];
    const foundRedundant = redundantPhrases.filter(phrase => text.toLowerCase().includes(phrase));
    if (foundRedundant.length > 0) {
      suggestions.push('Remove redundant phrases to improve compression');
    }

    return suggestions.length > 0 ? suggestions : ['Text appears well-optimized for compression'];
  }

  private generateEngineRecommendationReason(bestEngine: any, results: any[]): string {
    if (!bestEngine) return 'No engines succeeded';

    const reasons = [];
    
    if (bestEngine.compression > 40) {
      reasons.push(`excellent compression (${bestEngine.compression}%)`);
    }
    
    if (bestEngine.quality > 8) {
      reasons.push(`high quality score (${bestEngine.quality}/10)`);
    }
    
    if (bestEngine.processingTime < 200) {
      reasons.push('fast processing');
    }

    return reasons.length > 0 
      ? `Best balance of ${reasons.join(', ')}`
      : 'Optimal performance across metrics';
  }

  private calculateEstimatedCostSavings(tokensSaved: number): Record<string, string> {
    const models = {
      'GPT-4': 0.03,      // $0.03 per 1K tokens
      'Claude': 0.015,    // $0.015 per 1K tokens  
      'GPT-3.5': 0.001,   // $0.001 per 1K tokens
    };

    const savings: Record<string, string> = {};
    
    Object.entries(models).forEach(([model, costPer1K]) => {
      const dollarsPerToken = costPer1K / 1000;
      const totalSavings = tokensSaved * dollarsPerToken;
      savings[model] = `$${totalSavings.toFixed(4)}`;
    });

    return savings;
  }

  private getHealthRecommendation(health: any): string {
    if (health.overallStatus === 'healthy') {
      return 'System is operating optimally';
    } else if (health.overallStatus === 'warning') {
      return 'System is functional but has minor issues that should be addressed';
    } else {
      return 'System has critical issues that need immediate attention';
    }
  }

  private getOptimalStrategy(text: string, targetModel: string, maxTokens?: number, analysis?: any): {
    options: CompressionOptions;
    description: string;
  } {
    const textLength = text.length;
    const isCode = text.includes('```') || text.includes('function ') || text.includes('class ');
    
    // Model-specific optimizations
    const modelPreferences = {
      'gpt-4': { level: CompressionLevel.BALANCED, preserveCode: true },
      'claude': { level: CompressionLevel.BALANCED, preserveCode: isCode },
      'llama': { level: CompressionLevel.LIGHT, preserveCode: true }
    };

    const baseOptions = modelPreferences[targetModel as keyof typeof modelPreferences] || modelPreferences['claude'];
    
    // Adjust for token limit
    if (maxTokens) {
      const estimatedTokens = Math.ceil(textLength / 4);
      const compressionNeeded = 1 - (maxTokens / estimatedTokens);
      
      if (compressionNeeded > 0.5) {
        baseOptions.level = CompressionLevel.AGGRESSIVE;
      } else if (compressionNeeded > 0.3) {
        baseOptions.level = CompressionLevel.BALANCED;
      } else {
        baseOptions.level = CompressionLevel.LIGHT;
      }
    }

    return {
      options: baseOptions,
      description: `${baseOptions.level} compression optimized for ${targetModel}${maxTokens ? ` with ${maxTokens} token limit` : ''}`
    };
  }

  private async applyTokenLimitOptimization(result: CompressionResult, maxTokens: number): Promise<CompressionResult> {
    // If still over limit, try more aggressive compression
    if (result.compressedTokens > maxTokens) {
      try {
        const aggressiveResult = await this.cli.compress(result.originalText, {
          level: CompressionLevel.AGGRESSIVE,
          preserveCode: false,
          preserveUrls: true,
          preserveNumbers: false
        });
        
        return aggressiveResult.compressedTokens <= maxTokens ? aggressiveResult : result;
      } catch {
        return result;
      }
    }
    
    return result;
  }
}

// Export for MCPLOOP registration
export function createMCPLOOPIntegration(context: MCPLOOPContext): NeuralSemanticMCPLoop {
  return new NeuralSemanticMCPLoop(context);
}

// Default export
export default NeuralSemanticMCPLoop;