/**
 * Semantic Engine - Quality-focused compression with semantic validation
 */

import {
  CompressionResult,
  CompressionContext,
  CompressionLevel,
  PatternMatch,
  EngineType
} from '@neurosemantic/types';

import { BaseEngine } from './base-engine';

interface SemanticEngineDependencies {
  patternManager: any;
  vectorStore: any;
  qualityScorer: any;
  tokenizerManager: any;
  config: any;
}

export class SemanticEngine extends BaseEngine {
  private dependencies: SemanticEngineDependencies;
  private stats = {
    compressions: 0,
    totalTime: 0,
    avgQuality: 0,
    patternMatches: 0
  };

  constructor(dependencies: SemanticEngineDependencies) {
    super('SemanticEngine');
    this.dependencies = dependencies;
  }

  async compress(text: string, context: CompressionContext): Promise<CompressionResult> {
    const startTime = Date.now();
    
    try {
      // Step 1: Get relevant patterns
      const patterns = await this.dependencies.patternManager.getPatterns({
        domain: context.domain,
        language: context.language,
        limit: 50
      });

      // Step 2: Find pattern matches with semantic validation
      const patternMatches: PatternMatch[] = [];
      let compressedText = text;

      for (const pattern of patterns) {
        const matches = this.findPatternMatches(compressedText, pattern, context);
        
        // Validate each match semantically
        for (const match of matches) {
          if (await this.validateSemanticMatch(match, context)) {
            patternMatches.push(match);
            compressedText = compressedText.replace(match.originalText, match.compressedText);
          }
        }
      }

      // Step 3: Vector similarity enhancement (if available)
      if (this.dependencies.vectorStore.enabled) {
        const similarPatterns = await this.dependencies.vectorStore.findSimilarPatterns(
          text,
          5,
          0.8,
          context.domain
        );

        for (const similar of similarPatterns) {
          if (similar.confidence > 0.9) {
            // Apply high-confidence similar patterns
            compressedText = compressedText.replace(similar.original, similar.compressed);
          }
        }
      }

      // Step 4: Calculate tokens
      const [originalTokens, compressedTokens] = await Promise.all([
        this.dependencies.tokenizerManager.countTokens(text, context.targetModel || 'gpt-4'),
        this.dependencies.tokenizerManager.countTokens(compressedText, context.targetModel || 'gpt-4')
      ]);

      // Step 5: Build result
      const processingTime = Date.now() - startTime;
      const compressionRatio = compressedText.length / text.length;

      const result: CompressionResult = {
        originalText: text,
        compressedText,
        originalTokens: originalTokens.tokenCount,
        compressedTokens: compressedTokens.tokenCount,
        compressionRatio,
        qualityScore: 0, // Will be calculated by quality scorer
        patternMatches,
        processingTimeMs: processingTime,
        engineUsed: this.getName(),
        warnings: []
      };

      // Update stats
      this.updateStats(result);

      return result;

    } catch (error) {
      throw new Error(`Semantic engine compression failed: ${error}`);
    }
  }

  canHandle(context: CompressionContext): boolean {
    return context.level !== CompressionLevel.NONE;
  }

  getMetadata(): Record<string, any> {
    return {
      type: EngineType.SEMANTIC,
      description: 'Quality-focused compression with semantic validation',
      strengths: ['High quality', 'Semantic preservation', 'Conservative compression'],
      limitations: ['Lower compression ratios', 'Slower processing'],
      bestFor: ['Technical documentation', 'Important communications', 'Code comments']
    };
  }

  validate(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!this.dependencies.patternManager) {
      errors.push('Pattern manager is required');
    }

    if (!this.dependencies.tokenizerManager) {
      errors.push('Tokenizer manager is required');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  getStats(): Record<string, any> {
    return {
      ...this.stats,
      avgProcessingTime: this.stats.compressions > 0 ? this.stats.totalTime / this.stats.compressions : 0,
      avgPatternsPerCompression: this.stats.compressions > 0 ? this.stats.patternMatches / this.stats.compressions : 0
    };
  }

  private findPatternMatches(text: string, pattern: any, context: CompressionContext): PatternMatch[] {
    const matches: PatternMatch[] = [];
    const regex = new RegExp(this.escapeRegex(pattern.original), 'gi');
    let match;

    while ((match = regex.exec(text)) !== null) {
      // Check if we should preserve this based on context
      if (this.shouldPreserveMatch(match[0], context)) {
        continue;
      }

      matches.push({
        pattern,
        position: match.index,
        originalText: match[0],
        compressedText: pattern.compressed,
        confidence: this.calculatePatternConfidence(pattern, context),
        context: this.extractContext(text, match.index, 50)
      });
    }

    return matches;
  }

  private async validateSemanticMatch(match: PatternMatch, context: CompressionContext): Promise<boolean> {
    // For semantic engine, we're conservative - only apply high-confidence patterns
    if (match.confidence < 0.8) {
      return false;
    }

    // Check if the replacement maintains semantic meaning in context
    // This is a simplified version - in production, this would use more sophisticated NLP
    const originalWords = match.originalText.toLowerCase().split(/\s+/);
    const compressedWords = match.compressedText.toLowerCase().split(/\s+/);

    // Don't compress if it would result in ambiguity
    if (compressedWords.some(word => word.length <= 2 && originalWords.length > 1)) {
      return false;
    }

    return true;
  }

  private shouldPreserveMatch(text: string, context: CompressionContext): boolean {
    // Preserve code if requested
    if (context.preserveCode && this.looksLikeCode(text)) {
      return true;
    }

    // Preserve URLs if requested
    if (context.preserveUrls && /^https?:\/\//.test(text)) {
      return true;
    }

    // Preserve numbers if requested
    if (context.preserveNumbers && /^\d+(\.\d+)?$/.test(text)) {
      return true;
    }

    return false;
  }

  private looksLikeCode(text: string): boolean {
    // Simple heuristic to detect code-like patterns
    return /[{}();]/.test(text) || /\w+\.\w+/.test(text) || /^\w+\(/.test(text);
  }

  private calculatePatternConfidence(pattern: any, context: CompressionContext): number {
    let confidence = pattern.successRate || 0.5;

    // Boost confidence for domain-specific patterns
    if (pattern.domain === context.domain) {
      confidence += 0.2;
    }

    // Boost confidence for high-frequency patterns
    if (pattern.frequency > 10) {
      confidence += 0.1;
    }

    return Math.min(1.0, confidence);
  }

  private extractContext(text: string, position: number, radius: number): string {
    const start = Math.max(0, position - radius);
    const end = Math.min(text.length, position + radius);
    return text.substring(start, end);
  }

  private escapeRegex(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  private updateStats(result: CompressionResult): void {
    this.stats.compressions++;
    this.stats.totalTime += result.processingTimeMs;
    this.stats.avgQuality = (this.stats.avgQuality * (this.stats.compressions - 1) + result.qualityScore) / this.stats.compressions;
    this.stats.patternMatches += result.patternMatches.length;
  }
}