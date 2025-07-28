/**
 * Hybrid Engine - Balanced compression combining semantic validation with aggressive techniques
 */

import {
  CompressionResult,
  CompressionContext,
  CompressionLevel,
  PatternMatch,
  EngineType
} from '@neurosemantic/types';

import { BaseEngine } from './base-engine';

interface HybridEngineDependencies {
  patternManager: any;
  vectorStore: any;
  qualityScorer: any;
  tokenizerManager: any;
  config: any;
}

export class HybridEngine extends BaseEngine {
  private dependencies: HybridEngineDependencies;
  private stats = {
    compressions: 0,
    totalTime: 0,
    avgQuality: 0,
    avgCompressionRatio: 0,
    patternMatches: 0,
    rollbacks: 0
  };

  constructor(dependencies: HybridEngineDependencies) {
    super('HybridEngine');
    this.dependencies = dependencies;
  }

  async compress(text: string, context: CompressionContext): Promise<CompressionResult> {
    const startTime = Date.now();
    
    try {
      // Multi-stage compression with quality validation
      let currentText = text;
      const allPatternMatches: PatternMatch[] = [];
      const compressionStages: Array<{ name: string; text: string; quality?: number }> = [];

      // Stage 1: High-confidence pattern application (Semantic approach)
      const stage1Result = await this.applyHighConfidencePatterns(currentText, context);
      currentText = stage1Result.text;
      allPatternMatches.push(...stage1Result.matches);
      compressionStages.push({ name: 'high_confidence_patterns', text: currentText });

      // Stage 2: Vector similarity enhancement (if available)
      if (this.dependencies.vectorStore.enabled) {
        const stage2Result = await this.applySimilarityPatterns(currentText, context);
        currentText = stage2Result.text;
        allPatternMatches.push(...stage2Result.matches);
        compressionStages.push({ name: 'similarity_patterns', text: currentText });
      }

      // Stage 3: Selective aggressive compression
      const stage3Result = await this.applySelectiveAggression(currentText, context);
      currentText = stage3Result.text;
      allPatternMatches.push(...stage3Result.matches);
      compressionStages.push({ name: 'selective_aggression', text: currentText });

      // Stage 4: Quality validation and potential rollback
      const finalResult = await this.validateAndOptimize(text, currentText, compressionStages, context);
      currentText = finalResult.text;

      // Calculate final metrics
      const [originalTokens, compressedTokens] = await Promise.all([
        this.dependencies.tokenizerManager.countTokens(text, context.targetModel || 'gpt-4'),
        this.dependencies.tokenizerManager.countTokens(currentText, context.targetModel || 'gpt-4')
      ]);

      const processingTime = Date.now() - startTime;
      const compressionRatio = currentText.length / text.length;

      const result: CompressionResult = {
        originalText: text,
        compressedText: currentText,
        originalTokens: originalTokens.tokenCount,
        compressedTokens: compressedTokens.tokenCount,
        compressionRatio,
        qualityScore: 0, // Will be calculated by quality scorer
        patternMatches: allPatternMatches,
        processingTimeMs: processingTime,
        engineUsed: this.getName(),
        warnings: this.generateWarnings(compressionRatio, finalResult.hadRollback)
      };

      this.updateStats(result, finalResult.hadRollback);
      return result;

    } catch (error) {
      throw new Error(`Hybrid engine compression failed: ${error}`);
    }
  }

  canHandle(context: CompressionContext): boolean {
    return context.level !== CompressionLevel.NONE;
  }

  getMetadata(): Record<string, any> {
    return {
      type: EngineType.HYBRID,
      description: 'Balanced compression with quality validation and rollback capability',
      strengths: ['Balanced compression', 'Quality validation', 'Adaptive approach', 'Rollback safety'],
      limitations: ['Higher computational cost', 'Complex tuning'],
      bestFor: ['General purpose', 'Production systems', 'Mixed content types']
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

    if (!this.dependencies.qualityScorer) {
      errors.push('Quality scorer is recommended for hybrid engine');
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
      avgPatternsPerCompression: this.stats.compressions > 0 ? this.stats.patternMatches / this.stats.compressions : 0,
      rollbackRate: this.stats.compressions > 0 ? this.stats.rollbacks / this.stats.compressions : 0
    };
  }

  private async applyHighConfidencePatterns(text: string, context: CompressionContext): Promise<{ text: string; matches: PatternMatch[] }> {
    const patterns = await this.dependencies.patternManager.getPatterns({
      domain: context.domain,
      language: context.language,
      minSuccessRate: 0.8 // Only high-success patterns
    });

    let currentText = text;
    const matches: PatternMatch[] = [];

    for (const pattern of patterns) {
      const patternMatches = this.findPatternMatches(currentText, pattern, context);
      
      for (const match of patternMatches) {
        if (this.validateMatch(match, context)) {
          matches.push(match);
          currentText = currentText.replace(match.originalText, match.compressedText);
        }
      }
    }

    return { text: currentText, matches };
  }

  private async applySimilarityPatterns(text: string, context: CompressionContext): Promise<{ text: string; matches: PatternMatch[] }> {
    if (!this.dependencies.vectorStore.enabled) {
      return { text, matches: [] };
    }

    let currentText = text;
    const matches: PatternMatch[] = [];

    try {
      const similarPatterns = await this.dependencies.vectorStore.findSimilarPatterns(
        text,
        10,
        0.75, // Moderate similarity threshold
        context.domain
      );

      for (const similar of similarPatterns) {
        if (similar.confidence > 0.85) {
          // Apply only high-confidence similar patterns
          const regex = new RegExp(this.escapeRegex(similar.original), 'gi');
          if (regex.test(currentText)) {
            const match: PatternMatch = {
              pattern: similar,
              position: currentText.search(regex),
              originalText: similar.original,
              compressedText: similar.compressed,
              confidence: similar.confidence,
              context: this.extractContext(currentText, currentText.search(regex), 40)
            };

            matches.push(match);
            currentText = currentText.replace(regex, similar.compressed);
          }
        }
      }
    } catch (error) {
      // Vector store operations are optional - continue without them
      console.warn('Vector similarity patterns failed:', error);
    }

    return { text: currentText, matches };
  }

  private async applySelectiveAggression(text: string, context: CompressionContext): Promise<{ text: string; matches: PatternMatch[] }> {
    let currentText = text;
    const matches: PatternMatch[] = [];

    // Apply moderate compression techniques based on context
    if (context.level === CompressionLevel.BALANCED || context.level === CompressionLevel.AGGRESSIVE) {
      // Safe abbreviations
      const safeAbbreviations = {
        'information': 'info',
        'development': 'dev',
        'configuration': 'config',
        'application': 'app',
        'documentation': 'docs'
      };

      for (const [full, abbr] of Object.entries(safeAbbreviations)) {
        const regex = new RegExp(`\\b${full}\\b`, 'gi');
        if (regex.test(currentText)) {
          currentText = currentText.replace(regex, abbr);
          // Note: Not tracking these as pattern matches for simplicity
        }
      }

      // Safe structural improvements
      currentText = currentText.replace(/\bin order to\b/gi, 'to');
      currentText = currentText.replace(/\bdue to the fact that\b/gi, 'because');
      
      // Cleanup whitespace
      currentText = currentText.replace(/\s+/g, ' ').trim();
    }

    return { text: currentText, matches };
  }

  private async validateAndOptimize(
    originalText: string,
    compressedText: string,
    stages: Array<{ name: string; text: string; quality?: number }>,
    context: CompressionContext
  ): Promise<{ text: string; hadRollback: boolean }> {
    
    const compressionRatio = compressedText.length / originalText.length;
    let hadRollback = false;

    // Check if compression is too aggressive
    if (compressionRatio < 0.2) {
      // Too aggressive - rollback to earlier stage
      const safeStage = stages.find(stage => {
        const stageRatio = stage.text.length / originalText.length;
        return stageRatio >= 0.3;
      });

      if (safeStage) {
        hadRollback = true;
        return { text: safeStage.text, hadRollback };
      }
    }

    // Check quality requirements
    if (context.requiresHighQuality && this.dependencies.qualityScorer) {
      try {
        const quality = await this.dependencies.qualityScorer.estimateQuality(
          originalText,
          compressedText,
          context
        );

        if (quality < 0.8) {
          // Quality too low - try a more conservative stage
          const betterStage = stages.slice().reverse().find(stage => {
            const stageRatio = stage.text.length / originalText.length;
            return stageRatio > compressionRatio * 1.2; // Less aggressive
          });

          if (betterStage) {
            hadRollback = true;
            return { text: betterStage.text, hadRollback };
          }
        }
      } catch (error) {
        // Quality scoring failed - continue with current result
        console.warn('Quality validation failed:', error);
      }
    }

    return { text: compressedText, hadRollback };
  }

  private findPatternMatches(text: string, pattern: any, context: CompressionContext): PatternMatch[] {
    const matches: PatternMatch[] = [];
    const regex = new RegExp(this.escapeRegex(pattern.original), 'gi');
    let match;

    while ((match = regex.exec(text)) !== null) {
      if (this.shouldPreserveMatch(match[0], context)) {
        continue;
      }

      matches.push({
        pattern,
        position: match.index,
        originalText: match[0],
        compressedText: pattern.compressed,
        confidence: this.calculatePatternConfidence(pattern, context),
        context: this.extractContext(text, match.index, 40)
      });
    }

    return matches;
  }

  private validateMatch(match: PatternMatch, context: CompressionContext): boolean {
    // Hybrid validation - balance between conservative and aggressive
    const confidenceThreshold = context.requiresHighQuality ? 0.7 : 0.5;
    
    if (match.confidence < confidenceThreshold) {
      return false;
    }

    // Additional semantic checks for important contexts
    if (context.requiresHighQuality) {
      const originalWords = match.originalText.toLowerCase().split(/\s+/);
      const compressedWords = match.compressedText.toLowerCase().split(/\s+/);

      // Avoid extreme abbreviations in high-quality mode
      if (compressedWords.some(word => word.length <= 2 && originalWords.length > 2)) {
        return false;
      }
    }

    return true;
  }

  private shouldPreserveMatch(text: string, context: CompressionContext): boolean {
    if (context.preserveCode && this.looksLikeCode(text)) {
      return true;
    }

    if (context.preserveUrls && /^https?:\/\//.test(text)) {
      return true;
    }

    if (context.preserveNumbers && /^\d+(\.\d+)?$/.test(text)) {
      return true;
    }

    return false;
  }

  private calculatePatternConfidence(pattern: any, context: CompressionContext): number {
    let confidence = pattern.successRate || 0.5;

    // Boost for domain match
    if (pattern.domain === context.domain) {
      confidence += 0.2;
    }

    // Boost for frequency
    if (pattern.frequency > 5) {
      confidence += 0.1;
    }

    // Boost for priority
    if (pattern.priority > 700) {
      confidence += 0.1;
    }

    return Math.min(1.0, confidence);
  }

  private generateWarnings(compressionRatio: number, hadRollback: boolean): string[] {
    const warnings: string[] = [];

    if (hadRollback) {
      warnings.push('Compression was rolled back to maintain quality');
    }

    if (compressionRatio < 0.3) {
      warnings.push('Aggressive compression applied - review recommended');
    }

    return warnings;
  }

  private looksLikeCode(text: string): boolean {
    return /[{}();]/.test(text) || /\w+\.\w+/.test(text) || /^\w+\(/.test(text);
  }

  private extractContext(text: string, position: number, radius: number): string {
    const start = Math.max(0, position - radius);
    const end = Math.min(text.length, position + radius);
    return text.substring(start, end);
  }

  private escapeRegex(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  private updateStats(result: CompressionResult, hadRollback: boolean): void {
    this.stats.compressions++;
    this.stats.totalTime += result.processingTimeMs;
    this.stats.avgQuality = (this.stats.avgQuality * (this.stats.compressions - 1) + result.qualityScore) / this.stats.compressions;
    this.stats.avgCompressionRatio = (this.stats.avgCompressionRatio * (this.stats.compressions - 1) + result.compressionRatio) / this.stats.compressions;
    this.stats.patternMatches += result.patternMatches.length;
    
    if (hadRollback) {
      this.stats.rollbacks++;
    }
  }
}