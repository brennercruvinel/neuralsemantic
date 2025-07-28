/**
 * Extreme Engine - Maximum compression with aggressive techniques
 */

import {
  CompressionResult,
  CompressionContext,
  CompressionLevel,
  PatternMatch,
  EngineType
} from '@neurosemantic/types';

import { BaseEngine } from './base-engine';

interface ExtremeEngineDependencies {
  patternManager: any;
  vectorStore: any;
  qualityScorer: any;
  tokenizerManager: any;
  config: any;
}

export class ExtremeEngine extends BaseEngine {
  private dependencies: ExtremeEngineDependencies;
  private stats = {
    compressions: 0,
    totalTime: 0,
    avgCompressionRatio: 0,
    patternMatches: 0
  };

  constructor(dependencies: ExtremeEngineDependencies) {
    super('ExtremeEngine');
    this.dependencies = dependencies;
  }

  async compress(text: string, context: CompressionContext): Promise<CompressionResult> {
    const startTime = Date.now();
    
    try {
      let compressedText = text;
      const patternMatches: PatternMatch[] = [];

      // Step 1: Apply all available patterns aggressively
      const patterns = await this.dependencies.patternManager.getPatterns({
        domain: context.domain,
        language: context.language
      });

      // Sort patterns by compression potential (original length - compressed length)
      patterns.sort((a: any, b: any) => 
        (b.original.length - b.compressed.length) - (a.original.length - a.compressed.length)
      );

      for (const pattern of patterns) {
        const matches = this.findAllMatches(compressedText, pattern, context);
        for (const match of matches) {
          if (this.shouldApplyPattern(match, context)) {
            patternMatches.push(match);
            compressedText = compressedText.replace(match.originalText, match.compressedText);
          }
        }
      }

      // Step 2: Apply structural compression
      compressedText = this.applyStructuralCompression(compressedText, context);

      // Step 3: Apply extreme abbreviations
      compressedText = this.applyExtremeAbbreviations(compressedText, context);

      // Step 4: Apply symbol replacements
      compressedText = this.applySymbolReplacements(compressedText, context);

      // Step 5: Remove redundant spaces and punctuation
      compressedText = this.cleanupWhitespace(compressedText);

      // Step 6: Calculate tokens
      const [originalTokens, compressedTokens] = await Promise.all([
        this.dependencies.tokenizerManager.countTokens(text, context.targetModel || 'gpt-4'),
        this.dependencies.tokenizerManager.countTokens(compressedText, context.targetModel || 'gpt-4')
      ]);

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
        warnings: this.generateWarnings(compressionRatio)
      };

      this.updateStats(result);
      return result;

    } catch (error) {
      throw new Error(`Extreme engine compression failed: ${error}`);
    }
  }

  canHandle(context: CompressionContext): boolean {
    return context.level === CompressionLevel.AGGRESSIVE && !context.requiresHighQuality;
  }

  getMetadata(): Record<string, any> {
    return {
      type: EngineType.EXTREME,
      description: 'Maximum compression using aggressive techniques',
      strengths: ['High compression ratios', 'Fast processing', 'Token optimization'],
      limitations: ['Potential quality loss', 'Less readable', 'Context dependent'],
      bestFor: ['Internal notes', 'Data transmission', 'Quick summaries']
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

  private findAllMatches(text: string, pattern: any, context: CompressionContext): PatternMatch[] {
    const matches: PatternMatch[] = [];
    const regex = new RegExp(this.escapeRegex(pattern.original), 'gi');
    let match;

    while ((match = regex.exec(text)) !== null) {
      matches.push({
        pattern,
        position: match.index,
        originalText: match[0],
        compressedText: pattern.compressed,
        confidence: 1.0, // Extreme engine applies all patterns
        context: this.extractContext(text, match.index, 30)
      });
    }

    return matches;
  }

  private shouldApplyPattern(match: PatternMatch, context: CompressionContext): boolean {
    // In extreme mode, apply most patterns except for critical preservations
    if (context.preserveCode && this.looksLikeCode(match.originalText)) {
      return false;
    }

    if (context.preserveUrls && /^https?:\/\//.test(match.originalText)) {
      return false;
    }

    // Allow number compression in extreme mode unless explicitly preserved
    if (context.preserveNumbers && /^\d+(\.\d+)?$/.test(match.originalText)) {
      return false;
    }

    return true;
  }

  private applyStructuralCompression(text: string, context: CompressionContext): string {
    let compressed = text;

    // Remove redundant articles in extreme mode
    compressed = compressed.replace(/\b(the|a|an)\s+/gi, '');

    // Compress common phrases
    compressed = compressed.replace(/\bin order to\b/gi, 'to');
    compressed = compressed.replace(/\bas well as\b/gi, '&');
    compressed = compressed.replace(/\bdue to the fact that\b/gi, 'because');
    compressed = compressed.replace(/\bin the event that\b/gi, 'if');

    // Compress transitional phrases
    compressed = compressed.replace(/\bhowever,\s*/gi, 'but ');
    compressed = compressed.replace(/\btherefore,\s*/gi, 'so ');
    compressed = compressed.replace(/\bnevertheless,\s*/gi, 'yet ');

    return compressed;
  }

  private applyExtremeAbbreviations(text: string, context: CompressionContext): string {
    let compressed = text;

    // Extreme abbreviations map
    const abbreviations = {
      'information': 'info',
      'development': 'dev',
      'environment': 'env',
      'configuration': 'config',
      'application': 'app',
      'database': 'db',
      'repository': 'repo',
      'documentation': 'docs',
      'specification': 'spec',
      'implementation': 'impl',
      'requirements': 'reqs',
      'performance': 'perf',
      'management': 'mgmt',
      'administration': 'admin',
      'authentication': 'auth',
      'authorization': 'authz',
      'communication': 'comm',
      'infrastructure': 'infra',
      'architecture': 'arch'
    };

    for (const [full, abbr] of Object.entries(abbreviations)) {
      const regex = new RegExp(`\\b${full}\\b`, 'gi');
      compressed = compressed.replace(regex, abbr);
    }

    return compressed;
  }

  private applySymbolReplacements(text: string, context: CompressionContext): string {
    let compressed = text;

    // Symbol replacements for extreme compression
    compressed = compressed.replace(/\band\b/gi, '&');
    compressed = compressed.replace(/\bwith\b/gi, 'w/');
    compressed = compressed.replace(/\bwithout\b/gi, 'w/o');
    compressed = compressed.replace(/\bbefore\b/gi, 'b4');
    compressed = compressed.replace(/\bafter\b/gi, 'after'); // Keep as is
    compressed = compressed.replace(/\bnumber\b/gi, '#');
    compressed = compressed.replace(/\bat\b/gi, '@');

    return compressed;
  }

  private cleanupWhitespace(text: string): string {
    return text
      .replace(/\s+/g, ' ') // Multiple spaces to single space
      .replace(/\s*([.!?])\s*/g, '$1 ') // Clean punctuation spacing
      .replace(/\s*,\s*/g, ',') // Remove spaces around commas
      .trim();
  }

  private generateWarnings(compressionRatio: number): string[] {
    const warnings: string[] = [];

    if (compressionRatio < 0.3) {
      warnings.push('Extreme compression applied - readability may be significantly reduced');
    }

    if (compressionRatio < 0.2) {
      warnings.push('Very aggressive compression - manual review recommended');
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

  private updateStats(result: CompressionResult): void {
    this.stats.compressions++;
    this.stats.totalTime += result.processingTimeMs;
    this.stats.avgCompressionRatio = (this.stats.avgCompressionRatio * (this.stats.compressions - 1) + result.compressionRatio) / this.stats.compressions;
    this.stats.patternMatches += result.patternMatches.length;
  }
}