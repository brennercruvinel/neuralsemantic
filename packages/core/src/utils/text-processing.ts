/**
 * Text Processing Utilities for Neural Semantic Compiler
 */

import {
  TokenizationResult,
  TokenizerManager as ITokenizerManager,
  TextCharacteristics
} from '@neurosemantic/types';

export class TokenizerManager implements ITokenizerManager {
  private tokenizers: Map<string, any> = new Map();
  private cache: Map<string, TokenizationResult> = new Map();
  private maxCacheSize = 1000;

  constructor() {
    this.initializeTokenizers();
  }

  async countTokens(text: string, model: string): Promise<TokenizationResult> {
    const cacheKey = `${model}:${this.hashText(text)}`;
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    try {
      let result: TokenizationResult;

      if (this.tokenizers.has(model)) {
        // Use specific tokenizer
        result = await this.useSpecificTokenizer(text, model);
      } else {
        // Use estimation
        result = this.estimateTokens(text, model);
      }

      // Cache result
      this.addToCache(cacheKey, result);
      
      return result;
    } catch (error) {
      // Fallback to estimation
      return this.estimateTokens(text, model);
    }
  }

  async getOptimalCompressions(text: string, model: string): Promise<Record<string, any>> {
    const baseTokens = await this.countTokens(text, model);
    
    return {
      original_tokens: baseTokens.tokenCount,
      model: model,
      text_characteristics: this.analyzeText(text),
      compression_potential: this.estimateCompressionPotential(text),
      recommendations: this.getCompressionRecommendations(text, model)
    };
  }

  private async initializeTokenizers(): Promise<void> {
    try {
      // Try to initialize tiktoken for OpenAI models
      const { get_encoding } = await import('tiktoken');
      
      this.tokenizers.set('gpt-4', get_encoding('cl100k_base'));
      this.tokenizers.set('gpt-3.5-turbo', get_encoding('cl100k_base'));
      this.tokenizers.set('text-davinci-003', get_encoding('p50k_base'));
      
    } catch (error) {
      // tiktoken not available - will use estimation
      console.warn('tiktoken not available, using token estimation');
    }
  }

  private async useSpecificTokenizer(text: string, model: string): Promise<TokenizationResult> {
    const tokenizer = this.tokenizers.get(model);
    
    if (!tokenizer) {
      throw new Error(`Tokenizer not found for model: ${model}`);
    }

    const tokens = tokenizer.encode(text);
    const tokenStrings = tokens.map((id: number) => tokenizer.decode([id]));

    return {
      tokens: tokenStrings,
      tokenIds: tokens,
      tokenCount: tokens.length,
      modelName: model,
      estimated: false
    };
  }

  private estimateTokens(text: string, model: string): TokenizationResult {
    // Estimation based on model characteristics
    let tokensPerChar: number;
    
    switch (model) {
      case 'gpt-4':
      case 'gpt-3.5-turbo':
        tokensPerChar = 0.25; // ~4 chars per token
        break;
      case 'claude':
      case 'claude-instant':
        tokensPerChar = 0.24; // Similar to GPT
        break;
      default:
        tokensPerChar = 0.25; // Default estimation
    }

    const estimatedCount = Math.ceil(text.length * tokensPerChar);
    const words = text.split(/\s+/).filter(word => word.length > 0);
    
    return {
      tokens: words, // Approximate
      tokenIds: [], // Not available in estimation
      tokenCount: estimatedCount,
      modelName: model,
      estimated: true
    };
  }

  private analyzeText(text: string): TextCharacteristics {
    const length = text.length;
    const words = text.split(/\s+/).filter(word => word.length > 0);
    const wordCount = words.length;
    
    // Technical density (presence of technical terms)
    const technicalTerms = /\b(API|HTTP|JSON|XML|CSS|HTML|JavaScript|Python|SQL|REST|GraphQL|OAuth|JWT|CRUD|MVC|SPA|PWA|CI\/CD|DevOps|Kubernetes|Docker|AWS|Azure|GCP)\b/gi;
    const technicalMatches = (text.match(technicalTerms) || []).length;
    const technicalDensity = Math.min(1.0, technicalMatches / wordCount);
    
    // Structural complexity (nested structures, code blocks)
    const structuralIndicators = /[{}()\[\]<>]/g;
    const structuralMatches = (text.match(structuralIndicators) || []).length;
    const structuralComplexity = Math.min(1.0, structuralMatches / length);
    
    // Domain specificity
    const domainTerms = /\b(agile|scrum|sprint|kanban|standup|retrospective|velocity|backlog|epic|story|task|bug|feature|deployment|release|staging|production)\b/gi;
    const domainMatches = (text.match(domainTerms) || []).length;
    const domainSpecificity = Math.min(1.0, domainMatches / wordCount);
    
    // Compression potential
    const repetitivePatterns = this.detectRepetitivePatterns(text);
    const compressionPotential = Math.min(1.0, repetitivePatterns / wordCount);
    
    // Quality sensitivity (formal language, specific terminology)
    const formalIndicators = /\b(shall|must|should|requirement|specification|documentation|implementation|architecture|infrastructure)\b/gi;
    const formalMatches = (text.match(formalIndicators) || []).length;
    const qualitySensitivity = Math.min(1.0, formalMatches / wordCount);

    return {
      length,
      wordCount,
      technicalDensity,
      structuralComplexity,
      domainSpecificity,
      compressionPotential,
      qualitySensitivity
    };
  }

  private estimateCompressionPotential(text: string): number {
    const characteristics = this.analyzeText(text);
    
    // Base potential from text characteristics
    let potential = 0.3; // Base 30% compression potential
    
    // Boost for repetitive content
    potential += characteristics.compressionPotential * 0.4;
    
    // Boost for technical content (has abbreviations)
    potential += characteristics.technicalDensity * 0.2;
    
    // Boost for domain-specific content
    potential += characteristics.domainSpecificity * 0.15;
    
    // Reduce for high quality sensitivity
    potential -= characteristics.qualitySensitivity * 0.1;
    
    return Math.min(0.8, Math.max(0.1, potential));
  }

  private getCompressionRecommendations(text: string, model: string): string[] {
    const characteristics = this.analyzeText(text);
    const recommendations: string[] = [];
    
    if (characteristics.technicalDensity > 0.3) {
      recommendations.push('High technical content detected - use semantic engine for better preservation');
    }
    
    if (characteristics.structuralComplexity > 0.2) {
      recommendations.push('Complex structure detected - enable code preservation');
    }
    
    if (characteristics.qualitySensitivity > 0.4) {
      recommendations.push('Formal content detected - use conservative compression level');
    }
    
    if (characteristics.compressionPotential > 0.6) {
      recommendations.push('High compression potential - consider aggressive level');
    }
    
    if (characteristics.domainSpecificity > 0.4) {
      recommendations.push('Domain-specific content - specify domain for better patterns');
    }
    
    return recommendations;
  }

  private detectRepetitivePatterns(text: string): number {
    const words = text.toLowerCase().split(/\s+/);
    const wordCounts = new Map<string, number>();
    
    for (const word of words) {
      if (word.length > 3) { // Only count meaningful words
        wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
      }
    }
    
    let repetitiveCount = 0;
    for (const [word, count] of wordCounts) {
      if (count > 1) {
        repetitiveCount += count - 1; // Extra occurrences
      }
    }
    
    return repetitiveCount;
  }

  private hashText(text: string): string {
    // Simple hash for caching
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
  }

  private addToCache(key: string, result: TokenizationResult): void {
    if (this.cache.size >= this.maxCacheSize) {
      // Remove oldest entry
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    
    this.cache.set(key, result);
  }
}

export class TextProcessor {
  /**
   * Clean and normalize text for processing
   */
  static normalizeText(text: string): string {
    return text
      .replace(/\r\n/g, '\n') // Normalize line endings
      .replace(/\t/g, '  ') // Convert tabs to spaces
      .replace(/\u00A0/g, ' ') // Replace non-breaking spaces
      .trim();
  }

  /**
   * Extract code blocks from text
   */
  static extractCodeBlocks(text: string): Array<{ content: string; language?: string; position: number }> {
    const codeBlocks: Array<{ content: string; language?: string; position: number }> = [];
    
    // Match fenced code blocks
    const fencedRegex = /```(\w+)?\n([\s\S]*?)```/g;
    let match;
    
    while ((match = fencedRegex.exec(text)) !== null) {
      codeBlocks.push({
        content: match[2],
        language: match[1],
        position: match.index
      });
    }
    
    // Match inline code
    const inlineRegex = /`([^`]+)`/g;
    while ((match = inlineRegex.exec(text)) !== null) {
      codeBlocks.push({
        content: match[1],
        position: match.index
      });
    }
    
    return codeBlocks;
  }

  /**
   * Extract URLs from text
   */
  static extractUrls(text: string): Array<{ url: string; position: number }> {
    const urls: Array<{ url: string; position: number }> = [];
    const urlRegex = /https?:\/\/[^\s<>"{}|\\^`[\]]+/g;
    let match;
    
    while ((match = urlRegex.exec(text)) !== null) {
      urls.push({
        url: match[0],
        position: match.index
      });
    }
    
    return urls;
  }

  /**
   * Split text into sentences
   */
  static splitSentences(text: string): string[] {
    // Simple sentence splitting - could be enhanced with ML
    return text
      .split(/[.!?]+/)
      .map(sentence => sentence.trim())
      .filter(sentence => sentence.length > 0);
  }

  /**
   * Count words in text
   */
  static countWords(text: string): number {
    return text.split(/\s+/).filter(word => word.length > 0).length;
  }

  /**
   * Estimate reading time in minutes
   */
  static estimateReadingTime(text: string, wordsPerMinute: number = 200): number {
    const wordCount = this.countWords(text);
    return Math.ceil(wordCount / wordsPerMinute);
  }
}