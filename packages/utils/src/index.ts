/**
 * Neural Semantic Compiler - Utility Functions
 * 
 * Provides utility functions for token counting, text analysis, and compression helpers.
 */

import { TokenCounter, SimilarityCalculator } from '@neurosemantic/types';

// Token estimation utilities
export class SimpleTokenCounter implements TokenCounter {
  private static readonly CHARS_PER_TOKEN = 4;
  private static readonly ENCODING_PATTERNS = {
    // Common token patterns for better estimation
    WHITESPACE: /\s+/g,
    PUNCTUATION: /[^\w\s]/g,
    NUMBERS: /\d+/g,
    CAMELCASE: /[a-z][A-Z]/g,
    UPPERCASE_WORDS: /\b[A-Z]{2,}\b/g
  };

  count(text: string): number {
    if (!text) return 0;

    // Base estimation: ~4 characters per token
    let tokenCount = Math.ceil(text.length / SimpleTokenCounter.CHARS_PER_TOKEN);

    // Adjust for common patterns
    const whitespaceMatches = (text.match(SimpleTokenCounter.ENCODING_PATTERNS.WHITESPACE) || []).length;
    const punctuationMatches = (text.match(SimpleTokenCounter.ENCODING_PATTERNS.PUNCTUATION) || []).length;
    const numberMatches = (text.match(SimpleTokenCounter.ENCODING_PATTERNS.NUMBERS) || []).length;
    const camelCaseMatches = (text.match(SimpleTokenCounter.ENCODING_PATTERNS.CAMELCASE) || []).length;
    const uppercaseMatches = (text.match(SimpleTokenCounter.ENCODING_PATTERNS.UPPERCASE_WORDS) || []).length;

    // Refined estimation based on patterns
    const words = text.split(/\s+/).length;
    const adjustedTokenCount = Math.max(1, Math.round(
      words * 1.3 + // Base word multiplier
      punctuationMatches * 0.5 + // Punctuation adds tokens
      numberMatches * 0.3 + // Numbers are often single tokens
      camelCaseMatches * 0.2 + // CamelCase splits
      uppercaseMatches * 0.1 // Acronyms
    ));

    return Math.min(tokenCount, adjustedTokenCount);
  }

  encode(text: string): number[] {
    // Simple encoding simulation - not a real tokenizer
    const words = text.split(/\s+/);
    return words.map((word, index) => {
      // Create pseudo-token IDs based on word characteristics
      let tokenId = 0;
      for (let i = 0; i < word.length; i++) {
        tokenId += word.charCodeAt(i);
      }
      return (tokenId + index) % 50000; // Simulate vocabulary size
    });
  }

  decode(tokens: number[]): string {
    // Simple decoding simulation
    return tokens.map(token => `token_${token}`).join(' ');
  }
}

// Text analysis utilities
export class TextAnalyzer {
  /**
   * Calculate reading time estimate
   */
  static getReadingTime(text: string, wordsPerMinute: number = 200): number {
    const wordCount = text.trim().split(/\s+/).length;
    return Math.ceil(wordCount / wordsPerMinute);
  }

  /**
   * Estimate text complexity
   */
  static getComplexityScore(text: string): number {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const words = text.trim().split(/\s+/);
    
    if (sentences.length === 0 || words.length === 0) return 0;

    const avgWordsPerSentence = words.length / sentences.length;
    const avgCharsPerWord = words.reduce((sum, word) => sum + word.length, 0) / words.length;
    const uniqueWords = new Set(words.map(w => w.toLowerCase())).size;
    const lexicalDiversity = uniqueWords / words.length;

    // Complexity score combining multiple factors
    const complexityScore = (
      Math.min(avgWordsPerSentence / 20, 1) * 0.4 + // Sentence length
      Math.min(avgCharsPerWord / 8, 1) * 0.3 + // Word length
      lexicalDiversity * 0.3 // Vocabulary diversity
    );

    return Math.min(1, complexityScore);
  }

  /**
   * Detect domain/topic
   */
  static detectDomain(text: string): string {
    const lowerText = text.toLowerCase();
    
    const domainKeywords = {
      'web-development': ['react', 'javascript', 'html', 'css', 'api', 'frontend', 'backend', 'node', 'npm'],
      'agile': ['sprint', 'scrum', 'backlog', 'story', 'epic', 'kanban', 'retrospective'],
      'devops': ['docker', 'kubernetes', 'ci/cd', 'deployment', 'infrastructure', 'cloud'],
      'data-science': ['analysis', 'dataset', 'model', 'machine learning', 'python', 'pandas'],
      'business': ['strategy', 'revenue', 'customer', 'market', 'growth', 'profit'],
      'technical': ['algorithm', 'implementation', 'architecture', 'design', 'system']
    };

    let maxScore = 0;
    let detectedDomain = 'general';

    Object.entries(domainKeywords).forEach(([domain, keywords]) => {
      const score = keywords.reduce((count, keyword) => {
        return count + (lowerText.includes(keyword) ? 1 : 0);
      }, 0);

      if (score > maxScore) {
        maxScore = score;
        detectedDomain = domain;
      }
    });

    return maxScore > 0 ? detectedDomain : 'general';
  }

  /**
   * Extract key phrases
   */
  static extractKeyPhrases(text: string, maxPhrases: number = 10): string[] {
    const words = text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 3);

    // Remove common stop words
    const stopWords = new Set([
      'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
      'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
      'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
      'can', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
      'below', 'between', 'among', 'through', 'during', 'before', 'after', 'above', 'below'
    ]);

    const filteredWords = words.filter(word => !stopWords.has(word));

    // Count word frequency
    const wordFreq: Record<string, number> = {};
    filteredWords.forEach(word => {
      wordFreq[word] = (wordFreq[word] || 0) + 1;
    });

    // Extract 2-word phrases
    const phrases: Record<string, number> = {};
    for (let i = 0; i < filteredWords.length - 1; i++) {
      const phrase = `${filteredWords[i]} ${filteredWords[i + 1]}`;
      phrases[phrase] = (phrases[phrase] || 0) + 1;
    }

    // Combine single words and phrases, sort by frequency
    const allTerms = { ...wordFreq, ...phrases };
    const sortedTerms = Object.entries(allTerms)
      .sort(([, a], [, b]) => b - a)
      .slice(0, maxPhrases)
      .map(([term]) => term);

    return sortedTerms;
  }
}

// Similarity calculation utilities
export class SimpleSimilarityCalculator implements SimilarityCalculator {
  calculate(text1: string, text2: string): number {
    return this.jaccardSimilarity(text1, text2);
  }

  batchCalculate(query: string, candidates: string[]): number[] {
    return candidates.map(candidate => this.calculate(query, candidate));
  }

  private jaccardSimilarity(text1: string, text2: string): number {
    const words1 = new Set(this.tokenize(text1));
    const words2 = new Set(this.tokenize(text2));

    const intersection = new Set([...words1].filter(word => words2.has(word)));
    const union = new Set([...words1, ...words2]);

    return union.size === 0 ? 0 : intersection.size / union.size;
  }

  private tokenize(text: string): string[] {
    return text.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);
  }

  /**
   * Calculate cosine similarity between two texts
   */
  cosineSimilarity(text1: string, text2: string): number {
    const words1 = this.tokenize(text1);
    const words2 = this.tokenize(text2);

    // Build vocabulary
    const vocabulary = Array.from(new Set([...words1, ...words2]));
    
    if (vocabulary.length === 0) return 0;

    // Create frequency vectors
    const vector1 = vocabulary.map(word => words1.filter(w => w === word).length);
    const vector2 = vocabulary.map(word => words2.filter(w => w === word).length);

    // Calculate cosine similarity
    const dotProduct = vector1.reduce((sum, a, i) => sum + a * vector2[i], 0);
    const magnitude1 = Math.sqrt(vector1.reduce((sum, a) => sum + a * a, 0));
    const magnitude2 = Math.sqrt(vector2.reduce((sum, a) => sum + a * a, 0));

    return magnitude1 === 0 || magnitude2 === 0 ? 0 : dotProduct / (magnitude1 * magnitude2);
  }

  /**
   * Calculate edit distance (Levenshtein distance)
   */
  editDistance(text1: string, text2: string): number {
    const m = text1.length;
    const n = text2.length;

    // Create DP table
    const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

    // Initialize base cases
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;

    // Fill DP table
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (text1[i - 1] === text2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = 1 + Math.min(
            dp[i - 1][j],     // deletion
            dp[i][j - 1],     // insertion
            dp[i - 1][j - 1]  // substitution
          );
        }
      }
    }

    return dp[m][n];
  }
}

// Compression utilities
export class CompressionUtils {
  /**
   * Calculate estimated cost savings
   */
  static calculateCostSavings(originalTokens: number, compressedTokens: number, costPerToken: number): number {
    const tokenSavings = originalTokens - compressedTokens;
    return tokenSavings * costPerToken;
  }

  /**
   * Estimate processing time based on text length
   */
  static estimateProcessingTime(textLength: number): number {
    // Base processing time + length factor
    const baseTime = 100; // 100ms base
    const lengthFactor = Math.ceil(textLength / 1000) * 50; // 50ms per 1000 chars
    return baseTime + lengthFactor;
  }

  /**
   * Validate compression quality
   */
  static validateQuality(original: string, compressed: string): {
    isValid: boolean;
    issues: string[];
    score: number;
  } {
    const issues: string[] = [];
    let score = 10;

    // Check for over-compression
    const compressionRatio = compressed.length / original.length;
    if (compressionRatio < 0.3) {
      issues.push('Compression ratio too aggressive (< 30%)');
      score -= 3;
    }

    // Check for key information loss
    const originalWords = new Set(original.toLowerCase().split(/\s+/));
    const compressedWords = new Set(compressed.toLowerCase().split(/\s+/));
    const preservation = compressedWords.size / originalWords.size;
    
    if (preservation < 0.5) {
      issues.push('Significant word loss detected');
      score -= 2;
    }

    // Check for readability
    if (compressed.length > 0 && compressed.split(/\s+/).length < 3) {
      issues.push('Result too short for meaningful content');
      score -= 1;
    }

    // Check for broken syntax (basic)
    const unclosedParens = (compressed.split('(').length - 1) - (compressed.split(')').length - 1);
    const unclosedBrackets = (compressed.split('[').length - 1) - (compressed.split(']').length - 1);
    
    if (Math.abs(unclosedParens) > 0 || Math.abs(unclosedBrackets) > 0) {
      issues.push('Potential syntax issues detected');
      score -= 1;
    }

    return {
      isValid: issues.length === 0,
      issues,
      score: Math.max(0, score)
    };
  }

  /**
   * Generate compression report
   */
  static generateReport(original: string, compressed: string, processingTime: number): string {
    const originalLength = original.length;
    const compressedLength = compressed.length;
    const compressionRatio = compressedLength / originalLength;
    const savings = Math.round((1 - compressionRatio) * 100);
    
    const tokenCounter = new SimpleTokenCounter();
    const originalTokens = tokenCounter.count(original);
    const compressedTokens = tokenCounter.count(compressed);
    const tokenSavings = originalTokens - compressedTokens;

    const analyzer = new TextAnalyzer();
    const readingTime = TextAnalyzer.getReadingTime(original);
    const complexity = TextAnalyzer.getComplexityScore(original);
    const domain = TextAnalyzer.detectDomain(original);

    return `
 Neural Semantic Compression Report

 Compression Statistics:
  Original length: ${originalLength.toLocaleString()} characters
  Compressed length: ${compressedLength.toLocaleString()} characters
  Reduction: ${savings}% (${(originalLength - compressedLength).toLocaleString()} chars saved)
  
 Token Analysis:
  Original tokens: ${originalTokens.toLocaleString()}
  Compressed tokens: ${compressedTokens.toLocaleString()}
  Token savings: ${tokenSavings.toLocaleString()}
  
 Content Analysis:
  Reading time: ~${readingTime} minutes
  Complexity score: ${(complexity * 100).toFixed(1)}%
  Detected domain: ${domain}
  
 Performance:
  Processing time: ${processingTime}ms
  Rate: ${(originalLength / processingTime * 1000).toFixed(0)} chars/second
`.trim();
  }
}

// Format utilities
export class FormatUtils {
  /**
   * Format bytes with units
   */
  static formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  }

  /**
   * Format duration in milliseconds
   */
  static formatDuration(ms: number): string {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`;
    return `${(ms / 3600000).toFixed(1)}h`;
  }

  /**
   * Format percentage
   */
  static formatPercentage(value: number, decimals: number = 1): string {
    return `${(value * 100).toFixed(decimals)}%`;
  }

  /**
   * Format large numbers
   */
  static formatNumber(num: number): string {
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return num.toString();
  }
}

// Validation utilities
export class ValidationUtils {
  /**
   * Validate text input
   */
  static validateText(text: string): { isValid: boolean; error?: string } {
    if (typeof text !== 'string') {
      return { isValid: false, error: 'Input must be a string' };
    }
    
    if (text.trim().length === 0) {
      return { isValid: false, error: 'Text cannot be empty' };
    }
    
    if (text.length > 1000000) { // 1MB limit
      return { isValid: false, error: 'Text too large (max 1MB)' };
    }
    
    return { isValid: true };
  }

  /**
   * Validate compression options
   */
  static validateCompressionOptions(options: any): { isValid: boolean; error?: string } {
    if (options.level && !['light', 'balanced', 'aggressive'].includes(options.level)) {
      return { isValid: false, error: 'Invalid compression level' };
    }
    
    if (options.targetCompression && (options.targetCompression < 0 || options.targetCompression > 1)) {
      return { isValid: false, error: 'Target compression must be between 0 and 1' };
    }
    
    return { isValid: true };
  }
}

// Export all utilities
export {
  SimpleTokenCounter as TokenCounter,
  SimpleSimilarityCalculator as SimilarityCalculator
};

// Default export
export default {
  TextAnalyzer,
  CompressionUtils,
  FormatUtils,
  ValidationUtils,
  SimpleTokenCounter,
  SimpleSimilarityCalculator
};