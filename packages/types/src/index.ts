/**
 * Neural Semantic Compiler - TypeScript Type Definitions
 * 
 * Complete type definitions for the Neural Semantic Compiler ecosystem.
 */

// ===== CORE ENUMS =====

export enum CompressionLevel {
  NONE = 'none',
  LIGHT = 'light',
  BALANCED = 'balanced',
  AGGRESSIVE = 'aggressive'
}

export enum PatternType {
  PHRASE = 'phrase',
  COMPOUND = 'compound',
  WORD = 'word',
  ABBREVIATION = 'abbreviation',
  STRUCTURE = 'structure'
}

export enum EngineType {
  SEMANTIC = 'semantic',
  EXTREME = 'extreme',
  HYBRID = 'hybrid'
}

// ===== CORE INTERFACES =====

export interface Pattern {
  id?: number;
  original: string;
  compressed: string;
  patternType: PatternType;
  priority: number;
  domain: string;
  language: string;
  frequency: number;
  successRate: number;
  version: number;
  isActive: boolean;
  metadata?: Record<string, any>;
  createdAt?: number;
  updatedAt?: number;
}

export interface PatternMatch {
  pattern: Pattern;
  position: number;
  originalText: string;
  compressedText: string;
  confidence: number;
  context: string;
}

export interface CompressionContext {
  level: CompressionLevel;
  domain?: string;
  language: string;
  preserveCode: boolean;
  preserveUrls: boolean;
  preserveNumbers: boolean;
  targetCompression: number;
  requiresHighQuality: boolean;
  contextType: string;
  targetModel?: string;
  enginePreference?: EngineType;
}

export interface QualityMetrics {
  compositeScore: number;
  semanticPreservation: number;
  informationDensity: number;
  compressionEfficiency: number;
  llmInterpretability: number;
  structuralPreservation: number;
  linguisticCoherence: number;
  entityPreservation: number;
  breakdownDetails: Record<string, any>;
}

export interface CompressionResult {
  originalText: string;
  compressedText: string;
  originalTokens: number;
  compressedTokens: number;
  compressionRatio: number;
  qualityScore: number;
  patternMatches: PatternMatch[];
  processingTimeMs: number;
  engineUsed: string;
  warnings: string[];
  qualityMetrics?: QualityMetrics;
  sessionId?: string;
}

export interface SimilarPattern {
  original: string;
  compressed: string;
  similarity: number;
  patternType: string;
  domain: string;
  priority: number;
  confidence: number;
  semanticBreakdown?: Record<string, number>;
}

export interface NewPattern {
  original: string;
  compressed: string;
  frequency: number;
  estimatedSavings: number;
  confidence: number;
}

// ===== CONFIGURATION INTERFACES =====

export interface DatabaseConfig {
  path: string;
  connectionPoolSize: number;
  enableWalMode: boolean;
  cacheSizeMb: number;
}

export interface VectorConfig {
  modelName: string;
  persistDirectory: string;
  similarityThreshold: number;
  maxResults: number;
  enableGpu: boolean;
}

export interface CompressionConfig {
  defaultLevel: CompressionLevel;
  preserveCode: boolean;
  preserveUrls: boolean;
  preserveNumbers: boolean;
  minCompressionRatio: number;
  maxCompressionRatio: number;
  semanticThreshold: number;
  targetSemanticScore: number;
}

export interface LearningConfig {
  enableAutoDiscovery: boolean;
  minPatternFrequency: number;
  patternQualityThreshold: number;
  feedbackLearningRate: number;
}

export interface CompilerConfig {
  database: DatabaseConfig;
  vector: VectorConfig;
  compression: CompressionConfig;
  learning: LearningConfig;
  logLevel: string;
  logFile?: string;
  enableCaching: boolean;
  cacheTtlSeconds: number;
  maxCacheSize: number;
  activeDomains: string[];
  domainWeights: Record<string, number>;
}

// ===== ENGINE INTERFACES =====

export interface BaseEngine {
  compress(text: string, context: CompressionContext): Promise<CompressionResult>;
  getName(): string;
}

export interface EngineFactory {
  createEngine(type: EngineType): BaseEngine;
  getEngineForContext(context: CompressionContext): BaseEngine | null;
  getAvailableEngines(): EngineType[];
}

// ===== TOKENIZATION INTERFACES =====

export interface TokenizationResult {
  tokens: string[];
  tokenIds: number[];
  tokenCount: number;
  modelName: string;
  estimated: boolean;
}

export interface TokenizerManager {
  countTokens(text: string, model: string): Promise<TokenizationResult>;
  getOptimalCompressions(text: string, model: string): Promise<Record<string, any>>;
}

// ===== PATTERN MANAGEMENT INTERFACES =====

export interface PatternManager {
  getPatterns(domain?: string, patternType?: string): Promise<Pattern[]>;
  addPattern(pattern: Omit<Pattern, 'id'>): Promise<boolean>;
  updateUsageStats(patternId: number, success: boolean): Promise<void>;
  searchPatterns(query: string, limit?: number): Promise<Pattern[]>;
  getStatistics(): Promise<Record<string, any>>;
}

export interface ConflictResolver {
  resolveConflicts(matches: PatternMatch[], strategy?: string): PatternMatch[];
  detectPatternConflicts(patterns: Pattern[]): Array<Record<string, any>>;
}

// ===== VECTOR STORE INTERFACES =====

export interface VectorStore {
  enabled: boolean;
  addPattern(pattern: Pattern): Promise<void>;
  findSimilarPatterns(
    text: string, 
    nResults?: number, 
    threshold?: number, 
    domain?: string
  ): Promise<SimilarPattern[]>;
  bulkAddPatterns(patterns: Pattern[]): Promise<void>;
  getCollectionStats(): Promise<Record<string, any>>;
}

export interface EmbeddingManager {
  getEmbedding(text: string): Promise<number[]>;
  similarity(text1: string, text2: string): Promise<number>;
  batchEncode(texts: string[]): Promise<number[][]>;
  getCacheStats(): Record<string, any>;
}

// ===== ANALYTICS INTERFACES =====

export interface CompressionSession {
  sessionId: string;
  originalText: string;
  compressedText: string;
  originalTokens: number;
  compressedTokens: number;
  compressionRatio: number;
  qualityScore: number;
  engineUsed: string;
  processingTimeMs: number;
  contextType: string;
  createdAt: number;
}

export interface AnalyticsData {
  totalSessions: number;
  totalCostSavings: number;
  averageCompressionRatio: number;
  averageQualityScore: number;
  totalTokensSaved: number;
  sessionsToday: number;
  topDomains: Array<{ domain: string; count: number }>;
  enginePerformance: Record<string, Record<string, number>>;
}

// ===== MCPLOOP INTEGRATION INTERFACES =====

export interface MCPLoopIntegration {
  interceptPrompt(promptData: Record<string, any>): Promise<Record<string, any>>;
  getSessionAnalytics(sessionId?: string): AnalyticsData;
  generateSavingsReport(): string;
}

export interface MCPServer {
  handleRequest(request: Record<string, any>): Promise<Record<string, any>>;
  startServer(host?: string, port?: number): Promise<void>;
}

// ===== COMPILER INTERFACES =====

export interface NeuralSemanticCompiler {
  compress(text: string, options?: Partial<CompressionContext>): Promise<CompressionResult>;
  decompress(compressedText: string, options?: Record<string, any>): Promise<string>;
  addPattern(
    original: string, 
    compressed: string, 
    options?: Partial<Pattern>
  ): Promise<boolean>;
  getStatistics(): Promise<Record<string, any>>;
  benchmark(testTexts: string[], options?: Record<string, any>): Promise<Record<string, any>>;
  healthCheck(): Promise<Record<string, any>>;
  learnFromFeedback(sessionId: string, rating: number, feedback?: string): Promise<void>;
}

// ===== UTILITY INTERFACES =====

export interface TextCharacteristics {
  length: number;
  wordCount: number;
  technicalDensity: number;
  structuralComplexity: number;
  domainSpecificity: number;
  compressionPotential: number;
  qualitySensitivity: number;
}

export interface BenchmarkResult {
  engineName: string;
  averageCompressionRatio: number;
  averageProcessingTime: number;
  successRate: number;
  totalCompressions: number;
  totalErrors: number;
}

export interface HealthCheckResult {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  components: Record<string, {
    status: string;
    error?: string;
    [key: string]: any;
  }>;
  timestamp: number;
}

// ===== ERROR INTERFACES =====

export interface NSCError extends Error {
  code: string;
  context?: Record<string, any>;
}

export interface CompressionError extends NSCError {
  originalText?: string;
  partialResult?: Partial<CompressionResult>;
}

export interface PatternConflictError extends NSCError {
  conflictingPatterns: Pattern[];
  conflictType: string;
}

export interface QualityError extends NSCError {
  actualQuality: number;
  expectedQuality: number;
  result: CompressionResult;
}

// ===== CLI INTERFACES =====

export interface CLIOptions {
  level?: CompressionLevel;
  domain?: string;
  model?: string;
  showStats?: boolean;
  showTokens?: boolean;
  dryRun?: boolean;
  quiet?: boolean;
  verbose?: boolean;
}

export interface CLIResult {
  success: boolean;
  result?: CompressionResult;
  error?: string;
  warnings?: string[];
}

// ===== PLUGIN INTERFACES =====

export interface Plugin {
  name: string;
  version: string;
  initialize(compiler: NeuralSemanticCompiler): Promise<void>;
  cleanup(): Promise<void>;
}

export interface PluginManager {
  loadPlugin(plugin: Plugin): Promise<void>;
  unloadPlugin(pluginName: string): Promise<void>;
  getLoadedPlugins(): Plugin[];
}

// ===== EVENTS =====

export interface CompressionEvent {
  type: 'compression_started' | 'compression_completed' | 'compression_failed';
  sessionId: string;
  timestamp: Date;
  data: Record<string, any>;
}

export interface PatternEvent {
  type: 'pattern_added' | 'pattern_updated' | 'pattern_conflict';
  pattern: Pattern;
  timestamp: Date;
  data: Record<string, any>;
}

export type NSCEvent = CompressionEvent | PatternEvent;

export interface EventEmitter {
  on(event: string, listener: (event: NSCEvent) => void): void;
  emit(event: string, data: NSCEvent): void;
  off(event: string, listener: (event: NSCEvent) => void): void;
}

// ===== FACTORY TYPES =====

export type CompilerFactory = () => Promise<NeuralSemanticCompiler>;
export type ConfigFactory = () => CompilerConfig;
export type PatternFactory = (original: string, compressed: string) => Pattern;

// ===== UTILITY TYPES =====

export type Domain = 'general' | 'web-development' | 'agile' | 'devops' | string;
export type Language = 'en' | 'pt' | 'es' | 'fr' | 'de' | string;
export type ModelName = 'gpt-4' | 'gpt-3.5-turbo' | 'claude' | 'claude-instant' | string;

// ===== RESPONSE TYPES =====

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  timestamp: Date;
}

export interface CompressResponse extends APIResponse<CompressionResult> {}
export interface PatternsResponse extends APIResponse<Pattern[]> {}
export interface StatsResponse extends APIResponse<Record<string, any>> {}
export interface HealthResponse extends APIResponse<HealthCheckResult> {}

// ===== DEFAULT EXPORTS =====

export default {
  CompressionLevel,
  PatternType,
  EngineType
};

// ===== VERSION INFO =====

export const VERSION = '1.0.0';
export const BUILD_DATE = new Date().toISOString();
export const AUTHOR = 'Brenner Cruvinel (@brennercruvinel)';
export const HOMEPAGE = 'https://neurosemantic.tech';