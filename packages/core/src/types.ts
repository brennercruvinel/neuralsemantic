/**
 * Core types for Neural Semantic Compiler TypeScript implementation
 */

// Enums
export enum CompressionLevel {
  LIGHT = 'light',
  BALANCED = 'balanced',
  AGGRESSIVE = 'aggressive'
}

export enum PatternType {
  WORD = 'word',
  PHRASE = 'phrase',
  CONCEPT = 'concept',
  TECH_TERM = 'tech_term',
  ABBREVIATION = 'abbreviation',
  CUSTOM = 'custom'
}

export enum QualityDimension {
  SEMANTIC_ACCURACY = 'semantic_accuracy',
  CONTEXT_PRESERVATION = 'context_preservation',
  TECHNICAL_CORRECTNESS = 'technical_correctness',
  READABILITY = 'readability',
  COMPRESSION_EFFICIENCY = 'compression_efficiency',
  DOMAIN_APPROPRIATENESS = 'domain_appropriateness',
  REVERSIBILITY = 'reversibility',
  COHERENCE = 'coherence'
}

// Core interfaces
export interface Pattern {
  id?: string;
  original: string;
  compressed: string;
  pattern_type: PatternType;
  domain: string;
  priority: number;
  language: string;
  frequency: number;
  success_rate: number;
  created_at?: Date;
  updated_at?: Date;
  metadata?: Record<string, any>;
}

export interface PatternMatch {
  pattern: Pattern;
  position: number;
  length: number;
  confidence: number;
  context: string;
}

export interface CompressionContext {
  level: CompressionLevel;
  domain?: string;
  language: string;
  preserve_code: boolean;
  preserve_urls: boolean;
  preserve_numbers: boolean;
  target_compression: number;
  requires_high_quality: boolean;
  context_type: string;
  custom_patterns?: Pattern[];
  excluded_patterns?: string[];
}

export interface QualityScore {
  overall: number;
  dimensions: Record<QualityDimension, number>;
  confidence: number;
  details: string;
}

export interface CompressionResult {
  original_text: string;
  compressed_text: string;
  compression_ratio: number;
  quality_score: number;
  quality_details?: QualityScore;
  original_tokens: number;
  compressed_tokens: number;
  token_savings: number;
  savings_percentage: number;
  processing_time_ms: number;
  engine_used: string;
  pattern_matches: PatternMatch[];
  warnings: string[];
  session_id?: string;
  context: CompressionContext;
  metadata?: Record<string, any>;
}

export interface CompilerConfig {
  database: {
    path: string;
    enable_wal: boolean;
    cache_size: number;
  };
  compression: {
    default_level: CompressionLevel;
    preserve_code: boolean;
    preserve_urls: boolean;
    preserve_numbers: boolean;
    target_compression: number;
    min_quality_threshold: number;
  };
  vector?: {
    enabled: boolean;
    collection_name: string;
    embedding_model: string;
    similarity_threshold: number;
    max_results: number;
    persist_directory?: string;
  };
  patterns: {
    auto_learn: boolean;
    min_frequency: number;
    max_patterns: number;
    enable_custom: boolean;
  };
  engines: {
    default_engine: string;
    semantic_enabled: boolean;
    hybrid_enabled: boolean;
    extreme_enabled: boolean;
  };
  log_level: string;
  log_file?: string;
}

export interface EngineInfo {
  name: string;
  type: string;
  description: string;
  capabilities: string[];
  optimal_use_cases: string[];
  compression_range: [number, number];
  quality_range: [number, number];
}

export interface CompressionEngine {
  name: string;
  compress(text: string, context: CompressionContext): Promise<CompressionResult>;
  getEngineInfo(): EngineInfo;
}

export interface PatternManager {
  addPattern(pattern: Pattern): Promise<boolean>;
  getPatterns(domain?: string, type?: PatternType, limit?: number): Promise<Pattern[]>;
  searchPatterns(query: string, limit?: number): Promise<Pattern[]>;
  updateUsageStats(patternId: string, successful: boolean): Promise<void>;
  getStatistics(): Promise<Record<string, any>>;
}

export interface VectorStore {
  enabled: boolean;
  addPattern(pattern: Pattern): Promise<void>;
  findSimilarPatterns(text: string, limit?: number): Promise<Pattern[]>;
  getCollectionStats(): Promise<Record<string, any>>;
}

// Utility types
export type CompressionOptions = Partial<{
  level: CompressionLevel | string;
  domain: string;
  engine: string;
  preserve_code: boolean;
  preserve_urls: boolean;
  preserve_numbers: boolean;
  requires_high_quality: boolean;
  target_compression: number;
  language: string;
  context_type: string;
}>;

export type EngineComparison = Record<string, {
  compressed_text: string;
  compression_ratio: number;
  quality_score: number;
  token_savings: number;
  processing_time_ms: number;
  pattern_matches: number;
  warnings: string[];
  error?: string;
}>;

export type SystemHealth = {
  overall_status: 'healthy' | 'warning' | 'error';
  components: Record<string, {
    status: 'healthy' | 'warning' | 'error' | 'disabled';
    error?: string;
    stats?: Record<string, any>;
    pattern_count?: number;
  }>;
  issues: string[];
};

// Exception classes
export class NSCError extends Error {
  constructor(message: string, public code?: string) {
    super(message);
    this.name = 'NSCError';
  }
}

export class CompressionError extends NSCError {
  constructor(message: string) {
    super(message, 'COMPRESSION_ERROR');
    this.name = 'CompressionError';
  }
}

export class ConfigurationError extends NSCError {
  constructor(message: string) {
    super(message, 'CONFIGURATION_ERROR');
    this.name = 'ConfigurationError';
  }
}

export class PatternError extends NSCError {
  constructor(message: string) {
    super(message, 'PATTERN_ERROR');
    this.name = 'PatternError';
  }
}

export class EngineError extends NSCError {
  constructor(message: string) {
    super(message, 'ENGINE_ERROR');
    this.name = 'EngineError';
  }
}