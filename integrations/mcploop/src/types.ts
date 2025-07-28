/**
 * MCPLOOP Integration Type Definitions
 */

import { CompressionResult, CompressionOptions } from '@neurosemantic/types';

export interface MCPRequest {
  id: string;
  method: string;
  params: Record<string, any>;
  timestamp: number;
}

export interface MCPResponse {
  id: string;
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
  timestamp: number;
}

export interface PromptData {
  prompt: string;
  sessionId: string;
  files?: string[];
  context?: {
    command?: string;
    projectType?: string;
    language?: string;
    framework?: string;
  };
  metadata?: Record<string, any>;
}

export interface CompressedPromptData extends PromptData {
  originalPrompt: string;
  nscMetadata: {
    originalLength: number;
    compressedLength: number;
    compressionRatio: number;
    tokenSavings: number;
    costSavings: number;
    qualityScore: number;
    engineUsed: string;
    processingTimeMs: number;
    patternsApplied: number;
    warnings: string[];
  };
}

export interface SessionAnalytics {
  sessionId: string;
  totalPrompts: number;
  totalCompressions: number;
  totalTokensSaved: number;
  totalCostSaved: number;
  averageCompressionRatio: number;
  averageQualityScore: number;
  compressionsByEngine: Record<string, number>;
  domainBreakdown: Record<string, number>;
  timelineData: Array<{
    timestamp: number;
    compression: number;
    quality: number;
    tokensSaved: number;
  }>;
  startTime: number;
  lastActivity: number;
}

export interface CostCalculation {
  originalCost: number;
  compressedCost: number;
  savings: number;
  savingsPercentage: number;
  currency: string;
  pricingModel: {
    provider: string;
    model: string;
    inputCostPer1K: number;
    outputCostPer1K: number;
  };
}

export interface MCPLoopConfig {
  enabled: boolean;
  port: number;
  host: string;
  compressionOptions: CompressionOptions;
  costCalculation: {
    enabled: boolean;
    defaultProvider: 'openai' | 'anthropic' | 'custom';
    pricingModels: Record<string, {
      inputCostPer1K: number;
      outputCostPer1K: number;
    }>;
  };
  analytics: {
    enabled: boolean;
    sessionTimeout: number; // minutes
    retentionDays: number;
  };
  security: {
    enableAuth: boolean;
    apiKey?: string;
    allowedOrigins: string[];
  };
  logging: {
    level: 'debug' | 'info' | 'warn' | 'error';
    logFile?: string;
  };
}

export interface ContextHints {
  domain?: string;
  projectType?: 'web' | 'mobile' | 'backend' | 'ai' | 'data' | 'devops' | 'other';
  framework?: string;
  language?: string;
  complexity?: 'simple' | 'medium' | 'complex';
  urgency?: 'low' | 'medium' | 'high';
}

export interface CompressionStrategy {
  name: string;
  description: string;
  conditions: Array<{
    type: 'fileExtension' | 'keyword' | 'pattern' | 'context';
    value: string | RegExp;
    weight: number;
  }>;
  compressionOptions: CompressionOptions;
  priority: number;
}

export interface InterceptionRule {
  name: string;
  enabled: boolean;
  conditions: Array<{
    type: 'promptLength' | 'fileCount' | 'keyword' | 'pattern' | 'context';
    operator: 'gt' | 'lt' | 'eq' | 'contains' | 'matches';
    value: string | number | RegExp;
  }>;
  action: 'compress' | 'skip' | 'conditional';
  strategy?: string;
  fallback?: {
    enabled: boolean;
    strategy?: string;
  };
}

export interface MCPLoopMetrics {
  uptime: number;
  totalRequests: number;
  successfulCompressions: number;
  failedCompressions: number;
  totalTokensSaved: number;
  totalCostSaved: number;
  averageResponseTime: number;
  peakMemoryUsage: number;
  activeConnections: number;
  healthStatus: 'healthy' | 'degraded' | 'unhealthy';
  lastUpdated: number;
}

export interface DebugInfo {
  request: {
    method: string;
    params: any;
    timestamp: number;
    headers?: Record<string, string>;
  };
  processing: {
    contextDetection: {
      detectedDomain?: string;
      detectedFramework?: string;
      confidence: number;
    };
    compression: {
      engineSelected: string;
      processingSteps: string[];
      warnings: string[];
    };
    performance: {
      totalTime: number;
      compressionTime: number;
      overheadTime: number;
    };
  };
  response: {
    compressedPrompt: string;
    metadata: any;
    timestamp: number;
  };
}

export interface WebSocketMessage {
  type: 'compression' | 'analytics' | 'health' | 'debug' | 'error';
  data: any;
  timestamp: number;
  sessionId?: string;
}

export interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: {
    compiler: boolean;
    mcp_server: boolean;
    websocket: boolean;
    analytics: boolean;
  };
  metrics: MCPLoopMetrics;
  issues: string[];
  uptime: number;
}

// Events
export interface CompressionEvent {
  type: 'start' | 'complete' | 'error';
  sessionId: string;
  data: {
    originalPrompt?: string;
    compressedPrompt?: string;
    result?: CompressionResult;
    error?: Error;
  };
  timestamp: number;
}

export interface AnalyticsEvent {
  type: 'session_start' | 'session_end' | 'compression' | 'error';
  sessionId: string;
  data: any;
  timestamp: number;
}

// Configuration validation
export interface ConfigValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

// Plugin system for MCPLOOP
export interface MCPLoopPlugin {
  name: string;
  version: string;
  description: string;
  
  initialize(config: any): Promise<void>;
  beforeCompression?(data: PromptData): Promise<PromptData>;
  afterCompression?(result: CompressedPromptData): Promise<CompressedPromptData>;
  onAnalytics?(event: AnalyticsEvent): Promise<void>;
  shutdown?(): Promise<void>;
}

// Export utilities
export interface ExportOptions {
  format: 'json' | 'csv' | 'xlsx';
  dateRange?: {
    start: number;
    end: number;
  };
  includeSessionData: boolean;
  includeAnalytics: boolean;
  includeDebugInfo: boolean;
  compression?: 'none' | 'gzip';
}

export interface ImportOptions {
  format: 'json' | 'csv';
  source: 'file' | 'url' | 'clipboard';
  mergeStrategy: 'replace' | 'append' | 'merge';
  validateData: boolean;
}