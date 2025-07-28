/**
 * Neural Semantic Compiler - CLI Wrapper for Node.js
 * 
 * Provides a Node.js interface to the Python CLI backend.
 */

import { spawn, execSync } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';
import {
  CompressionResult,
  CompressionOptions,
  Pattern,
  CompressionStatistics,
  SystemHealth,
  CompressionError,
  ConfigurationError
} from '@neurosemantic/types';

export interface CLIConfig {
  pythonPath?: string;
  cliPath?: string;
  timeout?: number;
  workingDirectory?: string;
  verbose?: boolean;
}

export class NeuralSemanticCLI {
  private config: CLIConfig;
  private cliPath: string;
  private pythonPath: string;

  constructor(config: CLIConfig = {}) {
    this.config = {
      timeout: 30000, // 30 seconds default timeout
      verbose: false,
      ...config
    };

    this.pythonPath = config.pythonPath || this.findPython();
    this.cliPath = config.cliPath || this.findCLI();
    
    this.validateSetup();
  }

  /**
   * Check if Neural Semantic Compiler is available
   */
  static async isAvailable(): Promise<boolean> {
    try {
      const cli = new NeuralSemanticCLI();
      await cli.getHealth();
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Compress text using the Python CLI
   */
  async compress(text: string, options: CompressionOptions = {}): Promise<CompressionResult> {
    const args = ['compress'];
    
    // Add options as CLI arguments
    if (options.level) args.push('--level', options.level);
    if (options.domain) args.push('--domain', options.domain);
    if (options.engine) args.push('--engine', options.engine);
    if (options.preserveCode) args.push('--preserve-code');
    if (options.preserveUrls) args.push('--preserve-urls');
    if (options.preserveNumbers) args.push('--preserve-numbers');
    
    // Add JSON output flag
    args.push('--json-output');

    try {
      const result = await this.runCLICommand(args, text);
      return this.parseCompressionResult(result);
    } catch (error) {
      throw new CompressionError(`CLI compression failed: ${(error as Error).message}`);
    }
  }

  /**
   * Add a new compression pattern
   */
  async addPattern(
    original: string,
    compressed: string,
    options: {
      patternType?: string;
      domain?: string;
      priority?: number;
      language?: string;
    } = {}
  ): Promise<boolean> {
    const args = ['add-pattern', original, compressed];
    
    if (options.patternType) args.push('--type', options.patternType);
    if (options.domain) args.push('--domain', options.domain);
    if (options.priority) args.push('--priority', options.priority.toString());
    if (options.language) args.push('--language', options.language);

    try {
      await this.runCLICommand(args);
      return true;
    } catch (error) {
      if (this.config.verbose) {
        console.error('Failed to add pattern:', error);
      }
      return false;
    }
  }

  /**
   * Get compression patterns
   */
  async getPatterns(options: {
    domain?: string;
    patternType?: string;
    limit?: number;
    search?: string;
  } = {}): Promise<Pattern[]> {
    const args = ['patterns', '--json-output'];
    
    if (options.domain) args.push('--domain', options.domain);
    if (options.patternType) args.push('--type', options.patternType);
    if (options.limit) args.push('--limit', options.limit.toString());
    if (options.search) args.push('--search', options.search);

    try {
      const result = await this.runCLICommand(args);
      return JSON.parse(result);
    } catch (error) {
      if (this.config.verbose) {
        console.error('Failed to get patterns:', error);
      }
      return [];
    }
  }

  /**
   * Compare compression engines
   */
  async compareEngines(text: string, options: CompressionOptions = {}): Promise<Record<string, any>> {
    const args = ['compare', text, '--json-output'];
    
    if (options.level) args.push('--level', options.level);
    if (options.domain) args.push('--domain', options.domain);

    try {
      const result = await this.runCLICommand(args);
      return JSON.parse(result);
    } catch (error) {
      throw new CompressionError(`Engine comparison failed: ${(error as Error).message}`);
    }
  }

  /**
   * Get compression statistics
   */
  async getStatistics(): Promise<CompressionStatistics> {
    try {
      const result = await this.runCLICommand(['stats', '--json-output']);
      return JSON.parse(result);
    } catch (error) {
      throw new CompressionError(`Failed to get statistics: ${(error as Error).message}`);
    }
  }

  /**
   * Get system health
   */
  async getHealth(): Promise<SystemHealth> {
    try {
      const result = await this.runCLICommand(['health']);
      // Parse health output (CLI returns formatted text, not JSON)
      return this.parseHealthOutput(result);
    } catch (error) {
      throw new CompressionError(`Health check failed: ${(error as Error).message}`);
    }
  }

  /**
   * Batch compress multiple texts
   */
  async compressBatch(
    texts: string[],
    options: CompressionOptions = {}
  ): Promise<CompressionResult[]> {
    const results: CompressionResult[] = [];

    for (const text of texts) {
      try {
        const result = await this.compress(text, options);
        results.push(result);
      } catch (error) {
        if (this.config.verbose) {
          console.warn(`Batch compression failed for text: ${text.substring(0, 50)}...`, error);
        }
        // Continue with next text
      }
    }

    return results;
  }

  /**
   * Stream compression for large texts
   */
  async compressStream(
    text: string,
    options: CompressionOptions = {},
    onProgress?: (progress: number) => void
  ): Promise<CompressionResult> {
    // For large texts, we can implement chunking or streaming
    // For now, just use regular compression with progress callback
    if (onProgress) onProgress(0);
    
    const result = await this.compress(text, options);
    
    if (onProgress) onProgress(100);
    return result;
  }

  // Private helper methods

  private findPython(): string {
    // Try to find Python executable
    const candidates = ['python3', 'python', 'py'];
    
    for (const candidate of candidates) {
      try {
        execSync(`${candidate} --version`, { stdio: 'ignore' });
        return candidate;
      } catch {
        // Continue to next candidate
      }
    }
    
    throw new ConfigurationError('Python not found. Please install Python 3.7+ or specify pythonPath in config.');
  }

  private findCLI(): string {
    // Try to find the NSC CLI
    const candidates = [
      'nsc', // If globally installed
      'neuralsemantic', // Alternative command
      path.join(process.cwd(), 'python', 'neuralsemantic', 'cli', 'main.py'), // Local development
      path.join(__dirname, '..', '..', '..', 'python', 'neuralsemantic', 'cli', 'main.py') // Relative to package
    ];

    for (const candidate of candidates) {
      if (candidate.endsWith('.py')) {
        // Check if Python file exists
        if (fs.existsSync(candidate)) {
          return candidate;
        }
      } else {
        // Check if command is available
        try {
          execSync(`${candidate} --help`, { stdio: 'ignore' });
          return candidate;
        } catch {
          // Continue to next candidate
        }
      }
    }

    throw new ConfigurationError('Neural Semantic Compiler CLI not found. Please install the package or specify cliPath in config.');
  }

  private validateSetup(): void {
    try {
      // Validate Python
      execSync(`${this.pythonPath} --version`, { stdio: 'ignore' });
      
      // Validate CLI (if it's a Python file)
      if (this.cliPath.endsWith('.py')) {
        if (!fs.existsSync(this.cliPath)) {
          throw new Error('CLI Python file not found');
        }
      }
    } catch (error) {
      throw new ConfigurationError(`Setup validation failed: ${(error as Error).message}`);
    }
  }

  private async runCLICommand(args: string[], input?: string): Promise<string> {
    return new Promise((resolve, reject) => {
      let command: string;
      let commandArgs: string[];

      if (this.cliPath.endsWith('.py')) {
        // Run Python file
        command = this.pythonPath;
        commandArgs = [this.cliPath, ...args];
      } else {
        // Run CLI command directly
        command = this.cliPath;
        commandArgs = args;
      }

      const process = spawn(command, commandArgs, {
        cwd: this.config.workingDirectory,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      // Set timeout
      const timeout = setTimeout(() => {
        process.kill('SIGTERM');
        reject(new Error(`Command timed out after ${this.config.timeout}ms`));
      }, this.config.timeout);

      process.on('close', (code) => {
        clearTimeout(timeout);
        
        if (code === 0) {
          resolve(stdout.trim());
        } else {
          reject(new Error(`CLI command failed with code ${code}: ${stderr || stdout}`));
        }
      });

      process.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });

      // Send input if provided
      if (input) {
        process.stdin.write(input);
        process.stdin.end();
      }
    });
  }

  private parseCompressionResult(output: string): CompressionResult {
    try {
      const data = JSON.parse(output);
      
      return {
        originalText: data.original_text || '',
        compressedText: data.compressed_text || '',
        originalTokens: data.original_tokens || 0,
        compressedTokens: data.compressed_tokens || 0,
        compressionRatio: data.compression_ratio || 1.0,
        qualityScore: data.quality_score || 0,
        patternMatches: data.pattern_matches || [],
        processingTimeMs: data.processing_time_ms || 0,
        engineUsed: data.engine_used || 'unknown',
        warnings: data.warnings || []
      };
    } catch (error) {
      throw new CompressionError(`Failed to parse compression result: ${(error as Error).message}`);
    }
  }

  private parseHealthOutput(output: string): SystemHealth {
    // Basic parsing of health output
    // In a real implementation, we'd make the CLI output JSON for health too
    const isHealthy = output.includes('HEALTHY') || output.includes('');
    const hasWarnings = output.includes('WARNING') || output.includes('⚠️');
    const hasErrors = output.includes('ERROR') || output.includes('');

    let status: 'healthy' | 'warning' | 'error' = 'healthy';
    if (hasErrors) status = 'error';
    else if (hasWarnings) status = 'warning';

    return {
      overallStatus: status,
      components: {
        patternManager: { status: isHealthy ? 'healthy' : 'unknown' },
        vectorStore: { status: 'unknown' },
        engines: { overallStatus: 'unknown', engineStatus: {}, issues: [] }
      },
      issues: []
    };
  }
}

// Export convenience functions
export async function compress(text: string, options?: CompressionOptions): Promise<CompressionResult> {
  const cli = new NeuralSemanticCLI();
  return cli.compress(text, options);
}

export async function isNSCAvailable(): Promise<boolean> {
  return NeuralSemanticCLI.isAvailable();
}

export async function getHealth(): Promise<SystemHealth> {
  const cli = new NeuralSemanticCLI();
  return cli.getHealth();
}

export async function getStatistics(): Promise<CompressionStatistics> {
  const cli = new NeuralSemanticCLI();
  return cli.getStatistics();
}

// Default export
export default NeuralSemanticCLI;