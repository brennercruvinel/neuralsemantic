/**
 * MCP Server for Claude Code CLI Integration
 * 
 * This server implements the Model Context Protocol (MCP) to provide
 * Neural Semantic Compiler functionality to Claude Code CLI.
 */

import * as WebSocket from 'ws';
import { MCPLoopIntegration } from './integration';
import {
  MCPRequest,
  MCPResponse,
  PromptData,
  MCPLoopConfig,
  WebSocketMessage,
  HealthCheckResult
} from './types';
import { Logger } from './utils/logger';

export class MCPServer {
  private integration: MCPLoopIntegration;
  private server: WebSocket.Server;
  private logger: Logger;
  private config: MCPLoopConfig;
  private connections: Set<WebSocket> = new Set();
  private isRunning: boolean = false;

  constructor(config: Partial<MCPLoopConfig> = {}) {
    this.config = {
      enabled: true,
      port: 8765,
      host: 'localhost',
      ...config
    } as MCPLoopConfig;

    this.logger = new Logger(this.config.logging?.level || 'info');
    this.integration = new MCPLoopIntegration(this.config);
  }

  /**
   * Start the MCP server
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      this.logger.warn('Server is already running');
      return;
    }

    try {
      this.server = new WebSocket.Server({
        port: this.config.port,
        host: this.config.host
      });

      this.setupEventHandlers();
      this.isRunning = true;

      this.logger.info(`MCP Server started on ${this.config.host}:${this.config.port}`);
      this.logger.info('Neural Semantic Compiler is ready to enhance Claude Code CLI');

    } catch (error) {
      this.logger.error('Failed to start MCP server:', error);
      throw error;
    }
  }

  /**
   * Stop the MCP server
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    this.logger.info('Stopping MCP server...');

    // Close all connections
    this.connections.forEach(ws => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    });

    // Close server
    this.server.close();

    // Shutdown integration
    await this.integration.shutdown();

    this.isRunning = false;
    this.logger.info('MCP server stopped');
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupEventHandlers(): void {
    this.server.on('connection', (ws: WebSocket) => {
      this.connections.add(ws);
      this.logger.debug('New WebSocket connection established');

      ws.on('message', async (data: WebSocket.Data) => {
        try {
          const request: MCPRequest = JSON.parse(data.toString());
          const response = await this.handleRequest(request);
          
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(response));
          }
        } catch (error) {
          this.logger.error('Error handling WebSocket message:', error);
          
          const errorResponse: MCPResponse = {
            id: 'unknown',
            error: {
              code: -32700,
              message: 'Parse error',
              data: error.message
            },
            timestamp: Date.now()
          };

          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(errorResponse));
          }
        }
      });

      ws.on('close', () => {
        this.connections.delete(ws);
        this.logger.debug('WebSocket connection closed');
      });

      ws.on('error', (error) => {
        this.logger.error('WebSocket error:', error);
        this.connections.delete(ws);
      });

      // Send welcome message
      this.sendMessage(ws, {
        type: 'health',
        data: {
          status: 'connected',
          message: 'Neural Semantic Compiler ready',
          version: '1.0.0'
        },
        timestamp: Date.now()
      });
    });

    this.server.on('error', (error) => {
      this.logger.error('Server error:', error);
    });
  }

  /**
   * Handle MCP requests
   */
  private async handleRequest(request: MCPRequest): Promise<MCPResponse> {
    this.logger.debug('Handling MCP request:', { method: request.method, id: request.id });

    try {
      let result: any;

      switch (request.method) {
        case 'compress_prompt':
          result = await this.handleCompressPrompt(request.params);
          break;

        case 'get_analytics':
          result = this.handleGetAnalytics(request.params);
          break;

        case 'get_savings_report':
          result = this.handleGetSavingsReport();
          break;

        case 'get_metrics':
          result = this.handleGetMetrics();
          break;

        case 'health_check':
          result = await this.handleHealthCheck();
          break;

        case 'export_analytics':
          result = await this.handleExportAnalytics(request.params);
          break;

        case 'add_compression_strategy':
          result = this.handleAddCompressionStrategy(request.params);
          break;

        default:
          throw new Error(`Unknown method: ${request.method}`);
      }

      return {
        id: request.id,
        result,
        timestamp: Date.now()
      };

    } catch (error) {
      this.logger.error(`Error handling ${request.method}:`, error);

      return {
        id: request.id,
        error: {
          code: -32603,
          message: 'Internal error',
          data: error.message
        },
        timestamp: Date.now()
      };
    }
  }

  /**
   * Handle prompt compression request
   */
  private async handleCompressPrompt(params: any): Promise<any> {
    const promptData: PromptData = {
      prompt: params.prompt,
      sessionId: params.sessionId || this.generateSessionId(),
      files: params.files || [],
      context: params.context || {},
      metadata: params.metadata || {}
    };

    const result = await this.integration.interceptPrompt(promptData);

    // Broadcast compression event to all connections
    this.broadcastMessage({
      type: 'compression',
      data: {
        sessionId: result.sessionId,
        compressionRatio: result.nscMetadata.compressionRatio,
        tokenSavings: result.nscMetadata.tokenSavings,
        costSavings: result.nscMetadata.costSavings,
        qualityScore: result.nscMetadata.qualityScore
      },
      timestamp: Date.now(),
      sessionId: result.sessionId
    });

    return result;
  }

  /**
   * Handle analytics request
   */
  private handleGetAnalytics(params: any): any {
    const sessionId = params?.sessionId;
    return this.integration.getSessionAnalytics(sessionId);
  }

  /**
   * Handle savings report request
   */
  private handleGetSavingsReport(): any {
    return {
      report: this.integration.generateSavingsReport(),
      timestamp: Date.now()
    };
  }

  /**
   * Handle metrics request
   */
  private handleGetMetrics(): any {
    return this.integration.getMetrics();
  }

  /**
   * Handle health check request
   */
  private async handleHealthCheck(): Promise<HealthCheckResult> {
    const isHealthy = await this.integration.healthCheck();
    
    return {
      status: isHealthy ? 'healthy' : 'unhealthy',
      components: {
        compiler: isHealthy,
        mcp_server: this.isRunning,
        websocket: this.server.readyState === WebSocket.OPEN,
        analytics: true
      },
      metrics: {
        uptime: Date.now() - this.startTime,
        totalRequests: this.totalRequests,
        successfulCompressions: this.successfulCompressions,
        failedCompressions: this.failedCompressions,
        totalTokensSaved: 0, // Will be filled by integration
        totalCostSaved: 0,   // Will be filled by integration
        averageResponseTime: this.averageResponseTime,
        peakMemoryUsage: process.memoryUsage().heapUsed,
        activeConnections: this.connections.size,
        healthStatus: isHealthy ? 'healthy' : 'unhealthy',
        lastUpdated: Date.now()
      },
      issues: isHealthy ? [] : ['Integration health check failed'],
      uptime: Date.now() - this.startTime
    };
  }

  /**
   * Handle export analytics request
   */
  private async handleExportAnalytics(params: any): Promise<any> {
    const format = params?.format || 'json';
    const data = await this.integration.exportAnalytics(format);
    
    return {
      format,
      data,
      timestamp: Date.now()
    };
  }

  /**
   * Handle add compression strategy request
   */
  private handleAddCompressionStrategy(params: any): any {
    this.integration.addCompressionStrategy(params.strategy);
    
    return {
      success: true,
      message: `Compression strategy '${params.strategy.name}' added successfully`
    };
  }

  /**
   * Send message to specific WebSocket connection
   */
  private sendMessage(ws: WebSocket, message: WebSocketMessage): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  /**
   * Broadcast message to all connected clients
   */
  private broadcastMessage(message: WebSocketMessage): void {
    this.connections.forEach(ws => {
      this.sendMessage(ws, message);
    });
  }

  /**
   * Generate unique session ID
   */
  private generateSessionId(): string {
    return `mcploop_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Metrics tracking
  private startTime: number = Date.now();
  private totalRequests: number = 0;
  private successfulCompressions: number = 0;
  private failedCompressions: number = 0;
  private averageResponseTime: number = 0;

  /**
   * Get server status
   */
  getStatus() {
    return {
      isRunning: this.isRunning,
      port: this.config.port,
      host: this.config.host,
      connections: this.connections.size,
      uptime: Date.now() - this.startTime
    };
  }
}

// CLI entry point
if (require.main === module) {
  const server = new MCPServer();

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\nReceived SIGINT, shutting down gracefully...');
    await server.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    console.log('\nReceived SIGTERM, shutting down gracefully...');
    await server.stop();
    process.exit(0);
  });

  // Start server
  server.start().then(() => {
    console.log(' Neural Semantic Compiler MCP Server is running!');
    console.log('Ready to enhance Claude Code CLI with intelligent compression.');
    console.log('\nPress Ctrl+C to stop the server.');
  }).catch((error) => {
    console.error('Failed to start MCP server:', error);
    process.exit(1);
  });
}