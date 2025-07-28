import { AnalyticsData, CompressionResult, CompressionContext } from '@neurosemantic/types';
import { DatabaseManager } from '../database/database-manager';

export class AnalyticsManager {
  private sessionCache: Map<string, { result: CompressionResult; context: CompressionContext }> = new Map();
  private readonly MAX_CACHE_SIZE = 1000;

  constructor(private databaseManager: DatabaseManager) {}

  async storeCompressionSession(result: CompressionResult, context: CompressionContext): Promise<void> {
    if (!result.sessionId) {
      console.warn('Cannot store session without sessionId');
      return;
    }

    // Store in cache with size limit
    this.sessionCache.set(result.sessionId, { result, context });
    
    // Implement FIFO cache eviction
    if (this.sessionCache.size > this.MAX_CACHE_SIZE) {
      const firstKey = this.sessionCache.keys().next().value;
      this.sessionCache.delete(firstKey);
    }

    // TODO: Persist to database when database manager is fully implemented
  }

  async recordFeedback(sessionId: string, rating: number, feedback?: string): Promise<void> {
    if (!sessionId || rating < 1 || rating > 5) {
      throw new Error('Invalid feedback parameters');
    }

    const session = this.sessionCache.get(sessionId);
    if (!session) {
      console.warn(`Session ${sessionId} not found in cache`);
      return;
    }

    // TODO: Store feedback in database
  }

  async getAnalytics(): Promise<AnalyticsData> {
    // Calculate real analytics from cache
    const sessions = Array.from(this.sessionCache.values());
    const totalSessions = sessions.length;
    
    if (totalSessions === 0) {
      return {
        totalSessions: 0,
        totalCostSavings: 0,
        averageCompressionRatio: 0,
        averageQualityScore: 0,
        totalTokensSaved: 0,
        sessionsToday: 0,
        topDomains: [],
        enginePerformance: {}
      };
    }

    const totalTokensSaved = sessions.reduce((sum, s) => sum + (s.result.originalTokens - s.result.compressedTokens), 0);
    const avgCompressionRatio = sessions.reduce((sum, s) => sum + s.result.compressionRatio, 0) / totalSessions;
    const avgQualityScore = sessions.reduce((sum, s) => sum + s.result.qualityScore, 0) / totalSessions;

    return {
      totalSessions,
      totalCostSavings: totalTokensSaved * 0.00002, // Estimate based on token pricing
      averageCompressionRatio: avgCompressionRatio,
      averageQualityScore: avgQualityScore,
      totalTokensSaved,
      sessionsToday: totalSessions, // TODO: Filter by date
      topDomains: this.getTopDomains(sessions),
      enginePerformance: this.getEnginePerformance(sessions)
    };
  }

  private getTopDomains(sessions: Array<{ result: CompressionResult; context: CompressionContext }>): string[] {
    const domainCounts = new Map<string, number>();
    
    sessions.forEach(s => {
      const domain = s.context.domain || 'general';
      domainCounts.set(domain, (domainCounts.get(domain) || 0) + 1);
    });

    return Array.from(domainCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([domain]) => domain);
  }

  private getEnginePerformance(sessions: Array<{ result: CompressionResult; context: CompressionContext }>): Record<string, any> {
    const engineStats = new Map<string, { count: number; totalRatio: number; totalQuality: number }>();
    
    sessions.forEach(s => {
      const engine = s.result.engineUsed;
      const stats = engineStats.get(engine) || { count: 0, totalRatio: 0, totalQuality: 0 };
      
      stats.count++;
      stats.totalRatio += s.result.compressionRatio;
      stats.totalQuality += s.result.qualityScore;
      
      engineStats.set(engine, stats);
    });

    const performance: Record<string, any> = {};
    engineStats.forEach((stats, engine) => {
      performance[engine] = {
        usageCount: stats.count,
        averageCompressionRatio: stats.totalRatio / stats.count,
        averageQualityScore: stats.totalQuality / stats.count
      };
    });

    return performance;
  }

  clearCache(): void {
    this.sessionCache.clear();
  }
}