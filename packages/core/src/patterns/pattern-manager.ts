/**
 * Pattern Manager for Neural Semantic Compiler
 */

import {
  Pattern,
  PatternManager as IPatternManager,
  PatternType
} from '@neurosemantic/types';

export class PatternManager implements IPatternManager {
  private databaseManager: any;
  private cache: Map<string, Pattern[]> = new Map();
  private stats = {
    totalPatterns: 0,
    activePatterns: 0,
    domainCounts: {} as Record<string, number>
  };

  constructor(databaseManager: any) {
    this.databaseManager = databaseManager;
  }

  async initialize(): Promise<void> {
    // Initialize pattern cache
    await this.refreshCache();
  }

  async getPatterns(options: {
    domain?: string;
    patternType?: string;
    language?: string;
    limit?: number;
    minSuccessRate?: number;
  } = {}): Promise<Pattern[]> {
    const cacheKey = JSON.stringify(options);
    
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    try {
      const patterns = await this.queryPatterns(options);
      this.cache.set(cacheKey, patterns);
      return patterns;
    } catch (error) {
      console.error('Failed to get patterns:', error);
      return [];
    }
  }

  async addPattern(pattern: Omit<Pattern, 'id'>): Promise<boolean> {
    try {
      const result = await this.databaseManager.insertPattern(pattern);
      
      if (result.success) {
        // Clear cache to force refresh
        this.cache.clear();
        await this.refreshStats();
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Failed to add pattern:', error);
      return false;
    }
  }

  async updateUsageStats(patternId: number, success: boolean): Promise<void> {
    try {
      await this.databaseManager.updatePatternUsage(patternId, success);
    } catch (error) {
      console.error('Failed to update pattern usage:', error);
    }
  }

  async searchPatterns(query: string, limit: number = 20): Promise<Pattern[]> {
    try {
      return await this.databaseManager.searchPatterns(query, limit);
    } catch (error) {
      console.error('Failed to search patterns:', error);
      return [];
    }
  }

  async getStatistics(): Promise<Record<string, any>> {
    await this.refreshStats();
    
    return {
      total_patterns: this.stats.totalPatterns,
      active_patterns: this.stats.activePatterns,
      domain_counts: this.stats.domainCounts,
      cache_size: this.cache.size,
      pattern_types: await this.getPatternTypeCounts(),
      avg_success_rate: await this.getAverageSuccessRate()
    };
  }

  private async queryPatterns(options: {
    domain?: string;
    patternType?: string;
    language?: string;
    limit?: number;
    minSuccessRate?: number;
  }): Promise<Pattern[]> {
    const conditions: string[] = ['is_active = 1'];
    const params: any[] = [];

    if (options.domain) {
      conditions.push('domain = ?');
      params.push(options.domain);
    }

    if (options.patternType) {
      conditions.push('pattern_type = ?');
      params.push(options.patternType);
    }

    if (options.language) {
      conditions.push('language = ?');
      params.push(options.language);
    }

    if (options.minSuccessRate !== undefined) {
      conditions.push('success_rate >= ?');
      params.push(options.minSuccessRate);
    }

    const query = `
      SELECT * FROM patterns 
      WHERE ${conditions.join(' AND ')}
      ORDER BY priority DESC, success_rate DESC, frequency DESC
      ${options.limit ? 'LIMIT ?' : ''}
    `;

    if (options.limit) {
      params.push(options.limit);
    }

    return await this.databaseManager.query(query, params);
  }

  private async refreshCache(): Promise<void> {
    this.cache.clear();
    
    // Pre-load common queries
    const commonQueries = [
      { domain: 'general' },
      { domain: 'web-development' },
      { domain: 'agile' },
      { patternType: 'phrase' },
      { patternType: 'word' }
    ];

    for (const query of commonQueries) {
      await this.getPatterns(query);
    }
  }

  private async refreshStats(): Promise<void> {
    try {
      const totalResult = await this.databaseManager.query(
        'SELECT COUNT(*) as count FROM patterns'
      );
      this.stats.totalPatterns = totalResult[0]?.count || 0;

      const activeResult = await this.databaseManager.query(
        'SELECT COUNT(*) as count FROM patterns WHERE is_active = 1'
      );
      this.stats.activePatterns = activeResult[0]?.count || 0;

      const domainResult = await this.databaseManager.query(`
        SELECT domain, COUNT(*) as count 
        FROM patterns 
        WHERE is_active = 1 
        GROUP BY domain
      `);
      
      this.stats.domainCounts = {};
      for (const row of domainResult) {
        this.stats.domainCounts[row.domain] = row.count;
      }
      
    } catch (error) {
      console.error('Failed to refresh stats:', error);
    }
  }

  private async getPatternTypeCounts(): Promise<Record<string, number>> {
    try {
      const result = await this.databaseManager.query(`
        SELECT pattern_type, COUNT(*) as count 
        FROM patterns 
        WHERE is_active = 1 
        GROUP BY pattern_type
      `);
      
      const counts: Record<string, number> = {};
      for (const row of result) {
        counts[row.pattern_type] = row.count;
      }
      
      return counts;
    } catch (error) {
      console.error('Failed to get pattern type counts:', error);
      return {};
    }
  }

  private async getAverageSuccessRate(): Promise<number> {
    try {
      const result = await this.databaseManager.query(`
        SELECT AVG(success_rate) as avg_rate 
        FROM patterns 
        WHERE is_active = 1 AND frequency > 0
      `);
      
      return result[0]?.avg_rate || 0;
    } catch (error) {
      console.error('Failed to get average success rate:', error);
      return 0;
    }
  }
}