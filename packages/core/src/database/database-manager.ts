/**
 * Database Manager for Neural Semantic Compiler
 */

import { DatabaseConfig } from '@neurosemantic/types';

export class DatabaseManager {
  private config: DatabaseConfig;
  private db: any;
  private initialized = false;

  constructor(config: DatabaseConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      // Dynamic import of sqlite3 to handle optional dependency
      const sqlite3 = await import('sqlite3').then(m => m.default);
      
      this.db = new sqlite3.Database(this.config.path);
      
      // Enable WAL mode if configured
      if (this.config.enableWalMode) {
        await this.run('PRAGMA journal_mode = WAL');
      }

      // Set cache size
      const cachePages = Math.floor(this.config.cacheSizeMb * 1024 / 4); // 4KB per page
      await this.run(`PRAGMA cache_size = ${cachePages}`);

      // Create tables if they don't exist
      await this.createTables();

      this.initialized = true;
    } catch (error) {
      throw new Error(`Failed to initialize database: ${error}`);
    }
  }

  async query(sql: string, params: any[] = []): Promise<any[]> {
    return new Promise((resolve, reject) => {
      this.db.all(sql, params, (err: any, rows: any[]) => {
        if (err) {
          reject(err);
        } else {
          resolve(rows || []);
        }
      });
    });
  }

  async run(sql: string, params: any[] = []): Promise<{ lastID?: number; changes: number }> {
    return new Promise((resolve, reject) => {
      this.db.run(sql, params, function(this: any, err: any) {
        if (err) {
          reject(err);
        } else {
          resolve({
            lastID: this.lastID,
            changes: this.changes
          });
        }
      });
    });
  }

  async insertPattern(pattern: any): Promise<{ success: boolean; id?: number }> {
    try {
      const sql = `
        INSERT INTO patterns (
          original, compressed, pattern_type, priority, domain, language,
          frequency, success_rate, version, is_active, metadata, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `;

      const params = [
        pattern.original,
        pattern.compressed,
        pattern.patternType,
        pattern.priority,
        pattern.domain,
        pattern.language,
        pattern.frequency,
        pattern.successRate,
        pattern.version,
        pattern.isActive ? 1 : 0,
        JSON.stringify(pattern.metadata || {}),
        new Date().toISOString(),
        new Date().toISOString()
      ];

      const result = await this.run(sql, params);
      
      return {
        success: true,
        id: result.lastID
      };
    } catch (error) {
      console.error('Failed to insert pattern:', error);
      return { success: false };
    }
  }

  async updatePatternUsage(patternId: number, success: boolean): Promise<void> {
    try {
      // Get current stats
      const pattern = await this.query('SELECT frequency, success_rate FROM patterns WHERE id = ?', [patternId]);
      
      if (pattern.length === 0) {
        return;
      }

      const current = pattern[0];
      const newFrequency = current.frequency + 1;
      const newSuccessRate = success 
        ? (current.success_rate * current.frequency + 1) / newFrequency
        : (current.success_rate * current.frequency) / newFrequency;

      await this.run(`
        UPDATE patterns 
        SET frequency = ?, success_rate = ?, updated_at = ?
        WHERE id = ?
      `, [newFrequency, newSuccessRate, new Date().toISOString(), patternId]);

    } catch (error) {
      console.error('Failed to update pattern usage:', error);
    }
  }

  async searchPatterns(query: string, limit: number): Promise<any[]> {
    try {
      const sql = `
        SELECT * FROM patterns 
        WHERE is_active = 1 
          AND (original LIKE ? OR compressed LIKE ? OR domain LIKE ?)
        ORDER BY priority DESC, frequency DESC
        LIMIT ?
      `;

      const searchTerm = `%${query}%`;
      return await this.query(sql, [searchTerm, searchTerm, searchTerm, limit]);
    } catch (error) {
      console.error('Failed to search patterns:', error);
      return [];
    }
  }

  async close(): Promise<void> {
    if (this.db) {
      return new Promise((resolve, reject) => {
        this.db.close((err: any) => {
          if (err) {
            reject(err);
          } else {
            resolve();
          }
        });
      });
    }
  }

  private async createTables(): Promise<void> {
    const createPatternsTable = `
      CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original TEXT NOT NULL,
        compressed TEXT NOT NULL,
        pattern_type TEXT NOT NULL,
        priority INTEGER DEFAULT 500,
        domain TEXT DEFAULT 'general',
        language TEXT DEFAULT 'en',
        frequency INTEGER DEFAULT 0,
        success_rate REAL DEFAULT 0.0,
        version INTEGER DEFAULT 1,
        is_active BOOLEAN DEFAULT 1,
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    `;

    const createCompressionSessionsTable = `
      CREATE TABLE IF NOT EXISTS compression_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE NOT NULL,
        original_text TEXT NOT NULL,
        compressed_text TEXT NOT NULL,
        original_tokens INTEGER NOT NULL,
        compressed_tokens INTEGER NOT NULL,
        compression_ratio REAL NOT NULL,
        quality_score REAL NOT NULL,
        engine_used TEXT NOT NULL,
        processing_time_ms INTEGER NOT NULL,
        context_type TEXT,
        domain TEXT,
        created_at TEXT NOT NULL
      )
    `;

    const createFeedbackTable = `
      CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        rating INTEGER NOT NULL,
        feedback_text TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES compression_sessions(session_id)
      )
    `;

    // Create indexes for better performance
    const createIndexes = [
      'CREATE INDEX IF NOT EXISTS idx_patterns_domain ON patterns(domain)',
      'CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)',
      'CREATE INDEX IF NOT EXISTS idx_patterns_priority ON patterns(priority)',
      'CREATE INDEX IF NOT EXISTS idx_patterns_active ON patterns(is_active)',
      'CREATE INDEX IF NOT EXISTS idx_sessions_created ON compression_sessions(created_at)',
      'CREATE INDEX IF NOT EXISTS idx_sessions_engine ON compression_sessions(engine_used)'
    ];

    try {
      await this.run(createPatternsTable);
      await this.run(createCompressionSessionsTable);
      await this.run(createFeedbackTable);

      for (const indexSql of createIndexes) {
        await this.run(indexSql);
      }

      // Insert default patterns if table is empty
      await this.insertDefaultPatterns();

    } catch (error) {
      throw new Error(`Failed to create tables: ${error}`);
    }
  }

  private async insertDefaultPatterns(): Promise<void> {
    try {
      const count = await this.query('SELECT COUNT(*) as count FROM patterns');
      
      if (count[0].count > 0) {
        return; // Patterns already exist
      }

      const defaultPatterns = [
        {
          original: 'web development',
          compressed: 'web dev',
          patternType: 'phrase',
          priority: 800,
          domain: 'web-development',
          language: 'en',
          frequency: 10,
          successRate: 0.95,
          version: 1,
          isActive: true,
          metadata: { source: 'default' }
        },
        {
          original: 'user interface',
          compressed: 'UI',
          patternType: 'phrase',
          priority: 900,
          domain: 'web-development',
          language: 'en',
          frequency: 15,
          successRate: 0.98,
          version: 1,
          isActive: true,
          metadata: { source: 'default' }
        },
        {
          original: 'application programming interface',
          compressed: 'API',
          patternType: 'phrase',
          priority: 950,
          domain: 'web-development',
          language: 'en',
          frequency: 20,
          successRate: 0.99,
          version: 1,
          isActive: true,
          metadata: { source: 'default' }
        }
      ];

      for (const pattern of defaultPatterns) {
        await this.insertPattern(pattern);
      }

    } catch (error) {
      console.error('Failed to insert default patterns:', error);
    }
  }
}