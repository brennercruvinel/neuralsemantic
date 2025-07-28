-- Neural Semantic Compiler Database Schema
-- SQLite database for pattern storage and management

-- Core patterns table
CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original TEXT NOT NULL,
    compressed TEXT NOT NULL,
    pattern_type TEXT NOT NULL CHECK (pattern_type IN ('phrase', 'compound', 'word', 'abbreviation', 'structure')),
    priority INTEGER NOT NULL DEFAULT 500,
    language TEXT NOT NULL DEFAULT 'en',
    domain TEXT NOT NULL DEFAULT 'general',
    frequency INTEGER NOT NULL DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    UNIQUE(original, language, domain, version)
);

-- Pattern version history
CREATE TABLE IF NOT EXISTS pattern_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id INTEGER NOT NULL,
    version INTEGER NOT NULL,
    original TEXT NOT NULL,
    compressed TEXT NOT NULL,
    change_reason TEXT,
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES patterns(id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_patterns_priority ON patterns(priority DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_domain ON patterns(domain);
CREATE INDEX IF NOT EXISTS idx_patterns_frequency ON patterns(frequency DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_success_rate ON patterns(success_rate DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_original ON patterns(original);
CREATE INDEX IF NOT EXISTS idx_patterns_active ON patterns(is_active);
CREATE INDEX IF NOT EXISTS idx_patterns_composite ON patterns(domain, pattern_type, is_active, priority DESC);

-- Pattern usage tracking
CREATE TABLE IF NOT EXISTS pattern_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id INTEGER NOT NULL,
    context_type TEXT,
    compression_ratio REAL,
    quality_score REAL,
    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (pattern_id) REFERENCES patterns(id)
);

-- Conflict resolution tracking
CREATE TABLE IF NOT EXISTS pattern_conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern1_id INTEGER NOT NULL,
    pattern2_id INTEGER NOT NULL,
    conflict_type TEXT NOT NULL,
    resolution TEXT,
    resolved_at TIMESTAMP,
    FOREIGN KEY (pattern1_id) REFERENCES patterns(id),
    FOREIGN KEY (pattern2_id) REFERENCES patterns(id)
);

-- Compression sessions for analytics
CREATE TABLE IF NOT EXISTS compression_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    compression_ratio REAL,
    quality_score REAL,
    engine_used TEXT,
    context_type TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User feedback
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    feedback_text TEXT,
    improvement_suggestions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Configuration storage
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert core configuration defaults
INSERT OR IGNORE INTO config (key, value, description) VALUES
('schema_version', '1.0.0', 'Database schema version'),
('last_pattern_sync', datetime('now'), 'Last pattern synchronization'),
('compression_stats_enabled', 'true', 'Enable compression statistics collection'),
('auto_learning_enabled', 'true', 'Enable automatic pattern learning'),
('vector_store_enabled', 'true', 'Enable vector store integration');

-- Insert essential core patterns
INSERT OR IGNORE INTO patterns (original, compressed, pattern_type, priority, domain, language) VALUES
-- High priority web development patterns
('production-ready', 'prod-rdy', 'compound', 900, 'web-development', 'en'),
('authentication', 'auth', 'word', 850, 'web-development', 'en'),
('authorization', 'authz', 'word', 850, 'web-development', 'en'),
('application', 'app', 'word', 800, 'general', 'en'),
('development', 'dev', 'word', 800, 'general', 'en'),
('environment', 'env', 'word', 800, 'general', 'en'),
('configuration', 'config', 'word', 800, 'general', 'en'),
('implementation', 'impl', 'word', 780, 'general', 'en'),
('information', 'info', 'word', 750, 'general', 'en'),
('management', 'mgmt', 'word', 750, 'general', 'en'),

-- React/Frontend patterns
('React application', 'React app', 'compound', 900, 'web-development', 'en'),
('React component', 'React comp', 'compound', 880, 'web-development', 'en'),
('user interface', 'UI', 'compound', 850, 'web-development', 'en'),
('user experience', 'UX', 'compound', 850, 'web-development', 'en'),
('responsive design', 'responsive', 'compound', 800, 'web-development', 'en'),
('single page application', 'SPA', 'compound', 880, 'web-development', 'en'),

-- Backend patterns
('Application Programming Interface', 'API', 'phrase', 900, 'web-development', 'en'),
('Representational State Transfer', 'REST', 'phrase', 880, 'web-development', 'en'),
('database', 'DB', 'word', 850, 'web-development', 'en'),
('microservices', 'μsvc', 'word', 800, 'web-development', 'en'),
('Kubernetes', 'k8s', 'word', 880, 'web-development', 'en'),

-- Agile patterns
('Sprint planning', 'Spr plan', 'compound', 900, 'agile', 'en'),
('product owner', 'PO', 'compound', 850, 'agile', 'en'),
('scrum master', 'SM', 'compound', 850, 'agile', 'en'),
('user stories', 'usr stories', 'compound', 800, 'agile', 'en'),
('story points', 'story pts', 'compound', 800, 'agile', 'en'),
('sprint backlog', 'spr backlog', 'compound', 800, 'agile', 'en'),
('retrospective', 'retro', 'word', 750, 'agile', 'en'),

-- General programming patterns
('function', 'fn', 'word', 700, 'general', 'en'),
('variable', 'var', 'word', 700, 'general', 'en'),
('object', 'obj', 'word', 700, 'general', 'en'),
('repository', 'repo', 'word', 750, 'general', 'en'),
('documentation', 'docs', 'word', 750, 'general', 'en'),
('requirements', 'reqs', 'word', 750, 'general', 'en'),

-- Common abbreviations
('with', 'w/', 'word', 600, 'general', 'en'),
('without', 'w/o', 'word', 600, 'general', 'en'),
('and', '&', 'word', 500, 'general', 'en'),
('for', '4', 'word', 400, 'general', 'en'),
('to', '2', 'word', 300, 'general', 'en'),
('you', 'u', 'word', 300, 'general', 'en'),

-- Technical compounds
('machine learning', 'ML', 'compound', 900, 'general', 'en'),
('artificial intelligence', 'AI', 'compound', 900, 'general', 'en'),
('natural language processing', 'NLP', 'phrase', 900, 'general', 'en'),
('computer vision', 'CV', 'compound', 850, 'general', 'en'),
('deep learning', 'DL', 'compound', 850, 'general', 'en'),
('neural network', 'NN', 'compound', 850, 'general', 'en');