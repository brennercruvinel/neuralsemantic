"""Pattern management system with SQLite backend."""

import sqlite3
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..core.types import Pattern, PatternType
from ..core.exceptions import DatabaseError, PatternConflictError

logger = logging.getLogger(__name__)


class PatternManager:
    """Manages pattern database operations with caching and optimization."""

    def __init__(self, database_path: str):
        self.db_path = database_path
        self._pattern_cache: Dict[str, List[Pattern]] = {}
        self._cache_timestamp = 0
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database and create tables."""
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Read schema
            schema_path = Path(__file__).parent.parent / "data" / "schema.sql"
            
            if not schema_path.exists():
                # Create minimal schema if file doesn't exist
                self._create_minimal_schema()
            else:
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.executescript(schema_sql)
                    conn.commit()
            
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

    def _create_minimal_schema(self) -> None:
        """Create minimal schema if schema file doesn't exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original TEXT NOT NULL,
            compressed TEXT NOT NULL,
            pattern_type TEXT NOT NULL DEFAULT 'word',
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
            UNIQUE(original, language, domain, version)
        );
        
        CREATE INDEX IF NOT EXISTS idx_patterns_priority ON patterns(priority DESC);
        CREATE INDEX IF NOT EXISTS idx_patterns_domain ON patterns(domain);
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema)
            conn.commit()

    def get_patterns(self, domain: Optional[str] = None, 
                    pattern_type: Optional[str] = None,
                    limit: Optional[int] = None) -> List[Pattern]:
        """Retrieve patterns with intelligent caching and dynamic priority adjustment."""
        cache_key = f"{domain}:{pattern_type}:{limit}"
        current_time = time.time()

        # Check cache validity (5 minute TTL)
        if (cache_key in self._pattern_cache and
            current_time - self._cache_timestamp < 300):
            return self._pattern_cache[cache_key]

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Query database with dynamic priority adjustment
                query = """
                SELECT *,
                       (priority * 0.7 + success_rate * 300 * 0.3) as effective_priority
                FROM patterns
                WHERE (:domain IS NULL OR domain = :domain)
                AND (:pattern_type IS NULL OR pattern_type = :pattern_type)
                AND is_active = TRUE
                ORDER BY effective_priority DESC, frequency DESC
                """
                
                if limit:
                    query += f" LIMIT {limit}"

                cursor = conn.execute(query, {
                    'domain': domain,
                    'pattern_type': pattern_type
                })

                patterns = [Pattern.from_row(row) for row in cursor.fetchall()]

            # Update cache
            self._pattern_cache[cache_key] = patterns
            self._cache_timestamp = current_time

            return patterns

        except Exception as e:
            logger.error(f"Failed to retrieve patterns: {e}")
            raise DatabaseError(f"Pattern retrieval failed: {e}")

    def add_pattern(self, pattern: Pattern) -> bool:
        """Add new pattern with conflict detection."""
        # Check for conflicts
        if self._has_conflicts(pattern):
            logger.warning(f"Pattern conflict detected for: {pattern.original}")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                INSERT INTO patterns
                (original, compressed, pattern_type, priority, domain, language)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    pattern.original, 
                    pattern.compressed, 
                    pattern.pattern_type.value,
                    pattern.priority, 
                    pattern.domain, 
                    pattern.language
                ))
                conn.commit()

            # Invalidate cache
            self._pattern_cache.clear()
            logger.info(f"Added pattern: {pattern.original} → {pattern.compressed}")
            return True

        except sqlite3.IntegrityError:
            logger.warning(f"Pattern already exists: {pattern.original}")
            return False
        except Exception as e:
            logger.error(f"Failed to add pattern: {e}")
            raise DatabaseError(f"Pattern addition failed: {e}")

    def update_usage_stats(self, pattern_id: int, success: bool) -> None:
        """Update pattern usage statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                UPDATE patterns
                SET frequency = frequency + 1,
                    last_used = CURRENT_TIMESTAMP
                WHERE id = ?
                """, (pattern_id,))

                # Update success rate if we have usage tracking
                if success:
                    conn.execute("""
                    INSERT INTO pattern_usage (pattern_id, quality_score, used_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """, (pattern_id, 8.0 if success else 3.0))

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to update usage stats: {e}")

    def _has_conflicts(self, new_pattern: Pattern) -> bool:
        """Check for pattern conflicts (circular references, ambiguity)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if compressed form already exists as original
                cursor = conn.execute(
                    "SELECT id FROM patterns WHERE original = ?",
                    (new_pattern.compressed,)
                )
                if cursor.fetchone():
                    return True

                # Check if original already has different compression
                cursor = conn.execute(
                    "SELECT compressed FROM patterns WHERE original = ? AND compressed != ?",
                    (new_pattern.original, new_pattern.compressed)
                )
                if cursor.fetchone():
                    return True

            return False

        except Exception as e:
            logger.error(f"Conflict check failed: {e}")
            return True  # Assume conflict on error

    def get_pattern_by_id(self, pattern_id: int) -> Optional[Pattern]:
        """Get pattern by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM patterns WHERE id = ?", (pattern_id,))
                row = cursor.fetchone()
                
                if row:
                    return Pattern.from_row(row)
                return None

        except Exception as e:
            logger.error(f"Failed to get pattern by ID: {e}")
            return None

    def search_patterns(self, query: str, limit: int = 20) -> List[Pattern]:
        """Search patterns by text."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                SELECT * FROM patterns 
                WHERE (original LIKE ? OR compressed LIKE ?)
                AND is_active = TRUE
                ORDER BY priority DESC, frequency DESC
                LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
                
                return [Pattern.from_row(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Total patterns
                cursor = conn.execute("SELECT COUNT(*) FROM patterns WHERE is_active = TRUE")
                stats['total_patterns'] = cursor.fetchone()[0]
                
                # Patterns by domain
                cursor = conn.execute("""
                SELECT domain, COUNT(*) FROM patterns 
                WHERE is_active = TRUE 
                GROUP BY domain
                """)
                stats['patterns_by_domain'] = dict(cursor.fetchall())
                
                # Patterns by type
                cursor = conn.execute("""
                SELECT pattern_type, COUNT(*) FROM patterns 
                WHERE is_active = TRUE 
                GROUP BY pattern_type
                """)
                stats['patterns_by_type'] = dict(cursor.fetchall())
                
                # Most used patterns
                cursor = conn.execute("""
                SELECT original, compressed, frequency FROM patterns 
                WHERE is_active = TRUE 
                ORDER BY frequency DESC 
                LIMIT 10
                """)
                stats['most_used'] = cursor.fetchall()
                
                return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def bulk_add_patterns(self, patterns: List[Pattern]) -> int:
        """Add multiple patterns efficiently."""
        added_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for pattern in patterns:
                    try:
                        conn.execute("""
                        INSERT OR IGNORE INTO patterns
                        (original, compressed, pattern_type, priority, domain, language)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            pattern.original,
                            pattern.compressed,
                            pattern.pattern_type.value,
                            pattern.priority,
                            pattern.domain,
                            pattern.language
                        ))
                        
                        if conn.total_changes > 0:
                            added_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to add pattern {pattern.original}: {e}")
                        continue
                
                conn.commit()
                
            # Invalidate cache
            self._pattern_cache.clear()
            logger.info(f"Bulk added {added_count} patterns")
            
            return added_count

        except Exception as e:
            logger.error(f"Bulk pattern addition failed: {e}")
            raise DatabaseError(f"Bulk pattern addition failed: {e}")

    def delete_pattern(self, pattern_id: int) -> bool:
        """Delete pattern by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM patterns WHERE id = ?", (pattern_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    self._pattern_cache.clear()
                    logger.info(f"Deleted pattern ID: {pattern_id}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to delete pattern: {e}")
            return False