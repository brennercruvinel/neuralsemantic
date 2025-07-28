/**
 * Conflict Resolver for Pattern Management
 */

import {
  ConflictResolver as IConflictResolver,
  PatternMatch,
  Pattern
} from '@neurosemantic/types';

export class ConflictResolver implements IConflictResolver {
  
  /**
   * Resolve conflicts between overlapping pattern matches
   */
  resolveConflicts(matches: PatternMatch[], strategy: string = 'priority'): PatternMatch[] {
    if (matches.length <= 1) {
      return matches;
    }

    // Sort matches by position
    const sortedMatches = matches.sort((a, b) => a.position - b.position);
    
    // Detect overlapping matches
    const conflicts = this.detectOverlaps(sortedMatches);
    
    if (conflicts.length === 0) {
      return sortedMatches;
    }

    // Resolve conflicts based on strategy
    switch (strategy) {
      case 'priority':
        return this.resolveByyPriority(sortedMatches, conflicts);
      case 'confidence':
        return this.resolveByConfidence(sortedMatches, conflicts);
      case 'length':
        return this.resolveByLength(sortedMatches, conflicts);
      case 'compression':
        return this.resolveByCompression(sortedMatches, conflicts);
      default:
        return this.resolveByyPriority(sortedMatches, conflicts);
    }
  }

  /**
   * Detect potential conflicts between patterns
   */
  detectPatternConflicts(patterns: Pattern[]): Array<Record<string, any>> {
    const conflicts: Array<Record<string, any>> = [];
    
    for (let i = 0; i < patterns.length; i++) {
      for (let j = i + 1; j < patterns.length; j++) {
        const conflict = this.analyzePatternConflict(patterns[i], patterns[j]);
        if (conflict) {
          conflicts.push(conflict);
        }
      }
    }

    return conflicts;
  }

  private detectOverlaps(matches: PatternMatch[]): Array<{ indices: number[]; type: string }> {
    const overlaps: Array<{ indices: number[]; type: string }> = [];
    
    for (let i = 0; i < matches.length; i++) {
      const current = matches[i];
      const currentEnd = current.position + current.originalText.length;
      
      for (let j = i + 1; j < matches.length; j++) {
        const next = matches[j];
        
        // Check for overlap
        if (next.position < currentEnd) {
          const overlapType = this.determineOverlapType(current, next);
          overlaps.push({
            indices: [i, j],
            type: overlapType
          });
        } else {
          // No more overlaps with current match (sorted by position)
          break;
        }
      }
    }

    return overlaps;
  }

  private determineOverlapType(match1: PatternMatch, match2: PatternMatch): string {
    const end1 = match1.position + match1.originalText.length;
    const end2 = match2.position + match2.originalText.length;

    if (match1.position === match2.position && end1 === end2) {
      return 'exact';
    } else if (match1.position <= match2.position && end1 >= end2) {
      return 'contains';
    } else if (match2.position <= match1.position && end2 >= end1) {
      return 'contained';
    } else {
      return 'partial';
    }
  }

  private resolveByyPriority(matches: PatternMatch[], conflicts: Array<{ indices: number[]; type: string }>): PatternMatch[] {
    const toRemove = new Set<number>();

    for (const conflict of conflicts) {
      const [i, j] = conflict.indices;
      const match1 = matches[i];
      const match2 = matches[j];

      const priority1 = match1.pattern.priority || 500;
      const priority2 = match2.pattern.priority || 500;

      if (priority1 > priority2) {
        toRemove.add(j);
      } else if (priority2 > priority1) {
        toRemove.add(i);
      } else {
        // Same priority - use confidence as tiebreaker
        if (match1.confidence > match2.confidence) {
          toRemove.add(j);
        } else {
          toRemove.add(i);
        }
      }
    }

    return matches.filter((_, index) => !toRemove.has(index));
  }

  private resolveByConfidence(matches: PatternMatch[], conflicts: Array<{ indices: number[]; type: string }>): PatternMatch[] {
    const toRemove = new Set<number>();

    for (const conflict of conflicts) {
      const [i, j] = conflict.indices;
      const match1 = matches[i];
      const match2 = matches[j];

      if (match1.confidence > match2.confidence) {
        toRemove.add(j);
      } else if (match2.confidence > match1.confidence) {
        toRemove.add(i);
      } else {
        // Same confidence - use priority as tiebreaker
        const priority1 = match1.pattern.priority || 500;
        const priority2 = match2.pattern.priority || 500;
        
        if (priority1 > priority2) {
          toRemove.add(j);
        } else {
          toRemove.add(i);
        }
      }
    }

    return matches.filter((_, index) => !toRemove.has(index));
  }

  private resolveByLength(matches: PatternMatch[], conflicts: Array<{ indices: number[]; type: string }>): PatternMatch[] {
    const toRemove = new Set<number>();

    for (const conflict of conflicts) {
      const [i, j] = conflict.indices;
      const match1 = matches[i];
      const match2 = matches[j];

      // Prefer longer matches (more specific)
      if (match1.originalText.length > match2.originalText.length) {
        toRemove.add(j);
      } else if (match2.originalText.length > match1.originalText.length) {
        toRemove.add(i);
      } else {
        // Same length - use confidence as tiebreaker
        if (match1.confidence > match2.confidence) {
          toRemove.add(j);
        } else {
          toRemove.add(i);
        }
      }
    }

    return matches.filter((_, index) => !toRemove.has(index));
  }

  private resolveByCompression(matches: PatternMatch[], conflicts: Array<{ indices: number[]; type: string }>): PatternMatch[] {
    const toRemove = new Set<number>();

    for (const conflict of conflicts) {
      const [i, j] = conflict.indices;
      const match1 = matches[i];
      const match2 = matches[j];

      // Calculate compression ratios
      const ratio1 = match1.compressedText.length / match1.originalText.length;
      const ratio2 = match2.compressedText.length / match2.originalText.length;

      // Prefer better compression (lower ratio)
      if (ratio1 < ratio2) {
        toRemove.add(j);
      } else if (ratio2 < ratio1) {
        toRemove.add(i);
      } else {
        // Same compression - use confidence as tiebreaker
        if (match1.confidence > match2.confidence) {
          toRemove.add(j);
        } else {
          toRemove.add(i);
        }
      }
    }

    return matches.filter((_, index) => !toRemove.has(index));
  }

  private analyzePatternConflict(pattern1: Pattern, pattern2: Pattern): Record<string, any> | null {
    // Check for exact duplicates
    if (pattern1.original === pattern2.original && pattern1.compressed === pattern2.compressed) {
      return {
        type: 'duplicate',
        pattern1: pattern1.id,
        pattern2: pattern2.id,
        severity: 'high',
        description: 'Exact duplicate patterns'
      };
    }

    // Check for overlapping originals with different compressions
    if (pattern1.original === pattern2.original && pattern1.compressed !== pattern2.compressed) {
      return {
        type: 'conflicting_compression',
        pattern1: pattern1.id,
        pattern2: pattern2.id,
        severity: 'high',
        description: 'Same original text with different compressions'
      };
    }

    // Check for substring conflicts
    if (pattern1.original.includes(pattern2.original) || pattern2.original.includes(pattern1.original)) {
      return {
        type: 'substring_conflict',
        pattern1: pattern1.id,
        pattern2: pattern2.id,
        severity: 'medium',
        description: 'One pattern is a substring of another'
      };
    }

    // Check for similar compressed forms
    if (pattern1.compressed === pattern2.compressed && pattern1.original !== pattern2.original) {
      return {
        type: 'ambiguous_compression',
        pattern1: pattern1.id,
        pattern2: pattern2.id,
        severity: 'medium',
        description: 'Different originals compress to the same form'
      };
    }

    // Check for domain conflicts
    if (pattern1.domain !== pattern2.domain && 
        this.calculateSimilarity(pattern1.original, pattern2.original) > 0.8) {
      return {
        type: 'domain_conflict',
        pattern1: pattern1.id,
        pattern2: pattern2.id,
        severity: 'low',
        description: 'Similar patterns in different domains'
      };
    }

    return null;
  }

  private calculateSimilarity(text1: string, text2: string): number {
    // Simple Jaccard similarity
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));
    
    const intersection = new Set([...words1].filter(word => words2.has(word)));
    const union = new Set([...words1, ...words2]);
    
    return intersection.size / union.size;
  }
}