/**
 * Vector Store for Neural Semantic Compiler
 */

import {
  VectorStore as IVectorStore,
  VectorConfig,
  Pattern,
  SimilarPattern
} from '@neurosemantic/types';

export class VectorStore implements IVectorStore {
  public enabled: boolean = false;
  private config: VectorConfig;
  private chromaClient: any;
  private collection: any;
  private embeddingManager: any;

  constructor(config: VectorConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    try {
      // Try to initialize ChromaDB
      const { ChromaClient } = await import('chromadb');
      
      this.chromaClient = new ChromaClient({
        path: this.config.persistDirectory
      });

      // Get or create collection
      try {
        this.collection = await this.chromaClient.getCollection({
          name: 'neural_semantic_patterns'
        });
      } catch (error) {
        this.collection = await this.chromaClient.createCollection({
          name: 'neural_semantic_patterns',
          metadata: { description: 'Neural Semantic Compiler patterns' }
        });
      }

      this.enabled = true;
      console.log('Vector store initialized successfully');

    } catch (error) {
      console.warn('Vector store not available:', error);
      this.enabled = false;
    }
  }

  async addPattern(pattern: Pattern): Promise<void> {
    if (!this.enabled || !this.collection) {
      return;
    }

    try {
      await this.collection.add({
        ids: [`pattern_${pattern.id}`],
        documents: [pattern.original],
        metadatas: [{
          pattern_id: pattern.id,
          compressed: pattern.compressed,
          pattern_type: pattern.patternType,
          domain: pattern.domain,
          priority: pattern.priority,
          success_rate: pattern.successRate
        }]
      });
    } catch (error) {
      console.error('Failed to add pattern to vector store:', error);
    }
  }

  async findSimilarPatterns(
    text: string,
    nResults: number = 10,
    threshold: number = 0.7,
    domain?: string
  ): Promise<SimilarPattern[]> {
    if (!this.enabled || !this.collection) {
      return [];
    }

    try {
      const queryFilter = domain ? { domain } : undefined;

      const results = await this.collection.query({
        queryTexts: [text],
        nResults,
        where: queryFilter
      });

      const similarPatterns: SimilarPattern[] = [];

      if (results.documents && results.metadatas && results.distances) {
        for (let i = 0; i < results.documents[0].length; i++) {
          const distance = results.distances[0][i];
          const similarity = 1 - distance; // Convert distance to similarity

          if (similarity >= threshold) {
            const metadata = results.metadatas[0][i];
            
            similarPatterns.push({
              original: results.documents[0][i],
              compressed: metadata.compressed,
              similarity,
              patternType: metadata.pattern_type,
              domain: metadata.domain,
              priority: metadata.priority,
              confidence: similarity * (metadata.success_rate || 0.5)
            });
          }
        }
      }

      return similarPatterns.sort((a, b) => b.confidence - a.confidence);

    } catch (error) {
      console.error('Failed to find similar patterns:', error);
      return [];
    }
  }

  async bulkAddPatterns(patterns: Pattern[]): Promise<void> {
    if (!this.enabled || !this.collection) {
      return;
    }

    try {
      const ids = patterns.map(p => `pattern_${p.id}`);
      const documents = patterns.map(p => p.original);
      const metadatas = patterns.map(p => ({
        pattern_id: p.id,
        compressed: p.compressed,
        pattern_type: p.patternType,
        domain: p.domain,
        priority: p.priority,
        success_rate: p.successRate
      }));

      await this.collection.add({
        ids,
        documents,
        metadatas
      });

    } catch (error) {
      console.error('Failed to bulk add patterns:', error);
    }
  }

  async getCollectionStats(): Promise<Record<string, any>> {
    if (!this.enabled || !this.collection) {
      return {
        enabled: false,
        total_patterns: 0
      };
    }

    try {
      const count = await this.collection.count();
      
      return {
        enabled: true,
        total_patterns: count,
        collection_name: 'neural_semantic_patterns',
        config: {
          model_name: this.config.modelName,
          similarity_threshold: this.config.similarityThreshold,
          max_results: this.config.maxResults
        }
      };
    } catch (error) {
      console.error('Failed to get collection stats:', error);
      return {
        enabled: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  async clearCollection(): Promise<void> {
    if (!this.enabled || !this.chromaClient) {
      return;
    }

    try {
      await this.chromaClient.deleteCollection({
        name: 'neural_semantic_patterns'
      });

      // Recreate empty collection
      this.collection = await this.chromaClient.createCollection({
        name: 'neural_semantic_patterns',
        metadata: { description: 'Neural Semantic Compiler patterns' }
      });

    } catch (error) {
      console.error('Failed to clear collection:', error);
    }
  }
}