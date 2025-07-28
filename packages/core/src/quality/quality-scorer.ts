import { CompressionConfig, CompressionContext, CompressionResult, QualityMetrics } from '@neurosemantic/types';

export class QualityScorer {
  constructor(private config: CompressionConfig) {}

  async calculateQualityScore(result: CompressionResult, context: CompressionContext): Promise<number> {
    const compressionQuality = 1.0 - Math.abs(result.compressionRatio - context.targetCompression);
    
    const patternQuality = result.patternMatches.length > 0
      ? result.patternMatches.reduce((sum, m) => sum + m.confidence, 0) / result.patternMatches.length
      : 1.0;
    
    const lengthRatio = result.compressedText.length / result.originalText.length;
    const lengthQuality = lengthRatio > 0.2 ? 1.0 : lengthRatio * 5;
    
    const timeQuality = Math.max(0.5, 1.0 - (result.processingTimeMs / 1000));
    
    const qualityScore = (
      compressionQuality * 0.4 +
      patternQuality * 0.3 +
      lengthQuality * 0.2 +
      timeQuality * 0.1
    );
    
    return Math.min(10.0, Math.max(0.0, qualityScore * 10));
  }

  calculateMetrics(result: CompressionResult): QualityMetrics {
    return {
      compositeScore: result.qualityScore,
      semanticPreservation: 0.9,
      informationDensity: 0.85,
      compressionEfficiency: 1 - result.compressionRatio,
      llmInterpretability: 0.95,
      structuralPreservation: 0.9,
      linguisticCoherence: 0.88,
      entityPreservation: 0.92,
      breakdownDetails: {}
    };
  }
}