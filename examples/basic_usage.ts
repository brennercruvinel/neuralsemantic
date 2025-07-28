/**
 * Neural Semantic Compiler - TypeScript Usage Examples
 * 
 * This script demonstrates the basic functionality of the Neural Semantic Compiler
 * in a TypeScript/Node.js environment.
 */

import { NeuralSemanticCompiler } from '@neurosemantic/core';
import { CompressionLevel, PatternType } from '@neurosemantic/types';

async function main(): Promise<void> {
  console.log('Neural Semantic Compiler - TypeScript Usage Examples\n');

  // Create compiler with default configuration
  console.log('Initializing Neural Semantic Compiler...');
  const compiler = NeuralSemanticCompiler.createDefault();

  try {
    // Example 1: Basic compression
    console.log('\n Example 1: Basic Compression');
    const text1 = 'Build a production-ready React application with user authentication';
    const result1 = await compiler.compress(text1);

    console.log(`Original:   ${text1}`);
    console.log(`Compressed: ${result1.compressedText}`);
    console.log(`Reduction:  ${((1 - result1.compressionRatio) * 100).toFixed(1)}% (${result1.originalTokens - result1.compressedTokens} tokens saved)`);
    console.log(`Quality:    ${result1.qualityScore.toFixed(1)}/10`);

    // Example 2: Domain-specific compression
    console.log('\n Example 2: Web Development Domain');
    const webText = 'Create React components with TypeScript interfaces and Redux state management';
    const webResult = await compiler.compress(webText, { domain: 'web-development' });

    console.log(`Original:   ${webText}`);
    console.log(`Compressed: ${webResult.compressedText}`);
    console.log(`Reduction:  ${((1 - webResult.compressionRatio) * 100).toFixed(1)}%`);

    // Example 3: Agile domain
    console.log('\n Example 3: Agile Domain');
    const agileText = 'Sprint planning meeting with product owner and scrum master to review user stories';
    const agileResult = await compiler.compress(agileText, { domain: 'agile' });

    console.log(`Original:   ${agileText}`);
    console.log(`Compressed: ${agileResult.compressedText}`);
    console.log(`Reduction:  ${((1 - agileResult.compressionRatio) * 100).toFixed(1)}%`);

    // Example 4: Different compression levels
    console.log('\n Example 4: Compression Levels');
    const testText = 'Implement comprehensive user authentication and authorization system with role-based access control';

    const levels: Array<[CompressionLevel, string]> = [
      [CompressionLevel.LIGHT, 'Light'],
      [CompressionLevel.BALANCED, 'Balanced'],
      [CompressionLevel.AGGRESSIVE, 'Aggressive']
    ];

    for (const [level, name] of levels) {
      const result = await compiler.compress(testText, { level });
      const reduction = ((1 - result.compressionRatio) * 100).toFixed(1);
      console.log(`${name.padStart(10)}: ${result.compressedText} (${reduction}% reduction)`);
    }

    // Example 5: Adding custom patterns
    console.log('\n Example 5: Custom Patterns');
    console.log("Adding custom pattern: 'artificial intelligence' → 'AI'");

    const success = await compiler.addPattern(
      'artificial intelligence',
      'AI',
      {
        patternType: PatternType.COMPOUND,
        domain: 'ai',
        priority: 900
      }
    );

    if (success) {
      const aiText = 'Train artificial intelligence models using machine learning algorithms';
      const aiResult = await compiler.compress(aiText, { domain: 'ai' });
      console.log(`Original:   ${aiText}`);
      console.log(`Compressed: ${aiResult.compressedText}`);
    }

    // Example 6: Batch processing
    console.log('\n Example 6: Batch Processing');
    const texts = [
      'Build React application',
      'Implement user authentication',
      'Create database schema',
      'Setup CI/CD pipeline',
      'Deploy to production'
    ];

    const batchResults = await compiler.compressBatch(texts);
    console.log(`Processed ${batchResults.length} texts:`);

    batchResults.forEach((result, index) => {
      const reduction = ((1 - result.compressionRatio) * 100).toFixed(1);
      console.log(`  ${index + 1}. ${result.compressedText} (${reduction}% reduction)`);
    });

    // Example 7: Cost savings estimation
    console.log('\n Example 7: Cost Savings');
    const largeText = `
      Build a comprehensive e-commerce platform with the following features:
      - User registration and authentication system
      - Product catalog with search and filtering capabilities
      - Shopping cart and checkout process with payment integration
      - Order management and tracking system
      - Admin dashboard for inventory management
      - Real-time notifications and email alerts
      - Responsive design for mobile and desktop devices
      - Performance optimization and caching mechanisms
    `;

    const costResult = await compiler.compress(largeText);

    // Estimate cost savings (rough calculation based on GPT-4 pricing)
    const costPer1kTokens = 0.03; // $0.03 per 1K input tokens
    const tokenSavings = costResult.originalTokens - costResult.compressedTokens;
    const estimatedSavings = (tokenSavings / 1000) * costPer1kTokens;

    console.log(`Original tokens: ${costResult.originalTokens}`);
    console.log(`Compressed tokens: ${costResult.compressedTokens}`);
    console.log(`Token savings: ${tokenSavings}`);
    console.log(`Estimated cost savings: $${estimatedSavings.toFixed(4)} per prompt`);
    console.log(`Monthly savings (100 prompts/day): $${(estimatedSavings * 100 * 30).toFixed(2)}`);

    // Example 8: Engine comparison
    console.log('\n Example 8: Engine Comparison');
    const comparisonText = 'Implement microservices architecture with Docker containers and Kubernetes orchestration';
    const engineComparison = await compiler.compareEngines(comparisonText);

    console.log('Engine performance comparison:');
    Object.entries(engineComparison).forEach(([engineName, result]) => {
      if (!result.error) {
        const compression = (result.compressionRatio! * 100).toFixed(1);
        const quality = result.qualityScore!.toFixed(1);
        console.log(`  ${engineName.padStart(12)}: ${compression}% compression, ${quality}/10 quality`);
      }
    });

    // Example 9: Compression explanation
    console.log('\n Example 9: Compression Explanation');
    const explainText = 'Build production-ready React application';
    const explanation = await compiler.explainCompression(explainText);

    console.log(`Engine used: ${explanation.engine.name}`);
    console.log(`Processing time: ${explanation.engine.processingTimeMs}ms`);
    console.log(`Patterns applied: ${explanation.patternsApplied.length}`);

    if (explanation.patternsApplied.length > 0) {
      console.log('Top patterns:');
      explanation.patternsApplied.slice(0, 3).forEach((pattern, index) => {
        console.log(`  ${index + 1}. '${pattern.original}' → '${pattern.compressed}'`);
      });
    }

    // Example 10: Session statistics
    console.log('\n Example 10: Session Statistics');
    const stats = await compiler.getStatistics();
    const sessionStats = stats.session;

    console.log(`Total compressions: ${sessionStats.compressions}`);
    console.log(`Total tokens saved: ${sessionStats.totalSavings}`);
    console.log(`Average compression: ${(sessionStats.averageCompression * 100).toFixed(1)}%`);
    console.log(`Session duration: ${sessionStats.durationMinutes.toFixed(1)} minutes`);

    // Example 11: System health check
    console.log('\n Example 11: System Health Check');
    const health = await compiler.validateSystemHealth();
    console.log(`Overall status: ${health.overallStatus}`);
    console.log(`Pattern manager: ${health.components.patternManager.status}`);
    console.log(`Engines: ${health.components.engines.overallStatus}`);

    if (health.issues.length > 0) {
      console.log('Issues found:');
      health.issues.forEach(issue => console.log(`  - ${issue}`));
    }

    // Generate final report
    console.log('\n Final Session Report:');
    console.log(compiler.getSessionReport());

  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : error);
  } finally {
    // Always clean up
    console.log('\nCleaning up...');
    await compiler.close();
    console.log('Done!');
  }
}

// Export for module usage
export { main };

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}