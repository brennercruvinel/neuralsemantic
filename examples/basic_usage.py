#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Semantic Compiler - Basic Usage Examples

This script demonstrates the basic functionality of the Neural Semantic Compiler.
"""

from neuralsemantic import NeuralSemanticCompiler, CompressionLevel

def main():
    print(" Neural Semantic Compiler - Basic Usage Examples\n")
    
    # Create compiler with default configuration
    print("Initializing Neural Semantic Compiler...")
    compiler = NeuralSemanticCompiler.create_default()
    
    try:
        # Example 1: Basic compression
        print("\n Example 1: Basic Compression")
        text1 = "Build a production-ready React application with user authentication"
        result1 = compiler.compress(text1)
        
        print(f"Original:   {text1}")
        print(f"Compressed: {result1.compressed_text}")
        print(f"Reduction:  {result1.savings_percentage:.1f}% ({result1.token_savings} tokens saved)")
        print(f"Quality:    {result1.quality_score:.1f}/10")
        
        # Example 2: Domain-specific compression
        print("\n Example 2: Web Development Domain")
        web_text = "Create React components with TypeScript interfaces and Redux state management"
        web_result = compiler.compress(web_text, domain="web-development")
        
        print(f"Original:   {web_text}")
        print(f"Compressed: {web_result.compressed_text}")
        print(f"Reduction:  {web_result.savings_percentage:.1f}%")
        
        # Example 3: Agile domain
        print("\n Example 3: Agile Domain")
        agile_text = "Sprint planning meeting with product owner and scrum master to review user stories"
        agile_result = compiler.compress(agile_text, domain="agile")
        
        print(f"Original:   {agile_text}")
        print(f"Compressed: {agile_result.compressed_text}")
        print(f"Reduction:  {agile_result.savings_percentage:.1f}%")
        
        # Example 4: Different compression levels
        print("\n Example 4: Compression Levels")
        test_text = "Implement comprehensive user authentication and authorization system with role-based access control"
        
        levels = [
            (CompressionLevel.LIGHT, "Light"),
            (CompressionLevel.BALANCED, "Balanced"), 
            (CompressionLevel.AGGRESSIVE, "Aggressive")
        ]
        
        for level, name in levels:
            result = compiler.compress(test_text, level=level)
            print(f"{name:>10}: {result.compressed_text} ({result.savings_percentage:.1f}% reduction)")
        
        # Example 5: Adding custom patterns
        print("\n Example 5: Custom Patterns")
        print("Adding custom pattern: 'artificial intelligence' → 'AI'")
        
        success = compiler.add_pattern(
            original="artificial intelligence",
            compressed="AI",
            pattern_type="compound",
            domain="ai",
            priority=900
        )
        
        if success:
            ai_text = "Train artificial intelligence models using machine learning algorithms"
            ai_result = compiler.compress(ai_text, domain="ai")
            print(f"Original:   {ai_text}")
            print(f"Compressed: {ai_result.compressed_text}")
        
        # Example 6: Cost savings estimation
        print("\n Example 6: Cost Savings")
        large_text = """
        Build a comprehensive e-commerce platform with the following features:
        - User registration and authentication system
        - Product catalog with search and filtering capabilities
        - Shopping cart and checkout process with payment integration
        - Order management and tracking system
        - Admin dashboard for inventory management
        - Real-time notifications and email alerts
        - Responsive design for mobile and desktop devices
        - Performance optimization and caching mechanisms
        """
        
        cost_result = compiler.compress(large_text)
        
        # Estimate cost savings (rough calculation based on GPT-4 pricing)
        cost_per_1k_tokens = 0.03  # $0.03 per 1K input tokens
        estimated_savings = (cost_result.token_savings / 1000) * cost_per_1k_tokens
        
        print(f"Original tokens: {cost_result.original_tokens}")
        print(f"Compressed tokens: {cost_result.compressed_tokens}")
        print(f"Token savings: {cost_result.token_savings}")
        print(f"Estimated cost savings: ${estimated_savings:.4f} per prompt")
        print(f"Monthly savings (100 prompts/day): ${estimated_savings * 100 * 30:.2f}")
        
        # Example 7: Session statistics
        print("\n Example 7: Session Statistics")
        stats = compiler.get_statistics()
        session_stats = stats['session']
        
        print(f"Total compressions: {session_stats['compressions']}")
        print(f"Total tokens saved: {session_stats['total_savings']}")
        print(f"Average compression: {session_stats.get('average_compression', 0):.1%}")
        
        # Example 8: Engine comparison
        print("\n Example 8: Engine Comparison")
        comparison_text = "Implement microservices architecture with Docker containers and Kubernetes orchestration"
        engine_comparison = compiler.compare_engines(comparison_text)
        
        print("Engine performance comparison:")
        for engine_name, result in engine_comparison.items():
            if 'error' not in result:
                print(f"  {engine_name:>12}: {result['compression_ratio']:.1%} compression, "
                      f"{result['quality_score']:.1f}/10 quality")
        
        # Example 9: Compression explanation
        print("\n Example 9: Compression Explanation")
        explain_text = "Build production-ready React application"
        explanation = compiler.explain_compression(explain_text)
        
        print(f"Engine used: {explanation['engine']['name']}")
        print(f"Processing time: {explanation['engine']['processingTimeMs']}ms")
        print(f"Patterns applied: {len(explanation['patternsApplied'])}")
        
        if explanation['patternsApplied']:
            print("Top patterns:")
            for i, pattern in enumerate(explanation['patternsApplied'][:3]):
                print(f"  {i+1}. '{pattern['original']}' → '{pattern['compressed']}'")
        
        # Generate final report
        print("\n Final Session Report:")
        print(compiler.get_session_report())
        
    except Exception as e:
        print(f" Error: {e}")
    
    finally:
        # Always clean up
        print("\nCleaning up...")
        compiler.close()
        print(" Done!")

if __name__ == "__main__":
    main()