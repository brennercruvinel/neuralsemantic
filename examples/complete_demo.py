#!/usr/bin/env python3
"""
Teste completo do Neural Semantic Compiler
Demonstra todas as funcionalidades principais
"""

from neuralsemantic import NeuralSemanticCompiler, CompressionLevel
import json
from datetime import datetime

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def main():
    print(" TESTE COMPLETO DO NEURAL SEMANTIC COMPILER")
    print(f" Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Inicializar compilador
    print("Inicializando compilador...")
    compiler = NeuralSemanticCompiler.create_default()
    
    # Lista de textos para testar
    test_cases = [
        {
            "name": "Web Development",
            "text": "Create a React application with TypeScript, Redux for state management, and Material-UI components",
            "domain": "web-development"
        },
        {
            "name": "Agile/Scrum",
            "text": "Daily standup meeting with product owner and scrum master to discuss sprint backlog items",
            "domain": "agile"
        },
        {
            "name": "DevOps",
            "text": "Deploy microservices to Kubernetes cluster with Docker containers and configure load balancer",
            "domain": "devops"
        },
        {
            "name": "Database",
            "text": "Create database schema with foreign key constraints and indexes for performance optimization",
            "domain": "general"
        },
        {
            "name": "AI/ML",
            "text": "Train machine learning model using neural networks with TensorFlow and evaluate accuracy metrics",
            "domain": "ai"
        }
    ]
    
    # Resultados para relatório
    results_summary = []
    
    # Testar cada caso
    for i, test_case in enumerate(test_cases, 1):
        print_separator(f"TESTE {i}: {test_case['name']}")
        
        text = test_case['text']
        domain = test_case['domain']
        
        print(f" Texto original ({len(text)} chars):")
        print(f"   '{text}'")
        
        # Testar todos os níveis de compressão
        print("\n Testando níveis de compressão:")
        
        level_results = {}
        for level in [CompressionLevel.LIGHT, CompressionLevel.BALANCED, CompressionLevel.AGGRESSIVE]:
            result = compiler.compress(text, level=level, domain=domain)
            
            level_results[level.value] = {
                "compressed": result.compressed_text,
                "reduction": result.savings_percentage,
                "tokens_saved": result.token_savings,
                "quality": result.quality_score
            }
            
            print(f"\n   {level.value.upper()}:")
            print(f"   → '{result.compressed_text}'")
            print(f"   → Redução: {result.savings_percentage:.1f}% | Tokens: {result.token_savings} | Qualidade: {result.quality_score:.1f}/10")
        
        # Adicionar ao resumo
        results_summary.append({
            "test_case": test_case['name'],
            "original_length": len(text),
            "domain": domain,
            "results": level_results
        })
    
    # Teste de padrões customizados
    print_separator("TESTE DE PADRÕES CUSTOMIZADOS")
    
    # Adicionar alguns padrões
    custom_patterns = [
        ("application programming interface", "API", "compound"),
        ("continuous integration", "CI", "compound"),
        ("continuous deployment", "CD", "compound"),
        ("infrastructure as code", "IaC", "compound")
    ]
    
    print("Adicionando padrões customizados:")
    for original, compressed, pattern_type in custom_patterns:
        success = compiler.add_pattern(
            original=original,
            compressed=compressed,
            pattern_type=pattern_type,
            domain="devops",
            priority=950
        )
        print(f"  • '{original}' → '{compressed}' {'✓' if success else '✗'}")
    
    # Testar com os novos padrões
    print("\nTestando com padrões customizados:")
    test_text = "Setup continuous integration and continuous deployment pipeline with infrastructure as code"
    result = compiler.compress(test_text, domain="devops")
    
    print(f"\nOriginal:   '{test_text}'")
    print(f"Comprimido: '{result.compressed_text}'")
    print(f"Redução:    {result.savings_percentage:.1f}% ({result.token_savings} tokens)")
    
    # Estatísticas da sessão
    print_separator("ESTATÍSTICAS DA SESSÃO")
    
    stats = compiler.get_statistics()
    print(f"Total de compressões: {stats['total']['compressions']}")
    print(f"Tokens economizados:  {stats['total']['total_savings']}")
    print(f"Taxa média:           {stats['total'].get('average_compression', 0):.1%}")
    
    # Comparação de engines
    print_separator("COMPARAÇÃO DE ENGINES")
    
    comparison_text = "Implement microservices architecture with Docker containers and Kubernetes orchestration"
    comparison = compiler.compare_engines(comparison_text)
    
    print("Desempenho por engine:")
    for engine_name, result in comparison.items():
        if 'error' not in result:
            print(f"  • {engine_name:>12}: {result['compression_ratio']:.1%} compressão, "
                  f"{result['quality_score']:.1f}/10 qualidade, {result['processing_time_ms']}ms")
    
    # Salvar relatório JSON
    print_separator("SALVANDO RELATÓRIO")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_results": results_summary,
        "session_stats": stats,
        "engine_comparison": comparison
    }
    
    with open("neuralsemantic_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(" Relatório salvo em: neuralsemantic_test_report.json")
    
    # Gerar relatório de sessão
    session_report = compiler.get_session_report()
    
    with open("neuralsemantic_session_report.txt", "w") as f:
        f.write(session_report)
    
    print(" Relatório de sessão salvo em: neuralsemantic_session_report.txt")
    
    print("\n TESTE COMPLETO FINALIZADO!")

if __name__ == "__main__":
    main()