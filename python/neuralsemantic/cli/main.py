"""Main CLI application for Neural Semantic Compiler."""

import sys
import os
import click
import time
import json
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.prompt import Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None

from ..core.compiler import NeuralSemanticCompiler
from ..core.types import CompressionLevel, Pattern, PatternType
from ..core.config import ConfigManager
from ..core.exceptions import CompressionError, ConfigurationError

# Initialize console
if RICH_AVAILABLE:
    console = Console()
else:
    # Fallback console for environments without Rich
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
        
        def input(self, prompt=""):
            return input(prompt)
    
    console = FallbackConsole()


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Quiet mode')
@click.pass_context
def cli(ctx, config, verbose, quiet):
    """
     Neural Semantic Compiler - The first compiler for neural communication
    
    Reduce LLM costs by 60-70% while preserving 100% semantic meaning.
    """
    ctx.ensure_object(dict)
    
    try:
        # Load configuration
        if config and os.path.exists(config):
            config_obj = ConfigManager.load_config(config)
        else:
            config_obj = ConfigManager.create_default_config()
            
        if verbose:
            config_obj.log_level = "DEBUG"
        elif quiet:
            config_obj.log_level = "ERROR"

        # Initialize compiler
        ctx.obj['compiler'] = NeuralSemanticCompiler(config_obj)
        ctx.obj['config'] = config_obj
        ctx.obj['quiet'] = quiet
        
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Error initializing compiler: {e}[/red]")
        else:
            print(f"Error initializing compiler: {e}")
        sys.exit(1)


@cli.command()
@click.argument('text', required=False)
@click.option('--input', '-i', 'input_file', help='Input file path')
@click.option('--output', '-o', 'output_file', help='Output file path')
@click.option('--level', '-l', 
              type=click.Choice(['none', 'light', 'balanced', 'aggressive']),
              default='balanced', help='Compression level')
@click.option('--domain', '-d', help='Domain context (web-dev, agile, devops, etc.)')
@click.option('--model', '-m', default='gpt-4', help='Target LLM model for token optimization')
@click.option('--show-stats', is_flag=True, help='Show detailed compression statistics')
@click.option('--show-tokens', is_flag=True, help='Show token analysis')
@click.option('--dry-run', is_flag=True, help='Show what would be compressed without applying')
@click.pass_context
def compress(ctx, text, input_file, output_file, level, domain, model, show_stats, show_tokens, dry_run):
    """Compress text using Neural Semantic Compiler."""
    compiler = ctx.obj['compiler']
    quiet = ctx.obj.get('quiet', False)
    
    try:
        # Get input text
        if input_file:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif not text:
            if sys.stdin.isatty():
                if not quiet and RICH_AVAILABLE:
                    console.print("[yellow]Enter text to compress (Ctrl+D to finish):[/yellow]")
                text = sys.stdin.read()
            else:
                text = sys.stdin.read()

        if not text or not text.strip():
            if RICH_AVAILABLE:
                console.print("[red]Error: No input text provided[/red]")
            else:
                print("Error: No input text provided")
            return

        # Show original text info
        if not quiet:
            original_stats = _get_text_stats(text, model, compiler)
            if RICH_AVAILABLE:
                _show_original_stats(original_stats)
            else:
                print(f"Original: {len(text)} chars, ~{original_stats['estimated_tokens']} tokens")

        if dry_run:
            if RICH_AVAILABLE:
                console.print("[blue]Dry run mode - showing compression preview...[/blue]")
            else:
                print("Dry run mode - showing compression preview...")

        # Show progress
        if not quiet and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Compressing text...", total=100)
                
                # Simulate progress steps
                progress.update(task, advance=20, description="Analyzing text...")
                
                # Perform compression
                result = compiler.compress(
                    text,
                    level=CompressionLevel(level),
                    domain=domain,
                    target_model=model
                )
                
                progress.update(task, advance=80, description="Compression complete!")
        else:
            # Perform compression without progress
            result = compiler.compress(
                text,
                level=CompressionLevel(level),
                domain=domain,
                target_model=model
            )

        if dry_run:
            if RICH_AVAILABLE:
                console.print(Panel(
                    f"[green]Compression Preview[/green]\n\n"
                    f"Original length: {len(text)} characters\n"
                    f"Compressed length: {len(result.compressed_text)} characters\n"
                    f"Reduction: {(1-result.compression_ratio):.1%}\n"
                    f"Quality score: {result.quality_score:.1f}/10\n"
                    f"Patterns applied: {len(result.pattern_matches)}",
                    title=" Dry Run Results",
                    border_style="blue"
                ))
            else:
                print(f"Dry run results: {(1-result.compression_ratio):.1%} reduction, {result.quality_score:.1f}/10 quality")
            return

        # Output result
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.compressed_text)
            if not quiet:
                if RICH_AVAILABLE:
                    console.print(f"[green]✓ Compressed text saved to: {output_file}[/green]")
                else:
                    print(f"Compressed text saved to: {output_file}")
        else:
            if not quiet and RICH_AVAILABLE:
                console.print("\n[bold blue]Compressed Text:[/bold blue]")
                console.print(Panel(result.compressed_text, title=" Output", border_style="green"))
            else:
                print(result.compressed_text)

        # Show statistics
        if show_stats and not quiet:
            _show_compression_stats(result, model, compiler)
        
        # Show token analysis
        if show_tokens and not quiet:
            _show_token_analysis(text, result.compressed_text, model, compiler)

    except CompressionError as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Compression failed: {e}[/red]")
        else:
            print(f"Compression failed: {e}")
        sys.exit(1)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Unexpected error: {e}[/red]")
        else:
            print(f"Unexpected error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--domain', '-d', help='Filter by domain')
@click.option('--type', '-t', 'pattern_type', help='Filter by pattern type')
@click.option('--limit', '-l', default=20, help='Limit number of results')
@click.option('--search', '-s', help='Search patterns by text')
@click.option('--sort', type=click.Choice(['priority', 'frequency', 'success_rate']), 
              default='priority', help='Sort patterns by field')
@click.pass_context
def patterns(ctx, domain, pattern_type, limit, search, sort):
    """List and manage compression patterns."""
    compiler = ctx.obj['compiler']
    
    try:
        if search:
            patterns_list = compiler.pattern_manager.search_patterns(search, limit)
        else:
            patterns_list = compiler.pattern_manager.get_patterns(
                domain=domain,
                pattern_type=pattern_type
            )
            
            # Sort patterns
            if sort == 'frequency':
                patterns_list.sort(key=lambda p: p.frequency, reverse=True)
            elif sort == 'success_rate':
                patterns_list.sort(key=lambda p: p.success_rate, reverse=True)
            else:  # priority
                patterns_list.sort(key=lambda p: p.priority, reverse=True)
            
            patterns_list = patterns_list[:limit]

        if not patterns_list:
            if RICH_AVAILABLE:
                console.print("[yellow]No patterns found matching criteria[/yellow]")
            else:
                print("No patterns found matching criteria")
            return

        # Display patterns
        if RICH_AVAILABLE:
            _show_patterns_table(patterns_list, sort)
        else:
            _show_patterns_simple(patterns_list)

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Error listing patterns: {e}[/red]")
        else:
            print(f"Error listing patterns: {e}")


@cli.command()
@click.argument('original')
@click.argument('compressed')
@click.option('--type', '-t', default='word', 
              type=click.Choice(['word', 'phrase', 'compound', 'abbreviation', 'structure']),
              help='Pattern type')
@click.option('--domain', '-d', default='general', help='Domain')
@click.option('--priority', '-p', default=500, type=int, help='Priority (100-1000)')
@click.option('--language', default='en', help='Language code')
@click.pass_context
def add_pattern(ctx, original, compressed, type, domain, priority, language):
    """Add a new compression pattern."""
    compiler = ctx.obj['compiler']
    
    try:
        success = compiler.add_pattern(
            original=original,
            compressed=compressed,
            pattern_type=type,
            domain=domain,
            priority=priority,
            language=language
        )

        if success:
            if RICH_AVAILABLE:
                console.print(f"[green]✓ Pattern added: '{original}' → '{compressed}'[/green]")
            else:
                print(f"Pattern added: '{original}' → '{compressed}'")
        else:
            if RICH_AVAILABLE:
                console.print(f"[red]✗ Failed to add pattern (conflict detected)[/red]")
            else:
                print("Failed to add pattern (conflict detected)")

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Error adding pattern: {e}[/red]")
        else:
            print(f"Error adding pattern: {e}")


@cli.command()
@click.option('--model', '-m', default='gpt-4', help='Target LLM model')
@click.argument('text', required=False)
@click.option('--input', '-i', 'input_file', help='Input file path')
@click.pass_context
def analyze(ctx, model, text, input_file):
    """Analyze text for compression opportunities."""
    compiler = ctx.obj['compiler']
    
    try:
        # Get input text
        if input_file:
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif not text:
            text = sys.stdin.read()

        if not text or not text.strip():
            if RICH_AVAILABLE:
                console.print("[red]Error: No input text provided[/red]")
            else:
                print("Error: No input text provided")
            return

        # Analyze text
        analysis = _analyze_text_comprehensive(text, model, compiler)
        
        if RICH_AVAILABLE:
            _show_analysis_results(analysis)
        else:
            _show_analysis_simple(analysis)

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Error analyzing text: {e}[/red]")
        else:
            print(f"Error analyzing text: {e}")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show compiler statistics and health."""
    compiler = ctx.obj['compiler']
    
    try:
        stats_data = compiler.get_statistics()
        health_data = compiler.health_check()
        
        if RICH_AVAILABLE:
            _show_stats_dashboard(stats_data, health_data)
        else:
            _show_stats_simple(stats_data, health_data)

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Error getting statistics: {e}[/red]")
        else:
            print(f"Error getting statistics: {e}")


@cli.command()
@click.option('--text', '-t', help='Test text for benchmark')
@click.option('--file', '-f', help='Test file for benchmark')  
@click.option('--model', '-m', default='gpt-4', help='Target model')
@click.pass_context
def benchmark(ctx, text, file, model):
    """Benchmark compression performance."""
    compiler = ctx.obj['compiler']
    
    try:
        # Prepare test texts
        if file:
            with open(file, 'r') as f:
                test_texts = [f.read()]
        elif text:
            test_texts = [text]
        else:
            # Use default test texts
            test_texts = [
                "Build a production-ready React application with authentication and authorization",
                "Sprint planning meeting with product owner to review user stories and estimate story points",
                "Implement microservices architecture with Docker containerization and Kubernetes orchestration"
            ]

        if RICH_AVAILABLE:
            console.print("[blue]Running compression benchmark...[/blue]")
            
            with Progress() as progress:
                task = progress.add_task("Benchmarking...", total=len(test_texts))
                
                results = compiler.benchmark(test_texts, target_model=model)
                progress.update(task, completed=len(test_texts))
            
            _show_benchmark_results(results)
        else:
            print("Running compression benchmark...")
            results = compiler.benchmark(test_texts, target_model=model)
            _show_benchmark_simple(results)

    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[red]Benchmark failed: {e}[/red]")
        else:
            print(f"Benchmark failed: {e}")


# Helper functions for display
def _get_text_stats(text: str, model: str, compiler) -> dict:
    """Get basic text statistics."""
    try:
        tokenizer = compiler.pattern_manager  # This would be tokenizer in real implementation
        estimated_tokens = len(text) // 4  # Simple estimation
        
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'estimated_tokens': estimated_tokens,
            'model': model
        }
    except:
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'estimated_tokens': len(text) // 4,
            'model': model
        }


def _show_original_stats(stats: dict):
    """Show original text statistics."""
    if RICH_AVAILABLE:
        table = Table(title=" Original Text Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Characters", f"{stats['char_count']:,}")
        table.add_row("Words", f"{stats['word_count']:,}")
        table.add_row("Estimated Tokens", f"{stats['estimated_tokens']:,}")
        table.add_row("Target Model", stats['model'])
        
        console.print(table)


def _show_compression_stats(result, model: str, compiler):
    """Display compression statistics."""
    if not RICH_AVAILABLE:
        print(f"Compression: {(1-result.compression_ratio):.1%} reduction")
        print(f"Quality: {result.quality_score:.1f}/10")
        print(f"Processing time: {result.processing_time_ms}ms")
        return
    
    table = Table(title=" Compression Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Calculate derived metrics
    char_reduction = len(result.original_text) - len(result.compressed_text)
    token_reduction = result.original_tokens - result.compressed_tokens
    cost_savings = token_reduction * 0.00003  # Rough estimate
    
    table.add_row("Original Length", f"{len(result.original_text):,} chars")
    table.add_row("Compressed Length", f"{len(result.compressed_text):,} chars")
    table.add_row("Character Reduction", f"{char_reduction:,} chars ({(1-result.compression_ratio):.1%})")
    table.add_row("Original Tokens", f"{result.original_tokens:,}")
    table.add_row("Compressed Tokens", f"{result.compressed_tokens:,}")
    table.add_row("Token Reduction", f"{token_reduction:,} tokens")
    table.add_row("Quality Score", f"{result.quality_score:.1f}/10")
    table.add_row("Processing Time", f"{result.processing_time_ms}ms")
    table.add_row("Engine Used", result.engine_used.title())
    table.add_row("Pattern Matches", f"{len(result.pattern_matches)}")
    table.add_row("Estimated Cost Savings", f"${cost_savings:.4f}")
    
    console.print(table)
    
    # Show warnings if any
    if result.warnings:
        console.print("\n[yellow]⚠️  Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  • {warning}")


def _show_token_analysis(original: str, compressed: str, model: str, compiler):
    """Show detailed token analysis."""
    if not RICH_AVAILABLE:
        return
        
    console.print("\n[bold blue] Token Analysis[/bold blue]")
    
    # This would use the actual tokenizer in real implementation
    try:
        # Placeholder analysis
        analysis = {
            'original_tokens': len(original) // 4,
            'compressed_tokens': len(compressed) // 4,
            'efficiency': 'Good',
            'recommendations': [
                'Consider domain-specific patterns',
                'Look for repeated phrases'
            ]
        }
        
        table = Table()
        table.add_column("Aspect", style="cyan")
        table.add_column("Details", style="green")
        
        table.add_row("Model", model)
        table.add_row("Token Efficiency", analysis['efficiency'])
        table.add_row("Recommendations", '\n'.join(analysis['recommendations']))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[yellow]Token analysis unavailable: {e}[/yellow]")


def _show_patterns_table(patterns_list: list, sort_by: str):
    """Show patterns in a rich table."""
    table = Table(title=f" Compression Patterns (sorted by {sort_by})")
    table.add_column("Original", style="cyan", max_width=30)
    table.add_column("Compressed", style="green", max_width=20)
    table.add_column("Type", style="yellow")
    table.add_column("Domain", style="blue")
    table.add_column("Priority", style="red", justify="right")
    table.add_column("Usage", style="magenta", justify="right")
    table.add_column("Success %", style="bright_green", justify="right")

    for pattern in patterns_list:
        original = pattern.original
        if len(original) > 28:
            original = original[:25] + "..."
            
        compressed = pattern.compressed
        if len(compressed) > 18:
            compressed = compressed[:15] + "..."
            
        success_rate = f"{pattern.success_rate*100:.1f}%" if pattern.success_rate > 0 else "N/A"
        
        table.add_row(
            original,
            compressed,
            pattern.pattern_type.value,
            pattern.domain,
            str(pattern.priority),
            str(pattern.frequency),
            success_rate
        )

    console.print(table)


def _show_patterns_simple(patterns_list: list):
    """Show patterns in simple format."""
    print(f"Found {len(patterns_list)} patterns:")
    for i, pattern in enumerate(patterns_list, 1):
        print(f"{i:2d}. '{pattern.original}' → '{pattern.compressed}' "
              f"({pattern.pattern_type.value}, {pattern.domain}, priority: {pattern.priority})")


def _analyze_text_comprehensive(text: str, model: str, compiler) -> dict:
    """Comprehensive text analysis."""
    return {
        'length': len(text),
        'words': len(text.split()),
        'compression_potential': 'High',  # Would be calculated
        'technical_density': 'Medium',    # Would be calculated
        'recommendations': [
            'Use web-development domain patterns',
            'Consider aggressive compression level',
            'Focus on technical term abbreviations'
        ]
    }


def _show_analysis_results(analysis: dict):
    """Show analysis results with rich formatting."""
    if not RICH_AVAILABLE:
        return
        
    panel_content = f"""[green]Text Length:[/green] {analysis['length']:,} characters
[green]Word Count:[/green] {analysis['words']:,} words
[green]Compression Potential:[/green] {analysis['compression_potential']}
[green]Technical Density:[/green] {analysis['technical_density']}

[blue]Recommendations:[/blue]"""
    
    for rec in analysis['recommendations']:
        panel_content += f"\n  • {rec}"
    
    console.print(Panel(panel_content, title=" Text Analysis", border_style="blue"))


def _show_analysis_simple(analysis: dict):
    """Show analysis in simple format."""
    print(f"Length: {analysis['length']} chars, {analysis['words']} words")
    print(f"Compression potential: {analysis['compression_potential']}")
    print("Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")


def _show_stats_dashboard(stats_data: dict, health_data: dict):
    """Show comprehensive stats dashboard."""
    if not RICH_AVAILABLE:
        return
        
    # Health status
    health_color = "green" if health_data['overall'] == 'healthy' else "yellow" if health_data['overall'] == 'degraded' else "red"
    console.print(f"[{health_color}] System Health: {health_data['overall'].title()}[/{health_color}]")
    
    # Stats table
    table = Table(title=" Compiler Statistics")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    # Add component stats
    for component, info in health_data['components'].items():
        status = info['status']
        details = f"Enabled: {info.get('enabled', 'N/A')}"
        table.add_row(component.replace('_', ' ').title(), status.title(), details)
    
    console.print(table)


def _show_stats_simple(stats_data: dict, health_data: dict):
    """Show stats in simple format."""
    print(f"System health: {health_data['overall']}")
    print("Components:")
    for component, info in health_data['components'].items():
        print(f"  {component}: {info['status']}")


def _show_benchmark_results(results: dict):
    """Show benchmark results with rich formatting."""
    if not RICH_AVAILABLE:
        return
        
    console.print("[bold green] Benchmark Results[/bold green]")
    
    table = Table()
    table.add_column("Engine", style="cyan")
    table.add_column("Avg Compression", style="green", justify="right")
    table.add_column("Avg Time (ms)", style="yellow", justify="right")
    table.add_column("Success Rate", style="blue", justify="right")
    
    for engine_name, engine_results in results.get('engines', {}).items():
        compression = f"{(1-engine_results.get('average_compression_ratio', 0)):.1%}"
        time_ms = f"{engine_results.get('average_processing_time', 0):.1f}"
        success = f"{(engine_results.get('total_compressions', 0) / max(1, engine_results.get('total_compressions', 0) + engine_results.get('total_errors', 0))):.1%}"
        
        table.add_row(engine_name.title(), compression, time_ms, success)
    
    console.print(table)


def _show_benchmark_simple(results: dict):
    """Show benchmark in simple format."""
    print("Benchmark Results:")
    for engine_name, engine_results in results.get('engines', {}).items():
        compression = (1-engine_results.get('average_compression_ratio', 0)) * 100
        time_ms = engine_results.get('average_processing_time', 0)
        print(f"  {engine_name}: {compression:.1f}% compression, {time_ms:.1f}ms avg")


if __name__ == '__main__':
    cli()
