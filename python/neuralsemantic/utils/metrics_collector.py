"""Metrics collection and analytics for Neural Semantic Compiler."""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from ..core.types import CompressionResult, CompressionContext

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Individual compression metrics."""
    session_id: str
    timestamp: float
    original_length: int
    compressed_length: int
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    quality_score: float
    processing_time_ms: int
    engine_used: str
    domain: Optional[str]
    level: str
    pattern_matches_count: int
    success: bool


class MetricsCollector:
    """
    Collects and analyzes compression metrics for performance tracking.
    
    Tracks compression statistics, performance metrics, and usage patterns
    to provide insights and improve the system over time.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self.metrics: List[CompressionMetrics] = []
        self.session_start = time.time()
        self._load_persisted_metrics()

    def _load_persisted_metrics(self) -> None:
        """Load previously persisted metrics."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return
            
        try:
            with open(self.persist_path, 'r') as f:
                data = json.load(f)
                
            for item in data.get('metrics', []):
                metric = CompressionMetrics(**item)
                self.metrics.append(metric)
                
            logger.info(f"Loaded {len(self.metrics)} persisted metrics")
            
        except Exception as e:
            logger.warning(f"Failed to load persisted metrics: {e}")

    def record_compression(self, result: CompressionResult, 
                          context: CompressionContext) -> None:
        """Record a compression operation."""
        try:
            metric = CompressionMetrics(
                session_id=result.session_id or "unknown",
                timestamp=time.time(),
                original_length=len(result.original_text),
                compressed_length=len(result.compressed_text),
                original_tokens=result.original_tokens,
                compressed_tokens=result.compressed_tokens,
                compression_ratio=result.compression_ratio,
                quality_score=result.quality_score,
                processing_time_ms=result.processing_time_ms,
                engine_used=result.engine_used,
                domain=context.domain,
                level=context.level.value,
                pattern_matches_count=len(result.pattern_matches),
                success=len(result.warnings) == 0
            )
            
            self.metrics.append(metric)
            
            # Persist periodically
            if len(self.metrics) % 10 == 0:
                self._persist_metrics()
                
        except Exception as e:
            logger.error(f"Failed to record compression metrics: {e}")

    def _persist_metrics(self) -> None:
        """Persist metrics to disk."""
        if not self.persist_path:
            return
            
        try:
            # Ensure directory exists
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            data = {
                'session_start': self.session_start,
                'last_updated': time.time(),
                'metrics': [asdict(metric) for metric in self.metrics]
            }
            
            with open(self.persist_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        if not self.metrics:
            return {
                "total_compressions": 0,
                "session_duration_minutes": (time.time() - self.session_start) / 60
            }
        
        # Basic counts
        total_compressions = len(self.metrics)
        successful_compressions = sum(1 for m in self.metrics if m.success)
        
        # Token statistics
        total_input_tokens = sum(m.original_tokens for m in self.metrics)
        total_output_tokens = sum(m.compressed_tokens for m in self.metrics)
        total_token_savings = total_input_tokens - total_output_tokens
        
        # Compression ratios
        compression_ratios = [m.compression_ratio for m in self.metrics]
        avg_compression_ratio = sum(compression_ratios) / len(compression_ratios)
        
        # Quality scores
        quality_scores = [m.quality_score for m in self.metrics]
        avg_quality_score = sum(quality_scores) / len(quality_scores)
        
        # Performance metrics
        processing_times = [m.processing_time_ms for m in self.metrics]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Engine usage
        engine_usage = {}
        for metric in self.metrics:
            engine_usage[metric.engine_used] = engine_usage.get(metric.engine_used, 0) + 1
        
        # Domain usage
        domain_usage = {}
        for metric in self.metrics:
            domain = metric.domain or "unknown"
            domain_usage[domain] = domain_usage.get(domain, 0) + 1
        
        # Level usage
        level_usage = {}
        for metric in self.metrics:
            level_usage[metric.level] = level_usage.get(metric.level, 0) + 1
        
        return {
            "overview": {
                "total_compressions": total_compressions,
                "successful_compressions": successful_compressions,
                "success_rate": successful_compressions / total_compressions,
                "session_duration_minutes": (time.time() - self.session_start) / 60
            },
            "token_metrics": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_token_savings": total_token_savings,
                "average_savings_per_compression": total_token_savings / total_compressions,
                "overall_compression_ratio": total_output_tokens / total_input_tokens if total_input_tokens > 0 else 1.0
            },
            "quality_metrics": {
                "average_compression_ratio": avg_compression_ratio,
                "average_quality_score": avg_quality_score,
                "min_quality_score": min(quality_scores),
                "max_quality_score": max(quality_scores),
                "compression_efficiency": (1 - avg_compression_ratio) * avg_quality_score
            },
            "performance_metrics": {
                "average_processing_time_ms": avg_processing_time,
                "min_processing_time_ms": min(processing_times),
                "max_processing_time_ms": max(processing_times),
                "total_processing_time_ms": sum(processing_times)
            },
            "usage_patterns": {
                "engines": engine_usage,
                "domains": domain_usage,
                "levels": level_usage
            }
        }

    def get_time_series_data(self, interval_minutes: int = 5) -> Dict[str, List]:
        """Get time series data for visualization."""
        if not self.metrics:
            return {"timestamps": [], "compression_ratios": [], "quality_scores": []}
        
        # Sort by timestamp
        sorted_metrics = sorted(self.metrics, key=lambda m: m.timestamp)
        
        # Group by time intervals
        start_time = sorted_metrics[0].timestamp
        current_time = time.time()
        interval_seconds = interval_minutes * 60
        
        timestamps = []
        compression_ratios = []
        quality_scores = []
        token_savings = []
        
        current_interval_start = start_time
        while current_interval_start < current_time:
            interval_end = current_interval_start + interval_seconds
            
            # Find metrics in this interval
            interval_metrics = [
                m for m in sorted_metrics 
                if current_interval_start <= m.timestamp < interval_end
            ]
            
            if interval_metrics:
                avg_compression = sum(m.compression_ratio for m in interval_metrics) / len(interval_metrics)
                avg_quality = sum(m.quality_score for m in interval_metrics) / len(interval_metrics)
                total_savings = sum(m.original_tokens - m.compressed_tokens for m in interval_metrics)
                
                timestamps.append(current_interval_start)
                compression_ratios.append(avg_compression)
                quality_scores.append(avg_quality)
                token_savings.append(total_savings)
            
            current_interval_start = interval_end
        
        return {
            "timestamps": timestamps,
            "compression_ratios": compression_ratios,
            "quality_scores": quality_scores,
            "token_savings": token_savings
        }

    def get_engine_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare performance across different engines."""
        engine_stats = {}
        
        for metric in self.metrics:
            engine = metric.engine_used
            if engine not in engine_stats:
                engine_stats[engine] = {
                    "count": 0,
                    "compression_ratios": [],
                    "quality_scores": [],
                    "processing_times": [],
                    "token_savings": []
                }
            
            stats = engine_stats[engine]
            stats["count"] += 1
            stats["compression_ratios"].append(metric.compression_ratio)
            stats["quality_scores"].append(metric.quality_score)
            stats["processing_times"].append(metric.processing_time_ms)
            stats["token_savings"].append(metric.original_tokens - metric.compressed_tokens)
        
        # Calculate averages
        comparison = {}
        for engine, stats in engine_stats.items():
            if stats["count"] > 0:
                comparison[engine] = {
                    "usage_count": stats["count"],
                    "avg_compression_ratio": sum(stats["compression_ratios"]) / stats["count"],
                    "avg_quality_score": sum(stats["quality_scores"]) / stats["count"],
                    "avg_processing_time_ms": sum(stats["processing_times"]) / stats["count"],
                    "avg_token_savings": sum(stats["token_savings"]) / stats["count"],
                    "efficiency_score": (
                        (1 - sum(stats["compression_ratios"]) / stats["count"]) * 
                        (sum(stats["quality_scores"]) / stats["count"]) / 10
                    )
                }
        
        return comparison

    def get_domain_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Analyze compression performance by domain."""
        domain_stats = {}
        
        for metric in self.metrics:
            domain = metric.domain or "unknown"
            if domain not in domain_stats:
                domain_stats[domain] = {
                    "count": 0,
                    "compression_ratios": [],
                    "quality_scores": [],
                    "pattern_matches": []
                }
            
            stats = domain_stats[domain]
            stats["count"] += 1
            stats["compression_ratios"].append(metric.compression_ratio)
            stats["quality_scores"].append(metric.quality_score)
            stats["pattern_matches"].append(metric.pattern_matches_count)
        
        # Calculate domain-specific metrics
        analysis = {}
        for domain, stats in domain_stats.items():
            if stats["count"] > 0:
                analysis[domain] = {
                    "compression_count": stats["count"],
                    "avg_compression_ratio": sum(stats["compression_ratios"]) / stats["count"],
                    "avg_quality_score": sum(stats["quality_scores"]) / stats["count"],
                    "avg_pattern_matches": sum(stats["pattern_matches"]) / stats["count"],
                    "compression_effectiveness": (
                        (1 - sum(stats["compression_ratios"]) / stats["count"]) * 100
                    )
                }
        
        return analysis

    def export_metrics(self, output_path: str, format: str = "json") -> bool:
        """Export metrics to file."""
        try:
            data = {
                "summary": self.get_summary_stats(),
                "time_series": self.get_time_series_data(),
                "engine_comparison": self.get_engine_comparison(),
                "domain_analysis": self.get_domain_analysis(),
                "raw_metrics": [asdict(metric) for metric in self.metrics]
            }
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format == "csv":
                import csv
                with open(output_path, 'w', newline='') as f:
                    if self.metrics:
                        writer = csv.DictWriter(f, fieldnames=asdict(self.metrics[0]).keys())
                        writer.writeheader()
                        for metric in self.metrics:
                            writer.writerow(asdict(metric))
            
            logger.info(f"Exported {len(self.metrics)} metrics to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
        self.session_start = time.time()
        
        # Clear persisted file
        if self.persist_path and Path(self.persist_path).exists():
            try:
                Path(self.persist_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to clear persisted metrics: {e}")

    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations based on metrics."""
        recommendations = []
        
        if not self.metrics:
            return ["No metrics available for analysis"]
        
        stats = self.get_summary_stats()
        
        # Quality recommendations
        avg_quality = stats["quality_metrics"]["average_quality_score"]
        if avg_quality < 7.0:
            recommendations.append(
                f"Average quality score is {avg_quality:.1f}/10. "
                "Consider using more conservative compression levels."
            )
        
        # Performance recommendations
        avg_time = stats["performance_metrics"]["average_processing_time_ms"]
        if avg_time > 500:
            recommendations.append(
                f"Average processing time is {avg_time:.0f}ms. "
                "Consider reducing pattern count or optimizing vector search."
            )
        
        # Engine recommendations
        engine_comparison = self.get_engine_comparison()
        if len(engine_comparison) > 1:
            best_engine = max(
                engine_comparison.items(),
                key=lambda x: x[1]["efficiency_score"]
            )
            recommendations.append(
                f"Engine '{best_engine[0]}' shows best efficiency. "
                f"Consider using it more frequently."
            )
        
        # Success rate recommendations
        success_rate = stats["overview"]["success_rate"]
        if success_rate < 0.95:
            recommendations.append(
                f"Success rate is {success_rate:.1%}. "
                "Check for pattern conflicts or quality threshold issues."
            )
        
        return recommendations

    def __del__(self):
        """Persist metrics on cleanup."""
        try:
            self._persist_metrics()
        except Exception:
            pass  # Don't raise exceptions in destructor