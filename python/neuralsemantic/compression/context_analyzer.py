"""Context analysis for intelligent compression strategy selection."""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """
    Analyzes text context to optimize compression strategy.
    
    Detects domain, content type, technical density, and other factors
    to inform compression engine selection and parameter tuning.
    """

    def __init__(self):
        # Domain detection keywords
        self.domain_keywords = {
            'web-development': [
                'react', 'javascript', 'typescript', 'api', 'frontend', 'backend',
                'database', 'server', 'client', 'component', 'interface', 'framework',
                'html', 'css', 'node', 'express', 'webpack', 'npm', 'yarn', 'git',
                'repository', 'deployment', 'hosting', 'domain', 'endpoint', 'middleware'
            ],
            'agile': [
                'sprint', 'scrum', 'backlog', 'user story', 'story points', 'epic',
                'retrospective', 'standup', 'kanban', 'product owner', 'scrum master',
                'velocity', 'burndown', 'iteration', 'refinement', 'planning poker',
                'definition of done', 'acceptance criteria', 'product backlog'
            ],
            'devops': [
                'docker', 'kubernetes', 'ci/cd', 'jenkins', 'pipeline', 'deployment',
                'infrastructure', 'cloud', 'aws', 'azure', 'gcp', 'terraform',
                'ansible', 'monitoring', 'logging', 'metrics', 'scalability',
                'load balancer', 'microservices', 'containerization'
            ],
            'machine-learning': [
                'model', 'training', 'dataset', 'neural network', 'deep learning',
                'algorithm', 'feature', 'prediction', 'classification', 'regression',
                'supervised', 'unsupervised', 'tensorflow', 'pytorch', 'scikit-learn',
                'data science', 'artificial intelligence', 'natural language processing'
            ],
            'data-science': [
                'analysis', 'visualization', 'statistics', 'pandas', 'numpy',
                'matplotlib', 'seaborn', 'jupyter', 'notebook', 'correlation',
                'regression', 'clustering', 'data mining', 'big data', 'etl'
            ]
        }

        # Content type indicators
        self.content_patterns = {
            'code': [
                r'```[\w]*\n.*?\n```',  # Code blocks
                r'function\s+\w+\s*\(',  # Function definitions
                r'class\s+\w+\s*[:\(]',  # Class definitions
                r'def\s+\w+\s*\(',  # Python functions
                r'import\s+\w+',  # Import statements
                r'from\s+\w+\s+import',  # Python imports
                r'const\s+\w+\s*=',  # JavaScript constants
                r'let\s+\w+\s*=',  # JavaScript variables
                r'var\s+\w+\s*=',  # JavaScript variables
            ],
            'documentation': [
                r'^#+\s+',  # Markdown headers
                r'^\*\s+',  # Bulleted lists
                r'^\d+\.\s+',  # Numbered lists
                r'\[.*?\]\(.*?\)',  # Markdown links
                r'`[^`]+`',  # Inline code
            ],
            'configuration': [
                r'^\w+\s*[:=]\s*\w+',  # Key-value pairs
                r'^\s*-\s+\w+:',  # YAML lists
                r'{\s*"[^"]+"\s*:',  # JSON objects
                r'<\w+.*?/>',  # XML/HTML tags
            ],
            'prose': [
                r'\w+\.\s+[A-Z]',  # Sentence endings
                r',\s+\w+',  # Comma-separated phrases
                r'\w+\s+that\s+\w+',  # Relative clauses
                r'\w+\s+which\s+\w+',  # Relative clauses
            ]
        }

        # Technical density indicators
        self.technical_indicators = [
            'api', 'sdk', 'framework', 'library', 'protocol', 'algorithm',
            'architecture', 'infrastructure', 'implementation', 'optimization',
            'configuration', 'deployment', 'integration', 'authentication',
            'authorization', 'encryption', 'performance', 'scalability'
        ]

    def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        """Perform comprehensive context analysis."""
        analysis = {
            'domain': self.detect_domain(text),
            'content_type': self.analyze_content_type(text),
            'technical_density': self.calculate_technical_density(text),
            'complexity_score': self.calculate_complexity_score(text),
            'compression_readiness': self.assess_compression_readiness(text),
            'language_characteristics': self.analyze_language_characteristics(text),
            'structure_analysis': self.analyze_text_structure(text)
        }
        
        return analysis

    def detect_domain(self, text: str) -> str:
        """Detect the primary domain/topic of the text."""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences with weight based on keyword importance
                count = text_lower.count(keyword)
                if count > 0:
                    # Weight longer keywords more heavily
                    weight = len(keyword.split())
                    score += count * weight
            
            domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if no clear domain
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
        
        return 'general'

    def analyze_content_type(self, text: str) -> str:
        """Analyze the type of content (code, documentation, prose, etc.)."""
        type_scores = {}
        
        for content_type, patterns in self.content_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
                score += len(matches)
            type_scores[content_type] = score
        
        # Determine primary content type
        if type_scores:
            total_score = sum(type_scores.values())
            if total_score > 0:
                # Return type with highest relative score
                best_type = max(type_scores, key=type_scores.get)
                if type_scores[best_type] / total_score > 0.3:  # At least 30% of indicators
                    return best_type
        
        # Fallback analysis
        if len(text.split('\n')) > len(text.split('. ')):
            return 'structured'
        elif '.' in text and len(text.split('. ')) > 2:
            return 'prose'
        else:
            return 'general'

    def calculate_technical_density(self, text: str) -> float:
        """Calculate the technical density of the text (0.0 to 1.0)."""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
        
        # Count technical terms
        technical_count = sum(1 for word in words if word in self.technical_indicators)
        
        # Count technical patterns
        technical_patterns = [
            r'\b\w+[A-Z]\w*\b',  # CamelCase
            r'\b[A-Z]{2,}\b',    # Acronyms
            r'\b\w+_\w+\b',      # Snake_case
            r'\b\w+-\w+\b',      # Kebab-case
            r'\bv?\d+\.\d+',     # Version numbers
        ]
        
        pattern_matches = 0
        for pattern in technical_patterns:
            pattern_matches += len(re.findall(pattern, text))
        
        # Calculate density
        total_technical_indicators = technical_count + pattern_matches
        density = min(1.0, total_technical_indicators / len(words))
        
        return density

    def calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity score (0.0 to 1.0)."""
        if not text:
            return 0.0
        
        # Factors contributing to complexity
        complexity_factors = []
        
        # Sentence length variance
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            complexity_factors.append(min(1.0, avg_length / 20))  # Normalize to 20 words
        
        # Vocabulary diversity
        words = text.lower().split()
        if words:
            unique_words = len(set(words))
            lexical_diversity = unique_words / len(words)
            complexity_factors.append(lexical_diversity)
        
        # Technical terminology ratio
        technical_density = self.calculate_technical_density(text)
        complexity_factors.append(technical_density)
        
        # Nested structure indicators
        nesting_indicators = [
            len(re.findall(r'\([^)]*\)', text)),  # Parentheses
            len(re.findall(r'\[[^\]]*\]', text)),  # Brackets
            len(re.findall(r'{[^}]*}', text)),     # Braces
            text.count('\n\n'),                    # Paragraph breaks
        ]
        nesting_score = min(1.0, sum(nesting_indicators) / len(text.split()))
        complexity_factors.append(nesting_score)
        
        # Calculate average complexity
        if complexity_factors:
            return sum(complexity_factors) / len(complexity_factors)
        else:
            return 0.5  # Neutral complexity

    def assess_compression_readiness(self, text: str) -> Dict[str, Any]:
        """Assess how ready the text is for compression."""
        readiness = {
            'overall_score': 0.0,
            'factors': {},
            'recommendations': []
        }
        
        # Check for redundancy
        words = text.lower().split()
        if words:
            word_freq = Counter(words)
            common_words = [word for word, count in word_freq.most_common(10) if count > 1]
            redundancy_score = len(common_words) / len(set(words))
            readiness['factors']['redundancy'] = redundancy_score
        
        # Check for verbose patterns
        verbose_patterns = [
            r'\bvery\s+\w+\b',
            r'\breally\s+\w+\b',
            r'\bquite\s+\w+\b',
            r'\bextremely\s+\w+\b',
            r'\bin\s+order\s+to\b',
            r'\bit\s+is\s+important\s+to\b',
            r'\bfor\s+the\s+purpose\s+of\b',
        ]
        
        verbose_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                           for pattern in verbose_patterns)
        verbosity_score = min(1.0, verbose_count / len(words) * 10) if words else 0
        readiness['factors']['verbosity'] = verbosity_score
        
        # Check for abbreviation potential
        long_phrases = re.findall(r'\b\w{8,}\s+\w{8,}\b', text)
        abbreviation_potential = min(1.0, len(long_phrases) / max(1, len(text.split()) // 10))
        readiness['factors']['abbreviation_potential'] = abbreviation_potential
        
        # Calculate overall readiness
        overall_score = (
            redundancy_score * 0.4 +
            verbosity_score * 0.3 +
            abbreviation_potential * 0.3
        )
        readiness['overall_score'] = overall_score
        
        # Generate recommendations
        if redundancy_score > 0.3:
            readiness['recommendations'].append("High word redundancy detected - good for compression")
        if verbosity_score > 0.2:
            readiness['recommendations'].append("Verbose patterns found - consider aggressive compression")
        if abbreviation_potential > 0.3:
            readiness['recommendations'].append("Long phrases detected - ideal for pattern matching")
        
        return readiness

    def analyze_language_characteristics(self, text: str) -> Dict[str, Any]:
        """Analyze language-specific characteristics."""
        characteristics = {
            'avg_word_length': 0.0,
            'punctuation_density': 0.0,
            'capitalization_patterns': {},
            'language_complexity': 0.0
        }
        
        if not text:
            return characteristics
        
        words = text.split()
        if words:
            # Average word length
            word_lengths = [len(word.strip('.,!?;:"()[]{}')) for word in words]
            characteristics['avg_word_length'] = sum(word_lengths) / len(word_lengths)
        
        # Punctuation density
        punctuation_chars = '.,!?;:"()[]{}/-'
        punct_count = sum(1 for char in text if char in punctuation_chars)
        characteristics['punctuation_density'] = punct_count / len(text) if text else 0
        
        # Capitalization patterns
        cap_patterns = {
            'all_caps_words': len(re.findall(r'\b[A-Z]{2,}\b', text)),
            'title_case_words': len(re.findall(r'\b[A-Z][a-z]+\b', text)),
            'camel_case_words': len(re.findall(r'\b[a-z]+[A-Z]\w*\b', text)),
        }
        characteristics['capitalization_patterns'] = cap_patterns
        
        # Language complexity (based on sentence structure)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Complex sentences have more conjunctions and relative pronouns
            complex_indicators = ['which', 'that', 'where', 'when', 'because', 'although', 'however']
            complex_word_count = sum(text.lower().count(indicator) for indicator in complex_indicators)
            complexity = min(1.0, (avg_sentence_length / 15 + complex_word_count / len(words)) / 2)
            characteristics['language_complexity'] = complexity
        
        return characteristics

    def analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structural characteristics of the text."""
        structure = {
            'paragraph_count': 0,
            'sentence_count': 0,
            'list_items': 0,
            'has_headers': False,
            'has_code_blocks': False,
            'structure_type': 'flat'
        }
        
        # Paragraph analysis
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        structure['paragraph_count'] = len(paragraphs)
        
        # Sentence analysis
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        structure['sentence_count'] = len(sentences)
        
        # List detection
        list_patterns = [
            r'^\s*[-*•]\s+',  # Bulleted lists
            r'^\s*\d+\.\s+',  # Numbered lists
        ]
        for pattern in list_patterns:
            structure['list_items'] += len(re.findall(pattern, text, re.MULTILINE))
        
        # Header detection
        structure['has_headers'] = bool(re.search(r'^#+\s+', text, re.MULTILINE))
        
        # Code block detection
        structure['has_code_blocks'] = bool(re.search(r'```', text))
        
        # Determine structure type
        if structure['has_headers'] and structure['paragraph_count'] > 3:
            structure['structure_type'] = 'hierarchical'
        elif structure['list_items'] > 0:
            structure['structure_type'] = 'list-based'
        elif structure['paragraph_count'] > 1:
            structure['structure_type'] = 'multi-paragraph'
        elif structure['sentence_count'] > 5:
            structure['structure_type'] = 'narrative'
        else:
            structure['structure_type'] = 'flat'
        
        return structure

    def recommend_compression_strategy(self, text: str) -> Dict[str, Any]:
        """Recommend optimal compression strategy based on analysis."""
        analysis = self.analyze(text)
        
        recommendations = {
            'recommended_level': 'balanced',
            'recommended_engine': 'hybrid',
            'strategy_rationale': [],
            'parameter_suggestions': {}
        }
        
        # Domain-based recommendations
        domain = analysis['domain']
        if domain in ['web-development', 'devops']:
            recommendations['recommended_engine'] = 'semantic'
            recommendations['strategy_rationale'].append(f"Technical domain ({domain}) benefits from semantic compression")
        elif domain == 'agile':
            recommendations['recommended_engine'] = 'hybrid'
            recommendations['strategy_rationale'].append("Agile content has mixed technical and business terms")
        
        # Content type recommendations
        content_type = analysis['content_type']
        if content_type == 'code':
            recommendations['recommended_level'] = 'light'
            recommendations['parameter_suggestions']['preserve_code'] = True
            recommendations['strategy_rationale'].append("Code content requires careful preservation")
        elif content_type == 'documentation':
            recommendations['recommended_level'] = 'balanced'
            recommendations['strategy_rationale'].append("Documentation can handle moderate compression")
        elif content_type == 'prose':
            recommendations['recommended_level'] = 'aggressive'
            recommendations['strategy_rationale'].append("Prose text is highly compressible")
        
        # Technical density impact
        tech_density = analysis['technical_density']
        if tech_density > 0.7:
            recommendations['recommended_engine'] = 'semantic'
            recommendations['strategy_rationale'].append("High technical density requires semantic preservation")
        elif tech_density < 0.3:
            recommendations['recommended_level'] = 'aggressive'
            recommendations['strategy_rationale'].append("Low technical density allows aggressive compression")
        
        # Complexity considerations
        complexity = analysis['complexity_score']
        if complexity > 0.8:
            recommendations['recommended_level'] = 'light'
            recommendations['strategy_rationale'].append("High complexity requires conservative compression")
        elif complexity < 0.4:
            recommendations['recommended_level'] = 'aggressive'
            recommendations['strategy_rationale'].append("Low complexity allows aggressive compression")
        
        # Compression readiness
        readiness = analysis['compression_readiness']['overall_score']
        if readiness > 0.7:
            recommendations['strategy_rationale'].append("High compression readiness detected")
            if recommendations['recommended_level'] == 'balanced':
                recommendations['recommended_level'] = 'aggressive'
        elif readiness < 0.3:
            recommendations['strategy_rationale'].append("Low compression readiness - be conservative")
            if recommendations['recommended_level'] == 'aggressive':
                recommendations['recommended_level'] = 'balanced'
        
        return recommendations

    def get_domain_confidence(self, text: str) -> Dict[str, float]:
        """Get confidence scores for all domains."""
        text_lower = text.lower()
        domain_scores = {}
        total_score = 0
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    weight = len(keyword.split())
                    score += count * weight
            
            domain_scores[domain] = score
            total_score += score
        
        # Convert to confidence percentages
        if total_score > 0:
            domain_confidence = {
                domain: score / total_score 
                for domain, score in domain_scores.items()
            }
        else:
            domain_confidence = {domain: 0.0 for domain in self.domain_keywords.keys()}
        
        return domain_confidence