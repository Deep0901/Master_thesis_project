"""
Query Parser and Normalizer

Parses and normalizes user search queries for the fuzzy ranking system.
Handles multilingual queries (DE, FR, IT, EN) and extracts query intent.

Research Context:
- Part of Master Thesis: "Improving Access to Swiss OGD through Fuzzy HCIR"
- Supports RQ2: Multilingual search performance
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class QueryLanguage(Enum):
    """Supported query languages."""
    GERMAN = "de"
    FRENCH = "fr"
    ITALIAN = "it"
    ENGLISH = "en"
    UNKNOWN = "unknown"


class TemporalModifier(Enum):
    """Temporal modifiers in queries."""
    VERY_RECENT = "very_recent"
    RECENT = "recent"
    MODERATE = "moderate"
    OLD = "old"
    ANY = "any"


class QualityModifier(Enum):
    """Quality/completeness modifiers."""
    COMPLETE = "complete"
    MOSTLY_COMPLETE = "mostly_complete"
    ANY = "any"


@dataclass
class ParsedQuery:
    """
    Structured representation of a parsed user query.
    
    Attributes:
        raw_query: Original query string
        normalized_query: Cleaned and normalized query
        detected_language: Detected query language
        keywords: Extracted keywords
        temporal_modifier: Detected temporal requirements
        quality_modifier: Detected quality requirements
        themes: Detected thematic categories
        intent: Inferred query intent
    """
    raw_query: str
    normalized_query: str
    detected_language: QueryLanguage
    keywords: List[str]
    temporal_modifier: TemporalModifier
    quality_modifier: QualityModifier
    themes: List[str]
    intent: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "language": self.detected_language.value,
            "keywords": self.keywords,
            "temporal": self.temporal_modifier.value,
            "quality": self.quality_modifier.value,
            "themes": self.themes,
            "intent": self.intent
        }


class QueryParser:
    """
    Parses and normalizes user search queries.
    
    Extracts linguistic modifiers (fuzzy terms like "recent", "complete")
    and translates them into fuzzy system parameters.
    """
    
    # Temporal modifier patterns by language
    TEMPORAL_PATTERNS = {
        QueryLanguage.ENGLISH: {
            TemporalModifier.VERY_RECENT: [
                r'\bvery recent\b', r'\blatest\b', r'\bjust published\b',
                r'\bthis week\b', r'\btoday\b', r'\bnew\b'
            ],
            TemporalModifier.RECENT: [
                r'\brecent\b', r'\bthis month\b', r'\blast few weeks\b',
                r'\bnewly\b', r'\bupdated\b'
            ],
            TemporalModifier.OLD: [
                r'\bold\b', r'\bhistoric\b', r'\barchive\b',
                r'\bprevious years?\b'
            ]
        },
        QueryLanguage.GERMAN: {
            TemporalModifier.VERY_RECENT: [
                r'\bsehr aktuell\b', r'\bneuest\w*\b', r'\bgerade veröffentlicht\b',
                r'\bdiese woche\b', r'\bheute\b'
            ],
            TemporalModifier.RECENT: [
                r'\baktuell\b', r'\bkürzlich\b', r'\bneu\b',
                r'\bdiesen monat\b'
            ],
            TemporalModifier.OLD: [
                r'\balt\b', r'\bhistorisch\b', r'\barchiv\b'
            ]
        },
        QueryLanguage.FRENCH: {
            TemporalModifier.VERY_RECENT: [
                r'\btrès récent\b', r'\bdernier\b', r'\bjuste publié\b',
                r'\bcette semaine\b', r"\baujourd'hui\b"
            ],
            TemporalModifier.RECENT: [
                r'\brécent\b', r'\bce mois\b', r'\bnouveau\b',
                r'\bmis à jour\b'
            ],
            TemporalModifier.OLD: [
                r'\bancien\b', r'\bhistorique\b', r'\barchive\b'
            ]
        },
        QueryLanguage.ITALIAN: {
            TemporalModifier.VERY_RECENT: [
                r'\bmolto recente\b', r'\bultimo\b', r'\bappena pubblicato\b',
                r'\bquesta settimana\b', r'\boggi\b'
            ],
            TemporalModifier.RECENT: [
                r'\brecente\b', r'\bquesto mese\b', r'\bnuovo\b',
                r'\baggiornato\b'
            ],
            TemporalModifier.OLD: [
                r'\bvecchio\b', r'\bstorico\b', r'\barchivio\b'
            ]
        }
    }
    
    # Quality modifier patterns
    QUALITY_PATTERNS = {
        QueryLanguage.ENGLISH: {
            QualityModifier.COMPLETE: [
                r'\bcomplete\b', r'\bfull\b', r'\bcomprehensive\b',
                r'\bdetailed\b'
            ],
            QualityModifier.MOSTLY_COMPLETE: [
                r'\bmostly complete\b', r'\bpartial\b', r'\bsome\b'
            ]
        },
        QueryLanguage.GERMAN: {
            QualityModifier.COMPLETE: [
                r'\bvollständig\b', r'\bkomplett\b', r'\bumfassend\b'
            ],
            QualityModifier.MOSTLY_COMPLETE: [
                r'\bgrößtenteils\b', r'\bteilweise\b'
            ]
        },
        QueryLanguage.FRENCH: {
            QualityModifier.COMPLETE: [
                r'\bcomplet\b', r'\bintégral\b', r'\bdétaillé\b'
            ],
            QualityModifier.MOSTLY_COMPLETE: [
                r'\bpartiel\b', r'\ben partie\b'
            ]
        },
        QueryLanguage.ITALIAN: {
            QualityModifier.COMPLETE: [
                r'\bcompleto\b', r'\bintegrale\b', r'\bdettagliato\b'
            ],
            QualityModifier.MOSTLY_COMPLETE: [
                r'\bparziale\b', r'\bin parte\b'
            ]
        }
    }
    
    # Theme/domain keywords
    THEME_KEYWORDS = {
        "environment": [
            "environment", "umwelt", "environnement", "ambiente",
            "air quality", "luftqualität", "qualité de l'air",
            "pollution", "climate", "klima", "climat"
        ],
        "mobility": [
            "transport", "verkehr", "traffic", "mobilité", "mobilità",
            "road", "strasse", "route", "strada", "rail", "bahn",
            "mobility", "bicycle", "bike", "biking", "cycling", "cycle",
            "fahrrad", "velo", "vélo", "bicicletta", "transit", "transportation"
        ],
        "health": [
            "health", "gesundheit", "santé", "salute",
            "hospital", "krankenhaus", "hôpital", "ospedale"
        ],
        "education": [
            "education", "bildung", "éducation", "istruzione",
            "school", "schule", "école", "scuola"
        ],
        "economy": [
            "economy", "wirtschaft", "économie", "economia",
            "finance", "finanzen", "employment", "beschäftigung"
        ],
        "population": [
            "population", "bevölkerung", "demographic", "demographie"
        ]
    }
    
    # Language detection patterns
    LANGUAGE_INDICATORS = {
        QueryLanguage.GERMAN: [
            r'\b(und|mit|nach|für|über|unter|durch)\b',
            r'ä|ö|ü|ß'
        ],
        QueryLanguage.FRENCH: [
            r'\b(le|la|les|de|du|des|avec|pour)\b',
            r'é|è|ê|ë|à|â|ç|ô|î|û'
        ],
        QueryLanguage.ITALIAN: [
            r'\b(il|la|di|da|con|per|che)\b',
            r'à|è|ì|ò|ù'
        ],
        QueryLanguage.ENGLISH: [
            r'\b(the|of|and|to|in|for|with)\b'
        ]
    }
    
    def __init__(self):
        """Initialize the query parser."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._compiled_temporal = {}
        for lang, modifiers in self.TEMPORAL_PATTERNS.items():
            self._compiled_temporal[lang] = {}
            for modifier, patterns in modifiers.items():
                self._compiled_temporal[lang][modifier] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]
        
        self._compiled_quality = {}
        for lang, modifiers in self.QUALITY_PATTERNS.items():
            self._compiled_quality[lang] = {}
            for modifier, patterns in modifiers.items():
                self._compiled_quality[lang][modifier] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]
        
        self._compiled_language = {}
        for lang, patterns in self.LANGUAGE_INDICATORS.items():
            self._compiled_language[lang] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def detect_language(self, query: str) -> QueryLanguage:
        """
        Detect the language of a query.
        
        Args:
            query: Input query string
            
        Returns:
            Detected QueryLanguage
        """
        scores = {lang: 0 for lang in QueryLanguage if lang != QueryLanguage.UNKNOWN}
        
        for lang, patterns in self._compiled_language.items():
            for pattern in patterns:
                matches = pattern.findall(query)
                scores[lang] += len(matches)
        
        if max(scores.values()) == 0:
            return QueryLanguage.UNKNOWN
        
        return max(scores, key=scores.get)
    
    def extract_temporal_modifier(
        self, query: str, language: QueryLanguage
    ) -> TemporalModifier:
        """
        Extract temporal modifier from query.
        
        Args:
            query: Input query
            language: Detected language
            
        Returns:
            TemporalModifier or ANY if none detected
        """
        if language not in self._compiled_temporal:
            language = QueryLanguage.ENGLISH
        
        for modifier, patterns in self._compiled_temporal[language].items():
            for pattern in patterns:
                if pattern.search(query):
                    return modifier
        
        return TemporalModifier.ANY
    
    def extract_quality_modifier(
        self, query: str, language: QueryLanguage
    ) -> QualityModifier:
        """
        Extract quality/completeness modifier from query.
        
        Args:
            query: Input query
            language: Detected language
            
        Returns:
            QualityModifier or ANY if none detected
        """
        if language not in self._compiled_quality:
            language = QueryLanguage.ENGLISH
        
        for modifier, patterns in self._compiled_quality[language].items():
            for pattern in patterns:
                if pattern.search(query):
                    return modifier
        
        return QualityModifier.ANY
    
    def extract_themes(self, query: str) -> List[str]:
        """
        Extract thematic categories from query.
        
        Args:
            query: Input query
            
        Returns:
            List of detected theme names
        """
        query_lower = query.lower()
        detected_themes = []
        
        for theme, keywords in self.THEME_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    if theme not in detected_themes:
                        detected_themes.append(theme)
                    break
        
        return detected_themes
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract main keywords from query.
        
        Removes stop words and modifiers.
        
        Args:
            query: Input query
            
        Returns:
            List of keywords
        """
        # Remove common words and modifiers
        stop_patterns = [
            r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b',
            r'\b(der|die|das|und|oder|in|auf|mit|für|von|bei)\b',
            r'\b(le|la|les|de|du|des|et|ou|dans|sur|avec|pour)\b',
            r'\b(il|lo|la|di|da|e|o|in|su|con|per)\b',
            r'\b(data|dataset|datasets|related|show|statistics|statistic)\b',
            r'\b(recent|complete|full|new|old|latest)\b',
            r'\b(aktuell|vollständig|neu|alt)\b',
            r'\b(récent|complet|nouveau|ancien)\b',
            r'\b(recente|completo|nuovo|vecchio)\b'
        ]
        
        cleaned = query.lower()
        for pattern in stop_patterns:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Split into words and filter
        words = cleaned.split()
        keywords = [w.strip() for w in words if len(w.strip()) > 2]
        
        return keywords
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize a query string.
        
        Args:
            query: Raw query
            
        Returns:
            Normalized query
        """
        # Remove extra whitespace
        normalized = ' '.join(query.split())
        # Remove special characters except letters and spaces
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        return normalized.strip()
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a user query into structured form.
        
        Args:
            query: Raw user query
            
        Returns:
            ParsedQuery with extracted information
        """
        normalized = self.normalize_query(query)
        language = self.detect_language(query)
        
        return ParsedQuery(
            raw_query=query,
            normalized_query=normalized,
            detected_language=language,
            keywords=self.extract_keywords(normalized),
            temporal_modifier=self.extract_temporal_modifier(query, language),
            quality_modifier=self.extract_quality_modifier(query, language),
            themes=self.extract_themes(query),
            intent="dataset_search"  # Default intent
        )


def create_parser() -> QueryParser:
    """Create a new QueryParser instance."""
    return QueryParser()


if __name__ == "__main__":
    # Demo: Parse sample queries
    parser = QueryParser()
    
    test_queries = [
        "recent transport statistics in Zurich",
        "aktuelle Luftqualitätsdaten Schweiz",
        "données de santé complètes 2024",
        "statistiche sulla popolazione recente",
        "mostly complete financial data"
    ]
    
    print("=" * 60)
    print("QUERY PARSER DEMONSTRATION")
    print("=" * 60)
    
    for query in test_queries:
        result = parser.parse(query)
        print(f"\nQuery: '{query}'")
        print(f"  Language: {result.detected_language.value}")
        print(f"  Keywords: {result.keywords}")
        print(f"  Temporal: {result.temporal_modifier.value}")
        print(f"  Quality: {result.quality_modifier.value}")
        print(f"  Themes: {result.themes}")
