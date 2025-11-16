import re
import yaml
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.storage.database import get_db
from src.models import Category

@dataclass
class RuleResult:
    name: str
    priority: int
    pattern: str
    confidence: float = 1.0


class RuleEngine:
    """
    Deterministic rule-based categorization engine.

    Applies regex patterns to merchant names for high-confidence categorization.
    Rules are loaded from database and YAML config files.
    """

    def __init__(self):
        self.rules_cache: List[Dict] = []
        self._load_rules()

    def _load_rules(self):
        """Load rules from database and YAML config"""
        # Database rules (higher priority)
        db_rules = self._load_db_rules()

        # YAML rules (fallback/default rules)
        yaml_rules = self._load_yaml_rules()

        # Combine with database rules taking priority
        self.rules_cache = db_rules + yaml_rules

        # Sort by priority (higher priority first)
        self.rules_cache.sort(key=lambda x: x.get('priority', 0), reverse=True)

    def _load_db_rules(self) -> List[Dict]:
        """Load active rules from database"""
        try:
            db = next(get_db())
            result = db.execute(text(
                """
                SELECT pattern, category_name, priority
                FROM rules
                WHERE is_active = TRUE
                ORDER BY priority DESC
                """
            ))

            rules = []
            for row in result:
                rules.append({
                    'pattern': row[0],
                    'category': row[1],
                    'priority': row[2],
                    'source': 'database'
                })
            return rules
        except Exception as e:
            print(f"Warning: Could not load database rules: {e}")
            return []

    def _load_yaml_rules(self) -> List[Dict]:
        """Load default rules from YAML file"""
        try:
            with open('taxonomy/rules.yaml', 'r') as f:
                yaml_data = yaml.safe_load(f)

            rules = []
            for rule in yaml_data.get('rules', []):
                rules.append({
                    'pattern': rule['pattern'],
                    'category': rule['category'],
                    'priority': rule.get('priority', 1),
                    'source': 'yaml'
                })
            return rules
        except FileNotFoundError:
            # Return hardcoded defaults if YAML doesn't exist
            return [
                {
                    'pattern': r'(?i)netflix|hulu|spotify|disney\+',
                    'category': 'Entertainment',
                    'priority': 100,
                    'source': 'default'
                },
                {
                    'pattern': r'(?i)whole foods|kroger|safeway|trader joe',
                    'category': 'Food & Dining',
                    'priority': 100,
                    'source': 'default'
                },
                {
                    'pattern': r'(?i)shell|chevron|exxon|bp gas',
                    'category': 'Transportation',
                    'priority': 100,
                    'source': 'default'
                },
                {
                    'pattern': r'(?i)atm withdrawal|atm fee',
                    'category': 'Cash',
                    'priority': 90,
                    'source': 'default'
                },
                {
                    'pattern': r'(?i)amazon|amazon\.com|amzn',
                    'category': 'Shopping',
                    'priority': 80,
                    'source': 'default'
                }
            ]
        except Exception as e:
            print(f"Warning: Could not load YAML rules: {e}")
            return []

    def apply_rules(self, merchant: str, amount: Optional[float] = None) -> Optional[RuleResult]:
        """
        Apply rules to a merchant name.

        Returns Category with confidence=1.0 if rule matches, None otherwise.
        """
        # Normalize merchant for matching
        from src.preprocessing.normalize import normalize_merchant
        normalized_merchant = normalize_merchant(merchant)

        # Try each rule in priority order
        for rule in self.rules_cache:
            try:
                pattern = rule['pattern']
                if re.search(pattern, normalized_merchant):
                    return RuleResult(
                        name=rule['category'],
                        priority=rule.get('priority', 1),
                        pattern=pattern,
                        confidence=1.0,
                    )
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern}': {e}")
                continue

        return None

    def get_all_rules(self) -> List[Dict]:
        """Get all loaded rules for debugging/inspection"""
        return self.rules_cache.copy()

    def refresh_rules(self):
        """Reload rules from sources (call after rule updates)"""
        self._load_rules()

# Global rule engine instance
rule_engine = RuleEngine()