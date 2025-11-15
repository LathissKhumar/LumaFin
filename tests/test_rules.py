import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.rules.engine import rule_engine

def test_rule_engine():
    """Test the rule engine with various merchants"""

    # Test cases that should match rules
    test_cases = [
        ("Netflix", "Entertainment"),
        ("AMZN Mktp US", "Shopping"),
        ("Shell Gas Station", "Transportation"),
        ("ATM Withdrawal", "Cash"),
        ("Whole Foods Market", "Food & Dining"),
        ("Starbucks #123", "Food & Dining"),  # Should match YAML rule
    ]

    print("Testing Rule Engine...")
    print("=" * 50)

    for merchant, expected_category in test_cases:
        result = rule_engine.apply_rules(merchant)
        if result:
            status = "✓" if result.name == expected_category else "✗"
            print(f"{status} '{merchant}' -> {result.name} (expected: {expected_category})")
        else:
            print(f"✗ '{merchant}' -> No match (expected: {expected_category})")

    # Test case that should NOT match
    result = rule_engine.apply_rules("Unknown Store XYZ")
    if result is None:
        print("✓ 'Unknown Store XYZ' -> No match (correct)")
    else:
        print(f"✗ 'Unknown Store XYZ' -> {result.name} (should be no match)")

    print("\nAll loaded rules:")
    rules = rule_engine.get_all_rules()
    for rule in rules[:5]:  # Show first 5
        print(f"  {rule['pattern']} -> {rule['category']} (priority: {rule['priority']}, source: {rule['source']})")
    if len(rules) > 5:
        print(f"  ... and {len(rules) - 5} more rules")

if __name__ == "__main__":
    test_rule_engine()