import re
from typing import Dict

# Merchant name abbreviation mapping
MERCHANT_ABBREV = {
    "SQ *": "square",
    "AMZN": "amazon",
    "TST*": "toast",
    "PYPL *": "paypal",
    "GOOGLE *": "google",
}

def normalize_merchant(merchant: str) -> str:
    """
    Normalize merchant name for better matching.
    
    Steps:
    1. Lowercase
    2. Remove special characters
    3. Expand abbreviations
    4. Remove location codes
    """
    # Lowercase
    merchant = merchant.lower().strip()
    
    # Expand known abbreviations
    for abbrev, full_name in MERCHANT_ABBREV.items():
        if abbrev.lower() in merchant:
            merchant = merchant.replace(abbrev.lower(), full_name)
    
    # Remove common suffixes (location codes, store numbers)
    merchant = re.sub(r'#\d+', '', merchant)  # Remove #123
    merchant = re.sub(r'\d{3,}', '', merchant)  # Remove long numbers
    
    # Remove special characters but keep spaces
    merchant = re.sub(r'[^a-z0-9\s]', ' ', merchant)
    
    # Remove extra whitespace
    merchant = ' '.join(merchant.split())
    
    return merchant

def bucket_amount(amount: float) -> str:
    """Bucket transaction amount into ranges"""
    if amount < 10:
        return "0-10"
    elif amount < 50:
        return "10-50"
    elif amount < 100:
        return "50-100"
    elif amount < 500:
        return "100-500"
    else:
        return "500+"

def get_time_features(date) -> Dict[str, int]:
    """Extract time-based features"""
    return {
        "hour": date.hour if hasattr(date, 'hour') else 12,
        "day_of_week": date.weekday(),
        "is_weekend": 1 if date.weekday() >= 5 else 0,
    }