import pytest
from src.preprocessing.normalize import normalize_label, get_time_features
from datetime import datetime


def test_normalize_label_basic():
    assert normalize_label("Coffee #1") == "coffee 1"
    assert normalize_label("  Monthly Rent ") == "monthly rent"
    assert normalize_label(None) == ""


def test_get_time_features():
    dt = datetime(2025, 11, 15, 8, 30)
    feats = get_time_features(dt)
    assert 'hour_of_day' in feats
    assert feats['hour_of_day'] == 8
    assert feats['weekday'] == dt.weekday()
