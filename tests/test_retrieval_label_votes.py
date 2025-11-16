from src.retrieval.service import RetrievalService


def test_get_label_votes_empty():
    svc = RetrievalService()
    assert svc.get_label_votes([]) == {}


def test_get_label_votes_basic():
    svc = RetrievalService()
    results = [
        {'label': 'coffee', 'similarity': 0.8},
        {'label': 'coffee', 'similarity': 0.7},
        {'label': 'food', 'similarity': 0.6},
    ]
    votes = svc.get_label_votes(results)
    assert votes['coffee'] == 1.5
    assert votes['food'] == 0.6
