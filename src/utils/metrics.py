"""Metrics utilities for classification, calibration, and fairness checks.

Functions avoid heavy dependencies where possible and fall back gracefully if
optional libs (sklearn) are not available.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Dict, Any, Optional
import math
from collections import Counter, defaultdict


def _safe_div(a: float, b: float) -> float:
	return a / b if b else 0.0


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[List[str]] = None) -> Tuple[List[List[int]], List[str]]:
	if labels is None:
		labels = sorted(list(set(y_true) | set(y_pred)))
	idx = {l: i for i, l in enumerate(labels)}
	n = len(labels)
	mat = [[0 for _ in range(n)] for _ in range(n)]
	for t, p in zip(y_true, y_pred):
		if t not in idx:
			labels.append(t)
			idx[t] = len(labels) - 1
			# expand matrix
			for row in mat:
				row.append(0)
			mat.append([0 for _ in range(len(labels))])
		if p not in idx:
			labels.append(p)
			idx[p] = len(labels) - 1
			for row in mat:
				row.append(0)
			mat.append([0 for _ in range(len(labels))])
		mat[idx[t]][idx[p]] += 1
	return mat, labels


def precision_recall_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
	if labels is None:
		labels = sorted(list(set(y_true) | set(y_pred)))
	# per-class counts
	tp = Counter()
	fp = Counter()
	fn = Counter()
	for t, p in zip(y_true, y_pred):
		if t == p:
			tp[t] += 1
		else:
			fp[p] += 1
			fn[t] += 1
	metrics: Dict[str, Dict[str, float]] = {}
	for c in labels:
		prec = _safe_div(tp[c], tp[c] + fp[c])
		rec = _safe_div(tp[c], tp[c] + fn[c])
		f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
		metrics[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp[c] + fn[c]}
	# macro
	macro_p = sum(m["precision"] for m in metrics.values()) / len(metrics) if metrics else 0.0
	macro_r = sum(m["recall"] for m in metrics.values()) / len(metrics) if metrics else 0.0
	macro_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics) if metrics else 0.0
	metrics["macro_avg"] = {"precision": macro_p, "recall": macro_r, "f1": macro_f1}
	# micro
	total_tp = sum(tp.values())
	total_fp = sum(fp.values())
	total_fn = sum(fn.values())
	micro_p = _safe_div(total_tp, total_tp + total_fp)
	micro_r = _safe_div(total_tp, total_tp + total_fn)
	micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r) if (micro_p + micro_r) else 0.0
	metrics["micro_avg"] = {"precision": micro_p, "recall": micro_r, "f1": micro_f1}
	return metrics


def calibration_bins(y_true: Sequence[str], y_prob: Sequence[float], y_pred: Sequence[str], positive_label: Optional[str] = None, n_bins: int = 10) -> List[Dict[str, float]]:
	"""Compute calibration per probability bin for binary one-vs-rest case.

	Args:
		y_true: true labels
		y_prob: predicted probability for the predicted class (or positive label)
		y_pred: predicted labels
		positive_label: if provided, compute for that class vs rest; otherwise use correctness of prediction
		n_bins: number of bins
	Returns: list of dicts with keys: bin_lower, bin_upper, count, avg_conf, accuracy
	"""
	bins = [
		{
			"bin_lower": i / n_bins,
			"bin_upper": (i + 1) / n_bins,
			"count": 0,
			"sum_conf": 0.0,
			"correct": 0,
		}
		for i in range(n_bins)
	]
	for t, p_label, p_conf in zip(y_true, y_pred, y_prob):
		idx = min(n_bins - 1, int(p_conf * n_bins))
		bins[idx]["count"] += 1
		bins[idx]["sum_conf"] += p_conf
		if positive_label is None:
			correct = int(t == p_label)
		else:
			correct = int((t == positive_label) == (p_label == positive_label))
		bins[idx]["correct"] += correct
	out = []
	for b in bins:
		count = b["count"]
		avg_conf = _safe_div(b["sum_conf"], count)
		acc = _safe_div(b["correct"], count)
		out.append({
			"bin_lower": b["bin_lower"],
			"bin_upper": b["bin_upper"],
			"count": count,
			"avg_conf": avg_conf,
			"accuracy": acc,
			"gap": (avg_conf - acc) if count else 0.0,
		})
	return out


def amount_bucket(amount: float) -> str:
	if amount < 10:
		return "<10"
	if amount < 50:
		return "10-50"
	if amount < 100:
		return "50-100"
	if amount < 500:
		return "100-500"
	return ">=500"


def bucketed_f1(y_true: Sequence[str], y_pred: Sequence[str], amounts: Sequence[float]) -> Dict[str, float]:
	by_bucket: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
	for t, p, a in zip(y_true, y_pred, amounts):
		b = amount_bucket(float(a))
		by_bucket[b]["y_true"].append(t)
		by_bucket[b]["y_pred"].append(p)
	scores: Dict[str, float] = {}
	for b, data in by_bucket.items():
		metrics = precision_recall_f1(data["y_true"], data["y_pred"])  # macro avg
		scores[b] = metrics["macro_avg"]["f1"]
	return scores


__all__ = [
	"confusion_matrix",
	"precision_recall_f1",
	"calibration_bins",
	"amount_bucket",
	"bucketed_f1",
]

