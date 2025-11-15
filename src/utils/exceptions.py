"""Custom exception types for LumaFin services."""
from __future__ import annotations


class LumaFinError(Exception):
	"""Base error for domain-specific failures."""


class RetrievalError(LumaFinError):
	pass


class ClusteringError(LumaFinError):
	pass


class RerankerError(LumaFinError):
	pass


class ValidationError(LumaFinError):
	pass


__all__ = [
	"LumaFinError",
	"RetrievalError",
	"ClusteringError",
	"RerankerError",
	"ValidationError",
]

