"""Central logging utility for LumaFin.

Provides a consistent, structured logger that can emit either plain text or
JSON lines depending on the environment variable `LUMAFIN_LOG_FORMAT`.

Usage:
	from src.utils.logger import get_logger
	log = get_logger(__name__)
	log.info("Starting service", extra={"component": "api"})

The `extra` dict keys will be merged into the log record. For JSON format they
appear as top-level fields; for text format they are appended as key=value pairs.
"""
from __future__ import annotations

import logging
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

_LOG_FORMAT = os.getenv("LUMAFIN_LOG_FORMAT", "TEXT").upper()
_LEVEL = os.getenv("LUMAFIN_LOG_LEVEL", "INFO").upper()


class _KeyValueFormatter(logging.Formatter):
	def format(self, record: logging.LogRecord) -> str:
		base = f"{datetime.utcfromtimestamp(record.created).isoformat()}Z level={record.levelname} msg={record.getMessage()}"
		# Merge custom extra fields
		extras = {}
		for k, v in record.__dict__.items():
			if k.startswith('_'):
				continue
			if k in ("name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "message"):
				continue
			extras[k] = v
		if extras:
			extra_str = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in extras.items())
			base = f"{base} {extra_str}"
		return base


class _JSONFormatter(logging.Formatter):
	def format(self, record: logging.LogRecord) -> str:
		payload: Dict[str, Any] = {
			"ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
			"level": record.levelname,
			"message": record.getMessage(),
			"logger": record.name,
		}
		for k, v in record.__dict__.items():
			if k.startswith('_'):
				continue
			if k in ("name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "message"):
				continue
			payload[k] = v
		if record.exc_info:
			payload["exc_type"] = str(record.exc_info[0].__name__)
			payload["exc_value"] = str(record.exc_info[1])
		return json.dumps(payload, ensure_ascii=False)


def _build_handler() -> logging.Handler:
	handler = logging.StreamHandler(sys.stdout)
	formatter: logging.Formatter
	if _LOG_FORMAT == "JSON":
		formatter = _JSONFormatter()
	else:
		formatter = _KeyValueFormatter()
	handler.setFormatter(formatter)
	return handler


_handler = _build_handler()


def get_logger(name: str) -> logging.Logger:
	logger = logging.getLogger(name)
	if not logger.handlers:
		logger.setLevel(getattr(logging, _LEVEL, logging.INFO))
		logger.addHandler(_handler)
		logger.propagate = False
	return logger


__all__ = ["get_logger"]

