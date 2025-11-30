# Path: QAgents-workflos/database/__init__.py
# Purpose: Database module exports for storage, logging, memory, and circuit quality
# Relations: Provides unified access to all database functionality

"""Database module for storage, logging, memory, and circuit quality tracking."""

from .storage import (
    Database,
    MemoryType,
    MemoryEntry,
    LogEntry,
    ResultEntry,
    get_database
)

from .circuit_quality_db import (
    CircuitQualityDB,
    CircuitEvaluation,
    QualityMetrics,
    get_quality_db
)

__all__ = [
    # Original storage
    "Database",
    "MemoryType",
    "MemoryEntry",
    "LogEntry",
    "ResultEntry",
    "get_database",
    # Quality tracking (NEW)
    "CircuitQualityDB",
    "CircuitEvaluation",
    "QualityMetrics",
    "get_quality_db"
]
