"""
IntentForge AutoWorker - Autonomous Developer and AI Worker

Phase 1: Context Engine
- ProjectRegistry: Multi-project management
- FileIndexer: AST parsing and symbol extraction
- VectorStore: Semantic search with embeddings
- ContextEngine: Unified context retrieval
"""

from .base import BaseWorker, WorkerConfig, WorkerResult
from .context import (
    ContextEngine,
    FileIndexer,
    ProjectContext,
    ProjectRegistry,
    VectorStore,
)
from .orchestrator import WorkerOrchestrator

__all__ = [
    "BaseWorker",
    "ContextEngine",
    "FileIndexer",
    "ProjectContext",
    "ProjectRegistry",
    "VectorStore",
    "WorkerConfig",
    "WorkerOrchestrator",
    "WorkerResult",
]
