"""
Context Engine - Phase 1 Core Component

Provides:
- ProjectRegistry: Multi-project management with persistence
- FileIndexer: AST parsing, symbol extraction, embeddings
- VectorStore: Semantic search with Chroma/FAISS
- ContextEngine: Unified context retrieval for workers
"""

import ast
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProjectType(Enum):
    """Detected project types"""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    UNKNOWN = "unknown"


class IndexStatus(Enum):
    """Project indexing status"""

    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"
    STALE = "stale"


@dataclass
class Project:
    """Project metadata"""

    id: str
    name: str
    path: str
    type: ProjectType = ProjectType.UNKNOWN
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    index_status: IndexStatus = IndexStatus.PENDING
    last_indexed: str | None = None
    file_count: int = 0
    symbol_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Symbol:
    """Code symbol (function, class, variable)"""

    name: str
    type: str  # function, class, method, variable, import
    file: str
    line: int
    end_line: int | None = None
    signature: str = ""
    docstring: str = ""
    parent: str | None = None
    references: list[str] = field(default_factory=list)


@dataclass
class FileIndex:
    """Indexed file data"""

    path: str
    language: str
    size: int
    modified: str
    hash: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)


@dataclass
class SearchResult:
    """Semantic search result"""

    file: str
    content: str
    score: float
    line_start: int = 0
    line_end: int = 0
    symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectContext:
    """Full context for a project"""

    project_id: str
    project_path: str
    project_type: ProjectType
    file_tree: dict[str, list[str]] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    symbols: dict[str, list[Symbol]] = field(default_factory=dict)
    imports: dict[str, list[str]] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)
    recent_files: list[str] = field(default_factory=list)
    recent_errors: list[dict] = field(default_factory=list)
    recent_changes: list[dict] = field(default_factory=list)


class ProjectRegistry:
    """
    Registry for multi-project management.
    Stores project metadata in SQLite for persistence.

    Features:
    - Register/unregister projects
    - Switch context between projects
    - Track indexing status
    - Project configuration
    """

    def __init__(self, db_path: str = "projects.db"):
        self.db_path = db_path
        self._current_project_id: str | None = None
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT UNIQUE NOT NULL,
                type TEXT DEFAULT 'unknown',
                languages TEXT DEFAULT '[]',
                frameworks TEXT DEFAULT '[]',
                entry_points TEXT DEFAULT '[]',
                config_files TEXT DEFAULT '[]',
                index_status TEXT DEFAULT 'pending',
                last_indexed TEXT,
                file_count INTEGER DEFAULT 0,
                symbol_count INTEGER DEFAULT 0,
                created_at TEXT,
                config TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_projects_path ON projects(path)
        """)
        conn.commit()
        conn.close()

    def register(self, path: str, name: str | None = None, config: dict | None = None) -> Project:
        """
        Register a new project.

        Args:
            path: Absolute path to project root
            name: Project name (defaults to directory name)
            config: Optional project configuration

        Returns:
            Registered Project
        """
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise ValueError(f"Project path does not exist: {path}")

        project_id = hashlib.md5(path.encode()).hexdigest()[:12]
        project_name = name or os.path.basename(path)
        project_type = self._detect_project_type(path)

        project = Project(
            id=project_id,
            name=project_name,
            path=path,
            type=project_type,
            languages=self._detect_languages(path),
            frameworks=self._detect_frameworks(path),
            entry_points=self._find_entry_points(path, project_type),
            config_files=self._find_config_files(path),
            config=config or {},
        )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO projects
                (id, name, path, type, languages, frameworks, entry_points,
                 config_files, index_status, last_indexed, file_count,
                 symbol_count, created_at, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    project.id,
                    project.name,
                    project.path,
                    project.type.value,
                    json.dumps(project.languages),
                    json.dumps(project.frameworks),
                    json.dumps(project.entry_points),
                    json.dumps(project.config_files),
                    project.index_status.value,
                    project.last_indexed,
                    project.file_count,
                    project.symbol_count,
                    project.created_at,
                    json.dumps(project.config),
                ),
            )
            conn.commit()
            logger.info(f"Registered project: {project.name} ({project.type.value})")
        finally:
            conn.close()

        return project

    def unregister(self, project_id: str) -> bool:
        """Remove a project from registry"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get(self, project_id: str) -> Project | None:
        """Get project by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        conn.close()
        return self._row_to_project(row) if row else None

    def get_by_path(self, path: str) -> Project | None:
        """Get project by path"""
        path = os.path.abspath(path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM projects WHERE path = ?", (path,))
        row = cursor.fetchone()
        conn.close()
        return self._row_to_project(row) if row else None

    def list_all(self) -> list[Project]:
        """List all registered projects"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT * FROM projects ORDER BY name")
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_project(row) for row in rows]

    def switch_context(self, project_id: str) -> Project | None:
        """Switch to a different project context"""
        project = self.get(project_id)
        if project:
            self._current_project_id = project_id
            logger.info(f"Switched context to: {project.name}")
        return project

    def get_current(self) -> Project | None:
        """Get currently active project"""
        if self._current_project_id:
            return self.get(self._current_project_id)
        return None

    def update_index_status(
        self,
        project_id: str,
        status: IndexStatus,
        file_count: int = 0,
        symbol_count: int = 0,
    ):
        """Update project indexing status"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            UPDATE projects
            SET index_status = ?, last_indexed = ?, file_count = ?, symbol_count = ?
            WHERE id = ?
        """,
            (
                status.value,
                datetime.utcnow().isoformat() if status == IndexStatus.READY else None,
                file_count,
                symbol_count,
                project_id,
            ),
        )
        conn.commit()
        conn.close()

    def _detect_project_type(self, path: str) -> ProjectType:
        """Detect project type from files"""
        files = os.listdir(path)

        if "pyproject.toml" in files or "setup.py" in files or "requirements.txt" in files:
            return ProjectType.PYTHON
        if "package.json" in files:
            if "tsconfig.json" in files:
                return ProjectType.TYPESCRIPT
            return ProjectType.JAVASCRIPT
        if "go.mod" in files:
            return ProjectType.GO
        if "Cargo.toml" in files:
            return ProjectType.RUST
        if "pom.xml" in files or "build.gradle" in files:
            return ProjectType.JAVA

        return ProjectType.UNKNOWN

    def _detect_languages(self, path: str) -> list[str]:
        """Detect languages used in project"""
        languages = set()
        extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".php": "php",
            ".cpp": "c++",
            ".c": "c",
        }

        for root, _, files in os.walk(path):
            if ".git" in root or "node_modules" in root or "__pycache__" in root:
                continue
            for file in files[:100]:  # Limit for speed
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    languages.add(extensions[ext])

        return list(languages)

    def _detect_frameworks(self, path: str) -> list[str]:
        """Detect frameworks used in project"""
        frameworks = []

        # Check Python frameworks
        requirements = Path(path) / "requirements.txt"
        if requirements.exists():
            content = requirements.read_text()
            if "flask" in content.lower():
                frameworks.append("flask")
            if "fastapi" in content.lower():
                frameworks.append("fastapi")
            if "django" in content.lower():
                frameworks.append("django")
            if "pytest" in content.lower():
                frameworks.append("pytest")

        # Check Node frameworks
        package_json = Path(path) / "package.json"
        if package_json.exists():
            try:
                pkg = json.loads(package_json.read_text())
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "react" in deps:
                    frameworks.append("react")
                if "vue" in deps:
                    frameworks.append("vue")
                if "express" in deps:
                    frameworks.append("express")
                if "next" in deps:
                    frameworks.append("nextjs")
            except json.JSONDecodeError:
                pass

        return frameworks

    def _find_entry_points(self, path: str, project_type: ProjectType) -> list[str]:
        """Find project entry points"""
        entry_points = []

        common_entries = [
            "main.py",
            "app.py",
            "server.py",
            "index.py",
            "main.js",
            "index.js",
            "app.js",
            "server.js",
            "main.ts",
            "index.ts",
            "app.ts",
            "main.go",
            "cmd/main.go",
            "src/main.rs",
            "main.rs",
        ]

        for entry in common_entries:
            full_path = Path(path) / entry
            if full_path.exists():
                entry_points.append(entry)

        return entry_points

    def _find_config_files(self, path: str) -> list[str]:
        """Find configuration files"""
        config_patterns = [
            "*.toml",
            "*.yaml",
            "*.yml",
            "*.json",
            "*.ini",
            "*.cfg",
            ".env*",
            "Dockerfile*",
            "docker-compose*",
            "Makefile",
        ]

        config_files = []
        for file in os.listdir(path):
            file_lower = file.lower()
            if any(
                file_lower.endswith(p.replace("*", "")) or file_lower.startswith(p.replace("*", ""))
                for p in config_patterns
            ):
                config_files.append(file)

        return config_files[:20]  # Limit

    def _row_to_project(self, row: tuple) -> Project:
        """Convert database row to Project"""
        return Project(
            id=row[0],
            name=row[1],
            path=row[2],
            type=ProjectType(row[3]),
            languages=json.loads(row[4]),
            frameworks=json.loads(row[5]),
            entry_points=json.loads(row[6]),
            config_files=json.loads(row[7]),
            index_status=IndexStatus(row[8]),
            last_indexed=row[9],
            file_count=row[10],
            symbol_count=row[11],
            created_at=row[12],
            config=json.loads(row[13]),
        )


class FileIndexer:
    """
    File indexer with AST parsing and symbol extraction.

    Features:
    - Parse files for multiple languages
    - Extract symbols (functions, classes, variables)
    - Chunk files for embeddings
    - Watch for changes (incremental indexing)
    """

    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
    }

    IGNORE_DIRS = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        "eggs",
        "*.egg-info",
        ".eggs",
        "target",
        "vendor",
    }

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def index_project(self, project: Project) -> list[FileIndex]:
        """
        Index all files in a project.

        Args:
            project: Project to index

        Returns:
            List of indexed files
        """
        start_time = time.time()
        indexed_files = []

        files = self._get_indexable_files(project.path)
        logger.info(f"Indexing {len(files)} files in {project.name}")

        for file_path in files:
            try:
                file_index = await self.index_file(file_path)
                if file_index:
                    indexed_files.append(file_index)
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")

        duration = time.time() - start_time
        logger.info(
            f"Indexed {len(indexed_files)} files in {duration:.2f}s "
            f"({len(indexed_files) / duration:.1f} files/s)"
        )

        return indexed_files

    async def index_file(self, file_path: str) -> FileIndex | None:
        """
        Index a single file.

        Args:
            file_path: Path to file

        Returns:
            FileIndex or None if not indexable
        """
        path = Path(file_path)
        if not path.exists():
            return None

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return None

        language = self.SUPPORTED_EXTENSIONS[ext]

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return None

        stat = path.stat()
        file_hash = hashlib.md5(content.encode()).hexdigest()

        # Parse symbols
        symbols = self._extract_symbols(content, language, str(path))

        # Extract imports
        imports = self._extract_imports(content, language)

        # Create chunks for embeddings
        chunks = self._create_chunks(content, str(path))

        return FileIndex(
            path=str(path),
            language=language,
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            hash=file_hash,
            symbols=symbols,
            imports=imports,
            chunks=chunks,
        )

    def _get_indexable_files(self, root_path: str) -> list[str]:
        """Get all indexable files in directory"""
        files = []

        for root, dirs, filenames in os.walk(root_path):
            # Filter ignored directories
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, filename))

        return files

    def _extract_symbols(self, content: str, language: str, file_path: str) -> list[Symbol]:
        """Extract symbols from code"""
        if language == "python":
            return self._extract_python_symbols(content, file_path)
        # TODO: Add parsers for other languages
        return []

    def _extract_python_symbols(self, content: str, file_path: str) -> list[Symbol]:
        """Extract symbols from Python code using AST"""
        symbols = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return symbols

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # Get signature
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)
                signature = f"({', '.join(args)})"

                # Get return type
                if node.returns:
                    signature += f" -> {ast.unparse(node.returns)}"

                symbols.append(
                    Symbol(
                        name=node.name,
                        type="function" if isinstance(node, ast.FunctionDef) else "async_function",
                        file=file_path,
                        line=node.lineno,
                        end_line=node.end_lineno,
                        signature=signature,
                        docstring=ast.get_docstring(node) or "",
                    )
                )

            elif isinstance(node, ast.ClassDef):
                # Get base classes
                bases = [ast.unparse(base) for base in node.bases]
                signature = f"({', '.join(bases)})" if bases else ""

                symbols.append(
                    Symbol(
                        name=node.name,
                        type="class",
                        file=file_path,
                        line=node.lineno,
                        end_line=node.end_lineno,
                        signature=signature,
                        docstring=ast.get_docstring(node) or "",
                    )
                )

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Only module-level variables
                        if hasattr(node, "col_offset") and node.col_offset == 0:
                            symbols.append(
                                Symbol(
                                    name=target.id,
                                    type="variable",
                                    file=file_path,
                                    line=node.lineno,
                                )
                            )

        return symbols

    def _extract_imports(self, content: str, language: str) -> list[str]:
        """Extract imports from code"""
        imports = []

        if language == "python":
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
            except SyntaxError:
                # Fallback to regex
                imports = re.findall(r"^(?:from|import)\s+([\w.]+)", content, re.MULTILINE)

        elif language in ("javascript", "typescript"):
            # ES6 imports
            imports = re.findall(r"(?:import|from)\s+['\"]([^'\"]+)['\"]", content)
            # require
            imports.extend(re.findall(r"require\(['\"]([^'\"]+)['\"]\)", content))

        return list(set(imports))

    def _create_chunks(self, content: str, file_path: str) -> list[dict]:
        """Create chunks for embedding"""
        chunks = []
        lines = content.split("\n")

        chunk_id = 0
        i = 0
        while i < len(lines):
            # Get chunk of lines
            chunk_lines = lines[i : i + self.chunk_size // 50]  # ~50 chars per line estimate
            chunk_content = "\n".join(chunk_lines)

            if chunk_content.strip():
                chunks.append(
                    {
                        "id": f"{file_path}:chunk:{chunk_id}",
                        "content": chunk_content,
                        "start_line": i + 1,
                        "end_line": i + len(chunk_lines),
                        "file": file_path,
                    }
                )
                chunk_id += 1

            i += len(chunk_lines) - (self.chunk_overlap // 50)
            if i <= 0:
                i = len(chunk_lines)

        return chunks


class VectorStore:
    """
    Vector store for semantic search.

    Supports:
    - Chroma (default, local)
    - FAISS (fast, in-memory)
    - Simple fallback (no dependencies)
    """

    def __init__(self, store_path: str = "vectors", backend: str = "simple"):
        self.store_path = store_path
        self.backend = backend
        self._store: dict[str, list[dict]] = {}  # Simple in-memory store
        self._embeddings_cache: dict[str, list[float]] = {}

        Path(store_path).mkdir(parents=True, exist_ok=True)
        self._load_store()

    def _load_store(self):
        """Load store from disk"""
        store_file = Path(self.store_path) / "store.json"
        if store_file.exists():
            try:
                self._store = json.loads(store_file.read_text())
            except json.JSONDecodeError:
                self._store = {}

    def _save_store(self):
        """Save store to disk"""
        store_file = Path(self.store_path) / "store.json"
        store_file.write_text(json.dumps(self._store, indent=2))

    async def add_documents(
        self,
        project_id: str,
        documents: list[dict],
        embeddings: list[list[float]] | None = None,
    ):
        """
        Add documents to the vector store.

        Args:
            project_id: Project identifier
            documents: List of documents with 'id', 'content', 'metadata'
            embeddings: Optional pre-computed embeddings
        """
        if project_id not in self._store:
            self._store[project_id] = []

        for i, doc in enumerate(documents):
            entry = {
                "id": doc.get("id", f"{project_id}:{i}"),
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "embedding": embeddings[i]
                if embeddings
                else await self._compute_embedding(doc.get("content", "")),
            }
            self._store[project_id].append(entry)

        self._save_store()
        logger.info(f"Added {len(documents)} documents to project {project_id}")

    async def search(
        self,
        project_id: str,
        query: str,
        k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Semantic search in project.

        Args:
            project_id: Project identifier
            query: Search query
            k: Number of results
            filters: Optional metadata filters

        Returns:
            List of SearchResult sorted by relevance
        """
        if project_id not in self._store:
            return []

        query_embedding = await self._compute_embedding(query)
        results = []

        for doc in self._store[project_id]:
            # Apply filters
            if filters:
                match = all(
                    doc.get("metadata", {}).get(key) == value for key, value in filters.items()
                )
                if not match:
                    continue

            # Compute similarity
            score = self._cosine_similarity(query_embedding, doc.get("embedding", []))

            results.append(
                SearchResult(
                    file=doc.get("metadata", {}).get("file", ""),
                    content=doc.get("content", ""),
                    score=score,
                    line_start=doc.get("metadata", {}).get("start_line", 0),
                    line_end=doc.get("metadata", {}).get("end_line", 0),
                    metadata=doc.get("metadata", {}),
                )
            )

        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    async def delete_by_file(self, project_id: str, file_path: str):
        """Delete all documents from a specific file"""
        if project_id in self._store:
            self._store[project_id] = [
                doc
                for doc in self._store[project_id]
                if doc.get("metadata", {}).get("file") != file_path
            ]
            self._save_store()

    async def delete_project(self, project_id: str):
        """Delete all documents for a project"""
        if project_id in self._store:
            del self._store[project_id]
            self._save_store()

    async def _compute_embedding(self, text: str) -> list[float]:
        """
        Compute embedding for text.
        Uses simple bag-of-words as fallback, can be upgraded to sentence-transformers.
        """
        # Simple TF-IDF-like embedding
        words = re.findall(r"\w+", text.lower())
        word_freq: dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Create fixed-size vector using word hashes
        vector_size = 384
        embedding = [0.0] * vector_size

        for word, freq in word_freq.items():
            idx = hash(word) % vector_size
            embedding[idx] += freq

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors"""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)


class ContextEngine:
    """
    Main context engine for AutoWorker.

    Combines:
    - ProjectRegistry for project management
    - FileIndexer for code parsing
    - VectorStore for semantic search

    Provides unified interface for context retrieval.
    """

    def __init__(
        self,
        db_path: str = "autoworker.db",
        vector_path: str = "vectors",
    ):
        self.registry = ProjectRegistry(db_path)
        self.indexer = FileIndexer()
        self.vector_store = VectorStore(vector_path)
        self._current_context: ProjectContext | None = None

    async def register_project(
        self,
        path: str,
        name: str | None = None,
        auto_index: bool = True,
    ) -> Project:
        """
        Register and optionally index a project.

        Args:
            path: Project path
            name: Optional name
            auto_index: Whether to index immediately

        Returns:
            Registered Project
        """
        project = self.registry.register(path, name)

        if auto_index:
            await self.index_project(project.id)

        return project

    async def index_project(self, project_id: str) -> dict:
        """
        Index a project for semantic search.

        Args:
            project_id: Project ID

        Returns:
            Indexing statistics
        """
        project = self.registry.get(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        self.registry.update_index_status(project_id, IndexStatus.INDEXING)

        try:
            # Index files
            indexed_files = await self.indexer.index_project(project)

            # Add to vector store
            documents = []
            total_symbols = 0

            for file_index in indexed_files:
                total_symbols += len(file_index.symbols)

                for chunk in file_index.chunks:
                    documents.append(
                        {
                            "id": chunk["id"],
                            "content": chunk["content"],
                            "metadata": {
                                "file": file_index.path,
                                "language": file_index.language,
                                "start_line": chunk["start_line"],
                                "end_line": chunk["end_line"],
                            },
                        }
                    )

            await self.vector_store.add_documents(project_id, documents)

            # Update status
            self.registry.update_index_status(
                project_id,
                IndexStatus.READY,
                file_count=len(indexed_files),
                symbol_count=total_symbols,
            )

            return {
                "success": True,
                "files_indexed": len(indexed_files),
                "symbols_found": total_symbols,
                "chunks_created": len(documents),
            }

        except Exception:
            self.registry.update_index_status(project_id, IndexStatus.ERROR)
            logger.exception(f"Failed to index project {project_id}")
            raise

    async def switch_context(self, project_id: str) -> ProjectContext:
        """
        Switch to a project context.

        Args:
            project_id: Project ID

        Returns:
            ProjectContext for the project
        """
        project = self.registry.switch_context(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        # Build context
        self._current_context = ProjectContext(
            project_id=project.id,
            project_path=project.path,
            project_type=project.type,
            entry_points=project.entry_points,
            config_files=project.config_files,
        )

        return self._current_context

    async def get_relevant_context(
        self,
        query: str,
        max_results: int = 10,
        include_symbols: bool = True,
    ) -> dict:
        """
        Get relevant context for a query.

        Args:
            query: Search query or task description
            max_results: Maximum results
            include_symbols: Include symbol definitions

        Returns:
            Context dictionary with relevant code
        """
        project = self.registry.get_current()
        if not project:
            return {"error": "No project selected"}

        # Semantic search
        results = await self.vector_store.search(
            project.id,
            query,
            k=max_results,
        )

        context = {
            "project": {
                "name": project.name,
                "type": project.type.value,
                "path": project.path,
            },
            "results": [
                {
                    "file": r.file,
                    "content": r.content,
                    "score": r.score,
                    "lines": f"{r.line_start}-{r.line_end}",
                }
                for r in results
            ],
        }

        return context

    async def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Semantic search in current project"""
        project = self.registry.get_current()
        if not project:
            return []

        return await self.vector_store.search(project.id, query, k)

    def list_projects(self) -> list[Project]:
        """List all registered projects"""
        return self.registry.list_all()

    def get_current_project(self) -> Project | None:
        """Get current project"""
        return self.registry.get_current()
