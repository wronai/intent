"""
Code Generator - Pure LLM-powered code generation
All code is generated dynamically via user commands and LLM
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import LLMSettings, get_settings
from .core import Intent
from .llm import BaseLLMProvider, LLMConfig, get_llm_provider

logger = logging.getLogger(__name__)


class DSLType(Enum):
    """Supported DSL types for code generation"""

    SQL_QUERY = "sql_query"
    SQL_SCHEMA = "sql_schema"
    DOM_QUERY = "dom_query"
    DOM_MANIPULATION = "dom_manipulation"
    API_ENDPOINT = "api_endpoint"
    EVENT_HANDLER = "event_handler"
    FORM_HANDLER = "form_handler"
    DATABASE_CRUD = "database_crud"
    GENERIC = "generic"


@dataclass
class GenerationContext:
    """Context for code generation"""

    project_name: str = "app"
    database_type: str = "postgresql"
    framework: str = "fastapi"
    use_orm: bool = True
    orm_type: str = "sqlalchemy"
    auth_type: str | None = None
    env_prefix: str = ""

    # Schema information
    tables: dict[str, dict[str, str]] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)

    def to_prompt_context(self) -> str:
        """Convert to string for LLM prompt"""
        lines = [
            f"Project: {self.project_name}",
            f"Database: {self.database_type}",
            f"Framework: {self.framework}",
            f"ORM: {self.orm_type if self.use_orm else 'raw SQL'}",
        ]

        if self.tables:
            lines.append("\nAvailable tables:")
            for table, columns in self.tables.items():
                cols = ", ".join(f"{k}: {v}" for k, v in columns.items())
                lines.append(f"  - {table}({cols})")

        return "\n".join(lines)


class DSLGenerator(ABC):
    """Base class for DSL-specific generators"""

    @abstractmethod
    async def generate(
        self, intent: Intent, context: GenerationContext, llm: BaseLLMProvider
    ) -> tuple[str, str]:
        """
        Generate code from intent using LLM
        Returns (code, language)
        """
        pass

    @abstractmethod
    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        pass

    async def fix(self, code: str, errors: list[str], intent: Intent, llm: BaseLLMProvider) -> str:
        """Fix code based on validation errors"""
        system = "You are an expert code fixer. Fix the code based on the errors provided."
        prompt = f"""Original Code:
{code}

Validation Errors:
{json.dumps(errors, indent=2)}

Intent: {intent.description}

Fix the code and return ONLY the corrected code."""

        response = await llm.generate_code(prompt, system=system)
        return self.extract_code(response.content)

    def extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks"""
        # Try to find code block
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        return text.strip()


class UniversalGenerator(DSLGenerator):
    """
    Universal code generator - generates ALL code types via LLM.
    No hardcoded templates. Everything is dynamically generated based on user intent.
    """

    async def generate(
        self, intent: Intent, context: GenerationContext, llm: BaseLLMProvider
    ) -> tuple[str, str]:
        """Generate any type of code based on intent"""

        # Determine target language from platform
        language = self._detect_language(intent)

        system = f"""You are a code generator that creates simple, executable Python code.

CRITICAL RULES:
1. Return ONLY raw Python code - NO markdown, NO explanations.
2. For mock/demo data: Create it DIRECTLY in code using literals (lists, dicts).
3. DO NOT import: os, sys, subprocess, shutil, socket, http, urllib, sqlalchemy.
4. DO NOT use: database connections, file I/O, network calls, environment variables.
5. Assign the final result to a variable named 'result'.
6. Keep code simple and self-contained.

Example for "return 3 users":
```
result = [
    {{"id": 1, "name": "Alice", "email": "alice@example.com"}},
    {{"id": 2, "name": "Bob", "email": "bob@example.com"}},
    {{"id": 3, "name": "Charlie", "email": "charlie@example.com"}}
]
```

Target Language: {language}
"""

        prompt = f"""Generate simple Python code for: "{intent.description}"

REMEMBER:
- NO imports of os, sys, database libraries
- Create mock data DIRECTLY as Python literals
- Assign result to 'result' variable
- Return ONLY the code, nothing else"""

        response = await llm.generate_code(prompt, system=system)
        return self.extract_code(response.content), language

    def _detect_language(self, intent: Intent) -> str:
        """Detect target programming language from intent and platform"""
        desc = intent.description.lower()
        platform = intent.target_platform.value.lower()

        # Check intent description for language hints
        if any(word in desc for word in ["sql", "query", "select", "insert", "database"]):
            return "sql"
        if any(word in desc for word in ["javascript", "js", "dom", "html", "browser"]):
            return "javascript"
        if any(word in desc for word in ["cpp", "c++", "arduino"]):
            return "cpp"

        # Check platform
        if "node" in platform or "express" in platform:
            return "javascript"
        if "arduino" in platform or "cpp" in platform:
            return "cpp"

        # Default to Python
        return "python"

    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        """Basic security validation"""
        errors = []

        if not code.strip():
            errors.append("Generated code is empty")

        # Security checks
        dangerous_patterns = [
            (r"eval\(", "eval() is a security risk"),
            (r"exec\(", "exec() is a security risk"),
            (r"__import__\(", "__import__() is a security risk"),
            (r"os\.system\(", "os.system() is a security risk"),
            (r"subprocess\..*shell\s*=\s*True", "shell=True in subprocess is risky"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(message)

        # Check for hardcoded secrets
        if re.search(r'password\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
            if "os.getenv" not in code and "process.env" not in code:
                errors.append("Hardcoded password detected - use environment variables")

        return len(errors) == 0, errors


class CodeGenerator:
    """
    Main code generator - uses UniversalGenerator for all code types.
    All generation is done via LLM, no hardcoded templates.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider: str = "anthropic",
        settings: LLMSettings | None = None,
    ):
        self.settings = settings or get_settings().llm
        self.api_key = api_key or self.settings.api_key.get_secret_value()
        self.model = model or self.settings.model
        self.provider_name = provider

        # Single universal generator for all types
        self.generator = UniversalGenerator()

        self._client = None

    @property
    def client(self) -> BaseLLMProvider:
        """Lazy load LLM provider"""
        if self._client is None:
            config = LLMConfig(api_key=self.api_key)
            self._client = get_llm_provider(self.provider_name, model=self.model, config=config)
        return self._client

    async def generate(
        self,
        intent: Intent,
        context: GenerationContext | None = None,
        max_retries: int = 3,
    ) -> tuple[str, str]:
        """
        Generate code from intent using LLM.

        Args:
            intent: The user's intent
            context: Optional generation context
            max_retries: Number of retry attempts for validation failures

        Returns:
            Tuple of (generated_code, language)
        """
        ctx = context or GenerationContext()

        for attempt in range(max_retries):
            # Generate code
            code, language = await self.generator.generate(intent, ctx, self.client)

            # Validate
            is_valid, errors = self.generator.validate_output(code)

            if is_valid:
                logger.info(f"Code generated successfully on attempt {attempt + 1}")
                return code, language

            logger.warning(f"Validation failed (attempt {attempt + 1}): {errors}")

            # Try to fix
            if attempt < max_retries - 1:
                code = await self.generator.fix(code, errors, intent, self.client)

        # Return last attempt even if validation failed
        logger.warning("Returning code despite validation failures")
        return code, language

    async def generate_with_retry(
        self,
        intent: Intent,
        context: GenerationContext | None = None,
        max_retries: int = 3,
    ) -> tuple[str, str]:
        """Alias for generate() method"""
        return await self.generate(intent, context, max_retries)
