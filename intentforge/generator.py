"""
Code Generator - LLM-powered code generation with DSL support
Generates SQL, DOM manipulation, API endpoints, and firmware code
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import LLMSettings, get_settings
from .core import Intent, IntentType, TargetPlatform
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


class SQLGenerator(DSLGenerator):
    """Generate SQL queries and schemas using LLM"""

    async def generate(
        self, intent: Intent, context: GenerationContext, llm: BaseLLMProvider
    ) -> tuple[str, str]:
        """Generate SQL from intent"""
        system = f"""You are an expert SQL Generator.
        Context:
        {context.to_prompt_context()}

        Rules:
        1. Return ONLY the SQL code.
        2. Use parameterized queries where appropriate.
        3. Do not include markdown formatting unless requested, but prefer raw code or standard code block.
        4. Target platform: {context.database_type or "postgresql"}
        """

        prompt = f"""Generate SQL for: "{intent.description}"
        Intent Type: {intent.intent_type.value}
        Context: {json.dumps(intent.context)}
        Constraints: {intent.constraints}

        Return valid SQL."""

        response = await llm.generate_code(prompt, system=system)
        return self.extract_code(response.content), "sql"

    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        """Validate SQL output"""
        errors = []

        # Check for dangerous patterns
        dangerous = [
            (r"DROP\s+TABLE", "DROP TABLE detected"),
            (r"TRUNCATE", "TRUNCATE detected"),
            (r"DELETE\s+FROM\s+\w+\s*;", "DELETE without WHERE"),
            (r"UPDATE\s+\w+\s+SET.*;\s*$", "UPDATE without WHERE"),
        ]

        for pattern, message in dangerous:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(message)

        # Check for parameterized queries
        if re.search(r"['\"].*\+.*['\"]", code):
            errors.append("String concatenation in query - use parameterized queries")

        return len(errors) == 0, errors


class DOMGenerator(DSLGenerator):
    """Generate DOM manipulation code"""


class DOMGenerator(DSLGenerator):
    """Generate DOM manipulation code using LLM"""

    async def generate(
        self, intent: Intent, context: GenerationContext, llm: BaseLLMProvider
    ) -> tuple[str, str]:
        """Generate DOM manipulation code"""

        system = f"""You are an expert JavaScript/DOM Developer.
        Context:
        Project: {context.project_name}
        Framework: {context.framework}

        Rules:
        1. Return ONLY the JavaScript code.
        2. Use modern ES6+ syntax.
        3. Ensure code is secure (NO eval, NO innerHTML with user input).
        4. Handle errors gracefully.
        """

        prompt = f"""Generate DOM manipulation code for: "{intent.description}"
        Intent Type: {intent.intent_type.value}
        Context: {json.dumps(intent.context)}
        Constraints: {intent.constraints}

        Return valid JavaScript."""

        response = await llm.generate_code(prompt, system=system)
        return self.extract_code(response.content), "javascript"

    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        """Validate DOM code"""
        errors = []

        # Check for dangerous patterns
        if "eval(" in code:
            errors.append("eval() detected - security risk")
        if "innerHTML" in code and "user" in code.lower():
            errors.append("innerHTML with user input - XSS risk")
        if "document.write" in code:
            errors.append("document.write() is deprecated")

        return len(errors) == 0, errors


class APIEndpointGenerator(DSLGenerator):
    """Generate API endpoint code using LLM"""

    async def generate(
        self, intent: Intent, context: GenerationContext, llm: BaseLLMProvider
    ) -> tuple[str, str]:
        """Generate API endpoint based on intent"""
        framework = context.framework.lower()

        system = f"""You are an expert Backend Developer.
        Context:
        Project: {context.project_name}
        Framework: {framework} (e.g. FastAPI, Flask, Express)
        Database: {context.database_type}
        ORM: {context.orm_type if context.use_orm else "None"}

        Rules:
        1. Return ONLY the code for the endpoint/router.
        2. Use best practices for the specified framework.
        3. Include Pydantic models (for FastAPI) or equivalent validation.
        4. Handle database sessions correctly.
        5. Follow RESTful conventions.
        """

        prompt = f"""Generate API endpoint code:
        Intent: {intent.description}
        Type: {intent.intent_type.value}
        Context: {json.dumps(intent.context)}
        Constraints: {intent.constraints}

        Platform: {intent.target_platform.value}

        Return valid code."""

        start_time = time.time()

        response = await llm.generate_code(prompt, system=system)

        # Determine language
        language = "python"
        if framework in ("express", "nodejs", "react"):
            language = "javascript"

        return self.extract_code(response.content), language

    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        errors = []

        # Check for hardcoded credentials
        if re.search(r'password\s*=\s*["\'][^"\']+["\']', code, re.IGNORECASE):
            if "os.getenv" not in code and "process.env" not in code:
                errors.append("Hardcoded password detected - use environment variables")

        return len(errors) == 0, errors


class GenericGenerator(DSLGenerator):
    """Generate generic code (algorithms, workflows, etc)"""

    async def generate(
        self, intent: Intent, context: GenerationContext, llm: BaseLLMProvider
    ) -> tuple[str, str]:
        """Generate generic code"""

        system = f"""You are an expert Software Developer.
        Context:
        Project: {context.project_name}
        Platform: {intent.target_platform.value}

        Rules:
        1. Return ONLY the code.
        2. Follow best practices for the target language.
        3. Include docstrings and comments.
        4. Make code robust and handle errors.
        5. IMPORTANT: Assign the final return value (if any) to a global variable named 'result'.
        6. Use simple dictionaries for data objects, avoid defining custom classes.
        """

        prompt = f"""Generate code for: "{intent.description}"
        Intent Type: {intent.intent_type.value}
        Context: {json.dumps(intent.context)}
        Constraints: {intent.constraints}

        Return valid code."""

        response = await llm.generate_code(prompt, system=system)

        # Determine language
        language = "python"
        platform = intent.target_platform.value
        if "node" in platform or "js" in platform or "script" in platform:
            language = "javascript"
        elif "cpp" in platform or "arduino" in platform:
            language = "cpp"

        return self.extract_code(response.content), language

    def validate_output(self, code: str) -> tuple[bool, list[str]]:
        """Basic validation"""
        errors = []
        if not code.strip():
            errors.append("Code is empty")
        return len(errors) == 0, errors


class CodeGenerator:
    """
    Main code generator - orchestrates DSL generators and LLM
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

        # Initialize generators
        self.generators = {
            DSLType.SQL_QUERY: SQLGenerator(),
            DSLType.SQL_SCHEMA: SQLGenerator(),
            DSLType.DOM_MANIPULATION: DOMGenerator(),
            DSLType.DOM_QUERY: DOMGenerator(),
            DSLType.FORM_HANDLER: DOMGenerator(),
            DSLType.API_ENDPOINT: APIEndpointGenerator(),
            DSLType.GENERIC: GenericGenerator(),
            DSLType.DATABASE_CRUD: APIEndpointGenerator(),
        }

        self._client = None

    @property
    def client(self) -> BaseLLMProvider:
        """Lazy load LLM provider"""
        if self._client is None:
            config = LLMConfig(api_key=self.api_key)
            self._client = get_llm_provider(self.provider_name, model=self.model, config=config)
        return self._client

    async def generate(
        self, intent: Intent, context: GenerationContext | None = None
    ) -> tuple[str, str]:
        """
        Generate code from intent with validation and self-correction loop
        Returns (code, language)
        """
        context = context or GenerationContext()

        # Determine DSL type from intent
        dsl_type = self._determine_dsl_type(intent)
        generator = self.generators.get(dsl_type)

        # Default to API endpoint logic if something unknown or if fallback needed,
        # but here we rely on the specific generator to handle the prompt
        if not generator:
            # Fallback generic handling using generic DSL generator or just direct LLM
            # For now we reuse APIEndpointGenerator or define a GenericGenerator
            generator = self.generators[DSLType.API_ENDPOINT]  # fallback

        # 1. Generate
        logger.info(f"Generating code for {intent.intent_id} using {dsl_type}")
        code, language = await generator.generate(intent, context, self.client)

        # 2. Validation Loop
        max_retries = 3
        for attempt in range(max_retries):
            is_valid, errors = generator.validate_output(code)

            if is_valid:
                logger.info("Validation passed")
                return code, language

            logger.warning(f"Validation failed (Attempt {attempt + 1}/{max_retries}): {errors}")

            # 3. Fix
            try:
                code = await generator.fix(code, errors, intent, self.client)
                # Code returned from fix might be just code or markdown, ensure cleanup if needed
                # The fix method relies on generate_code which usually returns raw code if instructed
                # but let's be safe
            except Exception as e:
                logger.error(f"Fix failed: {e}")
                break  # Stop if fix fails hard

        # Return whatever we have (even if invalid, validation happens later in pipeline too)
        # But maybe we should mark it? core.py handles validation again
        return code, language

    def _determine_dsl_type(self, intent: Intent) -> DSLType:
        """Determine which DSL generator to use"""
        desc = intent.description.lower()
        intent_type = intent.intent_type

        if intent_type == IntentType.WORKFLOW or intent_type == IntentType.FIRMWARE_FUNCTION:
            return DSLType.GENERIC

        if any(word in desc for word in ["sql", "query", "select", "insert", "database", "tabela"]):
            return DSLType.SQL_QUERY

        if any(word in desc for word in ["form", "formularz", "submit"]):
            return DSLType.FORM_HANDLER

        if any(word in desc for word in ["dom", "element", "html", "click", "event"]):
            return DSLType.DOM_MANIPULATION

        if any(word in desc for word in ["api", "endpoint", "rest", "route"]):
            return DSLType.API_ENDPOINT

        if any(word in desc for word in ["crud", "resource"]):
            return DSLType.DATABASE_CRUD

        if intent_type == IntentType.API_ENDPOINT:
            return DSLType.API_ENDPOINT

        return DSLType.GENERIC  # Default to generic instead of API

    def _determine_language(self, intent: Intent, dsl_type: DSLType) -> str:
        """Determine output language"""
        platform = intent.target_platform

        if dsl_type in (DSLType.SQL_QUERY, DSLType.SQL_SCHEMA):
            return "sql"

        if dsl_type in (DSLType.DOM_MANIPULATION, DSLType.DOM_QUERY, DSLType.FORM_HANDLER):
            return "javascript"

        platform_languages = {
            TargetPlatform.PYTHON_FASTAPI: "python",
            TargetPlatform.PYTHON_FLASK: "python",
            TargetPlatform.NODEJS_EXPRESS: "javascript",
            TargetPlatform.ARDUINO_CPP: "cpp",
            TargetPlatform.ESP32_MICROPYTHON: "python",
            TargetPlatform.JETSON_PYTHON: "python",
            TargetPlatform.RASPBERRY_PI: "python",
        }

        return platform_languages.get(platform, "python")

    async def _generate_with_llm(
        self, intent: Intent, context: GenerationContext
    ) -> tuple[str, str]:
        """Generate code using LLM"""

        system_prompt = f"""You are a code generator. Generate production-ready code based on the user's intent.

Context:
{context.to_prompt_context()}

Rules:
1. Generate ONLY code, no explanations
2. Use environment variables for secrets (from .env file)
3. Include proper error handling
4. Follow security best practices
5. Use parameterized queries for SQL
6. Add appropriate comments

Target platform: {intent.target_platform.value}
"""

        user_prompt = f"""Generate code for the following intent:

Intent: {intent.description}
Type: {intent.intent_type.value}
Context: {json.dumps(intent.context)}
Constraints: {intent.constraints}

Return only the code."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            code = message.content[0].text

            # Extract code from markdown if present
            code = self._extract_code(code)
            language = self._determine_language(intent, self._determine_dsl_type(intent))

            return code, language

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks"""
        # Try to find code block
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        return text.strip()
