"""
IntentForge Simple API - One-liner interface for common tasks
No configuration needed, sensible defaults, progressive complexity
"""

import asyncio
import os
from pathlib import Path
from typing import Any

# Lazy imports to speed up initial load
_forge_instance = None
_env_loaded = False


def _get_forge():
    """Lazy load forge instance"""
    global _forge_instance
    if _forge_instance is None:
        from .core import IntentForge

        _forge_instance = IntentForge(
            mqtt_broker=os.getenv("MQTT_HOST", "localhost"),
            mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            enable_auto_deploy=False,
            sandbox_mode=True,
        )
    return _forge_instance


def _ensure_env():
    """Load .env if exists"""
    global _env_loaded
    if not _env_loaded:
        from dotenv import load_dotenv

        load_dotenv()
        _env_loaded = True


# =============================================================================
# ONE-LINER API
# =============================================================================


def generate(description: str, **kwargs) -> str:
    """
    Generate code from natural language - simplest possible API

    Usage:
        code = generate("Create user registration endpoint")
        code = generate("Form handler for contact form", platform="flask")
    """
    _ensure_env()

    from .core import Intent, TargetPlatform

    # Auto-detect intent type from description
    intent_type = _detect_intent_type(description)

    # Map simple platform names
    platform_map = {
        "fastapi": TargetPlatform.PYTHON_FASTAPI,
        "flask": TargetPlatform.PYTHON_FLASK,
        "express": TargetPlatform.NODEJS_EXPRESS,
        "arduino": TargetPlatform.ARDUINO_CPP,
        "esp32": TargetPlatform.ESP32_MICROPYTHON,
        "jetson": TargetPlatform.JETSON_PYTHON,
        "rpi": TargetPlatform.RASPBERRY_PI,
    }

    platform = platform_map.get(kwargs.get("platform", "fastapi"), TargetPlatform.PYTHON_FASTAPI)

    intent = Intent(
        description=description,
        intent_type=intent_type,
        target_platform=platform,
        context=kwargs.get("context", {}),
        constraints=kwargs.get("constraints", []),
    )

    # Run async in sync context
    result = asyncio.run(_get_forge().process_intent(intent))

    if result.success:
        return result.generated_code
    else:
        raise GenerationError(result.validation_errors)


def crud(table: str, fields: list[dict[str, Any]] = None, **kwargs) -> dict[str, str]:
    """
    Generate complete CRUD for a table - one function call

    Usage:
        files = crud("users", [
            {"name": "email", "type": "email", "required": True},
            {"name": "name", "type": "text", "required": True}
        ])

        # Even simpler - auto-detect from table name
        files = crud("contacts")
    """
    from .patterns import FullstackPatterns, PatternConfig, PatternType

    # Default fields based on common table names
    if fields is None:
        fields = _get_default_fields(table)

    config = PatternConfig(
        pattern_type=PatternType.CRUD_API,
        target_table=table,
        fields=fields,
        auth_required=kwargs.get("auth", False),
        use_validation=True,
        framework=kwargs.get("framework", "fastapi"),
        include_tests=kwargs.get("tests", True),
    )

    return FullstackPatterns.form_to_database(config)


def form(form_id: str, fields: list[dict[str, Any]], **kwargs) -> dict[str, str]:
    """
    Generate form with full backend integration

    Usage:
        files = form("contact", [
            {"name": "email", "type": "email", "required": True},
            {"name": "message", "type": "textarea"}
        ])
    """
    from .patterns import FullstackPatterns, PatternConfig, PatternType

    config = PatternConfig(
        pattern_type=PatternType.FORM_TO_DATABASE,
        target_table=form_id,
        fields=fields,
        auth_required=kwargs.get("auth", False),
        framework=kwargs.get("framework", "fastapi"),
    )

    return FullstackPatterns.form_to_database(config)


def query(description: str, table: str = None, **kwargs) -> tuple:
    """
    Generate safe SQL query from natural language

    Usage:
        sql, params = query("Get active users sorted by date", table="users")
        sql, params = query("Find orders over $100 from last week")
    """
    _ensure_env()

    from .core import Intent, IntentType
    from .generator import SQLGenerator

    intent = Intent(
        description=description,
        intent_type=IntentType.DATABASE_SCHEMA,
        context={"table": table} if table else {},
    )

    generator = SQLGenerator()
    from .generator import GenerationContext

    code = generator.generate(intent, GenerationContext())

    # Extract parameterized query
    return code, {}


def validate(code: str, language: str = "python") -> bool:
    """
    Validate code safety and correctness

    Usage:
        is_safe = validate(generated_code)
        is_safe = validate(js_code, "javascript")
    """
    from .validator import CodeValidator

    validator = CodeValidator(sandbox_mode=True)
    result = validator.validate_sync(code, language)

    return result.is_valid


def save(files: dict[str, str], output_dir: str = "generated") -> list[str]:
    """
    Save generated files to disk

    Usage:
        paths = save(crud("users"), "src/users")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for filename, content in files.items():
        # Determine extension
        if "html" in filename:
            ext = ".html"
        elif "js" in filename or "frontend_js" in filename:
            ext = ".js"
        elif "sql" in filename or "migration" in filename:
            ext = ".sql"
        else:
            ext = ".py"

        filepath = output_path / f"{filename}{ext}"
        filepath.write_text(content)
        saved.append(str(filepath))

    return saved


# =============================================================================
# FLUENT BUILDER API
# =============================================================================


class ForgeBuilder:
    """
    Fluent builder for complex generations

    Usage:
        result = (Forge("Create REST API for products")
            .platform("fastapi")
            .with_auth()
            .with_pagination()
            .fields(["name", "price", "stock"])
            .generate())
    """

    def __init__(self, description: str):
        self.description = description
        self._platform = "fastapi"
        self._auth = False
        self._pagination = False
        self._fields = []
        self._constraints = []
        self._context = {}

    def platform(self, name: str) -> "ForgeBuilder":
        self._platform = name
        return self

    def with_auth(self, auth_type: str = "jwt") -> "ForgeBuilder":
        self._auth = True
        self._context["auth_type"] = auth_type
        return self

    def with_pagination(self, default_limit: int = 50) -> "ForgeBuilder":
        self._pagination = True
        self._context["pagination"] = {"default_limit": default_limit}
        return self

    def fields(self, field_list: list[str | dict]) -> "ForgeBuilder":
        self._fields = [
            f if isinstance(f, dict) else {"name": f, "type": "text"} for f in field_list
        ]
        return self

    def constraint(self, rule: str) -> "ForgeBuilder":
        self._constraints.append(rule)
        return self

    def context(self, **kwargs) -> "ForgeBuilder":
        self._context.update(kwargs)
        return self

    def generate(self) -> str:
        return generate(
            self.description,
            platform=self._platform,
            context={
                **self._context,
                "auth_required": self._auth,
                "pagination": self._pagination,
                "fields": self._fields,
            },
            constraints=self._constraints,
        )

    def save(self, output_dir: str = "generated") -> list[str]:
        code = self.generate()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        filepath = path / "generated.py"
        filepath.write_text(code)
        return [str(filepath)]


# Alias for cleaner API
Forge = ForgeBuilder


# =============================================================================
# HELPERS
# =============================================================================


class GenerationError(Exception):
    """Error during code generation"""

    pass


def _detect_intent_type(description: str):
    """Auto-detect intent type from description"""
    from .core import IntentType

    desc_lower = description.lower()

    # Keywords mapping
    if any(w in desc_lower for w in ["endpoint", "api", "rest", "route", "crud"]):
        return IntentType.API_ENDPOINT

    if any(w in desc_lower for w in ["form", "formularz", "input", "submit"]):
        return IntentType.EVENT_HANDLER

    if any(w in desc_lower for w in ["sql", "query", "select", "database", "tabela"]):
        return IntentType.DATABASE_SCHEMA

    if any(w in desc_lower for w in ["gpio", "pin", "sensor", "led", "mqtt", "iot"]):
        return IntentType.FIRMWARE_FUNCTION

    if any(w in desc_lower for w in ["event", "click", "observer", "handler"]):
        return IntentType.EVENT_HANDLER

    return IntentType.API_ENDPOINT


def _get_default_fields(table: str) -> list[dict[str, Any]]:
    """Get sensible default fields based on table name"""

    defaults = {
        "users": [
            {"name": "email", "type": "email", "required": True},
            {"name": "name", "type": "text", "required": True},
            {"name": "password", "type": "password", "required": True},
        ],
        "contacts": [
            {"name": "name", "type": "text", "required": True},
            {"name": "email", "type": "email", "required": True},
            {"name": "phone", "type": "text"},
            {"name": "message", "type": "textarea", "required": True},
        ],
        "products": [
            {"name": "name", "type": "text", "required": True},
            {"name": "description", "type": "textarea"},
            {"name": "price", "type": "number", "required": True},
            {"name": "stock", "type": "number"},
        ],
        "orders": [
            {"name": "customer_id", "type": "number", "required": True},
            {"name": "total", "type": "float", "required": True},
            {"name": "status", "type": "select", "options": ["pending", "paid", "shipped"]},
        ],
        "posts": [
            {"name": "title", "type": "text", "required": True},
            {"name": "content", "type": "textarea", "required": True},
            {"name": "author_id", "type": "number"},
            {"name": "published", "type": "boolean"},
        ],
        "comments": [
            {"name": "post_id", "type": "number", "required": True},
            {"name": "author", "type": "text", "required": True},
            {"name": "content", "type": "textarea", "required": True},
        ],
    }

    # Return matching or generic
    return defaults.get(
        table,
        [
            {"name": "name", "type": "text", "required": True},
            {"name": "description", "type": "textarea"},
            {"name": "status", "type": "text"},
        ],
    )


# =============================================================================
# CLI SHORTCUTS
# =============================================================================


def main():
    """CLI entry point for quick generation"""
    import sys

    if len(sys.argv) < 2:
        print("IntentForge - Quick Code Generation")
        print("=" * 40)
        print("\nUsage:")
        print("  iforge 'Create user API with auth'")
        print("  iforge crud users")
        print("  iforge form contact")
        print("\nExamples:")
        print("  iforge 'REST endpoint for products'")
        print("  iforge crud orders --auth")
        print("  iforge form newsletter --output=src/")
        return

    command = sys.argv[1]

    if command == "crud" and len(sys.argv) > 2:
        table = sys.argv[2]
        files = crud(table)
        paths = save(files, f"generated/{table}")
        print(f"Generated {len(paths)} files in generated/{table}/")

    elif command == "form" and len(sys.argv) > 2:
        form_id = sys.argv[2]
        files = form(form_id, _get_default_fields(form_id))
        paths = save(files, f"generated/{form_id}")
        print(f"Generated {len(paths)} files in generated/{form_id}/")

    else:
        # Treat as description
        description = " ".join(sys.argv[1:])
        code = generate(description)
        print(code)


if __name__ == "__main__":
    main()
