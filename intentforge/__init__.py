"""
IntentForge - NLP-driven Code Generation Framework
Generates backend and firmware code from natural language intents via MQTT

LLM Providers:
    from intentforge.llm import get_llm_provider
    llm = get_llm_provider("ollama", model="llama3")

Plugins:
    from intentforge.plugins import hook, middleware
    @hook("form:submit")
    def on_form(ctx): pass
"""

__version__ = "0.2.0"
__author__ = "Softreck"

from .broker import MQTTIntentBroker
from .cache import IntentCache
from .config import Settings, get_settings
from .core import Intent, IntentForge, IntentResult, IntentType, TargetPlatform
from .env_handler import EnvConfig, EnvHandler, configure_env, get_env
from .executor import DynamicExecutor
from .generator import CodeGenerator, DSLType, GenerationContext
from .patterns import FullstackPatterns, PatternConfig, PatternType

# New modules
from .plugins import (
    BasePlugin,
    HookEvent,
    MiddlewarePhase,
    audit_log,
    cached,
    hook,
    hooks,
    middleware,
    plugins,
    rate_limit,
    retry,
    validate_input,
)
from .schema_registry import SchemaRegistry, SchemaType, get_registry
from .simple import Forge, crud, form, generate, query, save, validate
from .validator import CodeValidator, ValidationLevel, ValidationResult

__all__ = [
    # Core
    "IntentForge",
    "Intent",
    "IntentResult",
    "IntentType",
    "TargetPlatform",
    # Broker
    "MQTTIntentBroker",
    # Generator
    "CodeGenerator",
    "DSLType",
    "GenerationContext",
    # Validator
    "CodeValidator",
    "ValidationLevel",
    "ValidationResult",
    # Executor
    "DynamicExecutor",
    # Cache
    "IntentCache",
    # Config
    "get_settings",
    "Settings",
    # Schema Registry
    "SchemaRegistry",
    "get_registry",
    "SchemaType",
    # Environment Handler
    "EnvHandler",
    "get_env",
    "configure_env",
    "EnvConfig",
    # Patterns
    "FullstackPatterns",
    "PatternConfig",
    "PatternType",
    # Simple API (one-liners)
    "generate",
    "crud",
    "form",
    "query",
    "validate",
    "save",
    "Forge",
    # Plugins & Middleware
    "hook",
    "middleware",
    "hooks",
    "plugins",
    "HookEvent",
    "MiddlewarePhase",
    "BasePlugin",
    "cached",
    "retry",
    "rate_limit",
    "validate_input",
    "audit_log",
]
