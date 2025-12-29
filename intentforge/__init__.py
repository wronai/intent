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
    "BasePlugin",
    # Generator
    "CodeGenerator",
    # Validator
    "CodeValidator",
    "DSLType",
    # Executor
    "DynamicExecutor",
    "EnvConfig",
    # Environment Handler
    "EnvHandler",
    "Forge",
    # Patterns
    "FullstackPatterns",
    "GenerationContext",
    "HookEvent",
    "Intent",
    # Cache
    "IntentCache",
    # Core
    "IntentForge",
    "IntentResult",
    "IntentType",
    # Broker
    "MQTTIntentBroker",
    "MiddlewarePhase",
    "PatternConfig",
    "PatternType",
    # Schema Registry
    "SchemaRegistry",
    "SchemaType",
    "Settings",
    "TargetPlatform",
    "ValidationLevel",
    "ValidationResult",
    "audit_log",
    "cached",
    "configure_env",
    "crud",
    "form",
    # Simple API (one-liners)
    "generate",
    "get_env",
    "get_registry",
    # Config
    "get_settings",
    # Plugins & Middleware
    "hook",
    "hooks",
    "middleware",
    "plugins",
    "query",
    "rate_limit",
    "retry",
    "save",
    "validate",
    "validate_input",
]
